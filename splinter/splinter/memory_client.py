"""Client for the daemon-hosted native memory API."""

from __future__ import annotations

from typing import Any

import requests

DEFAULT_URL = "http://127.0.0.1:8765"


class NativeMemoryError(RuntimeError):
    """The daemon memory API is unavailable or rejected a request."""


class NativeMemoryClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 120.0) -> None:
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        import sys

        print(f"[memory] native daemon at {self.base_url}", file=sys.stderr, flush=True)

    def _post(self, conversation: str, action: str, payload: dict | None = None) -> dict:
        try:
            response = requests.post(
                f"{self.base_url}/v1/memory/{conversation}/{action}",
                json=payload or {},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise NativeMemoryError(
                f"native memory {action} failed at {self.base_url}: {exc}"
            ) from exc
        except ValueError as exc:
            raise NativeMemoryError(f"native memory returned invalid JSON: {exc}") from exc

    def load(
        self,
        conversation: str,
        chunks: list[dict[str, Any]],
        *,
        max_chunks: int | None = None,
        sanitize: bool = True,
        ingest_block_prefixes: list[str] | None = None,
    ) -> dict:
        return self._post(
            conversation,
            "load",
            {
                "chunks": chunks,
                "max_chunks": max_chunks,
                "sanitize": sanitize,
                "ingest_block_prefixes": ingest_block_prefixes,
            },
        )

    def add(
        self,
        conversation: str,
        turn: int,
        content: str,
        chunk_id: str | None = None,
    ) -> dict:
        return self._post(
            conversation,
            "add",
            {"turn": turn, "content": content, "chunk_id": chunk_id},
        )

    def assemble(
        self,
        conversation: str,
        query: str,
        *,
        current_turn: int,
        route: str = "ultra_small",
        max_context: int = 8192,
    ) -> dict:
        return self._post(
            conversation,
            "assemble",
            {
                "query": query,
                "current_turn": current_turn,
                "route": route,
                "max_context": max_context,
            },
        )

    def reset(self, conversation: str) -> dict:
        return self._post(conversation, "reset")

    def status(self, conversation: str) -> dict:
        try:
            response = requests.get(
                f"{self.base_url}/v1/memory/{conversation}/status",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise NativeMemoryError(f"native memory status failed: {exc}") from exc
