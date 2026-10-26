"""Tiny synchronous client for the running Studio's local HTTP API.

The CLI and the terminal approval overlay both talk to the same live Studio
process.  Keeping the transport here means both surfaces share the URL,
bearer-token, error, and JSON rules.  This module never starts a server and
never imports FastAPI.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8765"


class StudioAPIError(RuntimeError):
    """The Studio is unreachable or answered with something unusable."""


def base_url(url: str | None = None) -> str:
    return (url or os.environ.get("HIVE_STUDIO_URL")
            or DEFAULT_URL).rstrip("/")


def call(path: str, *, method: str = "GET",
         body: dict[str, Any] | None = None, url: str | None = None,
         timeout: float = 5.0) -> dict[str, Any]:
    """Call one Studio JSON endpoint and return its object response."""
    if not path.startswith("/"):
        raise StudioAPIError("Studio API path must start with '/'")
    target = base_url(url) + path
    headers = {"content-type": "application/json"}
    token = os.environ.get("HARNESS_TOKEN", "").strip()
    if token:
        headers["x-splinter-token"] = token
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(target, data=data, method=method,
                                     headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise StudioAPIError(
            f"Studio gateway HTTP {exc.code}: {detail[:300]}") from exc
    except OSError as exc:
        raise StudioAPIError(
            f"Studio gateway unreachable at {base_url(url)}: {exc}") from exc
    try:
        result = json.loads(raw)
    except ValueError as exc:
        raise StudioAPIError("Studio gateway returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise StudioAPIError("Studio gateway returned a non-object")
    return result
