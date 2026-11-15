"""Client for the daemon-hosted native memory API."""

from __future__ import annotations

from typing import Any, Optional

import requests

DEFAULT_URL = "http://127.0.0.1:8765"


class NativeMemoryError(RuntimeError):
    """The daemon memory API is unavailable or rejected a request."""


class NativeMemoryClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 120.0) -> None:
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        import sys

        print(f"[memory] native daemon at {self.base_url}", file=sys.stderr, flush=True)

    def _post(self, conversation: str, action: str, payload: dict | None = None) -> dict:
        try:
            response = self.session.post(
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
        relevance_floor: float = 0.25,
        max_chunk_share: float = 0.5,
        drift_threshold: float = 0.6,
        dedup_threshold: float = 0.92,
        stale_threshold: int | None = None,
        stale_factor: float | None = None,
        skip_remembrance: bool = False,
        skip_dedup: bool = False,
        comb_candidates: list[dict[str, Any]] | None = None,
        payload_fingerprints: Optional[set[str]] = None,
        dedup_against_payload: bool = True,
        drift_prior: str = "off",
        drift_prior_factor: float = 0.5,
        vocab_boost: float = 0.15,
        budget: int | None = None,
        score_override: dict[str, float] | None = None,
    ) -> dict:
        return self._post(
            conversation,
            "assemble",
            {
                "query": query,
                "current_turn": current_turn,
                "route": route,
                "max_context": max_context,
                "relevance_floor": relevance_floor,
                "max_chunk_share": max_chunk_share,
                "drift_threshold": drift_threshold,
                "dedup_threshold": dedup_threshold,
                "stale_threshold": stale_threshold,
                "stale_factor": stale_factor,
                "skip_remembrance": skip_remembrance,
                "skip_dedup": skip_dedup,
                "comb_candidates": comb_candidates or [],
                "payload_fingerprints": sorted(payload_fingerprints or []),
                "dedup_against_payload": dedup_against_payload,
                "drift_prior": drift_prior,
                "drift_prior_factor": drift_prior_factor,
                "vocab_boost": vocab_boost,
                "budget": budget,
                "score_override": score_override,
            },
        )

    def reset(self, conversation: str) -> dict:
        return self._post(conversation, "reset")

    def status(self, conversation: str) -> dict:
        try:
            response = self.session.get(
                f"{self.base_url}/v1/memory/{conversation}/status",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise NativeMemoryError(f"native memory status failed: {exc}") from exc


def apply_memory_updates(store, payload: dict) -> None:
    """Mirror the mutations native assembly made back onto the Python store."""
    for chunk_id, turn in (payload.get("refresh") or {}).items():
        chunk = getattr(store, "chunks", {}).get(chunk_id)
        if chunk is not None:
            chunk.last_referenced_turn = max(
                chunk.last_referenced_turn or 0, int(turn)
            )
    for chunk_id, history in (payload.get("relevance_updates") or {}).items():
        chunk = getattr(store, "chunks", {}).get(chunk_id)
        if chunk is not None:
            chunk.relevance_history = [list(item) for item in history][-10:]
    for result in payload.get("remembrance") or []:
        if not result.get("saved"):
            continue
        chunk = getattr(store, "chunks", {}).get(result.get("chunk_id"))
        if chunk is not None:
            chunk.times_saved = int(result.get("times_saved") or chunk.times_saved)
            chunk.decay_multiplier = float(
                result.get("new_decay") or chunk.decay_multiplier
            )


class NativeAssembler:
    """Drop-in `ContextAssembler.assemble` backed by the daemon memory API.

    The daemon owns the chunk store, decay, dedup, drift, and selection. This
    adapter synchronizes the Python store before each call (the protocol
    replays mutate it directly) and mirrors native state changes back so comb
    archiving and the next turn see the same decay/history bookkeeping.
    """

    def __init__(
        self,
        client: NativeMemoryClient,
        conversation: str = "protocol",
        *,
        vocab_boost: float = 0.15,
    ) -> None:
        self.client = client
        self.conversation = conversation
        self.vocab_boost = vocab_boost

    def assemble(
        self,
        query: str,
        current_turn: int,
        store,
        router,
        ultra_small,
        medium,
        escalation,
        dedup,
        drift_detector,
        budget,
        max_context: int = 8192,
        skip_remembrance: bool = False,
        skip_dedup: bool = False,
        comb_candidates: Optional[list] = None,
        payload_fingerprints: Optional[set] = None,
        dedup_against_payload: bool = True,
        relevance_floor: float = 0.25,
        max_chunk_share: float = 0.5,
        stale_threshold: Optional[int] = None,
        stale_factor: Optional[float] = None,
        drift_prior: str = "off",
        drift_prior_factor: float = 0.5,
        **_: Any,
    ):
        from focal.assembly import AssembledContext, normalize_for_scoring

        comb_candidates = list(comb_candidates or [])
        routing = router.route(query, store.get_turns())
        score_override = None
        if routing.route_to != "ultra_small":
            chunks = list(store.all_chunks()) + comb_candidates
            prefixes = getattr(store, "ingest_block_prefixes", None)
            pairs = [
                (chunk, normalize_for_scoring(chunk.content, prefixes))
                for chunk in chunks
            ]
            pairs = [(chunk, text) for chunk, text in pairs if text]
            scoring_query = normalize_for_scoring(query, prefixes)
            texts = [text for _, text in pairs]
            if routing.route_to == "escalation":
                scores = escalation.process(scoring_query, texts, ultra_small, medium)
            elif routing.route_to == "medium":
                scores = medium.score(scoring_query, texts)
            else:
                scores = ultra_small.score(scoring_query, texts)
            score_override = {
                chunk.id: float(score.relevance_score)
                for (chunk, _), score in zip(pairs, scores)
            }

        fixed_budget = None
        if type(budget).__name__ != "AdaptiveBudget":
            fixed_budget = int(budget.compute(routing.route_to, 0, max_context))

        self.client.load(
            self.conversation,
            [chunk.to_dict() for chunk in store.all_chunks()],
            max_chunks=store.max_chunks,
            sanitize=store.sanitize,
            ingest_block_prefixes=store.ingest_block_prefixes,
        )
        payload = self.client.assemble(
            self.conversation,
            query,
            current_turn=current_turn,
            route=routing.route_to,
            max_context=max_context,
            relevance_floor=relevance_floor,
            max_chunk_share=max_chunk_share,
            drift_threshold=getattr(drift_detector, "threshold", 0.6),
            dedup_threshold=getattr(dedup, "threshold", 0.92),
            stale_threshold=stale_threshold,
            stale_factor=stale_factor,
            skip_remembrance=skip_remembrance,
            skip_dedup=skip_dedup,
            comb_candidates=[candidate.to_json() for candidate in comb_candidates],
            payload_fingerprints=payload_fingerprints,
            dedup_against_payload=dedup_against_payload,
            drift_prior=drift_prior,
            drift_prior_factor=drift_prior_factor,
            vocab_boost=getattr(ultra_small, "vocab_boost", self.vocab_boost),
            budget=fixed_budget,
            score_override=score_override,
        )
        apply_memory_updates(store, payload)
        return AssembledContext(
            content=str(payload.get("content") or ""),
            token_count=int(payload.get("token_count") or 0),
            budget=int(payload.get("budget") or 0),
            chunks_used=len(payload.get("selected_ids") or []),
            routing_decision=routing,
            drift_detected=bool(payload.get("drift_detected")),
            selected_chunk_ids=list(payload.get("selected_ids") or []),
            top_raw_score=float(payload.get("top_raw_score") or 0.0),
            top_chunk_id=payload.get("top_chunk_id"),
            raw_scores={
                key: float(value)
                for key, value in (payload.get("raw_scores") or {}).items()
            },
            payload_dedup_skipped=int(payload.get("payload_dedup_skipped") or 0),
        )
