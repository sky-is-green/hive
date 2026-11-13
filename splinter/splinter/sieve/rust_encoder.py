"""Rust ONNX encoder client — the native Phase 3 embedding backend.

The daemon/sidecar can point at `hive-encoder serve` on loopback. The drone has
the same `embed`/`score` interface as `UltraSmallDrone`, so the Splinter memory
pipeline can run without importing torch or sentence-transformers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import requests

from sieve.embedding_cache import EmbeddingCache
from sieve.scores import ChunkScore
from sieve.ultra_small import cosine_similarity_rows
from sieve.vocabulary import Vocabulary

DEFAULT_URL = "http://127.0.0.1:8172"


class RustEncoderError(RuntimeError):
    """The native encoder service is unavailable or rejected a request."""


class RustEncoderDrone:
    """Client for `hive-encoder serve` (JSON, batched)."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        vocab: Optional[Vocabulary] = None,
        vocab_boost: float = 0.15,
        cache: Optional[EmbeddingCache] = None,
        timeout: float = 30.0,
        confidence_mode: str = "off",
    ) -> None:
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.vocab = vocab if vocab is not None else Vocabulary.load("code", "general")
        self.vocab_boost = vocab_boost
        self.cache = cache if cache is not None else EmbeddingCache()
        self.cache.model = "hive-encoder:paraphrase-MiniLM-L3-v2"
        self.timeout = timeout
        self.confidence_mode = confidence_mode
        self._dimension: Optional[int] = None
        import sys

        print(
            f"[encoder] native hive-encoder at {self.base_url}",
            file=sys.stderr,
            flush=True,
        )

    def _post(self, texts: list[str]) -> list[list[float]]:
        try:
            response = requests.post(
                f"{self.base_url}/encode",
                json={"texts": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RustEncoderError(
                f"native encoder unavailable at {self.base_url}: {exc}"
            ) from exc
        except ValueError as exc:
            raise RustEncoderError(f"native encoder returned invalid JSON: {exc}") from exc
        vectors = payload.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RustEncoderError(
                f"native encoder returned {len(vectors or [])} vectors for {len(texts)} texts"
            )
        self._dimension = int(payload.get("dimension") or len(vectors[0]))
        return [[float(value) for value in vector] for vector in vectors]

    def embed(self, text: str) -> np.ndarray:
        cached = self.cache.get(text)
        if cached is not None:
            return np.asarray(cached)
        vector = np.asarray(self._post([text])[0], dtype=np.float32)
        self.cache.put(text, vector)
        return vector

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        missing: list[str] = []
        missing_index: list[int] = []
        out: list[Optional[np.ndarray]] = [None] * len(texts)
        for index, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                out[index] = np.asarray(cached)
            else:
                missing.append(text)
                missing_index.append(index)
        if missing:
            vectors = self._post(missing)
            for index, text, vector in zip(missing_index, missing, vectors):
                array = np.asarray(vector, dtype=np.float32)
                self.cache.put(text, array)
                out[index] = array
        return [vector for vector in out if vector is not None]

    def score(self, query: str, chunks: list[str]) -> list[ChunkScore]:
        if not chunks:
            return []
        vectors = self.embed_batch([query, *chunks])
        q = vectors[0]
        rows = np.stack(vectors[1:])
        sims = cosine_similarity_rows(q, rows)
        results = []
        for index, similarity in enumerate(sims):
            boosted = float(similarity) + self._boost(chunks[index])
            results.append(ChunkScore(index, boosted, 1.0, source="rust_encoder"))
        return results

    def _boost(self, text: str) -> float:
        if self.vocab is not None and self.vocab.matches(text):
            return float(self.vocab_boost)
        return 0.0
