"""A lazy text-encoder boundary for the controller and memory experiments.

The base Hive install does not depend on torch or sentence-transformers.  This
module only imports the backend when ``embed`` is actually called, on the CPU,
and reports a typed unavailable state instead of pretending to have vectors.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


class EncoderError(RuntimeError):
    """The encoder could not produce a usable vector."""


class EncoderUnavailable(EncoderError):
    """The requested backend or model is not installed/cached."""


class TextEncoder:
    """Protocol-like base class for a local text encoder."""

    model: str
    dimensions: int | None = None

    def encode(self, texts: Sequence[str], *,
               batch_size: int = 8) -> list[list[float]]:
        raise NotImplementedError

    def encode_one(self, text: str) -> list[float]:
        rows = self.encode([text], batch_size=1)
        if len(rows) != 1:
            raise EncoderError("encoder returned the wrong number of rows")
        return rows[0]

    def status(self) -> dict[str, Any]:
        raise NotImplementedError


class SentenceTransformerEncoder(TextEncoder):
    """A sentence-transformers model loaded lazily on the CPU.

    ``local_files_only=True`` is the default: the controller must work from a
    model the user already chose and accepted, not download one as a side
    effect of opening a settings panel.  The CLI exposes an explicit opt-in.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str = "cpu",
        local_files_only: bool = True,
        normalize: bool = True,
        trust_remote_code: bool = False,
        batch_size: int = 8,
    ) -> None:
        self.model = str(model or "").strip()
        if not self.model:
            raise ValueError("an encoder model id is required")
        if device not in ("cpu", "cuda", "rocm", "mps"):
            raise ValueError(f"unsupported encoder device {device!r}")
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.normalize = bool(normalize)
        self.trust_remote_code = bool(trust_remote_code)
        self.batch_size = max(1, int(batch_size))
        self.dimensions = None
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001 - optional dependency
            raise EncoderUnavailable(
                "sentence-transformers is not installed; install the eval "
                "extra or use a served encoder") from exc
        try:
            self._model = SentenceTransformer(
                self.model,
                device=self.device,
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception as exc:  # noqa: BLE001 - a missing/corrupt model is typed
            raise EncoderUnavailable(
                f"cannot load encoder {self.model!r}: {exc}") from exc
        try:
            getter = (getattr(self._model, "get_embedding_dimension", None)
                      or self._model.get_sentence_embedding_dimension)
            dim = getter()
        except Exception:  # noqa: BLE001 - old/new backend shape
            dim = None
        if dim is not None:
            self.dimensions = int(dim)

    def encode(self, texts: Sequence[str], *,
               batch_size: int | None = None) -> list[list[float]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("encode expects a sequence of strings, not one string")
        values = [str(text) for text in texts]
        if not values:
            return []
        self._ensure()
        try:
            result = self._model.encode(
                values,
                batch_size=max(1, int(batch_size or self.batch_size)),
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001 - report encoder failure cleanly
            raise EncoderError(f"encoder failed: {exc}") from exc
        rows = result.tolist() if hasattr(result, "tolist") else result
        cleaned: list[list[float]] = []
        for row in rows:
            vector = [float(value) for value in row]
            if not vector:
                raise EncoderError("encoder returned an empty vector")
            if any(not math.isfinite(value) for value in vector):
                raise EncoderError("encoder returned a non-finite vector")
            if self.dimensions is not None and len(vector) != self.dimensions:
                raise EncoderError(
                    f"encoder dimension mismatch: expected {self.dimensions}, "
                    f"got {len(vector)}")
            cleaned.append(vector)
        if len(cleaned) != len(values):
            raise EncoderError("encoder returned the wrong number of rows")
        return cleaned

    def status(self) -> dict[str, Any]:
        available = False
        try:
            import importlib.util

            available = importlib.util.find_spec("sentence_transformers") is not None
        except Exception:  # noqa: BLE001
            available = False
        return {
            "kind": "sentence-transformers",
            "model": self.model,
            "device": self.device,
            "local_files_only": self.local_files_only,
            "available": available,
            "loaded": self._model is not None,
            "dimensions": self.dimensions,
            "normalize": self.normalize,
        }


def encoder_from_settings(settings: dict[str, Any], *,
                          local_files_only: bool = True) -> TextEncoder:
    """Build the configured local encoder without importing its backend.

    Served encoders are intentionally not guessed at here; the Studio already
    has a served-embedding drone, and the controller will get a typed served
    adapter when it needs one.
    """
    block = settings.get("encoder") if isinstance(settings, dict) else None
    block = block if isinstance(block, dict) else {}
    model = str(block.get("embedding_model") or "").strip()
    backend = str(block.get("embedding_backend") or "local").strip().lower()
    if backend == "served":
        raise EncoderUnavailable(
            "the controller's local encoder adapter does not serve yet; "
            "choose embedding_backend=local")
    return SentenceTransformerEncoder(model, local_files_only=local_files_only)
