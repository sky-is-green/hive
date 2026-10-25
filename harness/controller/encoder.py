"""A lazy text-encoder boundary for the controller and memory experiments.

The base Hive install does not depend on torch or sentence-transformers.  This
module only imports the backend when ``embed`` is actually called, on the CPU,
and reports a typed unavailable state instead of pretending to have vectors.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Sequence


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


class EncoderLifecycle:
    """Lazy encoder cache with an optional idle-unload timer.

    The timer is a daemon thread, so it never keeps Hive alive.  It only drops
    this service's reference to the model; it never touches the serving engine
    or its VRAM.  A request that arrives after expiry creates a fresh encoder
    through the same factory.
    """

    def __init__(
        self,
        factory: Callable[[], TextEncoder],
        *,
        idle_unload: bool = True,
        idle_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        timer_factory: Callable[..., Any] = threading.Timer,
    ) -> None:
        self._factory = factory
        self._clock = clock
        self._timer_factory = timer_factory
        self._lock = threading.RLock()
        self._encoder: TextEncoder | None = None
        self._timer: Any = None
        self._last_used: float | None = None
        self._generation = 0
        self._idle_unload = bool(idle_unload)
        self._idle_seconds = max(0.1, float(idle_seconds))

    def configure(self, *, idle_unload: bool,
                  idle_seconds: float) -> None:
        """Apply settings; an already-loaded model gets a fresh idle window."""
        with self._lock:
            self._idle_unload = bool(idle_unload)
            self._idle_seconds = max(0.1, float(idle_seconds))
            self._cancel_locked()
            if self._encoder is not None:
                self._last_used = self._clock()
                self._schedule_locked()

    def get(self) -> TextEncoder:
        with self._lock:
            if self._encoder is None:
                self._encoder = self._factory()
            self._last_used = self._clock()
            self._schedule_locked()
            return self._encoder

    def touch(self) -> None:
        """Reset the idle window after a successful encode."""
        with self._lock:
            if self._encoder is None:
                return
            self._last_used = self._clock()
            self._schedule_locked()

    def unload(self) -> bool:
        """Drop the model immediately; return whether one was loaded."""
        with self._lock:
            self._cancel_locked()
            was_loaded = self._encoder is not None
            self._encoder = None
            self._last_used = None
            self._generation += 1
            return was_loaded

    def close(self) -> None:
        self.unload()

    def status(self) -> dict[str, Any]:
        with self._lock:
            remaining = None
            if (self._encoder is not None and self._idle_unload
                    and self._last_used is not None):
                remaining = max(
                    0.0, self._last_used + self._idle_seconds - self._clock())
            return {
                "loaded": self._encoder is not None,
                "idle_unload": self._idle_unload,
                "idle_seconds": round(self._idle_seconds, 3),
                "idle_remaining_s": (round(remaining, 3)
                                     if remaining is not None else None),
            }

    def _cancel_locked(self) -> None:
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:  # noqa: BLE001 - a dead timer is harmless
                pass
            self._timer = None

    def _schedule_locked(self) -> None:
        self._cancel_locked()
        if (self._encoder is None or not self._idle_unload
                or self._last_used is None):
            return
        self._generation += 1
        generation = self._generation
        deadline = self._last_used + self._idle_seconds
        delay = max(0.0, deadline - self._clock())
        timer = self._timer_factory(delay, self._expire,
                                    args=(deadline, generation))
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _expire(self, deadline: float, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._timer = None
            if not self._idle_unload or self._encoder is None:
                return
            now = self._clock()
            if now + 1e-6 < deadline or (
                    self._last_used is not None and self._last_used > deadline):
                self._schedule_locked()
                return
            self._encoder = None
            self._last_used = None
            self._generation += 1


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
