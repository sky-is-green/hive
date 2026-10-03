"""Local answer validation — one forward pass, engine-agnostic.

The harness owns the validator so that any generator's output can be checked
through one service, whatever engine produced it:

    POST /v1/cascade/validate  {"task": {...}, "candidate": "..."}
      -> {"p_correct": 0.99, "accept": true, "threshold": 0.5, "ms": 42, ...}

Adapters (selected by model id, loaded lazily):

- **Tiny-Jev-1.7B** (default): the trust_remote_code Qwen3 model with a typed
  ``noul`` head; one forward pass, no generation.
- **Cloudflare Clef / Clef-Flash**: the Qwen3.5 backbone + joint schema head
  released Apache-2.0 (Jev/SystemOne compatible); the same ``noul`` question.

**CPU by default.**  A torch ROCm context beside the resident serving engine
can hang this host (two independent ROCm contexts); 1.7B on CPU is far cheaper
than one generation, and a future ternary Clef head is smaller still.
``device="cuda"`` is for offline evaluation with the GPUs actually free.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Optional

#: Default validator checkpoint (Tiny-Jev) and the environment overrides the
#: sidecar uses to swap it (e.g. ``HIVE_VALIDATOR_MODEL=Cloudflare/clef-flash``).
DEFAULT_MODEL = "lostargon/Tiny-Jev-1.7B"
DEFAULT_DEVICE = "cpu"
DEFAULT_THRESHOLD = 0.5

#: Shared validation instruction: one sentence, answer-only (the contract the
#: earlier judge work measured: judging reasoning raises false accepts).
INSTRUCTION = "The candidate answer is correct and complete for the task."


class TinyJevValidator:
    """Tiny-Jev's typed ``noul`` head: P(the candidate answer is correct)."""

    DEFAULT_MODEL = DEFAULT_MODEL
    INSTRUCTION = INSTRUCTION

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str = DEFAULT_DEVICE,
        _model: Any = None,
        _tokenizer: Any = None,
        _torch: Any = None,
    ) -> None:
        self.model_id = model_id
        if _model is not None:
            # Test seam: no transformers/torch import in the unit environment.
            self.tok = _tokenizer
            self.model = _model
            self.torch = _torch
            self.device = device
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        self.device = device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def verdict(self, task: dict, candidate: str) -> tuple[float, float]:
        """P(candidate is correct) and the forward-pass latency in ms."""
        state = {"task": task["prompt"], "candidate_answer": candidate}
        started = time.time()
        with self.torch.no_grad():
            p = float(self.model.noul(self.tok, state, self.INSTRUCTION))
        return p, (time.time() - started) * 1000.0


class ClefValidator:
    """Cloudflare Clef / Clef-Flash through its joint schema head (SystemOne).

    The release ships ``joint_schema_model.py`` with ``load_release_model`` and
    ``systemone``; this adapter asks one ``noul`` question and reads the
    probability.  Import/load happens lazily so the module stays importable in
    environments without transformers.
    """

    DEFAULT_MODEL = "Cloudflare/clef-flash"
    INSTRUCTION = INSTRUCTION

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str = DEFAULT_DEVICE,
        _model: Any = None,
        _processor: Any = None,
        _systemone: Optional[Callable[..., dict]] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        if _model is not None:
            # Test seam.
            self.model = _model
            self.processor = _processor
            self._systemone = _systemone
            return
        import sys

        from huggingface_hub import snapshot_download

        path = snapshot_download(model_id)
        if path not in sys.path:
            sys.path.insert(0, path)
        import joint_schema_model as js  # type: ignore

        self.model, self.processor = js.load_release_model(path, device=device)
        self._systemone = js.systemone

    def verdict(self, task: dict, candidate: str) -> tuple[float, float]:
        request = {
            "model": self.model_id.split("/")[-1],
            "state": {"task": task["prompt"], "candidate_answer": candidate},
            "questions": {
                "verdict": {"type": "noul", "instructions": self.INSTRUCTION},
            },
        }
        started = time.time()
        response = self._systemone(self.model, self.processor, request)
        p = float(response["answers"]["verdict"]["noul"])
        return p, (time.time() - started) * 1000.0


def load_validator(model_id: str, device: str) -> Any:
    """Adapter factory: Clef releases by id, Tiny-Jev otherwise."""
    if "clef" in str(model_id).lower():
        return ClefValidator(model_id, device=device)
    return TinyJevValidator(model_id, device=device)


class ValidatorService:
    """Lazy, serialized validation service over one adapter.

    ``factory`` is injectable for tests; the defaults come from the
    ``HIVE_VALIDATOR_*`` environment so a sidecar operator can swap the model
    without code (``HIVE_VALIDATOR_MODEL`` / ``HIVE_VALIDATOR_DEVICE`` /
    ``HIVE_VALIDATOR_THRESHOLD``).
    """

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        factory: Optional[Callable[[str, str], Any]] = None,
    ) -> None:
        self.model_id = model_id or os.environ.get("HIVE_VALIDATOR_MODEL", DEFAULT_MODEL)
        self.device = device or os.environ.get("HIVE_VALIDATOR_DEVICE", DEFAULT_DEVICE)
        env_threshold = os.environ.get("HIVE_VALIDATOR_THRESHOLD")
        if threshold is not None:
            self.threshold = float(threshold)
        elif env_threshold:
            self.threshold = float(env_threshold)
        else:
            self.threshold = DEFAULT_THRESHOLD
        self._factory = factory or load_validator
        self._lock = threading.RLock()
        self._validator: Any = None
        self._load_ms: Optional[float] = None
        self._checks = 0

    @property
    def loaded(self) -> bool:
        return self._validator is not None

    def _ensure(self) -> Any:
        with self._lock:
            if self._validator is None:
                started = time.time()
                self._validator = self._factory(self.model_id, self.device)
                self._load_ms = (time.time() - started) * 1000.0
            return self._validator

    def verdict(self, task: dict, candidate: str, *,
                threshold: Optional[float] = None) -> dict[str, Any]:
        """Score one candidate; serialized because the adapter is one model."""
        validator = self._ensure()
        with self._lock:
            p, ms = validator.verdict(task, candidate)
            self._checks += 1
        cut = self.threshold if threshold is None else float(threshold)
        return {
            "model": self.model_id,
            "device": self.device,
            "p_correct": round(float(p), 4),
            "accept": bool(p >= cut),
            "threshold": cut,
            "ms": round(float(ms), 1),
            "checks": self._checks,
        }

    def status(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "device": self.device,
            "threshold": self.threshold,
            "loaded": self.loaded,
            "load_ms": round(self._load_ms, 1) if self._load_ms is not None else None,
            "checks": self._checks,
        }
