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
- **Clef-Flash ternary** (``clef-ternary``): the deployed PQ2_0 body with the
  trained rank-512 corrections merged in, run by the llama.cpp CPU bridge with
  the joint head sidecar in torch fp32 on CPU; the 9B never enters torch, so
  the GPUs stay free and the check is engine-co-resident.

**CPU by default.**  A torch ROCm context beside the resident serving engine
can hang this host (two independent ROCm contexts); 1.7B on CPU is far cheaper
than one generation, and a future ternary Clef head is smaller still.
``device="cuda"`` is for offline evaluation with the GPUs actually free.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
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

    ``text_only`` (default) loads the backbone and the joint head directly
    with a tokenizer-only processor: the released multimodal processor pulls
    ``torchvision`` for its video processor, which text validation never uses.
    Set ``text_only=False`` with the full processor stack installed to accept
    images/videos.
    """

    DEFAULT_MODEL = "Cloudflare/clef-flash"
    INSTRUCTION = INSTRUCTION

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str = DEFAULT_DEVICE,
        text_only: bool = True,
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

        if text_only:
            import json as _json
            import types

            import torch
            from safetensors.torch import load_file
            from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

            backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
                path, dtype=torch.bfloat16, device_map={"": device})
            backbone.config.use_cache = False
            head = js.JointSchemaHead(
                **_json.loads((Path(path) / "joint_head_config.json").read_text()))
            head.load_state_dict(
                load_file(Path(path) / "joint_head.safetensors"), strict=True)
            head = head.to(device=device, dtype=torch.bfloat16)
            self.model = js.ClefModel(backbone, head).eval()
            self.processor = types.SimpleNamespace(
                tokenizer=AutoTokenizer.from_pretrained(path))
        else:
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


#: Ternary Clef release (packaged locally): deployed PQ2_0 body + merged
#: corrections, served through the fork's CPU bridge.
REPO_ROOT = Path(__file__).resolve().parents[2]              # hivebench/
WORKSPACE = REPO_ROOT.parent                                  # work/
DEFAULT_TERNARY_DIR = WORKSPACE / "models" / "clef-flash-ternary"
DEFAULT_TERNARY_BODY = "corrections/packaged/clef-flash-PQ2_0-corr-r512-g128-step78.gguf"
DEFAULT_TERNARY_BRIDGE = REPO_ROOT / "tools" / "clef-bridge" / "clef_embed"


class ClefTernaryValidator:
    """Clef-Flash ternary corrections through the llama.cpp CPU bridge.

    ``model_id`` may be a merged-body ``.gguf`` path, a model directory, or a
    label; the packaged release under ``models/clef-flash-ternary/`` is the
    default (env overrides: ``HIVE_TERNARY_DIR``, ``HIVE_TERNARY_BODY``,
    ``HIVE_TERNARY_BRIDGE``, ``HIVE_TERNARY_NCTX``).

    Per check: encode the one ``noul`` question with the release tokenizer, send
    the exact ids to a resident ``clef_embed --serve`` process (fresh context,
    no cross-check state), and run the joint head in torch fp32 on CPU over the
    per-token hidden states.
    """

    INSTRUCTION = INSTRUCTION

    def __init__(
        self,
        model_id: str = "clef-flash-ternary",
        *,
        device: str = DEFAULT_DEVICE,
        _runner: Any = None,
        _head_fn: Any = None,
        _encode_fn: Any = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._proc: Any = None
        self._work: Any = None
        self._log: Any = None
        if _head_fn is not None:
            # Test seam: injected encode/run/score, no torch and no bridge.
            self._runner = _runner
            self._head_fn = _head_fn
            self._encode_fn = _encode_fn
            return
        if device not in ("cpu", "auto"):
            raise ValueError(
                "ClefTernaryValidator runs the CPU bridge; use device='cpu' "
                "(the non-display-card bridge is the future GPU lane)")
        self._load()

    def _load(self) -> None:
        import json
        import subprocess
        import sys
        import tempfile

        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        explicit = Path(str(self.model_id)).expanduser()
        env_dir = os.environ.get("HIVE_TERNARY_DIR")
        env_body = os.environ.get("HIVE_TERNARY_BODY")
        if explicit.is_file():
            self._body = explicit
            model_dir = explicit.parent
        elif explicit.is_dir():
            model_dir = explicit
            self._body = Path(env_body) if env_body else model_dir / DEFAULT_TERNARY_BODY
        else:
            model_dir = Path(env_dir) if env_dir else DEFAULT_TERNARY_DIR
            self._body = Path(env_body) if env_body else model_dir / DEFAULT_TERNARY_BODY
        bridge = Path(os.environ.get("HIVE_TERNARY_BRIDGE", DEFAULT_TERNARY_BRIDGE))
        if not self._body.is_file():
            raise FileNotFoundError(f"ternary Clef body not found: {self._body}")
        if not bridge.is_file():
            raise FileNotFoundError(
                f"clef bridge not found: {bridge} (build tools/clef-bridge)")
        self.model_dir = model_dir
        self.bridge = bridge

        head_dir = model_dir / "hf-head"
        if str(head_dir) not in sys.path:
            sys.path.insert(0, str(head_dir))
        import joint_schema_model as js  # type: ignore
        self._js = js
        cfg = json.loads((head_dir / "joint_head_config.json").read_text())
        head = js.JointSchemaHead(**cfg)
        head.load_state_dict(
            load_file(str(head_dir / "joint_head.safetensors")), strict=True)
        self._head = head.to(device="cpu", dtype=torch.float32).eval()
        weights = load_file(str(model_dir / "lm_head.safetensors"))
        key = "lm_head.weight" if "lm_head.weight" in weights else next(iter(weights))
        self._lm_head = weights[key].to(device="cpu", dtype=torch.float32)
        self._tok = AutoTokenizer.from_pretrained(str(head_dir))
        self._hidden = int(cfg.get("hidden_size", 4096))

        self._n_ctx = int(os.environ.get("HIVE_TERNARY_NCTX", "1024"))
        self._work = Path(tempfile.mkdtemp(prefix="clef-ternary-"))
        self._log = open(self._work / "bridge.stderr", "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            [str(bridge), str(self._body), "--serve", str(self._n_ctx)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
        )
        ready = self._proc.stdout.readline().strip()
        if not ready.startswith("READY"):
            self.close()
            raise RuntimeError(f"clef bridge failed to start: {ready!r}")

        def encode(prompt: str, candidate: str) -> Any:
            request = {
                "model": "clef-flash",
                "state": {"task": prompt, "candidate_answer": candidate},
                "questions": {"verdict": {"type": "noul", "instructions": INSTRUCTION}},
            }
            return self._js.encode_record(self._tok, request, max_length=16384)

        self._encode_fn = encode
        self._runner = self._request
        self._head_fn = self._score

    def _request(self, ids: Any) -> Any:
        import numpy as np

        tok_path = self._work / "q.tok"
        out_path = self._work / "q.bin"
        tok_path.write_text(" ".join(str(int(t)) for t in ids), encoding="utf-8")
        self._proc.stdin.write(f"{tok_path} {out_path}\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline().strip()
        if not line.startswith("OK "):
            raise RuntimeError(f"clef bridge request failed: {line!r}")
        hidden = np.fromfile(out_path, dtype=np.float32)
        return hidden.reshape(-1, self._hidden)

    def _score(self, hidden: Any, enc: Any) -> float:
        import numpy as np
        import torch

        ids = torch.tensor(enc.input_ids, dtype=torch.long)
        with torch.no_grad():
            logits = self._head(
                torch.from_numpy(np.ascontiguousarray(hidden)).unsqueeze(0),
                ids.unsqueeze(0),
                torch.ones(1, ids.numel(), dtype=torch.long),
                [enc],
                self._lm_head,
            )[0][0]
        return float(logits.float().softmax(-1)[0])

    def verdict(self, task: dict, candidate: str) -> tuple[float, float]:
        """P(candidate is correct) and the bridge+head latency in ms."""
        enc = self._encode_fn(task["prompt"], candidate)
        started = time.time()
        hidden = self._runner(enc.input_ids)
        p = self._head_fn(hidden, enc)
        return float(p), (time.time() - started) * 1000.0

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._work is not None:
            import shutil

            shutil.rmtree(self._work, ignore_errors=True)
            self._work = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def load_validator(model_id: str, device: str) -> Any:
    """Adapter factory: ternary/deployed Clef by id, Tiny-Jev otherwise."""
    low = str(model_id).lower()
    if "ternary" in low:
        return ClefTernaryValidator(model_id, device=device)
    if "clef" in low:
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
