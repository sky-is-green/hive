"""The learned gate (C): a third trigger beside the deterministic detectors.

The deterministic verdict (``consistency`` + the task's own tests) catches
most local wrongs; the measured blind spot is the remainder — internally
consistent wrong answers the detectors stay silent on (3 of the 9 stored
wrongs on the organic-66, ``experiments/cascade/results/
scion176b-self-organic-20261018/MARGINAL.md``).  The learned gate scores a
candidate with a frozen head over a small decision model's hidden state and
fires — as *one more rejection finding, nothing else* — when the score is
below the frozen threshold.

The contract it honours (``experiments/cascade/LEARNED-GATE-PLAN.md``):

- **Blind-spot only.**  The runner scores a candidate only when the
  deterministic detectors are silent; a fired gate enters the verdict as an
  ordinary finding (same greedy repair, same ladder).  This module ships the
  scorer only — the trigger policy lives in the runner wiring.
- **Gold-free.**  The scorer sees (question, candidate) and nothing else —
  no ``expect``, no tests, no task metadata.
- **Frozen.**  Threshold 0.5 (the head's own boundary; anything tighter is
  slice-tuned) and head L0.75 (precision-first).  No per-run tuning.
- **Strict re-verify.**  After a repair the gate triggered, the repaired
  answer is re-scored; a persistent flag stays flagged.

Path B of the plan (v1): the readout backbone is
``internlm/Intern-Decision-0.8B`` on **CPU** (bfloat16, ~2 GiB resident) —
CPU-only on purpose: a ROCm context beside the live 176B serve is the
documented hard-reset hazard, so the device is not configurable here.  One
row per (question, candidate) goes through
``experiments/cascade/decision_readout.py``'s extraction path — the exact row
builder the heads were fitted on — and the frozen r2 head applies client-side
from ``experiments/cascade/results/decision-readout-r2-20261017/``.
Measured cost: 0.44 s/score median (the 1315-item r2 extraction).  The
backbone loads lazily on first use; importing this module (or the package)
runs nothing heavy.  Path A (the 176B serve's own ``/embedding`` readout) is
a future source behind the same interface.

Unit tests: :mod:`tests.unit.test_cascade_learned_gate` reproduce the stored
fitted p from the committed states/head npz; one live extraction cross-check
against a stored item is opt-in via ``HIVE_GATE_LIVE=1`` (it loads the
backbone).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
R2_DIR = REPO_ROOT / "experiments" / "cascade" / "results" / "decision-readout-r2-20261017"
#: the frozen r2 heads (ridge+Platt per depth, ridge over the concat, MLP).
HEADS_FILE = R2_DIR / "readout-head-intern-decision-0.8b-r2.npz"

#: path-B backbone: the smallest Intern-Decision checkpoint (the frozen
#: heads' training states were extracted from this exact id).
MODEL_ID = "internlm/Intern-Decision-0.8B"

#: depth fractions of the decision state the r2 extraction capped at, and
#: their state labels — the npz/head-pfrefix convention (``f"L{f:g}"``).
DEPTHS: tuple[float, ...] = (0.5, 0.75, 1.0)
STATE_LABELS: tuple[str, ...] = tuple(f"L{f:g}" for f in DEPTHS)

#: layer count of the r2 extraction's backbone
#: (``R2_DIR/extract-intern-decision-0.8b-r2.json``).  The heads read these
#: exact layers (12/18/24 for 24 layers), so a same-named checkpoint with a
#: different depth must fail loudly instead of scoring nonsense.
EXPECTED_LAYERS = 24

#: the frozen operating point (plan: the head's own boundary; the tighter
#: cuts that buy +1 catch sit 2e-4 from a correct answer — slice-tuned).
DEFAULT_HEAD = "L0.75"
THRESHOLD = 0.5

#: the r2 head names readout-head-intern-decision-0.8b-r2.npz holds, under
#: the names the plan and the marginal table use.
KNOWN_HEADS: tuple[str, ...] = STATE_LABELS + ("ridge-concat", "mlp-concat")

_RIDGE_PREFIX = {
    "L0.5": "ridge_L0.5",
    "L0.75": "ridge_L0.75",
    "L1": "ridge_L1",
    "ridge-concat": "ridge_concat",
}


def _logistic(x: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic, identical to the fitting code's."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class ReadoutHead:
    """One frozen head from the r2 npz, applied client-side to 0.8B states.

    ``score`` takes the decision-position states for one item (a mapping
    state-label → vector) and returns the Platt-scaled P(candidate correct).
    Ridge heads (``L0.5``/``L0.75``/``L1``) read one 1024-d layer state;
    ``ridge-concat``/``mlp-concat`` read the ordered concat of all three.
    """

    def __init__(self, name: str = DEFAULT_HEAD, path: Path | str = HEADS_FILE):
        if name not in KNOWN_HEADS:
            raise ValueError(
                f"unknown learned-gate head {name!r} (known: {', '.join(KNOWN_HEADS)})")
        self.name = name
        self.path = Path(path)
        self._h = np.load(self.path, allow_pickle=False)

        self.concat = name not in STATE_LABELS
        self.inputs: tuple[str, ...] = STATE_LABELS if self.concat else (name,)
        self.kind = "mlp" if name == "mlp-concat" else "ridge"
        if self.kind == "ridge":
            prefix = _RIDGE_PREFIX[name]
            missing = [k for k in (f"{prefix}_w", f"{prefix}_mu", f"{prefix}_sd",
                                   f"{prefix}_bias", f"{prefix}_platt")
                       if k not in self._h.files]
            if missing:
                raise KeyError(f"head file {self.path.name} lacks {missing} for {name!r}")

    def score(self, feats: Mapping[str, np.ndarray]) -> float:
        """Apply the frozen head to one item's decision states."""
        missing = [k for k in self.inputs if k not in feats]
        if missing:
            raise KeyError(
                f"head {self.name!r} needs decision state(s) {missing}; got "
                f"{sorted(feats)}")
        if self.concat:
            x = np.concatenate([np.asarray(feats[k], dtype=np.float64)
                                for k in STATE_LABELS])
        else:
            x = np.asarray(feats[self.name], dtype=np.float64)
        if self.kind == "ridge":
            prefix = _RIDGE_PREFIX[self.name]
            s = (((x - self._h[f"{prefix}_mu"]) / self._h[f"{prefix}_sd"])
                 @ self._h[f"{prefix}_w"] + float(self._h[f"{prefix}_bias"]))
            a, b = (float(v) for v in self._h[f"{prefix}_platt"])
            return float(_logistic(a * s + b))
        hidden = np.maximum(
            ((x - self._h["mlp_mu"]) / self._h["mlp_sd"]) @ self._h["mlp_0.weight"].T
            + self._h["mlp_0.bias"], 0.0)
        logit = (hidden @ self._h["mlp_2.weight"].T + self._h["mlp_2.bias"]).ravel()
        return float(_logistic(logit)[0])


class LearnedGate:
    """Path-B scorer: ``score(question, candidate) -> (p_correct, ms)``.

    Loads the 0.8B readout backbone on CPU once, lazily, then scores one
    (question, candidate) pair per forward pass through the extraction row
    builder the heads were fitted on and applies the frozen head to the
    decision-position states.  ``ms`` measures the extraction + head math of
    a warm call (the backbone load is paid before the first score and is not
    part of it, so reported ms stays comparable with the 0.44 s extraction
    median).

    ``states`` is a test seam — a callable producing the decision states for
    a (question, candidate), replacing the backbone entirely (the unit tests
    feed the stored r2 states through it); the real scorer never sets it.
    """

    def __init__(self, head: str = DEFAULT_HEAD, head_path: Path | str = HEADS_FILE,
                 model_id: str = MODEL_ID, dtype: str = "bfloat16",
                 states: Callable[[str, str], Mapping[str, np.ndarray]] | None = None):
        self.head = ReadoutHead(head, head_path)
        self.model_id = model_id
        self.dtype = dtype
        self._states_source = states
        self._impl: tuple | None = None
        self._dr = None  # the decision_readout module, bound at first backbone load

    # -- backbone (lazy; CPU-only on purpose) --------------------------------

    def _load(self):
        if self._impl is None:
            from experiments.cascade import decision_readout as dr
            self._dr = dr
            self._impl = dr.load_backend(self.model_id, "cpu", self.dtype)
        return self._impl

    # -- scoring -------------------------------------------------------------

    def states(self, question: str, candidate: str) -> dict[str, np.ndarray]:
        """Decision-position states for one candidate, keyed by ``STATE_LABELS``.

        One row through the fitted extraction path (the Intern-Decision
        decision-marker prompt, positions from ``encode``), one forward pass,
        the three depth-fraction states at the decision position.
        """
        if self._states_source is not None:
            return dict(self._states_source(question, candidate))

        inference, backend, torch = self._load()
        request = {
            "state": {"task": question, "candidate_answer": candidate},
            "questions": {"verdict": dict(self._dr.VERDICT_QUESTION)},
        }
        row = inference.validate_request(request)
        compiled, batch, positions = backend.encode(row)
        pos = int(positions[0])

        with torch.inference_mode():
            try:
                out = backend.model(**batch, use_cache=False,
                                    output_hidden_states=True,
                                    logits_to_keep=positions)
            except TypeError:  # older forwards without logits_to_keep
                out = backend.model(**batch, use_cache=False,
                                    output_hidden_states=True)
        hidden_states = out.hidden_states
        n_layers = len(hidden_states) - 1
        if n_layers != EXPECTED_LAYERS:
            raise RuntimeError(
                f"{self.model_id!r} reports {n_layers} transformer layers but the frozen "
                "heads were fitted on the "
                f"{EXPECTED_LAYERS}-layer r2 extraction — refusing to score")
        feats = {f"L{label}": hidden_states[idx][0, pos].float().numpy()
                 for label, idx in self._dr.layer_map(list(DEPTHS), n_layers)}
        return feats

    def score(self, question: str, candidate: str) -> tuple[float, float]:
        """Return ``(p_correct, ms)`` for one candidate — the frozen gate.

        ``p < THRESHOLD`` is a fired gate (the runner's trigger policy applies
        it blind-spot-only and as one more rejection finding).
        """
        started = time.perf_counter()
        feats = self.states(question, candidate)
        p = self.head.score(feats)
        return p, (time.perf_counter() - started) * 1000.0
