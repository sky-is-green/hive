"""Unit tests for the partner-head transfer evaluator.

The heavy frozen-head check is still local-only: it uses the existing Bonsai
J-Bench states and asserts that the exported weights reproduce their recorded
AUC exactly.  It does not start a model server.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.cascade.eval_partner_transfer import (
    apply_head,
    load_states,
    make_preview,
    sigmoid,
)
from experiments.cascade.readout_judge import auc

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "experiments/cascade/results/jev-readout-20261006/probe-states-bonsai27b.npz"
HEAD = REPO / "experiments/cascade/results/jev-readout-20261006/bonsai-2-27b-decision-head.json"
READOUT = REPO / (
    "experiments/cascade/results/decision-readout-r2-20261017/"
    "readout-states-intern-decision-0.8b-r2.npz"
)


def test_sigmoid_handles_extreme_scores_without_overflow():
    out = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert out.tolist() == [0.0, 0.5, 1.0]


def test_preview_uses_the_real_gate_renderer_and_marks_eval_labels():
    text = make_preview(
        [
            {"item_id": "t1", "bucket": "code", "prompt": "write add",
             "candidate": "print(1)", "p": 0.1, "label": 0},
            {"item_id": "t2", "bucket": "qa", "prompt": "q",
             "candidate": "A", "p": 0.3, "label": 1},
        ],
        threshold=0.58,
        api_target="http://api.example/v1",
        api_model="frontier-test",
    )
    assert "HUMAN GATE — local answer rejected" in text
    assert "first trigger" in text
    assert "[s]olve" in text and "[e]scalate" in text
    assert "write add" in text and "print(1)" in text
    assert "http://api.example/v1" in text and "frontier-test" in text
    assert "eval label: 0" in text and "eval label: 1" in text


def test_frozen_bonsai_head_reproduces_the_recorded_jbench_test_auc():
    if not (BASE.exists() and HEAD.exists() and READOUT.exists()):
        pytest.skip("Bonsai J-Bench artifacts are not present")
    head = json.loads(HEAD.read_text())
    states = load_states(BASE)
    with np.load(READOUT, allow_pickle=True) as z:
        ids = [str(x) for x in z["item_ids"]]
        splits = [str(x) for x in z["splits"]]
        sources = [str(x) for x in z["sources"]]
        labels = [int(x) for x in z["labels"]]
    selected = [
        n for n, item_id in enumerate(ids)
        if item_id in states and splits[n] == "test" and sources[n] == "bench"
    ]
    x = np.stack([states[ids[n]] for n in selected])
    p = apply_head(head, x)
    observed = auc(p.tolist(), [labels[n] for n in selected])

    assert len(selected) == 116
    assert observed == pytest.approx(head["test"]["auc"], abs=1e-10)
    assert observed == pytest.approx(0.9881093935790726, abs=1e-10)
