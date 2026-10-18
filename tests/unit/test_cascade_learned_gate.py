"""Head-math and seam tests for the learned gate scorer.

Everything here is numpy arithmetic over committed artifacts — the frozen r2
heads applied to the r2 states they were fitted on — so no model, no network,
no GPU.  The point of these tests is the contract ``LEARNED-GATE-PLAN.md``
promised: the harness's head math reproduces the stored fitted p exactly, the
frozen 0.5 cut behaves as the plan's operating table records, and the scorer
reports ``(p, ms)`` with a deterministic p.

The one live test (opt-in via ``HIVE_GATE_LIVE=1``, because it loads the
intern-decision-0.8B backbone on CPU) cross-checks one stored item's states
and score against the committed extraction.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import experiments.cascade.fusion_probe as fusion_probe
from harness.cascade.learned_gate import (
    DEFAULT_HEAD,
    EXPECTED_LAYERS,
    KNOWN_HEADS,
    STATE_LABELS,
    THRESHOLD,
    LearnedGate,
    ReadoutHead,
)

R2 = "experiments/cascade/results/decision-readout-r2-20261017"
STATES_FILE = f"{R2}/readout-states-intern-decision-0.8b-r2.npz"
HEADS_FILE = f"{R2}/readout-head-intern-decision-0.8b-r2.npz"
MARGINAL_FILE = ("experiments/cascade/results/scion176b-self-organic-20261018/"
                 "marginal-catch.json")
HARVEST_FILE = ("experiments/cascade/results/decision-harvest-r2-20261017/"
                "harvest-bench-r2.jsonl")
#: the blind spot (the three detector-missed stored wrongs) and the universal
#: miss — the items every anchor test below is pinned to.
MISSED = ("gsm8k-0009:hs2", "gsm8k-0015:hs2", "math-0020:hs2")
UNIVERSAL_MISS = "humaneval-0001:hs1"
#: every learned head keeps the universal miss above this p (MARGINAL.md).
UNIVERSAL_MISS_MIN_P = 0.76


@pytest.fixture(scope="module")
def r2():
    """Stored r2 states + the fitted p the analysis code recomputes from them."""
    if not (os.path.exists(STATES_FILE) and os.path.exists(HEADS_FILE)):
        pytest.skip("r2 states/head npz not present on this machine")
    meta, p_fitted = fusion_probe.load_r2()
    z = np.load(STATES_FILE, allow_pickle=False)
    states = {label: z[label] for label in STATE_LABELS}
    return meta, p_fitted, states


def _row(r2_data, item_id: str) -> tuple[int, dict, dict]:
    """Index of a stored item inside the r2 arrays, plus the fitted p."""
    meta, p_fitted, states = r2_data
    ids = [str(i) for i in meta["item_ids"]]
    n = ids.index(item_id)
    return n, p_fitted, states


def _feats_at(states, n: int) -> dict:
    return {label: states[label][n] for label in STATE_LABELS}


def _gate_over_row(states, n: int) -> LearnedGate:
    """A LearnedGate whose states seam feeds one stored r2 row."""
    return LearnedGate(states=lambda q, c: _feats_at(states, n))


# --------------------------------------------------------------------------
# the core claim: the harness head math reproduces the stored fitted p
# --------------------------------------------------------------------------

@pytest.mark.parametrize("head", KNOWN_HEADS)
def test_head_math_reproduces_stored_fitted_p(head, r2):
    meta, p_fitted, states = r2
    gate_head = ReadoutHead(head)
    n_items = len(meta["item_ids"])
    got = np.array([
        gate_head.score(_feats_at(states, i)) for i in range(n_items)
    ])
    want = np.asarray(p_fitted[head], dtype=np.float64)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_state_labels_match_the_stored_artifacts(r2):
    meta, p_fitted, states = r2
    assert tuple(states) == STATE_LABELS
    assert tuple(p_fitted) == ("L0.5", "L0.75", "L1", "ridge-concat",
                               "mlp-concat", "p_raw")
    assert all(states[label].shape[1] == 1024 for label in STATE_LABELS)


# --------------------------------------------------------------------------
# the frozen 0.5 cut behaves as the plan's operating table records
# --------------------------------------------------------------------------

@pytest.mark.parametrize("head,expected_fires", [
    # the plan's frozen operating point: L0.75 = precision-first (+1 catch,
    # 0 extra flags); math-0020 sits at 0.5912 — the +2@0 point needs the
    # tighter threshold the plan rejected as slice-tuned.
    ("L0.75", {"gsm8k-0015:hs2"}),
    # the recall dial: all three detector-missed wrongs fire, paid in flags.
    ("L0.5", set(MISSED)),
    # bench-strong but organic-noisier: two of the three.
    ("mlp-concat", {"gsm8k-0015:hs2", "math-0020:hs2"}),
    # ridge over the concat: its unique catch among the missed three is
    # math-0020.
    ("ridge-concat", {"math-0020:hs2"}),
])
def test_threshold_cut_separates_as_recorded(head, expected_fires, r2):
    meta, p_fitted, states = r2
    got = set()
    for item_id in MISSED:
        n, _, _ = _row(r2, item_id)
        if ReadoutHead(head).score(_feats_at(states, n)) < THRESHOLD:
            got.add(item_id)
    assert got == expected_fires


@pytest.mark.parametrize("head", ("L0.5", "L0.75", "L1", "ridge-concat",
                                  "mlp-concat"))
def test_universal_miss_stays_missed(head, r2):
    """humaneval-0001:hs1 stays above the line under every learned head."""
    meta, p_fitted, states = r2
    n, _, _ = _row(r2, UNIVERSAL_MISS)
    got = ReadoutHead(head).score(_feats_at(states, n))
    assert got >= UNIVERSAL_MISS_MIN_P, f"{head}: got p={got}"


def test_committed_marginal_table_joins_on_these_p(r2):
    """The analyzer's joinpoint equals the stored head p of its boundary item.

    cap 0 for L0.75 accepts the two wrongs up to 0.5912 (the pareto point the
    plan rejected); the threshold recorded there is exactly p(math-0020:hs2)
    under L0.75 — the identity that ties the committed table to these heads.
    """
    if not os.path.exists(MARGINAL_FILE):
        pytest.skip("marginal-catch.json not present on this machine")
    marginal = json.load(open(MARGINAL_FILE))["signals"]["0.8B-L0.75"]["marginal"]
    entry = marginal["0"]
    assert set(entry["caught"]) == {"gsm8k-0015:hs2", "math-0020:hs2"}
    n20, _, states = _row(r2, "math-0020:hs2")
    n15, _, _ = _row(r2, "gsm8k-0015:hs2")
    boundary = ReadoutHead("L0.75").score(_feats_at(states, n20))
    p15 = ReadoutHead("L0.75").score(_feats_at(states, n15))
    assert entry["threshold"] == pytest.approx(boundary, abs=1e-12)
    assert p15 < boundary <= 0.5 * 1.2  # 0.5912: just above the line


# --------------------------------------------------------------------------
# the scorer's surface contract
# --------------------------------------------------------------------------

def test_score_returns_p_and_ms_with_deterministic_p(r2):
    n, p_fitted, states = _row(r2, "gsm8k-0015:hs2")
    gate = _gate_over_row(states, n)
    p1, ms1 = gate.score("unused question", "unused candidate")
    p2, ms2 = gate.score("unused question", "unused candidate")
    assert p1 == pytest.approx(float(p_fitted["L0.75"][n]), abs=1e-12)
    assert p2 == p1
    assert ms1 >= 0.0 and ms2 >= 0.0


def test_threshold_fires_the_catch_through_the_gate(r2):
    """End to end through the production scorer: the L0.75 catch fires."""
    n, _, states = _row(r2, "gsm8k-0015:hs2")
    p, _ms = _gate_over_row(states, n).score("unused question",
                                             "unused candidate")
    assert p < THRESHOLD


def test_default_head_and_threshold_are_the_frozen_choice():
    assert DEFAULT_HEAD == "L0.75"
    assert THRESHOLD == 0.5
    if os.path.exists(HEADS_FILE):  # ReadoutHead reads the npz eagerly
        assert LearnedGate().head.name == DEFAULT_HEAD


def test_unknown_head_is_rejected():
    with pytest.raises(ValueError, match="unknown learned-gate head"):
        ReadoutHead("own-community")  # a 176B head — not a path-B name
    with pytest.raises(ValueError, match="unknown learned-gate head"):
        ReadoutHead("L0.9")


def test_missing_state_is_rejected():
    if not os.path.exists(HEADS_FILE):
        pytest.skip("r2 heads npz not present on this machine")
    head = ReadoutHead("L0.75")
    with pytest.raises(KeyError, match="needs decision state"):
        head.score({"L0.5": np.zeros(1024)})  # L0.75 head needs the L0.75 state
    concat = ReadoutHead("ridge-concat")
    with pytest.raises(KeyError, match="needs decision state"):
        concat.score({"L0.5": np.zeros(1024), "L0.75": np.zeros(1024)})


def test_expected_layers_constant_matches_the_extraction_json():
    path = f"{R2}/extract-intern-decision-0.8b-r2.json"
    if not os.path.exists(path):
        pytest.skip("r2 extraction json not present on this machine")
    extract = json.load(open(path))
    assert EXPECTED_LAYERS == extract["n_layers"] == 24


# --------------------------------------------------------------------------
# the live cross-check (opt-in; loads the backbone on CPU)
# --------------------------------------------------------------------------

def _harvest_row(item_id: str) -> dict:
    with open(HARVEST_FILE) as fh:
        for line in fh:
            row = json.loads(line)
            if row["item_id"] == item_id:
                return row
    raise AssertionError(f"{item_id} not in {HARVEST_FILE}")


@pytest.mark.skipif(os.environ.get("HIVE_GATE_LIVE") != "1",
                    reason="live backbone test; set HIVE_GATE_LIVE=1")
def test_live_extraction_matches_the_stored_item():
    if not (os.path.exists(STATES_FILE) and os.path.exists(HARVEST_FILE)):
        pytest.skip("r2 states npz or harvest rows not present on this machine")
    item = "gsm8k-0015:hs2"  # the L0.75 catch: the gate's flagship item
    row = _harvest_row(item)

    z = np.load(STATES_FILE, allow_pickle=False)
    ids = [str(i) for i in z["item_ids"]]
    n = ids.index(item)
    stored = {label: z[label][n].astype(np.float64) for label in STATE_LABELS}

    gate = LearnedGate()
    feats = gate.states(row["prompt"], row["candidate"])
    assert sorted(feats) == sorted(STATE_LABELS)
    for label in STATE_LABELS:
        live = feats[label].astype(np.float64)
        denom = np.linalg.norm(stored[label]) * np.linalg.norm(live)
        cos = float(stored[label] @ live / denom)
        assert cos >= 0.9999, f"{label}: state cosine {cos}"

    # and the whole chain: live states through the frozen head reproduce the
    # stored fitted p for this item (committed as the L0.75 catch).
    p, ms = gate.score(row["prompt"], row["candidate"])
    want = ReadoutHead().score({k: stored[k].astype(np.float64)
                                for k in STATE_LABELS})
    assert abs(p - want) <= 5e-3
    assert p < THRESHOLD  # it is the catch — a fired gate on the stored item
    assert ms > 0.0
    print(f"\n[live] {item}: p={p:.6f} (stored-fitted {want:.6f}), ms={ms:.1f}")
