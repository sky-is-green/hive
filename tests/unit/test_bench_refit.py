"""Unit tests for ``experiments.cascade.bench_refit`` (train-fit/test-report)."""

from __future__ import annotations

from experiments.cascade.bench_refit import choose, evaluate, sweep


def _row(rid: str, p: float, scion_ok: bool, api_ok: bool) -> dict:
    return {"id": rid, "p_local_correct": p, "scion_ok": scion_ok, "api_ok": api_ok}


def test_sweep_reports_quality_share_and_routing_accuracy():
    rows = [_row("a", 0.9, True, True), _row("b", 0.3, False, True)]
    by_threshold = {row["threshold"]: row for row in sweep(rows)}

    at_half = by_threshold[0.5]
    assert at_half["quality"] == 1.0          # a local right, b api right
    assert at_half["api_share"] == 0.5
    assert at_half["routing_accuracy"] == 1.0  # both routed correctly

    at_wide = by_threshold[0.95]
    assert at_wide["api_share"] == 1.0
    assert at_wide["quality"] == 1.0


def test_choose_prefers_quality_then_share_then_highest_threshold():
    # Local is right for a and wrong for b; the API is right for both.  The
    # quality-first rule must route b to the API, then keep the highest
    # threshold that still routes a locally.
    rows = [_row("a", 0.9, True, True), _row("b", 0.2, False, True)]
    chosen = choose(sweep(rows))

    assert chosen["quality"] == 1.0
    assert chosen["api_share"] == 0.5
    assert chosen["threshold"] == 0.9


def test_choose_prefers_the_local_answer_when_the_api_is_not_better():
    # Both local and API are wrong for b: routing b to the API buys nothing,
    # so the rule keeps API share at 0 and still takes the highest threshold.
    rows = [_row("a", 0.9, True, True), _row("b", 0.2, False, False)]
    chosen = choose(sweep(rows))

    assert chosen["quality"] == 0.5
    assert chosen["api_share"] == 0.0
    assert chosen["threshold"] == 0.2


def test_evaluate_applies_a_fixed_threshold_without_refitting():
    rows = [_row("a", 0.9, True, True), _row("b", 0.2, False, True)]

    fixed = evaluate(rows, 0.5)
    assert fixed["quality"] == 1.0 and fixed["api_share"] == 0.5

    all_api = evaluate(rows, 0.95)
    assert all_api["api_share"] == 1.0 and all_api["quality"] == 1.0
