"""Unit tests for the escalation lane (durable harvest of ladder-resolved items)."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.cascade.escalation_lane import (
    EscalationLane,
    in_lane,
    lane_record,
)
from experiments.cascade.run_cascade import build_parser


def _row(**overrides):
    row = {
        "id": "gsm8k-0015:hs2",
        "bucket": "gsm8k",
        "planned_path": "P5",
        "scion_answer": "draft answer",
        "scion_ok": False,
        "verdict": "reject",
        "verdict_confidence": 0.9,
        "verdict_reason": "inconsistent",
        "findings": [{"kind": "constraint", "msg": "2 != 3"}],
        "escalated": False,
        "gated": False,
        "repair": {"cleared": False, "text": "fixed answer"},
        "cascade_ok": False,
        "false_accept": False,
        "false_reject": False,
        "human_corrected": False,
        "human_gate": None,
        "api_answer": None,
        "bucket": "gsm8k",
    }
    row.update(overrides)
    return row


def _task():
    return {
        "id": "gsm8k-0015:hs2",
        "bucket": "gsm8k",
        "prompt": "what is 2+2?",
        "checker": {"kind": "numeric_gold", "expect": "4", "tol": 0.0},
    }


def test_disabled_drops_everything(tmp_path):
    lane = EscalationLane.disabled()
    assert lane.path is None
    assert lane.count == 0
    lane.append({"x": 1})
    assert lane.count == 0
    lane.close()  # no-op, must not raise
    assert not (tmp_path / "escalation.jsonl").exists()


def test_writes_jsonl_and_round_trips(tmp_path):
    out = tmp_path / "run" / "escalation.jsonl"
    lane = EscalationLane(out)
    rec = {"hello": 42, "nested": [1, 2]}
    lane.append(rec)
    assert lane.count == 1
    assert out.exists()
    lane.close()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == rec


def test_in_lane_predicate():
    # ladder touched: flagged
    assert in_lane(_row(flagged=True))
    # escalated local attempt
    assert in_lane(_row(escalated=True))
    # human-gated
    assert in_lane(_row(human_gate={"choice": "solve", "accepted_locally": True}))
    # router skip (no local attempt) -> not in lane even if escalated
    assert not in_lane(_row(scion_ok=None, escalated=True))
    # plain served correct row
    assert not in_lane(_row(scion_ok=True, verdict="accept", escalated=False, flagged=False))


def test_lane_record_shape_and_source():
    row = _row(escalated=True, scion_ok=False)
    task = _task()
    rec = lane_record(
        row, task,
        run="unit-run",
        draft="FULL DRAFT TEXT (not truncated)",
        cascade_answer="api answer",
        findings=row["findings"],
    )
    assert rec["run"] == "unit-run"
    assert rec["item_id"] == "gsm8k-0015:hs2"
    assert rec["task"] == task  # checker included on purpose
    assert rec["task"]["checker"]["expect"] == "4"
    assert rec["draft"] == "FULL DRAFT TEXT (not truncated)"
    assert rec["findings"] == row["findings"]
    assert rec["verdict"]["decision"] == "reject"
    assert rec["verdict"]["truncated"] is None or rec["verdict"]["truncated"] is False
    assert rec["resolution"]["source"] == "api"
    assert rec["resolution"]["answer"] == "api answer"
    assert rec["checker"]["cascade_ok"] is False

    # repair-cleared row -> source "repair"
    r2 = _row(repair={"cleared": True, "text": "fixed"})
    rec2 = lane_record(r2, task, run="x", draft="d", cascade_answer="fixed", findings=[])
    assert rec2["resolution"]["source"] == "repair"

    # human accepted locally -> source "human"
    r3 = _row(human_gate={"choice": "solve", "accepted_locally": True})
    rec3 = lane_record(r3, task, run="x", draft="d", cascade_answer="d", findings=[])
    assert rec3["resolution"]["source"] == "human"

    # neither -> source "draft"
    r4 = _row(repair={"cleared": False})
    rec4 = lane_record(r4, task, run="x", draft="d", cascade_answer="d", findings=[])
    assert rec4["resolution"]["source"] == "draft"


def test_no_escalation_lane_flag_parses():
    args = build_parser().parse_args(["--tasks", "t.json", "--no-escalation-lane"])
    assert args.no_escalation_lane is True
    args2 = build_parser().parse_args(["--tasks", "t.json"])
    assert args2.no_escalation_lane is False