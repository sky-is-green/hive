"""The Roles page placement gate: weights + KV must be provably placeable."""

from __future__ import annotations

import pytest

from harness.placement import Device, plan_roles


def device(index: int, *, display: bool, free_gb: float, total_gb: float = 24.0) -> Device:
    return Device(index=index, card=f"card-{index}", display=display,
                  total_gb=total_gb, free_gb=free_gb)


def meta() -> dict:
    """A dense 32-layer, 8-KV-head model; enough for attention_kv_estimate."""
    return {
        "general.architecture": "llama",
        "llama.block_count": 32,
        "llama.attention.head_count": 32,
        "llama.embedding_length": 4096,
        "llama.attention.head_count_kv": 8,
        "llama.attention.key_length": 128,
        "llama.attention.value_length": 128,
    }


def model(weight_gb: float, label: str, *, resident: bool = False,
          metadata: bool = True) -> dict:
    entry = {"weights_gb": weight_gb, "resident": resident, "label": label}
    if metadata:
        entry["gguf_metadata"] = meta()
    return entry


def by_role(result: dict) -> dict[str, dict]:
    return {plan["role"]: plan for plan in result["plans"]}


def test_auto_prefers_a_non_display_card_and_charges_a_shared_model_once():
    roles = {"face": "m1", "router": "m1", "judge": "m2", "worker": "m3"}
    placements = {
        "face": "auto", "router": "auto", "judge": "display",
        "worker": "non_display",
    }
    models = {
        "m1": model(6, "M1"),
        "m2": model(2, "M2"),
        "m3": model(12, "M3"),
    }
    devices = [
        device(0, display=True, free_gb=22),
        device(1, display=False, free_gb=20),
    ]

    result = plan_roles(roles, placements, models, devices)
    plans = by_role(result)

    # M1 is charged once on the spare card, so M3 still fits beside it.
    assert plans["face"]["device"] == 1
    assert plans["router"]["device"] == 1
    assert plans["worker"]["device"] == 1
    assert plans["judge"]["device"] == 0
    assert all(plan["fits"] for plan in plans.values())
    assert result["summary"] == {"total": 4, "ok": 4, "warn": 0, "blocked": 0}


def test_over_budget_is_blocked():
    result = plan_roles(
        {"face": "m1"}, {"face": "non_display"},
        {"m1": model(25, "M1")},
        [device(0, display=False, free_gb=20)],
    )
    plan = result["plans"][0]

    assert plan["fits"] is False
    assert plan["device"] == 0
    assert "needs" in plan["reason"]
    assert plan["need_gb"] > 25
    assert result["summary"]["blocked"] == 1


def test_kv_cache_is_added_to_the_weight_budget():
    def run(free_gb: float):
        return plan_roles(
            {"face": "m1"}, {"face": "non_display"},
            {"m1": model(10, "M1")},
            [device(0, display=False, free_gb=free_gb)],
            ctx=8192, kv="q8_0",
        )["plans"][0]

    tight = run(10.4)
    roomy = run(10.6)

    assert tight["fits"] is False
    assert 0.5 < tight["kv_gb"] < 0.6
    assert tight["need_gb"] == pytest.approx(10 + tight["kv_gb"])
    assert "KV" in tight["reason"]
    assert roomy["fits"] is True


def test_context_and_kv_type_change_the_need():
    def need(ctx: int, kv: str) -> float:
        return plan_roles(
            {"face": "m1"}, {"face": "non_display"},
            {"m1": model(10, "M1")},
            [device(0, display=False, free_gb=20)],
            ctx=ctx, kv=kv,
        )["plans"][0]["need_gb"]

    small = need(8192, "q8_0")
    large = need(32768, "q8_0")
    f16 = need(8192, "f16")

    assert large > small
    assert f16 > small


def test_resident_and_policy_seats_do_not_consume_budget():
    result = plan_roles(
        {"face": "m1", "risk": "guards", "worker": "m2"},
        {"face": "display", "risk": "", "worker": "display"},
        {
            "m1": model(10, "M1", resident=True),
            "guards": {"weights_gb": 0, "resident": False, "label": "Guards"},
            "m2": model(2, "M2"),
        },
        [device(0, display=True, free_gb=10)],
    )
    plans = by_role(result)

    assert plans["face"]["resident"] is True
    assert plans["risk"]["resident"] is True
    assert plans["worker"]["device"] == 0
    assert all(plan["fits"] for plan in plans.values())


def test_explicit_card_policy_is_honoured():
    result = plan_roles(
        {"worker": "m1"}, {"worker": "1"},
        {"m1": model(2, "M1")},
        [
            device(0, display=True, free_gb=20),
            device(1, display=False, free_gb=4),
        ],
    )

    assert result["plans"][0]["device"] == 1


def test_unknown_model_size_is_blocked():
    result = plan_roles(
        {"face": "m1"}, {"face": "auto"},
        {"m1": {"weights_gb": None, "resident": False, "label": "M1"}},
        [device(0, display=False, free_gb=20)],
    )
    plan = result["plans"][0]

    assert plan["fits"] is False
    assert "size is unknown" in plan["reason"]


def test_no_card_matches_the_policy():
    result = plan_roles(
        {"face": "m1"}, {"face": "display"},
        {"m1": model(2, "M1")},
        [device(0, display=False, free_gb=20)],
    )
    plan = result["plans"][0]

    assert plan["fits"] is False
    assert "no card matches" in plan["reason"]


def test_no_card_policy_auto_places_with_a_warning():
    result = plan_roles(
        {"face": "m1"}, {"face": ""},
        {"m1": model(2, "M1")},
        [device(0, display=False, free_gb=20)],
    )
    plan = result["plans"][0]

    assert plan["fits"] is True
    assert any("no card policy" in warning for warning in plan["warnings"])
    assert result["summary"]["warn"] == 1


def test_missing_metadata_falls_back_to_weights_only():
    result = plan_roles(
        {"face": "m1"}, {"face": "non_display"},
        {"m1": model(2, "M1", metadata=False)},
        [device(0, display=False, free_gb=20)],
    )
    plan = result["plans"][0]

    assert plan["fits"] is True
    assert plan["kv_gb"] == 0
    assert plan["need_gb"] == 2
    assert any("weights only" in warning for warning in plan["warnings"])
    assert result["summary"]["warn"] == 1


def test_shared_model_with_a_conflicting_policy_warns_but_stays_on_one_card():
    result = plan_roles(
        {"face": "m1", "router": "m1"},
        {"face": "auto", "router": "display"},
        {"m1": model(2, "M1")},
        [
            device(0, display=True, free_gb=20),
            device(1, display=False, free_gb=20),
        ],
    )
    plans = by_role(result)

    assert plans["face"]["device"] == 1
    assert plans["router"]["device"] == 1
    assert plans["router"]["warnings"]
    assert result["summary"]["warn"] == 1
