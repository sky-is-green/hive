"""Unit tests for the harness validator service and its adapters.

No torch/transformers here: adapters take injected models and the service
takes an injected factory, so the lazy-load, threshold and request-framing
contracts are pinned without loading anything.
"""

from __future__ import annotations

import pytest

from harness.cascade.validator import (
    ClefTernaryValidator,
    ClefValidator,
    TinyJevValidator,
    ValidatorService,
    apply_calibration,
    calibration_from_file,
    load_validator,
)


class _FakeValidator:
    def __init__(self, p: float = 0.9) -> None:
        self.p = p
        self.calls: list[tuple[dict, str]] = []

    def verdict(self, task: dict, candidate: str) -> tuple[float, float]:
        self.calls.append((task, candidate))
        return self.p, 12.5


def test_service_lazy_loads_once_and_reports_status():
    built: list[tuple[str, str]] = []

    def factory(model_id: str, device: str) -> _FakeValidator:
        built.append((model_id, device))
        return _FakeValidator()

    service = ValidatorService(model_id="m", device="cpu", threshold=0.7,
                               factory=factory)

    assert service.status()["loaded"] is False
    out = service.verdict({"prompt": "p"}, "a")
    assert built == [("m", "cpu")]
    assert out["p_correct"] == 0.9
    assert out["accept"] is True
    assert out["threshold"] == 0.7
    assert out["checks"] == 1

    service.verdict({"prompt": "p"}, "a", threshold=0.95)
    assert built == [("m", "cpu")]  # loaded exactly once
    assert service.status()["checks"] == 2
    assert service.status()["loaded"] is True


def test_service_rejects_below_threshold():
    service = ValidatorService(threshold=0.8,
                               factory=lambda m, d: _FakeValidator(0.5))
    out = service.verdict({"prompt": "p"}, "a")
    assert out["accept"] is False
    assert out["p_correct"] == 0.5


def test_load_validator_selects_clef_by_model_id(monkeypatch):
    seen: dict[str, tuple[str, str]] = {}

    def clef(model_id, *, device):
        seen["clef"] = (model_id, device)
        return "clef"

    def tiny(model_id, *, device):
        seen["tiny"] = (model_id, device)
        return "tiny"

    monkeypatch.setattr("harness.cascade.validator.ClefValidator", clef)
    monkeypatch.setattr("harness.cascade.validator.TinyJevValidator", tiny)

    assert load_validator("Cloudflare/clef-flash", "cpu") == "clef"
    assert load_validator("lostargon/Tiny-Jev-1.7B", "cpu") == "tiny"
    assert seen["clef"] == ("Cloudflare/clef-flash", "cpu")


def test_load_validator_selects_ternary_by_model_id(monkeypatch):
    seen: dict[str, tuple[str, str]] = {}

    def ternary(model_id, *, device):
        seen["ternary"] = (model_id, device)
        return "ternary"

    monkeypatch.setattr("harness.cascade.validator.ClefTernaryValidator", ternary)

    assert load_validator("clef-flash-ternary", "cpu") == "ternary"
    assert seen["ternary"] == ("clef-flash-ternary", "cpu")


class _FakeEnc:
    input_ids = [1, 2, 3]


def test_ternary_adapter_frames_one_noul_question():
    calls: dict[str, object] = {}

    def encode(prompt: str, candidate: str) -> _FakeEnc:
        calls["encode"] = (prompt, candidate)
        return _FakeEnc()

    def runner(ids):
        calls["ids"] = list(ids)
        return "hidden"

    def head(hidden, enc):
        calls["head"] = (hidden, enc)
        return 0.81

    validator = ClefTernaryValidator(
        "clef-flash-ternary", _runner=runner, _head_fn=head, _encode_fn=encode)
    p, ms = validator.verdict({"prompt": "2+2?"}, "4")

    assert p == 0.81 and ms >= 0.0
    assert calls["encode"] == ("2+2?", "4")
    assert calls["ids"] == [1, 2, 3]
    assert calls["head"][0] == "hidden"
    validator.close()  # no process in the seam path; must not raise


def test_clef_adapter_frames_one_noul_question():
    requests: list[dict] = []

    def fake_systemone(model, processor, request):
        requests.append(request)
        return {
            "model": request["model"],
            "answers": {"verdict": {"type": "noul", "noul": 0.73}},
            "usage": {},
        }

    validator = ClefValidator(
        "Cloudflare/clef-flash", _model="m", _processor="p",
        _systemone=fake_systemone,
    )
    p, ms = validator.verdict({"prompt": "2+2?"}, "4")

    assert p == 0.73 and ms >= 0.0
    request = requests[0]
    assert request["model"] == "clef-flash"
    assert request["state"] == {"task": "2+2?", "candidate_answer": "4"}
    assert request["questions"]["verdict"]["type"] == "noul"
    assert request["questions"]["verdict"]["instructions"] == TinyJevValidator.INSTRUCTION


def test_calibration_boundary_and_file(tmp_path):
    import json as _json

    cal = tmp_path / "cal.json"
    cal.write_text(_json.dumps({"calibration": {"a": 2.0, "threshold_raw": 0.6}}))
    assert calibration_from_file(cal) == (2.0, 0.6)
    assert abs(apply_calibration(0.6, 2.0, 0.6) - 0.5) < 1e-12  # boundary
    assert apply_calibration(0.7, 2.0, 0.6) > 0.5
    assert apply_calibration(0.5, 2.0, 0.6) < 0.5


def test_ternary_adapter_applies_injected_calibration():
    def build(cal):
        return ClefTernaryValidator(
            "x", _runner=lambda ids: "h", _head_fn=lambda h, e: 0.7,
            _encode_fn=lambda p, c: _FakeEnc(), _calibration=cal)

    p_low, _ = build((1.0, 0.8)).verdict({"prompt": "p"}, "a")   # raw below boundary
    p_high, _ = build((1.0, 0.6)).verdict({"prompt": "p"}, "a")  # raw above boundary
    assert p_low < 0.5 < p_high
