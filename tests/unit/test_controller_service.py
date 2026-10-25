"""Applying is explicit, reversible, and leaves an audit trail."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pytest

from harness.controller.profile import HardwareProfiler
from harness.controller.service import (
    ConfirmationRequired,
    ControllerError,
    ControllerService,
)
from harness.state_store import StatePaths


def _summary():
    return {
        "cpu_count": 16,
        "total_ram_gb": 30.9,
        "available_ram_gb": 18.2,
        "vram_source": "sysfs",
        "devices": [
            {"index": 0, "name": "AMD GPU 0", "backend": "rocm",
             "memory_gb": 21.46, "used_gb": 19.32, "free_gb": 2.14,
             "display": True, "visible": True},
            {"index": 1, "name": "AMD GPU 1", "backend": "rocm",
             "memory_gb": 21.46, "used_gb": 0.55, "free_gb": 20.91,
             "display": False, "visible": True},
        ],
        "disk": {"models": {"free_gb": 23.7, "total_gb": 100.0}},
    }


class _Settings:
    def __init__(self):
        self.data = {
            "runtime_defaults": {"threads": 4, "ctx_size": 8192, "ngl": 0},
            "memory_defaults": {},
        }

    def load(self, _root):
        return copy.deepcopy(self.data)

    def save(self, changes, _root):
        for block, values in changes.items():
            self.data.setdefault(block, {}).update(values)
        return copy.deepcopy(self.data)


def _service(tmp_path):
    state = StatePaths(tmp_path / "state")
    profiler = HardwareProfiler(state, probe=_summary)
    profiler.scan("hardware", profile_id="local-test")
    settings = _Settings()
    service = ControllerService(
        state, profiler=profiler,
        settings_loader=settings.load, settings_saver=settings.save,
        now=lambda: datetime(2026, 10, 25, 21, tzinfo=timezone.utc),
    )
    return service, settings, state


def test_proposal_is_persisted_and_listed(tmp_path):
    service, _settings, _state = _service(tmp_path)
    proposal = service.propose(model_gib=12.0, native_ctx=32768, desired_ctx=16384)
    assert proposal["changes"]["runtime_defaults.ngl"] == 999
    status = service.status()
    assert status["profile"]["profile_id"] == "local-test"
    assert status["latest_proposal"]["id"] == proposal["id"]
    assert service.get_proposal(proposal["id"])["id"] == proposal["id"]


def test_apply_requires_confirmation(tmp_path):
    service, _settings, _state = _service(tmp_path)
    proposal = service.propose(model_gib=12.0, native_ctx=32768, desired_ctx=16384)
    with pytest.raises(ConfirmationRequired):
        service.apply(proposal["id"])


def test_apply_saves_the_change_and_rollback_restores_it(tmp_path):
    service, settings, state = _service(tmp_path)
    proposal = service.propose(model_gib=12.0, native_ctx=32768, desired_ctx=16384)
    result = service.apply(proposal["id"], confirmed=True)
    assert result["ok"] is True
    assert settings.data["runtime_defaults"]["threads"] == 8
    assert settings.data["runtime_defaults"]["ctx_size"] == 16384
    assert settings.data["runtime_defaults"]["ngl"] == 999
    controller = json.loads(state.controller_file().read_text(encoding="utf-8"))
    assert controller["last_apply"]["proposal_id"] == proposal["id"]
    assert controller["last_apply"]["previous"]["runtime_defaults.threads"] == 4

    reverted = service.revert(confirmed=True)
    assert reverted["ok"] is True
    assert settings.data["runtime_defaults"]["threads"] == 4
    assert settings.data["runtime_defaults"]["ctx_size"] == 8192
    assert settings.data["runtime_defaults"]["ngl"] == 0
    assert json.loads(state.controller_file().read_text(
        encoding="utf-8"))["last_apply"] is None


def test_revert_needs_a_previous_apply(tmp_path):
    service, _settings, _state = _service(tmp_path)
    with pytest.raises(ConfirmationRequired):
        service.revert()
    with pytest.raises(ControllerError, match="no applied"):
        service.revert(confirmed=True)


def test_apply_and_revert_are_audited(tmp_path):
    service, _settings, state = _service(tmp_path)
    proposal = service.propose(model_gib=12.0, native_ctx=32768, desired_ctx=16384)
    service.apply(proposal["id"], confirmed=True)
    service.revert(confirmed=True)
    rows = [json.loads(line) for line in state.audit_file().read_text(
        encoding="utf-8").splitlines()]
    events = [row["event"] for row in rows]
    assert events == ["controller.propose", "controller.apply", "controller.revert"]


def test_request_features_are_traced_but_raw_text_is_not(tmp_path):
    service, _settings, state = _service(tmp_path)
    proposal = service.propose(
        model_gib=12.0, native_ctx=32768, desired_ctx=32768,
        request="delete production with sudo; this text must not be stored")
    assert proposal["request_features"]["risk"] == "high"
    trace = state.controller_trace_file().read_text(encoding="utf-8")
    assert "delete production" not in trace
    assert "request_features" in trace
    assert "controller.proposal" in trace

    service.apply(proposal["id"], confirmed=True)
    service.revert(confirmed=True)
    trace = state.controller_trace_file().read_text(encoding="utf-8")
    assert '"outcome": "applied"' in trace
    assert '"outcome": "reverted"' in trace


def test_record_outcome_is_append_only_and_typed(tmp_path):
    service, _settings, state = _service(tmp_path)
    with pytest.raises(ControllerError, match="outcome must be"):
        service.record_outcome("rec-x", "maybe")
    service.record_outcome("rec-x", "accepted", metrics={"latency_ms": 12})
    row = json.loads(state.controller_trace_file().read_text(
        encoding="utf-8").splitlines()[-1])
    assert row["outcome"] == "accepted"
    assert row["metrics"]["latency_ms"] == 12


def test_unknown_proposal_is_a_clear_error(tmp_path):
    service, _settings, _state = _service(tmp_path)
    with pytest.raises(ControllerError, match="no saved recommendation"):
        service.apply("rec-20990101T000000-nope", confirmed=True)


def test_propose_without_a_profile_says_what_to_do(tmp_path):
    state = StatePaths(tmp_path / "state")
    profiler = HardwareProfiler(state, probe=_summary)
    service = ControllerService(state, profiler=profiler)
    with pytest.raises(ControllerError, match="profile scan"):
        service.propose(model_gib=1.0)
