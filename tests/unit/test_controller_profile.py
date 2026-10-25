"""Hardware profiles need consent and survive a restart as JSON."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from harness.controller.profile import (
    Consent,
    ConsentRequired,
    HardwareProfiler,
    build_profile,
)
from harness.state_store import StatePaths


def _summary():
    return {
        "cpu_count": 16,
        "total_ram_gb": 30.9,
        "available_ram_gb": 18.2,
        "swap_total_gb": 8.0,
        "swap_used_gb": 1.0,
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
        "visibility": {"indices": [0, 1]},
    }


def test_consent_parsing_accepts_strings_dicts_and_lists():
    assert Consent.parse("hardware").hardware is True
    assert Consent.parse("hardware,models").models is True
    assert Consent.parse({"hardware": True}).hardware is True
    assert Consent.parse(["hardware", "traces"]).traces is True
    with pytest.raises(ValueError, match="unknown consent"):
        Consent.parse("everything")


def test_build_profile_requires_hardware_consent():
    with pytest.raises(ConsentRequired):
        build_profile(_summary(), Consent(), profile_id="local-test")


def test_build_profile_keeps_only_the_policy_fields():
    profile = build_profile(
        _summary(), Consent(hardware=True), profile_id="local-test",
        now=datetime(2026, 10, 25, 21, tzinfo=timezone.utc))
    assert profile["schema_version"] == 1
    assert profile["profile_id"] == "local-test"
    assert profile["scanned_at"] == "2026-10-25T21:00:00Z"
    assert profile["consent"] == {
        "hardware": True, "models": False, "traces": False, "export": False}
    assert len(profile["gpus"]) == 2
    assert profile["gpus"][1]["display"] is False
    assert profile["gpus"][1]["free_gb"] == 20.91
    assert profile["storage"]["models_free_gb"] == 23.7
    assert profile["backends"] == ["rocm"]


def test_scan_saves_and_loads_one_profile(tmp_path):
    state = StatePaths(tmp_path / "state")
    profiler = HardwareProfiler(state, probe=_summary)
    with pytest.raises(ConsentRequired):
        profiler.scan(None)
    saved = profiler.scan("hardware", profile_id="local-test")
    assert saved["profile_id"] == "local-test"
    assert profiler.get("local-test")["ram"]["available_gb"] == 18.2
    assert profiler.latest()["profile_id"] == "local-test"
    assert [row["profile_id"] for row in profiler.list()] == ["local-test"]


def test_delete_removes_the_snapshot(tmp_path):
    state = StatePaths(tmp_path / "state")
    profiler = HardwareProfiler(state, probe=_summary)
    profiler.scan("hardware", profile_id="local-test")
    assert profiler.delete("local-test") is True
    assert profiler.get("local-test") is None
    assert profiler.delete("local-test") is False


def test_a_corrupt_profile_is_not_returned(tmp_path):
    state = StatePaths(tmp_path / "state").ensure()
    path = state.profile_file("local-bad")
    path.write_text("{not json", encoding="utf-8")
    profiler = HardwareProfiler(state, probe=_summary)
    assert profiler.get("local-bad") is None
    assert list(state.profiles_dir.glob("corrupt-*-hardware-local-bad.json"))
