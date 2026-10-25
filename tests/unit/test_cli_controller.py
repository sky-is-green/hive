"""The CLI exposes the same consent and confirmation gates as the API."""

from __future__ import annotations

import json

from harness import cli
from harness.controller.profile import HardwareProfiler


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
    }


def _fake_profiler(monkeypatch):
    class FakeProfiler(HardwareProfiler):
        def __init__(self, state):
            super().__init__(state, probe=_summary)

    monkeypatch.setattr("harness.controller.profile.HardwareProfiler", FakeProfiler)
    return FakeProfiler


def test_profile_scan_needs_consent(tmp_path, capsys):
    assert cli.main(["profile", "scan", "--state-dir", str(tmp_path)]) == 2
    assert "consent hardware" in capsys.readouterr().err


def test_profile_scan_show_and_controller_flow(tmp_path, capsys, monkeypatch):
    _fake_profiler(monkeypatch)
    state = str(tmp_path / "state")
    assert cli.main(["profile", "scan", "--consent", "hardware",
                     "--state-dir", state, "--json"]) == 0
    profile = json.loads(capsys.readouterr().out)
    assert profile["gpus"][1]["display"] is False

    assert cli.main(["profile", "show", "--state-dir", state, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["profile_id"] == profile["profile_id"]

    assert cli.main(["controller", "propose", "--state-dir", state,
                     "--model-gib", "12", "--native-ctx", "32768",
                     "--ctx", "16384", "--request", "How do I test this?",
                     "--json"]) == 0
    proposal = json.loads(capsys.readouterr().out)
    assert proposal["changes"]["runtime_defaults.ngl"] == 999
    assert proposal["request_features"]["intent"] == "qa"

    assert cli.main(["controller", "outcome", proposal["id"], "accepted",
                     "--metric", "latency_ms=12", "--state-dir", state,
                     "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "accepted"

    assert cli.main(["controller", "apply", proposal["id"], "--state-dir", state,
                     "--yes", "--json"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["settings"]["runtime_defaults"]["ngl"] == 999

    assert cli.main(["controller", "revert", "--state-dir", state,
                     "--yes", "--json"]) == 0
    reverted = json.loads(capsys.readouterr().out)
    assert "ngl" not in reverted["settings"]["runtime_defaults"]


def test_apply_without_yes_is_refused(tmp_path, capsys, monkeypatch):
    _fake_profiler(monkeypatch)
    state = str(tmp_path / "state")
    assert cli.main(["profile", "scan", "--consent", "hardware",
                     "--state-dir", state]) == 0
    capsys.readouterr()
    assert cli.main(["controller", "propose", "--state-dir", state,
                     "--model-gib", "1", "--json"]) == 0
    proposal = json.loads(capsys.readouterr().out)
    assert cli.main(["controller", "apply", proposal["id"],
                     "--state-dir", state]) == 2
    assert "pass --yes" in capsys.readouterr().err


def test_encoder_status_is_lazy(tmp_path, capsys):
    assert cli.main(["encoder", "status", "--model", "fake-model",
                     "--state-dir", str(tmp_path), "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["model"] == "fake-model"
    assert body["loaded"] is False
