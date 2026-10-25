"""The controller's state files are atomic, versioned and inspectable."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harness.state_store import (
    AuditLog,
    JsonStore,
    SchemaMismatch,
    StatePaths,
    StateCorrupt,
)


def test_layout_is_stable(tmp_path):
    state = StatePaths(tmp_path / "state").ensure()
    assert state.controller_file() == state.settings_dir / "controller.json"
    assert state.profile_file("local-a") == state.profiles_dir / "hardware-local-a.json"
    assert (state.audit_dir).is_dir()


def test_paths_reject_traversal(tmp_path):
    state = StatePaths(tmp_path)
    with pytest.raises(ValueError):
        state.profile_file("../../etc/passwd")
    with pytest.raises(ValueError):
        state.settings_file("../controller")


def test_atomic_round_trip_and_version(tmp_path):
    path = tmp_path / "settings" / "controller.json"
    store = JsonStore(path)
    written = store.write({"answer": 42})
    assert written["schema_version"] == 1
    assert store.read()["answer"] == 42
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1
    if os.name != "nt":
        assert (path.stat().st_mode & 0o777) == 0o600


def test_missing_file_returns_the_safe_default(tmp_path):
    store = JsonStore(tmp_path / "missing.json", default={"enabled": True})
    assert store.read() == {"enabled": True}
    assert store.read() is not store.default


def test_corrupt_file_is_quarantined_not_reused(tmp_path):
    path = tmp_path / "profiles" / "hardware-x.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    store = JsonStore(path, default={"safe": True})
    assert store.read() == {"safe": True}
    assert not path.exists()
    quarantined = list(path.parent.glob("corrupt-*-hardware-x.json"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{not json"


def test_non_object_is_quarantined(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert JsonStore(path, default={"safe": True}).read() == {"safe": True}
    assert list(tmp_path.glob("corrupt-*-x.json"))


def test_schema_mismatch_is_not_silently_downgraded(tmp_path):
    path = tmp_path / "future.json"
    path.write_text('{"schema_version": 9, "value": 1}', encoding="utf-8")
    with pytest.raises(SchemaMismatch):
        JsonStore(path).read()
    assert path.exists()


def test_strict_read_names_a_missing_file(tmp_path):
    with pytest.raises(StateCorrupt):
        JsonStore(tmp_path / "none.json").read_strict()


def test_update_preserves_other_keys(tmp_path):
    store = JsonStore(tmp_path / "grants.json", default={"grants": []})
    store.write({"grants": [{"id": "a"}]})

    def add(current):
        current["grants"].append({"id": "b"})
        return current

    result = store.update(add)
    assert [row["id"] for row in result["grants"]] == ["a", "b"]
    assert store.read()["grants"][1]["id"] == "b"


def test_audit_redacts_credentials_and_appends(tmp_path):
    path = tmp_path / "audit" / "settings-202610.jsonl"
    log = AuditLog(path)
    log.append({"event": "x", "token": "secret", "nested": {"password": "p"}})
    log.append({"event": "y", "api_key": "k"})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["token"] == "[redacted]"
    assert rows[0]["nested"]["password"] == "[redacted]"
    assert rows[1]["api_key"] == "[redacted]"
    assert rows[0]["event"] == "x" and "ts" in rows[0]


def test_audit_truncates_a_very_long_field(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append({"event": "x", "detail": "a" * 5000})
    row = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert row["detail"].endswith("[truncated]")
    assert len(row["detail"]) == 4096 + len("…[truncated]")


def test_ensure_is_idempotent(tmp_path):
    state = StatePaths(tmp_path / "state")
    state.ensure()
    state.ensure()
    assert state.profiles_dir.is_dir()
