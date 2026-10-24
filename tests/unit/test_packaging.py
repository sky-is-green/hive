"""The serve-only package contract: lazy Studio imports and HIVE_HOME paths."""

from __future__ import annotations

import json
import subprocess
import sys

from harness import paths


def test_packaged_backend_manifest_matches_the_checkout_copy():
    source = paths.SOURCE_ROOT / "tools" / "backends" / "manifest.json"
    packaged = paths.PACKAGED_MANIFEST
    assert source.is_file()
    assert packaged.is_file()
    assert json.loads(source.read_text(encoding="utf-8")) == json.loads(
        packaged.read_text(encoding="utf-8"))


def test_cli_import_does_not_load_the_studio():
    code = (
        "import sys, harness.cli; "
        "print(int('harness.app' in sys.modules), int('fastapi' in sys.modules))"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code], text=True, timeout=60).strip()
    assert output == "0 0"


def test_hive_home_moves_all_mutable_serve_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_HOME", str(tmp_path))
    monkeypatch.delenv("HIVE_STATE_DIR", raising=False)

    from harness import backends, serve_state
    from harness.stack import schema

    assert paths.home() == tmp_path
    assert paths.stacks_dir() == tmp_path / "stacks"
    assert paths.models_dir() == tmp_path / "models" / "gguf"
    assert paths.logs_dir() == tmp_path / "logs"
    assert schema.stacks_dir() == tmp_path / "stacks"
    assert serve_state.state_dir() == tmp_path / "harness_state"
    assert backends.resolve("vulkan")["binary"].startswith(
        str(tmp_path / "tools" / "backends" / "vulkan"))
