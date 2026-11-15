"""The Rust install owns `hive`; the Python package keeps `hive-py`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "install-rust.sh"


def test_rust_install_layout_and_entry_point(tmp_path):
    binaries = REPO / "rust" / "target" / "debug"
    if not (binaries / "hive").is_file():
        pytest.skip("hive is not built")
    prefix = tmp_path / "prefix"
    result = subprocess.run(
        [
            str(SCRIPT),
            "--prefix",
            str(prefix),
            "--bin-dir",
            str(binaries),
            "--skip-build",
            "--no-systemd",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for name in ("hive", "hive-daemon", "hive-encoder", "hive-memory", "hive-sandbox"):
        assert (prefix / "bin" / name).is_file(), name
    assert (prefix / "share" / "hive" / "webui" / "index.html").is_file()
    install = json.loads((prefix / "share" / "hive" / "install.json").read_text())
    assert install["name"] == "hive"
    assert install["version"]

    env = {
        **os.environ,
        "HIVE_HOME": str(tmp_path / "home"),
        "HIVE_PYTHON": sys.executable,
    }
    version = subprocess.run(
        [str(prefix / "bin" / "hive"), "version", "--json"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert version.returncode == 0, version.stderr
    payload = json.loads(version.stdout)
    assert payload["hive"] == install["version"]

    help_result = subprocess.run(
        [str(prefix / "bin" / "hive-daemon"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--port" in help_result.stdout

    pyproject = (REPO / "pyproject.toml").read_text()
    assert 'hive-py = "harness.cli:main"' in pyproject
    assert 'hive = "harness.cli:main"' not in pyproject

    result = subprocess.run(
        [str(SCRIPT), "--prefix", str(prefix), "--uninstall"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not (prefix / "bin" / "hive").exists()
    assert not (prefix / "share" / "hive").exists()
