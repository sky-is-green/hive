from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLUGIN_KEYS = REPO / "scripts" / "sign_plugin.py"
SIGN_ARTIFACT = REPO / "scripts" / "sign_artifact.py"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_sign_and_verify_a_release_artifact(tmp_path):
    key = tmp_path / "release.json"
    result = _run(
        PLUGIN_KEYS,
        "generate",
        "--key-file",
        str(key),
        "--key-id",
        "release-test",
    )
    assert result.returncode == 0, result.stderr
    artifact = tmp_path / "release-manifest.json"
    artifact.write_text('{"name":"hive"}\n')
    signature = tmp_path / "release-manifest.json.sig"
    result = _run(
        SIGN_ARTIFACT,
        "sign",
        "--key-file",
        str(key),
        "--file",
        str(artifact),
        "--signature",
        str(signature),
    )
    assert result.returncode == 0, result.stderr
    assert signature.is_file()

    result = _run(
        SIGN_ARTIFACT,
        "verify",
        "--key-file",
        str(key),
        "--file",
        str(artifact),
        "--signature",
        str(signature),
    )
    assert result.returncode == 0, result.stderr

    artifact.write_text('{"name":"tampered"}\n')
    result = _run(
        SIGN_ARTIFACT,
        "verify",
        "--key-file",
        str(key),
        "--file",
        str(artifact),
        "--signature",
        str(signature),
    )
    assert result.returncode != 0
