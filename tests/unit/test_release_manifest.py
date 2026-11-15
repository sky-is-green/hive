from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "release_manifest.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_release_manifest_checksums_and_sbom(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "hive").write_bytes(b"hive-binary")
    (binaries / "hive-daemon").write_bytes(b"daemon-binary")
    out = tmp_path / "dist"
    result = _run(
        "--bin-dir",
        str(binaries),
        "--out",
        str(out),
        "--allow-missing",
        "--no-cargo-metadata",
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "release-manifest.json").read_text())
    assert manifest["name"] == "hive"
    assert manifest["version"]
    assert {item["name"] for item in manifest["files"]} == {"hive", "hive-daemon"}
    assert (out / "SHA256SUMS").is_file()
    sbom = json.loads((out / "sbom.cdx.json").read_text())
    assert sbom["bomFormat"] == "CycloneDX"
    assert len(sbom["components"]) == 2
    assert sbom["components"][0]["hashes"][0]["alg"] == "SHA-256"

    for name in ("hive", "hive-daemon"):
        (out / name).write_bytes((binaries / name).read_bytes())
    result = _run("--verify", str(out))
    assert result.returncode == 0, result.stderr

    (out / "hive").write_bytes(b"tampered")
    result = _run("--verify", str(out))
    assert result.returncode == 1
    assert "MISMATCH" in result.stderr


def test_release_manifest_refuses_missing_binaries(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    out = tmp_path / "dist"
    result = _run("--bin-dir", str(binaries), "--out", str(out))
    assert result.returncode != 0
    assert "missing release binary" in result.stderr
