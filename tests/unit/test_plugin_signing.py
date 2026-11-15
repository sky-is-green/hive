from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "sign_plugin.py"
SPEC = importlib.util.spec_from_file_location("sign_plugin", SCRIPT)
SIGN_PLUGIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIGN_PLUGIN)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_plugin_keygen_sign_and_verify(tmp_path):
    key_file = tmp_path / "keys" / "plugin.json"
    trusted = tmp_path / "plugins" / "trusted-keys.json"
    trusted.parent.mkdir()
    result = _run(
        "generate",
        "--key-file",
        str(key_file),
        "--key-id",
        "release-test",
        "--trusted-file",
        str(trusted),
    )
    assert result.returncode == 0, result.stderr
    assert "public_key:" in result.stdout
    assert key_file.stat().st_mode & 0o777 == 0o600
    keys = json.loads(trusted.read_text())["keys"]
    assert "release-test" in keys

    plugin = tmp_path / "plugins" / "demo"
    plugin.mkdir()
    module = plugin / "module.wasm"
    module.write_bytes(b"not-a-real-wasm")
    manifest = {
        "name": "demo",
        "version": "0.1.0",
        "module": "module.wasm",
        "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "fuel": 1000,
        "memory_mb": 1,
    }
    (plugin / "plugin.json").write_text(json.dumps(manifest))
    result = _run("sign", "--plugin-dir", str(plugin), "--key-file", str(key_file))
    assert result.returncode == 0, result.stderr
    signed = json.loads((plugin / "plugin.json").read_text())
    assert signed["key_id"] == "release-test"
    assert signed["signature"]

    result = _run("verify", "--plugin-dir", str(plugin), "--key-file", str(key_file))
    assert result.returncode == 0, result.stderr

    signed["signature"] = "AAAA"
    (plugin / "plugin.json").write_text(json.dumps(signed))
    result = _run("verify", "--plugin-dir", str(plugin), "--key-file", str(key_file))
    assert result.returncode != 0


def test_signing_payload_matches_the_rust_contract():
    payload = SIGN_PLUGIN.payload(
        {
            "name": "demo",
            "version": "0.1.0",
            "module": "module.wasm",
            "sha256": "abc",
            "fuel": 1000,
            "memory_mb": 1,
            "max_output_bytes": 65536,
        },
        "release-test",
    )
    assert payload == (
        b'{"name":"demo","version":"0.1.0","module":"module.wasm","sha256":"abc",'
        b'"fuel":1000,"memory_mb":1,"max_output_bytes":65536,"key_id":"release-test"}'
    )
