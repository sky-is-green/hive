#!/usr/bin/env python3
"""Generate plugin signing keys and sign/verify plugin manifests.

The signing payload must match ``PluginManifest::signing_payload`` in
``rust/crates/hive-plugins/src/lib.rs`` exactly: JSON with these fields in this
order and no spaces.

Signing uses the OpenSSL CLI (Ed25519), so the script has no Python crypto
dependency. The private key is a PKCS#8 PEM stored in the key JSON with mode
0600; the public key is raw 32-byte base64 for the Rust trust store.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

DEFAULT_FUEL = 10_000_000
DEFAULT_MEMORY_MB = 64
DEFAULT_MAX_OUTPUT = 1_048_576


def payload(manifest: dict, key_id: str) -> bytes:
    body = {
        "name": manifest["name"],
        "version": manifest.get("version", ""),
        "module": manifest["module"],
        "sha256": manifest.get("sha256", ""),
        "fuel": manifest.get("fuel", DEFAULT_FUEL),
        "memory_mb": manifest.get("memory_mb", DEFAULT_MEMORY_MB),
        "max_output_bytes": manifest.get("max_output_bytes", DEFAULT_MAX_OUTPUT),
        "key_id": key_id,
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def read_key(path: Path) -> tuple[str, str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["key_id"], data["private_key_pem"], data["public_key_pem"]


def run_openssl(args: list[str], input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["openssl", *args],
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.decode("utf-8", "replace").strip())
    return result.stdout


def public_der(private_key_path: Path) -> bytes:
    return run_openssl(
        ["pkey", "-in", str(private_key_path), "-pubout", "-outform", "DER"]
    )


def generate(key_file: Path, key_id: str, trusted_file: Path | None) -> None:
    key_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        private_path = Path(temporary) / "key.pem"
        run_openssl(["genpkey", "-algorithm", "ED25519", "-out", str(private_path)])
        private_pem = private_path.read_text(encoding="utf-8")
        der = public_der(private_path)
        public_pem = run_openssl(
            ["pkey", "-in", str(private_path), "-pubout"]
        ).decode("utf-8")
    if len(der) < 32:
        raise SystemExit("openssl returned a short Ed25519 public key")
    public_raw = der[-32:]
    key_file.write_text(
        json.dumps(
            {
                "key_id": key_id,
                "private_key_pem": private_pem,
                "public_key_pem": public_pem,
                "public_key": base64.b64encode(public_raw).decode("ascii"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(key_file, 0o600)
    if trusted_file:
        trusted_file.parent.mkdir(parents=True, exist_ok=True)
        trusted_file.write_text(
            json.dumps(
                {"keys": {key_id: base64.b64encode(public_raw).decode("ascii")}},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"key_id: {key_id}")
    print(f"public_key: {base64.b64encode(public_raw).decode('ascii')}")
    print(f"wrote {key_file}" + (f" and {trusted_file}" if trusted_file else ""))


def sign(plugin_dir: Path, key_file: Path) -> None:
    key_id, private_pem, _ = read_key(key_file)
    manifest_path = plugin_dir / "plugin.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("name") != plugin_dir.name:
        raise SystemExit("plugin manifest name must match its directory")
    module = plugin_dir / manifest["module"]
    if not module.is_file():
        raise SystemExit(f"plugin module not found: {module}")
    manifest["sha256"] = hashlib.sha256(module.read_bytes()).hexdigest()
    manifest["key_id"] = key_id
    with tempfile.TemporaryDirectory() as temporary:
        private_path = Path(temporary) / "key.pem"
        payload_path = Path(temporary) / "payload"
        private_path.write_text(private_pem, encoding="utf-8")
        payload_path.write_bytes(payload(manifest, key_id))
        signature = run_openssl(
            [
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(private_path),
                "-in",
                str(payload_path),
            ]
        )
    manifest["signature"] = base64.b64encode(signature).decode("ascii")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"signed {manifest_path} with {key_id}")


def verify(plugin_dir: Path, key_file: Path) -> None:
    key_id, _, public_pem = read_key(key_file)
    manifest = json.loads((plugin_dir / "plugin.json").read_text(encoding="utf-8"))
    if manifest.get("key_id") != key_id:
        raise SystemExit("manifest key_id does not match the key file")
    signature = base64.b64decode(manifest["signature"], validate=True)
    with tempfile.TemporaryDirectory() as temporary:
        public_path = Path(temporary) / "key.pub"
        signature_path = Path(temporary) / "signature"
        payload_path = Path(temporary) / "payload"
        public_path.write_text(public_pem, encoding="utf-8")
        signature_path.write_bytes(signature)
        payload_path.write_bytes(payload(manifest, key_id))
        result = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-verify",
                "-pubin",
                "-inkey",
                str(public_path),
                "-rawin",
                "-sigfile",
                str(signature_path),
                "-in",
                str(payload_path),
            ],
            capture_output=True,
            check=False,
        )
    if result.returncode != 0:
        raise SystemExit(result.stderr.decode("utf-8", "replace").strip())
    print(f"verified {plugin_dir / 'plugin.json'} with {key_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--key-file", type=Path, required=True)
    gen.add_argument("--key-id", required=True)
    gen.add_argument("--trusted-file", type=Path)
    sign_cmd = sub.add_parser("sign")
    sign_cmd.add_argument("--plugin-dir", type=Path, required=True)
    sign_cmd.add_argument("--key-file", type=Path, required=True)
    verify_cmd = sub.add_parser("verify")
    verify_cmd.add_argument("--plugin-dir", type=Path, required=True)
    verify_cmd.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(args.key_file, args.key_id, args.trusted_file)
    elif args.command == "sign":
        sign(args.plugin_dir, args.key_file)
    else:
        verify(args.plugin_dir, args.key_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
