#!/usr/bin/env python3
"""Sign or verify an arbitrary release artifact with an Ed25519 key.

Key files use the same JSON format as ``sign_plugin.py``:
``key_id``, ``private_key_pem``, ``public_key_pem``, ``public_key``.
Signatures are base64 text in a ``.sig`` sidecar.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path


def read_key(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def openssl(args: list[str]) -> bytes:
    result = subprocess.run(["openssl", *args], capture_output=True, check=False)
    if result.returncode != 0:
        raise SystemExit(result.stderr.decode("utf-8", "replace").strip())
    return result.stdout


def sign(key_file: Path, artifact: Path, signature: Path) -> None:
    key = read_key(key_file)
    with tempfile.TemporaryDirectory() as temporary:
        private_path = Path(temporary) / "key.pem"
        private_path.write_text(key["private_key_pem"], encoding="utf-8")
        raw = openssl(
            [
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(private_path),
                "-in",
                str(artifact),
            ]
        )
    signature.write_text(base64.b64encode(raw).decode("ascii") + "\n", encoding="utf-8")
    print(f"signed {artifact} with {key['key_id']} -> {signature}")


def verify(key_file: Path, artifact: Path, signature: Path) -> None:
    key = read_key(key_file)
    raw = base64.b64decode(signature.read_text(encoding="utf-8").strip(), validate=True)
    with tempfile.TemporaryDirectory() as temporary:
        public_path = Path(temporary) / "key.pub"
        signature_path = Path(temporary) / "signature"
        public_path.write_text(key["public_key_pem"], encoding="utf-8")
        signature_path.write_bytes(raw)
        result = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-verify",
                "-pubin",
                "-inkey",
                str(public_path),
                "-rawin",
                "-in",
                str(artifact),
                "-sigfile",
                str(signature_path),
            ],
            capture_output=True,
            check=False,
        )
    if result.returncode != 0:
        raise SystemExit(result.stderr.decode("utf-8", "replace").strip())
    print(f"verified {artifact} with {key['key_id']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sign", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--key-file", type=Path, required=True)
        command.add_argument("--file", type=Path, required=True)
        command.add_argument("--signature", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "sign":
        sign(args.key_file, args.file, args.signature)
    else:
        verify(args.key_file, args.file, args.signature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
