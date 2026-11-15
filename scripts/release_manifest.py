#!/usr/bin/env python3
"""Build the release manifest, SHA256SUMS, and CycloneDX SBOM.

The script is deliberately offline: it hashes the release binaries, reads the
workspace version/commit, and (unless ``--no-cargo-metadata``) records the
Cargo dependency graph. Signing is delegated to minisign so the private key
never has to live in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BINARIES = ("hive", "hive-daemon", "hive-encoder", "hive-memory", "hive-sandbox")


def workspace_version() -> str:
    text = (REPO / "rust" / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit("cannot find the workspace version in rust/Cargo.toml")
    return match.group(1)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def target_name() -> str:
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    return f"{machine}-{platform.system().lower()}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cargo_components() -> list[dict]:
    try:
        result = subprocess.run(
            ["cargo", "metadata", "--format-version", "1"],
            cwd=REPO / "rust",
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    data = json.loads(result.stdout)
    components = []
    for package in data.get("packages", []):
        component = {
            "type": "library",
            "name": package.get("name", ""),
            "version": package.get("version", ""),
            "purl": f"pkg:cargo/{package.get('name', '')}@{package.get('version', '')}",
        }
        license = package.get("license")
        if license:
            component["licenses"] = [{"license": {"id": license}}]
        components.append(component)
    return components


def build_manifest(bin_dir: Path, out_dir: Path, allow_missing: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name in DEFAULT_BINARIES:
        path = bin_dir / name
        if not path.is_file():
            if allow_missing:
                continue
            raise SystemExit(f"missing release binary: {path}")
        files.append(
            {
                "name": name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    if not files:
        raise SystemExit("no release binaries found")
    version = workspace_version()
    manifest = {
        "schema": 1,
        "name": "hive",
        "version": version,
        "commit": git_commit(),
        "target": target_name(),
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "files": files,
        "checksums": "SHA256SUMS",
        "sbom": "sbom.cdx.json",
    }
    (out_dir / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {item['name']}\n" for item in files),
        encoding="utf-8",
    )
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": manifest["generated_at"],
            "component": {
                "type": "application",
                "name": "hive",
                "version": version,
            },
        },
        "components": [
            {
                "type": "application",
                "name": item["name"],
                "version": version,
                "hashes": [{"alg": "SHA-256", "content": item["sha256"]}],
                "properties": [
                    {"name": "hive:target", "value": manifest["target"]},
                    {"name": "hive:bytes", "value": str(item["bytes"])},
                ],
            }
            for item in files
        ],
    }
    (out_dir / "sbom.cdx.json").write_text(
        json.dumps(sbom, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify(out_dir: Path) -> int:
    checksums = out_dir / "SHA256SUMS"
    if not checksums.is_file():
        print(f"missing {checksums}", file=sys.stderr)
        return 2
    failures = 0
    for line in checksums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        path = out_dir / name
        if not path.is_file():
            print(f"MISSING {name}", file=sys.stderr)
            failures += 1
            continue
        actual = sha256(path)
        if actual != expected:
            print(f"MISMATCH {name}: {actual} != {expected}", file=sys.stderr)
            failures += 1
    if failures:
        return 1
    print(f"verified {checksums}")
    return 0


def sign(out_dir: Path, key: Path) -> int:
    minisign = shutil.which("minisign")
    if minisign is None:
        print("minisign is not installed", file=sys.stderr)
        return 2
    if not key.is_file():
        print(f"signing key not found: {key}", file=sys.stderr)
        return 2
    for name in ("release-manifest.json", "SHA256SUMS"):
        target = out_dir / name
        result = subprocess.run(
            [minisign, "-S", "-s", str(key), "-m", str(target)],
            check=False,
        )
        if result.returncode != 0:
            return result.returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=REPO / "rust" / "target" / "release")
    parser.add_argument("--out", type=Path, default=REPO / "dist")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="hash the binaries that exist instead of failing",
    )
    parser.add_argument(
        "--no-cargo-metadata",
        action="store_true",
        help="skip third-party components in the SBOM",
    )
    parser.add_argument("--verify", type=Path, metavar="DIST_DIR")
    parser.add_argument("--sign-key", type=Path, metavar="MINISIGN_KEY")
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.verify)
    if args.sign_key:
        return sign(args.out, args.sign_key)
    manifest = build_manifest(args.bin_dir, args.out, args.allow_missing)
    if not args.no_cargo_metadata:
        sbom_path = args.out / "sbom.cdx.json"
        sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
        sbom["components"].extend(cargo_components())
        sbom_path.write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out / 'release-manifest.json'} "
        f"({len(manifest['files'])} binaries, hive {manifest['version']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
