#!/usr/bin/env python
"""Start the five Hive 1.7B models as local Studio providers.

The Studio is already running (``python -m harness``). This script asks its
``/v1/server/start`` endpoint to load each GGUF on its canonical port, with the
GGUF alias the role resolver expects. It is deliberately a thin client: the
manager still owns the process, the provider registration and the engine
profile.

    python scripts/load_roster.py --dry-run
    python scripts/load_roster.py --backend vulkan

The model files must already be in the Studio's GGUF library. Download them
with the console Models page or ``hive models download`` first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

ROSTER = (
    # key, file stem, port, alias
    ("hive-conductor-1.7b-v1-Q4_K_M", "hive-conductor-1.7b-v1-Q4_K_M",
     8156, "conductor-1.7b"),
    ("hive-reader-1.7b-v1-Q4_K_M", "hive-reader-1.7b-v1-Q4_K_M",
     8159, "reader-1.7b"),
    ("hive-auditor-1.7b-v1-Q4_K_M", "hive-auditor-1.7b-v1-Q4_K_M",
     8162, "auditor-1.7b"),
    ("hive-dispatcher-1.7b-v1-Q4_K_M", "hive-dispatcher-1.7b-v1-Q4_K_M",
     8163, "dispatcher-1.7b"),
    ("hive-clerk-1.7b-v1-Q4_K_M", "hive-clerk-1.7b-v1-Q4_K_M",
     8164, "clerk-1.7b"),
)


def _post(base: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/server/start",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    # localhost must not go through an HTTP proxy
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--backend", default=os.environ.get("HIVE_BACKEND", "vulkan"))
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    failed = False
    for key, stem, port, alias in ROSTER:
        payload = {
            "model": f"{stem}.gguf",
            "key": key,
            "port": port,
            "ctx_size": args.ctx,
            "ngl": args.ngl,
            "backend": args.backend,
            "alias": alias,
            "reasoning": "off",
            "register_provider": True,
            "claim_default": False,
        }
        print(f"{alias:16s} :{port}  <- {stem}.gguf")
        if args.dry_run:
            print("  ", json.dumps(payload, sort_keys=True))
            continue
        try:
            result = _post(args.base_url, payload, args.timeout)
        except RuntimeError as exc:
            failed = True
            print(f"  FAIL {exc}", file=sys.stderr)
            continue
        print(f"  up  {result.get('base_url') or result.get('port')} "
              f"provider={result.get('provider') or '?'}")
    if failed:
        print("one or more models failed; the others were left running",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
