#!/usr/bin/env python3
"""Regenerate the frozen CLI golden corpus from the Python implementation.

This is a deliberate, reviewed step: run it only when the CLI contract changes,
inspect the diff under ``tests/conformance/golden/``, and commit it with the
contract change. Tests compare the Rust binary to these files and do not need
Python at run time.

    python tests/conformance/golden/generate.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import golden_support  # noqa: E402


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="hive-golden-"))
    try:
        golden_support.build_library(root)
        golden_support.build_stacks(root)
        env = golden_support.build_env(root, sys.executable)
        for name, args in golden_support.CASES:
            payload = golden_support.render(
                [sys.executable], args, root, env, module=True
            )
            if name == "help" and isinstance(payload, dict):
                payload = golden_support.trim_help_root(payload)
            path = HERE / f"{name}.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"wrote {path.relative_to(golden_support.REPO)}")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
