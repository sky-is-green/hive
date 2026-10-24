"""``hive serve`` — apply one stack, keep it up, unload it on exit.

The command is deliberately engine-blind: a stack tier either names an engine
through its ``launch`` block (``strata``, ``deepinfer``, or whatever adapter is
registered) or is a llama-server tier with a ``backend``; an endpoint tier is
adopted after a health check.  This module only owns the lifecycle — it does not
know how any engine works, and adding one does not change this file.

Usage::

    hive serve peer-2tier            # a stack under stacks/
    hive serve stacks/my-stack.json  # or a path
    hive serve peer-2tier --once     # apply, print, unload (smoke test)
    hive serve --list                # what is available

Exit codes: 0 on a clean stop, 1 when the stack cannot be applied, 2 on a usage
error.  SIGINT/SIGTERM unload the stack before exiting.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

from harness.models import LlamaServerManager
from harness.stack.manager import StackManager
from harness.stack.schema import Stack, list_stacks, load_stack, validate_shape


def _load_stack_arg(value: str, stacks_dir: str | None) -> Stack:
    """A saved stack name (``stacks/<name>.json``) or an explicit path."""
    path = Path(value)
    if path.suffix == ".json" or path.is_file() or "/" in value:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"no stack file at {path}") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"stack file {path} is not valid JSON: {exc}") from exc
        stack = Stack.from_dict(data)
        problems = validate_shape(stack)
        if problems:
            raise ValueError(
                f"stack {stack.name!r} has an invalid shape: " + "; ".join(problems))
        return stack
    return load_stack(value, root=stacks_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hive serve",
        description="Apply a stack (one process per tier, any engine) and keep it serving.")
    parser.add_argument("stack", nargs="?",
                        help="stack name under stacks/, or a path to a .json stack")
    parser.add_argument("--list", action="store_true",
                        help="list the saved stacks and exit")
    parser.add_argument("--stacks-dir", default=None,
                        help="directory of stack files (default: <repo>/stacks)")
    parser.add_argument("--models-dir", default=None, help="GGUF model directory")
    parser.add_argument("--llama-server", default=None, help="llama-server binary")
    parser.add_argument("--log-dir", default=None, help="engine log directory")
    parser.add_argument("--host", default="127.0.0.1",
                        help="host the tiers bind/publish on (default 127.0.0.1)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-tier startup timeout in seconds (default 300)")
    parser.add_argument("--once", action="store_true",
                        help="apply, print the endpoints, unload, exit (smoke test)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="print the applied stack as JSON instead of a table")
    return parser


def _endpoint_of(row: dict[str, Any], host: str) -> str:
    base = row.get("base_url")
    if base:
        return str(base)
    port = row.get("port")
    return f"http://{host}:{port}/v1" if port else ""


def _print_table(rows: list[dict[str, Any]], host: str) -> None:
    width = max((len(str(r.get("role", ""))) for r in rows), default=4)
    for row in rows:
        role = str(row.get("role", "?"))
        endpoint = _endpoint_of(row, host)
        ctx = row.get("ctx")
        backend = row.get("backend") or ""
        extra = " ".join(str(x) for x in (backend, f"ctx {ctx}" if ctx else "") if x)
        print(f"  {role:<{width}}  {endpoint}")
        if extra:
            print(f"  {'':<{width}}  {extra}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        rows = list_stacks(root=args.stacks_dir)
        if not rows:
            print("no saved stacks")
            return 0
        for row in rows:
            roles = ",".join(row.get("roles", []))
            print(f"  {row.get('name', '?'):<24} {roles}")
        return 0

    if not args.stack:
        parser.error("a stack name or path is required (or --list)")

    try:
        stack = _load_stack_arg(args.stack, args.stacks_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"hive serve: {exc}", file=sys.stderr)
        return 2

    models = LlamaServerManager(
        binary=Path(args.llama_server) if args.llama_server else None,
        models_dir=Path(args.models_dir) if args.models_dir else None,
        host=args.host,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        startup_timeout=args.timeout,
    )
    manager = StackManager(models, stacks_root=args.stacks_dir)

    try:
        result = manager.apply(stack)
    except RuntimeError as exc:
        print(f"hive serve: {exc}", file=sys.stderr)
        return 1

    status = manager.status()
    rows = list(status.get("tiers", []))
    if args.as_json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(f"stack {stack.name!r} applied")
        _print_table(rows, args.host)
        for warning in status.get("warnings", []) or []:
            print(f"  ! {warning}")

    if args.once:
        manager.unload()
        return 0

    print("serving; Ctrl-C to stop", flush=True)

    def _stop(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        manager.unload()
    print("stopped", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
