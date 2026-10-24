"""``hive serve`` — apply one stack, keep it up, unload it on exit.

The command is deliberately engine-blind: a stack tier either names an engine
through its ``launch`` block (``strata``, ``deepinfer``, or whatever adapter is
registered) or is a llama-server tier with a ``backend``; an endpoint tier is
adopted after a health check.  This module only owns the lifecycle — it does not
know how any engine works, and adding one does not change this file.

For the common case it can also write the stack for you: ``--model`` turns a
HuggingFace repo (or a local GGUF) and the cards it finds into a stack, so a
two-card box does not need a hand-edited JSON.

Usage::

    hive serve peer-2tier                  # a stack under stacks/
    hive serve stacks/my-stack.json        # or a path
    hive serve --model org/repo:model.gguf --cards 2
    hive serve --model /models/model.gguf --cards 1
    hive serve --model org/repo --file model.gguf --cards 2 --save my-stack
    hive serve peer-2tier --detach         # background; use hive status/stop/logs
    hive serve peer-2tier --once           # apply, print, unload (smoke test)
    hive serve --list                      # what is available

Exit codes: 0 on a clean stop, 1 when the stack cannot be applied, 2 on a usage
error.  SIGINT/SIGTERM unload the stack before exiting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

from harness import backends as backend_manifest
from harness import serve_state, termui
from harness.hardware import linux_amd_devices
from harness.models import LlamaServerManager
from harness.stack.manager import StackManager
from harness.stack.schema import (
    Stack, list_stacks_in, load_stack_in, stack_path_in, stacks_dir,
    validate_shape)

_NAME = "serve"


def _stacks_directory(value: str | None) -> Path:
    """The stack directory: the flag if given, else the repo's ``stacks/``."""
    return Path(value).expanduser() if value else stacks_dir()


def _load_stack_arg(value: str, directory: str | None) -> Stack:
    """A name in the stack directory, or an explicit JSON path."""
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
    return load_stack_in(value, _stacks_directory(directory))


def _detect_devices() -> list[dict]:
    """The cards this machine can serve on, in HIP index order."""
    try:
        return linux_amd_devices()
    except OSError:
        return []


def _select_devices(devices: list[dict], count: int | None,
                    requested: str | None) -> list[dict]:
    if requested:
        try:
            wanted = [int(x) for x in requested.replace(" ", "").split(",") if x]
        except ValueError:
            raise ValueError("--devices must be comma-separated HIP indices, e.g. 0,1") from None
        by_index = {int(d["index"]): d for d in devices}
        missing = [i for i in wanted if i not in by_index]
        if missing:
            raise ValueError(
                f"--devices {missing} not among the detected cards {sorted(by_index)}")
        return [by_index[i] for i in wanted]
    if count is not None and count > len(devices):
        raise ValueError(f"--cards {count} but only {len(devices)} GPU(s) detected")
    n = count if count is not None else len(devices)
    return devices[:n] if n else []


def _pick_backend(devices: list[dict], requested: str | None) -> str | None:
    """The installed backend to use, from the manifest (not by convention).

    ``None`` means nothing suitable is installed; the caller then uses the
    default llama-server binary rather than guess a backend that would fail at
    load time.
    """
    if requested:
        return requested
    return backend_manifest.for_accelerator("amd" if devices else "cpu")


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    return slug[:64] or "stack"


def _ensure_hf_model(model: str, file: str, models_dir: str | None) -> None:
    """Make sure a ``repo:file`` model is in the local library.

    ``plan_residency`` sizes a tier from a real local file, so a remote-only
    stack is refused before the engine is spawned.  Downloading here also
    gives the CLI a progress bar and one place for ``HF_TOKEN`` errors.
    """
    manager = LlamaServerManager(
        models_dir=Path(models_dir).expanduser() if models_dir else None)
    resolver = getattr(manager, "resolve_model", None)
    if resolver is not None and resolver(file) is not None:
        return
    models_dir_path = getattr(manager, "models_dir", None)
    if models_dir_path is None:  # a test double without a library
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed; install it or pass a local .gguf")
    print(termui.info(f"downloading {model}:{file} ...", stream=sys.stderr),
          file=sys.stderr, flush=True)
    try:
        path = Path(hf_hub_download(
            repo_id=model, filename=file, local_dir=str(models_dir_path)))
    except Exception as exc:  # noqa: BLE001 - gated/network/disk are one line
        raise RuntimeError(f"cannot download {model}:{file}: {exc}") from exc
    size = path.stat().st_size / (1024 ** 3) if path.is_file() else 0.0
    print(f"  {termui.dim(path, stream=sys.stderr)} ({size:.2f} GiB)",
          file=sys.stderr, flush=True)


def _stack_from_model(*, model: str, file: str, devices: list[dict], ctx: int,
                      backend: str | None, role: str,
                      name: str | None) -> tuple[dict, list[str]]:
    tier: dict[str, Any] = {
        "role": role, "repo": model, "file": file, "ctx": ctx, "ngl": 99,
        "cache_k": "q8_0", "cache_v": "q8_0",
    }
    if backend:
        tier["backend"] = backend
    warnings: list[str] = []
    if devices:
        tier["pin"] = ",".join(str(d["index"]) for d in devices)
        warnings += [
            f"card {d['index']} drives a display; expect desktop stutter"
            for d in devices if d.get("display")
        ]
        if len(devices) > 1:
            gb = [float(d.get("memory_gb") or 1.0) for d in devices]
            if max(gb) - min(gb) < 0.5:
                tier["ts"] = ",".join("1" for _ in gb)
            else:
                tier["ts"] = ",".join(f"{x:.0f}" for x in gb)
    stem = _slug(Path(file).stem)
    cards = f"{len(devices)}card" if devices else "cpu"
    data = {"name": name or _slug(f"{stem}-{cards}"), "version": 1, "tiers": [tier]}
    return data, warnings


# ---------------------------------------------------------------------------
# argument surface (shared by ``hive serve`` and the root ``hive`` parser)
# ---------------------------------------------------------------------------

def _add_serve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("stack", nargs="?",
                        help="stack name under stacks/, or a path to a .json stack")
    parser.add_argument("--list", action="store_true",
                        help="list the saved stacks and exit")
    parser.add_argument("--backends", action="store_true",
                        help="list the declared server builds and exit")
    parser.add_argument("--model", default=None,
                        help="HuggingFace repo (or repo:file), or a local .gguf path")
    parser.add_argument("--file", default=None,
                        help="GGUF file inside the repo (with --model repo)")
    parser.add_argument("--cards", type=int, default=None,
                        help="how many detected cards to use (default: all)")
    parser.add_argument("--devices", default=None,
                        help="explicit HIP indices, e.g. 0,1 (overrides --cards)")
    parser.add_argument("--ctx", type=int, default=8192,
                        help="context size for a generated stack (default 8192)")
    parser.add_argument("--backend", default=None,
                        help="server backend (default: the first installed of rocm, vulkan)")
    parser.add_argument("--fetch", action="store_true",
                        help="fetch a missing engine backend from the manifest before applying")
    parser.add_argument("--role", default="face",
                        help="role to give a generated tier (default: face)")
    parser.add_argument("--save", default=None,
                        help="write a generated stack to stacks/<name>.json")
    parser.add_argument("--name", default=None,
                        help="name this serve (default: the stack name)")
    parser.add_argument("--detach", action="store_true",
                        help="run in the background; manage with status/stop/logs")
    parser.add_argument("--replace", action="store_true",
                        help="stop a running serve with the same name first")
    parser.add_argument("--stacks-dir", default=None,
                        help="directory of stack files (default: <repo>/stacks)")
    parser.add_argument("--models-dir", default=None, help="GGUF model directory")
    parser.add_argument("--llama-server", default=None, help="server binary")
    parser.add_argument("--log-dir", default=None, help="engine log directory")
    parser.add_argument("--host", default="127.0.0.1",
                        help="host the tiers bind/publish on (default 127.0.0.1)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-tier startup timeout in seconds (default 300)")
    parser.add_argument("--once", action="store_true",
                        help="apply, print the endpoints, unload, exit (smoke test)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="print status as JSON instead of a table")
    parser.add_argument("--_child", dest="_child", action="store_true",
                        help=argparse.SUPPRESS)


def _standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hive serve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Apply a stack (one process per tier, any engine) and keep it serving.",
        epilog=(
            "examples:\n"
            "  hive serve peer-2tier --detach\n"
            "  hive serve --model org/repo:model.Q4_K_M.gguf --cards 2\n"
            "  hive serve --model /models/model.gguf --once\n"
            "  hive serve --list\n"))
    _add_serve_args(parser)
    return parser


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``hive serve`` on the root CLI's subparser group."""
    parser = subparsers.add_parser(
        "serve",
        help="apply a stack and serve it (foreground or detached)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Apply a stack (one process per tier, any engine) and keep it serving.",
        epilog=(
            "examples:\n"
            "  hive serve peer-2tier --detach\n"
            "  hive serve --model org/repo:model.Q4_K_M.gguf --cards 2\n"
            "  hive serve --model /models/model.gguf --once\n"))
    _add_serve_args(parser)
    parser.set_defaults(func=run)
    return parser


# ---------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------

def _prepare(args: argparse.Namespace,
             parser: argparse.ArgumentParser) -> tuple[Stack, dict | None, list[str]]:
    """Validate the request and return ``(stack, generated, warnings)``."""
    generated: dict | None = None
    warnings: list[str] = []
    if args.model:
        if args.stack:
            parser.error("give either a stack name or --model, not both")
        raw = str(args.model)
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            if args.file:
                raise ValueError("--file is only for a HuggingFace repo, not a local path")
            model, file = "local", str(candidate.resolve())
        elif ":" in raw and "/" in raw.partition(":")[0]:
            # org/repo:file.gguf — a Hugging Face file, not a missing local path
            model, _, file = raw.partition(":")
            if not file:
                raise ValueError(
                    "--model needs the GGUF file: --file NAME or --model repo:file")
        elif candidate.suffix.lower() == ".gguf":
            raise FileNotFoundError(f"no such GGUF file: {candidate}")
        else:
            model, file = raw, args.file
            if file is None and ":" in model:
                model, _, file = model.partition(":")
            if not file:
                raise ValueError(
                    "--model needs the GGUF file: --file NAME or --model repo:file")
        if args.ctx <= 0:
            raise ValueError("--ctx must be positive")
        try:
            devices = _select_devices(_detect_devices(), args.cards, args.devices)
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        backend = _pick_backend(devices, args.backend)
        if args.backend and not args.fetch:
            try:
                chosen = backend_manifest.resolve(args.backend)
            except KeyError as exc:
                raise ValueError(str(exc)) from None
            if not chosen.get("installed"):
                raise ValueError(
                    f"backend {args.backend!r} is not installed "
                    f"({chosen.get('binary')}); run `hive backends fetch "
                    f"{args.backend}` or add --fetch")
        # validate the flags before a multi-GB download
        if model != "local":
            _ensure_hf_model(model, file, args.models_dir)
        generated, warnings = _stack_from_model(
            model=model, file=file, devices=devices, ctx=args.ctx,
            backend=backend, role=args.role, name=args.save or args.name)
    elif not args.stack:
        parser.error("a stack name, a path, or --model is required (or --list)")

    if generated is not None:
        stack = Stack.from_dict(generated)
        problems = validate_shape(stack)
        if problems:
            raise ValueError(
                f"generated stack {stack.name!r} is invalid: " + "; ".join(problems))
        if args.save:
            path = stack_path_in(args.save, _stacks_directory(args.stacks_dir))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")
            if not args.as_json:
                print(f"wrote {path}")
    else:
        stack = _load_stack_arg(args.stack, args.stacks_dir)
    return stack, generated, warnings


def _stack_file(args: argparse.Namespace) -> str | None:
    if not args.stack:
        return None
    path = Path(args.stack).expanduser()
    if path.suffix == ".json" or path.is_file() or "/" in args.stack:
        return str(path.resolve())
    return None


def _check_existing(name: str, replace: bool) -> dict | None:
    """Refuse a duplicate name, or stop it when ``--replace`` was given."""
    try:
        record = serve_state.read_record(name)
    except ValueError:
        serve_state.remove_record(name)
        return None
    if record is None:
        return None
    if serve_state.pid_matches(record):
        if not replace:
            raise RuntimeError(
                f"serve {name!r} is already running (pid {record['pid']}); "
                "use --replace or --name")
        ok, message = serve_state.terminate(record["pid"], force=False, timeout=15.0)
        if not ok:
            raise RuntimeError(f"cannot replace serve {name!r}: {message}")
    serve_state.remove_record(name)
    return record


# ---------------------------------------------------------------------------
# detached lifecycle
# ---------------------------------------------------------------------------

def _child_argv(args: argparse.Namespace, name: str) -> list[str]:
    """The argv for the detached child, with every meaningful flag preserved."""
    argv = [sys.executable, "-m", "harness.cli", "serve"]
    if args.model:
        argv += ["--model", str(args.model)]
        if args.file:
            argv += ["--file", str(args.file)]
    else:
        target = str(args.stack)
        path = Path(target).expanduser()
        if path.suffix == ".json" or path.is_file() or "/" in target:
            target = str(path.resolve())
        argv.append(target)
    for flag, attr in (
        ("--cards", "cards"),
        ("--devices", "devices"),
        ("--ctx", "ctx"),
        ("--backend", "backend"),
        ("--role", "role"),
        ("--save", "save"),
        ("--stacks-dir", "stacks_dir"),
        ("--models-dir", "models_dir"),
        ("--llama-server", "llama_server"),
        ("--log-dir", "log_dir"),
        ("--host", "host"),
        ("--timeout", "timeout"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            argv += [flag, str(value)]
    argv += ["--name", name, "--_child"]
    return argv


def _detach(args: argparse.Namespace, stack: Stack, warnings: list[str]) -> int:
    if args.once:
        print("hive serve: --detach and --once are mutually exclusive",
              file=sys.stderr)
        return 2
    name = args.name or stack.name
    try:
        serve_state.validate_name(name)
    except ValueError as exc:
        print(f"hive serve: {exc}", file=sys.stderr)
        return 2
    try:
        _check_existing(name, args.replace)
    except RuntimeError as exc:
        print(f"hive serve: {exc}", file=sys.stderr)
        return 1

    log = serve_state.log_path(name)
    argv = _child_argv(args, name)
    proc = serve_state.spawn(argv, log)
    deadline = time.monotonic() + max(10.0, float(args.timeout) + 15.0)
    record: dict | None = None
    try:
        with termui.spinner(f"starting {name}"):
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    record = serve_state.read_record(name)
                except ValueError:
                    record = None
                if record and int(record.get("pid") or 0) == proc.pid:
                    break
                time.sleep(0.2)
    except KeyboardInterrupt:
        serve_state.terminate(proc.pid, force=True, timeout=5.0)
        print("hive serve: interrupted while starting; stopped", file=sys.stderr)
        return 130

    if not record or int(record.get("pid") or 0) != proc.pid:
        if proc.poll() is None:
            serve_state.terminate(proc.pid, force=True, timeout=5.0)
            reason = f"did not become ready within {deadline - time.monotonic():.0f}s"
        else:
            reason = f"exited with code {proc.returncode}"
        print(termui.fail(f"{name!r} {reason}", stream=sys.stderr), file=sys.stderr)
        print(f"  {termui.dim('log')}  {log}", file=sys.stderr)
        try:
            for line in serve_state.tail_lines(log, 20):
                print(f"  {termui.dim(line, stream=sys.stderr)}", file=sys.stderr)
        except OSError:
            pass
        return 1

    if args.as_json:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    print(termui.ok(f"serving {name!r} · pid {record['pid']} · detached"))
    _print_table(list(record.get("tiers") or []), args.host)
    print(f"  {termui.dim('log')}  {log}")
    for warning in list(warnings) + list(record.get("warnings") or []):
        print(f"  {termui.warn(warning)}")
    return 0


# ---------------------------------------------------------------------------
# output and lifecycle
# ---------------------------------------------------------------------------

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
        print(f"  {termui.accent(role.ljust(width))}  {endpoint}")
        if extra:
            print(f"  {'':<{width}}  {termui.dim(extra)}")


def _record(args: argparse.Namespace, stack: Stack, status: dict[str, Any],
            models: LlamaServerManager, warnings: list[str]) -> dict[str, Any]:
    name = args.name or stack.name
    return {
        "version": 1,
        "name": name,
        "pid": os.getpid(),
        "mode": "detached" if args._child else "foreground",
        "stack": stack.name,
        "stack_file": _stack_file(args),
        "host": args.host,
        "started_at": time.time(),
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cmdline": serve_state.pid_cmdline(os.getpid()),
        "log_path": str(serve_state.log_path(name)) if args._child else None,
        "log_dir": str(models.log_dir),
        "tiers": list(status.get("tiers") or []),
        "warnings": list(warnings) + list(status.get("warnings") or []),
    }


def _ensure_fetch(args: argparse.Namespace, stack: Stack) -> None:
    """Fetch any missing local backend before the stack is applied."""
    missing: list[str] = []
    for tier in stack.tiers:
        if getattr(tier, "endpoint", None):
            continue
        name = str(tier.backend or "vulkan")
        if name in missing:
            continue
        try:
            installed = backend_manifest.resolve(name).get("installed")
        except KeyError as exc:
            raise RuntimeError(str(exc)) from None
        if not installed:
            missing.append(name)
    for name in missing:
        print(ui.info(f"fetching {name} ...", stream=sys.stderr),
              file=sys.stderr, flush=True)
        last = {"pct": -1}

        def progress(done: int, total: int) -> None:
            if getattr(args, "as_json", False):
                return
            if total > 0:
                pct = int(done * 100 / total)
                if pct != last["pct"]:
                    last["pct"] = pct
                    sys.stderr.write(
                        f"\r  {pct:3d}%  {done / 1048576:.1f} / {total / 1048576:.1f} MiB")
                    if done >= total:
                        sys.stderr.write("\n")
            else:
                sys.stderr.write(f"\r  {done / 1048576:.1f} MiB")
            sys.stderr.flush()

        try:
            backend_manifest.fetch_backend(name, progress=progress)
        except (KeyError, RuntimeError, OSError) as exc:
            raise RuntimeError(f"cannot fetch backend {name!r}: {exc}") from exc
        print(ui.ok(f"installed {name}", stream=sys.stderr), file=sys.stderr)


def run(args: argparse.Namespace) -> int:
    if args.backends:
        rows = backend_manifest.rows()
        if args.as_json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            termui.header("backends")
            termui.table(["NAME", "STATUS", "ACCELERATORS", "LABEL"],
                     [[row["name"], "installed" if row["installed"] else "-",
                       ",".join(row.get("accelerators") or []),
                       row.get("label", "")] for row in rows])
        return 0

    if args.list:
        directory = _stacks_directory(args.stacks_dir)
        rows = list_stacks_in(directory)
        if args.as_json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            termui.header("stacks", str(directory))
            if not rows:
                print(f"  {termui.dim('no saved stacks')}")
            else:
                termui.table(["NAME", "ROLES"],
                         [[row.get("name", "?"),
                           ",".join(row.get("roles", []))] for row in rows])
        return 0

    parser = _standalone_parser()
    try:
        stack, _generated, warnings = _prepare(args, parser)
    except (FileNotFoundError, ValueError) as exc:
        print(f"hive serve: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"hive serve: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "fetch", False):
        try:
            _ensure_fetch(args, stack)
        except RuntimeError as exc:
            print(f"hive serve: {exc}", file=sys.stderr)
            return 1

    if args.detach and not args._child:
        return _detach(args, stack, warnings)

    name = args.name or stack.name
    try:
        serve_state.validate_name(name)
    except ValueError as exc:
        print(f"hive serve: {exc}", file=sys.stderr)
        return 2
    if not args.once:
        try:
            _check_existing(name, args.replace)
        except RuntimeError as exc:
            print(f"hive serve: {exc}", file=sys.stderr)
            return 1

    models = LlamaServerManager(
        binary=Path(args.llama_server) if args.llama_server else None,
        models_dir=Path(args.models_dir) if args.models_dir else None,
        host=args.host,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        startup_timeout=args.timeout,
    )
    manager = StackManager(models, stacks_root=args.stacks_dir)

    try:
        with termui.spinner(f"applying {stack.name}"):
            manager.apply(stack)
    except RuntimeError as exc:
        print(f"hive serve: {exc}", file=sys.stderr)
        return 1

    status = manager.status()
    rows = list(status.get("tiers", []))
    record: dict[str, Any] | None = None
    if not args.once:
        record = _record(args, stack, status, models, warnings)
        try:
            serve_state.write_record(record)
        except (OSError, ValueError) as exc:
            print(f"hive serve: cannot write the serve record: {exc}", file=sys.stderr)
            record = None

    if args.as_json:
        print(json.dumps(record or status, indent=2, sort_keys=True))
    else:
        print(termui.ok(f"stack {stack.name!r} applied"))
        _print_table(rows, args.host)
        for warning in warnings + list(status.get("warnings", []) or []):
            print(f"  {termui.warn(warning)}")
        if record is not None:
            print(f"  {termui.dim('name')} {record['name']}  "
                  f"{termui.dim('pid')} {record['pid']}  "
                  f"{termui.dim(record['mode'])}")
            if args._child:
                print(f"  {termui.dim('log')}  {record['log_path']}")

    if args.once:
        manager.unload()
        if not args.as_json:
            print(f"  {termui.dim('unloaded')}")
        return 0

    if not args.as_json:
        print(f"  {termui.accent('serving')}  {termui.dim('Ctrl-C to stop')}", flush=True)

    def _stop(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):  # pragma: no cover - platform differences
            pass
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            manager.unload()
        finally:
            if record is not None:
                serve_state.remove_record(record["name"])
    print("stopped", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point (``hive serve`` without the root command)."""
    parser = _standalone_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
