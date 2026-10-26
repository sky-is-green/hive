"""``hive`` — the command-line surface for the harness.

The Studio is one way to drive Hive; this module is the other.  It owns the
operational layer and nothing else: serve a stack (foreground or detached),
find it again, read its logs, stop it, manage the local GGUF library and the
saved stacks, check the box, and print the resolved paths.  Every command has a
``--json`` form so a script can consume it.

Design rules:

* one command per verb, no hidden modes;
* expected failures are one line on stderr and exit 1, usage errors exit 2,
  Ctrl-C exits 130, and no traceback unless ``HIVE_DEBUG`` is set;
* a detached serve is recorded as JSON under ``harness_state/serve/`` so
  ``status``/``stop``/``logs`` work from a different process;
* status re-checks the PID and the recorded command line — records are
  advisory, never trusted blindly.

Usage::

    hive doctor
    hive models list
    hive stacks validate ds41-strata
    hive serve ds41-strata --detach
    hive status
    hive logs ds41-strata --tier face -f
    hive stop ds41-strata

The full reference is ``docs/CLI.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from harness import backends as backend_manifest
from harness import serve, serve_state, termui
from harness.hardware import linux_amd_devices
from harness.models import LlamaServerManager
from harness.stack import schema

REPO_ROOT = Path(__file__).resolve().parents[1]
CASCADE_RUNS = REPO_ROOT / "experiments" / "cascade" / "runs"
CASCADE_RUNNER = REPO_ROOT / "experiments" / "cascade" / "run_cascade.py"


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------

def _version() -> str:
    try:
        from importlib.metadata import version as package_version
        return package_version("hivebench")
    except Exception:  # noqa: BLE001 - a source checkout may not be installed
        return "0.2.0"


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _table(headers: list[str], rows: list[list[Any]], indent: str = "  ") -> None:
    termui.table(headers, rows, indent=indent)


def _models_manager(args: argparse.Namespace, *,
                    host: str = "127.0.0.1") -> LlamaServerManager:
    models_dir = getattr(args, "models_dir", None)
    return LlamaServerManager(
        models_dir=Path(models_dir).expanduser() if models_dir else None,
        host=host,
    )


def _stacks_directory(value: str | None) -> Path:
    return Path(value).expanduser() if value else schema.stacks_dir()


def _load_stack_any(value: str, stacks_dir: str | None) -> schema.Stack:
    path = Path(value).expanduser()
    if path.suffix == ".json" or path.is_file() or "/" in value:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"no stack file at {path}") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"stack file {path} is not valid JSON: {exc}") from exc
        stack = schema.Stack.from_dict(data)
        problems = schema.validate_shape(stack)
        if problems:
            raise ValueError(
                f"stack {stack.name!r} has an invalid shape: " + "; ".join(problems))
        return stack
    return schema.load_stack_in(value, _stacks_directory(stacks_dir))


def _tier_endpoint(record: dict, tier: dict) -> str:
    base = tier.get("base_url")
    if base:
        return str(base)
    port = tier.get("port")
    host = record.get("host") or "127.0.0.1"
    return f"http://{host}:{port}/v1" if port else ""


def _probe(base_url: str, timeout: float = 0.8) -> dict:
    """One cheap liveness + model check against an OpenAI-compatible base."""
    if not base_url:
        return {"up": False, "error": "no endpoint"}
    url = base_url.rstrip("/") + "/models"
    try:
        # never send a loopback health check through an HTTP proxy
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=timeout) as response:
            raw = response.read(65536)
        data = json.loads(raw)
        ids = [str(item.get("id")) for item in (data.get("data") or [])
               if isinstance(item, dict) and item.get("id")]
        return {"up": True, "model": ids[0] if ids else None}
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ValueError, TimeoutError) as exc:
        return {"up": False, "error": str(exc)[:160]}


def _resolve_one(records: list[dict], name: str | None) -> dict:
    if name:
        for record in records:
            if record.get("name") == name:
                return record
        raise ValueError(f"no such serve {name!r}")
    if not records:
        raise ValueError("nothing is running")
    if len(records) > 1:
        names = ", ".join(str(record.get("name")) for record in records)
        raise ValueError(f"several serves are running ({names}); name one")
    return records[0]


def _tui_available() -> bool:
    """Whether a bare ``hive`` should take over the terminal."""
    if os.environ.get("HIVE_NO_TUI"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except (AttributeError, OSError):
        return False
    try:
        import curses  # noqa: F401 - availability probe
    except ImportError:
        return False
    return True


def cmd_tui(_args: argparse.Namespace) -> int:
    if not _tui_available():
        print("hive tui: an interactive terminal is required "
              "(use the plain commands in a pipe, or HIVE_NO_TUI=1)", file=sys.stderr)
        return 1
    from harness import tui
    return tui.run()


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


# ---------------------------------------------------------------------------
# landing page / status / stop / logs
# ---------------------------------------------------------------------------

def cmd_home(_args: argparse.Namespace | None = None) -> int:
    """A bare ``hive``: the product's front door, not a usage error."""
    records = serve_state.list_records()
    running = sum(1 for record in records if serve_state.pid_matches(record))
    stale = len(records) - running
    try:
        manager = _models_manager(argparse.Namespace(models_dir=None))
        files = list(manager.models_dir.rglob("*.gguf"))
        total = sum(path.stat().st_size for path in files) / (1024 ** 3)
        model_text = f"{len(files)} · {total:.1f} GiB"
    except OSError:
        model_text = "unavailable"
    try:
        stacks = len(schema.list_stacks_in(_stacks_directory(None)))
    except (OSError, ValueError):
        stacks = 0
    try:
        installed = [row["name"] for row in backend_manifest.rows()
                     if row["installed"]]
    except (OSError, ValueError):
        installed = []
    run_text = "none" if not running else str(running)
    if stale:
        run_text += f" · {stale} stale"
    termui.banner(
        _version(),
        [("running", run_text),
         ("models", model_text),
         ("stacks", str(stacks)),
         ("backends", ", ".join(installed) or "none")],
        ["hive doctor",
         "hive serve peer-2tier --detach",
         "hive status",
         "hive models list"])
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    records = serve_state.list_records()
    pruned: list[str] = []
    views: list[dict] = []
    for record in records:
        alive = serve_state.pid_alive(record["pid"])
        matches = serve_state.pid_matches(record) if alive else False
        if args.prune and not matches:
            serve_state.remove_record(record["name"])
            pruned.append(record["name"])
            continue
        uptime = max(0.0, time.time() - float(record.get("started_at") or time.time()))
        views.append({
            **record,
            "alive": alive,
            "matches": matches,
            "state": "running" if matches else "stale",
            "uptime_s": round(uptime, 1),
            "tiers": list(record.get("tiers") or []),
        })

    if not args.no_probe:
        endpoints: dict[str, str] = {}
        for view in views:
            if view["state"] != "running":
                continue
            for tier in view["tiers"]:
                endpoint = _tier_endpoint(view, tier)
                if endpoint:
                    endpoints[endpoint] = endpoint
        if endpoints:
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(_probe, endpoints))
            probes = dict(zip(endpoints, results))
            for view in views:
                for tier in view["tiers"]:
                    tier["probe"] = probes.get(_tier_endpoint(view, tier))

    if args.as_json:
        _json({"ok": True, "serves": views, "pruned": pruned})
        return 0

    if not views:
        print(f"  {termui.dim('nothing running · start one with: hive serve <stack>')}")
        if pruned:
            print(f"  {termui.ok('pruned ' + ', '.join(pruned))}")
        return 0
    termui.header("serves")
    for view in views:
        name = view.get("name", "?")
        state = (termui.ok("running") if view["state"] == "running"
                 else termui.fail("stale"))
        print(f"  {termui.accent(name)}  {termui.dim('pid')} {view['pid']}  "
              f"{termui.dim(view.get('mode') or 'foreground')}  "
              f"{termui.dim('up ' + serve_state.format_duration(view['uptime_s']))}  "
              f"{state}")
        for tier in view["tiers"]:
            endpoint = _tier_endpoint(view, tier)
            probe = tier.get("probe") or {}
            if probe.get("up"):
                status = termui.ok("up")
            elif probe:
                status = termui.fail("down")
            else:
                status = termui.dim("-")
            ctx = tier.get("ctx") or ""
            backend = tier.get("backend") or ""
            role = str(tier.get("role") or "?")
            print(f"  {termui.accent(role.ljust(8))} {endpoint:<42} "
                  f"{termui.dim(f'ctx {ctx:<6} {backend:<8}')} {status}")
        if view.get("log_path"):
            print(f"  {termui.dim('log')}  {view['log_path']}")
        elif view["state"] == "running":
            print(f"  {termui.dim('log')}  (foreground; logs are on its terminal)")
        if view["state"] != "running":
            print(f"  {termui.warn('prune with: hive status --prune')}")
    if pruned:
        print(f"  {termui.ok('pruned ' + ', '.join(pruned))}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    records = serve_state.list_records()
    if args.all and args.name:
        print("hive stop: give a name or --all, not both", file=sys.stderr)
        return 2
    if args.all:
        targets = records
    elif args.name:
        try:
            targets = [serve_state.read_record(args.name) or _resolve_one(records, args.name)]
        except ValueError as exc:
            print(f"hive stop: {exc}", file=sys.stderr)
            return 1
        targets = [record for record in targets if record]
    elif len(records) == 1:
        targets = records
    elif not records:
        print(f"  {termui.dim('nothing running')}")
        return 0
    else:
        names = ", ".join(str(record.get("name")) for record in records)
        print(f"hive stop: several serves are running ({names}); name one or use --all",
              file=sys.stderr)
        return 2

    if not targets:
        print(f"  {termui.dim('nothing running')}")
        return 0

    results: list[dict] = []
    failed = False
    for record in targets:
        name = str(record.get("name"))
        if not serve_state.pid_matches(record):
            serve_state.remove_record(name)
            results.append({"name": name, "stopped": True, "state": "stale",
                            "message": "removed stale record (process was gone)"})
            continue
        with termui.spinner(f"stopping {name}"):
            ok, message = serve_state.terminate(
                record["pid"], force=args.force, timeout=args.timeout)
        if ok:
            serve_state.remove_record(name)
        else:
            failed = True
        results.append({"name": name, "stopped": ok, "state": "running",
                        "message": message})

    if args.as_json:
        _json({"ok": not failed, "serves": results})
    else:
        for result in results:
            detail = "" if result["message"] in ("stopped", "killed") else (
                f" · {result['message']}")
            text = f"{result['name']}{detail}"
            print(f"  {termui.ok('stopped ' + text) if result['stopped'] else termui.fail('could not stop ' + text)}")
    return 1 if failed else 0


def cmd_logs(args: argparse.Namespace) -> int:
    records = serve_state.list_records()
    try:
        record = _resolve_one(records, args.name)
    except ValueError as exc:
        print(f"hive logs: {exc}", file=sys.stderr)
        return 1

    path: Path | None = None
    source = "supervisor"
    if args.tier:
        tiers = list(record.get("tiers") or [])
        tier = next((row for row in tiers if row.get("role") == args.tier), None)
        if tier is None:
            roles = ", ".join(str(row.get("role")) for row in tiers) or "none"
            print(f"hive logs: tier {args.tier!r} is not in this serve (roles: {roles})",
                  file=sys.stderr)
            return 2
        if tier.get("base_url") or not tier.get("port"):
            print(f"hive logs: tier {args.tier!r} is an endpoint; its log belongs "
                  "to that endpoint", file=sys.stderr)
            return 1
        path = Path(str(record.get("log_dir") or "logs")) / (
            f"llama_server_{int(tier['port'])}.log")
        source = f"tier {args.tier}"
    else:
        raw = record.get("log_path")
        if not raw:
            print("hive logs: this serve is in the foreground; its output is on "
                  "that terminal (use --tier for the engine log)", file=sys.stderr)
            return 1
        path = Path(str(raw))

    try:
        lines = serve_state.tail_lines(path, args.lines)
    except OSError as exc:
        print(f"hive logs: cannot read {path}: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        _json({"name": record["name"], "source": source, "path": str(path),
               "lines": lines})
        return 0
    for line in lines:
        print(line)
    if not args.follow:
        return 0

    print(termui.dim(f"-- following {path} (Ctrl-C to stop) --",
                     stream=sys.stderr), file=sys.stderr)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, os.SEEK_END)
            while True:
                line = handle.readline()
                if line:
                    print(line, end="", flush=True)
                    continue
                if not serve_state.pid_alive(int(record["pid"])):
                    print(termui.dim("-- the serve has stopped --",
                                     stream=sys.stderr), file=sys.stderr)
                    break
                time.sleep(0.3)
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        print(f"hive logs: cannot follow {path}: {exc}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def cmd_models_list(args: argparse.Namespace) -> int:
    manager = _models_manager(args)
    if args.as_json:
        rows = manager.list_local(use_cache=False)
    else:
        with termui.spinner("reading the model library"):
            rows = manager.list_local(use_cache=False)
    if args.as_json:
        _json(rows)
        return 0
    termui.header("models", str(manager.models_dir))
    if not rows:
        print(f"  {termui.dim('no GGUF models here yet')}")
        print(f"  {termui.info('hive models download <repo>[:<file>]')}")
        return 0
    _table(
        ["NAME", "SIZE", "QUANT", "CTX", "FILE"],
        [[row.get("name"), f"{float(row.get('size_gb') or 0):.2f} GiB",
          row.get("quantization") or "-", row.get("context_length") or "-",
          row.get("file")] for row in rows])
    print(f"\n  {termui.dim(f'{len(rows)} model(s)')}")
    return 0


def cmd_models_search(args: argparse.Namespace) -> int:
    manager = _models_manager(args)
    try:
        rows = manager.hub_search(args.query, limit=args.limit)
    except Exception as exc:  # noqa: BLE001 - network/API message is the error
        print(f"hive models search: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(rows)
        return 0
    termui.header("models · search", args.query or "popular")
    if not rows:
        print(f"  {termui.dim('no GGUF repos matched')}")
        return 0
    _table(["REPO", "DOWNLOADS", "LIKES", "MODIFIED", "LICENSE"],
           [[row.get("repo"), row.get("downloads"), row.get("likes"),
             row.get("last_modified"), row.get("license") or "-"] for row in rows])
    return 0


def cmd_models_files(args: argparse.Namespace) -> int:
    manager = _models_manager(args)
    try:
        rows = manager.hub_files(args.repo)
    except Exception as exc:  # noqa: BLE001 - network/API message is the error
        print(f"hive models files: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(rows)
        return 0
    termui.header("models · files", args.repo)
    if not rows:
        print(f"  {termui.dim(f'{args.repo} has no .gguf files')}")
        return 0
    _table(["FILE", "SIZE"], [[row.get("file"),
                               f"{float(row.get('size_gb') or 0):.2f} GiB"]
                              for row in rows])
    return 0


def cmd_models_dir(args: argparse.Namespace) -> int:
    manager = _models_manager(args)
    target = args.set or args.path
    if target:
        result = manager.set_models_dir(target)
        if args.as_json:
            _json(result)
        else:
            print(result.get("models_dir") or manager.models_dir)
        return 0
    count = len(list(manager.models_dir.rglob("*.gguf")))
    payload = {"models_dir": str(manager.models_dir), "exists": manager.models_dir.is_dir(),
               "gguf_count": count}
    if args.as_json:
        _json(payload)
    else:
        print(payload["models_dir"])
    return 0


def _split_repo_file(repo: str, file: str | None) -> tuple[str, str | None]:
    if file:
        return repo, file
    if ":" in repo:
        name, _, suffix = repo.partition(":")
        return name, suffix or None
    return repo, None


def cmd_models_download(args: argparse.Namespace) -> int:
    manager = _models_manager(args)
    repo, file = _split_repo_file(args.repo, args.file)
    if not file:
        try:
            files = manager.hub_files(repo)
        except Exception as exc:  # noqa: BLE001 - network/API message is the error
            print(f"hive models download: cannot list {repo}: {exc}", file=sys.stderr)
            return 1
        if not files:
            print(f"hive models download: {repo} has no .gguf files", file=sys.stderr)
            return 1
        if len(files) > 1:
            print(f"hive models download: {repo} has several GGUFs; "
                  "choose one with --file:", file=sys.stderr)
            for row in files:
                print(f"  {row.get('file')}  ({row.get('size_gb')} GiB)", file=sys.stderr)
            return 2
        file = str(files[0].get("file"))
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("hive models download: huggingface_hub is not installed", file=sys.stderr)
        return 1
    if not args.as_json:
        print(termui.info(f"downloading {repo}:{file}"), file=sys.stderr, flush=True)
    try:
        path = Path(hf_hub_download(
            repo_id=repo, filename=file, revision=args.revision,
            local_dir=str(manager.models_dir)))
    except Exception as exc:  # noqa: BLE001 - gated/network/disk all become one line
        print(f"hive models download: {repo}:{file}: {exc}", file=sys.stderr)
        return 1
    size = path.stat().st_size / (1024 ** 3) if path.is_file() else 0.0
    payload = {"repo": repo, "file": file, "path": str(path),
               "size_gb": round(size, 2)}
    if args.as_json:
        _json(payload)
    else:
        print(termui.ok(f"downloaded {repo}:{file}"))
        termui.kv([("path", path), ("size", f"{size:.2f} GiB")])
    return 0


def cmd_models_import(args: argparse.Namespace) -> int:
    manager = _models_manager(args)
    source = Path(args.path).expanduser()
    if source.is_dir():
        result = manager.import_local_path(str(source))
        if args.as_json:
            _json(result)
        else:
            print(ui.ok(f"linked {source.name or source}"))
            for item in result.get("imported", [])[:20]:
                print(f"  {ui.dim(item)}")
        return 0
    if not source.is_file():
        print(f"hive models import: no such file or folder: {source}", file=sys.stderr)
        return 1
    if source.suffix.lower() != ".gguf":
        print(f"hive models import: not a .gguf file: {source.name}", file=sys.stderr)
        return 2
    if source.resolve().parent == manager.models_dir.resolve():
        payload = {"imported": source.name, "path": str(source), "linked": False}
        _json(payload) if args.as_json else print(source)
        return 0
    dest = manager.models_dir / source.name
    if dest.exists():
        print(f"hive models import: {dest.name} is already in {manager.models_dir}",
              file=sys.stderr)
        return 1
    how = "copy"
    if not args.copy:
        try:
            os.link(source, dest)
            how = "hardlink"
        except OSError:
            try:
                dest.symlink_to(source)
                how = "symlink"
            except OSError:
                with termui.spinner(f"copying {source.name}"):
                    shutil.copy2(source, dest)
                how = "copy"
    else:
        with termui.spinner(f"copying {source.name}"):
            shutil.copy2(source, dest)
    payload = {"imported": dest.name, "path": str(dest), "linked": how != "copy",
               "how": how}
    if args.as_json:
        _json(payload)
    else:
        print(termui.ok(f"imported {source.name} · {how}"))
        print(f"  {termui.dim(dest)}")
    return 0


def cmd_models_remove(args: argparse.Namespace) -> int:
    if not args.yes:
        print("hive models remove: refusing to delete without --yes", file=sys.stderr)
        return 2
    manager = _models_manager(args)
    try:
        result = manager.delete_local(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"hive models remove: {exc}", file=sys.stderr)
        return 1
    manager.invalidate_cache()
    if args.as_json:
        _json(result)
    else:
        print(termui.ok(f"removed {result['deleted']}"))
    return 0


# ---------------------------------------------------------------------------
# stacks
# ---------------------------------------------------------------------------

def cmd_stacks_list(args: argparse.Namespace) -> int:
    directory = _stacks_directory(args.stacks_dir)
    rows = schema.list_stacks_in(directory)
    if args.as_json:
        _json(rows)
        return 0
    termui.header("stacks", str(directory))
    if not rows:
        print(f"  {termui.dim('no saved stacks here yet')}")
        return 0
    _table(["NAME", "TIERS", "ROLES"],
           [[row.get("name"), row.get("tiers"),
             ",".join(row.get("roles") or [])] for row in rows])
    return 0


def cmd_stacks_show(args: argparse.Namespace) -> int:
    stack = _load_stack_any(args.stack, args.stacks_dir)
    print(json.dumps(stack.to_dict(), indent=2))
    return 0


def _tier_validation(stack: schema.Stack, args: argparse.Namespace) -> tuple[list[str], list[str], list[dict]]:
    errors: list[str] = []
    warnings: list[str] = []
    details: list[dict] = []
    manager = _models_manager(args)
    for tier in stack.tiers:
        row = {"role": tier.role, "status": "ok", "detail": ""}
        launch_ok = True
        if tier.launch:
            repo = Path(str(tier.launch.get("repo") or "")).expanduser()
            if str(repo) and not repo.exists():
                launch_ok = False
                errors.append(f"tier {tier.role!r}: launch repo {repo} does not exist")
                row.update(status="fail", detail="launch repo missing")
        if tier.endpoint:
            if launch_ok:
                row["detail"] = f"endpoint {tier.endpoint}"
            details.append(row)
            continue
        backend_ok = True
        try:
            backend = backend_manifest.resolve(tier.backend)
            if not backend.get("installed"):
                backend_ok = False
                message = (f"tier {tier.role!r}: backend {tier.backend!r} is not "
                           f"installed ({backend.get('binary')})")
                errors.append(message)
                row.update(status="fail", detail="backend missing")
        except KeyError as exc:
            backend_ok = False
            errors.append(f"tier {tier.role!r}: {exc}")
            row.update(status="fail", detail=str(exc))
        if backend_ok:
            model = Path(tier.file).expanduser()
            found = model.is_file() or manager.resolve_model(tier.file) is not None
            if found:
                row["detail"] = f"{tier.backend} · model ok"
            elif tier.repo and tier.repo != "local":
                row["detail"] = f"{tier.backend} · remote (downloaded on serve)"
                warnings.append(
                    f"tier {tier.role!r}: model is not local; it will be fetched "
                    f"from {tier.repo} on first serve")
                row["status"] = "warn"
            else:
                errors.append(f"tier {tier.role!r}: model {tier.file!r} is not in "
                              f"{manager.models_dir}")
                row.update(status="fail", detail="model missing")
        details.append(row)
    return errors, warnings, details


def cmd_stacks_validate(args: argparse.Namespace) -> int:
    stack = _load_stack_any(args.stack, args.stacks_dir)
    problems = schema.validate_shape(stack)
    errors: list[str] = list(problems)
    warnings: list[str] = []
    details: list[dict] = []
    if not problems:
        errors, warnings, details = _tier_validation(stack, args)
    ok = not errors and not (args.strict and warnings)
    if args.as_json:
        _json({"ok": ok, "stack": stack.name, "errors": errors,
               "warnings": warnings, "tiers": details})
        return 0 if ok else 1
    termui.header("stacks · validate", stack.name)
    print(f"  {termui.dim(f'{len(stack.tiers)} tier(s)')}")
    for problem in problems:
        print(f"  {termui.fail(problem)}")
    for row in details:
        mark = {"ok": termui.ok, "warn": termui.warn, "fail": termui.fail}[row["status"]]
        print(f"  {mark(str(row['role']).ljust(8))} {termui.dim(row['detail'])}")
    for error in errors:
        if error not in problems:
            print(f"  {termui.fail(error)}")
    for warning in warnings:
        print(f"  {termui.warn(warning)}")
    print(f"  {termui.ok('ok') if ok else termui.fail('validation failed')}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# cascade
# ---------------------------------------------------------------------------

def cmd_cascade_runs(args: argparse.Namespace) -> int:
    runs: list[dict] = []
    if CASCADE_RUNS.is_dir():
        for path in sorted(CASCADE_RUNS.iterdir(), key=lambda p: p.name, reverse=True):
            if not path.is_dir():
                continue
            report = path / "report.json"
            if not report.is_file():
                continue
            summary: dict = {}
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
                summary = dict(data.get("summary") or {})
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            runs.append({
                "run": path.name,
                "path": str(report),
                "modified": time.strftime("%Y-%m-%d %H:%M",
                                          time.localtime(report.stat().st_mtime)),
                "summary": summary,
            })
    if args.as_json:
        _json(runs)
        return 0
    termui.header("cascade", str(CASCADE_RUNS))
    if not runs:
        print(f"  {termui.dim('no runs yet')}")
        return 0
    rows = []
    for row in runs:
        summary = row["summary"]
        tasks = summary.get("tasks", "-")
        acc = summary.get("cascade_accuracy")
        esc = summary.get("escalation_rate")
        mean = summary.get("cascade_mean_ms", summary.get("scion_mean_ms"))
        rows.append([
            row["run"], tasks,
            f"{float(acc):.1%}" if isinstance(acc, (int, float)) else "-",
            f"{float(esc):.1%}" if isinstance(esc, (int, float)) else "-",
            f"{float(mean):.0f}" if isinstance(mean, (int, float)) else "-",
            row["modified"],
        ])
    _table(["RUN", "TASKS", "CASCADE", "ESCALATE", "MEAN MS", "MODIFIED"], rows)
    return 0


def cmd_cascade_report(args: argparse.Namespace) -> int:
    name = str(args.run)
    if Path(name).name != name or name in ("", ".", ".."):
        print(f"hive cascade report: invalid run name {name!r}", file=sys.stderr)
        return 2
    path = CASCADE_RUNS / name / "report.json"
    if not path.is_file():
        print(f"hive cascade report: no report at {path}", file=sys.stderr)
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"hive cascade report: cannot read {path}: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(data)
    else:
        termui.header("cascade · report", name)
        summary = dict(data.get("summary") or {})
        if not summary:
            print(f"  {termui.dim('no summary in this report')}")
            return 0
        termui.kv([(key, _metric(key, value)) for key, value in summary.items()])
    return 0


def _metric(key: str, value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if "rate" in key or "accuracy" in key:
            return f"{float(value):.1%}"
        if key.endswith("_ms"):
            return f"{float(value):.0f} ms"
        if isinstance(value, float):
            return f"{float(value):.3f}"
    return str(value)


def cmd_cascade_run(args: argparse.Namespace) -> int:
    if not CASCADE_RUNNER.is_file():
        print(f"hive cascade run: runner not found at {CASCADE_RUNNER}",
              file=sys.stderr)
        return 1
    forwarded = list(args.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    command = [sys.executable, str(CASCADE_RUNNER), *forwarded]
    try:
        return int(subprocess.call(command, cwd=str(REPO_ROOT)))
    except OSError as exc:
        print(f"hive cascade run: cannot start the runner: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# backends / doctor / config / version
# ---------------------------------------------------------------------------

def cmd_backends(args: argparse.Namespace) -> int:
    rows = backend_manifest.rows()
    if args.as_json:
        _json(rows)
        return 0
    termui.header("backends")
    _table(["NAME", "STATUS", "ACCELERATORS", "BINARY", "LABEL"],
           [[row["name"], "installed" if row["installed"] else "-",
             ",".join(row.get("accelerators") or []), row.get("binary"),
             row.get("label")] for row in rows])
    print(f"\n  {termui.dim('hive backends fetch <name> installs a portable release')}")
    return 0


def cmd_backends_fetch(args: argparse.Namespace) -> int:
    name = args.name or backend_manifest.default_fetch_backend()
    last = {"pct": -1}

    def progress(done: int, total: int) -> None:
        if args.as_json:
            return
        if total > 0:
            pct = int(done * 100 / total)
            if pct == last["pct"]:
                return
            last["pct"] = pct
            sys.stderr.write(
                f"\r  {pct:3d}%  {done / 1048576:.1f} / {total / 1048576:.1f} MiB")
            if done >= total:
                sys.stderr.write("\n")
        else:
            sys.stderr.write(f"\r  {done / 1048576:.1f} MiB")
        sys.stderr.flush()

    if not args.as_json:
        print(termui.info(f"resolving {name} for this host"), file=sys.stderr)
    try:
        result = backend_manifest.fetch_backend(
            name, tag=args.tag, force=args.force, progress=progress)
    except (KeyError, RuntimeError, OSError) as exc:
        print(f"hive backends fetch: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(result)
        return 0
    if result.get("already"):
        print(termui.dim(
            f"{name} is already installed at {result['path']} "
            "(use --force to replace it)"))
        return 0
    print(termui.ok(f"installed {name} · {result.get('tag') or 'release'}"))
    termui.kv([("binary", result["binary"]),
               ("asset", result.get("asset") or "-")])
    print(f"  {termui.dim('hive backends')}  {termui.dim('to see every build')}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """One-command onboarding: a working engine and the writable paths."""
    installed = backend_manifest.installed()
    fetched = None
    if not installed:
        if args.no_fetch:
            print("hive setup: no engine backend installed; run "
                  "`hive backends fetch vulkan`", file=sys.stderr)
            return 1
        name = backend_manifest.default_fetch_backend()
        last = {"pct": -1}

        def progress(done: int, total: int) -> None:
            if args.as_json:
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

        if not args.as_json:
            print(termui.info(f"no engine installed; fetching {name}"), file=sys.stderr)
        try:
            fetched = backend_manifest.fetch_backend(name, progress=progress)
        except (KeyError, RuntimeError, OSError) as exc:
            print(f"hive setup: cannot fetch {name}: {exc}", file=sys.stderr)
            return 1
        installed = backend_manifest.installed()

    manager = _models_manager(args)
    stacks = _stacks_directory(args.stacks_dir)
    stacks.mkdir(parents=True, exist_ok=True)
    state = serve_state.serve_dir()
    payload = {
        "ok": True,
        "fetched": fetched,
        "backends": installed,
        "models_dir": str(manager.models_dir),
        "stacks_dir": str(stacks),
        "state_dir": str(state),
    }
    if args.as_json:
        _json(payload)
        return 0
    termui.header("setup")
    print(f"  {termui.ok('engine: ' + (', '.join(installed) or 'none'))}")
    termui.kv([("models", manager.models_dir), ("stacks", stacks),
               ("state", state)])
    print()
    print(f"  {termui.dim('next')}")
    print(f"    {termui.accent('›')} hive models download <repo>[:<file>]")
    print(f"    {termui.accent('›')} hive serve --model <repo:file> --cards 1")
    print(f"    {termui.accent('›')} hive profile scan --consent hardware")
    print(f"    {termui.accent('›')} hive doctor")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[dict] = []

    def add(status: str, name: str, detail: str) -> None:
        checks.append({"status": status, "name": name, "detail": detail})

    version = sys.version_info
    add("ok" if version >= (3, 10) else "fail", "python",
        f"{platform.python_version()} ({sys.executable})")
    add("ok", "hive", _version())
    add("ok", "platform", f"{platform.system()} {platform.machine()}")

    try:
        manager = _models_manager(args)
        models_dir = manager.models_dir
        writable = os.access(models_dir, os.W_OK)
        count = len(list(models_dir.rglob("*.gguf")))
        free = shutil.disk_usage(models_dir).free / (1024 ** 3)
        add("ok" if writable else "fail", "models",
            f"{models_dir} · {count} GGUF · {free:.1f} GiB free")
    except OSError as exc:
        add("fail", "models", f"cannot use the models directory: {exc}")

    stacks_dir = _stacks_directory(args.stacks_dir)
    stack_count = len(list(stacks_dir.glob("*.json"))) if stacks_dir.is_dir() else 0
    add("ok" if stacks_dir.is_dir() else "warn", "stacks",
        f"{stacks_dir} · {stack_count} stack(s)")

    try:
        state = serve_state.serve_dir()
        add("ok" if os.access(state, os.W_OK) else "fail", "state", str(state))
    except OSError as exc:
        add("fail", "state", f"cannot use the state directory: {exc}")

    try:
        rows = backend_manifest.rows()
        installed = [row["name"] for row in rows if row["installed"]]
        if installed:
            add("ok", "backends", ", ".join(installed))
        else:
            add("fail", "backends",
                "none installed; run `hive backends fetch vulkan` "
                "(or see `hive backends`)")
    except (OSError, ValueError) as exc:
        add("fail", "backends", f"backend manifest: {exc}")

    try:
        devices = linux_amd_devices()
    except OSError as exc:
        devices = []
        add("warn", "cards", f"cannot read amdgpu sysfs: {exc}")
    else:
        if devices:
            def _card(d: dict) -> str:
                total = d.get("memory_gb")
                used = d.get("used_gb")
                free = d.get("free_gb")
                if used is not None and free is not None:
                    state = f"{used} used / {free} free GiB"
                else:
                    state = f"{total} GiB"
                return (f"card {d.get('index')}: {state}"
                        f"{' (display)' if d.get('display') else ''}")

            detail = "; ".join(_card(d) for d in devices)
            add("ok", "cards", detail)
        elif platform.system() == "Linux":
            add("warn", "cards", "no amdgpu cards found via sysfs")
        else:
            add("warn", "cards", "GPU probing is Linux-only")

    try:
        import psutil

        vm = psutil.virtual_memory()
        detail = (f"{vm.available / (1024 ** 3):.1f}/"
                  f"{vm.total / (1024 ** 3):.1f} GiB free")
        gtt_used = [d.get("gtt_used_gb") for d in devices
                    if isinstance(d.get("gtt_used_gb"), (int, float))]
        gtt_total = [d.get("gtt_total_gb") for d in devices
                     if isinstance(d.get("gtt_total_gb"), (int, float))]
        if gtt_used and gtt_total:
            detail += f" · GTT {sum(gtt_used):.1f}/{sum(gtt_total):.1f} GiB"
        add("ok", "memory", detail)
    except Exception as exc:  # noqa: BLE001 - psutil is best-effort
        add("warn", "memory", f"cannot read host memory: {exc}")

    rocm = shutil.which("rocminfo") or shutil.which("rocm-smi")
    if rocm:
        add("ok", "rocm", rocm)
    elif devices:
        add("warn", "rocm", "rocminfo/rocm-smi not in PATH")
    else:
        add("warn", "rocm", "not found (only needed for a ROCm backend)")

    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or any(path.is_file() for path in (
                 Path.home() / ".cache" / "huggingface" / "token",
                 Path.home() / ".huggingface" / "token")))
    add("ok" if token else "warn", "hf-token",
        "set" if token else "not set (only needed for gated/private repos)")

    for port in (1234, 8150):
        add("ok" if _port_free("127.0.0.1", port) else "warn", f"port {port}",
            "free" if _port_free("127.0.0.1", port) else "in use (a serve may be running)")

    failures = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    ok = not failures and not (args.strict and warnings)
    if args.as_json:
        _json({"ok": ok, "checks": checks,
               "summary": {"ok": len(checks) - len(failures) - len(warnings),
                           "warn": len(warnings), "fail": len(failures)}})
        return 0 if ok else 1
    termui.header("doctor", f"{platform.system()} {platform.machine()}")
    width = max((len(check["name"]) for check in checks), default=0)
    for check in checks:
        mark = {"ok": termui.ok, "warn": termui.warn, "fail": termui.fail}[check["status"]]
        print(f"  {mark(check['name'].ljust(width))}  {termui.dim(check['detail'])}")
    print()
    if failures:
        print(f"  {termui.fail(f'{len(failures)} failure(s) · fix them before serving')}")
    elif args.strict and warnings:
        print(f"  {termui.warn(f'{len(warnings)} warning(s) · --strict')}")
    elif warnings:
        print(f"  {termui.warn(f'ready · {len(warnings)} warning(s)')}")
    else:
        print(f"  {termui.ok('ready')}")
    return 0 if ok else 1


def cmd_config(args: argparse.Namespace) -> int:
    manager = _models_manager(args)
    installed = []
    try:
        installed = [row["name"] for row in backend_manifest.rows() if row["installed"]]
    except (OSError, ValueError):
        pass
    default_backend = backend_manifest.for_accelerator("amd") or backend_manifest.for_accelerator("cpu")
    payload = {
        "hive": _version(),
        "python": sys.executable,
        "repo": str(Path(__file__).resolve().parents[1]),
        "models_dir": str(manager.models_dir),
        "stacks_dir": str(_stacks_directory(args.stacks_dir)),
        "state_dir": str(serve_state.state_dir()),
        "serve_dir": str(serve_state.serve_dir()),
        "log_dir": str(manager.log_dir),
        "default_backend": default_backend,
        "installed_backends": installed,
    }
    if args.as_json:
        _json(payload)
    else:
        termui.header("config")
        termui.kv([(key, ", ".join(str(item) for item in value) or "-"
                if isinstance(value, list) else value)
               for key, value in payload.items()])
    return 0


def _state_paths(args: argparse.Namespace):
    from harness.state_store import StatePaths

    raw = getattr(args, "state_dir", None)
    return StatePaths(Path(raw).expanduser()) if raw else StatePaths.default()


def _print_profile(profile: dict) -> None:
    termui.header("hardware profile", str(profile.get("profile_id") or "-"))
    cpu = profile.get("cpu") or {}
    ram = profile.get("ram") or {}
    termui.kv([
        ("scanned", profile.get("scanned_at") or "-"),
        ("cpu", f"{cpu.get('model') or 'unknown'} · "
                f"{cpu.get('physical') or '?'} physical / "
                f"{cpu.get('logical') or '?'} logical"),
        ("ram", f"{ram.get('available_gb') or '?'} available / "
                f"{ram.get('total_gb') or '?'} GiB"),
        ("consent", ", ".join(k for k, v in (profile.get("consent") or {}).items() if v)
         or "none"),
    ])
    for card in profile.get("gpus") or []:
        state = (f"{card.get('used_gb')} used / {card.get('free_gb')} free GiB"
                 if card.get("free_gb") is not None else
                 f"{card.get('memory_gb')} GiB")
        print(f"  {termui.accent('›')} card {card.get('index')}: {state}"
              f"{' (display)' if card.get('display') else ''}")


def cmd_profile_scan(args: argparse.Namespace) -> int:
    from harness.controller.profile import Consent, ConsentRequired, HardwareProfiler

    try:
        consent = Consent.parse(args.consent)
    except ValueError as exc:
        print(f"hive profile scan: {exc}", file=sys.stderr)
        return 2
    if not consent.hardware:
        print("hive profile scan: explicit --consent hardware is required",
              file=sys.stderr)
        return 2
    try:
        profile = HardwareProfiler(_state_paths(args)).scan(
            consent, profile_id=args.profile_id)
    except (ConsentRequired, OSError, ValueError) as exc:
        print(f"hive profile scan: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(profile)
    else:
        _print_profile(profile)
        print(f"\n  {termui.dim('next')}")
        print(f"    {termui.accent('›')} hive controller propose --model <path.gguf>")
    return 0


def cmd_profile_show(args: argparse.Namespace) -> int:
    from harness.controller.profile import HardwareProfiler

    profiler = HardwareProfiler(_state_paths(args))
    profile = profiler.get(args.profile_id) if args.profile_id else profiler.latest()
    if profile is None:
        print("hive profile show: no hardware profile saved; run "
              "`hive profile scan --consent hardware`", file=sys.stderr)
        return 1
    if args.as_json:
        _json(profile)
    else:
        _print_profile(profile)
    return 0


def cmd_profile_list(args: argparse.Namespace) -> int:
    from harness.controller.profile import HardwareProfiler

    rows = HardwareProfiler(_state_paths(args)).list()
    if args.as_json:
        _json(rows)
    elif not rows:
        print("  no hardware profiles yet")
    else:
        _table(["profile", "scanned", "cpu", "ram"],
               [[row.get("profile_id"), row.get("scanned_at"),
                 (row.get("cpu") or {}).get("model"),
                 (row.get("ram") or {}).get("available_gb")]
                for row in rows])
    return 0


def cmd_profile_delete(args: argparse.Namespace) -> int:
    from harness.controller.profile import HardwareProfiler

    if not args.yes:
        print("hive profile delete: pass --yes to confirm", file=sys.stderr)
        return 2
    profiler = HardwareProfiler(_state_paths(args))
    profile = profiler.get(args.profile_id) if args.profile_id else profiler.latest()
    if profile is None:
        print("hive profile delete: no such profile", file=sys.stderr)
        return 1
    deleted = profiler.delete(profile["profile_id"])
    if args.as_json:
        _json({"deleted": deleted, "profile_id": profile["profile_id"]})
    else:
        print(f"  deleted {profile['profile_id']}")
    return 0


def _print_proposal(proposal: dict) -> None:
    termui.header("controller proposal", str(proposal.get("id") or "-"))
    termui.kv([
        ("profile", proposal.get("profile_id") or "-"),
        ("reason", proposal.get("reason") or "-"),
        ("confidence", proposal.get("confidence")),
    ])
    print(f"\n  {termui.accent('changes')}")
    for key, value in (proposal.get("changes") or {}).items():
        print(f"    {key} = {value}")
    print(f"\n  {termui.accent('evidence')}")
    for line in proposal.get("evidence") or []:
        print(f"    · {line}")
    warnings = proposal.get("warnings") or []
    if warnings:
        print(f"\n  {termui.warn('warnings')}")
        for line in warnings:
            print(f"    · {line}")


def cmd_controller_status(args: argparse.Namespace) -> int:
    from harness.controller.service import ControllerService

    status = ControllerService(_state_paths(args)).status()
    if args.as_json:
        _json(status)
    else:
        profile = status.get("profile") or {}
        latest = status.get("latest_proposal") or {}
        controller = status.get("controller") or {}
        encoder = status.get("encoder") or {}
        trace = status.get("trace_summary") or {}
        if not encoder.get("loaded"):
            encoder_text = "not loaded"
        else:
            encoder_text = (f"loaded · idle unload "
                            f"{'on' if encoder.get('idle_unload') else 'off'}"
                            f" · {encoder.get('idle_remaining_s')}s left")
        termui.header("controller")
        termui.kv([
            ("profile", profile.get("profile_id") or "none"),
            ("last proposal", latest.get("id") or "none"),
            ("applied", (controller.get("last_apply") or {}).get("proposal_id") or "none"),
            ("encoder", encoder_text),
            ("trace labels", f"{trace.get('labelled', 0)} / "
                             f"{trace.get('target_labelled', 200)}"),
            ("E1 ready", "yes" if trace.get("ready") else "no — "
             + "; ".join(trace.get("missing") or ["collect labelled traces"])),
            ("state", status.get("state_dir")),
        ])
    return 0


def cmd_controller_propose(args: argparse.Namespace) -> int:
    from harness.controller.service import ControllerError, ControllerService

    service = ControllerService(_state_paths(args))
    try:
        proposal = service.propose(
            profile_id=args.profile,
            model_path=args.model,
            model_gib=args.model_gib,
            native_ctx=args.native_ctx,
            desired_ctx=args.ctx,
            kv_type=args.kv_type,
            request=args.request,
            embed_request=args.embed_request,
        )
    except (ControllerError, OSError, ValueError) as exc:
        print(f"hive controller propose: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(proposal)
    else:
        _print_proposal(proposal)
        print(f"\n  {termui.dim('apply with')} "
              f"{termui.accent('hive controller apply ' + proposal['id'] + ' --yes')}")
    return 0


def cmd_controller_apply(args: argparse.Namespace) -> int:
    from harness.controller.service import (
        ConfirmationRequired,
        ControllerError,
        ControllerService,
    )

    if not args.yes:
        print("hive controller apply: pass --yes to confirm a settings change",
              file=sys.stderr)
        return 2
    service = ControllerService(_state_paths(args))
    proposal_id = args.proposal
    if not proposal_id:
        latest = service.status().get("latest_proposal") or {}
        proposal_id = latest.get("id")
    if not proposal_id:
        print("hive controller apply: no proposal to apply", file=sys.stderr)
        return 1
    try:
        result = service.apply(proposal_id, confirmed=True, profile_id=args.profile)
    except (ConfirmationRequired, ControllerError) as exc:
        print(f"hive controller apply: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(result)
    else:
        termui.header("controller apply", proposal_id)
        termui.kv([("changes", ", ".join(result["proposal"]["changes"]))])
        print(f"  {termui.ok('applied')} · revert with "
              f"{termui.accent('hive controller revert --yes')}")
    return 0


def cmd_controller_outcome(args: argparse.Namespace) -> int:
    from harness.controller.service import ControllerError, ControllerService

    metrics: dict[str, Any] = {}
    for item in args.metric or []:
        key, sep, raw = str(item).partition("=")
        if not sep or not key.strip():
            print("hive controller outcome: --metric needs key=value",
                  file=sys.stderr)
            return 2
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        metrics[key.strip()] = value
    try:
        ControllerService(_state_paths(args)).record_outcome(
            args.proposal, args.outcome, metrics=metrics)
    except ControllerError as exc:
        print(f"hive controller outcome: {exc}", file=sys.stderr)
        return 1
    payload = {"ok": True, "proposal_id": args.proposal,
               "outcome": args.outcome, "metrics": metrics}
    if args.as_json:
        _json(payload)
    else:
        print(f"  recorded {args.outcome} for {args.proposal}")
    return 0


def cmd_controller_revert(args: argparse.Namespace) -> int:
    from harness.controller.service import ConfirmationRequired, ControllerError, ControllerService

    if not args.yes:
        print("hive controller revert: pass --yes to confirm rollback", file=sys.stderr)
        return 2
    try:
        result = ControllerService(_state_paths(args)).revert(confirmed=True)
    except (ConfirmationRequired, ControllerError) as exc:
        print(f"hive controller revert: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(result)
    else:
        print(f"  {termui.ok('reverted')} the last controller change")
    return 0


def cmd_encoder_status(args: argparse.Namespace) -> int:
    from harness.controller.encoder import (
        EncoderUnavailable,
        SentenceTransformerEncoder,
        encoder_from_settings,
    )
    from harness.studio_settings import load_settings

    try:
        if args.model:
            encoder = SentenceTransformerEncoder(
                args.model, local_files_only=not args.allow_download)
        else:
            encoder = encoder_from_settings(
                load_settings(_state_paths(args).root),
                local_files_only=not args.allow_download)
        status = encoder.status()
    except (EncoderUnavailable, ValueError) as exc:
        print(f"hive encoder status: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(status)
    else:
        termui.header("encoder")
        termui.kv([
            ("model", status.get("model")),
            ("backend", status.get("kind")),
            ("device", status.get("device")),
            ("available", status.get("available")),
            ("dimensions", status.get("dimensions") or "not loaded"),
        ])
    return 0


def cmd_encoder_embed(args: argparse.Namespace) -> int:
    from harness.controller.encoder import (
        EncoderError,
        SentenceTransformerEncoder,
        encoder_from_settings,
    )
    from harness.studio_settings import load_settings

    text = args.text
    if text is None:
        text = sys.stdin.read()
    if not text.strip():
        print("hive encoder embed: text is empty", file=sys.stderr)
        return 2
    try:
        if args.model:
            encoder = SentenceTransformerEncoder(
                args.model, local_files_only=not args.allow_download)
        else:
            encoder = encoder_from_settings(
                load_settings(_state_paths(args).root),
                local_files_only=not args.allow_download)
        vector = encoder.encode_one(text)
    except (EncoderError, OSError, ValueError) as exc:
        print(f"hive encoder embed: {exc}", file=sys.stderr)
        return 1
    payload = {
        "model": encoder.model,
        "dimensions": len(vector),
        "vector": vector,
    }
    if args.as_json:
        _json(payload)
    else:
        print(f"  {termui.accent('model')} {payload['model']}")
        print(f"  {termui.accent('dimensions')} {payload['dimensions']}")
        preview = ", ".join(f"{value:.4f}" for value in vector[:8])
        print(f"  {termui.accent('vector')} [{preview}{' …' if len(vector) > 8 else ''}]")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Switch between the terminal dashboard and the web Studio."""
    mode = args.mode or "status"
    if mode == "terminal":
        return cmd_tui(args)
    if mode == "web":
        argv = [sys.executable, "-m", "harness"]
        if args.host:
            argv += ["--host", args.host]
        if args.port:
            argv += ["--port", str(args.port)]
        if args.detach:
            argv.append("--detach")
        if args.no_open:
            argv.append("--no-open")
        if args.mock:
            argv.append("--mock")
        state_dir = getattr(args, "state_dir", None) or str(serve_state.state_dir())
        argv += ["--state-dir", state_dir]
        env = dict(os.environ)
        env["HIVE_TOOL_URL"] = f"http://127.0.0.1:{args.port or 8765}"
        return subprocess.call(argv, cwd=str(REPO_ROOT), env=env)
    # status: show the web endpoint and how to switch without starting it
    url = f"http://{args.host or '127.0.0.1'}:{args.port or 8765}"
    alive = False
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=0.8) as response:
            alive = response.status == 200
    except Exception:  # noqa: BLE001 - "not running" is the useful answer
        alive = False
    payload = {
        "web": {"running": alive, "url": url},
        "terminal": {"command": "hive ui terminal"},
        "switch": {"web": "hive ui web", "terminal": "hive ui terminal"},
    }
    if args.as_json:
        _json(payload)
    else:
        termui.header("interfaces")
        termui.kv([
            ("web", f"{url} · {'running' if alive else 'not running'}"),
            ("terminal", "hive ui terminal"),
            ("start web", "hive ui web"),
        ])
        print(f"  {termui.dim('hive ui web --detach starts the Studio and opens the browser')}")
    return 0


def _tool_api(args: argparse.Namespace, path: str, *,
              method: str = "GET", body: dict | None = None) -> dict:
    from harness import studio_api

    try:
        return studio_api.call(
            path, method=method, body=body,
            url=getattr(args, "url", None))
    except studio_api.StudioAPIError as exc:
        raise RuntimeError(str(exc)) from exc


def cmd_tools_pending(args: argparse.Namespace) -> int:
    try:
        payload = _tool_api(args, "/v1/tools/pending"
                            + (f"?session_id={args.session}" if args.session else ""))
    except RuntimeError as exc:
        print(f"hive tools pending: {exc}", file=sys.stderr)
        return 1
    rows = payload.get("requests") or []
    if args.as_json:
        _json(rows)
    elif not rows:
        print("  no pending tool approvals")
    else:
        _table(["request", "session", "tool", "action", "why"],
               [[row.get("request_id"), row.get("session_id"),
                 row.get("tool_identity"), row.get("action"),
                 row.get("reason")] for row in rows])
    return 0


def cmd_tools_decide(args: argparse.Namespace) -> int:
    try:
        payload = _tool_api(args, "/v1/tools/decide", method="POST", body={
            "session_id": args.session,
            "request_id": args.request,
            "choice": args.choice,
            "scope": args.scope,
        })
    except RuntimeError as exc:
        print(f"hive tools decide: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(payload)
    else:
        termui.header("tool decision", args.request)
        termui.kv([("result", payload.get("status") or "?"),
                   ("reason", payload.get("reason") or "-")])
    return 0


def cmd_tools_status(args: argparse.Namespace) -> int:
    from harness.security import GrantStore, SecurityStore

    state = _state_paths(args)
    policy = SecurityStore(state).read()
    grants = GrantStore(state).list()
    payload = {"tier": policy["tier"], "master": policy["master"],
               "grants": grants, "state_dir": str(state.root)}
    if args.as_json:
        _json(payload)
    else:
        termui.header("tools")
        termui.kv([("tier", payload["tier"]),
                   ("approval prompts", policy["master"].get("approval_prompts")),
                   ("sandbox", policy["master"].get("sandbox")),
                   ("private network", policy["master"].get("private_network")),
                   ("grants", len(grants)),
                   ("state", payload["state_dir"])])
    return 0


def cmd_tools_policy(args: argparse.Namespace) -> int:
    from harness.security import InvalidTier, SecurityStore

    store = SecurityStore(_state_paths(args))
    if args.tier:
        try:
            policy = store.set_tier(args.tier)
        except InvalidTier as exc:
            print(f"hive tools policy: {exc}", file=sys.stderr)
            return 2
    else:
        policy = store.read()
    if args.as_json:
        _json(policy)
    else:
        termui.header("tool policy")
        termui.kv([("tier", policy["tier"]),
                   ("updated", policy.get("updated_at") or "never")])
    return 0


def cmd_tools_grants(args: argparse.Namespace) -> int:
    from harness.security import GrantStore

    rows = GrantStore(_state_paths(args)).list(session_id=args.session)
    if args.as_json:
        _json(rows)
    elif not rows:
        print("  no tool grants")
    else:
        _table(["tool", "scope", "resource", "session", "granted"],
               [[row.get("tool_identity"), row.get("scope"),
                 row.get("resource") or "-", row.get("session_id") or "-",
                 row.get("granted_at")]
                for row in rows])
    return 0


def cmd_tools_revoke(args: argparse.Namespace) -> int:
    from harness.security import GrantStore

    if not args.identity and not args.all:
        print("hive tools revoke: name a tool or pass --all", file=sys.stderr)
        return 2
    if not args.yes:
        print("hive tools revoke: pass --yes to confirm", file=sys.stderr)
        return 2
    result = GrantStore(_state_paths(args)).revoke(
        args.identity, session_id=args.session)
    if args.as_json:
        _json(result)
    else:
        print(f"  revoked {result['global']} global and "
              f"{result['session']} session grant(s)")
    return 0


def cmd_tools_audit(args: argparse.Namespace) -> int:
    from harness.state_store import StatePaths

    path = StatePaths(_state_paths(args).root).tool_audit_file()
    rows: list[dict] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines()[-args.limit:]:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    if args.as_json:
        _json(rows)
    elif not rows:
        print("  no tool audit records")
    else:
        _table(["when", "event", "tool", "decision"],
               [[row.get("ts"), row.get("event"), row.get("tool_identity") or "-",
                 row.get("decision") or row.get("status") or "-"]
                for row in rows])
    return 0


def cmd_sandbox_status(args: argparse.Namespace) -> int:
    from harness import sandbox

    info = sandbox.status(sandbox.SandboxPolicy(_state_paths(args)))
    if args.as_json:
        _json(info)
        return 0
    termui.header("sandbox")
    termui.kv([
        ("docker", "installed" if info["installed"] else "missing"),
        ("daemon", "running" if info["daemon"] else info["detail"]),
        ("image", info["policy"]["docker_image"]),
        ("pinned", "yes" if info["pinned"] else info["pin_error"] or "no"),
        ("network", info["policy"]["network"]),
        ("fallback", info["policy"]["local_fallback"]),
    ])
    if not info["pinned"]:
        print("  pin it with: hive sandbox pin")
    return 0


def cmd_sandbox_policy(args: argparse.Namespace) -> int:
    from harness import sandbox

    policy = sandbox.SandboxPolicy(_state_paths(args))
    fields = {
        "docker_image": args.image,
        "image_digest": args.digest,
        "local_fallback": args.local_fallback,
        "network": args.network,
        "memory": args.memory,
        "cpus": args.cpus,
        "pids_limit": args.pids_limit,
        "timeout_s": args.timeout,
    }
    try:
        data = policy.update(**fields) if any(
            value is not None for value in fields.values()) else policy.read()
    except sandbox.SandboxError as exc:
        print(f"hive sandbox policy: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        _json(data)
    else:
        termui.header("sandbox policy")
        termui.kv([(key, data.get(key)) for key in (
            "docker_image", "image_digest", "network", "memory", "cpus",
            "pids_limit", "timeout_s", "local_fallback", "updated_at")])
    return 0


def cmd_sandbox_pin(args: argparse.Namespace) -> int:
    from harness import sandbox

    policy = sandbox.SandboxPolicy(_state_paths(args))
    try:
        data = sandbox.pin_image(policy, args.image, args.digest)
    except sandbox.SandboxError as exc:
        print(f"hive sandbox pin: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(data)
    else:
        termui.header("sandbox image pinned")
        termui.kv([("image", data["docker_image"]),
                   ("digest", data["image_digest"])])
    return 0


def cmd_web_status(args: argparse.Namespace) -> int:
    from harness import web_fetch

    body = web_fetch.status(_state_paths(args))
    if args.as_json:
        _json(body)
        return 0
    policy = body["policy"]
    termui.header("secure fetch")
    termui.kv([
        ("max bytes", policy["max_bytes"]),
        ("max upload", policy["max_upload_bytes"]),
        ("redirects", policy["max_redirects"]),
        ("timeout", f"{policy['timeout_s']}s"),
        ("public", "allowed" if policy["allow_public"] else "grant only"),
        ("downloads", "allowed" if policy["allow_downloads"] else "off"),
        ("uploads", "allowed" if policy["allow_uploads"] else "off"),
        ("downloads dir", body["download_dir"]),
        ("uploads dir", body["upload_dir"]),
        ("grants", len(body["grants"])),
    ])
    return 0


def cmd_web_fetch(args: argparse.Namespace) -> int:
    from harness import web_fetch

    headers: dict[str, str] = {}
    for raw in args.header or []:
        if "=" not in raw:
            print("hive web fetch: --header expects NAME=VALUE", file=sys.stderr)
            return 2
        name, value = raw.split("=", 1)
        headers[name.strip()] = value.strip()
    try:
        result = web_fetch.fetch(
            args.url, method=args.method, body=args.body, headers=headers,
            session_id=args.session, allow_private=args.allow_private,
            allow_download=bool(args.download),
            download_name=args.download, upload_path=args.upload,
            allow_upload=args.allow_upload, state=_state_paths(args))
    except web_fetch.WebFetchError as exc:
        print(f"hive web fetch: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(result)
        return 0
    termui.header("fetch", result["url"])
    termui.kv([
        ("status", result["status"]),
        ("content type", result["content_type"] or "-"),
        ("bytes", result["bytes"]),
    ])
    if result.get("download"):
        download = result["download"]
        termui.kv([("saved", download["path"]),
                   ("sha256", download["sha256"])])
    if result.get("upload"):
        upload = result["upload"]
        termui.kv([("uploaded", upload["name"]),
                   ("upload bytes", upload["bytes"])])
    if result.get("body") is not None:
        print()
        print(result["body"])
    return 0


def cmd_web_grants(args: argparse.Namespace) -> int:
    from harness import web_fetch

    rows = web_fetch.BrowserGrantStore(_state_paths(args)).list(
        session_id=args.session)
    if args.as_json:
        _json(rows)
    elif not rows:
        print("  no fetch grants")
    else:
        _table(["origin", "access", "scope", "session"],
               [[row.get("origin"), row.get("access"), row.get("scope"),
                 row.get("session_id") or "-"] for row in rows])
    return 0


def cmd_web_grant(args: argparse.Namespace) -> int:
    from harness import web_fetch

    try:
        row = web_fetch.BrowserGrantStore(_state_paths(args)).grant(
            args.origin, access=args.access, scope=args.scope,
            session_id=args.session, reason=args.reason,
            expires_at=args.expires)
    except web_fetch.WebFetchError as exc:
        print(f"hive web grant: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        _json(row)
    else:
        termui.header("fetch grant", row["origin"])
        termui.kv([("access", row["access"]), ("scope", row["scope"]),
                   ("session", row.get("session_id") or "-")])
    return 0


def cmd_web_revoke(args: argparse.Namespace) -> int:
    from harness import web_fetch

    try:
        result = web_fetch.BrowserGrantStore(_state_paths(args)).revoke(
            args.origin, session_id=args.session)
    except web_fetch.WebFetchError as exc:
        print(f"hive web revoke: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        _json(result)
    else:
        print(f"  revoked {result['global']} global and "
              f"{result['session']} session grant(s)")
    return 0


def cmd_web_policy(args: argparse.Namespace) -> int:
    from harness import web_fetch

    fields = {
        "max_bytes": args.max_bytes,
        "max_upload_bytes": args.max_upload_bytes,
        "max_redirects": args.max_redirects,
        "timeout_s": args.timeout,
        "allow_public": args.allow_public,
        "allow_downloads": args.allow_downloads,
        "allow_uploads": args.allow_uploads,
    }
    try:
        policy = web_fetch.WebFetchPolicy(_state_paths(args))
        body = (policy.update(**fields) if any(
            value is not None for value in fields.values()) else policy.read())
    except web_fetch.WebFetchError as exc:
        print(f"hive web policy: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        _json(body)
    else:
        termui.header("fetch policy")
        termui.kv([(key, body.get(key)) for key in (
            "max_bytes", "max_upload_bytes", "max_redirects", "timeout_s",
            "allow_public", "allow_downloads", "allow_uploads", "updated_at")])
    return 0


def cmd_browser_status(args: argparse.Namespace) -> int:
    from harness import browser_adapter

    body = browser_adapter.status()
    if args.as_json:
        _json(body)
        return 0
    termui.header("browser")
    if body.get("available"):
        termui.kv([("backend", body.get("browser", "chromium")),
                   ("executable", body.get("executable", "-"))])
    else:
        termui.kv([("available", "no"), ("reason", body.get("reason") or "?")])
        print(f"  install: {body.get('install')}")
    return 0


def cmd_browser_browse(args: argparse.Namespace) -> int:
    from harness import browser_adapter
    from harness.web_fetch import WebFetchError

    try:
        body = browser_adapter.browse(
            args.url, session_id=args.session, allow_private=args.allow_private,
            screenshot_name=args.screenshot, wait_until=args.wait_until,
            timeout_s=args.timeout, max_text=args.max_text,
            state=_state_paths(args))
    except WebFetchError as exc:
        print(f"hive browser browse: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _json(body)
        return 0
    termui.header("browser", body["url"])
    termui.kv([("title", body["title"] or "-"),
               ("bytes", len(body["text"]))])
    if body.get("screenshot"):
        termui.kv([("screenshot", body["screenshot"]["path"])])
    if body["text"]:
        print()
        print(body["text"])
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    payload = {"hive": _version(), "python": platform.python_version(),
               "platform": f"{platform.system()} {platform.machine()}"}
    if args.as_json:
        _json(payload)
    else:
        print(f"{termui.accent('hive')} {payload['hive']}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def _add_models_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models-dir", default=None,
                        help="GGUF model directory (default: the configured library)")


def _add_stacks_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stacks-dir", default=None,
                        help="directory of stack files (default: <repo>/stacks)")


def _add_state_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", default=None,
                        help="state root (default: HIVE_STATE_DIR or harness_state)")


def _normalize_help(parser: argparse.ArgumentParser) -> None:
    """Fill the generic help missing from repeated ``--json`` flags.

    The flags were added in many branches over time; this keeps their wording
    identical without duplicating the string at every call site.  Explicit
    help (for example ``stacks show``) is never overwritten.
    """
    seen: set[int] = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                if id(child) in seen:
                    continue
                seen.add(id(child))
                _normalize_help(child)
        elif "--json" in action.option_strings and not action.help:
            action.help = "machine-readable JSON on stdout"


def _action_schema(action: argparse.Action) -> dict[str, Any] | None:
    if action.help == argparse.SUPPRESS:
        return None
    item: dict[str, Any] = {
        "kind": "option" if action.option_strings else "argument",
        "name": action.dest,
        "flags": list(action.option_strings),
        "help": action.help or "",
    }
    if action.nargs not in (None, 1):
        item["nargs"] = action.nargs
    if action.required:
        item["required"] = True
    if action.default not in (None, argparse.SUPPRESS):
        item["default"] = action.default
    choices = getattr(action, "choices", None)
    if choices and not isinstance(action, argparse._SubParsersAction):
        item["choices"] = list(choices)
    if action.type in (int, float, str):
        item["type"] = action.type.__name__
    return item


def _command_schema(parser: argparse.ArgumentParser,
                    path: list[str]) -> dict[str, Any]:
    """Describe a parser tree for agents and docs generators.

    The shape is deliberately boring and stable: commands, options, arguments,
    defaults, and choices.  It is derived from the live parser, so it cannot
    drift from ``--help``.
    """
    options: list[dict[str, Any]] = []
    arguments: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            grouped: dict[int, list[str]] = {}
            by_id: dict[int, argparse.ArgumentParser] = {}
            for alias, child in action.choices.items():
                key = id(child)
                grouped.setdefault(key, []).append(alias)
                by_id[key] = child
            for key, aliases in grouped.items():
                child = by_id[key]
                row = _command_schema(child, path + [aliases[0]])
                if len(aliases) > 1:
                    row["aliases"] = aliases[1:]
                commands.append(row)
            continue
        row = _action_schema(action)
        if row is None:
            continue
        if row["kind"] == "option":
            options.append(row)
        else:
            arguments.append(row)
    return {
        "name": path[-1],
        "path": " ".join(path),
        "prog": parser.prog,
        "description": parser.description or getattr(parser, "_help", "") or "",
        "options": options,
        "arguments": arguments,
        "commands": commands,
    }


def _parser_for_topic(topic: str) -> argparse.ArgumentParser | None:
    parser = build_parser()
    for token in shlex.split(topic):
        action = next(
            (a for a in parser._actions
             if isinstance(a, argparse._SubParsersAction)),
            None,
        )
        if action is None or token not in action.choices:
            return None
        parser = action.choices[token]
    return parser


def cmd_help(args: argparse.Namespace) -> int:
    topic = " ".join(args.topic or [])
    parser = _parser_for_topic(topic) if topic else build_parser()
    if parser is None:
        print(f"hive help: unknown command {topic!r}", file=sys.stderr)
        return 2
    if args.as_json:
        path = ["hive"] + (shlex.split(topic) if topic else [])
        _json(_command_schema(parser, path))
        return 0
    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Hive — serve and manage local GGUF stacks, engine-blind.",
        epilog=(
            "docs: docs/CLI.md · quickstart: docs/SERVE-AMD.md\n"
            "data commands accept --json; 'hive help --json' emits the full "
            "command catalog; run 'hive <command> --help' for details."),
    )
    parser.add_argument("--version", action="version", version=f"hive {_version()}")
    parser.add_argument("--no-color", action="store_true",
                        help="disable colour even on a TTY")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    serve.add_parser(sub)

    tui = sub.add_parser(
        "tui",
        help="interactive terminal dashboard",
        description="Open the interactive dashboard (Status, Models, Stacks, Runs).")
    tui.set_defaults(func=cmd_tui)

    status = sub.add_parser(
        "status", aliases=["ps"],
        help="show running serves and their endpoints",
        description="List recorded serves, their tier endpoints, and live health. "
                    "Records are advisory and checked against their process.")
    status.add_argument("--json", action="store_true", dest="as_json",
                        help="machine-readable output")
    status.add_argument("--no-probe", action="store_true",
                        help="do not call the endpoints")
    status.add_argument("--prune", action="store_true",
                        help="remove records whose process is gone")
    status.set_defaults(func=cmd_status)

    stop = sub.add_parser(
        "stop",
        help="stop a serve (name; or --all)",
        description="Stop one recorded serve, the only running serve, or every "
                    "recorded serve with --all. Stale records are cleaned without "
                    "signalling a reused PID.")
    stop.add_argument("name", nargs="?", help="serve name (the only one if omitted)")
    stop.add_argument("--all", action="store_true", help="stop every recorded serve")
    stop.add_argument("--force", action="store_true",
                      help="SIGKILL after the timeout")
    stop.add_argument("--timeout", type=float, default=15.0,
                      help="seconds to wait for a clean stop (default 15)")
    stop.add_argument("--json", action="store_true", dest="as_json")
    stop.set_defaults(func=cmd_stop)

    logs = sub.add_parser(
        "logs",
        help="show or follow a serve's log",
        description="Show a serve's supervisor log, or one tier's engine log "
                    "with --tier. Use --follow to stream new lines.")
    logs.add_argument("name", nargs="?", help="serve name (the only one if omitted)")
    logs.add_argument("--tier", default=None, help="a tier role's engine log")
    logs.add_argument("-n", "--lines", type=int, default=80,
                      help="lines to show; 0 means all (default 80)")
    logs.add_argument("-f", "--follow", action="store_true", help="follow the file")
    logs.add_argument("--json", action="store_true", dest="as_json")
    logs.set_defaults(func=cmd_logs)

    models = sub.add_parser(
        "models",
        help="local GGUF library",
        description="Browse, search, download, import, or remove GGUF models "
                    "in the configured local library.")
    models_sub = models.add_subparsers(dest="models_command", metavar="<action>",
                                       required=True)
    ml = models_sub.add_parser(
        "list", help="list local GGUFs",
        description="List GGUF files in the configured local model library.")
    _add_models_dir(ml)
    ml.add_argument("--json", action="store_true", dest="as_json")
    ml.set_defaults(func=cmd_models_list)

    msearch = models_sub.add_parser(
        "search", help="search Hugging Face for GGUF repos",
        description="Search the Hugging Face Hub for GGUF repositories. Omit "
                    "the query to list popular repos.")
    msearch.add_argument("query", nargs="?", default="",
                         help="words to search for (omit for popular repos)")
    msearch.add_argument("--limit", type=int, default=12,
                         help="number of repos (default 12)")
    _add_models_dir(msearch)
    msearch.add_argument("--json", action="store_true", dest="as_json")
    msearch.set_defaults(func=cmd_models_search)

    mfiles = models_sub.add_parser(
        "files", help="list the GGUFs in a Hugging Face repo",
        description="List the GGUF files and sizes in one Hugging Face repo.")
    mfiles.add_argument("repo", help="org/repo")
    _add_models_dir(mfiles)
    mfiles.add_argument("--json", action="store_true", dest="as_json")
    mfiles.set_defaults(func=cmd_models_files)

    md = models_sub.add_parser(
        "dir", help="show or set the models directory",
        description="Print the resolved GGUF library path, or set a new one "
                    "with PATH/--set. The setting persists.")
    md.add_argument("path", nargs="?", help="a directory to use from now on")
    md.add_argument("--set", dest="set", default=None,
                    help="set the models directory (same as the positional)")
    _add_models_dir(md)
    md.add_argument("--json", action="store_true", dest="as_json")
    md.set_defaults(func=cmd_models_dir)

    mdn = models_sub.add_parser(
        "download", help="download one GGUF from Hugging Face",
        description="Download one GGUF from a Hugging Face repo into the local "
                    "library. Use --file or repo:file to select the file.")
    mdn.add_argument("repo", help="org/repo, or org/repo:file.gguf")
    mdn.add_argument("--file", default=None, help="GGUF file inside the repo")
    mdn.add_argument("--revision", default="main", help="HF revision (default main)")
    _add_models_dir(mdn)
    mdn.add_argument("--json", action="store_true", dest="as_json")
    mdn.set_defaults(func=cmd_models_download)

    mi = models_sub.add_parser(
        "import", help="link a local GGUF or folder into the library",
        description="Link one local .gguf file, or every .gguf in a folder, "
                    "into the library. --copy copies instead of linking.")
    mi.add_argument("path", help="a .gguf file or a folder containing GGUFs")
    mi.add_argument("--copy", action="store_true",
                    help="copy the file instead of hardlink/symlink")
    _add_models_dir(mi)
    mi.add_argument("--json", action="store_true", dest="as_json")
    mi.set_defaults(func=cmd_models_import)

    mr = models_sub.add_parser(
        "remove", help="delete one GGUF from the library",
        description="Delete one GGUF from the library. --yes is required; the "
                    "command never deletes outside the configured library.")
    mr.add_argument("file", help="file name or relative path inside the library")
    mr.add_argument("--yes", action="store_true", help="confirm the deletion")
    _add_models_dir(mr)
    mr.add_argument("--json", action="store_true", dest="as_json")
    mr.set_defaults(func=cmd_models_remove)

    stacks = sub.add_parser(
        "stacks",
        help="saved stacks",
        description="List, inspect, or validate saved stack files without "
                    "starting an engine.")
    stacks_sub = stacks.add_subparsers(dest="stacks_command", metavar="<action>",
                                       required=True)
    sl = stacks_sub.add_parser(
        "list", help="list saved stacks",
        description="List stack files in the stacks directory.")
    _add_stacks_dir(sl)
    sl.add_argument("--json", action="store_true", dest="as_json")
    sl.set_defaults(func=cmd_stacks_list)

    ss = stacks_sub.add_parser(
        "show", help="print one stack as JSON",
        description="Print one stack's resolved JSON. --json is accepted for "
                    "symmetry; this command always prints JSON.")
    ss.add_argument("stack", help="stack name or path")
    _add_stacks_dir(ss)
    ss.add_argument("--json", action="store_true", dest="as_json",
                    help="accepted for symmetry; show always prints JSON")
    ss.set_defaults(func=cmd_stacks_show)

    sv = stacks_sub.add_parser(
        "validate", help="check a stack without starting it",
        description="Validate a stack file without launching an engine. "
                    "--strict treats warnings as failures.")
    sv.add_argument("stack", help="stack name or path")
    _add_stacks_dir(sv)
    _add_models_dir(sv)
    sv.add_argument("--strict", action="store_true",
                    help="treat warnings as failures")
    sv.add_argument("--json", action="store_true", dest="as_json")
    sv.set_defaults(func=cmd_stacks_validate)

    cascade = sub.add_parser(
        "cascade", help="the verification / repair / escalation runner",
        description="Run or inspect the Hive cascade: generate, detect, repair, "
                    "and escalate answers.")
    cascade_sub = cascade.add_subparsers(dest="cascade_command", metavar="<action>",
                                         required=True)
    cruns = cascade_sub.add_parser(
        "runs", help="list recorded cascade runs",
        description="List recorded cascade runs, newest first.")
    cruns.add_argument("--json", action="store_true", dest="as_json")
    cruns.set_defaults(func=cmd_cascade_runs)

    creport = cascade_sub.add_parser(
        "report", help="print one run's report",
        description="Print a run's summary, or its full report with --json.")
    creport.add_argument("run", help="run directory name")
    creport.add_argument("--json", action="store_true", dest="as_json",
                         help="the full report instead of the summary")
    creport.set_defaults(func=cmd_cascade_report)

    crun = cascade_sub.add_parser(
        "run", add_help=False,
        help="run the cascade runner (all arguments are forwarded)",
        description="Run the cascade runner; every argument after 'run' is "
                    "forwarded to experiments.cascade.run_cascade. This "
                    "command does not parse the runner's own flags.")
    crun.add_argument("args", nargs=argparse.REMAINDER,
                      help="arguments for experiments.cascade.run_cascade")
    crun.set_defaults(func=cmd_cascade_run)

    backends = sub.add_parser(
        "backends",
        help="server builds declared, installed or fetched",
        description="Inspect the declared engine server builds, or fetch a "
                    "portable llama.cpp release.")
    backends.add_argument("--json", action="store_true", dest="as_json",
                          help="machine-readable output")
    backends_sub = backends.add_subparsers(dest="backends_command",
                                           metavar="<action>")
    blist = backends_sub.add_parser(
        "list", help="list declared backends",
        description="List declared backends and whether each is installed.")
    blist.add_argument("--json", action="store_true", dest="as_json")
    blist.set_defaults(func=cmd_backends)
    bfetch = backends_sub.add_parser(
        "fetch", help="download a portable llama.cpp release",
        description="Download and install a portable llama.cpp backend "
                    "release. --force replaces an existing directory.")
    bfetch.add_argument("name", nargs="?", default=None,
                        help="backend to install (default: vulkan)")
    bfetch.add_argument("--tag", default=None,
                        help="GitHub release tag (default: newest build)")
    bfetch.add_argument("--force", action="store_true",
                        help="replace an existing backend directory")
    bfetch.add_argument("--json", action="store_true", dest="as_json")
    bfetch.set_defaults(func=cmd_backends_fetch)
    backends.set_defaults(func=cmd_backends)

    setup = sub.add_parser(
        "setup",
        help="onboard this box: engine, model/stacks/state paths",
        description="Create missing paths, fetch a portable engine when needed, "
                    "and print the next commands. It does not download a model.")
    _add_models_dir(setup)
    _add_stacks_dir(setup)
    setup.add_argument("--no-fetch", action="store_true",
                       help="do not fetch a portable engine if none is installed")
    setup.add_argument("--json", action="store_true", dest="as_json")
    setup.set_defaults(func=cmd_setup)

    doctor = sub.add_parser(
        "doctor",
        help="check the box before serving",
        description="Check Python, paths, engines, cards, host memory, ports, "
                    "and the Hugging Face token before a serve. --strict treats "
                    "warnings as failures.")
    _add_models_dir(doctor)
    _add_stacks_dir(doctor)
    doctor.add_argument("--strict", action="store_true",
                        help="treat warnings as failures")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    doctor.set_defaults(func=cmd_doctor)

    config = sub.add_parser(
        "config", help="resolved paths and versions",
        description="Print the resolved model, stack, state, log, and backend "
                    "paths plus versions. Read-only.")
    _add_models_dir(config)
    _add_stacks_dir(config)
    config.add_argument("--json", action="store_true", dest="as_json")
    config.set_defaults(func=cmd_config)

    profile = sub.add_parser(
        "profile", help="saved hardware profiles",
        description="Scan or inspect the local hardware snapshot used by the "
                    "deterministic controller. A scan is opt-in and local-only.")
    profile_sub = profile.add_subparsers(dest="profile_command",
                                         metavar="<action>", required=True)
    ps = profile_sub.add_parser(
        "scan", help="scan this machine with consent",
        description="Probe CPU, RAM, GPUs, VRAM/GTT and storage, then save a "
                    "versioned profile. Nothing leaves the machine.")
    ps.add_argument("--consent", default="",
                    help="comma list of permitted reads; hardware is required")
    ps.add_argument("--profile-id", default=None,
                    help="stable profile id (default: generated from date + random)")
    _add_state_dir(ps)
    ps.add_argument("--json", action="store_true", dest="as_json")
    ps.set_defaults(func=cmd_profile_scan)

    pshow = profile_sub.add_parser(
        "show", help="print one saved hardware profile",
        description="Print the newest saved profile, or the one named by "
                    "PROFILE_ID.")
    pshow.add_argument("profile_id", nargs="?",
                       help="profile id (default: newest)")
    _add_state_dir(pshow)
    pshow.add_argument("--json", action="store_true", dest="as_json")
    pshow.set_defaults(func=cmd_profile_show)

    plist = profile_sub.add_parser(
        "list", help="list saved hardware profiles",
        description="List every readable hardware snapshot, oldest first.")
    _add_state_dir(plist)
    plist.add_argument("--json", action="store_true", dest="as_json")
    plist.set_defaults(func=cmd_profile_list)

    pdel = profile_sub.add_parser(
        "delete", help="delete one saved profile",
        description="Delete one profile by id, or the newest when omitted. "
                    "--yes is required.")
    pdel.add_argument("profile_id", nargs="?",
                      help="profile id (default: newest)")
    pdel.add_argument("--yes", action="store_true",
                      help="confirm the deletion")
    _add_state_dir(pdel)
    pdel.add_argument("--json", action="store_true", dest="as_json")
    pdel.set_defaults(func=cmd_profile_delete)

    controller = sub.add_parser(
        "controller", help="the deterministic settings controller",
        description="Propose, inspect, apply, or revert allowlisted harness "
                    "settings derived from the saved hardware profile.")
    controller_sub = controller.add_subparsers(
        dest="controller_command", metavar="<action>", required=True)
    cstatus = controller_sub.add_parser(
        "status", help="show the active profile and proposal",
        description="Show the saved hardware profile, latest proposal, and "
                    "last applied controller change.")
    _add_state_dir(cstatus)
    cstatus.add_argument("--json", action="store_true", dest="as_json")
    cstatus.set_defaults(func=cmd_controller_status)

    cprop = controller_sub.add_parser(
        "propose", help="build a settings proposal",
        description="Build an explainable E0 proposal. A model path or size is "
                    "optional; without it only a conservative thread count is "
                    "proposed.")
    cprop.add_argument("--profile", default=None,
                       help="hardware profile id (default: newest)")
    cprop.add_argument("--model", default=None,
                       help="GGUF path for size and native context")
    cprop.add_argument("--model-gib", type=float, default=None,
                       help="model weight size in GiB (when no path is given)")
    cprop.add_argument("--native-ctx", type=int, default=None,
                       help="model's native context length")
    cprop.add_argument("--ctx", type=int, default=None,
                       help="largest context the proposal may choose")
    cprop.add_argument("--kv-type", default="q8_0",
                       help="KV cache type used for the estimate (default q8_0)")
    cprop.add_argument("--request", default=None,
                       help="request text for the deterministic feature baseline")
    cprop.add_argument("--embed-request", action="store_true",
                       help="also encode the request on CPU and trace the vector")
    _add_state_dir(cprop)
    cprop.add_argument("--json", action="store_true", dest="as_json")
    cprop.set_defaults(func=cmd_controller_propose)

    capply = controller_sub.add_parser(
        "apply", help="apply a proposal with confirmation",
        description="Validate and apply one allowlisted proposal. The previous "
                    "values are saved for rollback; --yes is required.")
    capply.add_argument("proposal", nargs="?",
                        help="proposal id (default: latest)")
    capply.add_argument("--profile", default=None,
                        help="profile id to search (default: all profiles)")
    capply.add_argument("--yes", action="store_true",
                        help="confirm the settings change")
    _add_state_dir(capply)
    capply.add_argument("--json", action="store_true", dest="as_json")
    capply.set_defaults(func=cmd_controller_apply)

    coutcome = controller_sub.add_parser(
        "outcome", help="label a proposal for the E1 training trace",
        description="Append an accepted/rejected/wrong label and optional "
                    "metrics to the privacy-preserving controller trace.")
    coutcome.add_argument("proposal", help="proposal id to label")
    coutcome.add_argument("outcome", choices=(
        "accepted", "rejected", "wrong", "unknown", "abandoned"),
        help="the observed result")
    coutcome.add_argument("--metric", action="append", default=[],
                          help="key=value metric; repeatable")
    _add_state_dir(coutcome)
    coutcome.add_argument("--json", action="store_true", dest="as_json")
    coutcome.set_defaults(func=cmd_controller_outcome)

    crevert = controller_sub.add_parser(
        "revert", help="roll back the last applied proposal",
        description="Restore the values captured before the last apply. "
                    "--yes is required.")
    crevert.add_argument("--yes", action="store_true",
                         help="confirm the rollback")
    _add_state_dir(crevert)
    crevert.add_argument("--json", action="store_true", dest="as_json")
    crevert.set_defaults(func=cmd_controller_revert)

    encoder = sub.add_parser(
        "encoder", help="the local text encoder boundary",
        description="Inspect or lazy-load the configured text encoder on the "
                    "CPU. Nothing is downloaded unless --allow-download is set.")
    encoder_sub = encoder.add_subparsers(dest="encoder_command",
                                         metavar="<action>", required=True)
    estatus = encoder_sub.add_parser(
        "status", help="show the configured encoder without loading it",
        description="Report the model, backend, cache policy and whether the "
                    "sentence-transformers backend is importable.")
    estatus.add_argument("--model", default=None,
                         help="model id to inspect (default: Studio encoder block)")
    estatus.add_argument("--allow-download", action="store_true",
                         help="allow the backend to download a missing model")
    _add_state_dir(estatus)
    estatus.add_argument("--json", action="store_true", dest="as_json")
    estatus.set_defaults(func=cmd_encoder_status)

    eembed = encoder_sub.add_parser(
        "embed", help="embed one text on the CPU",
        description="Load the configured encoder lazily and print one vector. "
                    "Use --allow-download only when a model download is wanted.")
    eembed.add_argument("text", nargs="?",
                        help="text to embed (default: read stdin)")
    eembed.add_argument("--model", default=None,
                        help="model id (default: Studio encoder block)")
    eembed.add_argument("--allow-download", action="store_true",
                        help="allow the backend to download a missing model")
    _add_state_dir(eembed)
    eembed.add_argument("--json", action="store_true", dest="as_json")
    eembed.set_defaults(func=cmd_encoder_embed)

    ui = sub.add_parser(
        "ui", help="switch between terminal and web interfaces",
        description="Use the terminal dashboard or the web Studio. With no "
                    "mode, report the web endpoint and the switch commands.")
    ui.add_argument("mode", nargs="?", choices=("web", "terminal", "status"),
                    default="status",
                    help="target interface (default: status)")
    ui.add_argument("--host", default="127.0.0.1",
                    help="web bind host (default 127.0.0.1)")
    ui.add_argument("--port", type=int, default=8765,
                    help="web port (default 8765)")
    ui.add_argument("--detach", action="store_true",
                    help="start the web Studio in the background")
    ui.add_argument("--no-open", action="store_true",
                    help="do not open a browser")
    ui.add_argument("--mock", action="store_true",
                    help="start the Studio in model-free mock mode")
    ui.add_argument("--state-dir", default=None,
                    help="studio state directory (passed through to the web UI)")
    ui.add_argument("--json", action="store_true", dest="as_json")
    ui.set_defaults(func=cmd_ui)

    tools = sub.add_parser(
        "tools", help="tool policy, grants and audit",
        description="Inspect or change the tool gateway policy. The gateway "
                    "owns approvals; this command never executes a tool.")
    tools_sub = tools.add_subparsers(dest="tools_command", metavar="<action>",
                                     required=True)
    tstatus = tools_sub.add_parser(
        "status", help="show the policy and grant count",
        description="Show the current safety tier, master switches, and grants.")
    _add_state_dir(tstatus)
    tstatus.add_argument("--json", action="store_true", dest="as_json")
    tstatus.set_defaults(func=cmd_tools_status)

    tpending = tools_sub.add_parser(
        "pending", help="list pending tool approvals from a running Studio",
        description="Ask the local Studio gateway for pending approval requests. "
                    "The Studio must be running.")
    tpending.add_argument("--session", default=None,
                          help="only show this session")
    tpending.add_argument("--url", default=None,
                          help="Studio URL (default HIVE_STUDIO_URL or :8765)")
    tpending.add_argument("--json", action="store_true", dest="as_json")
    tpending.set_defaults(func=cmd_tools_pending)

    tdecide = tools_sub.add_parser(
        "decide", help="approve or deny a pending tool request",
        description="Resolve a pending request in a running Studio: once, "
                    "session, always (with --scope global), or deny.")
    tdecide.add_argument("request", help="pending request id")
    tdecide.add_argument("--session", required=True, help="requesting session id")
    tdecide.add_argument("--choice", required=True, choices=(
        "once", "session", "always", "deny"), help="decision")
    tdecide.add_argument("--scope", default="session", choices=(
        "session", "global"), help="grant scope for always (default session)")
    tdecide.add_argument("--url", default=None,
                         help="Studio URL (default HIVE_STUDIO_URL or :8765)")
    tdecide.add_argument("--json", action="store_true", dest="as_json")
    tdecide.set_defaults(func=cmd_tools_decide)

    tpolicy = tools_sub.add_parser(
        "policy", help="show or set the safety tier",
        description="Show the current policy, or set strict/guarded/trusted/open. "
                    "Open still keeps the hard guards.")
    tpolicy.add_argument("tier", nargs="?", choices=(
        "strict", "guarded", "trusted", "open"),
        help="new tier (omit to show the current policy)")
    _add_state_dir(tpolicy)
    tpolicy.add_argument("--json", action="store_true", dest="as_json")
    tpolicy.set_defaults(func=cmd_tools_policy)

    tgrants = tools_sub.add_parser(
        "grants", help="list persistent and session grants",
        description="List global grants and, with --session, in-memory grants "
                    "for one session.")
    tgrants.add_argument("--session", default=None,
                         help="also show this session's grants")
    _add_state_dir(tgrants)
    tgrants.add_argument("--json", action="store_true", dest="as_json")
    tgrants.set_defaults(func=cmd_tools_grants)

    trevoke = tools_sub.add_parser(
        "revoke", help="revoke a tool grant",
        description="Remove one tool identity, or every grant with --all. "
                    "--yes is required.")
    trevoke.add_argument("identity", nargs="?",
                         help="source:name@version identity")
    trevoke.add_argument("--all", action="store_true", help="revoke every grant")
    trevoke.add_argument("--session", default=None,
                         help="also revoke this session's grants")
    trevoke.add_argument("--yes", action="store_true", help="confirm the removal")
    _add_state_dir(trevoke)
    trevoke.add_argument("--json", action="store_true", dest="as_json")
    trevoke.set_defaults(func=cmd_tools_revoke)

    taudit = tools_sub.add_parser(
        "audit", help="show recent tool decisions",
        description="Print the newest tool-gateway audit records. Arguments "
                    "are stored as hashes, not raw values.")
    taudit.add_argument("--limit", type=int, default=20,
                        help="number of recent records (default 20)")
    _add_state_dir(taudit)
    taudit.add_argument("--json", action="store_true", dest="as_json")
    taudit.set_defaults(func=cmd_tools_audit)

    sandbox = sub.add_parser(
        "sandbox", help="Docker sandbox policy and image pinning",
        description="Inspect the sandbox, pin the container image by digest, "
                    "or change the local-fallback rule. A host fallback is "
                    "never silent.")
    sandbox_sub = sandbox.add_subparsers(
        dest="sandbox_command", metavar="<action>", required=True)
    sstatus = sandbox_sub.add_parser(
        "status", help="show Docker, image pin, and fallback state",
        description="Probe Docker live and show the image pin, network and "
                    "local-fallback policy.")
    _add_state_dir(sstatus)
    sstatus.add_argument("--json", action="store_true", dest="as_json")
    sstatus.set_defaults(func=cmd_sandbox_status)

    spolicy = sandbox_sub.add_parser(
        "policy", help="show or change the sandbox policy",
        description="Show or change the pinned image, resource limits, network, "
                    "and the approval rule for a local host fallback.")
    spolicy.add_argument("--image", default=None, help="container image name")
    spolicy.add_argument("--digest", default=None,
                         help="sha256 digest (64 hex characters)")
    spolicy.add_argument("--local-fallback", default=None,
                         choices=("approval", "allow", "deny"),
                         help="host fallback rule")
    spolicy.add_argument("--network", default=None,
                         help="Docker network mode (default none)")
    spolicy.add_argument("--memory", default=None, help="memory limit")
    spolicy.add_argument("--cpus", default=None, help="CPU limit")
    spolicy.add_argument("--pids-limit", type=int, default=None,
                         dest="pids_limit", help="process limit")
    spolicy.add_argument("--timeout", type=int, default=None,
                         dest="timeout", help="maximum timeout in seconds")
    _add_state_dir(spolicy)
    spolicy.add_argument("--json", action="store_true", dest="as_json")
    spolicy.set_defaults(func=cmd_sandbox_policy)

    spin = sandbox_sub.add_parser(
        "pin", help="resolve and save the image digest",
        description="Resolve the image's sha256 digest with Docker, or record "
                    "a digest supplied offline, so the tag cannot move.")
    spin.add_argument("--image", default=None,
                      help="image to pin (default: the policy image)")
    spin.add_argument("--digest", default=None,
                      help="pin this 64-character sha256 digest offline")
    _add_state_dir(spin)
    spin.add_argument("--json", action="store_true", dest="as_json")
    spin.set_defaults(func=cmd_sandbox_pin)

    web = sub.add_parser(
        "web", help="secure HTTP fetch and per-origin grants",
        description="Fetch public/private HTTP(S) URLs with manual redirect "
                    "checks, size/type limits, and explicit download/upload "
                    "gates. This is not a JavaScript browser.")
    web_sub = web.add_subparsers(dest="web_command", metavar="<action>",
                                 required=True)
    wstatus = web_sub.add_parser(
        "status", help="show fetch limits, directories, and grants",
        description="Show the secure fetch policy and the current grants.")
    _add_state_dir(wstatus)
    wstatus.add_argument("--json", action="store_true", dest="as_json",
                         help="machine-readable JSON on stdout")
    wstatus.set_defaults(func=cmd_web_status)

    wfetch = web_sub.add_parser(
        "fetch", help="fetch one URL with per-hop SSRF checks",
        description="Fetch a URL. A private origin needs a grant or "
                    "--allow-private with --session. Redirects are checked "
                    "again before each hop.")
    wfetch.add_argument("url", help="http(s) URL to fetch")
    wfetch.add_argument("--method", default="GET", choices=("GET", "POST"),
                         help="HTTP method (default GET)")
    wfetch.add_argument("--body", default=None, help="text body for POST")
    wfetch.add_argument("--header", action="append", default=[],
                        metavar="NAME=VALUE",
                        help="safe request header (repeatable)")
    wfetch.add_argument("--session", default=None,
                        help="session id for a private-origin approval")
    wfetch.add_argument("--allow-private", action="store_true",
                        help="approve the original private origin")
    wfetch.add_argument("--download", default=None, metavar="NAME",
                        help="write the response to the downloads directory")
    wfetch.add_argument("--upload", default=None, metavar="PATH",
                        help="read the POST body from the uploads directory")
    wfetch.add_argument("--allow-upload", action="store_true",
                         help="send --upload as the POST body")
    _add_state_dir(wfetch)
    wfetch.add_argument("--json", action="store_true", dest="as_json",
                         help="machine-readable JSON on stdout")
    wfetch.set_defaults(func=cmd_web_fetch)

    wgrants = web_sub.add_parser(
        "grants", help="list per-origin fetch grants",
        description="List global grants and, with --session, session grants.")
    wgrants.add_argument("--session", default=None,
                          help="also show this session's grants")
    _add_state_dir(wgrants)
    wgrants.add_argument("--json", action="store_true", dest="as_json",
                         help="machine-readable JSON on stdout")
    wgrants.set_defaults(func=cmd_web_grants)

    wgrant = web_sub.add_parser(
        "grant", help="grant public/private access to one origin",
        description="Approve one exact origin for public or private fetches. "
                    "Session grants stay in this process; global grants persist.")
    wgrant.add_argument("origin", help="origin such as https://example.com")
    wgrant.add_argument("--access", default="private",
                        choices=("public", "private"),
                        help="which access class this grant covers")
    wgrant.add_argument("--scope", default="global",
                        choices=("global", "session"),
                        help="persist globally or only for this process session")
    wgrant.add_argument("--session", default=None,
                         help="session id for a session-scoped grant")
    wgrant.add_argument("--reason", default="user approved fetch origin",
                         help="reason recorded in the audit log")
    wgrant.add_argument("--expires", type=float, default=None,
                         help="optional Unix expiry time")
    _add_state_dir(wgrant)
    wgrant.add_argument("--json", action="store_true", dest="as_json",
                         help="machine-readable JSON on stdout")
    wgrant.set_defaults(func=cmd_web_grant)

    wrevoke = web_sub.add_parser(
        "revoke", help="revoke one origin, or all with no origin",
        description="Remove one exact-origin grant, or every grant when no "
                    "origin is supplied.")
    wrevoke.add_argument("origin", nargs="?", default=None,
                          help="origin to revoke (omit to revoke all)")
    wrevoke.add_argument("--session", default=None,
                          help="also revoke this session's grants")
    _add_state_dir(wrevoke)
    wrevoke.add_argument("--json", action="store_true", dest="as_json",
                         help="machine-readable JSON on stdout")
    wrevoke.set_defaults(func=cmd_web_revoke)

    wpolicy = web_sub.add_parser(
        "policy", help="show or change fetch limits and gates",
        description="Show or change response/upload limits, redirects, timeout, "
                    "and the public/download/upload gates.")
    wpolicy.add_argument("--max-bytes", type=int, default=None,
                         dest="max_bytes", help="response byte limit")
    wpolicy.add_argument("--max-upload-bytes", type=int, default=None,
                         dest="max_upload_bytes", help="upload/body byte limit")
    wpolicy.add_argument("--max-redirects", type=int, default=None,
                         dest="max_redirects", help="redirect hop limit")
    wpolicy.add_argument("--timeout", type=int, default=None,
                         help="per-request timeout in seconds")
    wpolicy.add_argument("--allow-public", dest="allow_public",
                         action="store_true", default=None,
                         help="allow public origins without a grant")
    wpolicy.add_argument("--no-public", dest="allow_public",
                         action="store_false",
                         help="require a public-origin grant")
    wpolicy.add_argument("--allow-downloads", dest="allow_downloads",
                         action="store_true", default=None,
                         help="permit explicit downloads")
    wpolicy.add_argument("--no-downloads", dest="allow_downloads",
                         action="store_false",
                         help="refuse all downloads")
    wpolicy.add_argument("--allow-uploads", dest="allow_uploads",
                         action="store_true", default=None,
                         help="permit explicit uploads")
    wpolicy.add_argument("--no-uploads", dest="allow_uploads",
                         action="store_false",
                         help="refuse all uploads")
    _add_state_dir(wpolicy)
    wpolicy.add_argument("--json", action="store_true", dest="as_json",
                         help="machine-readable JSON on stdout")
    wpolicy.set_defaults(func=cmd_web_policy)

    browser = sub.add_parser(
        "browser", help="optional Playwright browser over a validating proxy",
        description="Open a page in a fresh Chromium context. Every HTTP "
                    "request and HTTPS CONNECT is checked by the local proxy, "
                    "including redirects. Requires the optional browser extra.")
    browser_sub = browser.add_subparsers(
        dest="browser_command", metavar="<action>", required=True)
    bstatus = browser_sub.add_parser(
        "status", help="check the optional browser dependency",
        description="Report whether Playwright can resolve a Chromium binary.")
    bstatus.add_argument("--json", action="store_true", dest="as_json",
                         help="machine-readable JSON on stdout")
    bstatus.set_defaults(func=cmd_browser_status)
    bbrowse = browser_sub.add_parser(
        "browse", help="open one URL in a fresh browser context",
        description="Open a page through the validating proxy. Private "
                    "origins need a grant or --allow-private with --session.")
    bbrowse.add_argument("url", help="http(s) URL to open")
    bbrowse.add_argument("--session", default=None,
                         help="session id for a private-origin approval")
    bbrowse.add_argument("--allow-private", action="store_true",
                         help="approve the original private origin")
    bbrowse.add_argument("--screenshot", default=None, metavar="NAME",
                         help="save a screenshot under browser/screenshots")
    bbrowse.add_argument("--wait-until", default="domcontentloaded",
                         choices=("domcontentloaded", "load", "networkidle"),
                         help="navigation readiness event")
    bbrowse.add_argument("--timeout", type=int, default=30,
                         help="navigation timeout in seconds")
    bbrowse.add_argument("--max-text", type=int, default=20000,
                         help="maximum body text characters returned")
    _add_state_dir(bbrowse)
    bbrowse.add_argument("--json", action="store_true", dest="as_json",
                         help="machine-readable JSON on stdout")
    bbrowse.set_defaults(func=cmd_browser_browse)

    version = sub.add_parser(
        "version", help="print the Hive version",
        description="Print Hive, Python, and platform versions.")
    version.add_argument("--json", action="store_true", dest="as_json",
                         help="machine-readable JSON on stdout")
    version.set_defaults(func=cmd_version)

    help_parser = sub.add_parser(
        "help",
        help="show help or a machine-readable command catalog",
        description="Print human help for a command, or emit the complete "
                    "command/option catalog as JSON for agents and scripts.")
    help_parser.add_argument(
        "topic", nargs="*", default=None,
        help="command or subcommand, e.g. 'serve' or 'models list'")
    help_parser.add_argument("--json", action="store_true", dest="as_json",
                             help="emit the command/option catalog as JSON")
    help_parser.set_defaults(func=cmd_help)
    _normalize_help(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    passthrough = len(raw) >= 2 and raw[0] == "cascade" and raw[1] == "run"
    # Accept --no-color before or after the command, not only where argparse
    # would look for a global option.
    if not passthrough and "--no-color" in raw:
        termui.set_color(False)
        raw = [item for item in raw if item != "--no-color"]
    # `cascade run` is a pass-through: the runner owns its argument surface, so
    # do not let argparse reject its flags (including --help) before forwarding.
    if passthrough:
        return cmd_cascade_run(argparse.Namespace(args=raw[2:]))
    parser = build_parser()
    args = parser.parse_args(raw)
    if not getattr(args, "func", None):
        if _tui_available():
            from harness import tui
            code = tui.run()
            if code == 0:
                return 0
            print("hive: the interactive dashboard could not start; "
                  "showing the static one", file=sys.stderr)
        return cmd_home(args)
    try:
        return int(args.func(args) or 0)
    except BrokenPipeError:  # pragma: no cover - `hive models list | head`
        return 0
    except KeyboardInterrupt:
        print("hive: interrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"hive: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - one line for the operator
        if os.environ.get("HIVE_DEBUG"):
            raise
        print(f"hive: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
