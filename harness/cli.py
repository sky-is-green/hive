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
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from harness import backends as backend_manifest
from harness import serve, serve_state
from harness.hardware import linux_amd_devices
from harness.models import LlamaServerManager
from harness.stack import schema


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------

def _version() -> str:
    try:
        from importlib.metadata import version as package_version
        return package_version("hivebench")
    except Exception:  # noqa: BLE001 - a source checkout may not be installed
        return "0.1.0"


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _table(headers: list[str], rows: list[list[Any]], indent: str = "  ") -> None:
    data = [[("" if cell is None else str(cell)) for cell in row] for row in rows]
    widths = []
    for index, header in enumerate(headers):
        width = len(header)
        for row in data:
            if index < len(row):
                width = max(width, len(row[index]))
        widths.append(width)
    print(indent + "  ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)).rstrip())
    for row in data:
        cells = [row[index] if index < len(row) else "" for index in range(len(headers))]
        print(indent + "  ".join(
            cells[index].ljust(widths[index]) for index in range(len(headers))).rstrip())


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


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


# ---------------------------------------------------------------------------
# status / stop / logs
# ---------------------------------------------------------------------------

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
        print("nothing running (start one with: hive serve <stack>)")
        if pruned:
            print(f"pruned stale record(s): {', '.join(pruned)}")
        return 0
    for view in views:
        name = view.get("name", "?")
        head = (f"{name}  pid {view['pid']}  {view.get('mode') or 'foreground'}  "
                f"up {serve_state.format_duration(view['uptime_s'])}")
        if view["state"] != "running":
            head += "  STALE (process is gone or the pid was reused)"
        print(head)
        for tier in view["tiers"]:
            endpoint = _tier_endpoint(view, tier)
            probe = tier.get("probe") or {}
            status = "-"
            if probe.get("up"):
                status = "up"
            elif probe:
                status = "down"
            ctx = tier.get("ctx") or ""
            backend = tier.get("backend") or ""
            print(f"  {str(tier.get('role') or '?'):<8} {endpoint:<42} "
                  f"ctx {str(ctx):<6} {backend:<8} {status}")
        if view.get("log_path"):
            print(f"  log  {view['log_path']}")
        elif view["state"] == "running":
            print("  log  (foreground; logs are on its terminal)")
        if view["state"] != "running":
            print(f"  prune with: hive status --prune")
    if pruned:
        print(f"pruned stale record(s): {', '.join(pruned)}")
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
        print("nothing running")
        return 0
    else:
        names = ", ".join(str(record.get("name")) for record in records)
        print(f"hive stop: several serves are running ({names}); name one or use --all",
              file=sys.stderr)
        return 2

    if not targets:
        print("nothing running")
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
            print(f"{'stopped' if result['stopped'] else 'could not stop'} "
                  f"{result['name']}: {result['message']}")
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

    print(f"-- following {path} (Ctrl-C to stop) --", file=sys.stderr)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, os.SEEK_END)
            while True:
                line = handle.readline()
                if line:
                    print(line, end="", flush=True)
                    continue
                if not serve_state.pid_alive(int(record["pid"])):
                    print("-- the serve has stopped --", file=sys.stderr)
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
    rows = manager.list_local(use_cache=False)
    if args.as_json:
        _json(rows)
        return 0
    if not rows:
        print(f"no GGUF models in {manager.models_dir}")
        print("  download one with: hive models download <repo>[:<file>]")
        return 0
    _table(
        ["NAME", "SIZE", "QUANT", "CTX", "FILE"],
        [[row.get("name"), f"{float(row.get('size_gb') or 0):.2f} GiB",
          row.get("quantization") or "-", row.get("context_length") or "-",
          row.get("file")] for row in rows])
    print(f"\n  {len(rows)} model(s) in {manager.models_dir}")
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
    if not rows:
        print("no GGUF repos matched")
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
    if not rows:
        print(f"{args.repo} has no .gguf files")
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
        print(f"downloaded {repo}:{file}")
        print(f"  {path}  ({size:.2f} GiB)")
    return 0


def cmd_models_import(args: argparse.Namespace) -> int:
    manager = _models_manager(args)
    source = Path(args.path).expanduser()
    if source.is_dir():
        result = manager.import_local_path(str(source))
        if args.as_json:
            _json(result)
        else:
            print(f"linked {source} into {manager.models_dir}")
            for item in result.get("imported", [])[:20]:
                print(f"  {item}")
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
                shutil.copy2(source, dest)
                how = "copy"
    else:
        shutil.copy2(source, dest)
    payload = {"imported": dest.name, "path": str(dest), "linked": how != "copy",
               "how": how}
    if args.as_json:
        _json(payload)
    else:
        print(f"imported {source.name} ({how}) -> {dest}")
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
        print(f"removed {result['deleted']}")
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
    if not rows:
        print(f"no saved stacks in {directory}")
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
    print(f"stack {stack.name!r} ({len(stack.tiers)} tier(s))")
    for problem in problems:
        print(f"  [fail] {problem}")
    for row in details:
        mark = {"ok": "ok", "warn": "warn", "fail": "fail"}[row["status"]]
        print(f"  [{mark:<4}] {row['role']:<8} {row['detail']}")
    for warning in warnings:
        print(f"  [warn] {warning}")
    print("  ok" if ok else "  validation failed")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# backends / doctor / config / version
# ---------------------------------------------------------------------------

def cmd_backends(args: argparse.Namespace) -> int:
    rows = backend_manifest.rows()
    if args.as_json:
        _json(rows)
        return 0
    _table(["NAME", "STATUS", "ACCELERATORS", "BINARY", "LABEL"],
           [[row["name"], "installed" if row["installed"] else "-",
             ",".join(row.get("accelerators") or []), row.get("binary"),
             row.get("label")] for row in rows])
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
                "none installed; run scripts/install_amd_engine.sh or "
                "hive backends to see what is declared")
    except (OSError, ValueError) as exc:
        add("fail", "backends", f"backend manifest: {exc}")

    try:
        devices = linux_amd_devices()
    except OSError as exc:
        devices = []
        add("warn", "cards", f"cannot read amdgpu sysfs: {exc}")
    else:
        if devices:
            detail = "; ".join(
                f"card {d.get('index')}: {d.get('memory_gb')} GiB"
                f"{' (display)' if d.get('display') else ''}" for d in devices)
            add("ok", "cards", detail)
        elif platform.system() == "Linux":
            add("warn", "cards", "no amdgpu cards found via sysfs")
        else:
            add("warn", "cards", "GPU probing is Linux-only")

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
    width = max((len(check["name"]) for check in checks), default=0)
    for check in checks:
        print(f"  [{check['status']:<4}] {check['name']:<{width}}  {check['detail']}")
    if failures:
        print(f"\n  {len(failures)} failure(s); fix them before serving")
    elif args.strict and warnings:
        print(f"\n  {len(warnings)} warning(s) (--strict)")
    else:
        print("\n  ready" if not warnings else f"\n  ready ({len(warnings)} warning(s))")
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
        for key, value in payload.items():
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value) or "-"
            print(f"{key:<18} {value}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    payload = {"hive": _version(), "python": platform.python_version(),
               "platform": f"{platform.system()} {platform.machine()}"}
    if args.as_json:
        _json(payload)
    else:
        print(f"hive {payload['hive']}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Hive — serve and manage local GGUF stacks, engine-blind.",
        epilog=(
            "docs: docs/CLI.md · quickstart: docs/SERVE-AMD.md\n"
            "every command accepts --json; run 'hive <command> --help' for details."),
    )
    parser.add_argument("--version", action="version", version=f"hive {_version()}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    serve.add_parser(sub)

    status = sub.add_parser("status", aliases=["ps"],
                            help="show running serves and their endpoints")
    status.add_argument("--json", action="store_true", dest="as_json",
                        help="machine-readable output")
    status.add_argument("--no-probe", action="store_true",
                        help="do not call the endpoints")
    status.add_argument("--prune", action="store_true",
                        help="remove records whose process is gone")
    status.set_defaults(func=cmd_status)

    stop = sub.add_parser("stop", help="stop a serve (name; or --all)")
    stop.add_argument("name", nargs="?", help="serve name (the only one if omitted)")
    stop.add_argument("--all", action="store_true", help="stop every recorded serve")
    stop.add_argument("--force", action="store_true",
                      help="SIGKILL after the timeout")
    stop.add_argument("--timeout", type=float, default=15.0,
                      help="seconds to wait for a clean stop (default 15)")
    stop.add_argument("--json", action="store_true", dest="as_json")
    stop.set_defaults(func=cmd_stop)

    logs = sub.add_parser("logs", help="show or follow a serve's log")
    logs.add_argument("name", nargs="?", help="serve name (the only one if omitted)")
    logs.add_argument("--tier", default=None, help="a tier role's engine log")
    logs.add_argument("-n", "--lines", type=int, default=80,
                      help="lines to show; 0 means all (default 80)")
    logs.add_argument("-f", "--follow", action="store_true", help="follow the file")
    logs.add_argument("--json", action="store_true", dest="as_json")
    logs.set_defaults(func=cmd_logs)

    models = sub.add_parser("models", help="local GGUF library")
    models_sub = models.add_subparsers(dest="models_command", metavar="<action>",
                                       required=True)
    ml = models_sub.add_parser("list", help="list local GGUFs")
    _add_models_dir(ml)
    ml.add_argument("--json", action="store_true", dest="as_json")
    ml.set_defaults(func=cmd_models_list)

    msearch = models_sub.add_parser("search", help="search Hugging Face for GGUF repos")
    msearch.add_argument("query", help="words to search for")
    msearch.add_argument("--limit", type=int, default=12,
                         help="number of repos (default 12)")
    _add_models_dir(msearch)
    msearch.add_argument("--json", action="store_true", dest="as_json")
    msearch.set_defaults(func=cmd_models_search)

    mfiles = models_sub.add_parser("files", help="list the GGUFs in a Hugging Face repo")
    mfiles.add_argument("repo", help="org/repo")
    _add_models_dir(mfiles)
    mfiles.add_argument("--json", action="store_true", dest="as_json")
    mfiles.set_defaults(func=cmd_models_files)

    md = models_sub.add_parser("dir", help="show or set the models directory")
    md.add_argument("path", nargs="?", help="a directory to use from now on")
    md.add_argument("--set", dest="set", default=None,
                    help="set the models directory (same as the positional)")
    _add_models_dir(md)
    md.add_argument("--json", action="store_true", dest="as_json")
    md.set_defaults(func=cmd_models_dir)

    mdn = models_sub.add_parser("download", help="download one GGUF from Hugging Face")
    mdn.add_argument("repo", help="org/repo, or org/repo:file.gguf")
    mdn.add_argument("--file", default=None, help="GGUF file inside the repo")
    mdn.add_argument("--revision", default="main", help="HF revision (default main)")
    _add_models_dir(mdn)
    mdn.add_argument("--json", action="store_true", dest="as_json")
    mdn.set_defaults(func=cmd_models_download)

    mi = models_sub.add_parser("import", help="link a local GGUF or folder into the library")
    mi.add_argument("path", help="a .gguf file or a folder containing GGUFs")
    mi.add_argument("--copy", action="store_true",
                    help="copy the file instead of hardlink/symlink")
    _add_models_dir(mi)
    mi.add_argument("--json", action="store_true", dest="as_json")
    mi.set_defaults(func=cmd_models_import)

    mr = models_sub.add_parser("remove", help="delete one GGUF from the library")
    mr.add_argument("file", help="file name or relative path inside the library")
    mr.add_argument("--yes", action="store_true", help="confirm the deletion")
    _add_models_dir(mr)
    mr.add_argument("--json", action="store_true", dest="as_json")
    mr.set_defaults(func=cmd_models_remove)

    stacks = sub.add_parser("stacks", help="saved stacks")
    stacks_sub = stacks.add_subparsers(dest="stacks_command", metavar="<action>",
                                       required=True)
    sl = stacks_sub.add_parser("list", help="list saved stacks")
    _add_stacks_dir(sl)
    sl.add_argument("--json", action="store_true", dest="as_json")
    sl.set_defaults(func=cmd_stacks_list)

    ss = stacks_sub.add_parser("show", help="print one stack as JSON")
    ss.add_argument("stack", help="stack name or path")
    _add_stacks_dir(ss)
    ss.add_argument("--json", action="store_true", dest="as_json",
                    help="accepted for symmetry; show always prints JSON")
    ss.set_defaults(func=cmd_stacks_show)

    sv = stacks_sub.add_parser("validate", help="check a stack without starting it")
    sv.add_argument("stack", help="stack name or path")
    _add_stacks_dir(sv)
    _add_models_dir(sv)
    sv.add_argument("--strict", action="store_true",
                    help="treat warnings as failures")
    sv.add_argument("--json", action="store_true", dest="as_json")
    sv.set_defaults(func=cmd_stacks_validate)

    backends = sub.add_parser("backends", help="server builds declared and installed")
    backends.add_argument("--json", action="store_true", dest="as_json")
    backends.set_defaults(func=cmd_backends)

    doctor = sub.add_parser("doctor", help="check the box before serving")
    _add_models_dir(doctor)
    _add_stacks_dir(doctor)
    doctor.add_argument("--strict", action="store_true",
                        help="treat warnings as failures")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    doctor.set_defaults(func=cmd_doctor)

    config = sub.add_parser("config", help="resolved paths and versions")
    _add_models_dir(config)
    _add_stacks_dir(config)
    config.add_argument("--json", action="store_true", dest="as_json")
    config.set_defaults(func=cmd_config)

    version = sub.add_parser("version", help="print the Hive version")
    version.add_argument("--json", action="store_true", dest="as_json")
    version.set_defaults(func=cmd_version)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help(sys.stderr)
        return 2
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
