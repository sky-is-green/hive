"""Records for running ``hive serve`` instances.

A serve is one supervisor process (``hive serve``) plus the engine children it
started.  The record is written once the stack is applied and removed on a
clean exit.  It is deliberately *advisory*: ``hive status`` re-checks the PID
and the command line it recorded, and a stale record is pruned instead of
trusted.  There is no daemon and no database — one JSON file per serve under
``harness_state/serve/`` (``HIVE_STATE_DIR`` overrides the root).

The record is what makes the CLI an alternative to the Studio: a different
process can answer "what is running?", "stop it", and "show its log".
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_DIR = REPO_ROOT / "harness_state"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


# ---------------------------------------------------------------------------
# paths and records
# ---------------------------------------------------------------------------

def state_dir() -> Path:
    """The state root (``HIVE_STATE_DIR`` or ``<repo>/harness_state``)."""
    raw = os.environ.get("HIVE_STATE_DIR") or DEFAULT_STATE_DIR
    return Path(raw).expanduser()


def serve_dir() -> Path:
    path = state_dir() / "serve"
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_name(name: str) -> str:
    text = str(name or "").strip()
    if not _NAME_RE.match(text):
        raise ValueError(
            f"invalid serve name {name!r}: use 1-64 letters, digits, '.', '_' or '-'")
    return text


def record_path(name: str) -> Path:
    return serve_dir() / f"{validate_name(name)}.json"


def log_path(name: str) -> Path:
    return serve_dir() / f"{validate_name(name)}.log"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _coerce_record(data: Any, path: Path) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"serve record {path} is not a JSON object")
    name = validate_name(str(data.get("name") or path.stem))
    try:
        pid = int(data.get("pid") or 0)
    except (TypeError, ValueError):
        raise ValueError(f"serve record {path} has no numeric pid") from None
    if pid <= 0:
        raise ValueError(f"serve record {path} has no valid pid")
    record = dict(data)
    record["name"] = name
    record["pid"] = pid
    record.setdefault("version", 1)
    record.setdefault("started_at", 0.0)
    record.setdefault("tiers", [])
    return record


def write_record(record: dict) -> Path:
    """Write one record atomically and return its path."""
    record = dict(record)
    record["name"] = validate_name(str(record.get("name") or ""))
    record["pid"] = int(record.get("pid") or 0)
    if record["pid"] <= 0:
        raise ValueError("serve record needs a positive pid")
    record["updated_at"] = time.time()
    path = record_path(record["name"])
    _atomic_write(path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def read_record(name: str) -> Optional[dict]:
    """One record by name, or ``None`` when it is not there.

    A corrupt record raises ``ValueError`` rather than disappearing; the
    caller can decide to prune it.  ``list_records`` skips corrupt files so
    one hand-edited record cannot take the whole status view down.
    """
    path = record_path(name)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"cannot read serve record {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"serve record {path} is not valid JSON: {exc}") from exc
    return _coerce_record(data, path)


def list_records() -> list[dict]:
    """Every valid record, oldest first.  Corrupt files are skipped."""
    directory = serve_dir()
    records: list[dict] = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.name):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            record = _coerce_record(data, path)
        except (OSError, ValueError):
            continue
        record["record_path"] = str(path)
        records.append(record)
    records.sort(key=lambda r: float(r.get("started_at") or 0.0))
    return records


def remove_record(name: str) -> bool:
    path = record_path(name)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------

def pid_alive(pid: int) -> bool:
    """True when ``pid`` names a live process.

    On Windows ``os.kill(pid, 0)`` is *not* a probe: any signal other than
    CTRL_C/CTRL_BREAK terminates the process.  Use OpenProcess there.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:  # noqa: BLE001 - a failed probe is not alive
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    # A zombie still answers signal 0.  It is not a running serve, and waiting
    # for it to disappear would turn every stop into a timeout when the
    # detached parent is still alive (pytest, a wrapper script, a supervisor).
    if _pid_state(pid) == "Z":
        return False
    return True


def _pid_state(pid: int) -> str:
    """Linux process state letter (``R``/``S``/``Z``...), or ``''``."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ""
    end = raw.rfind(")")
    return raw[end + 2:end + 3] if end >= 0 else ""


def pid_cmdline(pid: int) -> str:
    """The process command line, best effort (``''`` where unavailable)."""
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, ValueError):
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def pid_matches(record: dict) -> bool:
    """Whether the live PID still looks like the serve that wrote the record.

    PID reuse is rare but not impossible; a foreground serve's command line is
    the console script path, so the exact recorded line is the fingerprint.
    On platforms without ``/proc`` the check degrades to "alive".
    """
    pid = int(record.get("pid") or 0)
    if not pid_alive(pid):
        return False
    stored = str(record.get("cmdline") or "").strip()
    if not stored:
        return True
    current = pid_cmdline(pid)
    return not current or current == stored


def terminate(pid: int, *, force: bool = False,
              timeout: float = 15.0) -> tuple[bool, str]:
    """Ask one serve supervisor to stop; force only after the timeout.

    On POSIX the child is a session leader, so ``killpg`` also reaches the
    engine processes if the supervisor cannot handle SIGTERM.  On Windows
    ``taskkill /T`` does the same job.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return True, "no pid"
    if pid <= 0 or not pid_alive(pid):
        return True, "already stopped"
    if os.name == "nt":
        return _terminate_windows(pid, force=force, timeout=timeout)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True, "already stopped"
    except PermissionError:
        return False, f"permission denied for pid {pid}"
    deadline = time.monotonic() + max(0.5, float(timeout))
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True, "stopped"
        time.sleep(0.2)
    if not force:
        return False, f"pid {pid} still running after {timeout:g}s (use --force)"
    try:
        pgid = os.getpgid(pid)
        if pgid == pid:
            # a detached serve is its own session/process group; take the
            # engine children with the supervisor
            os.killpg(pgid, signal.SIGKILL)
        else:
            # a foreground serve shares the terminal's group: never kill the
            # shell that started it
            os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True, "killed"
        time.sleep(0.1)
    return False, f"pid {pid} survived SIGKILL"


def _terminate_windows(pid: int, *, force: bool, timeout: float) -> tuple[bool, str]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    attempts = [["taskkill", "/PID", str(pid), "/T"]]
    if force:
        attempts.append(["taskkill", "/PID", str(pid), "/T", "/F"])
    for argv in attempts:
        try:
            subprocess.run(argv, check=False, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, creationflags=flags)
        except OSError as exc:
            return False, f"taskkill failed: {exc}"
        deadline = time.monotonic() + max(0.5, float(timeout))
        while time.monotonic() < deadline:
            if not pid_alive(pid):
                return True, "stopped"
            time.sleep(0.2)
    return False, f"pid {pid} still running (use --force)"


def spawn(argv: list[str], log_path: Path) -> subprocess.Popen:
    """Start one detached supervisor, with its output in ``log_path``."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "ab")  # noqa: SIM115 - the child owns the handle
    env = {**os.environ, "HIVE_SERVE_CHILD": "1", "PYTHONUNBUFFERED": "1"}
    # A source checkout can be launched from any cwd; the child needs the same
    # import path the parent had (the editable install already covers this,
    # but the explicit path keeps `python -m harness.cli` working in CI).
    root = str(REPO_ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        [root] + [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part])
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": handle,
        "stderr": subprocess.STDOUT,
        "env": env,
        "cwd": os.getcwd(),
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(list(argv), **kwargs)
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# small helpers shared by the CLI
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def tail_lines(path: str | Path, count: int = 80) -> list[str]:
    """The last ``count`` lines of a file (all of it when ``count <= 0``)."""
    target = Path(path)
    with target.open("r", encoding="utf-8", errors="replace") as handle:
        if count <= 0:
            return [line.rstrip("\n") for line in handle]
        return [line.rstrip("\n") for line in deque(handle, maxlen=count)]
