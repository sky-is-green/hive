"""Small, inspectable state files for the controller and future tool policy.

The controller writes hardware profiles, recommendations and the last applied
settings snapshot.  Those files are user-visible policy data, so they are plain
JSON with a schema version, written atomically under a per-file lock.  A corrupt
file is moved aside instead of being silently replaced; a missing or corrupt
file never turns into an allow-everything default.

This module owns storage only.  It does not know what a recommendation means,
and it does not import FastAPI, torch or the hardware probes.

Layout (under ``harness.paths.state_dir()``)::

    settings/controller.json
    profiles/hardware-<id>.json
    profiles/recommendations-<id>.json
    audit/settings-YYYYMM.jsonl
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from harness import paths

SCHEMA_VERSION = 1
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,96}$")
_REDACT_KEYS = frozenset(
    {
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "set-cookie",
        "password",
        "passwd",
        "secret",
        "client_secret",
    }
)


def file_lock(path: str | Path) -> Iterator[None]:
    """Public per-file lock for legacy settings that have not migrated yet."""
    return _file_lock(Path(path))


class StateError(RuntimeError):
    """A state file could not be read or written safely."""


class StateCorrupt(StateError):
    """The file exists but does not contain a usable JSON object."""


class SchemaMismatch(StateError):
    """The file's schema version is not one this build understands."""


class StateLockError(StateError):
    """The per-file lock could not be acquired."""


def _utc_stamp(compact: bool = False) -> str:
    now = datetime.now(timezone.utc)
    if compact:
        return now.strftime("%Y%m%dT%H%M%S%fZ")
    return now.isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_token(value: str, what: str) -> str:
    text = str(value or "").strip()
    if not _SAFE_TOKEN.match(text):
        raise ValueError(
            f"invalid {what} {value!r}: use letters, digits, '.', '_' or '-'")
    return text


@dataclass(frozen=True)
class StatePaths:
    """Resolve every durable controller path beneath one state root."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())

    @classmethod
    def default(cls) -> "StatePaths":
        return cls(paths.state_dir())

    @property
    def settings_dir(self) -> Path:
        return self.root / "settings"

    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    @property
    def controllers_dir(self) -> Path:
        return self.root / "controllers"

    @property
    def audit_dir(self) -> Path:
        return self.root / "audit"

    @property
    def browser_dir(self) -> Path:
        return self.root / "browser"

    def ensure(self) -> "StatePaths":
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (self.settings_dir, self.profiles_dir,
                          self.controllers_dir, self.audit_dir):
            directory.mkdir(parents=True, exist_ok=True)
            _chmod(directory, 0o700)
        return self

    def settings_file(self, name: str) -> Path:
        return self.settings_dir / f"{_safe_token(name, 'settings file')}.json"

    def profile_file(self, profile_id: str) -> Path:
        return self.profiles_dir / (
            f"hardware-{_safe_token(profile_id, 'profile id')}.json")

    def recommendations_file(self, profile_id: str) -> Path:
        return self.profiles_dir / (
            f"recommendations-{_safe_token(profile_id, 'profile id')}.json")

    def controller_file(self) -> Path:
        return self.settings_file("controller")

    def audit_file(self, month: str | None = None) -> Path:
        token = month or datetime.now(timezone.utc).strftime("%Y%m")
        return self.audit_dir / f"settings-{_safe_token(token, 'audit month')}.jsonl"

    def controller_trace_file(self, month: str | None = None) -> Path:
        token = month or datetime.now(timezone.utc).strftime("%Y%m")
        return self.profiles_dir / (
            f"controller-traces-{_safe_token(token, 'trace month')}.jsonl")


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        # Windows and some filesystems do not expose POSIX modes.  The write is
        # still atomic; the caller's OS profile is the remaining boundary.
        pass


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Exclusive lock on ``path + '.lock'``, held for the with-block."""
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    try:
        handle = open(lock_path, "a+b")
    except OSError as exc:
        raise StateLockError(f"cannot open lock for {path}: {exc}") from exc
    try:
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            raise StateLockError(f"cannot lock {path}: {exc}") from exc
        _chmod(lock_path, 0o600)
        yield
    finally:
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def _fsync_dir(path: Path) -> None:
    """Make a rename durable where the platform supports directory fsync."""
    if os.name == "nt":  # pragma: no cover - Windows has no directory fsync
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod(path.parent, 0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _chmod(path, 0o600)
        _fsync_dir(path.parent)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


class JsonStore:
    """One JSON object, with a schema version and atomic replacement."""

    def __init__(self, path: str | Path, *, schema_version: int = SCHEMA_VERSION,
                 default: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.schema_version = int(schema_version)
        self.default = copy.deepcopy(default or {})

    def read(self, default: dict[str, Any] | None = None,
             *, quarantine: bool = True) -> dict[str, Any]:
        """Read the object, or the safe default when it is absent/corrupt.

        A schema mismatch raises :class:`SchemaMismatch`: a newer file must not
        be silently treated as an old one.  Callers that can safely fall back
        catch that explicitly.
        """
        fallback = self.default if default is None else copy.deepcopy(default)
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return copy.deepcopy(fallback)
        except OSError as exc:
            raise StateError(f"cannot read {self.path}: {exc}") from exc
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            if quarantine:
                self._quarantine()
            return copy.deepcopy(fallback)
        if not isinstance(data, dict):
            if quarantine:
                self._quarantine()
            return copy.deepcopy(fallback)
        version = data.get("schema_version")
        if version != self.schema_version:
            raise SchemaMismatch(
                f"{self.path} has schema_version={version!r}; "
                f"this build expects {self.schema_version}")
        return data

    def read_strict(self) -> dict[str, Any]:
        """Read without a fallback; missing/corrupt files raise."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise StateCorrupt(f"{self.path} does not exist") from exc
        except OSError as exc:
            raise StateError(f"cannot read {self.path}: {exc}") from exc
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise StateCorrupt(f"{self.path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise StateCorrupt(f"{self.path} is not a JSON object")
        if data.get("schema_version") != self.schema_version:
            raise SchemaMismatch(
                f"{self.path} has schema_version={data.get('schema_version')!r}; "
                f"this build expects {self.schema_version}")
        return data

    def write(self, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise TypeError("state data must be a JSON object")
        payload = copy.deepcopy(data)
        version = payload.setdefault("schema_version", self.schema_version)
        if version != self.schema_version:
            raise SchemaMismatch(
                f"refusing to write schema_version={version!r}; "
                f"this build writes {self.schema_version}")
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with _file_lock(self.path):
            _atomic_write(self.path, text)
        return payload

    def update(self, mutator: Callable[[dict[str, Any]], dict[str, Any] | None]
               ) -> dict[str, Any]:
        """Read-modify-write under one lock; the mutator returns the new object."""
        with _file_lock(self.path):
            current = self.read()
            result = mutator(current)
            if result is None:
                result = current
            if not isinstance(result, dict):
                raise TypeError("state mutator must return a JSON object")
            payload = copy.deepcopy(result)
            version = payload.setdefault("schema_version", self.schema_version)
            if version != self.schema_version:
                raise SchemaMismatch(
                    f"refusing to write schema_version={version!r}; "
                    f"this build writes {self.schema_version}")
            _atomic_write(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
            return payload

    def delete(self) -> bool:
        with _file_lock(self.path):
            try:
                self.path.unlink()
            except FileNotFoundError:
                return False
            except OSError as exc:
                raise StateError(f"cannot remove {self.path}: {exc}") from exc
            _fsync_dir(self.path.parent)
            return True

    def _quarantine(self) -> Path | None:
        if not self.path.exists():
            return None
        target = self.path.with_name(
            f"corrupt-{_utc_stamp(compact=True)}-{self.path.name}")
        try:
            os.replace(self.path, target)
            _chmod(target, 0o600)
        except OSError:
            return None
        return target


def redact(value: Any, *, key: str | None = None) -> Any:
    """Remove credentials from an audit payload, recursively."""
    if key is not None and key.strip().lower() in _REDACT_KEYS:
        return "[redacted]"
    if isinstance(value, dict):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str) and len(value) > 4096:
        return value[:4096] + "…[truncated]"
    return value


class AuditLog:
    """Append-only JSONL records; one lock and one fsync per append."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        if not isinstance(record, dict):
            raise TypeError("audit record must be a JSON object")
        data = dict(record)
        data.setdefault("ts", _utc_stamp())
        payload = json.dumps(redact(data), sort_keys=True,
                             ensure_ascii=False) + "\n"
        with _file_lock(self.path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _chmod(self.path.parent, 0o700)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                with os.fdopen(fd, "a", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                raise
            _chmod(self.path, 0o600)
