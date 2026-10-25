"""Hardware profiles: an explicit, local snapshot with a consent record.

Scanning is deterministic probing, not inference.  It reads the same sysfs and
psutil data as ``hive doctor`` but stores a stable snapshot so a recommendation
can be explained and audited later.  Nothing leaves the machine, and a scan
requires ``consent.hardware`` to be true.
"""

from __future__ import annotations

import copy
import os
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from harness.hardware import hardware_summary
from harness.state_store import SchemaMismatch, StatePaths, JsonStore

CONSENT_KEYS: tuple[str, ...] = ("hardware", "models", "traces", "export")


class ConsentRequired(PermissionError):
    """The caller asked for a scan without the matching consent."""


@dataclass(frozen=True)
class Consent:
    """What the user allowed the profiler to read.

    The E0 profiler only reads hardware today.  The other flags are stored so a
    later model/trace profiler can consume the same consent record without a
    second first-run prompt.
    """

    hardware: bool = False
    models: bool = False
    traces: bool = False
    export: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {key: bool(getattr(self, key)) for key in CONSENT_KEYS}

    @classmethod
    def parse(cls, value: Any) -> "Consent":
        """Accept a dict, a comma list, a single string, or None."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            names = {part.strip().lower() for part in value.split(",") if part.strip()}
            unknown = names - set(CONSENT_KEYS)
            if unknown:
                raise ValueError(
                    f"unknown consent value(s): {', '.join(sorted(unknown))}")
            return cls(**{name: name in names for name in CONSENT_KEYS})
        if isinstance(value, dict):
            unknown = {str(k) for k in value} - set(CONSENT_KEYS)
            if unknown:
                raise ValueError(
                    f"unknown consent key(s): {', '.join(sorted(unknown))}")
            return cls(**{name: bool(value.get(name, False)) for name in CONSENT_KEYS})
        if isinstance(value, Iterable):
            return cls.parse(",".join(str(item) for item in value))
        raise ValueError("consent must be a dict, string, or list of names")


def _cpu_model() -> str:
    """A local CPU label without importing a probing library."""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def _physical_cores() -> int | None:
    try:
        import psutil

        value = psutil.cpu_count(logical=False)
        return int(value) if value else None
    except Exception:  # noqa: BLE001 - a profile may be partial
        return None


def _device_role(device: dict[str, Any], position: int) -> dict[str, Any]:
    """Keep the fields the policy needs, without dumping unrelated probe data."""
    return {
        "index": int(device.get("index", position)),
        "name": str(device.get("name") or f"GPU {position}"),
        "backend": str(device.get("backend") or "unknown"),
        "bdf": device.get("bdf"),
        "memory_gb": device.get("memory_gb"),
        "used_gb": device.get("used_gb"),
        "free_gb": device.get("free_gb"),
        "gtt_total_gb": device.get("gtt_total_gb"),
        "gtt_used_gb": device.get("gtt_used_gb"),
        "display": bool(device.get("display", False)),
        "visible": bool(device.get("visible", True)),
    }


def _confidence(summary: dict[str, Any], gpus: list[dict[str, Any]],
                storage: dict[str, Any]) -> dict[str, str]:
    source = str(summary.get("vram_source") or "")
    gpu_confidence = "high" if source in ("sysfs", "nvidia-smi") else (
        "medium" if gpus else "unknown")
    return {
        "cpu": "high" if summary.get("cpu_count") else "unknown",
        "ram": "high" if summary.get("total_ram_gb") else "unknown",
        "gpus": gpu_confidence,
        "storage": "high" if storage.get("models_free_gb") is not None else "unknown",
    }


def build_profile(summary: dict[str, Any], consent: Consent | dict[str, Any] | str,
                  *, profile_id: str | None = None,
                  now: datetime | None = None) -> dict[str, Any]:
    """Turn a ``hardware_summary()`` dict into a versioned profile payload."""
    if not isinstance(consent, Consent):
        consent = Consent.parse(consent)
    if not consent.hardware:
        raise ConsentRequired(
            "hardware scan needs explicit consent; pass --consent hardware")
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    pid = profile_id or (
        f"local-{stamp.strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:4]}")
    gpus = [
        _device_role(device, position)
        for position, device in enumerate(summary.get("devices") or [])
        if isinstance(device, dict)
    ]
    devices = summary.get("disk") if isinstance(summary.get("disk"), dict) else {}
    models_disk = devices.get("models") if isinstance(devices.get("models"), dict) else {}
    storage = {
        "models_free_gb": models_disk.get("free_gb"),
        "models_total_gb": models_disk.get("total_gb"),
        "paths": devices,
    }
    backends = sorted({
        str(device.get("backend") or "")
        for device in gpus
        if device.get("backend")
    })
    logical = summary.get("cpu_count") or os.cpu_count()
    physical = _physical_cores()
    return {
        "schema_version": 1,
        "profile_id": pid,
        "scanned_at": stamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "consent": consent.as_dict(),
        "scan_scope": ["hardware"] if consent.hardware else [],
        "cpu": {
            "model": _cpu_model(),
            "logical": int(logical) if logical else None,
            "physical": physical,
        },
        "ram": {
            "total_gb": summary.get("total_ram_gb"),
            "available_gb": summary.get("available_ram_gb"),
            "swap_total_gb": summary.get("swap_total_gb"),
            "swap_used_gb": summary.get("swap_used_gb"),
        },
        "gpus": gpus,
        "backends": backends,
        "storage": storage,
        "confidence": _confidence(summary, gpus, storage),
        "source": str(summary.get("vram_source") or "unknown"),
        "visibility": copy.deepcopy(summary.get("visibility") or {}),
    }


class HardwareProfiler:
    """Scan, save, load and delete hardware snapshots under a state root."""

    def __init__(self, state: StatePaths | None = None, *,
                 probe: Callable[[], dict[str, Any]] = hardware_summary,
                 now: Callable[[], datetime] | None = None) -> None:
        self.state = (state or StatePaths.default()).ensure()
        self._probe = probe
        self._now = now or (lambda: datetime.now(timezone.utc))

    def _store(self, profile_id: str) -> JsonStore:
        return JsonStore(self.state.profile_file(profile_id))

    def scan(self, consent: Consent | dict[str, Any] | str | None,
             *, profile_id: str | None = None) -> dict[str, Any]:
        parsed = Consent.parse(consent)
        if not parsed.hardware:
            raise ConsentRequired(
                "hardware scan needs explicit consent; pass --consent hardware")
        summary = self._probe()
        if not isinstance(summary, dict):
            raise ValueError("hardware probe returned a non-object")
        profile = build_profile(summary, parsed, profile_id=profile_id,
                                now=self._now())
        return self._store(profile["profile_id"]).write(profile)

    def get(self, profile_id: str) -> dict[str, Any] | None:
        path = self.state.profile_file(profile_id)
        if not path.is_file():
            return None
        try:
            data = self._store(profile_id).read()
        except SchemaMismatch:
            return None
        if not data.get("profile_id"):
            return None
        return data

    def latest(self) -> dict[str, Any] | None:
        rows = self.list()
        if not rows:
            return None
        return max(rows, key=lambda row: (
            str(row.get("scanned_at") or ""), str(row.get("profile_id") or "")))

    def list(self) -> list[dict[str, Any]]:
        directory = self.state.profiles_dir
        if not directory.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(directory.glob("hardware-*.json")):
            profile_id = path.stem[len("hardware-"):]
            profile = self.get(profile_id)
            if profile is not None:
                rows.append(profile)
        rows.sort(key=lambda row: (
            str(row.get("scanned_at") or ""), str(row.get("profile_id") or "")))
        return rows

    def delete(self, profile_id: str) -> bool:
        return self._store(profile_id).delete()
