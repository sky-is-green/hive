"""Host hardware introspection for fit estimates and GPU selection.

Linux/AMD is sysfs-primary (this host has no ``rocm-smi``/``amdsmi``): per-card
total and **free** VRAM, GTT (the CPU-offload budget on AMD), PCI BDF, driver
and whether a display connector is attached. Per-process VRAM comes from
``/proc/<pid>/fdinfo`` (``drm-total-vram``), which is how the desktop tools
attribute usage.

Deliberately no thermal/power/clock polling: that is real-time telemetry for a
dedicated monitor, not something the harness should spend cycles on.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _optional_psutil():
    """psutil is a base install dependency, but hardware probes stay best-effort."""
    try:
        import psutil
        return psutil
    except Exception:  # noqa: BLE001 - optional at import time
        return None


def parse_visible_indices(value: Optional[str]) -> Optional[list[int]]:
    """Parse ``HIP_VISIBLE_DEVICES``/``CUDA_VISIBLE_DEVICES`` into indices.

    None when unset (all devices visible) or when the value is not a plain
    index list (UUIDs/MIG), so callers fall back to "all visible"."""
    if value is None or str(value).strip() == "":
        return None
    tokens = [t.strip() for t in str(value).split(",") if t.strip() != ""]
    if not tokens:
        return None
    indices: list[int] = []
    for token in tokens:
        if not token.lstrip("-").isdigit():
            return None
        indices.append(int(token))
    return indices


def _gb(value) -> Optional[float]:
    try:
        return round(int(value) / (1024 ** 3), 2)
    except (TypeError, ValueError):
        return None


def linux_amd_devices() -> list[dict]:
    """Enumerate amdgpu cards from sysfs.

    Sorted by PCI BDF ascending — the order ROCm/HIP assigns device indices,
    which is why ``HIP_VISIBLE_DEVICES=1`` selects the card whose BDF sorts
    second (0000:07:00.0 here, not sysfs card0)."""
    devices: list[dict] = []
    for dev in glob.glob("/sys/class/drm/card[0-9]*/device"):
        card = dev.split("/")[-2]

        def read(name: str) -> Optional[str]:
            try:
                with open(f"{dev}/{name}") as handle:
                    return handle.read().strip()
            except OSError:
                return None

        total = read("mem_info_vram_total")
        if not total:
            continue
        bdf = driver = None
        try:
            with open(f"{dev}/uevent") as handle:
                for line in handle:
                    if line.startswith("PCI_SLOT_NAME="):
                        bdf = line.split("=", 1)[1].strip()
                    elif line.startswith("DRIVER="):
                        driver = line.split("=", 1)[1].strip()
        except OSError:
            pass
        display = False
        for status_path in glob.glob(f"/sys/class/drm/{card}-*/status"):
            try:
                with open(status_path) as handle:
                    if handle.read().strip() == "connected":
                        display = True
                        break
            except OSError:
                continue
        if not display:
            # the boot VGA card is the display even when the connector
            # status is not readable (headless over SSH, a KVM)
            boot_vga = read("boot_vga")
            display = boot_vga == "1"
        busy = read("gpu_busy_percent")
        vendor, device_id = read("vendor"), read("device")
        mem = _gb(total)
        used = _gb(read("mem_info_vram_used"))
        devices.append({
            "card": card,
            "bdf": bdf,
            "driver": driver,
            "vendor": vendor,
            "device": device_id,
            "backend": "rocm",
            "name": f"AMD GPU {vendor or ''}:{device_id or ''}".strip(),
            "memory_gb": mem,
            "used_gb": used,
            "free_gb": round(max(0.0, mem - used), 2)
            if mem is not None and used is not None else None,
            "gtt_total_gb": _gb(read("mem_info_gtt_total")),
            "gtt_used_gb": _gb(read("mem_info_gtt_used")),
            "util_pct": int(busy) if busy and busy.isdigit() else None,
            "display": display,
        })
    devices.sort(key=lambda d: d.get("bdf") or "")
    for index, device in enumerate(devices):
        device["index"] = index
    return devices


def non_display_index(devices: list[dict] | None = None) -> Optional[int]:
    """The HIP index of the card that is *not* the display.

    The desktop's card belongs to the desktop: an engine that takes it
    makes the window stutter, and the launch fails outright when the
    display's VRAM is nearly full.  So a launch that does not name a
    card gets the spare one.

    The order is the HIP order (the devices are sorted by BDF, which is
    what ROCm enumerates), so the returned index is what
    ``HIP_VISIBLE_DEVICES`` wants.
    """
    if devices is None:
        try:
            devices = linux_amd_devices()
        except Exception:  # noqa: BLE001 - the probe is best effort
            return None
    for device in devices:
        if not device.get("display"):
            index = device.get("index")
            if index is not None:
                return int(index)
    return None


def _fdinfo_vram_bytes(pid: int) -> int:
    """Sum ``drm-total-vram`` across a process's fds (bytes; fdinfo is KiB)."""
    total_kib = 0
    try:
        for path in glob.glob(f"/proc/{pid}/fdinfo/*"):
            try:
                with open(path) as handle:
                    for line in handle:
                        if line.startswith("drm-total-vram:"):
                            total_kib += int(line.split(":", 1)[1].strip().split()[0])
            except (OSError, ValueError):
                continue
    except Exception:  # noqa: BLE001 - process may vanish
        return 0
    return total_kib * 1024


def gpu_processes() -> list[dict]:
    """Processes holding GPU memory, ranked by VRAM (on-demand; not polled)."""
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return []
    rows: list[dict] = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pid = int(proc.info["pid"])
            used = _fdinfo_vram_bytes(pid)
            if used <= 0:
                continue
            rows.append({"pid": pid, "name": proc.info.get("name") or "",
                         "vram_mb": round(used / (1024 ** 2), 1)})
        except Exception:  # noqa: BLE001 - process vanished
            continue
    rows.sort(key=lambda r: r["vram_mb"], reverse=True)
    return rows



def hardware_summary(repo_root: Path | None = None) -> dict:
    """Host VRAM/RAM summary for fit estimates (nvidia-smi + AMD registry + psutil).

    This lives with the other lightweight hardware probes so serving does not
    need to import the Studio application.  ``repo_root`` is kept for the
    Studio, whose historical repo root points at its workspace.
    """
    psutil = _optional_psutil()
    repo = Path(repo_root) if repo_root is not None else REPO_ROOT
    try:
        vm = psutil.virtual_memory()
        total_ram_gb = round(vm.total / (1024 ** 3), 2)
        available_ram_gb = round(vm.available / (1024 ** 3), 2)
    except Exception:
        total_ram_gb = 8.0
        available_ram_gb = 8.0
    vram_gb: Optional[float] = None
    combined_vram_gb: Optional[float] = None
    devices: list[dict] = []
    # Nvidia
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            timeout=2, text=True, stderr=subprocess.DEVNULL,
            creationflags=_NO_WINDOW,
        )
        for line in out.strip().splitlines():
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                total = round(int(parts[1]) / 1024, 2)
                used = round(int(parts[2]) / 1024, 2)
            except ValueError:
                continue
            vram_gb = (vram_gb or 0) + total  # sum for combined
            devices.append({"backend": "cuda", "name": parts[0],
                            "memory_gb": total, "used_gb": used,
                            "free_gb": round(max(0.0, total - used), 2)})
    except Exception:
        pass
    # AMD fallback via registry + WMI count (avoids duplicate registry entries counting 4x)
    if not devices:
        try:
            import winreg

            per_card_gb = None
            desc_name = None
            base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as k:
                for i in range(100):
                    try:
                        sub = winreg.EnumKey(k, i)
                    except OSError:
                        break
                    try:
                        with winreg.OpenKey(k, sub) as sk:
                            try:
                                desc = winreg.QueryValueEx(sk, "DriverDesc")[0]
                            except OSError:
                                continue
                            if "AMD" not in str(desc) and "Radeon" not in str(desc):
                                continue
                            try:
                                qmem = winreg.QueryValueEx(sk, "HardwareInformation.qwMemorySize")[0]
                                gb = round(int(qmem) / (1024 ** 3), 2)
                                if gb < 1 or gb > 64:
                                    continue
                                per_card_gb = gb
                                desc_name = str(desc)
                                break
                            except OSError:
                                continue
                    except OSError:
                        continue
            if per_card_gb:
                # count physical AMD GPUs via WMI (registry has duplicates)
                count = 1
                try:
                    out = subprocess.check_output(
                        ["powershell", "-Command", "(Get-CimInstance Win32_VideoController | Where-Object {$_.Name -like '*AMD*' -or $_.Name -like '*Radeon*'}).Count"],
                        timeout=2, text=True, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW,
                    )
                    # output may be "2" or blank
                    txt = out.strip().split()[0] if out.strip() else "1"
                    count = int(txt)
                    if count < 1:
                        count = 1
                except Exception:
                    count = 1
                vram_gb = per_card_gb * count
                for _ in range(count):
                    devices.append({"backend": "rocm", "name": desc_name, "memory_gb": per_card_gb})
        except Exception:
            pass
    # Linux fallback (cross-platform)
    if not devices and sys.platform != "win32":
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram"],
                timeout=2, text=True, stderr=subprocess.DEVNULL,
                creationflags=_NO_WINDOW if sys.platform == "win32" else 0,
            )
            import re as _re2

            for m in _re2.finditer(r"(\d{9,})\s*B", out):
                val = int(m.group(1))
                gb = round(val / (1024 ** 3), 2)
                if 1 < gb < 128:
                    vram_gb = (vram_gb or 0) + gb
                    devices.append({"backend": "rocm", "name": "AMD GPU", "memory_gb": gb})
        except Exception:
            pass
        try:
            from harness.hardware import linux_amd_devices
            if not devices:
                devices.extend(linux_amd_devices())
        except Exception:
            pass
    # macOS fallback
    if not devices and sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                timeout=3, text=True, stderr=subprocess.DEVNULL,
            )
            import re as _re3

            for line in out.splitlines():
                if "VRAM" in line and "GB" in line:
                    m = _re3.search(r"(\d+)\s*GB", line)
                    if m:
                        gb = float(m.group(1))
                        if 1 < gb < 128:
                            vram_gb = (vram_gb or 0) + gb
                            devices.append({"backend": "metal", "name": "Apple GPU", "memory_gb": gb})
        except Exception:
            pass
    for index, device in enumerate(devices):
        device.setdefault("index", index)
    visibility_env = ("HIP_VISIBLE_DEVICES"
                      if any(d.get("backend") == "rocm" for d in devices)
                      else "CUDA_VISIBLE_DEVICES")
    visibility_value = os.environ.get("HIP_VISIBLE_DEVICES")
    if visibility_value is None and any(d.get("backend") == "cuda" for d in devices):
        visibility_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_indices = parse_visible_indices(visibility_value)
    for device in devices:
        device["visible"] = (visible_indices is None
                             or device.get("index") in visible_indices)
    visible = [d for d in devices if d.get("visible")]

    def _sum(items, key):
        vals = [d[key] for d in items if d.get(key) is not None]
        return round(sum(vals), 2) if vals else None

    vram_total = _sum(visible, "memory_gb")
    vram_free = _sum(visible, "free_gb")
    combined_vram_gb = _sum(devices, "memory_gb")
    if vram_total is not None:
        vram_gb = vram_total
        if any(d.get("backend") == "cuda" for d in devices):
            source = "nvidia-smi"
        elif any(d.get("backend") == "rocm" for d in devices):
            source = "amd-registry" if os.name == "nt" else "sysfs"
        elif devices:
            source = "sysfs"
        else:
            source = "ram"
        available_gb = vram_free if vram_free is not None else vram_total
    else:
        vram_gb = None
        source = "ram"
        available_gb = total_ram_gb
    try:
        swap = psutil.swap_memory()
        swap_total_gb = round(swap.total / (1024 ** 3), 2)
        swap_used_gb = round(swap.used / (1024 ** 3), 2)
    except Exception:
        swap_total_gb = swap_used_gb = None
    disk = disk_summary({
        "repo": str(repo),
        "models": str(repo / "models" / "gguf"),
        "tmp": "/tmp" if os.name != "nt" else os.environ.get("TEMP", "/tmp"),
    })
    return {
        "cpu_count": os.cpu_count(),
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": available_ram_gb,
        "swap_total_gb": swap_total_gb,
        "swap_used_gb": swap_used_gb,
        "disk": disk,
        "vram_gb": vram_gb,
        "vram_free_gb": vram_free,
        "combined_vram_gb": combined_vram_gb,
        "gtt_total_gb": _sum(visible, "gtt_total_gb"),
        "gtt_used_gb": _sum(visible, "gtt_used_gb"),
        "available_gb": round(available_gb, 2) if available_gb is not None else None,
        "devices": devices,
        "vram_source": source,
        "visibility": {"env": visibility_env if visible_indices is not None else None,
                       "value": visibility_value,
                       "indices": visible_indices},
    }

def disk_summary(paths: dict[str, str]) -> dict:
    """Free/total GB per labelled path, skipping unreadable ones."""
    out: dict[str, dict] = {}
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return out
    for label, path in paths.items():
        try:
            usage = psutil.disk_usage(path)
            out[label] = {"free_gb": round(usage.free / (1024 ** 3), 2),
                          "total_gb": round(usage.total / (1024 ** 3), 2)}
        except Exception:  # noqa: BLE001
            continue
    return out


# ---------------------------------------------------------------------------
# Where does it fit?
# ---------------------------------------------------------------------------
#: GB of KV cache per 1024 tokens, for a big MoE at q8_0 (the FP8 estimate
#: the DSH side uses for its own tiering).  It scales with the KV type.
KV_GB_PER_1K: dict[str, float] = {
    "f16": 0.14, "q8_0": 0.07, "q4_0": 0.035, "": 0.07,
}
#: above this the disk starts to hurt the prompt read
IO_WARN_CTX = 131072
#: the disk is "full" past this share
DISK_FULL_SHARE = 0.8


def fit_plan(model_gb: float, ctx_tokens: int, *, vram_gb: float,
             ram_gb: float, kv_type: str = "q8_0",
             disk_free_gb: Optional[float] = None) -> dict:
    """Where a model and its KV cache fit, and the context that fits.

    Three tiers, in the order the memory is used: the VRAM (the weights
    that fit), the RAM (the rest of the weights plus the KV cache that
    fits), the disk (whatever is left over).  The flags are the two that
    matter to the person launching: the disk will hurt the prompt read, and
    the disk is nearly full.

    The KV size is an estimate (a flat per-token figure for a big MoE,
    scaled by the KV type) - it is a plan, not a measurement.
    """
    model_gb = max(0.0, float(model_gb or 0.0))
    ctx = max(0, int(ctx_tokens or 0))
    per_1k = KV_GB_PER_1K.get((kv_type or "").lower(), KV_GB_PER_1K["q8_0"])
    kv_gb = (ctx / 1024.0) * per_1k

    tier1 = min(model_gb, max(0.0, float(vram_gb or 0.0)))
    remaining = model_gb - tier1
    ram = max(0.0, float(ram_gb or 0.0))
    ram_weights = min(remaining, ram)
    weights_disk = remaining - ram_weights
    ram_left = ram - ram_weights
    ram_kv = min(kv_gb, max(0.0, ram_left))
    kv_disk = kv_gb - ram_kv
    tier2 = ram_weights + ram_kv
    tier3 = weights_disk + kv_disk

    disk_full = False
    if disk_free_gb is not None:
        disk_full = (tier3 > 0.0 and
                     disk_free_gb - tier3 < disk_free_gb * (1.0 - DISK_FULL_SHARE))
    io_warning = kv_disk > 0.0 or ctx > IO_WARN_CTX

    # the largest context whose KV cache still fits in the RAM
    cap = 0
    if per_1k > 0:
        cap = int(max(0.0, ram_left) * 1024.0 / per_1k)
    return {
        "metrics": {
            "total_gb": round(model_gb + kv_gb, 2),
            "model_gb": round(model_gb, 2),
            "kv_gb": round(kv_gb, 2),
            "tier1_vram_gb": round(tier1, 2),
            "tier2_ram_gb": round(tier2, 2),
            "tier3_disk_gb": round(tier3, 2),
        },
        "flags": {
            "io_latency_warning": io_warning,
            "disk_full": disk_full,
            "recommend_cap": cap,
        },
    }
