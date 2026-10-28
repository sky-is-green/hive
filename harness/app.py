"""The harness sidecar FastAPI application.

State model
-----------
- One ``Splinter`` instance per conversation_id (fresh store + comb per
  conversation — per-conversation isolation is mandatory).
  Instances are created lazily on the first turn and dropped by /v1/splinter/reset.
- Conversations persist to ``state_dir`` (default ./harness_state, one atomic
  JSON per conversation using the same store serialization as the benchmark's
  checkpoint/resume) and reload lazily on first touch after a restart, so the
  splinter survives sidecar restarts. /v1/splinter/reset deletes memory AND disk.
- One shared ultra-small drone across conversations (a per-conversation encoder
  would multiply VRAM/RAM for nothing); inference is read-only.
- Per-conversation locks serialize turns within a conversation; different
  conversations may proceed in parallel. Generation calls are blocking
  (streaming is a v2 concern) — sync endpoints run in FastAPI's threadpool.
- Providers: loaded from providers.local.json at startup (or --providers-file),
  replaceable at runtime via POST /v1/provider/config.

Secrets: api_key values are never echoed back (masked as "***") and are only
written to the local providers file; NDJSON event logs redact separately.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

import psutil
import requests
import anyio
import asyncio
from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
import queue

from harness.activity import ActivityError, ActivityHub
from harness.agent import DshAgentService
from harness.commands import ConsoleCommands
from harness.commands import local_shell_command
from harness.studio_settings import load_settings as load_studio_settings
from harness.studio_settings import save_settings as save_studio_settings
from harness.trainer import (
    draft_candidate,
    evaluate_candidate,
    mine_evidence,
    summarize_evidence,)

from backend.engines import (
    EngineProfile,
    EngineRegistry,
    engines_path,
    load_engines,
    save_engines,
)
from backend.openai_compat import OpenAICompatBackend
from backend.providers import (
    MASK,
    Provider,
    ProviderRegistry,
    backend_kwargs,
    load_registry,
    providers_path,
    save_registry,
)
from cortex.config import SplinterConfig
from cortex.splinter import Splinter
from experiments.model_probe import _list_models, probe_model
from harness.hardware import (disk_summary, non_display_index,
                               parse_visible_indices)
from harness.models import drain_slots, page_cache_share, prefetch_weights
from harness.models import LlamaServerManager
from harness.cascade import api as cascade_api
from harness.reports import (
    render_report_page,
    render_runs_page,
    render_server_page,
    render_cascade_page,
    resolve_run_dir,
)
from harness.stack import (
    StackManager,
    create_router as create_stack_router,
    list_stacks,
)
from harness.stack_runtime import default_base_config, generate_runtime_config
from harness.training import (
    ENGINE_HIVE_TERNARY,
    build_run_report,
    engine_catalog,
    ternary_python,
    ternary_runner,
    launch as launch_training,
    run_status as training_run_status,
    write_run_report,
)
from harness.ui import invocation_cards, stack_tab
from logs.event_logger import EventLogger
from retention.store import ContextStore
try:  # current sibling layout: <home>/splinter
    from splinter.server import create_app as create_splinter_app
except ModuleNotFoundError as exc:  # legacy pre-rename checkout: <home>/strata
    if (exc.name or "").split(".")[0] != "splinter":
        raise
    from strata.server import create_app as create_splinter_app


def _list_runs(runs_root: Path) -> list[dict]:
    """Available run bundles under runs_root (newest first)."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    entries = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime,
                        reverse=True):
        if not child.is_dir():
            continue
        report_path = child / "run_report.json"
        kind = "benchmark"
        if report_path.is_file():
            try:
                kind = json.loads(report_path.read_text(encoding="utf-8")).get("kind") or kind
            except (OSError, ValueError):
                pass
        entries.append({
            "name": child.name,
            "has_report": report_path.is_file(),
            "kind": kind,
            "modified": datetime.fromtimestamp(child.stat().st_mtime)
            .strftime("%Y-%m-%d %H:%M:%S"),
        })
    return entries

REPO_ROOT = Path(__file__).resolve().parents[2]
# Splinter mode (AFK) canonical state - workspace-level so all projects share one source.
MODE_FILE = Path(os.environ.get("HIVE_MODE_FILE", str(Path(REPO_ROOT).parent / "SPLINTER-MODE.json")))
RESEARCH_QUEUE = Path(os.environ.get(
    "HIVE_RESEARCH_QUEUE", str(Path(REPO_ROOT).parent / "RESEARCH-QUEUE.md")))
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_PORT = 8765


def dsh_session_root() -> Path:
    """Where DSH persists session logs.

    A deployment must not write under the read-only application tree, so the
    unit sets ``HIVE_DSH_SESSION_ROOT`` to the AI user's ``dsh/sessions``.
    A source checkout keeps the historical ``harness_state/dsh_sessions``.
    """
    override = os.environ.get("HIVE_DSH_SESSION_ROOT", "").strip()
    return (Path(override).expanduser() if override
            else REPO_ROOT / "harness_state" / "dsh_sessions")

# generate_data flags the sidecar may forward (whitelist: no arbitrary CLI).
PROTOCOL_FLAGS_INT = {"max_convs": "--max-convs", "max_turns": "--max-turns",
                      "checkpoint_every": "--checkpoint-every"}
PROTOCOL_FLAGS_STR = {"model": "--model", "base_url": "--base-url",
                      "provider": "--provider", "conversations": "--conversations"}
PROTOCOL_FLAGS_BOOL = {"protocol": "--protocol", "baselines": "--baselines",
                       "no_thinking": "--no-thinking"}

_popen = subprocess.Popen  # module-level so tests can intercept
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # suppress console flash for polling helpers

# HTML consoles change with the code; never let a browser cache them.
_NO_STORE = {"Cache-Control": "no-store"}

# ---------------------------------------------------------------------------
# Stack tab page shell (T44).  The fragments in harness/ui/* are pure
# renderers; this shell inlines them and their module-symbol CSS, and wires the
# data-action buttons to the §B3 routes.  Kept as plain strings so the fragment
# CSS can carry braces without format-string escaping.
# ---------------------------------------------------------------------------
_STACK_PAGE_SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stacks — Splinter Studio</title>
<style>
:root { color-scheme: dark; }
body { margin: 0; padding: 1.25rem; background: #141100; color: #FFDD00;
       font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
a { color: #FFDD00; }
.stk-page { max-width: 76rem; margin: 0 auto; }
.stk-page .stk-back { display: inline-block; margin-bottom: .8rem; font-size: .86rem; }
__STACK_CSS__
__CARDS_CSS__
</style>
</head>
<body>
<div class="stk-page">
<a class="stk-back" href="/server">&larr; Studio</a>
__STACK_TAB__
<section>
<div class="stk-head"><h2 style="margin-top:0">Invocations</h2>
<span class="stk-note">tier dispatches — harness.agent_events / invocation_cards (T39/T42)</span>
</div>
__CARDS__
</section>
</div>
<script>
__SCRIPT__
</script>
</body>
</html>
"""

_STACK_PAGE_SCRIPT = """(function () {
  "use strict";
  function msg(text) {
    var el = document.getElementById("stk-msg-main");
    if (el) { el.textContent = text; }
  }
  function call(method, path, body) {
    var opts = { method: method, headers: { "Accept": "application/json" } };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(path, opts).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (data) {
        if (!response.ok) {
          throw new Error(data.detail || (path + ": HTTP " + response.status));
        }
        return data;
      });
    });
  }
  function stackName() {
    var el = document.getElementById("stk-name");
    return el && el.value ? el.value.trim() : "";
  }
  function reloadSoon() {
    window.setTimeout(function () { window.location.reload(); }, 600);
  }
  document.addEventListener("click", function (event) {
    var target = event.target;
    var button = target && target.closest ? target.closest("[data-action]") : null;
    if (!button) { return; }
    var action = button.getAttribute("data-action");
    var stack = button.getAttribute("data-stack") || stackName();
    if (!stack && action !== "stack-unload") {
      msg("name a stack first");
      return;
    }
    event.preventDefault();
    var run;
    if (action === "stack-load") {
      run = call("GET", "/v1/stacks/" + encodeURIComponent(stack)).then(function (data) {
        var input = document.getElementById("stk-name");
        if (input) { input.value = data.name || stack; }
        msg("loaded " + stack + " — review fields, then Validate or Apply");
      });
    } else if (action === "stack-delete") {
      run = call("DELETE", "/v1/stacks/" + encodeURIComponent(stack)).then(reloadSoon);
    } else if (action === "stack-validate") {
      run = call("POST", "/v1/stacks/" + encodeURIComponent(stack) + "/validate").then(function (data) {
        var cards = (data.per_card || []).map(function (card) {
          return card.card + ": " + card.total + "/" + card.budget + " GiB";
        }).join(" · ");
        msg("validate " + (data.ok ? "OK" : "REFUSED") + (cards ? " — " + cards : ""));
      });
    } else if (action === "stack-apply") {
      run = call("POST", "/v1/stacks/" + encodeURIComponent(stack) + "/apply").then(function (data) {
        msg("applied " + stack + " — " + (data.tiers || []).length + " tier(s)");
        reloadSoon();
      });
    } else if (action === "stack-unload") {
      run = call("POST", "/v1/stacks/" + encodeURIComponent(stack) + "/unload").then(function () {
        msg("unloaded");
        reloadSoon();
      });
    } else {
      return;
    }
    run.catch(function (error) { msg(String(error.message || error)); });
  });
})();
"""


def _render_stack_page(tab_html: str, cards_html: str, stack_css: str,
                       cards_css: str) -> str:
    """The ``/stacks`` page: both fragments, their CSS, and the action script."""
    page = _STACK_PAGE_SHELL
    for token, value in (
        ("__STACK_CSS__", stack_css),
        ("__CARDS_CSS__", cards_css),
        ("__STACK_TAB__", tab_html),
        ("__CARDS__", cards_html),
        ("__SCRIPT__", _STACK_PAGE_SCRIPT),
    ):
        page = page.replace(token, value)
    return page

def _hardware_summary() -> dict:
    """Host summary through the lightweight hardware module (no Studio import)."""
    from harness.hardware import hardware_summary

    return hardware_summary(repo_root=REPO_ROOT)

def _read_gguf_metadata(path: Path) -> dict:
    """Best-effort GGUF header parse for auto-preset (block_count, context, etc).

    Reads the GGUF magic + version + metadata KV section and extracts a handful
    of known keys (general.architecture, *block_count, *context_length,
    *embedding_length) without requiring a full gguf library. Missing or
    unreadable files return {}. This is the gguf-metadata source the Auto
    button combines with GET /v1/models/local size_gb and GET /v1/server/status
    hardware.
    """
    import struct

    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                return {}
            ver = struct.unpack("<I", fh.read(4))[0]
            # v3 uses 64-bit counts, v1/v2 32-bit; try 64 then fallback
            pos = fh.tell()
            try:
                tc = struct.unpack("<Q", fh.read(8))[0]
                mc = struct.unpack("<Q", fh.read(8))[0]
                # sanity: metadata count should be reasonable (< 10000)
                if mc > 10000:
                    raise ValueError("implausible")
            except Exception:
                fh.seek(pos)
                tc = struct.unpack("<I", fh.read(4))[0]
                mc = struct.unpack("<I", fh.read(4))[0]
            out: dict = {}
            for _ in range(int(mc)):
                try:
                    klen = struct.unpack("<Q" if ver >= 3 else "<I", fh.read(8 if ver >= 3 else 4))[0]
                except Exception:
                    break
                if klen > 500:
                    break
                key = fh.read(int(klen)).decode("utf-8", errors="ignore")
                try:
                    ktype = struct.unpack("<I", fh.read(4))[0]
                except Exception:
                    break
                # we care about a few string/uint32/uint64 keys
                try:
                    if ktype == 0:  # uint8
                        val = struct.unpack("<B", fh.read(1))[0]
                    elif ktype == 1:  # int8
                        val = struct.unpack("<b", fh.read(1))[0]
                    elif ktype == 2:  # uint16
                        val = struct.unpack("<H", fh.read(2))[0]
                    elif ktype == 3:  # int16
                        val = struct.unpack("<h", fh.read(2))[0]
                    elif ktype == 4:  # uint32
                        val = struct.unpack("<I", fh.read(4))[0]
                    elif ktype == 5:  # int32
                        val = struct.unpack("<i", fh.read(4))[0]
                    elif ktype == 6:  # float32
                        val = struct.unpack("<f", fh.read(4))[0]
                    elif ktype == 7:  # bool
                        val = bool(struct.unpack("<B", fh.read(1))[0])
                    elif ktype == 8:  # string
                        slen = struct.unpack("<Q" if ver >= 3 else "<I", fh.read(8 if ver >= 3 else 4))[0]
                        val = fh.read(int(slen)).decode("utf-8", errors="ignore")
                    elif ktype == 9:  # array
                        atype = struct.unpack("<I", fh.read(4))[0]
                        alen = struct.unpack("<Q" if ver >= 3 else "<I", fh.read(8 if ver >= 3 else 4))[0]
                        # skip arrays — not needed for presets
                        if atype == 8:  # string array
                            for __ in range(int(alen)):
                                sl = struct.unpack("<Q" if ver >= 3 else "<I", fh.read(8 if ver >= 3 else 4))[0]
                                fh.read(int(sl))
                            val = f"<array:{alen}>"
                        else:
                            size = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(atype, 1)
                            fh.read(int(alen)*size)
                            val = f"<array:{alen}>"
                    elif ktype == 10:  # uint64
                        val = struct.unpack("<Q", fh.read(8))[0]
                    elif ktype == 11:  # int64
                        val = struct.unpack("<q", fh.read(8))[0]
                    elif ktype == 12:  # float64
                        val = struct.unpack("<d", fh.read(8))[0]
                    else:
                        break
                except Exception:
                    break
                # keep only interesting keys
                if any(key.endswith(s) for s in (".block_count", ".context_length", ".embedding_length", ".feed_forward_length")) \
                   or key in ("general.architecture", "general.name", "general.parameter_count", "general.quantization_version"):
                    out[key] = val
                # early exit after we have architecture + block_count
                if "general.architecture" in out and any(k.endswith(".block_count") for k in out):
                    # keep reading a few more but not the whole file
                    if len(out) > 12:
                        break
            return out
    except Exception:
        return {}


def _auto_preset_load_options(size_gb: float, hardware_gb: float, file_name: str, gguf_meta: dict) -> dict:
    """Compute Auto preset load_options from hardware + model size + gguf-metadata.

    Matches the UI contract: qwen3-4b on 8GB → gpu_layers 28 + 8k ctx,
    qwen3-32b on 8GB → 12 layers + 4k, larger VRAM → larger offload/ctx.
    Other load options (threads, flash_attn, kv quant) are tuned with context.
    """
    name = (file_name or "").lower()
    meta = gguf_meta or {}
    # parameter count from gguf if available
    params_b = None
    for k in ("general.parameter_count", "general.parameter_count", "parameter_count"):
        if k in meta:
            try:
                params_b = float(meta[k]) / 1e9
                break
            except Exception:
                pass
    # block_count → total layers
    block_count = None
    for k, v in meta.items():
        if k.endswith(".block_count"):
            try:
                block_count = int(v)
                break
            except Exception:
                pass
    est_layers = block_count
    if not est_layers:
        if "32b" in name or "30b" in name or (params_b and params_b >= 30) or size_gb > 15:
            est_layers = 62
        elif "14b" in name or "13b" in name or (params_b and params_b >= 13):
            est_layers = 40
        elif "7b" in name or "8b" in name or (params_b and params_b >= 7):
            est_layers = 32
        elif "4b" in name or "3b" in name or "qwen3-4b" in name or (params_b and params_b >= 3):
            est_layers = 36
        else:
            est_layers = 32
    is_32 = "32b" in name or "30b" in name or (params_b and params_b >= 30) or size_gb > 15
    is_4 = "4b" in name or "3b" in name or "qwen3-4b" in name or (params_b is not None and 3 <= params_b < 6) or (2 <= size_gb < 6)
    avail = float(hardware_gb) if hardware_gb else 8.0
    if is_32:
        if avail <= 9:
            gpu_layers, ctx = 12, 4096
        elif avail <= 16:
            gpu_layers, ctx = 20, 8192
        elif avail <= 24:
            gpu_layers, ctx = min(40, est_layers), 8192
        else:
            gpu_layers, ctx = 999, 16384
    elif is_4:
        if avail <= 9:
            gpu_layers, ctx = 28, 8192
        elif avail <= 16:
            gpu_layers, ctx = 999, 16384
        else:
            gpu_layers, ctx = 999, 32768
    else:
        if size_gb + 2.0 <= avail * 0.9:
            gpu_layers, ctx = 999, 8192
        elif size_gb + 1.0 <= avail * 1.1:
            gpu_layers, ctx = min(28, est_layers), 8192
        else:
            gpu_layers, ctx = min(12, est_layers), 4096
    if gpu_layers != 999:
        gpu_layers = min(gpu_layers, est_layers)
    out = {"gpu_layers": gpu_layers, "context": ctx, "ctx_size": ctx}
    # advisory extras
    out["flash_attn"] = ctx >= 8192
    if ctx > 16384:
        out["cache_type_k"] = "q8_0"
        out["cache_type_v"] = "q8_0"
    return out


def _setup_tier(context_tokens: int = 32768, dual: bool = False, vhdx: str | None = None, model_gb: float | None = None) -> dict:
    """Tiered allocation — model-agnostic, no hardcoded 104.

    - MODEL_GB: detected from actual GGUFs in /mnt/dsh_storage/models or passed model_gb, not DeepSeek assumption
    - T3 spill: overflow for that model, with estimated speeds from system (no fallback)
    - cap: floor((VRAM+RAM+driveFree - MODEL)/KV) without artificial 131k clamp — shows real max
    """
    # Model size — try passed, then scan actual GGUFs, else 104 fallback
    if model_gb is not None:
        try:
            MODEL_GB = float(model_gb)
        except Exception:
            MODEL_GB = None
    else:
        MODEL_GB = None
    if MODEL_GB is None:
        try:
            out = subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", "du -sb /mnt/dsh_storage/models 2>&1 | cut -f1"],
                timeout=3, text=True, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW,
            )
            total_bytes = int(out.strip().split()[0])
            gb = total_bytes / (1024 ** 3)
            if gb >= 1:
                MODEL_GB = gb
        except Exception:
            pass
    if MODEL_GB is None:
        try:
            p = Path(r"\\wsl.localhost\Ubuntu\mnt\dsh_storage\models")
            total = 0
            for f in p.glob("*.gguf"):
                try:
                    total += f.stat().st_size
                except Exception:
                    continue
            if total:
                MODEL_GB = total / (1024 ** 3)
        except Exception:
            pass
    if MODEL_GB is None:
        MODEL_GB = 104.0  # fallback only if no models yet
    try:
        hw = _hardware_summary()
        vram_detected = hw.get("combined_vram_gb") or hw.get("vram_gb")
        VRAM = float(vram_detected) if vram_detected else (40.0 if dual else 20.0)
        WSL_RAM = float(hw.get("total_ram_gb") or hw.get("available_ram_gb") or 24.0)
    except Exception:
        VRAM = 40.0 if dual else 20.0
        WSL_RAM = 24.0
    vhdx_path = vhdx or os.environ.get("VHDX_PATH", r"E:\dsh_storage.vhdx")
    drive = os.path.splitdrive(vhdx_path)[0] or os.path.splitdrive(os.environ.get("VHDX_PATH", r"E:\dsh_storage.vhdx"))[0]
    if not drive:
        drive = "E:"
    try:
        du = psutil.disk_usage(drive + "\\" if not drive.endswith("\\") else drive)
        drive_total_gb = du.total / (1024 ** 3)
        drive_free_gb = du.free / (1024 ** 3)
        drive_percent = du.percent
    except Exception:
        drive_total_gb = 1000.0
        drive_free_gb = 500.0
        drive_percent = 0
    KV_PER_TOKEN_GB = 0.07 / 1024.0
    kv = context_tokens * KV_PER_TOKEN_GB
    tier1 = min(MODEL_GB, VRAM)
    rem = MODEL_GB - tier1
    ram_w = min(rem, WSL_RAM)
    w_nvme = rem - ram_w
    ram_left = WSL_RAM - ram_w
    ram_kv = min(kv, max(0, ram_left))
    kv_nvme = kv - ram_kv
    tier3 = w_nvme + kv_nvme
    tier2 = ram_w + ram_kv
    # cap without artificial clamp — real leftover
    leftover = VRAM + WSL_RAM + drive_free_gb - MODEL_GB
    cap = int(leftover / KV_PER_TOKEN_GB) if leftover > 0 else 0
    # speeds — no hardcoded fallback, detect accurately
    VRAM_BW = None
    RAM_BW = None
    NVME_BW = None
    try:
        out = subprocess.check_output(
            ["powershell", "-Command", "Get-CimInstance Win32_PhysicalMemory | Select-Object -ExpandProperty Speed"],
            timeout=2, text=True, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW,
        )
        speeds = [int(s) for s in out.split() if s.strip().isdigit()]
        if speeds:
            avg = sum(speeds) / len(speeds)
            RAM_BW = round(max(30, min(90, avg * 0.016)), 1)
    except Exception:
        pass
    if RAM_BW is None and sys.platform != "win32":
        try:
            out = subprocess.check_output(
                ["dmidecode", "--type", "memory"], timeout=2, text=True, stderr=subprocess.DEVNULL
            )
            import re as _re

            m = _re.search(r"Speed:\s*(\d+)\s*MT/s", out)
            if m:
                RAM_BW = round(max(30, min(90, int(m.group(1)) * 0.016)), 1)
        except Exception:
            pass
    try:
        out = subprocess.check_output(
            ["powershell", "-Command", "Get-PhysicalDisk | Where-Object BusType -eq 'NVMe' | Select-Object -ExpandProperty FriendlyName"],
            timeout=2, text=True, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW,
        )
        if out.strip():
            if "Gen5" in out or "PCIe5" in out:
                NVME_BW = 14.0
            elif "Gen3" in out:
                NVME_BW = 3.5
            else:
                NVME_BW = 7.5
    except Exception:
        pass
    if NVME_BW is None and sys.platform != "win32":
        try:
            out2 = subprocess.check_output(["lsblk", "-d", "-o", "NAME,TRAN"], timeout=2, text=True, stderr=subprocess.DEVNULL)
            if "nvme" in out2.lower():
                NVME_BW = 7.5
        except Exception:
            pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.mem", "--format=csv,noheader,nounits"],
            timeout=2, text=True, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW,
        )
        if out.strip():
            VRAM_BW = 1000.0
    except Exception:
        pass
    # AMD VRAM bandwidth — detect via rocm-smi or known GPU table
    if VRAM_BW is None:
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showclocks"],
                timeout=2, text=True, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW if sys.platform == "win32" else 0,
            )
            if out.strip():
                VRAM_BW = 800.0
        except Exception:
            pass
        if VRAM_BW is None:
            try:
                hw2 = _hardware_summary()
                for d in hw2.get("devices", []):
                    name = (d.get("name") or "").lower()
                    if "7900 xtx" in name:
                        VRAM_BW = 960.0
                        break
                    elif "7900 xt" in name:
                        VRAM_BW = 800.0
                        break
                    elif "7900" in name:
                        VRAM_BW = 800.0
                        break
                    elif "7800" in name:
                        VRAM_BW = 520.0
                        break
                    elif "6800" in name or "6900" in name:
                        VRAM_BW = 512.0
                        break
                if VRAM_BW is None and any("amd" in (d.get("name") or "").lower() or "radeon" in (d.get("name") or "").lower() for d in hw2.get("devices", [])):
                    VRAM_BW = 800.0
            except Exception:
                pass
    # macOS Apple Silicon — unified memory, no discrete VRAM
    if VRAM_BW is None and sys.platform == "darwin":
        try:
            out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], timeout=3, text=True, stderr=subprocess.DEVNULL)
            low = out.lower()
            if "m4 max" in low or "m3 max" in low or "m1 max" in low or "m2 max" in low:
                VRAM_BW = 400.0
            elif "m4 pro" in low or "m3 pro" in low or "m2 pro" in low or "m1 pro" in low:
                VRAM_BW = 200.0
            elif "m4" in low:
                VRAM_BW = 120.0
            elif "m3" in low:
                VRAM_BW = 100.0
            elif "m2" in low:
                VRAM_BW = 100.0
            elif "m1" in low:
                VRAM_BW = 68.0
            else:
                VRAM_BW = 100.0
        except Exception:
            pass
        if VRAM_BW is None and RAM_BW is not None:
            VRAM_BW = RAM_BW  # unified, same as RAM
    if VRAM_BW is not None and RAM_BW is not None and NVME_BW is not None:
        total_for_bw = tier1 + tier2 + tier3
        weighted_bw = (tier1 * VRAM_BW + tier2 * RAM_BW + tier3 * NVME_BW) / total_for_bw if total_for_bw else None
        weighted_bw = round(weighted_bw, 1) if weighted_bw is not None else None
    else:
        weighted_bw = None
    free_after_spill = drive_free_gb - tier3
    io_warn = bool(tier3 > 10)
    disk_full = bool(drive_percent > 95 or free_after_spill < 10)
    return {
        "metrics": {
            "tier1VramGb": round(tier1, 2),
            "tier2RamGb": round(tier2, 2),
            "tier3NvmeGb": round(tier3, 2),
            "driveTotalGb": round(drive_total_gb, 1),
            "driveFreeGb": round(drive_free_gb, 1),
            "freeAfterSpillGb": round(free_after_spill, 1),
            "estVramBw": VRAM_BW,
            "estRamBw": RAM_BW,
            "estNvmeBw": NVME_BW,
            "estEffectiveBw": weighted_bw,
            "modelGb": round(MODEL_GB, 2),
        },
        "flags": {
            "ioLatencyWarning": io_warn,
            "diskFull": disk_full,
            "recommendCap": cap,
            "drivePercent": drive_percent,
        },
    }


def _ensure_vhdx(vhdx: str, size_gb: int = 250) -> dict:
    """Create VHDX at vhdx if missing (first-time user). Returns status."""
    if os.path.exists(vhdx):
        return {"ok": True, "already": True, "path": vhdx}
    # Ensure parent drive exists and has space
    parent = os.path.dirname(vhdx) or "."
    try:
        os.makedirs(parent, exist_ok=True)
    except Exception as exc:
        return {"ok": False, "error": f"cannot create {parent}: {exc}"}
    # Try New-VHD (Hyper-V) first, then fsutil fallback (sparse file, then will be formatted as ext4)
    size_bytes = size_gb * 1024 * 1024 * 1024
    # No free-space gate: VHDX is dynamic/sparse (250GB max, initially small) and
    # is formatted to ext4 on first mount. Host free space is checked only if
    # the underlying OS call fails.
    # Try New-VHD (requires Hyper-V)
    try:
        out = subprocess.check_output(
            ["powershell", "-Command", f"New-VHD -Path '{vhdx}' -SizeBytes {size_bytes} -Dynamic -ErrorAction Stop | Out-String"],
            text=True, timeout=30, stderr=subprocess.STDOUT,
        )
        return {"ok": True, "created": True, "path": vhdx, "output": out[:400], "method": "New-VHD"}
    except subprocess.CalledProcessError as exc:
        # Fallback: diskpart create vdisk (works without Hyper-V)
        try:
            script = f"create vdisk file=\"{vhdx}\" maximum={size_gb * 1024} type=expandable\nselect vdisk file=\"{vhdx}\"\nattach vdisk\ncreate partition primary\nformat fs=ntfs quick label=dsh_storage\ndetach vdisk\nexit\n"
            # Use diskpart to create a VHDX, then we will reformat to ext4 in WSL
            with open(os.path.join(os.environ.get("TEMP", "."), "dsh_create_vhdx.txt"), "w", encoding="utf-8") as fh:
                fh.write(script)
                tmp = fh.name
            out2 = subprocess.check_output(
                ["diskpart", "/s", tmp],
                text=True, timeout=30, stderr=subprocess.STDOUT,
            )
            # diskpart creates NTFS, but WSL bare mount will reformat to ext4 on first mount if needed (bootstrap does mkfs.ext4)
            return {"ok": True, "created": True, "path": vhdx, "output": out2[:400], "method": "diskpart"}
        except Exception as exc2:
            # Last fallback: fsutil sparse file (will be formatted as ext4 on mount)
            try:
                subprocess.check_output(["fsutil", "file", "createnew", vhdx, str(size_bytes)], text=True, timeout=30, stderr=subprocess.STDOUT)
                return {"ok": True, "created": True, "path": vhdx, "method": "fsutil"}
            except Exception as exc3:
                return {"ok": False, "error": f"New-VHD failed: {(exc.output or str(exc))[:300]}; diskpart failed: {str(exc2)[:300]}; fsutil failed: {str(exc3)[:300]}"}


def _list_drives() -> list[dict]:
    """List fixed drives with free space for auto-detect (first-time user)."""
    import shutil
    drives: list[dict] = []
    for part in psutil.disk_partitions(all=False):
        try:
            mp = part.mountpoint
            if not mp or not os.path.exists(mp):
                continue
            # Only local fixed drives (ignore network, ramdisk)
            if part.fstype.lower() in ("", "tmpfs", "devtmpfs", "squashfs"):
                continue
            usage = shutil.disk_usage(mp)
            total_gb = round(usage.total / 1024**3, 1)
            if total_gb < 50:
                continue
            drives.append({
                "mount": mp,
                "fstype": part.fstype,
                "total_gb": total_gb,
                "free_gb": round(usage.free / 1024**3, 1),
            })
        except Exception:
            continue
    drives.sort(key=lambda d: d["free_gb"], reverse=True)
    return drives


def _best_vhdx_path(requested: str | None = None) -> str:
    """Auto-detect best drive for VHDX (first-time user with no E:)."""
    default = requested or os.environ.get("VHDX_PATH", r"E:\dsh_storage.vhdx")
    parent = os.path.dirname(default) or "."
    try:
        import shutil
        if os.path.exists(parent):
            free = shutil.disk_usage(parent).free // (1024**3)
            if free > 260:
                return default
    except Exception:
        pass
    # Pick drive with most free space >260GB
    for d in _list_drives():
        if d["free_gb"] > 260:
            drive = d["mount"].rstrip("\\/")
            if len(drive) == 2 and drive[1] == ":":
                drive += "\\"
            return os.path.join(drive, "dsh_storage.vhdx")
    return default


def _setup_health(vhdx: str | None = None) -> dict:
    """Collect VHDX / mount / shards / Docker health for the wizard."""
    import glob as _glob

    # Use requested path directly (for health display); auto-detect only when no vhdx given
    if vhdx is None:
        vhdx = os.environ.get("VHDX_PATH", r"E:\dsh_storage.vhdx")
    # Do not auto-pick best for health — show status of the requested path as-is
    mount = "/mnt/dsh_storage"
    model_dir = f"{mount}/models/DeepSeek-V4-Flash-0731-GGUF"
    vhdx_exists = os.path.exists(vhdx)
    mounted = False
    try:
        out = subprocess.check_output(
            ["wsl", "-d", "Ubuntu", "-e", "mount"], text=True, timeout=2, stderr=subprocess.DEVNULL,
            creationflags=_NO_WINDOW,
        )
        mounted = mount in out
    except Exception:
        try:
            with open("/proc/mounts", "r", encoding="utf-8") as fh:
                mounted = mount in fh.read()
        except Exception:
            mounted = False
    shards = False
    shard_path = ""
    try:
        found = _glob.glob(f"{model_dir}/*-00001-of-*.gguf")
        if not found:
            out = subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", f"ls {model_dir}/*-00001-of-*.gguf 2>&1"],
                text=True, timeout=2, stderr=subprocess.DEVNULL,
                creationflags=_NO_WINDOW,
            )
            shards = "00001-of-" in out
            if shards:
                shard_path = out.strip().splitlines()[0]
        else:
            shards = True
            shard_path = found[0]
    except Exception:
        shards = False
    docker_ok = False
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as resp:
            docker_ok = resp.status == 200
    except Exception:
        docker_ok = False
    engine = os.environ.get("ENGINE", "windows-vulkan")
    if engine not in ("windows-vulkan", "linux-rocm-docker"):
        engine = "windows-vulkan"
    return {
        "engine": engine,
        "vhdxPath": vhdx,
        "vhdxExists": vhdx_exists,
        "mountPoint": mount,
        "mounted": mounted,
        "modelDir": model_dir,
        "shardsFound": shards,
        "shardPath": shard_path,
        "dockerRunning": docker_ok,
    }


# Streaming upstream (llama-server / any OpenAI-compatible provider); module
# level so tests can inject a fake SSE transport.
_upstream_stream = requests.post


class _State:
    """Mutable app state: hives, locks, providers, engines, factories."""

    def __init__(
        self,
        ultra_factory: Callable[[], object],
        backend_factory: Callable[[Optional[str]], object],
        runs_root: Path,
        providers_file: Optional[Path],
        log_dir: str,
        state_dir: Optional[Path] = None,
        engines_file: Optional[Path] = None,
        drone_builder: Optional[Callable[[dict], object]] = None,
    ) -> None:
        self.ultra_factory = ultra_factory
        # a choice-aware drone builder: it gets the encoder choice (the
        # model, the backend, the scoring knobs) and returns the drone
        # that does the retrieval.  The tests inject ``ultra_factory``
        # instead - a no-arg fake - and that one wins.
        self.drone_builder = drone_builder
        self.backend_factory = backend_factory
        self.runs_root = runs_root
        self.providers_file = providers_file
        self.engines_file = engines_file
        self.log_dir = log_dir
        # Conversations persist here across restarts; None/empty disables.
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ProviderRegistry()
        self.engines = EngineRegistry()
        # One drone per encoder choice: the Global tab can swap the model
        # at any time and the next conversation gets a drone for the new
        # choice.  The old ones are kept (a handful), so switching back
        # is instant.
        self._ultras: dict[tuple, object] = {}
        # its own lock: the global lock is held while a conversation is
        # built, and the build asks for a drone
        self._drone_lock = threading.Lock()
        self.hives: dict[str, Splinter] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.global_lock = threading.Lock()
        # Conversation lifecycle: LRU-bounded so a long-running sidecar cannot
        # accumulate hives/loggers from every browser session that ever opened.
        self.max_conversations = int(os.environ.get("HARNESS_MAX_CONVERSATIONS", "50"))
        self._last_access: dict[str, float] = {}
        self._inflight: set[str] = set()
        self._loggers: dict[str, EventLogger] = {}
        self._conv_provider: dict[str, str] = {}  # per-conversation override

    # ------------------------------------------------------------------
    # The controls: the Global tab's blocks, read from the settings file
    # ------------------------------------------------------------------
    def settings(self) -> dict:
        """The Studio's settings, read fresh (the Global tab writes them)."""
        from harness.studio_settings import load_settings

        return load_settings(self.state_dir)

    def controls(self) -> dict:
        """The resolved baselines: curation, encoder, sampling, quality."""
        from harness import controls

        settings = self.settings()
        return {
            "memory": controls.memory_of(settings),
            "encoder": controls.encoder_of(settings),
            "sampling": controls.sampling_of(settings),
            "launch": controls.launch_of(settings),
            "quality": controls.quality_of(settings),
            "mcp": controls.mcp_of(settings),
        }

    def drone_choice(self, config: Optional[dict] = None) -> dict:
        """The drone a conversation needs: the encoder the user chose."""
        from harness import controls

        return controls.drone_choice(self.settings(), config)

    def ultra(self, choice: Optional[dict] = None):
        """The drone that does the retrieval, for the given encoder choice.

        Cached per choice: the model, the backend and the scoring knobs
        are the key, so a conversation that runs with a different encoder
        gets its own drone and the cache never mixes two models' vectors.
        """
        choice = choice or {}
        key = tuple(sorted((str(k), str(v)) for k, v in choice.items()))
        with self._drone_lock:
            drone = self._ultras.get(key)
            if drone is None:
                if self.drone_builder is not None:
                    drone = self.drone_builder(choice)
                else:
                    drone = self.ultra_factory()
                if len(self._ultras) >= 4:
                    self._ultras.pop(next(iter(self._ultras)))
                self._ultras[key] = drone
            return drone

    def _conv_path(self, conversation_id: str) -> Optional[Path]:
        """Per-conversation state file. Content-hashed name: arbitrary ids
        (session UUIDs, workspace keys, user input) stay safe on disk."""
        if self.state_dir is None:
            return None
        digest = hashlib.md5(conversation_id.encode("utf-8")).hexdigest()[:16]
        return self.state_dir / f"conv-{digest}.json"

    def save_conversation(self, conversation_id: str, splinter: Splinter) -> None:
        """Persist one conversation atomically (tmp file + os.replace)."""
        path = self._conv_path(conversation_id)
        if path is None:
            return
        payload = {
            "conversation_id": conversation_id,
            "turn": splinter.turn,
            "with_backend": splinter.backend is not None,
            "config": splinter.config.to_dict(),
            "store": splinter.store.to_dict(),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)

    def drop_conversation(self, conversation_id: str) -> None:
        path = self._conv_path(conversation_id)
        if path is not None and path.exists():
            path.unlink()

    def splinter_for(
        self, conversation_id: str, config_overrides: dict | None,
        with_backend: bool = True, engine: Optional[str] = None,
    ) -> Splinter:
        """Get or lazily create the conversation's splinter.

        A conversation not in memory but present in ``state_dir`` restores
        from disk (same serialization as the benchmark's checkpoint/resume),
        so the splinter survives sidecar restarts. In-memory hives are LRU-bounded
        (``HARNESS_MAX_CONVERSATIONS``); evicted conversations are persisted
        first and transparently restore on their next touch.

        ``with_backend=False`` (the curate/observe flow, where the caller's
        own shell generates) creates the splinter without an LLM backend; a
        conversation is driven either fully (/v1/splinter/turn) or externally
        (curate + observe), whichever touches it first wins.

        The Global tab's baseline is applied here, under the caller's
        overrides: every conversation - the chat's, the MCP client's, the
        agent's - starts from the harness-wide curation, the encoder the
        user chose and the runtime's sampling.  A caller that names its
        own value wins.
        """
        with self.global_lock:
            splinter = self.hives.get(conversation_id)
            if splinter is not None:
                self._last_access[conversation_id] = time.monotonic()
                return splinter

            baseline = self.controls()
            overrides = dict(config_overrides or {})
            conv_sampling = overrides.pop("sampling", None)

            def build(cfg: SplinterConfig, backend: object | None) -> Splinter:
                logger = self._loggers.get(conversation_id)
                if logger is None:
                    logger = EventLogger(log_dir=self.log_dir)
                    self._loggers[conversation_id] = logger
                h = Splinter(
                    config=cfg,
                    ultra=self.ultra(self.drone_choice(cfg.to_dict())),
                    backend=backend,
                    logger=logger,
                )
                self.hives[conversation_id] = h
                self.locks.setdefault(conversation_id, threading.Lock())
                return h

            path = self._conv_path(conversation_id)
            if path is not None and path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    splinter = build(SplinterConfig.from_dict(data["config"]),
                                 self.backend_factory(None)
                                 if data.get("with_backend") else None)
                    splinter.store = ContextStore.from_dict(
                        data["store"], embed_fn=splinter.ultra.embed
                    )
                    splinter.turn = int(data["turn"])
                    self._last_access[conversation_id] = time.monotonic()
                    self._evict_locked(exclude=conversation_id)
                    return splinter
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    print(f"harness: restoring {conversation_id} failed ({exc}); "
                          "starting fresh", file=sys.stderr)

            config = SplinterConfig(confidence_mode="off")
            # the harness-wide baseline first, then this conversation's
            # own overrides on top
            merged = {**config.to_dict(), **baseline["memory"], **overrides}
            config = SplinterConfig.from_dict(merged)
            # the encoder the user chose (a conversation may name its own
            # model; the baseline fills the gap the config's default
            # leaves)
            if not overrides.get("ultra_model") and baseline["encoder"]["model"]:
                config.ultra_model = baseline["encoder"]["model"]
            if not overrides.get("medium_model") and baseline["encoder"]["medium_model"]:
                config.medium_model = baseline["encoder"]["medium_model"]
            # the sampling: the runtime's baseline, then the engine
            # profile's, then this conversation's own
            profile_sampling: dict = {}
            if self.engines.engines:
                try:
                    profile = self.engines.resolve(engine)
                except LookupError:
                    profile = None
                if profile is not None:
                    profile_sampling = dict(profile.sampling or {})
            config.sampling = {**baseline["sampling"], **profile_sampling,
                               **(conv_sampling or {})}
            splinter = build(config, self.backend_factory(None) if with_backend else None)
            self._last_access[conversation_id] = time.monotonic()
            self._evict_locked(exclude=conversation_id)
            return splinter

    def _evict_locked(self, exclude: str) -> int:
        """LRU-evict idle conversations beyond the cap. Caller holds the
        global lock; in-flight conversations are never evicted, and evicted
        state is persisted first (restore-on-touch keeps it reachable)."""
        evicted = 0
        while len(self.hives) > self.max_conversations:
            candidates = [cid for cid in self.hives
                          if cid != exclude and cid not in self._inflight]
            if not candidates:
                break
            oldest = min(candidates, key=lambda c: self._last_access.get(c, 0.0))
            if oldest in self._inflight:
                # a turn started since the scan — skip it this round
                candidates.remove(oldest)
                continue
            self.save_conversation(oldest, self.hives[oldest])
            logger = self._loggers.pop(oldest, None)
            if logger is not None:
                try:
                    logger.close()
                except Exception:  # noqa: BLE001 - eviction must not fail
                    pass
            self.hives.pop(oldest, None)
            self.locks.pop(oldest, None)
            self._last_access.pop(oldest, None)
            evicted += 1
        return evicted

    def begin(self, conversation_id: str) -> None:
        self._inflight.add(conversation_id)
        self._last_access[conversation_id] = time.monotonic()

    def end(self, conversation_id: str) -> None:
        self._inflight.discard(conversation_id)

    def drop(self, conversation_id: str) -> None:
        with self.global_lock:
            self.hives.pop(conversation_id, None)
            self.locks.pop(conversation_id, None)
            self._last_access.pop(conversation_id, None)
            logger = self._loggers.pop(conversation_id, None)
        if logger is not None:
            try:
                logger.close()
            except Exception:  # noqa: BLE001
                pass
        self.drop_conversation(conversation_id)

    def lock_for(self, conversation_id: str) -> threading.Lock:
        with self.global_lock:
            return self.locks.setdefault(conversation_id, threading.Lock())


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ProtocolRunRequest(BaseModel):
    mode: str = "mock"  # live | mock
    args: dict = {}


class TrainingLaunchRequest(BaseModel):
    """One Training-tab run: engine + recipe + the form's fields."""
    engine: str = ENGINE_HIVE_TERNARY
    method: str = "ste-rotate"
    name: str = ""
    fields: dict = {}


class ProviderEntry(BaseModel):
    name: str
    base_url: str
    api_key: str = ""
    model: str = ""
    headers: dict = {}


class ProviderConfigRequest(BaseModel):
    providers: list[ProviderEntry]
    default: str = ""
    persist: bool = False
    expected_revision: str = Field(
        "", alias="expectedRevision",
        description="Revision the edit was based on; a mismatch is a 409 conflict.",
    )


class SandboxExecRequest(BaseModel):
    """A command to run in the sandbox."""

    command: str = Field(..., min_length=1, description="Shell command to execute.")
    timeout: int = Field(30, ge=1, le=120, description="Hard timeout in seconds.")
    use_docker: bool = Field(True, description="Use Docker isolation when the daemon is up.")
    image: Optional[str] = Field(
        None, description="Deprecated; must match the pinned sandbox image.")
    session_id: Optional[str] = Field(
        None, description="Session receiving a local-fallback approval.")
    parent_session_id: Optional[str] = Field(
        None, description="Parent session for a child container's cease label.")
    workspace: Optional[str] = Field(
        None, description="Host workspace to mount at /workspace; must be under "
                          "/srv/hive-ai/workspaces.")
    allow_local_fallback: bool = Field(
        False, description="Explicitly approve a host subprocess for this call.")


class WebFetchRequest(BaseModel):
    """One secure fetch. Private origins need a grant or explicit approval."""

    url: str = Field(..., min_length=1)
    method: str = Field("GET", description="GET or POST.")
    body: Optional[str] = Field(None, description="Text body for POST.")
    headers: dict[str, str] = Field(default_factory=dict)
    session_id: Optional[str] = None
    allow_private: bool = Field(
        False, description="Explicitly approve the original private origin.")
    allow_download: bool = Field(False, description="Write the response to disk.")
    download_name: Optional[str] = None
    upload_path: Optional[str] = Field(
        None, description="File under the configured upload directory.")
    allow_upload: bool = Field(False, description="Send upload_path as the body.")


class BrowserBrowseRequest(BaseModel):
    """One optional Playwright/Chromium page load behind the validating proxy."""

    url: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    allow_private: bool = Field(
        False, description="Explicitly approve the original private origin.")
    screenshot_name: Optional[str] = Field(
        None, description="Plain filename under browser/screenshots.")
    wait_until: str = Field("domcontentloaded")
    timeout_s: float = Field(30, ge=1, le=120)
    max_text: int = Field(20000, ge=1, le=200000)


class ToolRegisterRequest(BaseModel):
    tool: dict[str, Any] = Field(..., description="ToolSpec fields.")
    verified: bool = Field(
        False, description="Only verified metadata overrides conservative defaults.")
    server: str = Field("", description="MCP server name, when applicable.")
    note: str = Field("", description="Operator note.")


class WebGrantRequest(BaseModel):
    origin: str = Field(..., min_length=1)
    access: str = Field("private", description="public or private.")
    scope: str = Field("global", description="global or session.")
    session_id: Optional[str] = None
    reason: str = ""
    expires_at: Optional[float] = None


class WebRevokeRequest(BaseModel):
    origin: Optional[str] = None
    session_id: Optional[str] = None


class WebPolicyRequest(BaseModel):
    max_bytes: Optional[int] = None
    max_upload_bytes: Optional[int] = None
    max_redirects: Optional[int] = None
    timeout_s: Optional[int] = None
    allow_public: Optional[bool] = None
    allow_downloads: Optional[bool] = None
    allow_uploads: Optional[bool] = None


class ServerStartRequest(BaseModel):
    model: Optional[str] = None  # local library name or path
    hf_repo: Optional[str] = None  # passthrough to llama-server --hf-repo
    hf_file: Optional[str] = None
    key: Optional[str] = None  # instance key (defaults to the model stem)
    port: Optional[int] = None
    # None = the harness-wide runtime baseline (the Global tab's Runtime
    # block) decides, and the manager's own default when it says nothing
    ctx_size: Optional[int] = None
    ngl: Optional[int] = None  # GPU layers (Vulkan build: all layers on the RX 7900 XT)
    auto_fit: bool = False  # compute ngl from free VRAM + KV (unsloth-style)
    visible_devices: Optional[str] = None  # e.g. "1" or "0,1" -> HIP/CUDA_VISIBLE_DEVICES for this server
    register_provider: bool = True
    claim_default: bool = True  # first load claims the default provider slot
    # extra llama-server launch flags (wired to the UI settings panel)
    threads: Optional[int] = None
    flash_attn: bool = False
    parallel_slots: Optional[int] = None
    cache_type_k: Optional[str] = None  # f16 | q8_0 | q4_0 ...
    cache_type_v: Optional[str] = None
    batch_size: Optional[int] = None
    ubatch_size: Optional[int] = None
    alias: Optional[str] = None
    mlock: bool = False
    no_mmap: bool = False
    api_key: Optional[str] = None  # protect llama-server (--api-key)
    backend: Optional[str] = None  # vulkan | rocm | cuda | cpu | sycl (binary under tools/backends/<backend>/)
    embedding: bool = False  # when True, launch with --embedding (bge-m3, nomic-embed, etc.)
    pooling: Optional[str] = None  # mean | cls | last (requires embedding)
    mmproj: Optional[str] = None  # path to mmproj file for multimodal/vision models

    def extra_args(self, threads: Optional[int] = None) -> list[str]:
        args: list[str] = []
        threads = threads if threads is not None else self.threads
        if threads:
            args += ["-t", str(threads)]
        if self.flash_attn:
            # current llama.cpp: -fa takes on|off|auto (a bare -fa would eat
            # the next flag as its value)
            args += ["-fa", "on"]
        if self.parallel_slots:
            args += ["-np", str(self.parallel_slots)]
        if self.cache_type_k:
            args += ["--cache-type-k", self.cache_type_k]
        if self.cache_type_v:
            args += ["--cache-type-v", self.cache_type_v]
        if self.batch_size:
            args += ["-b", str(self.batch_size)]
        if self.ubatch_size:
            args += ["-ub", str(self.ubatch_size)]
        if self.alias:
            args += ["--alias", self.alias]
        if self.mlock:
            args += ["--mlock"]
        if self.no_mmap:
            args += ["--no-mmap"]
        if self.api_key:
            args += ["--api-key", self.api_key]
        if self.mmproj:
            args += ["--mmproj", self.mmproj]
        return args

    def load_options(self, ctx_size: Optional[int] = None,
                     ngl: Optional[int] = None,
                     threads: Optional[int] = None) -> dict:
        """Advisory engine record of the launch configuration actually used."""
        out = {"context": ctx_size if ctx_size is not None else self.ctx_size,
               "gpu_layers": ngl if ngl is not None else self.ngl}
        for key, value in (("threads", threads if threads is not None else self.threads),
                           ("flash_attn", self.flash_attn or None),
                           ("parallel_slots", self.parallel_slots),
                           ("cache_type_k", self.cache_type_k),
                           ("cache_type_v", self.cache_type_v),
                           ("batch_size", self.batch_size),
                           ("ubatch_size", self.ubatch_size),
                           ("alias", self.alias),
                           ("mlock", self.mlock or None),
                           ("no_mmap", self.no_mmap or None),
                           ("embedding", self.embedding or None),
                           ("pooling", self.pooling),
                           ("mmproj", self.mmproj)):
            if value is not None:
                out[key] = value
        return out


class HubDownloadRequest(BaseModel):
    repo: str
    file: str


class ServerUnloadRequest(BaseModel):
    key: str


class StrataLaunchRequest(BaseModel):
    """Start the Strata engine from the Launch panel.

    The panel's Strata tab is the engine's own configuration: the config
    profile, the engine flags, the VRAM reserve, the prefill chunk, the
    speculative depth.  The Studio owns the process it starts.
    """
    repo: str = ""                     # the strata checkout
    config: str = ""                   # the base config (strata-<model>.json)
    python: str = ""                   # the interpreter (default .venv/bin/python)
    port: int = 8081
    ctx: int = 0                      # --max-context (0 = the runtime baseline)
    engine_args: list[str] = []        # the engine flags
    options: dict = {}                 # merged into the generated config
    timeout_s: float = 600


class ChatTurnRequest(BaseModel):
    query: str
    conversation_id: str = "default"
    config: Optional[dict] = None


class ChatStopRequest(BaseModel):
    conversation_id: str = "default"


class AgentMessageRequest(BaseModel):
    message: str
    conversation_id: str = "default"
    assistant: bool = Field(
        False,
        description="App-assistant mode: give the agent the Studio's own API map.",
    )


class CommandRunRequest(BaseModel):
    line: str
    conversation_id: str = "default"


class EngineEntry(BaseModel):
    name: str
    kind: str = "lmstudio"
    base_url: str = ""
    load_options: dict = {}
    capabilities: list[str] = []
    sampling: dict = {}


class EngineConfigRequest(BaseModel):
    engines: list[EngineEntry]
    default: str = ""
    persist: bool = False
    expected_revision: str = Field(
        "", alias="expectedRevision",
        description="Revision the edit was based on; a mismatch is a 409 conflict.",
    )


def _config_revision(payload: object) -> str:
    """A short, stable hash of a config payload (the optimistic-concurrency token).

    Derived from the content, so it survives restarts and needs no extra
    persisted state.
    """
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


_ASSISTANT_CONTEXT = """\
You are the HiveBench Studio assistant. The Studio is the local-first AI
harness the user is looking at; it runs on this machine, and you have bash,
so you can read and drive it through its own HTTP API:

  base: http://127.0.0.1:8765
  GET  /v1/server/status       hardware (VRAM/RAM/CPU) + loaded models + health
  GET  /v1/models/local       the local GGUF library
  GET  /v1/engines            engine profiles (model, context, gpu, sampling)
  GET  /v1/provider/config    providers (endpoints, keys masked)
  GET  /v1/runs               run bundles
  GET  /v1/cascade/registry   role coverage for the cascade
  GET  /v1/sandbox/status     docker availability

Read one with e.g. `curl -s http://127.0.0.1:8765/v1/server/status | head -c 2000`.
To change something, POST to the matching endpoint (see hivebench/API.md).

Rules: answer from what the API actually returned - never guess the user's
hardware, models, or settings. If an endpoint is not in the list above, say
so and point at hivebench/API.md. Keep answers short and concrete.
"""


def _cors_origins() -> list[str]:
    """Console origins. Default: localhost dev origins only — agent mode can
    execute code, so blanket CORS (*) is opt-in via HARNESS_CORS_ORIGINS=*."""
    raw = os.environ.get("HARNESS_CORS_ORIGINS", "").strip()
    if raw == "*":
        return ["*"]
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:3000", "http://127.0.0.1:3000",
            "http://localhost:8765", "http://127.0.0.1:8765"]


def _required_token() -> str:
    """When HARNESS_TOKEN is set, /v1/* mutations require this bearer token."""
    return os.environ.get("HARNESS_TOKEN", "").strip()


def _title_from_docstrings(app: FastAPI) -> None:
    """Use each route's docstring first line as its OpenAPI summary.

    FastAPI otherwise titles an operation after the Python function name
    ("get_providers" -> "Get Providers").  The docstring is the human
    sentence we want, so promote its first line to the title and keep the
    rest as the description.
    """
    import inspect

    def _walk(routes):
        # an included router is one node holding the original router
        for route in routes:
            original = getattr(route, "original_router", None)
            if original is not None:
                yield from _walk(original.routes)
            else:
                yield route

    for route in _walk(app.routes):
        doc = getattr(route, "description", None) or inspect.getdoc(
            getattr(route, "endpoint", None) or (lambda: None))
        if not doc:
            continue
        lines = [ln.strip() for ln in doc.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        route.summary = lines[0]
        rest = " ".join(lines[1:]).strip()
        route.description = rest or None


def create_app(
    ultra_factory: Optional[Callable[[], object]] = None,
    backend_factory: Optional[Callable[[Optional[str]], object]] = None,
    runs_root: Optional[Path] = None,
    providers_file: Optional[Path] = None,
    log_dir: str = "logs",
    state_dir: Optional[Path] = None,
    engines_file: Optional[Path] = None,
    models_manager: Optional[LlamaServerManager] = None,
    llama_port: int = 1234,
) -> FastAPI:
    """Build the sidecar app.

    ``ultra_factory`` / ``backend_factory`` are injectable for offline tests;
    defaults build the real L3-v2 drone and a provider-driven OpenAI-compat
    backend (LM Studio on localhost:1234 when no providers are configured).
    ``state_dir=None`` defaults to ./harness_state (conversations survive
    restarts); passing an empty string disables persistence.
    """
    from sieve.ultra_small import UltraSmallDrone

    def _default_drone(choice: dict):
        """The drone that does the retrieval, for the encoder that was chosen.

        The choice comes from the Global tab (the model, the backend, the
        scoring knobs); the env still wins when it is set, because a
        deployment may pin the model.
        """
        backend = (choice.get("backend")
                   or os.environ.get("HARNESS_EMBEDDING_BACKEND", "local"))
        url = (choice.get("url")
               or os.environ.get("HARNESS_EMBEDDING_URL", ""))
        model = (choice.get("model")
                 or os.environ.get("HARNESS_EMBEDDING_MODEL", "default"))
        if backend == "served" and url:
            from sieve.served import ServedEmbeddingDrone

            return ServedEmbeddingDrone(base_url=url,
                                        model=model)
        # CPU by design: the encoder must never contend with llama-server
        # for VRAM. MiniLM-L3 is 12M params; CPU embedding costs ~5ms/turn.
        # Single-threaded on purpose: llama.cpp already holds a large
        # thread pool, and letting PyTorch OpenMP spawn 16 more exhausts
        # the sandbox pids cap (EAGAIN -> silent worker death). One
        # thread is still ~5ms for a 384-dim embedding.
        import torch

        torch.set_num_threads(1)
        return UltraSmallDrone(
            model_name=model,
            vocab_boost=float(choice.get("vocab_boost") or 0.15),
            confidence_mode=str(choice.get("confidence_mode") or "off"),
            device="cpu",
        )

    def _default_backend(model: Optional[str], provider: Optional[str] = None):
        kw = backend_kwargs(st.registry.resolve(provider))
        if model:
            kw["model"] = model
        return OpenAICompatBackend(**kw)

    app = FastAPI(title="Splinter Studio", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def token_guard(request: Request, call_next):
        required = _required_token()
        if required and request.url.path.startswith("/v1/"):
            supplied = request.headers.get("x-splinter-token", "")
            if supplied != required:
                from fastapi.responses import JSONResponse

                return JSONResponse({"detail": "invalid or missing token"},
                                    status_code=401)
        return await call_next(request)

    st = _State(
        ultra_factory=ultra_factory or (lambda: _default_drone({})),
        backend_factory=backend_factory or _default_backend,
        runs_root=Path(runs_root) if runs_root else DEFAULT_RUNS_ROOT,
        providers_file=Path(providers_file) if providers_file else None,
        log_dir=log_dir,
        state_dir=state_dir if state_dir is not None else Path("harness_state"),
        engines_file=Path(engines_file) if engines_file else None,
        # the choice-aware builder is the production path: it knows the
        # encoder the user picked.  A test that injects ``ultra_factory``
        # (a no-arg fake) keeps that fake - see _State.ultra.
        drone_builder=None if ultra_factory else _default_drone,
    )
    try:
        st.registry = load_registry(providers_file)
    except (ValueError, OSError) as exc:
        print(f"harness: ignoring unreadable providers config ({exc})", file=sys.stderr)
    try:
        st.engines = load_engines(engines_file)
    except (ValueError, OSError) as exc:
        print(f"harness: ignoring unreadable engines config ({exc})", file=sys.stderr)
    app.state.harness = st
    # Eagerly load the encoder at startup, not lazily on first request.
    # By first-request time uvicorn's event loop + connection threads are
    # alive and tip the process over its thread budget, so MiniLM's OpenMP
    # pool fails to spawn (EAGAIN) and the worker dies silently. Loading
    # here - before uvicorn serves - keeps the load in a clean state.
    try:
        st.ultra(st.drone_choice())
    except Exception as exc:  # noqa: BLE001 - never block startup on encoder
        print(f"harness: encoder pre-load failed ({exc}); will retry lazily",
              file=sys.stderr)

    @app.get("/health")
    def health():
        """Liveness probe: the sidecar is up."""
        return {"ok": True, "conversations": len(st.hives)}

    @app.get("/v1/setup/status")
    def setup_status(context: int = 32768, dual: bool = False, vhdx: str | None = None, model_gb: float | None = None):
        """Setup wizard status — engine + drive + health + tier (splinter console)."""
        health_info = _setup_health(vhdx)
        tier = _setup_tier(context, dual, vhdx, model_gb)
        complete = bool(health_info["vhdxExists"] and health_info["mounted"] and health_info["shardsFound"] and health_info["dockerRunning"] and not tier["flags"]["diskFull"])
        return {
            "state": {
                "engine": health_info["engine"],
                "vhdxPath": health_info["vhdxPath"],
                "modelsDir": health_info["modelDir"],
                "mountPoint": health_info["mountPoint"],
            },
            "health": {
                "windows": {"state": "running", "port": 8765},
                "linux": {
                    "state": "running" if health_info["dockerRunning"] else "stopped",
                    "port": 8000,
                    "vhdxMounted": health_info["mounted"],
                    "dockerRunning": health_info["dockerRunning"],
                    "shardsFound": health_info["shardsFound"],
                    "shardPath": health_info["shardPath"],
                    "vhdxExists": health_info["vhdxExists"],
                },
            },
            "tier": tier,
            "complete": complete,
        }

    @app.get("/v1/setup/health")
    def setup_health(vhdx: str | None = None):
        """Lightweight health for the wizard (VHDX/mount/shards/docker)."""
        return _setup_health(vhdx)

    @app.get("/v1/setup/docker-models")
    def setup_docker_models():
        """Proxy to Docker 8000 /v1/models (avoids browser CORS)."""
        import json as _json
        import urllib.request as _url

        try:
            with _url.urlopen("http://127.0.0.1:8000/v1/models", timeout=3) as resp:
                body = resp.read()
                ctype = resp.headers.get("content-type", "")
                if "json" in ctype or body.strip().startswith(b"{"):
                    try:
                        return _json.loads(body)
                    except Exception:
                        pass
                # fallback: try parse anyway
                try:
                    return _json.loads(body.decode("utf-8", errors="ignore"))
                except Exception:
                    return {"raw": body.decode("utf-8", errors="ignore")[:2000], "status": resp.status}
        except Exception as exc:
            raise HTTPException(500, f"Docker 8000 /v1/models unreachable: {exc}")

    @app.get("/v1/setup/drives")
    def setup_drives():
        """List drives and best VHDX path for first-time user (auto-detect)."""
        return {"drives": _list_drives(), "best": _best_vhdx_path(), "default": os.environ.get("VHDX_PATH", r"E:\dsh_storage.vhdx")}

    @app.post("/v1/setup/bootstrap")
    async def setup_bootstrap(request: Request):
        """One-click bootstrap for Docker → WebUI (mount + compose up). No Admin bare mount.

        Steps:
        1) If VHDX bare device not exposed, fail with Admin fix (Mount_AI_Drive.bat).
        2) If not mounted at /mnt/dsh_storage, mount the exposed device.
        3) docker compose up -d dsh-compute-backend
        Returns health after.
        """
        try:
            body = await request.json()
            vhdx_req = body.get("vhdx") if isinstance(body, dict) else None
        except Exception:
            vhdx_req = None
        steps: list[dict] = []
        # Require explicit location — do not auto-create without user picking drive
        if vhdx_req:
            vhdx = vhdx_req
        else:
            vhdx = os.environ.get("VHDX_PATH", r"E:\dsh_storage.vhdx")
        mount = "/mnt/dsh_storage"
        compose = os.environ.get("COMPOSE_PATH", os.path.expanduser(os.path.join("~", "Documents", "hivebench-studio", "docker-compose.yml")))
        # 1) VHDX exists — no auto-create (explicit Create Drive required)
        if not os.path.exists(vhdx):
            raise HTTPException(400, f"VHDX not found at {vhdx} — select a drive in Setup and click Create Drive first (explicit location required)")
        steps.append({"step": "vhdx", "ok": True, "path": vhdx})
        # 2) Bare device exposed? Check lsblk for sdd/sde etc with 230G
        bare_ok = False
        try:
            out = subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "lsblk", "-o", "NAME,SIZE,MOUNTPOINT", "-n"],
                text=True, timeout=3, stderr=subprocess.DEVNULL,
                creationflags=_NO_WINDOW,
            )
            # Look for a disk ~230G that is the VHDX (sdd 234.4G)
            bare_ok = "sdd" in out or "sde" in out
            steps.append({"step": "bare", "ok": bare_ok, "lsblk": out[:600]})
        except Exception as exc:
            steps.append({"step": "bare", "ok": False, "error": str(exc)[:300]})
        if not bare_ok:
            raise HTTPException(
                400,
                f"VHDX not exposed as bare device — run Mount_AI_Drive.bat as Administrator (wsl --mount --vhd {vhdx} --bare). lsblk: {steps[-1].get('lsblk','')[:200]}",
            )
        # 3) Mount at /mnt/dsh_storage if not already
        try:
            mout = subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "mount"], text=True, timeout=2, stderr=subprocess.DEVNULL,
                creationflags=_NO_WINDOW,
            )
            mounted = mount in mout
        except Exception:
            mounted = False
        if not mounted:
            # Try common devices: sdd1, sdd, sde1, sde (needs sudo inside WSL)
            mounted_now = False
            last_err = ""
            for dev in ["/dev/sdd1", "/dev/sdd", "/dev/sde1", "/dev/sde"]:
                try:
                    subprocess.check_output(
                        ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", f"sudo mkdir -p {mount} && sudo mount {dev} {mount}"],
                        text=True, timeout=10, stderr=subprocess.STDOUT,
                        creationflags=_NO_WINDOW,
                    )
                    mounted_now = True
                    steps.append({"step": "mount", "ok": True, "dev": dev})
                    break
                except Exception as exc:
                    # CalledProcessError, TimeoutExpired, etc. — capture output
                    last_err = (getattr(exc, 'output', None) or str(exc))[:800]
                    low = last_err.lower()
                    # First-time VHDX is NTFS/sparse — format to ext4 once, then retry mount (never if already mounted)
                    if any(s in low for s in ("wrong fs type", "unknown filesystem", "you must specify the filesystem type", "no such file or directory", "wrong fs type", "does not exist")):
                        try:
                            fmt_out = subprocess.check_output(
                                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", f"sudo mkfs.ext4 -F {dev}"],
                                text=True, timeout=30, stderr=subprocess.STDOUT,
                                creationflags=_NO_WINDOW,
                            )
                            steps.append({"step": "mkfs", "ok": True, "dev": dev, "output": fmt_out[:400]})
                            # retry mount after format
                            subprocess.check_output(
                                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", f"sudo mount {dev} {mount}"],
                                text=True, timeout=10, stderr=subprocess.STDOUT,
                                creationflags=_NO_WINDOW,
                            )
                            mounted_now = True
                            steps.append({"step": "mount", "ok": True, "dev": dev, "after_mkfs": True})
                            break
                        except Exception as exc2:
                            last_err = (getattr(exc2, 'output', None) or str(exc2))[:800]
            if not mounted_now:
                raise HTTPException(400, f"Mount {mount} failed — try wsl -d Ubuntu -e lsblk and mount manually. {last_err}")
        else:
            steps.append({"step": "mount", "ok": True, "already": True})
        # Ensure models directory exists inside the VHDX (visible as \\wsl.localhost\Ubuntu\mnt\dsh_storage\models)
        try:
            subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", f"sudo mkdir -p {mount}/models && sudo chmod 777 {mount}/models"],
                text=True, timeout=5, stderr=subprocess.STDOUT,
                creationflags=_NO_WINDOW,
            )
            steps.append({"step": "models_dir", "ok": True, "path": f"{mount}/models"})
        except Exception as exc:
            steps.append({"step": "models_dir", "ok": False, "error": str(exc)[:300]})
        # 4) Docker compose up (easy: auto-retag upstream if custom image missing)
        def _docker_up():
            if os.path.exists(compose):
                return subprocess.check_output(
                    ["docker", "compose", "-f", compose, "up", "-d", "dsh-compute-backend"],
                    text=True, timeout=60, stderr=subprocess.STDOUT,
                    creationflags=_NO_WINDOW,
                )
            return subprocess.check_output(
                ["docker", "compose", "up", "-d", "dsh-compute-backend"],
                text=True, timeout=60, stderr=subprocess.STDOUT,
                creationflags=_NO_WINDOW,
            )

        try:
            out = _docker_up()
            steps.append({"step": "docker", "ok": True, "output": out[:800]})
        except subprocess.CalledProcessError as exc:
            err = (exc.output or str(exc))
            # Easy fix: custom image not found → retag the pulled upstream
            if "custom-dsh-rocm-backend" in err and "pull access denied" in err:
                try:
                    tag_out = subprocess.check_output(
                        ["docker", "tag", "ghcr.io/ggml-org/llama.cpp:server-rocm", "custom-dsh-rocm-backend:latest"],
                        text=True, timeout=30, stderr=subprocess.STDOUT,
                        creationflags=_NO_WINDOW,
                    )
                    steps.append({"step": "docker-tag", "ok": True, "from": "ghcr.io/ggml-org/llama.cpp:server-rocm", "output": tag_out[:400]})
                    out = _docker_up()
                    steps.append({"step": "docker", "ok": True, "output": out[:800], "retried": True})
                except subprocess.CalledProcessError as exc2:
                    raise HTTPException(
                        500,
                        f"docker compose up failed (after tag retry): {(exc2.output or str(exc2))[:600]} — fix: docker pull ghcr.io/ggml-org/llama.cpp:server-rocm && docker tag ghcr.io/ggml-org/llama.cpp:server-rocm custom-dsh-rocm-backend:latest",
                    )
            elif "no such file or directory" in err and "/dev/kfd" in err:
                raise HTTPException(
                    500,
                    f"WSL GPU not available: /dev/kfd missing — fix: ensure Windows 11 + WSL2 GPU passthrough: wsl --update && wsl --shutdown, install AMD Adrenalin 24.10+, check wsl -d Ubuntu -e ls /dev/kfd — or run on Linux host with ROCm. Raw: {err[:400]}",
                )
            else:
                raise HTTPException(500, f"docker compose up failed: {err[:600]}")
        except Exception as exc:
            raise HTTPException(500, f"docker not running: {exc}")
        # 5) Health after
        time.sleep(2)
        health_after = _setup_health()
        steps.append({"step": "health", "ok": health_after["dockerRunning"], "health": health_after})
        return {"ok": health_after["dockerRunning"], "steps": steps, "health": health_after}

    @app.post("/v1/setup/mount-bare")
    async def setup_mount_bare(request: Request):
        """Bare-expose VHDX to WSL (requires Admin). Called by Setup UI button."""
        try:
            body = await request.json()
            vhdx_req = body.get("vhdx") if isinstance(body, dict) else None
        except Exception:
            vhdx_req = None
        # Require explicit location — do not auto-create without user picking drive
        if vhdx_req:
            vhdx = vhdx_req
            ensure = _ensure_vhdx(vhdx)
            if not ensure["ok"]:
                raise HTTPException(400, f"VHDX auto-create failed at {vhdx}: {ensure.get('error','')}")
        else:
            vhdx = os.environ.get("VHDX_PATH", r"E:\dsh_storage.vhdx")
            if not os.path.exists(vhdx):
                raise HTTPException(400, f"VHDX not found at {vhdx} — select a drive in Setup and click Create Drive first (explicit location required, auto-create disabled)")
            vhdx = _best_vhdx_path(vhdx) if os.path.exists(vhdx) else vhdx
        # Already bare-mounted? (sdd 234G from lsblk)
        try:
            out = subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "lsblk", "-o", "NAME,SIZE", "-n"],
                text=True, timeout=3, stderr=subprocess.DEVNULL,
                creationflags=_NO_WINDOW,
            )
            if "sdd" in out or "sde" in out:
                return {"ok": True, "output": "already bare-mounted", "already": True, "lsblk": out[:600]}
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                ["wsl", "--mount", "--vhd", vhdx, "--bare"],
                text=True, timeout=10, stderr=subprocess.STDOUT,
                creationflags=_NO_WINDOW,
            )
            return {"ok": True, "output": out[:800]}
        except subprocess.CalledProcessError as exc:
            err = (exc.output or str(exc))
            low = err.lower()
            if "already" in low or "exists" in low:
                return {"ok": True, "output": "already bare-mounted", "already": True, "raw": err[:400]}
            if "administra" in low or "elevation" in low or "access is denied" in low:
                # Pop UAC on the host desktop (sidecar and browser are same machine on localhost)
                try:
                    subprocess.Popen(
                        ["powershell", "-Command", f"Start-Process wsl -ArgumentList '--mount','--vhd','{vhdx}','--bare' -Verb RunAs -Wait"],
                        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
                    )
                    return {
                        "ok": False,
                        "needs_elevation": True,
                        "error": err[:600],
                        "hint": "UAC prompt shown on host — click Yes, then Bootstrap Docker",
                    }
                except Exception as e2:
                    raise HTTPException(400, f"wsl --mount --bare needs Admin (UAC failed: {e2}): {err[:400]} — fix: Right-click Mount_AI_Drive.bat → Run as administrator")
            raise HTTPException(400, f"wsl --mount --bare failed: {err[:600]}")

    @app.post("/v1/setup/create-vhdx")
    async def setup_create_vhdx(request: Request):
        """Explicit VHDX creation at user-picked location (first-time user)."""
        try:
            body = await request.json()
            vhdx = body.get("vhdx") if isinstance(body, dict) else None
            size_gb = body.get("size_gb") if isinstance(body, dict) else None
        except Exception:
            vhdx = None
            size_gb = None
        if not vhdx:
            raise HTTPException(400, "vhdx path required — select a drive in Setup and click Create Drive")
        try:
            size = int(size_gb) if size_gb is not None else 250
        except (ValueError, TypeError):
            size = 250
        if size < 10 or size > 2000:
            raise HTTPException(400, f"size_gb must be 10-2000, got {size}")
        # Explicit location — now auto-create is allowed
        result = _ensure_vhdx(vhdx, size_gb=size)
        if not result["ok"]:
            raise HTTPException(400, f"Create failed at {vhdx}: {result.get('error','')}")
        return result

    # ------------------------------------------------------------------
    @app.get("/v1/models")
    def models(
        probe: bool = Query(default=False),
        provider: Optional[str] = Query(default=None),
        base_url: Optional[str] = Query(default=None),
    ):
        """List served models and their status (setup tooling shape)."""
        target = base_url
        if not target:
            try:
                target = st.registry.resolve(provider).base_url
            except LookupError:
                target = "http://localhost:1234"
        try:
            ids = _list_models(target)
        except Exception as exc:  # noqa: BLE001 - surfaced as 502 to the caller
            raise HTTPException(502, f"cannot list models from {target}: {exc}")
        out = {"base_url": target, "models": ids, "probe": None}
        if probe:
            results = [probe_model(target, m).__dict__ for m in ids]
            out["probe"] = results
        return out

    # ------------------------------------------------------------------
    # Built-in mock OpenAI-compatible chat completions: pairs with
    # `python -m harness --mock` so a dsh shell (pi-ai openai-completions
    # route) can run end-to-end offline. The reply deterministically echoes
    # what the request actually contained — context size and whether splinter
    # content reached the model — which makes it a live probe of Seam A.
    # When the conversation asks for the benchmark, it emits a proper
    # hive_bench_run tool call and then acknowledges the tool result, so the
    # full agent loop (request -> tool_call -> tool/result -> answer) is
    # exercised offline.
    def _mock_reply(payload: dict) -> str:
        messages = payload.get("messages") or []
        system_txt = ""
        user_txt = ""
        for m in messages:
            if m.get("role") == "system" and not system_txt:
                system_txt = str(m.get("content") or "")
            elif m.get("role") == "user":
                user_txt = str(m.get("content") or "")
        # the exact marker dsh-splinter appends as a snapshot user message
        curated = any(
            "splinter-curated-context" in str(m.get("content") or "")
            for m in messages
        )
        head = " ".join(system_txt.split())[:160]
        return (
            f"[splinter-mock] model={payload.get('model', '?')} "
            f"system={len(system_txt)}ch user={len(user_txt)}ch "
            f"splinter_context={'yes' if curated else 'no'} "
            f"context_head={head!r}"
        )

    def _message_text(message: object) -> str:
        if isinstance(message, str):
            return message
        if isinstance(message, list):
            parts = []
            for block in message:
                if isinstance(block, dict):
                    parts.append(str(block.get("text") or ""))
                else:
                    parts.append(str(block))
            return "".join(parts)
        return str(message or "")

    def _mock_chat_decision(payload: dict) -> dict:
        """Return {'reply': str} or {'tool_call': (name, arguments_json)}."""
        messages = payload.get("messages") or []
        tool_results = [m for m in messages if m.get("role") == "tool"]
        if tool_results:
            last = tool_results[-1]
            text = _message_text(last.get("content"))
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            verdicts = next(
                (ln for ln in lines if "|" in ln), ""
            )
            pes = next((ln for ln in lines if "PES" in ln.upper()), "")
            reply = "The HiveBench run completed."
            if pes:
                reply += f" {pes}."
            if verdicts:
                reply += f" Verdicts: {verdicts}"
            return {"reply": reply}
        user_text = " ".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "user"
        ).lower()
        if "hive_bench" in user_text or "benchmark" in user_text \
                or "p1-p11" in user_text or "p1–p11" in user_text:
            match = re.search(r"(\d+)\s+conv", user_text)
            max_convs = int(match.group(1)) if match else 2
            return {"tool_call": (
                "hive_bench_run",
                json.dumps({"mode": "mock", "max_convs": max_convs,
                            "protocol": True}),
            )}
        return {"reply": _mock_reply(payload)}

    def _mock_completion_payload(payload: dict, decision: dict,
                                 cid: str, created: int) -> tuple[dict, dict]:
        usage = {
            "prompt_tokens": sum(len(str(m.get("content") or "").split())
                                 for m in (payload.get("messages") or [])),
            "completion_tokens": 40, "total_tokens": 0,
        }
        model = payload.get("model", "mock")
        if "tool_call" in decision:
            name, arguments = decision["tool_call"]
            message = {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }],
            }
            finish = "tool_calls"
            completion_tokens = len(arguments.split()) + 6
        else:
            message = {"role": "assistant", "content": decision["reply"]}
            finish = "stop"
            completion_tokens = len(decision["reply"].split())
        usage["completion_tokens"] = completion_tokens
        usage["total_tokens"] = usage["prompt_tokens"] + completion_tokens
        return message, {"model": model, "finish": finish, "usage": usage}

    @app.post("/v1/chat/completions")
    async def mock_chat_completions(request: Request):
        """OpenAI-compatible chat passthrough for the mock backend."""
        payload = await request.json()
        debug_dir = os.environ.get("HARNESS_DEBUG_CHAT")
        if debug_dir:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(debug_dir) / f"chat_{int(time.time() * 1000)}.json",
                      "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1, default=str)
        decision = _mock_chat_decision(payload)
        cid = f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        message, meta = _mock_completion_payload(payload, decision, cid, created)

        if not payload.get("stream"):
            return {
                "id": cid, "object": "chat.completion", "created": created,
                "model": meta["model"],
                "choices": [{
                    "index": 0, "message": message,
                    "finish_reason": meta["finish"],
                }],
                "usage": meta["usage"],
            }

        if "tool_call" in decision:
            tc = message["tool_calls"][0]
            chunks = [
                {"delta": {"role": "assistant", "tool_calls": [{
                    "index": 0, "id": tc["id"], "type": "function",
                    "function": {"name": tc["function"]["name"],
                                 "arguments": ""},
                }]}},
                {"delta": {"tool_calls": [{"index": 0, "function": {
                    "arguments": tc["function"]["arguments"]}}]}},
                {"delta": {}, "finish_reason": meta["finish"]},
            ]
        else:
            content = message["content"]
            pieces = [content[i:i + 24] for i in range(0, len(content), 24)] or [""]
            chunks = [{"delta": {"role": "assistant", "content": p}} for p in pieces]
            chunks.append({"delta": {}, "finish_reason": "stop"})

        chunks[-1]["finish_reason"] = meta["finish"]

        def sse():
            for part in chunks:
                choice = {"index": 0,
                          "delta": part.get("delta", {}),
                          "finish_reason": part.get("finish_reason")}
                body = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": meta["model"],
                    "choices": [choice],
                }
                if part is chunks[-1]:
                    body["usage"] = meta["usage"]
                yield "data: " + json.dumps(body) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Real OpenAI-compatible passthrough (curated) — Mode A integration
    # (OpenCode, dsh, any OpenAI client): standard /chat/completions wire
    # shape, curated system context, the reply observed back into the
    # store. Conversation key: X-Splinter-Conversation header > payload "user"
    # > "default".
    # OpenAI-shape model list for clients that probe {base_url}/models
    # (Unsloth Studio's connection test) when pointed at the curated
    # /v1/openai passthrough. /v1/models above is the setup tooling shape;
    # this one is the wire shape external clients expect.
    @app.post("/v1/provider/config")
    def set_providers(req: ProviderConfigRequest):
        """Replace the provider list (name, base_url, api_key, model)."""
        current = _config_revision({
            "providers": st.registry.redacted(), "default": st.registry.default,
        })
        if req.expected_revision and req.expected_revision != current:
            raise HTTPException(
                409,
                "stale provider revision — reload and retry "
                f"(expected {req.expected_revision}, current {current})",
            )
        reg = ProviderRegistry(default=req.default)
        for entry in req.providers:
            data = entry.model_dump()
            if data.get("api_key") == MASK:
                # the UI echoes the mask back for untouched keys — keep the
                # stored secret instead of overwriting it with "***"
                previous = [p for p in st.registry.providers
                            if p.name.lower() == str(data.get("name", "")).lower()]
                data["api_key"] = previous[0].api_key if previous else ""
            try:
                reg.providers.append(Provider.from_dict(data))
            except ValueError as exc:
                raise HTTPException(422, str(exc))
        st.registry = reg
        persisted = None
        if req.persist:
            path = save_registry(reg, st.providers_file)
            persisted = str(path)
        return {"ok": True, "default": reg.default,
                "providers": reg.redacted(), "persisted_to": persisted,
                "revision": _config_revision({
                    "providers": reg.redacted(), "default": reg.default,
                })}

    @app.get("/v1/provider/config")
    def get_providers():
        """List configured providers and the default one."""
        return {
            "default": st.registry.default,
            "providers": st.registry.redacted(),
            "file": str(providers_path(st.providers_file)),
            "revision": _config_revision({
                "providers": st.registry.redacted(), "default": st.registry.default,
            }),
        }

    # ------------------------------------------------------------------
    @app.post("/v1/engines")
    def set_engines(req: EngineConfigRequest):
        """Replace the engine profile list (name, kind, model, load options, sampling)."""
        current = _config_revision({
            "engines": [e.to_dict() for e in st.engines.engines],
            "default": st.engines.default,
        })
        if req.expected_revision and req.expected_revision != current:
            raise HTTPException(
                409,
                "stale engine revision — reload and retry "
                f"(expected {req.expected_revision}, current {current})",
            )
        reg = EngineRegistry(default=req.default)
        for entry in req.engines:
            try:
                reg.engines.append(EngineProfile.from_dict(entry.model_dump()))
            except ValueError as exc:
                raise HTTPException(422, str(exc))
        st.engines = reg
        persisted = None
        if req.persist:
            path = save_engines(reg, st.engines_file)
            persisted = str(path)
        return {"ok": True, "default": reg.default,
                "engines": [e.to_dict() for e in reg.engines],
                "persisted_to": persisted,
                "revision": _config_revision({
                    "engines": [e.to_dict() for e in reg.engines],
                    "default": reg.default,
                })}

    @app.get("/v1/engines")
    def get_engines():
        """List engine profiles and the default one."""
        return {
            "default": st.engines.default,
            "engines": [e.to_dict() for e in st.engines.engines],
            "file": str(engines_path(st.engines_file)),
            "revision": _config_revision({
                "engines": [e.to_dict() for e in st.engines.engines],
                "default": st.engines.default,
            }),
        }

    @app.get("/v1/engines/preset")
    def engines_preset(file: str = Query(...)):
        """Auto preset for the Engine profiles section.

        Combines GET /v1/server/status hardware + GET /v1/models/local
        size_gb + gguf-metadata (parsed from the GGUF header) into a
        ready-to-apply load_options preset. The Studio's Auto button calls
        this and then saves the profile. Examples: qwen3-4b on 8GB →
        gpu_layers 28 + 8k ctx, qwen3-32b → 12 + 4k.
        """
        # hardware: available_gb prefers VRAM when present, else RAM
        hw = _hardware_summary()
        hardware_gb = float(hw.get("available_gb") or hw.get("vram_gb") or 8.0)
        # model size + gguf metadata
        size_gb = 0.0
        gguf_meta: dict = {}
        try:
            models = models_manager.list_local()
            entry = next((m for m in models if m.get("file") == file), None)
            if entry is not None:
                size_gb = float(entry.get("size_gb") or 0)
            # try GGUF header even if not in list (e.g. absolute path)
            cand = models_manager.resolve_model(file) if file else None
            if cand and cand.is_file():
                gguf_meta = _read_gguf_metadata(cand)
        except Exception:
            pass
        # fallback: infer from file name when model not yet local
        if size_gb == 0:
            low = file.lower()
            if "32b" in low:
                size_gb = 18.0
            elif "14b" in low or "13b" in low:
                size_gb = 8.0
            elif "7b" in low or "8b" in low:
                size_gb = 4.5
            elif "4b" in low:
                size_gb = 2.5
        preset = _auto_preset_load_options(size_gb, hardware_gb, file, gguf_meta)
        return {
            "file": file,
            "hardware": hw,
            "model": {"file": file, "size_gb": size_gb, "gguf_metadata": gguf_meta},
            "preset": preset,
            "load_options": preset,
        }

    # ------------------------------------------------------------------
    # Model management (M4): own llama.cpp server + live Hugging Face hub.
    if models_manager is None:
        models_manager = LlamaServerManager(log_dir=Path(log_dir),
                                            port=llama_port)
    app.state.models = models_manager

    # Local Stack Control (LSC): the §B3 router is a factory (T38) and this is
    # the single integration mount (T44).  The manager shares the app's
    # LlamaServerManager, so stack tiers show up in /v1/models/local like any
    # other loaded model.
    stack_manager = StackManager(models_manager)
    app.state.stack_manager = stack_manager
    app.include_router(create_stack_router(stack_manager))

    # Cascade pilot (Track C): the console contract.  The router wraps
    # harness.cascade and the experiments/cascade run artifacts; it is a
    # factory so tests can point it at a tmp runs root with a fake launcher.
    # The validator is lazy (no model loads until /v1/cascade/validate is
    # called) and CPU-default; HIVE_VALIDATOR_MODEL swaps the checkpoint.
    from harness.cascade.validator import ValidatorService

    app.include_router(cascade_api.create_router(
        validator=ValidatorService(),
        # the Global tab's Quality block is the baseline every run starts
        # from (a run that names its own knob wins)
        defaults=lambda: st.controls()["quality"],
    ))

    def _retire_orphan_providers() -> list[str]:
        """Retire the local providers whose server is gone.

        Unloading only knows the instances it holds.  An instance that died
        on its own (a crash, a restart of the Studio) leaves its provider
        and engine profile behind, pointing at a dead port - every chat to
        them fails.  A local provider whose port is not listening is an
        orphan, so it goes with the unload.
        """
        from urllib.parse import urlparse

        from harness.models import _port_in_use

        retired: list[str] = []
        for prov in list(st.registry.providers):
            if not prov.name.lower().startswith("local-"):
                continue
            parsed = urlparse(prov.base_url)
            try:
                port = int(parsed.port or 0)
            except ValueError:
                port = 0
            if not port:
                continue
            if _port_in_use(models_manager.host, port, timeout=0.4):
                continue          # the server is still there
            st.registry.providers = [
                p for p in st.registry.providers
                if p.name.lower() != prov.name.lower()
            ]
            st.engines.engines = [
                e for e in st.engines.engines
                if e.name.lower() != prov.name.lower()
            ]
            if (st.registry.default or "").lower() == prov.name.lower():
                st.registry.default = next(
                    (p.name for p in st.registry.providers), "")
            if (st.engines.default or "").lower() == prov.name.lower():
                st.engines.default = next(
                    (e.name for e in st.engines.engines), "")
            retired.append(prov.name)
        return retired

    def register_local_remove(key: str) -> None:
        """Retire a local instance's provider + engine profile.

        The default has to move too: a default that points at a provider
        whose server is gone is an orphan - the Studio would keep talking to
        a dead port.
        """
        prov_name = f"local-{key}"
        st.registry.providers = [
            p for p in st.registry.providers
            if p.name.lower() != prov_name.lower()
        ]
        st.engines.engines = [
            e for e in st.engines.engines if e.name.lower() != prov_name.lower()
        ]
        if (st.registry.default or "").lower() == prov_name.lower():
            st.registry.default = next(
                (p.name for p in st.registry.providers), "")
        if (st.engines.default or "").lower() == prov_name.lower():
            st.engines.default = next(
                (e.name for e in st.engines.engines), "")

    def register_local(info: dict, load_options: Optional[dict] = None,
                       api_key: Optional[str] = None,
                       key: Optional[str] = None,
                       claim_default: bool = True) -> str:
        """Register one manager-loaded llama-server as provider
        ``local-<key>`` + engine profile with the real launch options. Shared
        by the start endpoint and CLI auto-start. Returns the provider name.
        Multi-model: every loaded instance gets its own provider; the first
        (or an explicitly re-loaded default) claims the default slot."""
        if not info.get("running"):
            return ""
        inst_key = key or "local"
        prov_name = f"local-{inst_key}"
        base_url = f"http://{models_manager.host}:{info['port']}"
        prov = Provider(name=prov_name, base_url=base_url,
                        api_key=api_key or "lm-studio",
                        model=str(info.get("model") or ""))
        st.registry.providers = [
            p for p in st.registry.providers if p.name.lower() != prov_name.lower()
        ] + [prov]
        if claim_default:
            # a managed launch IS the studio's brain: claim the default
            # (the user explicitly loaded this model; /provider switches back)
            st.registry.default = prov_name
        try:
            save_registry(st.registry, st.providers_file)
        except OSError as exc:
            print(f"harness: could not persist providers config ({exc})",
                  file=sys.stderr)
        profile = EngineProfile(
            name=prov_name, kind="llama_cpp", base_url=base_url,
            load_options=load_options
            or {"context": 8192, "gpu_layers": 999},
            capabilities=["streaming", "prefix_caching"],
        )
        st.engines.engines = [
            e for e in st.engines.engines if e.name.lower() != prov_name.lower()
        ] + [profile]
        if not st.engines.default or st.engines.default.startswith("local"):
            st.engines.default = prov_name
        # persist so CLI auto-start (and /model) reuse these launch settings
        try:
            save_engines(st.engines, st.engines_file)
        except OSError as exc:
            print(f"harness: could not persist engines config ({exc})",
                  file=sys.stderr)
        return prov_name

    app.state.register_local = register_local

    # ------------------------------------------------------------------
    # A/B compare — bench two Engine profiles side-by-side:
    #   start A on basePort, B on basePort+1, then POST 3 prompts to
    #   each POST /v1/chat/completions and compare tok/s to determine winner.
    #   Frontend grid cell uses this via the Bench button and winner badge.
    # ------------------------------------------------------------------
    _AB_DEFAULT_PROMPTS = [
        "Hello, how are you?",
        "Write a short story about a cat.",
        "Explain quantum physics briefly.",
    ]

    def _ab_ensure_started(profile: EngineProfile, port: int) -> str:
        """Ensure the profile's model is serving on ``port``; return base_url.

        Best-effort: if a server is already on that port, reuse it; if
        ``load()`` fails because the model is not found or already loaded,
        still return the constructed URL so the bench can proceed (tests mock
        the upstream POST). The helper prefers a local GGUF matching the
        profile name when available.
        """
        base_url = f"http://{getattr(models_manager, 'host', '127.0.0.1')}:{port}"
        try:
            # already serving on this port?
            try:
                stt = models_manager.status()
                for inst in stt.get("instances", []) or []:
                    if int(inst.get("port", -1)) == int(port):
                        return base_url
            except Exception:
                pass
            # pick a model file for the load
            model_file = None
            try:
                local = models_manager.list_local()
                if local:
                    for m in local:
                        f = str(m.get("file") or "")
                        n = str(m.get("name") or "")
                        if profile.name.lower() in f.lower() or profile.name.lower() in n.lower():
                            model_file = f
                            break
                    if not model_file:
                        model_file = local[0].get("file")
            except Exception:
                pass
            opts = profile.load_options if isinstance(profile.load_options, dict) else {}
            try:
                ctx_sz = int(opts.get("context") or opts.get("ctx_size") or opts.get("contextLength") or 8192)
            except Exception:
                ctx_sz = 8192
            try:
                ngl = int(opts.get("gpu_layers") or opts.get("ngl") or 999)
            except Exception:
                ngl = 999
            # attempt load only if not already serving
            if not any(int(inst.get("port", -1)) == int(port) for inst in (models_manager.status().get("instances", []) or []) if isinstance(inst, dict)):
                try:
                    models_manager.load(model=model_file, port=port, ctx_size=ctx_sz, ngl=ngl, extra_args=[])
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "already loaded" in msg or "already serving" in msg or "port" in msg or "neither a local" in msg or "not found" in msg:
                        pass
                    else:
                        raise
        except Exception:
            pass
        return base_url

    def _ab_bench_one(base_url: str, prompts: list[str]) -> dict:
        """POST 3 prompts to ``base_url/v1/chat/completions`` and measure tok/s."""
        total_tokens = 0
        total_time = 0.0
        results: list[dict] = []
        for prompt in prompts[:3]:
            payload = {
                "model": "bench",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
            }
            start = time.time()
            resp = requests.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json() if hasattr(resp, "json") else json.loads(resp.text)
            usage = data.get("usage") if isinstance(data, dict) else {}
            comp = None
            if isinstance(usage, dict):
                comp = usage.get("completion_tokens")
                if comp is None:
                    comp = usage.get("completionTokens")
            if comp is None:
                try:
                    content = data["choices"][0]["message"]["content"] or ""
                    comp = len(str(content).split())
                except Exception:
                    comp = 0
            try:
                comp = int(comp)
            except Exception:
                comp = 0
            elapsed = max(time.time() - start, 1e-6)
            tps = comp / elapsed if elapsed else 0
            total_tokens += comp
            total_time += elapsed
            results.append({"prompt": prompt, "tokens": comp, "seconds": round(elapsed, 4), "tok_per_sec": round(tps, 2)})
        avg = (total_tokens / total_time) if total_time else 0
        return {"tok_per_sec": avg, "total_tokens": total_tokens, "total_seconds": round(total_time, 4), "results": results}

    async def _ab_handle(request: Request):
        body = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        # flexible profile keys
        profile_a = body.get("profile_a") or body.get("profileA") or body.get("a") or body.get("A") or body.get("profile_a_name")
        profile_b = body.get("profile_b") or body.get("profileB") or body.get("b") or body.get("B") or body.get("profile_b_name")
        base_port_raw = body.get("basePort") if body.get("basePort") is not None else body.get("base_port") if body.get("base_port") is not None else body.get("baseport")
        prompts = body.get("prompts") or body.get("prompt_list") or _AB_DEFAULT_PROMPTS
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [str(p) for p in (prompts or _AB_DEFAULT_PROMPTS)][:3]
        while len(prompts) < 3:
            prompts += _AB_DEFAULT_PROMPTS[len(prompts):3]
        if not profile_a or not profile_b:
            raise HTTPException(422, "profile_a and profile_b are required")
        try:
            eng_a = st.engines.resolve(profile_a)
        except LookupError as exc:
            raise HTTPException(404, str(exc))
        try:
            eng_b = st.engines.resolve(profile_b)
        except LookupError as exc:
            raise HTTPException(404, str(exc))
        try:
            bp = int(base_port_raw) if base_port_raw is not None else int(getattr(models_manager, "port", 1234))
        except Exception:
            bp = 1234
        if bp < 1024 or bp > 65534:
            raise HTTPException(422, "basePort out of range (1024-65534)")
        port_a = bp
        port_b = bp + 1
        # start both on basePort and basePort+1
        url_a = _ab_ensure_started(eng_a, port_a)
        url_b = _ab_ensure_started(eng_b, port_b)
        # bench each — 3 POST /v1/chat/completions, measure tok/s
        try:
            res_a = _ab_bench_one(url_a, prompts)
        except Exception as exc:
            raise HTTPException(502, f"A bench failed on {url_a}: {exc}")
        try:
            res_b = _ab_bench_one(url_b, prompts)
        except Exception as exc:
            raise HTTPException(502, f"B bench failed on {url_b}: {exc}")
        a_tps = float(res_a.get("tok_per_sec") or 0)
        b_tps = float(res_b.get("tok_per_sec") or 0)
        if a_tps > b_tps:
            winner = "A"
        elif b_tps > a_tps:
            winner = "B"
        else:
            winner = "tie"
        # small difference within 2% counts as tie (noise)
        if winner != "tie":
            mx = max(a_tps, b_tps)
            if mx and abs(a_tps - b_tps) / mx < 0.02:
                winner = "tie"
        return {
            "winner": winner,
            "a_tok_per_sec": round(a_tps, 2),
            "b_tok_per_sec": round(b_tps, 2),
            "a": {"profile": eng_a.name, "port": port_a, "base_url": url_a, "tok_per_sec": round(a_tps, 2), **res_a},
            "b": {"profile": eng_b.name, "port": port_b, "base_url": url_b, "tok_per_sec": round(b_tps, 2), **res_b},
            "basePort": port_a,
            "base_port": port_a,
            "port_a": port_a,
            "port_b": port_b,
            "prompts": prompts,
            # aliases for frontend fallbacks
            "tokens_per_sec_a": round(a_tps, 2),
            "tokens_per_sec_b": round(b_tps, 2),
            "profile_a": eng_a.name,
            "profile_b": eng_b.name,
        }

    @app.post("/v1/engines/ab/bench")
    async def engines_ab_bench(request: Request):
        """Benchmark two engine profiles head to head on the same port base."""
        return await _ab_handle(request)

    @app.post("/v1/engines/bench")
    async def engines_bench_alias(request: Request):
        """Alias of the A/B bench (kept for older clients)."""
        return await _ab_handle(request)

    @app.post("/v1/ab/bench")
    async def ab_bench_alias(request: Request):
        """Alias of the A/B bench (kept for older clients)."""
        return await _ab_handle(request)

    def _stack_runtime_config() -> Optional[str]:
        """The dsh profile for the currently applied stack (T48).

        Called lazily before every runtime build, so a stack apply or unload
        is picked up on the next agent turn without touching the SSE path.
        ``None`` keeps the runtime's bundled profile (no delegation) when no
        stack is applied or no base profile can be resolved.
        """
        try:
            status = stack_manager.status()
            home = Path(os.environ.get("DSH_HOME") or (REPO_ROOT / ".dsh-home"))
            generated = generate_runtime_config(
                status, base_path=default_base_config(), out_dir=home)
        except Exception:  # noqa: BLE001 - a broken generator must not block chat
            return None
        return str(generated.path) if generated.path else None

    from harness.state_store import StatePaths

    # ``state_dir=""`` disables persistence; the activity ledger is part of
    # that state, so do not create even its directories in that mode.
    activity = (ActivityHub(StatePaths(Path(st.state_dir)))
                if st.state_dir else None)
    app.state.activity = activity
    agent_service = DshAgentService(
        default_cwd=REPO_ROOT,
        session_root=dsh_session_root(),
        runtime_config=_stack_runtime_config,
        activity=activity,
    )
    app.state.agent = agent_service

    def _cease_stop(*, conversation_id: str = "", session_id: str = "",
                    reason: str = "", force: bool = True) -> dict[str, Any]:
        result = agent_service.cease_stop(
            conversation_id=conversation_id, session_id=session_id,
            reason=reason, force=force)
        try:
            from harness.sandbox import kill_session_containers

            killed: list[str] = []
            errors: list[str] = []
            for target in {session_id, conversation_id} - {""}:
                outcome = kill_session_containers(target)
                killed.extend(outcome.get("killed") or [])
                if outcome.get("error"):
                    errors.append(str(outcome["error"]))
                errors.extend(str(item) for item in outcome.get("errors") or [])
            result["containers"] = {"killed": killed, "errors": errors}
        except Exception as exc:  # noqa: BLE001 - the block is the contract
            result["containers"] = {"killed": [], "error": str(exc)}
        return result

    if activity is not None:
        activity.register_cease_callback(_cease_stop)

    commands = ConsoleCommands(
        st=st,
        models=models_manager,
        agent=agent_service,
        transcripts_dir=REPO_ROOT / "transcripts",
    )
    commands._app = app  # register_local lives on the app instance
    app.state.commands = commands

    @app.get("/v1/commands")
    def list_commands():
        """List console commands (name, description, input hint)."""
        return {"commands": commands.descriptors()}

    @app.post("/v1/commands/run")
    def run_command(req: CommandRunRequest):
        """Run a console command line, e.g. '/status'."""
        result = commands.run(req.line, req.conversation_id)
        return result.to_dict()

    # ------------------------------------------------------------------
    # The chat turn: the Studio owns the provider connection
    # ------------------------------------------------------------------
    # Both engines stop generating when the client connection drops —
    # llama.cpp through its should_stop check, strata through its client
    # watcher — so the Studio keeps the provider request itself instead of
    # handing it to the mount below.  That is what makes Stop a real stop:
    # it closes this connection and the engine actually stops, instead of
    # the tokens merely disappearing from the browser.
    _live_chats: dict[str, dict] = {}

    @app.post("/v1/chat/stop")
    def chat_stop(req: ChatStopRequest):
        """Stop the running turn by closing the provider connection.

        The engine sees the disconnect and stops generating (llama.cpp and
        strata both do), so the GPU is freed at once rather than after the
        model finishes the sentence nobody is waiting for.
        """
        live = _live_chats.get(req.conversation_id)
        if live is None:
            return {"stopped": False, "conversation_id": req.conversation_id}
        live["stopped"] = True
        response = live.get("response")
        if response is not None:
            try:
                response.close()      # the socket drops; the engine stops
            except Exception:  # noqa: BLE001 - best effort
                pass
        return {"stopped": True, "conversation_id": req.conversation_id}

    # ------------------------------------------------------------------
    # The memory over MCP, with the Global tab's MCP block in force.
    # The sibling's own endpoint is mounted at the root and would answer
    # unconditionally; this route stands in front of it so the block
    # (expose the memory? read-only? which tools?) is the one in charge.
    # ------------------------------------------------------------------
    MCP_TOOL_NAMES = ("splinter_remember", "splinter_search")

    def _mcp_withheld(settings: dict) -> set[str]:
        """The tools the client may not use, per the MCP block."""
        from harness import controls as _controls

        mcp = _controls.mcp_of(settings)
        withheld: set[str] = set()
        if mcp["read_only"]:
            withheld.add("splinter_remember")
        allowed = {str(t).strip().lower() for t in mcp["tools"]}
        if allowed:
            for tool in MCP_TOOL_NAMES:
                short = tool.split("_", 1)[1]
                if short not in allowed and tool not in allowed:
                    withheld.add(tool)
        return withheld

    def _mcp_filter_tools(payload, withheld: set[str]):
        """Drop the withheld tools from a tools/list answer."""
        if not withheld:
            return payload
        if isinstance(payload, list):
            return [_mcp_filter_tools(item, withheld) for item in payload]
        if not isinstance(payload, dict):
            return payload
        result = payload.get("result")
        if isinstance(result, dict) and isinstance(result.get("tools"), list):
            result["tools"] = [t for t in result["tools"]
                               if not (isinstance(t, dict)
                                       and t.get("name") in withheld)]
        return payload

    @app.post("/v1/mcp")
    async def mcp_endpoint(request: Request):
        """The memory over MCP: the tools the Global tab allows."""
        from splinter.mcp.server import McpContext, handle_message
        from splinter.mcp.tools import remember as mcp_remember
        from splinter.mcp.tools import search as mcp_search

        settings = st.settings()
        from harness import controls as _controls

        mcp = _controls.mcp_of(settings)
        if not mcp["enabled"]:
            raise HTTPException(
                403, "the memory over MCP is off (the Global tab's MCP block)")
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "invalid JSON body")
        withheld = _mcp_withheld(settings)

        def do_remember(conversation_id: str, text: str) -> dict:
            if "splinter_remember" in withheld:
                raise ValueError(
                    "the memory is read-only (the Global tab's MCP block)")
            splinter = st.splinter_for(conversation_id, None, with_backend=False)
            with st.lock_for(conversation_id):
                payload = mcp_remember(splinter, text)
                st.save_conversation(conversation_id, splinter)
            return {"conversation_id": conversation_id, **payload}

        def do_search(conversation_id: str, query: str, top_k: int) -> dict:
            if "splinter_search" in withheld:
                raise ValueError(
                    "the search tool is withheld (the Global tab's MCP block)")
            splinter = st.splinter_for(conversation_id, None, with_backend=False)
            with st.lock_for(conversation_id):
                payload = mcp_search(splinter, query, top_k)
                st.save_conversation(conversation_id, splinter)
            return {"conversation_id": conversation_id, **payload}

        response = handle_message(body, McpContext(remember=do_remember,
                                                   search=do_search))
        if response is None:  # notifications only - nothing to answer
            return Response(status_code=202)
        return _mcp_filter_tools(response, withheld)

    @app.post("/v1/splinter/stream")
    async def splinter_stream(req: ChatTurnRequest, request: Request):
        """One chat turn: curate here, generate upstream, stream the deltas."""
        query = (req.query or "").strip()
        if not query:
            raise HTTPException(422, "query must not be empty")
        try:
            provider = st.registry.resolve(None)
        except LookupError as exc:
            raise HTTPException(502, f"no provider configured: {exc}")
        base_url = provider.base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {provider.api_key or 'lm-studio'}",
                   **provider.extra_headers}
        splinter = st.splinter_for(
            req.conversation_id,
            # the Global tab's baseline is applied inside (the curation,
            # the encoder and the sampling); this conversation's own
            # config rides on top of it
            req.config or {},
            with_backend=False)
        st.begin(req.conversation_id)
        try:
            with st.lock_for(req.conversation_id):
                result = splinter.process_turn(query,
                                               conversation_id=req.conversation_id)
                st.save_conversation(req.conversation_id, splinter)
        finally:
            st.end(req.conversation_id)
        assembled = result.assembled
        curated = assembled.content if assembled is not None else ""
        payload = {
            "model": provider.model or "local",
            "messages": [
                {"role": "system",
                 "content": curated or "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            **(splinter.config.sampling or {}),
        }
        if splinter.config.max_tokens:
            payload["max_tokens"] = splinter.config.max_tokens

        live: dict = {"response": None, "stopped": False, "done": False}

        def close_live() -> None:
            """Close the provider connection - the engine stops on the drop."""
            response = live.get("response")
            if response is not None:
                try:
                    response.close()
                except Exception:  # noqa: BLE001 - best effort
                    pass

        async def watch_client() -> None:
            """Close the provider connection when the browser goes away.

            A closed tab must not leave the engine generating into the void:
            both engines stop on the disconnect (llama.cpp through its
            should_stop check, strata through its client watcher), so the
            connection is closed as soon as the client is gone.  This runs as
            its own task because the response task is cancelled on a
            disconnect - the generator's own cleanup never gets to run.
            """
            while not live.get("done"):
                try:
                    if await request.is_disconnected():
                        live["stopped"] = True
                        close_live()
                        return
                except Exception:  # noqa: BLE001 - the request is over
                    return
                await anyio.sleep(0.2)

        def sse():
            from harness import controls as _controls

            yield "data: " + json.dumps({
                "type": "meta", "turn": result.turn,
                "token_count": result.token_count, "budget": result.budget,
                "curated_chars": len(curated), "mode": result.mode,
                # what the turn actually ran with: the encoder the user
                # chose, the curation the Global tab asked for, the
                # sampling the reply was drawn with
                "harness": _controls.effective_summary(
                    st.settings(), splinter.config.to_dict()),
            }) + "\n\n"
            started = time.time()
            parts: list[str] = []
            usage: dict = {}
            response = None
            _live_chats[req.conversation_id] = live
            try:
                response = requests.post(
                    f"{base_url}/v1/chat/completions", json=payload,
                    headers=headers, stream=True, timeout=600,
                )
                live["response"] = response
                response.raise_for_status()
                for raw in response.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw[6:].strip() if raw.startswith("data:") else raw.strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if text:
                            parts.append(text)
                            yield "data: " + json.dumps({
                                "type": "delta", "text": text}) + "\n\n"
            except Exception as exc:  # noqa: BLE001 - surfaced as an event
                if not live["stopped"]:
                    yield "data: " + json.dumps({
                        "type": "error", "error": str(exc)}) + "\n\n"
            finally:
                live["done"] = True
                # closing the response is what stops the engine when the
                # browser goes away (or when Stop closes it)
                if response is not None:
                    try:
                        response.close()
                    except Exception:  # noqa: BLE001
                        pass
                if _live_chats.get(req.conversation_id) is live:
                    _live_chats.pop(req.conversation_id, None)

            if live["stopped"]:
                yield "data: " + json.dumps({
                    "type": "stopped", "text": "".join(parts)}) + "\n\n"
                return

            reply = "".join(parts)
            stored = False
            if reply.strip() and not (
                splinter.config.filter_hedge_replies
                and Splinter._is_hedge_reply(reply)
            ):
                stored = splinter.store.add_chunk(splinter.turn, reply) is not None
                if stored:
                    st.save_conversation(req.conversation_id, splinter)
            # the harvest: the turn, with the curation that was used, for
            # the encoder's training data.  Off unless the user asked.
            try:
                if st.settings().get("harvest_enabled"):
                    from harness import harvest as _harvest

                    _harvest.append_turn(st.state_dir, _harvest.build_record(
                        query=query, reply=reply,
                        conversation_id=req.conversation_id,
                        turn=result.turn,
                        selected=list(getattr(assembled, "selected_chunk_ids", []) or []),
                        curated=curated,
                        candidates=dict(getattr(assembled, "raw_scores", {}) or {}),
                        config=splinter.config.to_dict(),
                        encoder=splinter.config.ultra_model,
                        token_count=result.token_count, budget=result.budget,
                        mode=result.mode))
            except Exception:  # noqa: BLE001 - a failed harvest loses nothing
                pass
            elapsed = max(time.time() - started, 1e-6)
            completion_tokens = (usage or {}).get("completion_tokens") or 0
            yield "data: " + json.dumps({
                "type": "done", "stored": stored,
                "tokens": completion_tokens,
                "seconds": round(elapsed, 2),
                "tokens_per_sec": round(completion_tokens / elapsed, 1)
                if completion_tokens else None,
            }) + "\n\n"

        watcher = asyncio.create_task(watch_client())

        async def stream():
            # The provider read blocks, so it runs in a thread; between
            # chunks the client is checked, and a client that went away
            # closes the provider connection - the engine stops instead of
            # finishing a reply nobody is reading.
            iterator = sse()
            try:
                while True:
                    if await request.is_disconnected():
                        live["stopped"] = True
                        close_live()
                        break
                    chunk = await anyio.to_thread.run_sync(
                        lambda: next(iterator, None))
                    if chunk is None:
                        break
                    yield chunk
            finally:
                live["stopped"] = True
                close_live()
                await anyio.to_thread.run_sync(iterator.close)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/agent/stream")
    async def agent_stream(req: AgentMessageRequest):
        """Stream an agent (dsh) turn with tool use over SSE."""
        message = (req.message or "").strip()
        if not message:
            raise HTTPException(422, "message must not be empty")
        if not _load_studio_settings()["tools_enabled"]:
            raise HTTPException(403, "tool use is off (Studio settings)")
        try:
            provider = st.registry.resolve(None)
        except LookupError as exc:
            raise HTTPException(502, str(exc))
        base_url = provider.base_url.rstrip("/") + "/v1"
        q: "queue.Queue[dict]" = queue.Queue()

        prompt = message
        if req.assistant:
            prompt = _ASSISTANT_CONTEXT + "\n---\n" + message

        def worker():
            st.begin(req.conversation_id)
            try:
                out = agent_service.run_turn(
                    req.conversation_id, prompt,
                    base_url=base_url,
                    api_key=provider.api_key or "lm-studio",
                    model=provider.model or "local",
                    on_event=q.put,
                )
                q.put({"type": "done", **out})
            except Exception as exc:  # noqa: BLE001 - surfaced as an event
                q.put({"type": "error", "error": str(exc)})
            finally:
                st.end(req.conversation_id)

        threading.Thread(target=worker, daemon=True).start()

        def sse():
            while True:
                event = q.get()
                yield "data: " + json.dumps(event, default=str) + "\n\n"
                if event.get("type") in ("done", "error"):
                    return

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.get("/v1/agent/status")
    def agent_status():
        """Agent service state: running, model, last error, gateway status."""
        from harness.security import SecurityStore

        return {"runtime_running": agent_service.runtime_running,
                "permission_policy": agent_service.permission_policy,
                # True only after an external DSH runtime passed the hook
                # preflight; a configured-but-unverified hook stays false.
                "gateway_enforced": agent_service.gateway_enforced,
                "gateway_configured": agent_service.gateway_configured,
                "gateway_hook": agent_service.hook_status,
                "security_tier": SecurityStore(_state_paths()).read()["tier"],
                "busy": agent_service._inflight,
                "ceased": (app.state.activity.cease.list()
                           if app.state.activity else []),
                "activity_rules_enabled":
                    (app.state.activity.rules.read()["enabled"]
                     if app.state.activity else False)}

    @app.post("/v1/agent/cancel")
    def agent_cancel():
        """Cancel the in-flight agent turn for a conversation."""
        return agent_service.cancel()

    # ------------------------------------------------------------------
    # Activity ledger, live feed, tripwires, and emergency cease
    def _activity_hub() -> ActivityHub:
        hub = getattr(app.state, "activity", None)
        if hub is None:
            raise HTTPException(
                503, "activity ledger is disabled (no persistent state dir)")
        return hub

    @app.get("/v1/activity")
    def activity_recent(
        conversation_id: Optional[str] = Query(default=None),
        limit: int = Query(default=200, ge=1, le=2000),
    ):
        """Recent activity events, current tripwire rules, and cease blocks."""
        hub = _activity_hub()
        return {
            "events": hub.ledger.read_file(conversation_id, limit=limit),
            "rules": hub.rules.read(),
            "blocks": hub.cease.list(),
        }

    @app.get("/v1/activity/stream")
    async def activity_stream(
        request: Request,
        conversation_id: Optional[str] = Query(default=None),
    ):
        """Live activity feed (SSE).  Events are the same records as JSONL."""
        hub = _activity_hub()
        subscriber = hub.ledger.subscribe()

        async def sse():
            try:
                yield ": connected\n\n"
                last_keepalive = time.monotonic()
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = subscriber.get_nowait()
                    except queue.Empty:
                        if time.monotonic() - last_keepalive >= 10:
                            yield ": keepalive\n\n"
                            last_keepalive = time.monotonic()
                        await anyio.sleep(0.2)
                        continue
                    if conversation_id and \
                            event.get("conversation_id") != conversation_id:
                        continue
                    yield "data: " + json.dumps(event, default=str) + "\n\n"
            finally:
                hub.ledger.unsubscribe(subscriber)

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.post("/v1/activity/control")
    def activity_control(body: dict[str, Any] = Body(default={})):
        """Cease a session (block + stop + revoke) or resume it explicitly."""
        hub = _activity_hub()
        action = str(body.get("action") or "cease").strip().lower()
        conversation_id = str(body.get("conversation_id") or "").strip()
        session_id = str(body.get("session_id") or "").strip()
        reason = str(body.get("reason") or "user requested").strip()
        source = str(body.get("source") or "studio").strip()
        try:
            if action == "cease":
                return hub.cease_now(
                    conversation_id=conversation_id,
                    session_id=session_id,
                    reason=reason or "user requested",
                    source=source,
                    rule=str(body.get("rule") or ""),
                    revoke_grants=body.get("revoke_grants") is not False,
                    revoke_all=body.get("revoke_all") is True,
                    force=body.get("force") is not False,
                )
            if action == "resume":
                return hub.resume(
                    conversation_id=conversation_id,
                    session_id=session_id,
                    source=source,
                )
        except ActivityError as exc:
            raise HTTPException(422, str(exc)) from exc
        raise HTTPException(422, "action must be cease or resume")

    @app.get("/v1/activity/rules")
    def activity_rules():
        """The tripwire rules and whether automatic evaluation is enabled."""
        return _activity_hub().rules.read()

    @app.put("/v1/activity/rules")
    def activity_rules_put(body: dict[str, Any] = Body(default={})):
        """Replace the tripwire rules, or toggle them with ``enabled``."""
        hub = _activity_hub()
        rules = body.get("rules")
        try:
            if rules is None:
                if body.get("enabled") is None:
                    raise HTTPException(422, "rules or enabled is required")
                return hub.rules.set_enabled(bool(body["enabled"]))
            if not isinstance(rules, list):
                raise HTTPException(422, "rules must be a list")
            return hub.rules.write(rules, enabled=body.get("enabled"))
        except ActivityError as exc:
            raise HTTPException(422, str(exc)) from exc

    # ------------------------------------------------------------------
    # Preset trainer (X15-X18)
    @app.get("/v1/trainer/evidence")
    def trainer_evidence():
        """Mine session logs for tool-use patterns and outcomes."""
        session_root = dsh_session_root()
        all_tools = ["bash", "str_replace_editor", "fs_search", "web",
                     "subagent", "todo", "code_runtime", "skill"]
        evidences = mine_evidence(session_root, all_tools)
        return {"evidence": [e.to_dict() for e in evidences],
                "summary": summarize_evidence(evidences)}

    @app.post("/v1/trainer/draft")
    def trainer_draft(body: dict):
        """Draft a candidate preset from the evidence."""
        name = body.get("name") or f"candidate-{int(time.time())}"
        session_root = dsh_session_root()
        all_tools = body.get("all_tools") or [
            "bash", "str_replace_editor", "fs_search", "web",
            "subagent", "todo", "code_runtime", "skill"]
        evidences = mine_evidence(session_root, all_tools)
        summary = summarize_evidence(evidences)
        baseline = Path(body.get("baseline") or "")
        if not baseline.is_file():
            raise HTTPException(422, f"baseline preset not found: {baseline}")
        output = REPO_ROOT / "harness_state" / "trainer_candidates"
        candidate = draft_candidate(summary, baseline, output, name)
        return candidate.to_dict()

    @app.post("/v1/trainer/evaluate")
    def trainer_evaluate(body: dict):
        """Run candidate vs baseline on bench tasks, score, compare."""
        candidate_path = Path(body.get("candidate") or "")
        meta_path = candidate_path.with_suffix("").with_suffix(".trainer-meta.json") \
            if candidate_path.suffix == ".yml" else \
            candidate_path.parent / (candidate_path.stem + ".trainer-meta.json")
        if not meta_path.is_file():
            raise HTTPException(404, f"trainer meta not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        from deepseek_harness import DeepSeekHarness

        def factory(config_path):
            return DeepSeekHarness(
                provider="deepseek-official",
                base_url=os.environ.get("HARNESS_EMBEDDING_URL",
                                        "http://127.0.0.1:1235/v1"),
                api_key="lm-studio",
                model=os.environ.get("HARNESS_AGENT_MODEL", "default"),
                cwd=str(REPO_ROOT),
                session_root=str(dsh_session_root()),
                cordis=str(config_path),
            )

        tasks = body.get("tasks") or [
            "List the files in the current directory.",
            "Create a file named trainer-test.txt containing 'hello'.",
            "Search for the word 'splinter' in .py files and report matches.",
        ]
        candidate = draft_candidate.__wrapped__ if hasattr(
            draft_candidate, "__wrapped__") else None
        from harness.trainer import CandidatePreset

        cand = CandidatePreset(
            name=candidate_path.stem, baseline=meta.get("baseline", ""),
            changes=meta.get("changes", {}),
            evidence_summary=meta.get("evidence_summary", {}),
            path=str(candidate_path),
        )
        results = evaluate_candidate(cand, factory, tasks)
        meta["eval_result"] = results
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return results

    @app.get("/v1/server/status")
    def server_status():
        """Hardware (VRAM/RAM) plus loaded model health and instances."""
        data = models_manager.status()
        try:
            data["hardware"] = _hardware_summary()
        except Exception:
            data["hardware"] = {
                "available_gb": 8.0, "total_ram_gb": 8.0,
                "vram_gb": None, "available_ram_gb": 8.0,
                "devices": [], "vram_source": "ram",
            }
        # the Strata engine is a separate serve: the Studio does not launch
        # it, but it is a loaded model the user is talking to.
        strata = _probe_strata()
        data["strata"] = strata
        loaded = [{
            "backend": "llama.cpp", "key": inst.get("key"),
            "model": inst.get("model"), "port": inst.get("port"),
            "base_url": inst.get("base_url"), "healthy": bool(inst.get("healthy")),
        } for inst in data.get("instances") or []]
        if strata.get("alive") and strata.get("loaded"):
            # alive is not loaded: an engine whose model was unloaded serves
            # nothing, so it must not be listed as a loaded model
            loaded.append({
                "backend": "strata", "key": "strata",
                "model": strata.get("model"), "port": 8081,
                "base_url": strata.get("base_url"), "healthy": True,
            })
        data["loaded"] = loaded
        data["loaded_model"] = (loaded[0]["model"] if loaded else None)
        # how warm the loaded model's file is in the page cache.  This is
        # the number VRAM cannot show: a 132 GB model streaming its
        # experts evicts the other engine's weights from the same cache,
        # so the second engine starts cold and runs slow without either
        # number looking wrong.
        for item in loaded:
            try:
                item["cache_share"] = page_cache_share(item.get("model"))
            except Exception:  # noqa: BLE001 - the readout is best effort
                item["cache_share"] = None
        return data

    @app.get("/v1/server/fit")
    def server_fit(model_gb: float = 0.0, ctx: int = 8192, kv: str = "q8_0"):
        """Where a model and its KV cache fit, and the context that fits.

        Three tiers in the order the memory is used (the VRAM, the RAM, the
        disk) plus the two flags that matter when launching: the disk will
        hurt the prompt read, and the disk is nearly full.
        """
        from harness.hardware import fit_plan

        hw = _hardware_summary()
        vram = float(hw.get("combined_vram_gb") or hw.get("vram_gb") or 0.0)
        ram = float(hw.get("available_ram_gb") or hw.get("total_ram_gb") or 0.0)
        disk = (hw.get("disk") or {})
        repo_disk = disk.get("repo") or {}
        disk_free = repo_disk.get("free_gb")
        return fit_plan(model_gb, ctx, vram_gb=vram, ram_gb=ram, kv_type=kv,
                        disk_free_gb=disk_free)

    @app.get("/v1/server/log")
    def server_log(tail: int = 120):
        """Last lines of the llama-server logs (startup errors live here)."""
        return {"lines": models_manager.server_log(tail)}

    @app.get("/v1/gpu/processes")
    def gpu_processes_endpoint():
        """Per-process VRAM (from /proc/<pid>/fdinfo), on-demand.

        Kept out of /v1/server/status so the hardware poll stays cheap."""
        try:
            from harness.hardware import gpu_processes
            return {"processes": gpu_processes()}
        except Exception as exc:  # noqa: BLE001
            return {"processes": [], "error": str(exc)}

    @app.get("/v1/server/memory")
    def server_memory():
        """Sidecar RSS + conversation accounting — the leak-detection probe."""
        import psutil

        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        return {
            "rss_mb": round(rss_mb, 1),
            "conversations_in_memory": len(st.hives),
            "max_conversations": st.max_conversations,
            "loggers": len(st._loggers),
            "downloads": len(models_manager._downloads),
            "threads": threading.active_count(),
        }

    @app.get("/v1/server/metrics")
    def server_metrics():
        """Proxy llama-server's prometheus /metrics (kv usage, throughput)."""
        return models_manager.server_metrics()

    @app.get("/v1/processes")
    def list_processes():
        """Process manager: spawned servers (llama-servers) + child shells.

        Each entry carries pid, name, cmdline, cpu_percent, memory_mb, and a
        kill affordance. The table powers the S4 Studio panel (CPU/RAM + kill
        buttons)."""
        import psutil

        processes: list[dict] = []
        # snapshot of managed llama-servers
        try:
            status = models_manager.status()
            instances = status.get("instances", []) or []
        except Exception:
            instances = []
        pid_to_inst = {int(inst.get("pid")): inst for inst in instances if inst.get("pid")}
        # helper to build one row
        def row_for_pid(pid: int, kind: str, inst: Optional[dict] = None) -> Optional[dict]:
            try:
                proc = psutil.Process(pid)
                with proc.oneshot():
                    name = proc.name() or ""
                    try:
                        cmdline = proc.cmdline() or []
                    except psutil.AccessDenied:
                        cmdline = []
                    try:
                        cpu = float(proc.cpu_percent(interval=None))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        cpu = 0.0
                    try:
                        rss = proc.memory_info().rss
                        mem_mb = round(rss / (1024 * 1024), 1)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        mem_mb = 0.0
                    try:
                        p_status = proc.status()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        p_status = "unknown"
                return {
                    "pid": pid,
                    "key": inst.get("key") if inst else None,
                    "kind": kind,
                    "name": name,
                    "cmdline": cmdline,
                    "cpu_percent": round(cpu, 1),
                    "memory_mb": mem_mb,
                    "status": p_status,
                    "port": inst.get("port") if inst else None,
                    "embedding": bool(inst.get("embedding")) if inst else False,
                    "adopted": bool(inst.get("adopted")) if inst else False,
                }
            except psutil.NoSuchProcess:
                if inst is not None:
                    return {
                        "pid": pid,
                        "key": inst.get("key"),
                        "kind": kind,
                        "name": "llama-server",
                        "cmdline": [],
                        "cpu_percent": 0.0,
                        "memory_mb": 0.0,
                        "status": "not-found",
                        "port": inst.get("port"),
                        "embedding": bool(inst.get("embedding")),
                        "adopted": bool(inst.get("adopted")),
                    }
                return None
            except psutil.AccessDenied:
                return {
                    "pid": pid,
                    "key": inst.get("key") if inst else None,
                    "kind": kind,
                    "name": "unknown",
                    "cmdline": [],
                    "cpu_percent": 0.0,
                    "memory_mb": 0.0,
                    "status": "access-denied",
                    "port": inst.get("port") if inst else None,
                    "embedding": bool(inst.get("embedding")) if inst else False,
                    "adopted": bool(inst.get("adopted")) if inst else False,
                }

        # llama-server rows first (so UI shows servers at top)
        for inst in instances:
            pid = inst.get("pid")
            if pid is None:
                continue
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            row = row_for_pid(pid_int, "llama-server", inst)
            if row:
                processes.append(row)
        # child processes of the sidecar (shells, etc.) not already listed
        try:
            me = psutil.Process()
            children = me.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []
        seen_pids = {p["pid"] for p in processes}
        for child in children:
            try:
                cpid = int(child.pid)
            except (TypeError, ValueError):
                continue
            if cpid in seen_pids or cpid in pid_to_inst:
                continue
            row = row_for_pid(cpid, "shell", None)
            if row:
                processes.append(row)
        # sidecar itself
        try:
            me = psutil.Process()
            with me.oneshot():
                sidecar_cpu = float(me.cpu_percent(interval=None))
                sidecar_mem = round(me.memory_info().rss / (1024 * 1024), 1)
                sidecar_status = me.status()
                sidecar_name = me.name() or "sidecar"
                sidecar_pid = int(me.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            sidecar_cpu = 0.0
            sidecar_mem = 0.0
            sidecar_status = "unknown"
            sidecar_name = "sidecar"
            try:
                sidecar_pid = int(psutil.Process().pid)
            except Exception:
                sidecar_pid = 0
        return {
            "processes": sorted(processes, key=lambda p: (0 if p["kind"] == "llama-server" else 1, p["pid"])),
            "sidecar": {
                "pid": sidecar_pid,
                "name": sidecar_name,
                "cpu_percent": round(sidecar_cpu, 1),
                "memory_mb": sidecar_mem,
                "status": sidecar_status,
            },
        }

    @app.post("/v1/processes/kill")
    async def kill_process(request: Request):
        """Kill one process by pid or by managed key (S4 kill button).

        Body: {"pid": 1234} or {"key": "bge"} or both (key preferred when
        it matches a managed instance so unload() is used and provider
        cleanup runs)."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(422, "invalid JSON")
        pid_raw = body.get("pid")
        key = str(body.get("key") or "").strip() if body.get("key") else None
        # key path: managed llama-server
        if key:
            try:
                result = models_manager.unload(key)
                return {"ok": True, "killed": result.get("unloaded", key), "by": "key"}
            except RuntimeError as exc:
                # key not a managed instance — fall through to pid kill if pid given
                if pid_raw is None:
                    raise HTTPException(404, str(exc))
        if pid_raw is None:
            raise HTTPException(422, "pid or key is required")
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            raise HTTPException(422, "pid must be an integer")
        if pid <= 0:
            raise HTTPException(422, "invalid pid")
        # try to map pid to a managed instance first (so unload path is used)
        try:
            status = models_manager.status()
            for inst in status.get("instances", []) or []:
                if int(inst.get("pid") or -1) == pid:
                    k = inst.get("key")
                    try:
                        models_manager.unload(k)
                        return {"ok": True, "killed": pid, "by": "key", "key": k}
                    except RuntimeError:
                        break
        except Exception:
            pass
        # generic pid kill via psutil
        import psutil

        try:
            proc = psutil.Process(pid)
            # guard: do not kill the sidecar itself or pid 1
            me = psutil.Process()
            if pid == int(me.pid) or pid == 1:
                raise HTTPException(400, "refusing to kill sidecar or init")
            name = ""
            try:
                name = proc.name() or ""
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                name = ""
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            return {"ok": True, "killed": pid, "name": name, "by": "pid"}
        except psutil.NoSuchProcess:
            raise HTTPException(404, f"no such process: {pid}")
        except psutil.AccessDenied as exc:
            raise HTTPException(403, f"access denied killing {pid}: {exc}")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, str(exc))

    @app.delete("/v1/models/local")
    def delete_local_model(file: str = Query(...)):
        """Delete one GGUF from the local library."""
        try:
            out=models_manager.delete_local(file)
            models_manager.invalidate_cache()
            return out
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc))

    def _wsl_to_windows(path: str) -> str:
        # /mnt/c/Users/... -> C:\Users\...
        if path.startswith("/mnt/") and len(path) > 6 and path[6] == "/":
            drive = path[5].upper()
            rest = path[7:].replace("/", "\\")
            return f"{drive}:\\{rest}"
        return path

    def _windows_to_wsl(path: str) -> str:
        # C:\Users\... -> /mnt/c/Users/...
        if len(path) >= 2 and path[1] == ":":
            drive = path[0].lower()
            rest = path[2:].lstrip("\\/").replace("\\", "/")
            return f"/mnt/{drive}/{rest}"
        return path.replace("\\", "/")

    @app.get("/v1/models/linux")
    def list_linux_models():
        """List GGUFs in /mnt/dsh_storage/models (WSL ext4)."""
        import time as _time
        t0=_time.time()
        try:
            out = subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", "find /mnt/dsh_storage/models -type f \\( -name '*.gguf' -o -name '*.GGUF' \\) -exec stat -c '%s %n' {} \\; 2>/dev/null | head -n 200"],
                text=True, timeout=10, stderr=subprocess.STDOUT, creationflags=_NO_WINDOW,
            )
        except Exception as exc:
            # Not mounted or WSL not running -> empty list, not 500
            print(f"list_linux_models: find failed in {(_time.time()-t0):.2f}s: {exc}")
            return {"models_dir": "/mnt/dsh_storage/models", "models": [], "error": str(exc)[:300], "mounted": False}
        files: list[dict] = []
        for line in out.strip().splitlines():
            line=line.strip()
            if not line:
                continue
            # stat -c '%s %n' -> "12345 /mnt/dsh_storage/models/foo/bar.gguf"
            try:
                sz_str, path = line.split(" ", 1)
                size_gb = round(int(sz_str) / (1024**3), 2)
                path=path.strip()
            except Exception:
                continue
            if not path.lower().endswith(".gguf"):
                continue
            try:
                rel = path[len("/mnt/dsh_storage/models/"): ] if path.startswith("/mnt/dsh_storage/models/") else Path(path).name
            except Exception:
                rel = Path(path).name
            files.append({
                "name": Path(path).stem,
                "file": rel,
                "path": path,
                "size_gb": size_gb,
                "sizeGb": size_gb,
                "location": "linux",
            })
        print(f"list_linux_models: {len(files)} files in {(_time.time()-t0):.2f}s")
        # sort by name
        files.sort(key=lambda x: x["file"].lower())
        return {"models_dir": "/mnt/dsh_storage/models", "models": files, "mounted": True}

    @app.post("/v1/models/move-to-linux")
    async def move_to_linux(request: Request):
        """Move a System GGUF to /mnt/dsh_storage/models via WSL."""
        try:
            body = await request.json()
            file = str(body.get("file") or "").strip()
        except Exception:
            file = ""
        if not file:
            raise HTTPException(422, "file is required (relative to models_dir)")
        src = models_manager.resolve_model(file)
        if src is None or not src.is_file():
            raise HTTPException(404, f"model not found: {file}")
        # ensure linux models dir exists
        try:
            subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", "sudo mkdir -p /mnt/dsh_storage/models && sudo chmod 777 /mnt/dsh_storage/models"],
                text=True, timeout=5, stderr=subprocess.STDOUT, creationflags=_NO_WINDOW,
            )
        except Exception as exc:
            raise HTTPException(500, f"cannot prepare linux models dir: {exc}")
        wsl_src = _windows_to_wsl(str(src))
        fname = Path(src).name
        dest_wsl = f"/mnt/dsh_storage/models/{fname}"
        # Use wsl cp with sudo, then verify
        try:
            # copy, not move, so original stays tracked until verified
            subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", f"sudo cp '{wsl_src}' '{dest_wsl}' && sudo chmod 644 '{dest_wsl}'"],
                text=True, timeout=300, stderr=subprocess.STDOUT, creationflags=_NO_WINDOW,
            )
        except subprocess.CalledProcessError as exc:
            raise HTTPException(500, f"copy to linux failed: {(exc.output or str(exc))[:600]}")
        except Exception as exc:
            raise HTTPException(500, f"copy to linux failed: {str(exc)[:600]}")
        # verify dest exists
        try:
            subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", f"test -f '{dest_wsl}' && stat -c %s '{dest_wsl}'"],
                text=True, timeout=5, stderr=subprocess.STDOUT, creationflags=_NO_WINDOW,
            )
        except Exception as exc:
            raise HTTPException(500, f"verify failed: {exc}")
        models_manager.invalidate_cache()
        return {"ok": True, "file": fname, "src": str(src), "dest": dest_wsl, "location": "linux"}

    @app.post("/v1/models/move-to-windows")
    async def move_to_windows(request: Request):
        """Move a Linux GGUF back to System library (host models_dir)."""
        try:
            body = await request.json()
            file = str(body.get("file") or "").strip()
        except Exception:
            file = ""
        if not file:
            raise HTTPException(422, "file is required (relative to /mnt/dsh_storage/models)")
        # sanitize: no traversal
        if ".." in file or file.startswith("/") or file.startswith("\\"):
            raise HTTPException(400, "invalid file")
        wsl_path = f"/mnt/dsh_storage/models/{file}"
        # check exists in linux
        try:
            subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", f"test -f '{wsl_path}'"],
                text=True, timeout=5, stderr=subprocess.STDOUT, creationflags=_NO_WINDOW,
            )
        except Exception:
            raise HTTPException(404, f"linux model not found: {file}")
        dest_win = models_manager.models_dir / Path(file).name
        dest_wsl = _windows_to_wsl(str(dest_win))
        try:
            subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", f"sudo cp '{wsl_path}' '{dest_wsl}'"],
                text=True, timeout=300, stderr=subprocess.STDOUT, creationflags=_NO_WINDOW,
            )
        except Exception as exc:
            raise HTTPException(500, f"copy to windows failed: {str(exc)[:600]}")
        if not dest_win.is_file():
            raise HTTPException(500, "copy verified failed on host")
        models_manager.invalidate_cache()
        return {"ok": True, "file": Path(file).name, "src": wsl_path, "dest": str(dest_win), "location": "system"}

    @app.delete("/v1/models/linux")
    def delete_linux_model(file: str = Query(...)):
        """Delete a GGUF from /mnt/dsh_storage/models."""
        if ".." in file or file.startswith("/") or file.startswith("\\"):
            raise HTTPException(400, "invalid file")
        wsl_path = f"/mnt/dsh_storage/models/{file}"
        try:
            subprocess.check_output(
                ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", f"sudo rm -f '{wsl_path}' && echo ok"],
                text=True, timeout=10, stderr=subprocess.STDOUT, creationflags=_NO_WINDOW,
            )
        except Exception as exc:
            raise HTTPException(500, f"delete failed: {str(exc)[:400]}")
        return {"ok": True, "deleted": file, "location": "linux"}


    @app.get("/v1/research/queue")
    def research_queue_get():
        """Pending deep-research questions. Execution is QUEEN-only: entries are
        picked up when the primary session next wakes."""
        if not RESEARCH_QUEUE.exists():
            return {"items": []}
        items = []
        for line in RESEARCH_QUEUE.read_text(encoding="utf-8-sig").splitlines():
            s = line.strip()
            if s.startswith("- [ ] "):
                items.append(s[6:])
        return {"items": items}

    @app.post("/v1/research/queue")
    async def research_queue_add(req: Request):
        """Queue a deep-research question for the background worker."""
        body = await req.json()
        q = str(body.get("question", "")).strip()
        if not q:
            raise HTTPException(422, "question is required")
        q = q[:500]
        with RESEARCH_QUEUE.open("a", encoding="utf-8") as fh:
            fh.write(f"- [ ] {q}\n")
        return {"queued": True, "question": q}


    # AFK mode removed (shelved). The idea: QUEEN autonomy toggle for
    # unattended operation. Revisit when we revisit the shelf.

    @app.post("/v1/server/stop")
    def server_stop(key: Optional[str] = Query(default=None)):
        """Unload one instance (key) or the whole local fleet (no key)."""
        if key:
            try:
                return models_manager.unload(key)
            except RuntimeError as exc:
                raise HTTPException(404, str(exc))
        for inst in list(models_manager.status()["instances"]):
            register_local_remove(inst["key"])
        return models_manager.stop_all()

    @app.post("/v1/server/start")
    def server_start(req: ServerStartRequest):
        """Launch a local model server (llama.cpp/strata) with the given options."""
        # embedding validation mirrors manager.load pooling check
        if req.pooling is not None and not req.embedding:
            raise HTTPException(422, "pooling requires embedding=True")
        if req.pooling is not None and req.pooling.strip().lower() not in (
                "mean", "cls", "last"):
            raise HTTPException(422, "unknown pooling; known: mean, cls, last")
        # the Global tab's Runtime block is the baseline a launch starts
        # from; the request's own value wins.  The fallbacks are the
        # manager's own defaults (LlamaServerManager.load).
        launch = st.controls()["launch"]
        ctx_size = req.ctx_size if req.ctx_size is not None else launch.get("ctx_size", 8192)
        ngl = req.ngl if req.ngl is not None else launch.get("ngl", 999)
        threads = req.threads if req.threads is not None else launch.get("threads")
        try:
            env = None
            visible = req.visible_devices
            if not visible:
                # a launch that does not name a card takes the spare one:
                # the display's card belongs to the desktop, and taking it
                # makes the window stutter (or fails outright when the
                # display's VRAM is nearly full)
                spare = non_display_index()
                if spare is not None:
                    visible = str(spare)
            if visible:
                vis_env = ("CUDA_VISIBLE_DEVICES"
                           if (req.backend or "").strip().lower() == "cuda"
                           else "HIP_VISIBLE_DEVICES")
                env = {vis_env: visible}
            # Ask the kernel to read the weights into the page cache
            # first: a model that fits in the cache (the 35B is 11 GB,
            # the cache is 13) then starts warm instead of streaming
            # itself off the disk.  The kernel reads at its own pace and
            # takes the pages back when the machine needs the RAM.
            if req.model:
                try:
                    prefetch_weights(str(req.model))
                except Exception:  # noqa: BLE001 - a warm-up is a nicety
                    pass
            info = models_manager.load(
                model=req.model, hf_repo=req.hf_repo, hf_file=req.hf_file,
                key=req.key, port=req.port, ctx_size=ctx_size,
                ngl=("auto" if req.auto_fit else ngl),
                extra_args=req.extra_args(threads=threads),
                env=env,
                backend=req.backend,
                embedding=bool(req.embedding),
                pooling=req.pooling.strip().lower() if req.pooling else None,
                mmproj=req.mmproj,
            )
        except RuntimeError as exc:
            message = str(exc)
            code = 502
            if "already loaded" in message or "already serving" in message:
                code = 409
            elif "not found at" in message or "neither a local file" in message:
                code = 400
            raise HTTPException(code, message)
        prov_name = ""
        if req.register_provider:
            prov_name = register_local(
                info,
                load_options=req.load_options(ctx_size=ctx_size, ngl=ngl,
                                              threads=threads),
                api_key=req.api_key,
                key=info["key"], claim_default=bool(req.claim_default),
            )
            save_registry(st.registry, st.providers_file)
        return {**info, "provider": prov_name,
                "provider_registered": bool(req.register_provider)}

    def _strata_repo() -> str:
        """The Strata checkout: the setting, else the sibling checkout.

        The engine is launched from its own repository (the serve, the
        configs, the packs), so the Studio needs to know where that is.
        """
        configured = (_load_studio_settings().get("strata_repo") or "").strip()
        if configured:
            return configured
        for candidate in (REPO_ROOT / "strata", REPO_ROOT.parent / "strata"):
            if candidate.is_dir():
                return str(candidate)
        return ""

    @app.get("/v1/strata/configs")
    def strata_configs():
        """The Strata config profiles in the checkout (the panel's picker)."""
        repo = Path(_strata_repo())
        out: dict = {"repo": str(repo), "configs": []}
        if not repo.is_dir():
            return out
        for path in sorted(repo.glob("strata-*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            out["configs"].append({
                "file": path.name,
                "model_name": data.get("model_name") or "",
                "backend": data.get("backend") or "",
                "port": data.get("port"),
                "gpus": data.get("gpu") or [],
                "args": len(data.get("args") or []),
            })
        return out

    @app.post("/v1/strata/launch")
    def strata_launch(req: StrataLaunchRequest):
        """Start the Strata engine from the Launch panel.

        The Studio owns the process it starts: the unload stops it again
        (an engine the user started by hand is never killed - it is only
        asked to give its VRAM back).
        """
        from harness.stack.schema import Tier
        from harness.stack.strata import StrataLauncher

        # The config is required regardless of the host's checkout, so reject
        # its absence first; otherwise this answer depends on whether the
        # machine happens to have a Strata sibling.
        config = (req.config or "").strip()
        if not config:
            raise HTTPException(400, "a strata config profile is required")
        repo = (req.repo or _strata_repo()).strip()
        if not repo:
            raise HTTPException(400, "no strata checkout: set strata_repo in the "
                                       "Studio settings or give a repo path")
        if not Path(repo).is_dir():
            # The checkout is checked before anything else: the launch
            # adopts a live engine on the port (so a second engine never
            # starts), and without this a bogus path would quietly get
            # the running engine instead of an error.
            raise HTTPException(400, f"no strata checkout at {repo!r}: the "
                                       "path must be the strata repository")
        port = int(req.port or 8081)
        endpoint = f"http://127.0.0.1:{port}"
        # The launch baseline's resident budget: the Strata engine's resident
        # tier, which keeps the experts the VRAM cache cannot hold in ordinary
        # RAM so they are never re-read from the disk.  Measured on the 176B:
        # 8 GiB took the cold first request from 33.8 s to 17.3 s and the
        # storage read on the second request from 1,415 MB to 204 MB, with the
        # warm rate unchanged.  0 (the default) leaves the engine's own
        # default alone.
        engine_args = [str(a) for a in (req.engine_args or [])]
        resident_gib = st.controls()["launch"].get("resident_gib")
        if resident_gib:
            try:
                gib = float(resident_gib)
            except (TypeError, ValueError):
                gib = 0.0
            if gib > 0:
                engine_args += ["--resident-budget-gib", str(gib)]
        launch = {
            "kind": "strata",
            "repo": repo,
            "config": config,
            "python": req.python or ".venv/bin/python",
            "engine_args": engine_args,
            "options": dict(req.options or {}),
            "timeout_s": float(req.timeout_s or 600),
        }
        tier = Tier(role="strata", repo=repo, file="",
                    ctx=int(req.ctx or st.controls()["launch"].get("ctx_size")
                            or 8192),
                    launch=launch)
        # adopt a serve that is already up: never start a second one on a
        # port that already answers (the same contract the stacks use)
        try:
            from harness.stack.endpoint import probe_endpoint

            health = probe_endpoint(endpoint)
            _register_strata_provider({"alive": True, "base_url": endpoint,
                                       "model": ""})
            return {"started": False, "adopted": True, "port": port,
                    "base_url": endpoint + "/v1", "health": health}
        except Exception:  # noqa: BLE001 - not running: we start it
            pass
        launcher = getattr(stack_manager, "strata_launcher", None)
        if launcher is None:
            launcher = StrataLauncher(
                state_dir=(st.state_dir / "strata") if st.state_dir else None,
                log_dir=REPO_ROOT / "logs",
            )
        try:
            info = launcher.start(stack_name="studio", role="strata",
                                  tier=tier, endpoint=endpoint)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))
        # the chat has to be able to reach it
        _register_strata_provider({"alive": True, "base_url": endpoint,
                                   "model": (req.options or {}).get("model_name")
                                   or ""})
        return {"started": True, "port": port, "base_url": endpoint + "/v1",
                **info}

    @app.post("/v1/server/unload-loaded")
    def server_unload_loaded(drain_timeout: float = 60.0):
        """Free the VRAM: unload whatever model is loaded.

        This is not the same as stopping a turn (that is the Stop button in
        the chat).  It releases the model itself: the local llama.cpp
        instances and the Strata engine (the 176B serve frees its own VRAM
        and reloads on the next request).  The engine profiles and the
        providers stay, so Start works again right away.
        """
        out: dict = {"instances": [], "retired": [], "strata": None}
        for inst in list(models_manager.status()["instances"]):
            try:
                models_manager.unload(inst["key"])
                out["instances"].append(inst["key"])
                # the provider and the engine profile of a dead server are
                # orphans: retire them with it
                register_local_remove(inst["key"])
                out["retired"].append(f"local-{inst['key']}")
            except RuntimeError:
                pass
        # providers whose server is gone (an instance that died on its own)
        for name in _retire_orphan_providers():
            if name not in out["retired"]:
                out["retired"].append(name)
        base = _strata_base()
        root = base[:-3] if base.endswith("/v1") else base
        # Drain first: a 176B reply is minutes of work, and killing the
        # engine mid-generation throws the whole answer away.  The wait is
        # bounded - an engine that never goes idle is stopped anyway.
        out["drained"] = drain_slots(root, timeout=drain_timeout)
        try:
            resp = requests.post(f"{root}/v1/unload", json={}, timeout=15)
            out["strata"] = resp.status_code
            if resp.status_code >= 400:
                # "busy" (a request is running) is a valid answer, not a failure
                out["strata"] = f"{resp.status_code}: {resp.text[:120]}"
        except Exception as exc:  # noqa: BLE001 - "not running" is valid
            out["strata"] = f"not reachable: {exc}"
        # a serve the STUDIO started is the Studio's to stop; one the user
        # started by hand is only asked to give its VRAM back
        launcher = getattr(stack_manager, "strata_launcher", None)
        if launcher is not None and launcher.managed("strata"):
            try:
                launcher.stop("strata")
                out["strata_stopped"] = True
            except Exception as exc:  # noqa: BLE001
                out["strata_stopped"] = f"could not stop: {exc}"
        return out

    @app.post("/v1/server/unload")
    def server_unload(req: ServerUnloadRequest):
        """Unload one instance and retire its provider + engine profile."""
        try:
            result = models_manager.unload(req.key)
        except RuntimeError as exc:
            raise HTTPException(404, str(exc))
        prov_name = f"local-{req.key}"
        st.registry.providers = [
            p for p in st.registry.providers
            if p.name.lower() != prov_name.lower()
        ]
        if st.registry.default == prov_name:
            st.registry.default = next(
                (p.name for p in st.registry.providers), "")
        st.engines.engines = [
            e for e in st.engines.engines if e.name.lower() != prov_name.lower()
        ]
        if st.engines.default == prov_name:
            st.engines.default = next(
                (e.name for e in st.engines.engines), "")
        try:
            save_registry(st.registry, st.providers_file)
            save_engines(st.engines, st.engines_file)
        except OSError:
            pass
        return {**result, "provider": prov_name}

    @app.get("/v1/models/local")
    def local_models():
        """List GGUF models in the local library directory."""
        import time as _time
        t0=_time.time()
        models=models_manager.list_local()
        print(f"list_local: {len(models)} models in {(_time.time()-t0):.2f}s from {models_manager.models_dir}")
        return {"models_dir": str(models_manager.models_dir),
                "models": models}

    @app.post("/v1/embeddings")
    async def create_embeddings(request: Request):
        """OpenAI-compatible embeddings endpoint.

        Proxies to a loaded embedding llama-server (``--embedding``) when
        available, otherwise computes embeddings locally via the splinter's
        ultra-small drone (offline fallback). Accepts the same wire shape
        as ``POST /v1/embeddings`` from llama-server / OpenAI.
        """
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(422, "invalid JSON")
        raw_input = body.get("input")
        if raw_input is None:
            raise HTTPException(422, "input is required")
        # normalize input to list[str]
        if isinstance(raw_input, str):
            inputs: list[str] = [raw_input]
        elif isinstance(raw_input, list):
            if not raw_input:
                raise HTTPException(422, "input must not be empty")
            # reject token-array inputs (list of ints) — not text embeddings
            if any(isinstance(x, int) for x in raw_input):
                raise HTTPException(422, "input must be text, not token ids")
            if any(not isinstance(x, str) for x in raw_input):
                raise HTTPException(422, "input must be string or array of strings")
            inputs = [str(x) for x in raw_input]
        else:
            raise HTTPException(422, "input must be string or array of strings")
        if any(not s.strip() for s in inputs):
            raise HTTPException(422, "input strings must not be empty")
        model_name = str(body.get("model") or body.get("model_name") or "default")
        encoding_format = body.get("encoding_format", "float")
        if encoding_format not in ("float", "base64"):
            raise HTTPException(422, "encoding_format must be float or base64")
        # Try to proxy to a loaded embedding server
        embedding_base: Optional[str] = None
        try:
            status = models_manager.status()
            for inst in status.get("instances", []) or []:
                if inst.get("embedding"):
                    embedding_base = inst.get("base_url")
                    if embedding_base:
                        model_name = str(inst.get("model") or model_name)
                    break
        except Exception:
            embedding_base = None
        # optional env-based served embedding backend
        if embedding_base is None:
            env_url = os.environ.get("HARNESS_EMBEDDING_URL", "").strip()
            if env_url and os.environ.get("HARNESS_EMBEDDING_BACKEND", "") == "served":
                embedding_base = env_url.rstrip("/")
        if embedding_base:
            # proxy to llama-server's /v1/embeddings
            base = embedding_base.rstrip("/")
            # if base already ends with /v1, avoid double
            if base.endswith("/v1"):
                url = f"{base}/embeddings"
            else:
                url = f"{base}/v1/embeddings"
            payload = {"model": model_name, "input": inputs}
            # forward optional OpenAI fields when present
            for k in ("encoding_format", "dimensions", "user"):
                if k in body:
                    payload[k] = body[k]
            try:
                resp = requests.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                # ensure OpenAI shape even if upstream is slightly off
                if "data" in data and isinstance(data["data"], list):
                    return data
                # wrap raw embeddings
                items = data.get("data") or data.get("embeddings") or []
                if items and isinstance(items[0], list):
                    # items is list of vectors
                    wrapped = [{"object": "embedding", "embedding": vec, "index": i}
                               for i, vec in enumerate(items)]
                    return {"object": "list", "data": wrapped, "model": model_name,
                            "usage": {"prompt_tokens": sum(len(s.split()) for s in inputs),
                                      "total_tokens": sum(len(s.split()) for s in inputs)}}
                return data
            except requests.RequestException as exc:
                raise HTTPException(502, f"embedding backend error: {exc}")
        # Fallback: local ultra-small drone
        try:
            drone = st.ultra()
        except Exception as exc:
            raise HTTPException(502, f"no embedding backend available: {exc}")
        vectors: list[list[float]] = []
        for text in inputs:
            try:
                vec = drone.embed(text)
            except Exception as exc:
                raise HTTPException(502, f"embedding failed: {exc}")
            # drone may return numpy array
            try:
                arr = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            except Exception:
                arr = [float(x) for x in vec]
            vectors.append([float(x) for x in arr])
        # encoding_format base64: encode each vector as base64 of float32 bytes
        if encoding_format == "base64":
            import base64
            import struct

            b64_vectors = []
            for vec in vectors:
                packed = struct.pack(f"{len(vec)}f", *vec)
                b64_vectors.append(base64.b64encode(packed).decode("ascii"))
            data = [{"object": "embedding", "embedding": b, "index": i}
                    for i, b in enumerate(b64_vectors)]
        else:
            data = [{"object": "embedding", "embedding": vec, "index": i}
                    for i, vec in enumerate(vectors)]
        prompt_tokens = sum(len(s.split()) for s in inputs)
        return {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }

    @app.post("/v1/models/local/import")
    async def import_local_models(files: List[UploadFile] = File(...)):
        """Import GGUF files from a local folder (webkitdirectory upload).

        Accepts multiple .gguf files via multipart/form-data; preserves
        webkitRelativePath subfolders when present, path-traversal safe.
        """
        imported: List[str] = []
        errors: List[str] = []
        root = models_manager.models_dir.resolve()
        for f in files:
            filename = (f.filename or "").strip()
            if not filename:
                continue
            if not filename.lower().endswith(".gguf"):
                errors.append(f"{filename}: not a .gguf file")
                continue
            # webkitRelativePath may be like "my-models/sub/model.gguf"
            p = Path(filename)
            if p.is_absolute() or ".." in p.parts:
                errors.append(f"{filename}: invalid path")
                continue
            # Resolve destination: try preserving relative path, fallback to basename
            dest = (models_manager.models_dir / p).resolve()
            if root not in dest.parents and dest != root:
                dest = (models_manager.models_dir / p.name).resolve()
                if root not in dest.parents and dest != root:
                    errors.append(f"{filename}: path traversal detected")
                    continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as out:
                    while True:
                        chunk = await f.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                try:
                    rel = str(dest.relative_to(root))
                except ValueError:
                    rel = dest.name
                imported.append(rel)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{filename}: {exc}")
        return {"imported": imported, "errors": errors, "models_dir": str(models_manager.models_dir)}

    @app.post("/v1/models/local/import-path")
    async def import_local_path(request: Request):
        """Link an external folder without copying (instant)."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(422, "invalid JSON")
        folder = str(body.get("folder") or "").strip()
        if not folder:
            raise HTTPException(422, "folder is required")
        try:
            return models_manager.import_local_path(folder)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc))

    @app.post("/v1/models/local/path")
    async def set_local_path(request: Request):
        """Point the library at a chosen folder (no copy, instant).

        Default is LM Studio's ``.lmstudio/models`` when it exists, else
        ``models/gguf`` (created for Hugging Face downloads).
        """
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(422, "invalid JSON")
        folder = str(body.get("folder") or body.get("path") or "").strip()
        if not folder:
            raise HTTPException(422, "folder is required")
        try:
            return models_manager.set_models_dir(folder)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except (NotADirectoryError, PermissionError, ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(400, str(exc))

    @app.post("/v1/agent-presets/copy")
    async def agent_presets_copy(request: Request):
        """Copy an agent preset to a new id."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(422, "invalid JSON")
        src = str(body.get("from") or body.get("source") or "").strip()
        dst = str(body.get("agentPreset") or body.get("id") or "").strip()
        if not src or not dst:
            raise HTTPException(422, "from and agentPreset required")
        # Minimal copy: duplicate directory from shipped presets
        import shutil
        shipped = REPO_ROOT.parent / "hivebench-studio" / "apps" / "cli" / "config" / "agent-presets" / src
        if not shipped.is_dir():
            shipped = Path.home() / ".dsh" / ".agent-presets" / src
        if not shipped.is_dir():
            raise HTTPException(404, f"preset not found: {src}")
        user_root = Path.home() / ".dsh" / ".agent-presets" / dst
        if user_root.exists():
            raise HTTPException(409, f"preset already exists: {dst}")
        try:
            shutil.copytree(shipped, user_root)
        except Exception as exc:
            raise HTTPException(500, str(exc))
        return {"agentPreset": dst}

    @app.post("/v1/agent-presets/remove")
    async def agent_presets_remove(request: Request):
        """Delete a user agent preset."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(422, "invalid JSON")
        pid = str(body.get("agentPreset") or body.get("id") or "").strip()
        if not pid:
            raise HTTPException(422, "agentPreset required")
        target = Path.home() / ".dsh" / ".agent-presets" / pid
        if not target.is_dir():
            raise HTTPException(404, f"not a user preset: {pid}")
        import shutil
        try:
            shutil.rmtree(target)
        except Exception as exc:
            raise HTTPException(500, str(exc))
        return {}

    @app.post("/v1/agent-presets/read")
    async def agent_presets_read(request: Request):
        """Read one agent preset's content."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(422, "invalid JSON")
        pid = str(body.get("agentPreset") or body.get("id") or "").strip()
        if not pid:
            raise HTTPException(422, "agentPreset required")
        # Search shipped then user
        candidates = [
            REPO_ROOT.parent / "hivebench-studio" / "apps" / "cli" / "config" / "agent-presets" / pid / "agent.cordis.yml",
            Path.home() / ".dsh" / ".agent-presets" / pid / "agent.cordis.yml",
        ]
        for cand in candidates:
            if cand.is_file():
                return {"agentPreset": pid, "content": cand.read_text(encoding="utf-8"), "trust": "user" if ".dsh" in str(cand) else "system"}
        raise HTTPException(404, f"preset not found: {pid}")

    @app.get("/v1/agent-presets/list")
    def agent_presets_list():
        """List available agent presets (system and user)."""
        # English display names for built-in presets. The preset.yml files on
        # disk are Chinese (the source shipping language), but the Studio
        # console is English-first, and the DSH web side localizes via
        # locales.ts — so the sidecar also returns English for built-ins to
        # avoid showing Chinese to an English-speaking operator.
        _EN_PRESET = {
            "standard": ("Standard mode",
                         "Full coding agent with file editing, shell, file and web search, skills, planning, goals, subagents, and workflows.", 1),
            "code": ("PTC mode",
                     "All Standard mode capabilities, with tools exposed through the Code Mode SDK so the model can combine multi-step operations in one TypeScript program.", 2),
            "minimal": ("Minimal mode",
                        "Two-tool coding agent with persistent bash and str_replace_editor.", 3),
            "cordis": ("Creator mode",
                       "Built for creating custom agent presets, with all Standard mode capabilities plus runtime inspection, plugin experiments, and preset-authoring guidance.", 4),
            "splinter-curator": ("Splinter Curator",
                             "Standard coding agent plus Splinter curation: each turn the local sidecar assembles relevant context and observes replies, ideal for long tasks and cross-session memory. Requires a local Splinter sidecar; offline it degrades to Standard.", 4),
            "local-first": ("Local-first",
                            "Lean composition for local model routing: no cloud search or other cloud surfaces, keeps shell, filesystem, jobs, skills, goals, and planning. Use with a local inference backend or any in-session model.", 5),
        }

        def _preset_meta(dir_path: Path) -> dict:
            meta = {"name": dir_path.name, "description": None, "order": 999, "broken": None}
            preset_yml = dir_path / "preset.yml"
            cordis_yml = dir_path / "agent.cordis.yml"
            if not cordis_yml.is_file():
                meta["broken"] = "missing agent.cordis.yml"
            if preset_yml.is_file():
                try:
                    text = preset_yml.read_text(encoding="utf-8")
                    for line in text.splitlines():
                        line = line.strip()
                        if line.startswith("name:"):
                            v = line.split(":", 1)[1].strip().strip('"').strip("'")
                            if v:
                                meta["name"] = v
                        elif line.startswith("description:"):
                            v = line.split(":", 1)[1].strip().strip('"').strip("'")
                            if v:
                                meta["description"] = v
                        elif line.startswith("order:"):
                            try:
                                meta["order"] = int(line.split(":", 1)[1].strip())
                            except Exception:
                                pass
                except Exception:
                    pass
            # Override built-ins to English for the English-first Studio console
            if dir_path.name in _EN_PRESET:
                en_name, en_desc, en_order = _EN_PRESET[dir_path.name]
                meta["name"] = en_name
                meta["description"] = en_desc
                meta["order"] = en_order
            return meta

        presets = []
        shipped_root = REPO_ROOT.parent / "hivebench-studio" / "apps" / "cli" / "config" / "agent-presets"
        if shipped_root.is_dir():
            for d in shipped_root.iterdir():
                if d.is_dir() and re.match(r"^[a-z0-9][a-z0-9-]*$", d.name):
                    m = _preset_meta(d)
                    presets.append({
                        "id": d.name,
                        "name": m["name"],
                        "description": m["description"],
                        "trust": "system",
                        "broken": m["broken"],
                        "order": m["order"],
                        "isDefault": d.name == "standard",
                    })
        user_root = Path.home() / ".dsh" / ".agent-presets"
        if user_root.is_dir():
            for d in user_root.iterdir():
                if d.is_dir() and re.match(r"^[a-z0-9][a-z0-9-]*$", d.name):
                    presets = [p for p in presets if p["id"] != d.name]
                    m = _preset_meta(d)
                    presets.append({
                        "id": d.name,
                        "name": m["name"],
                        "description": m["description"],
                        "trust": "user",
                        "broken": m["broken"],
                        "order": m["order"],
                        "isDefault": False,
                    })
        # Order by explicit order then id, user presets keep their order but appear after system by order value
        presets.sort(key=lambda x: (x.get("order", 999), x["id"]))
        # Strip internal order before returning, keep isDefault/broken/description for UI
        for p in presets:
            p.pop("order", None)
            # Ensure description is present even if None -> UI shows id
            if p.get("description") is None:
                p.pop("description", None)
        return {"presets": presets, "authorable": True, "hasDocument": True}

    @app.get("/v1/models/hub")
    def hub_search(q: str = "", limit: int = 12):
        """Search Hugging Face for GGUF repos."""
        try:
            return {"results": models_manager.hub_search(q, limit)}
        except Exception as exc:  # noqa: BLE001 - network errors surface as 502
            raise HTTPException(502, f"hugging face search failed: {exc}")

    @app.get("/v1/models/hub/files/{repo:path}")
    def hub_files(repo: str):
        """List the GGUF files inside a Hugging Face repo."""
        if not repo.strip():
            raise HTTPException(422, "repo must not be empty")
        try:
            return {"repo": repo, "files": models_manager.hub_files(repo)}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"cannot list '{repo}': {exc}")

    @app.post("/v1/models/hub/download")
    def hub_download(req: HubDownloadRequest):
        """Start downloading a model file from Hugging Face."""
        return models_manager.download(req.repo, req.file)

    @app.get("/v1/models/hub/downloads")
    def hub_downloads():
        """Download job states (queued/downloading/done/error)."""
        return {"downloads": models_manager.downloads_status()}

    @app.get("/server", response_class=HTMLResponse)
    def server_page():
        """The Studio console page (chat, models, engines, providers, hub, settings)."""
        return HTMLResponse(render_server_page(), headers=_NO_STORE)

    # ------------------------------------------------------------------
    # Stack tab (T44 mount): the T41 fragment + T42 invocation cards, fed by
    # the §B3 /v1/stacks/* routes registered above.
    @app.get("/stacks", response_class=HTMLResponse)
    def stacks_page():
        """The stack management page."""
        try:
            stacks = list_stacks()
        except Exception:  # noqa: BLE001 - an empty library is a valid page
            stacks = []
        try:
            models = models_manager.list_local()
        except Exception:  # noqa: BLE001 - the picker degrades to empty
            models = []
        tab = stack_tab.render_stack_tab(
            stacks=stacks, models=models, status=stack_manager.status())
        cards = invocation_cards.render_invocation_cards(())
        return HTMLResponse(
            _render_stack_page(tab, cards, stack_tab.css(),
                               invocation_cards.css()),
            headers=_NO_STORE,
        )

    # ------------------------------------------------------------------
    @app.post("/v1/protocol/run")
    def protocol_run(req: ProtocolRunRequest):
        """Run the mock or live HiveBench protocol into a run directory."""
        mode = req.mode if req.mode in ("live", "mock") else ""
        if not mode:
            raise HTTPException(422, "mode must be 'live' or 'mock'")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = st.runs_root / f"protocol_{stamp}"
        cmd = [sys.executable, "-m", "experiments.generate_data",
               f"--{mode}", "--output", str(run_dir)]
        a = req.args or {}
        for key, flag in PROTOCOL_FLAGS_INT.items():
            if key in a:
                cmd += [flag, str(int(a[key]))]
        for key, flag in PROTOCOL_FLAGS_STR.items():
            if key in a and a[key]:
                cmd += [flag, str(a[key])]
        for key, flag in PROTOCOL_FLAGS_BOOL.items():
            if a.get(key):
                cmd.append(flag)
        run_dir.mkdir(parents=True, exist_ok=True)
        # the child inherits the log fd; the parent's copy must be closed
        log_fh = open(run_dir / "run_stdout.log", "ab")
        try:
            proc = _popen(
                cmd, cwd=str(REPO_ROOT),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
        finally:
            log_fh.close()
        return {"run_dir": str(run_dir), "pid": proc.pid}

    # ------------------------------------------------------------------
    # Training tab (engine switch + Hive-Ternary launch/monitor). Unsloth Core
    # is catalogue-only for now; the ternary engine is the launchable one.
    @app.get("/v1/training/engines")
    def training_engines():
        """List training engines and the ternary runner."""
        return {
            "engines": engine_catalog(),
            "python": ternary_python(),
            "runner": ternary_runner(),
        }

    @app.post("/v1/training/launch")
    def training_launch(req: TrainingLaunchRequest):
        """Launch a training run for an engine with the given fields."""
        try:
            return launch_training(
                st.runs_root, req.engine, req.method,
                dict(req.fields or {}), name=req.name)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/v1/training/status/{run_dir:path}")
    def training_status(run_dir: str):
        """Training run progress for a run directory."""
        target = resolve_run_dir(st.runs_root, run_dir)
        if not target.is_dir():
            raise HTTPException(404, f"no run under {target}")
        return training_run_status(target)

    @app.post("/v1/training/report/{run_dir:path}")
    def training_report(run_dir: str):
        """Write and return the training report for a run directory."""
        target = resolve_run_dir(st.runs_root, run_dir)
        if not (target / "training.json").is_file():
            raise HTTPException(404, f"not a training run: {target}")
        write_run_report(target)
        return build_run_report(target)

    @app.get("/v1/report/{run_dir:path}")
    def report(run_dir: str):
        """Return the run_report.json bundle for a run directory."""
        target = resolve_run_dir(st.runs_root, run_dir)
        path = target / "run_report.json"
        if not path.is_file():
            raise HTTPException(404, f"no run_report.json under {target}")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/v1/runs")
    def runs_index():
        """List run bundles available for viewing."""
        return {"runs": _list_runs(st.runs_root)}

    # ------------------------------------------------------------------
    # Report views (Seam B): server-rendered HTML over run bundles.
    @app.get("/view/{run_dir:path}", response_class=HTMLResponse)
    def view_report(run_dir: str):
        """Rendered HTML report page for a run bundle."""
        target = resolve_run_dir(st.runs_root, run_dir)
        path = target / "run_report.json"
        if not path.is_file():
            raise HTTPException(404, f"no run_report.json under {target}")
        report = json.loads(path.read_text(encoding="utf-8"))
        return HTMLResponse(render_report_page(report, target.name), headers=_NO_STORE)

    @app.get("/runs", response_class=HTMLResponse)
    def view_runs():
        """HTML index of run bundles."""
        return HTMLResponse(render_runs_page(_list_runs(st.runs_root)), headers=_NO_STORE)

    # ------------------------------------------------------------------
    # Sandbox (Docker-backed, optional): run a command in an isolated
    # container when the daemon is up; fall back to a local subprocess with
    # a hard timeout otherwise.  Feeds the cascade's code_exec detector and
    # gives the agent a safe execution surface.
    # ------------------------------------------------------------------
    # Studio settings (the toggles the chat footer owns)
    # ------------------------------------------------------------------
    def _load_studio_settings() -> dict:
        """The Studio's own toggles (sandbox, tool use, auto-compaction, dev mode)."""
        return load_studio_settings(st.state_dir)

    def _save_studio_settings(data: dict) -> dict:
        return save_studio_settings(data, st.state_dir)

    # ------------------------------------------------------------------
    # Compaction + engine surgery (the /compact command and its offer)
    # ------------------------------------------------------------------
    def _chat(base_url: str, model: str, api_key: str, prompt: str,
              messages: list[dict]) -> str:
        """One OpenAI-compatible chat call; returns the reply text."""
        import requests

        payload = {"model": model or "local", "messages": messages,
                   "temperature": 0, "max_tokens": 1200}
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        resp = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                             json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"].get("content") or "").strip()

    def _default_provider():
        return st.registry.resolve(None)

    @app.post("/v1/compaction/run")
    def compaction_run(body: dict[str, Any] = Body(default={})):
        """Compact a transcript into a checkpoint (manual or auto trigger)."""
        from harness.compaction.core import Message, Settings, Trigger, compact

        raw = body.get("messages") or []
        messages = [Message(role=str(m.get("role") or "user"),
                             text=str(m.get("text") or ""))
                    for m in raw if isinstance(m, dict)]
        if not messages:
            raise HTTPException(422, "messages is required")
        reason = str(body.get("trigger") or "manual")
        trigger = (Trigger.manual() if reason == "manual"
                   else Trigger.auto() if reason == "auto"
                   else Trigger.overflow(float(reason) if reason.replace('.', '', 1).isdigit() else 1.0))
        settings = Settings(auto=bool(body.get("auto", True)),
                            keep=int(body.get("keep") or 15_000),
                            buffer=body.get("buffer"))

        def call_model(prompt: str, older) -> str:
            try:
                provider = _default_provider()
            except LookupError:
                return ""          # no model: the deterministic summary is used
            msgs = [{"role": "user", "content": prompt}]
            for m in older[-12:]:
                msgs.append({"role": "user" if m.role == "user" else "assistant",
                             "content": m.text})
            return _chat(provider.base_url, provider.model, provider.api_key,
                         prompt, msgs)

        from harness.compaction.studio import make_summarizer

        events: list[dict] = []
        outcome = compact(messages, trigger, settings,
                          summarize=make_summarizer(call_model),
                          emit=lambda kind, payload: events.append({"event": kind, **payload}),
                          limit=body.get("limit"))
        out: dict[str, Any] = {"status": outcome.status, "events": events}
        if outcome.error:
            out["error"] = outcome.error
        checkpoint = getattr(outcome, "checkpoint", None)
        if checkpoint is not None:
            out["checkpoint"] = {"summary": checkpoint.summary,
                                 "recent": checkpoint.recent,
                                 "rendered": checkpoint.render()}
        return out

    @app.post("/v1/compaction/surgery")
    def compaction_surgery(body: dict[str, Any] = Body(default={})):
        """Offer an engine switch at a checkpoint; commit only if the gate passes."""
        from harness.compaction.core import Checkpoint
        from harness.compaction.surgery import offer_surgery
        from harness.compaction.studio import make_candidate

        cp = body.get("checkpoint") or {}
        checkpoint = Checkpoint(summary=str(cp.get("summary") or ""),
                                recent=str(cp.get("recent") or ""))
        current = str(body.get("current") or "")
        names = [str(n) for n in (body.get("candidates") or [])]
        try:
            provider = _default_provider()
        except LookupError as exc:
            return {"status": "unavailable", "engine": "", "reason": str(exc),
                    "events": [], "results": []}

        def ask_for(base_url: str, model: str, key: str):
            def ask(text: str) -> str:
                return _chat(base_url, model, key, text,
                             [{"role": "user", "content": text}])
            return ask

        candidates = []
        for name in names:
            if name == current:
                continue
            target = st.registry.resolve(name)
            candidates.append(make_candidate(
                name, target.base_url,
                ask_for(target.base_url, target.model, target.api_key),
                current_ask=(ask_for(provider.base_url, provider.model, provider.api_key)
                             if current == provider.name else None),
            ))

        def apply(engine: str) -> None:
            # the switch: this provider becomes the default
            st.registry.default = engine
            save_registry(st.registry, st.providers_file)

        events: list[dict] = []
        outcome = offer_surgery(str(body.get("session") or "default"),
                                checkpoint, current, candidates, apply,
                                emit=lambda kind, payload: events.append({"event": kind, **payload}))
        return {
            "status": outcome.status,
            "engine": outcome.engine,
            "reason": outcome.reason,
            "events": events,
            "results": [{"name": n, "probe_ok": p.ok, "probe_model": p.model,
                          "gate_passed": (g.passed if g else None),
                          "score": (g.score if g else None),
                          "baseline": (g.baseline if g else None),
                          "detail": (g.detail if g else None)}
                         for n, p, g in outcome.results],
        }

    @app.get("/v1/compaction/status")
    def compaction_status():
        """Whether automatic compaction is on, and the default settings."""
        from harness.compaction.core import SUMMARY_HEADINGS, Settings

        settings = Settings(auto=bool(_load_studio_settings()["auto_compact"]))
        return {"auto": settings.auto, "keep": settings.keep,
                "buffer": settings.buffer,
                "headings": list(SUMMARY_HEADINGS)}

    @app.get("/v1/studio/settings")
    def studio_settings():
        """The Studio's toggles: sandbox, tool use, auto-compaction."""
        return _load_studio_settings()

    @app.get("/v1/studio/harvest")
    def studio_harvest(limit: int = Query(default=0, ge=0, le=1000)):
        """The harvested turns - the data the encoder is trained on.

        Off unless ``harvest_enabled`` is on: the file holds the user's
        own words, so it is the user's call.
        """
        from harness import harvest as _harvest

        return {
            "enabled": bool(_load_studio_settings().get("harvest_enabled")),
            "path": str(_harvest.harvest_path(st.state_dir)),
            "turns": _harvest.count_turns(st.state_dir),
            "records": _harvest.read_turns(st.state_dir, limit=limit),
        }

    @app.delete("/v1/studio/harvest")
    def studio_harvest_clear():
        """Throw the harvested turns away."""
        from harness import harvest as _harvest

        return {"cleared": _harvest.clear(st.state_dir)}

    def _strata_base() -> str:
        return (_load_studio_settings().get("strata_base")
                or os.environ.get("CASCADE_SCION_BASE")
                or "http://127.0.0.1:8081/v1").rstrip("/")

    def _sync_strata_provider() -> dict:
        """Keep a provider for the Strata engine, and prefer it when nothing else serves.

        The Strata serve is external to the Studio: the user starts it, and the
        chat has to be able to reach it.  When it is the only live backend, it
        becomes the default so the chat is not talking to a dead endpoint.
        """
        info = _probe_strata()
        if not info.get("alive"):
            return info
        default = st.registry.default
        if not default or default.lower() == "strata":
            return info
        prov = next((p for p in st.registry.providers
                     if p.name.lower() == default.lower()), None)
        if prov is None or not prov.base_url.startswith("auto://"):
            return info
        # auto:// points at the Studio's own llama-server: if none is running,
        # strata is the only thing that can serve, so take it over as default.
        if models_manager.status().get("running"):
            return info
        _register_strata_provider(info)
        st.registry.default = "strata"
        return info

    def _register_strata_provider(info: dict) -> None:
        """Add (or refresh) the provider that points at the Strata engine.

        A provider's base_url is the server ROOT: the clients append
        ``/v1/chat/completions`` themselves, so a base that already ends in
        ``/v1`` would produce ``/v1/v1/...``.
        """
        base = (info.get("base_url") or "").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        model = info.get("model") or ""
        existing = [p for p in st.registry.providers
                    if p.name.lower() == "strata" or p.base_url.rstrip("/") == base]
        if existing:
            for p in existing:
                if p.model != model:
                    p.model = model
                if p.base_url.rstrip("/") != base:
                    p.base_url = base
            return
        try:
            st.registry.providers.append(
                Provider.from_dict({"name": "strata", "base_url": base,
                                    "api_key": "", "model": model}))
        except ValueError:
            pass

    def _probe_strata() -> dict:
        """Is the Strata engine (the local 176B serve) up, and does it hold the model?

        "alive" is the serve process; "loaded" is the model in VRAM.  They
        are different: an engine that is up with its model unloaded serves
        nothing, and the Studio must not claim a model is loaded then.
        """
        import requests

        base = _strata_base()
        root = base[:-3] if base.endswith("/v1") else base
        out = {"base_url": base, "alive": False, "loaded": False,
               "model": None, "models": []}
        try:
            # /v1/status says both: the serve is up AND the model is loaded
            resp = requests.get(f"{root}/v1/status", timeout=4)
            if resp.ok:
                data = resp.json()
                out["alive"] = True
                out["loaded"] = bool(data.get("loaded"))
                out["model"] = data.get("model")
                out["models"] = [out["model"]] if out["model"] else []
                return out
        except Exception:  # noqa: BLE001 - fall back to the model list
            pass
        try:
            resp = requests.get(f"{base}/models", timeout=4)
            if resp.ok:
                data = resp.json()
                ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
                out["alive"] = True
                out["models"] = ids
                out["model"] = ids[0] if ids else None
        except Exception:  # noqa: BLE001 - "not running" is a valid answer
            pass
        return out

    @app.get("/v1/studio/strata")
    def studio_strata():
        """Probe the Strata engine (the local 176B serve) and list its models."""
        return _sync_strata_provider()

    @app.put("/v1/studio/settings")
    def studio_settings_put(body: dict[str, Any] = Body(default={})):
        """Update the Studio's toggles (sandbox, tool use, auto-compaction)."""
        return _save_studio_settings(body or {})

    # ------------------------------------------------------------------
    # Hardware profile + deterministic controller
    # ------------------------------------------------------------------
    def _state_paths():
        from harness.state_store import StatePaths

        return StatePaths(Path(st.state_dir))

    def _controller():
        from harness.controller.service import ControllerService

        return ControllerService(_state_paths())

    def _tool_hub():
        from harness.security import GatewayHub

        hub = getattr(app.state, "tool_hub", None)
        if hub is None:
            hub = GatewayHub(_state_paths())
            app.state.tool_hub = hub
            # Session grants live in this GatewayHub's GrantStore.  A cease
            # must revoke those exact in-memory grants, not a fresh store.
            if app.state.activity is not None:
                app.state.activity.grants = hub.grants
        return hub

    agent_service.lineage_sink = (
        lambda parent, child: _tool_hub().claim_child(parent, child))

    @app.get("/v1/profiles/hardware")
    def profile_hardware(profile_id: Optional[str] = Query(default=None)):
        """The newest saved hardware profile, or the one named by profile_id."""
        from harness.controller.profile import HardwareProfiler

        profiler = HardwareProfiler(_state_paths())
        profile = profiler.get(profile_id) if profile_id else profiler.latest()
        if profile is None:
            raise HTTPException(404, "no hardware profile saved")
        return {"profile": profile}

    @app.post("/v1/profiles/scan")
    def profile_scan(body: dict[str, Any] = Body(default={})):
        """Scan hardware locally after explicit consent."""
        from harness.controller.profile import (
            ConsentRequired,
            HardwareProfiler,
        )

        try:
            profile = HardwareProfiler(_state_paths()).scan(
                body.get("consent"),
                profile_id=body.get("profile_id"),
            )
        except ConsentRequired as exc:
            raise HTTPException(403, str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"profile": profile}

    @app.get("/v1/controller/status")
    def controller_status():
        """Show the saved profile, proposals, and last applied change."""
        return _controller().status()

    @app.post("/v1/controller/propose")
    def controller_propose(body: dict[str, Any] = Body(default={})):
        """Build an allowlisted E0 settings proposal."""
        from harness.controller.service import ControllerError

        try:
            proposal = _controller().propose(
                profile_id=body.get("profile_id"),
                model_path=body.get("model_path"),
                model_gib=(float(body["model_gib"])
                           if body.get("model_gib") is not None else None),
                native_ctx=(int(body["native_ctx"])
                            if body.get("native_ctx") is not None else None),
                desired_ctx=(int(body["desired_ctx"])
                             if body.get("desired_ctx") is not None else None),
                kv_type=str(body.get("kv_type") or "q8_0"),
                request=body.get("request"),
                embed_request=body.get("embed_request") is True,
            )
        except (ControllerError, OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"proposal": proposal}

    @app.post("/v1/controller/apply")
    def controller_apply(body: dict[str, Any] = Body(default={})):
        """Apply one proposal only after an explicit confirmation."""
        from harness.controller.service import (
            ConfirmationRequired,
            ControllerError,
        )

        if body.get("confirm") is not True:
            raise HTTPException(403, "controller apply requires confirm=true")
        proposal_id = str(body.get("proposal_id") or "").strip()
        if not proposal_id:
            raise HTTPException(422, "proposal_id is required")
        try:
            result = _controller().apply(
                proposal_id, confirmed=True,
                profile_id=body.get("profile_id"))
        except ConfirmationRequired as exc:
            raise HTTPException(403, str(exc)) from exc
        except ControllerError as exc:
            raise HTTPException(422, str(exc)) from exc
        return result

    @app.post("/v1/controller/revert")
    def controller_revert(body: dict[str, Any] = Body(default={})):
        """Roll back the last confirmed controller change."""
        from harness.controller.service import (
            ConfirmationRequired,
            ControllerError,
        )

        if body.get("confirm") is not True:
            raise HTTPException(403, "controller revert requires confirm=true")
        try:
            return _controller().revert(confirmed=True)
        except ConfirmationRequired as exc:
            raise HTTPException(403, str(exc)) from exc
        except ControllerError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/v1/controller/outcome")
    def controller_outcome(body: dict[str, Any] = Body(default={})):
        """Append an accepted/rejected/wrong label to a proposal trace."""
        from harness.controller.service import ControllerError

        proposal_id = str(body.get("proposal_id") or "").strip()
        outcome = str(body.get("outcome") or "").strip()
        if not proposal_id or not outcome:
            raise HTTPException(422, "proposal_id and outcome are required")
        metrics = body.get("metrics")
        if metrics is not None and not isinstance(metrics, dict):
            raise HTTPException(422, "metrics must be an object")
        try:
            _controller().record_outcome(proposal_id, outcome,
                                         metrics=metrics or {})
        except ControllerError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"ok": True, "proposal_id": proposal_id, "outcome": outcome,
                "metrics": metrics or {}}

    @app.get("/v1/tools")
    def tools_registry():
        """List verified and unverified external tool metadata."""
        from harness.security import ToolRegistry

        return {"tools": ToolRegistry(_state_paths()).list()}

    @app.post("/v1/tools/register")
    def tools_register(req: ToolRegisterRequest):
        """Register external (for example MCP) tool metadata."""
        from harness.security import SecurityError, ToolRegistry, ToolSpec

        try:
            spec = ToolSpec.from_dict(req.tool)
            return ToolRegistry(_state_paths()).register(
                spec, verified=req.verified, server=req.server, note=req.note)
        except SecurityError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/v1/tools/unregister")
    def tools_unregister(body: dict[str, Any] = Body(default={})):
        """Remove one registered tool identity."""
        from harness.security import SecurityError, ToolRegistry

        source = str(body.get("source") or "").strip()
        name = str(body.get("name") or "").strip()
        version = str(body.get("version") or "1.0.0").strip()
        if not source or not name:
            raise HTTPException(422, "source and name are required")
        try:
            return {"removed": ToolRegistry(_state_paths()).revoke(
                source, name, version)}
        except SecurityError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/tools/policy")
    def tools_policy():
        """Show the safety tier, master switches, and saved grants."""
        from harness.security import GrantStore, SecurityStore

        state = _state_paths()
        return {"policy": SecurityStore(state).read(),
                "grants": GrantStore(state).list()}

    @app.put("/v1/tools/policy")
    def tools_policy_put(body: dict[str, Any] = Body(default={})):
        """Set the safety tier and/or master switches."""
        from harness.security import SecurityError, SecurityStore

        store = SecurityStore(_state_paths())
        try:
            if body.get("tier") is not None:
                store.set_tier(str(body["tier"]))
            master = body.get("master")
            if isinstance(master, dict) and master:
                store.set_master(**master)
            elif master is not None and not isinstance(master, dict):
                raise SecurityError("master must be an object")
        except SecurityError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"policy": store.read()}

    @app.post("/v1/tools/revoke")
    def tools_revoke(body: dict[str, Any] = Body(default={})):
        """Revoke one global tool grant, all grants, or one session's grants."""
        from harness.security import GrantStore

        identity = str(body.get("identity") or "").strip() or None
        session_id = str(body.get("session_id") or "").strip() or None
        all_grants = body.get("all") is True
        session_only = body.get("session_only") is True
        # Session grants are held in memory by the live GatewayHub.  A fresh
        # GrantStore would remove the durable file but miss them completely.
        store = _tool_hub().grants
        if session_only:
            if not session_id:
                raise HTTPException(
                    422, "session_only needs a session_id")
            removed = store.revoke_session(session_id)
            return {"ok": True, "removed": removed}
        if not identity and not all_grants:
            raise HTTPException(422, "identity or all=true is required")
        removed = store.revoke(identity, session_id=session_id)
        return {"ok": True, "removed": removed}

    @app.post("/v1/tools/call")
    def tools_call(body: dict[str, Any] = Body(default={})):
        """Authorize one tool call; return approved, pending, or denied."""
        from harness.security import (
            SecurityError,
            ToolRegistry,
            ToolRequest,
            ToolSpec,
        )

        session_id = str(body.get("session_id") or "").strip()
        spec = body.get("tool")
        if not session_id or not isinstance(spec, dict):
            raise HTTPException(422, "session_id and tool metadata are required")
        try:
            spec_object = ToolSpec.from_dict(spec)
            registered = ToolRegistry(_state_paths()).get(
                spec_object.source, spec_object.name, spec_object.version)
            if registered is not None:
                spec_object = registered
            request = ToolRequest(
                session_id=session_id,
                action=str(body.get("action") or ""),
                reason=str(body.get("reason") or ""),
                args=(body.get("args") if isinstance(body.get("args"), dict)
                      else {}),
                resource=str(body.get("resource") or ""),
                turn_id=body.get("turn_id"),
                timeout_s=float(body.get("timeout_s") or 600),
            )
            result = _tool_hub().gateway(
                session_id, headless=body.get("headless") is True
            ).authorize(spec_object, request,
                        headless=body.get("headless") is True)
        except (SecurityError, TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return result

    @app.get("/v1/tools/pending")
    def tools_pending(session_id: Optional[str] = Query(default=None)):
        """List pending gateway requests, optionally for one session."""
        return {"requests": _tool_hub().pending(session_id)}

    @app.post("/v1/tools/decide")
    def tools_decide(body: dict[str, Any] = Body(default={})):
        """Resolve a pending request: once, session, always, or deny."""
        from harness.security import ApprovalError, SecurityError

        session_id = str(body.get("session_id") or "").strip()
        request_id = str(body.get("request_id") or "").strip()
        choice = str(body.get("choice") or "").strip()
        if not session_id or not request_id or not choice:
            raise HTTPException(422, "session_id, request_id and choice are required")
        try:
            return _tool_hub().decide(
                session_id, request_id, choice,
                scope=str(body.get("scope") or "session"))
        except (ApprovalError, SecurityError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/tools/decision")
    def tools_decision(session_id: str = Query(...),
                       request_id: str = Query(...)):
        """Return a settled pending decision for the hook poller."""
        result = _tool_hub().decision(session_id, request_id)
        if result is None:
            raise HTTPException(404, "decision not ready")
        return result

    @app.get("/v1/tools/audit")
    def tools_audit(limit: int = Query(default=50, ge=1, le=1000)):
        """Show recent tool-gateway audit records (arguments are hashed)."""
        from harness.state_store import StatePaths

        path = StatePaths(Path(st.state_dir)).tool_audit_file()
        rows: list[dict[str, Any]] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return {"records": rows, "path": str(path)}

    @app.get("/v1/encoder/status")
    def encoder_status(model: Optional[str] = Query(default=None),
                       allow_download: bool = Query(default=False)):
        """Report the configured local encoder without loading its weights."""
        from harness.controller.encoder import (
            EncoderUnavailable,
            SentenceTransformerEncoder,
            encoder_from_settings,
        )

        try:
            encoder = (SentenceTransformerEncoder(
                model, local_files_only=not allow_download)
                if model else encoder_from_settings(
                    load_studio_settings(st.state_dir),
                    local_files_only=not allow_download))
        except (EncoderUnavailable, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc
        return encoder.status()

    @app.post("/v1/sandbox/exec")
    def sandbox_exec(req: SandboxExecRequest):
        """Run in the pinned Docker sandbox, or a host command only after approval."""
        import subprocess as _sp
        from harness import sandbox as sandbox_mod

        if not _load_studio_settings()["sandbox_enabled"]:
            raise HTTPException(403, "the sandbox is off (Studio settings)")
        command = req.command
        if not command.strip():
            raise HTTPException(422, "command is required")
        policy = sandbox_mod.SandboxPolicy(_state_paths())
        data = policy.read()
        if req.image and req.image != data["docker_image"]:
            raise HTTPException(
                403, "the sandbox image is fixed by policy; use /v1/sandbox/status")
        timeout = max(1, min(int(req.timeout),
                             int(data.get("timeout_s") or 60), 120))
        docker = sandbox_mod.docker_status()
        backend = "local"
        image_ref = ""
        fallback_reason = ""
        workspace = ""
        if req.workspace:
            try:
                workspace = sandbox_mod.workspace_path(req.workspace)
            except sandbox_mod.SandboxError as exc:
                raise HTTPException(422, str(exc)) from exc
        if workspace and not (req.use_docker and docker["daemon"]):
            raise HTTPException(
                409, "a workspace sandbox requires Docker; local fallback is disabled")
        container_name = ""
        if req.use_docker and docker["daemon"]:
            try:
                image_ref = policy.image_ref()
            except sandbox_mod.SandboxError as exc:
                raise HTTPException(409, str(exc)) from exc
            import uuid

            container_name = f"hive-sandbox-{uuid.uuid4().hex[:12]}"
            cmd = [
                docker["binary"], "run", "--rm",
                "--name", container_name,
                "--network", str(data.get("network") or "none"),
                "--memory", str(data.get("memory") or "512m"),
                "--cpus", str(data.get("cpus") or "1"),
                "--pids-limit", str(int(data.get("pids_limit") or 128)),
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
            ]
            if workspace:
                cmd += ["-v", f"{workspace}:/workspace:rw", "-w", "/workspace"]
            cmd += [
                "--label", "hive.managed=1",
                "--label",
                "hive.session=" + sandbox_mod.label_value(
                    req.session_id or "global"),
            ]
            if req.parent_session_id:
                cmd += ["--label", "hive.parent=" + sandbox_mod.label_value(
                    req.parent_session_id)]
            if workspace:
                cmd += ["--label", "hive.workspace=" + sandbox_mod.label_value(
                    workspace)]
            runtime = str(data.get("runtime") or "").strip()
            if runtime:
                cmd += ["--runtime", runtime]
            cmd += [image_ref, "sh", "-c", command]
            backend = "docker"
        else:
            allowed, why = policy.local_fallback(
                req.session_id, approved=req.allow_local_fallback)
            if not allowed:
                raise HTTPException(409, detail={
                    "error": why,
                    "docker": docker,
                    "requires_local_fallback_approval":
                        data.get("local_fallback") == "approval",
                })
            try:
                cmd = local_shell_command(command)
            except OSError as exc:
                raise HTTPException(
                    500, f"local shell is unavailable: {exc}") from exc
            fallback_reason = why
            if req.session_id:
                policy.record_fallback(req.session_id, command, why)

        CAP = 10000
        try:
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=timeout)
            timed_out = False
            rc = proc.returncode
            out = (proc.stdout or "")[:CAP]
            err = (proc.stderr or "")[:CAP]
        except _sp.TimeoutExpired as exc:
            timed_out = True
            rc = 124
            if backend == "docker" and container_name:
                for args in (["kill", container_name],
                             ["rm", "-f", container_name]):
                    try:
                        _sp.run([docker["binary"], *args],
                                capture_output=True, timeout=10)
                    except Exception:  # noqa: BLE001 - best-effort cleanup
                        pass
            out = ((exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace"))[:CAP]
            err = ((exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace"))[:CAP]
        except Exception as exc:  # noqa: BLE001
            return {"backend": backend, "error": str(exc), "exit_code": 1,
                    "stdout": "", "stderr": str(exc), "timed_out": False,
                    "image": image_ref, "sandbox_policy": data,
                    "workspace": workspace, "container": container_name}
        return {
            "backend": backend,
            "docker_available": docker["daemon"],
            "image": image_ref,
            "sandbox_policy": data,
            "fallback_reason": fallback_reason,
            "workspace": workspace,
            "container": container_name,
            "exit_code": rc,
            "timed_out": timed_out,
            "stdout": out,
            "stderr": err,
        }

    @app.get("/v1/sandbox/status")
    def sandbox_status():
        """Sandbox availability, image pin state, and fallback policy."""
        from harness import sandbox as sandbox_mod

        info = sandbox_mod.status(sandbox_mod.SandboxPolicy(_state_paths()))
        fallback = ("local subprocess allowed by policy"
                    if info["policy"]["local_fallback"] == "allow" else
                    "local subprocess requires explicit approval"
                    if info["policy"]["local_fallback"] == "approval" else
                    "local subprocess is disabled")
        return {
            "docker_installed": info["installed"],
            "docker_daemon": info["daemon"],
            "fallback": fallback,
            **info,
        }

    @app.post("/v1/web/fetch")
    def web_fetch_call(req: WebFetchRequest):
        """Fetch one URL with per-hop SSRF checks and explicit file gates."""
        from harness import web_fetch as web

        try:
            return web.fetch(
                req.url, method=req.method, body=req.body,
                headers=req.headers, session_id=req.session_id,
                allow_private=req.allow_private,
                allow_download=req.allow_download,
                download_name=req.download_name,
                upload_path=req.upload_path,
                allow_upload=req.allow_upload,
                state=_state_paths())
        except web.WebFetchError as exc:
            raise HTTPException(exc.status, detail={
                "code": exc.code, "message": str(exc), **exc.detail}) from exc

    @app.get("/v1/web/status")
    def web_fetch_status(session_id: Optional[str] = Query(default=None)):
        """Fetch policy, directories, and grants for this state root."""
        from harness import web_fetch as web

        body = web.status(_state_paths())
        if session_id:
            body["grants"] = web.BrowserGrantStore(
                _state_paths()).list(session_id=session_id)
        return body

    @app.get("/v1/web/grants")
    def web_fetch_grants(session_id: Optional[str] = Query(default=None)):
        from harness import web_fetch as web

        return {"grants": web.BrowserGrantStore(
            _state_paths()).list(session_id=session_id)}

    @app.post("/v1/web/grant")
    def web_fetch_grant(req: WebGrantRequest):
        from harness import web_fetch as web

        try:
            row = web.BrowserGrantStore(_state_paths()).grant(
                req.origin, access=req.access, scope=req.scope,
                session_id=req.session_id, reason=req.reason,
                expires_at=req.expires_at)
        except web.WebFetchError as exc:
            raise HTTPException(exc.status, detail={
                "code": exc.code, "message": str(exc), **exc.detail}) from exc
        return row

    @app.post("/v1/web/revoke")
    def web_fetch_revoke(req: WebRevokeRequest):
        from harness import web_fetch as web

        try:
            return web.BrowserGrantStore(_state_paths()).revoke(
                req.origin, session_id=req.session_id)
        except web.WebFetchError as exc:
            raise HTTPException(exc.status, detail={
                "code": exc.code, "message": str(exc), **exc.detail}) from exc

    @app.post("/v1/web/policy")
    def web_fetch_policy(req: WebPolicyRequest):
        from harness import web_fetch as web

        try:
            return web.WebFetchPolicy(_state_paths()).update(**{
                key: value for key, value in req.model_dump().items()
                if value is not None})
        except web.WebFetchError as exc:
            raise HTTPException(422, detail={
                "code": exc.code, "message": str(exc), **exc.detail}) from exc

    @app.post("/v1/browser/browse")
    def browser_browse(req: BrowserBrowseRequest):
        """Open a fresh Chromium context behind the validating local proxy."""
        from harness import browser_adapter
        from harness.web_fetch import WebFetchError

        try:
            return browser_adapter.browse(
                req.url, session_id=req.session_id,
                allow_private=req.allow_private,
                screenshot_name=req.screenshot_name,
                wait_until=req.wait_until, timeout_s=req.timeout_s,
                max_text=req.max_text, state=_state_paths())
        except WebFetchError as exc:
            raise HTTPException(exc.status, detail={
                "code": exc.code, "message": str(exc),
                **exc.detail}) from exc

    @app.get("/v1/browser/status")
    def browser_status():
        """Whether the optional Playwright/Chromium adapter is available."""
        from harness import browser_adapter

        return browser_adapter.status()

    @app.get("/cascade", response_class=HTMLResponse)
    def view_cascade():
        """Hive Cascade launch console (every runner knob, collapsible advanced)."""
        task_files = sorted(
            p.name for p in cascade_api.CASCADE_DIR.glob("*.json")
        ) if cascade_api.CASCADE_DIR.is_dir() else []
        return HTMLResponse(
            render_cascade_page(task_files, defaults=st.controls()["quality"]),
            headers=_NO_STORE)

    @app.get("/favicon.svg")
    def favicon_svg():
        """Studio favicon (hive hexagon)."""
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
               "<polygon points='32,4 58,18 58,46 32,60 6,46 6,18' fill='#FFB703' stroke='#000' stroke-width='3'/>"
               "<text x='32' y='42' font-size='28' text-anchor='middle' fill='#000' font-family='sans-serif' font-weight='900'>H</text>"
               # two small bees
               "<g transform='translate(10,8) scale(0.35)'>"
               "<ellipse cx='12' cy='10' rx='6' ry='4' fill='#000'/>"
               "<ellipse cx='12' cy='10' rx='3' ry='2' fill='#FFDD00'/>"
               "<circle cx='16' cy='8' r='3' fill='#000'/>"
               "</g>"
               "<g transform='translate(38,4) scale(0.35)'>"
               "<ellipse cx='12' cy='10' rx='6' ry='4' fill='#000'/>"
               "<ellipse cx='12' cy='10' rx='3' ry='2' fill='#FFDD00'/>"
               "<circle cx='16' cy='8' r='3' fill='#000'/>"
               "</g>"
               "</svg>")
        return Response(content=svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/bee.svg")
    def bee_svg():
        """Bee icon used by the Studio chrome."""
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
               "<ellipse cx='32' cy='36' rx='18' ry='12' fill='#000'/>"
               "<ellipse cx='32' cy='36' rx='10' ry='6' fill='#FFB703'/>"
               "<circle cx='46' cy='28' r='8' fill='#000'/>"
               "<circle cx='49' cy='26' r='2' fill='#FFDD00'/>"
               "<ellipse cx='22' cy='22' rx='10' ry='6' fill='#FFDD00' opacity='0.85' transform='rotate(-20 22 22)'/>"
               "<ellipse cx='42' cy='22' rx='10' ry='6' fill='#FFDD00' opacity='0.85' transform='rotate(20 42 22)'/>"
               "</svg>")
        return Response(content=svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/", response_class=HTMLResponse)
    def index():
        """Redirect to the run index."""
        return RedirectResponse("/runs")

    # ------------------------------------------------------------------
    # Splinter conversation loop (Phase 2 plug-in): attach the standalone
    # splinter server LAST - host routes above keep precedence, and splinter's
    # paths (/v1/splinter/*, /v1/openai/*, /v1/mcp) fall through to it. The
    # shared registry means one conversation store for host + splinter.
    app.mount("/", create_splinter_app(app_state=st))

    # human titles for the OpenAPI docs (every route is registered by now)
    _title_from_docstrings(app)

    # pick up the Strata engine in the background: the chat must be able to
    # reach whatever is actually running, not a dead auto:// endpoint
    def _sync_strata_bg() -> None:
        try:
            _sync_strata_provider()
        except Exception:  # noqa: BLE001 - never block startup
            pass

    threading.Thread(target=_sync_strata_bg, daemon=True).start()

    return app
