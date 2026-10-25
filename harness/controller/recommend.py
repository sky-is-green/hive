"""The deterministic E0 recommender.

E0 is a constrained fit problem, not a language-model judgement.  It reads a
saved hardware profile and (optionally) a model's size/context metadata, then
emits a small proposal.  The proposal is explainable, allowlisted, and never
applied by this module.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from harness.controller.features import TextFeatures
from harness.hardware import KV_GB_PER_1K, fit_plan

GIB = 1024.0 ** 3
MAX_E0_CTX = 32768
CTX_STEPS: tuple[int, ...] = (32768, 24576, 16384, 12288, 8192, 6144, 4096)
_CANDIDATE_ID = re.compile(r"^rec-[A-Za-z0-9._-]{1,96}$")

#: The only settings E0 may propose.  This is deliberately narrower than the
#: five settings blocks: launch/memory/quality keys are proposals for a later
#: rung, not for a first-run hardware fit.
CONTROLLER_KEYS: dict[str, tuple[type, float, float]] = {
    "runtime_defaults.threads": (int, 1, 512),
    "runtime_defaults.ctx_size": (int, 1024, 131072),
    "runtime_defaults.ngl": (int, 0, 999),
    "runtime_defaults.resident_gib": (float, 0.0, 256.0),
}


class ProposalError(ValueError):
    """A proposal is malformed or outside the controller allowlist."""


@dataclass(frozen=True)
class ModelRequest:
    """The model facts E0 is allowed to use.

    The controller never guesses a model's layer count or native context.  When
    those facts are absent, it proposes only settings that do not depend on
    them.
    """

    name: str = ""
    weights_gib: float | None = None
    native_ctx: int | None = None
    desired_ctx: int | None = None
    kv_type: str = "q8_0"
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights_gib": self.weights_gib,
            "native_ctx": self.native_ctx,
            "desired_ctx": self.desired_ctx,
            "kv_type": self.kv_type,
            "path": self.path,
        }


def model_request_from_path(path: str | Path, *,
                            desired_ctx: int | None = None) -> ModelRequest:
    """Read size and native context from a GGUF without loading weights."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"no model file at {target}")
    weights = target.stat().st_size / GIB
    native: int | None = None
    name = target.name
    try:
        # The model manager's GGUF reader is deliberately conservative: it
        # parses metadata only and returns {} for a malformed header.
        from harness.models import _read_gguf_metadata

        meta = _read_gguf_metadata(target)
        if isinstance(meta, dict):
            value = meta.get("context_length")
            if isinstance(value, int) and value > 0:
                native = value
            label = meta.get("general.name")
            if isinstance(label, str) and label.strip():
                name = label.strip()
    except Exception:  # noqa: BLE001 - the size is still useful
        pass
    return ModelRequest(
        name=name, weights_gib=round(weights, 3), native_ctx=native,
        desired_ctx=desired_ctx, kv_type="q8_0", path=str(target))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _visible_cards(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for position, raw in enumerate(profile.get("gpus") or []):
        if not isinstance(raw, Mapping):
            continue
        if raw.get("visible") is False:
            continue
        total = _number(raw.get("memory_gb"))
        if total is None or total <= 0:
            continue
        card = dict(raw)
        card["index"] = int(raw.get("index", position))
        card["_total_gb"] = total
        cards.append(card)
    return cards


def _pick_card(cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Prefer a non-display card, then the one with the most free VRAM."""
    if not cards:
        return None
    candidates = [card for card in cards if not card.get("display")] or cards
    def free_gb(card: dict[str, Any]) -> float:
        measured = _number(card.get("free_gb"))
        return float(measured if measured is not None else card["_total_gb"])
    return max(candidates, key=lambda card: (
        free_gb(card), -int(card["index"])))


def _card_budget(card: Mapping[str, Any]) -> tuple[float, float, list[str]]:
    """Free VRAM, display reserve, and evidence lines."""
    total = float(card["_total_gb"])
    free = _number(card.get("free_gb"))
    if free is None:
        used = _number(card.get("used_gb")) or 0.0
        free = max(0.0, total - used)
    if card.get("display"):
        used = _number(card.get("used_gb"))
        reserve = max(1.0, min(4.0, used if used is not None else 2.0))
        line = (f"card {card['index']} drives a display: reserving "
                f"{reserve:.1f} GiB for the desktop")
    else:
        reserve = 0.5
        line = (f"card {card['index']} is not the display card: reserving "
                f"{reserve:.1f} GiB for runtime overhead")
    return max(0.0, free - reserve), reserve, [line]


def _thread_count(profile: Mapping[str, Any]) -> tuple[int, str]:
    cpu = profile.get("cpu") if isinstance(profile.get("cpu"), Mapping) else {}
    physical = _number(cpu.get("physical"))
    logical = _number(cpu.get("logical"))
    if physical and physical >= 1:
        value = int(min(32, max(1, physical)))
        return value, f"CPU has {int(physical)} physical / {int(logical or physical)} logical cores"
    if logical and logical >= 1:
        value = int(min(32, max(1, math.ceil(logical / 2))))
        return value, f"CPU has {int(logical)} logical cores; using half for the engine"
    return 4, "CPU topology is unknown; using the conservative 4 threads"


def _ctx_options(model: ModelRequest, cap: int = MAX_E0_CTX) -> list[int]:
    limit = min(MAX_E0_CTX, max(1024, int(cap)))
    if model.native_ctx:
        limit = min(limit, int(model.native_ctx))
    if model.desired_ctx:
        limit = min(limit, int(model.desired_ctx))
    steps = [ctx for ctx in CTX_STEPS if ctx <= limit]
    if limit >= 4096 and limit not in steps:
        steps.insert(0, limit)
    return steps


def request_ctx_cap(request: TextFeatures | None) -> int:
    """The largest context a request class should reserve.

    This is the deterministic request baseline, not a learned head.  It exists
    so a short chat does not reserve a 32k KV cache merely because the card can
    afford it, and so code/reasoning gets room before a human has to tune it.
    """
    if request is None:
        return MAX_E0_CTX
    cap = {"short": 8192, "medium": 16384, "long": MAX_E0_CTX}.get(
        request.context_pressure, MAX_E0_CTX)
    if request.intent in ("code", "reasoning"):
        cap = max(cap, 16384)
    return cap


def _kv_gib(ctx: int, kv_type: str) -> float:
    rate = KV_GB_PER_1K.get(kv_type, KV_GB_PER_1K["q8_0"])
    return (ctx / 1024.0) * rate


def validate_changes(changes: Mapping[str, Any],
                     profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize a proposal's changes, rejecting every non-allowlisted key."""
    if not isinstance(changes, Mapping):
        raise ProposalError("changes must be an object")
    out: dict[str, Any] = {}
    ram = None
    if profile is not None:
        ram_block = profile.get("ram") if isinstance(profile.get("ram"), Mapping) else {}
        ram = _number(ram_block.get("available_gb"))
    for key, value in changes.items():
        spec = CONTROLLER_KEYS.get(str(key))
        if spec is None:
            raise ProposalError(
                f"{key!r} is not in the controller allowlist; only "
                f"{', '.join(sorted(CONTROLLER_KEYS))}")
        kind, lower, upper = spec
        if kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ProposalError(f"{key} must be an integer")
            if not lower <= value <= upper:
                raise ProposalError(f"{key}={value} is outside {lower}..{upper}")
        else:
            number = _number(value)
            if number is None:
                raise ProposalError(f"{key} must be a finite number")
            if not lower <= number <= upper:
                raise ProposalError(f"{key}={value} is outside {lower}..{upper}")
            value = number
        if key == "runtime_defaults.ctx_size" and value % 1024:
            raise ProposalError("runtime_defaults.ctx_size must be a multiple of 1024")
        if key == "runtime_defaults.resident_gib" and ram is not None and value > ram:
            raise ProposalError(
                f"runtime_defaults.resident_gib={value} exceeds available RAM {ram}")
        out[str(key)] = value
    if not out:
        raise ProposalError("a proposal must change at least one setting")
    return out


def propose(profile: Mapping[str, Any], *,
            model: ModelRequest | None = None,
            request: TextFeatures | None = None,
            now: datetime | None = None) -> dict[str, Any]:
    """Build one E0 proposal from a saved hardware profile.

    No model means no context/ngl guess: the controller still recommends a
    conservative thread count, but says plainly that the model fit is unknown.
    """
    if not isinstance(profile, Mapping) or not profile.get("profile_id"):
        raise ValueError("a saved hardware profile is required")
    stamp = (now or _utc_now()).astimezone(timezone.utc)
    cards = _visible_cards(profile)
    card = _pick_card(cards)
    threads, thread_evidence = _thread_count(profile)
    changes: dict[str, Any] = {"runtime_defaults.threads": threads}
    evidence = [thread_evidence]
    warnings: list[str] = []
    confidence = 0.55
    reason = "conservative hardware-only baseline"
    request_cap = request_ctx_cap(request)
    if request is not None:
        evidence.append(
            f"request baseline: intent={request.intent}, "
            f"context={request.context_pressure}, risk={request.risk} "
            "(features only; no raw text is stored)")
        if request.risk == "high":
            warnings.append(
                "the request baseline flags high risk; this proposal still "
                "changes runtime settings only and requires confirmation")
        if request_cap < MAX_E0_CTX:
            evidence.append(
                f"request context cap {request_cap} limits over-reserving KV "
                "for this request class")

    if model is None or model.weights_gib is None:
        warnings.append(
            "no model size was supplied; context, GPU layers and resident "
            "experts are left unchanged")
    else:
        weights = max(0.0, float(model.weights_gib))
        if card is None:
            warnings.append("no visible GPU with usable VRAM; GPU-layer and "
                            "context proposals are disabled")
        else:
            budget, reserve, lines = _card_budget(card)
            evidence.extend(lines)
            ram_block = profile.get("ram") if isinstance(profile.get("ram"), Mapping) else {}
            ram_available = _number(ram_block.get("available_gb")) or 0.0
            ram_budget = max(0.0, ram_available - 2.0)
            evidence.append(
                f"host RAM: {ram_available:.1f} GiB available; reserving 2.0 GiB "
                f"for the OS and desktop")
            evidence.append(
                f"model {model.name or '(unnamed)'}: {weights:.2f} GiB weights; "
                f"VRAM budget {budget:.2f} GiB after the {reserve:.1f} GiB reserve")
            chosen: tuple[int, float, dict[str, Any]] | None = None
            for ctx in _ctx_options(model, request_cap):
                kv = _kv_gib(ctx, model.kv_type)
                fit = fit_plan(weights, ctx, vram_gb=budget, ram_gb=ram_budget,
                               kv_type=model.kv_type)
                if fit["metrics"]["tier3_disk_gb"] <= 0.01:
                    chosen = (ctx, kv, fit)
                    break
            if chosen is None:
                warnings.append(
                    "the model plus a 4k KV cache does not fit in usable "
                    "VRAM+RAM; no context or GPU-layer proposal is safe")
            else:
                ctx, kv, fit = chosen
                changes["runtime_defaults.ctx_size"] = ctx
                evidence.append(
                    f"ctx {ctx} is the largest tested window with no disk "
                    f"overflow ({kv:.2f} GiB KV, {fit['metrics']['tier3_disk_gb']:.2f} GiB disk)")
                if weights + kv <= budget:
                    changes["runtime_defaults.ngl"] = 999
                    evidence.append(
                        f"weights + KV fit in VRAM ({weights + kv:.2f} <= "
                        f"{budget:.2f} GiB), so all layers can be offloaded")
                    confidence = 0.88
                else:
                    warnings.append(
                        "the model does not fully fit in the selected card's "
                        "VRAM; GPU layers are left to the stack/engine profile")
                    confidence = 0.72
                # The resident tier only helps when experts spill out of VRAM.
                if weights + kv > budget and ram_budget >= 4.0:
                    resident = int(min(8, max(0, math.floor(ram_budget / 4.0))))
                    if resident > 0:
                        changes["runtime_defaults.resident_gib"] = resident
                        evidence.append(
                            f"resident expert budget {resident} GiB fits in the "
                            f"remaining host RAM ({ram_budget:.1f} GiB after reserve)")
                        confidence = min(confidence, 0.62)
                reason = (
                    f"{'code/long-context' if ctx >= 16384 else 'short-context'} "
                    f"model fit on card {card['index']}")
                if request is not None:
                    reason = f"{request.intent}/{request.context_pressure}; {reason}"
    changes = validate_changes(changes, profile)
    proposal_id = f"rec-{stamp.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    return {
        "schema_version": 1,
        "id": proposal_id,
        "profile_id": str(profile["profile_id"]),
        "created_at": stamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "scope": f"profile:{profile['profile_id']}",
        "reason": reason,
        "changes": changes,
        "confidence": round(float(confidence), 3),
        "evidence": evidence,
        "warnings": warnings,
        "requires_confirmation": True,
        "revert": "previous studio settings snapshot",
        "model": model.as_dict() if model is not None else None,
        "request_features": request.as_dict() if request is not None else None,
    }


def validate_proposal(proposal: Mapping[str, Any],
                      profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate a stored proposal before it is applied."""
    if not isinstance(proposal, Mapping):
        raise ProposalError("proposal must be an object")
    ident = str(proposal.get("id") or "")
    if not _CANDIDATE_ID.match(ident):
        raise ProposalError(f"invalid proposal id {ident!r}")
    changes = validate_changes(proposal.get("changes") or {}, profile)
    return {**dict(proposal), "id": ident, "changes": changes}
