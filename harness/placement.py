"""Role placement preflight for the Studio Roles page.

The Roles page asks one question before it saves: can every role's model fit
on the card the operator chose? This module answers it from the data the page
already has: role -> model, role -> placement policy, model -> weight/resident/
GGUF metadata, the context/KV choice, and the detected devices.

The budget is the model's weights plus the architecture-aware KV cache from
``attention_kv_estimate`` at the requested context. When a role points at a
hosted/unknown model (no local GGUF metadata), it falls back to weights-only
and records a warning.

Placement policies:

- ``""``          no card policy; auto-place on the spare card with a warning
- ``"auto"``      prefer a non-display card, then any card
- ``"display"``   the display card
- ``"non_display"`` any non-display card
- ``"<index>"``   that exact device index

A model shared by several roles is charged once. A model that is already
resident (or has no GPU weight, such as the policy guards) consumes no budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from harness.models import attention_kv_estimate

GIB = 1024 ** 3
DEFAULT_CTX = 8192
DEFAULT_KV = "q8_0"


@dataclass(frozen=True)
class Device:
    index: int
    card: str
    display: bool
    total_gb: float
    free_gb: float


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


def _policy_candidates(policy: str, devices: Sequence[Device]) -> list[Device]:
    if not policy or policy == "auto":
        spare = [d for d in devices if not d.display]
        return spare or list(devices)
    if policy == "display":
        return [d for d in devices if d.display]
    if policy == "non_display":
        return [d for d in devices if not d.display]
    try:
        wanted = int(policy)
    except (TypeError, ValueError):
        return []
    return [d for d in devices if int(d.index) == wanted]


def _kv_gib(
    info: Mapping[str, Any],
    *,
    ctx: int,
    kv: str,
    warnings: list[str],
) -> float:
    """KV cache for one model at ``ctx``/``kv``, or 0 with a warning."""
    meta = info.get("gguf_metadata") or info.get("ggufMetadata")
    if not isinstance(meta, dict) or not meta:
        warnings.append(
            "KV estimate unavailable (no local GGUF metadata); weights only")
        return 0.0
    estimate = attention_kv_estimate(gguf_meta=meta, kv_cache_type=kv)
    if estimate is None:
        warnings.append(
            "KV estimate unavailable (thin GGUF header); weights only")
        return 0.0
    return estimate.bytes_at(ctx, kv) / GIB


def _need_text(weight: float, kv_gib: float, ctx: int) -> str:
    if kv_gib > 0:
        return (f"{weight:.1f} GB weights + {kv_gib:.1f} GB KV "
                f"at {ctx} = {weight + kv_gib:.1f} GB")
    return f"{weight:.1f} GB weights"


def plan_roles(
    roles: Mapping[str, str],
    placements: Mapping[str, str],
    models: Mapping[str, Mapping[str, Any]],
    devices: Sequence[Device],
    *,
    ctx: int = DEFAULT_CTX,
    kv: str = DEFAULT_KV,
) -> dict[str, Any]:
    """Return ``{"plans": [...], "summary": {...}}`` for the role assignment.

    Each plan carries ``role``, ``model``, ``placement``, ``device``,
    ``resident``, ``weights_gb``, ``kv_gb``, ``need_gb``, ``fits``, ``reason``
    and ``warnings``. The caller saves only when no plan has
    ``fits == False``.
    """
    ctx = max(0, int(ctx))
    kv = str(kv or DEFAULT_KV).strip().lower()
    free = {int(d.index): max(0.0, d.free_gb) for d in devices}
    placed: dict[str, dict[str, Any]] = {}
    plans: list[dict[str, Any]] = []
    ok = warn = blocked = 0

    for role, model in roles.items():
        model = str(model or "")
        policy = str(placements.get(role) or "")
        info = models.get(model) or {}
        weight = _number(info.get("weights_gb"))
        resident = bool(info.get("resident"))
        label = str(info.get("label") or model or "(none)")
        warnings: list[str] = []

        if model in placed:
            result = dict(placed[model])
            warnings.extend(result.get("warnings") or [])
            if policy and result.get("device") is not None:
                candidates = _policy_candidates(policy, devices)
                allowed = {int(d.index) for d in candidates}
                if allowed and int(result["device"]) not in allowed:
                    warnings.append(
                        f"{label} is already placed on card "
                        f"#{result['device']} for another role")
            plan = {
                "role": role,
                "model": model,
                "placement": policy,
                "device": result.get("device"),
                "resident": bool(result.get("resident")),
                "weights_gb": result.get("weights_gb"),
                "kv_gb": result.get("kv_gb", 0.0),
                "need_gb": result.get("need_gb"),
                "fits": bool(result.get("fits")),
                "reason": result.get("reason", ""),
                "warnings": warnings,
            }
        elif not model:
            plan = {
                "role": role, "model": model, "placement": policy,
                "device": None, "resident": False,
                "weights_gb": None, "kv_gb": 0.0, "need_gb": None,
                "fits": False, "reason": "no model assigned", "warnings": [],
            }
        elif resident or (weight is not None and weight <= 0):
            weights_gb = weight if weight is not None else 0.0
            result = {
                "device": None,
                "resident": True,
                "weights_gb": weights_gb,
                "kv_gb": 0.0,
                "need_gb": 0.0,
                "fits": True,
                "reason": "already resident" if resident else "policy seat (no GPU)",
                "warnings": [],
            }
            placed[model] = result
            plan = {
                "role": role, "model": model, "placement": policy,
                "device": None, "resident": True,
                "weights_gb": weights_gb, "kv_gb": 0.0, "need_gb": 0.0,
                "fits": True, "reason": result["reason"], "warnings": [],
            }
        elif weight is None:
            result = {
                "device": None, "resident": False,
                "weights_gb": None, "kv_gb": 0.0, "need_gb": None,
                "fits": False,
                "reason": f"{label}: model size is unknown; fit cannot be proven",
                "warnings": [],
            }
            placed[model] = result
            plan = {
                "role": role, "model": model, "placement": policy,
                "device": None, "resident": False,
                "weights_gb": None, "kv_gb": 0.0, "need_gb": None,
                "fits": False, "reason": result["reason"], "warnings": [],
            }
        else:
            kv_gib = _kv_gib(info, ctx=ctx, kv=kv, warnings=warnings)
            need = weight + kv_gib
            need_text = _need_text(weight, kv_gib, ctx)
            candidates = _policy_candidates(policy, devices)
            if not candidates:
                result = {
                    "device": None, "resident": False,
                    "weights_gb": weight, "kv_gb": kv_gib, "need_gb": need,
                    "fits": False,
                    "reason": f"{label}: no card matches placement "
                              f"{policy or 'auto'}",
                    "warnings": list(warnings),
                }
                placed[model] = result
                plan = {
                    "role": role, "model": model, "placement": policy,
                    "device": None, "resident": False,
                    "weights_gb": weight, "kv_gb": kv_gib, "need_gb": need,
                    "fits": False, "reason": result["reason"],
                    "warnings": warnings,
                }
            else:
                fitting = [
                    d for d in candidates if free[int(d.index)] >= need
                ]
                if fitting:
                    chosen = max(fitting, key=lambda d: free[int(d.index)])
                    before = free[int(chosen.index)]
                    free[int(chosen.index)] = max(0.0, before - need)
                    result = {
                        "device": int(chosen.index), "resident": False,
                        "weights_gb": weight, "kv_gb": kv_gib, "need_gb": need,
                        "fits": True,
                        "reason": f"{label}: {need_text}; fits on card "
                                  f"#{chosen.index} ({before:.1f} GB free)",
                        "warnings": list(warnings),
                    }
                    if not policy:
                        warnings.append(
                            "no card policy; auto-placed on the spare card")
                        result["warnings"] = list(warnings)
                else:
                    chosen = max(candidates, key=lambda d: free[int(d.index)])
                    result = {
                        "device": int(chosen.index), "resident": False,
                        "weights_gb": weight, "kv_gb": kv_gib, "need_gb": need,
                        "fits": False,
                        "reason": f"{label}: needs {need_text}, card "
                                  f"#{chosen.index} has "
                                  f"{free[int(chosen.index)]:.1f} GB free",
                        "warnings": list(warnings),
                    }
                placed[model] = result
                plan = {
                    "role": role, "model": model, "placement": policy,
                    "device": result["device"], "resident": False,
                    "weights_gb": weight, "kv_gb": kv_gib, "need_gb": need,
                    "fits": result["fits"], "reason": result["reason"],
                    "warnings": warnings,
                }

        if plan["fits"] and plan["warnings"]:
            warn += 1
        elif plan["fits"]:
            ok += 1
        else:
            blocked += 1
        plans.append(plan)

    return {
        "plans": plans,
        "summary": {
            "total": len(plans),
            "ok": ok,
            "warn": warn,
            "blocked": blocked,
        },
    }
