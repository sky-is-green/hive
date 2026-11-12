"""Role placement preflight for the Studio Roles page.

The Roles page asks one question before it saves: can every role's model fit
on the card the operator chose? This module answers it from the data the page
already has: role -> model, role -> placement policy, model -> weight/resident,
and the detected devices.

It is deliberately weights-only. Context/KV is the stack residency planner's
job; this gate catches the gross overcommit (a 69 GB model on a 20 GB card)
before a role file points at a load that cannot work.

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


def plan_roles(
    roles: Mapping[str, str],
    placements: Mapping[str, str],
    models: Mapping[str, Mapping[str, Any]],
    devices: Sequence[Device],
) -> dict[str, Any]:
    """Return ``{"plans": [...], "summary": {...}}`` for the role assignment.

    Each plan carries ``role``, ``model``, ``placement``, ``device``,
    ``resident``, ``fits``, ``reason`` and ``warnings``. The caller saves only
    when no plan has ``fits == False``.
    """
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
                "fits": bool(result.get("fits")),
                "reason": result.get("reason", ""),
                "warnings": warnings,
            }
        elif not model:
            plan = {
                "role": role, "model": model, "placement": policy,
                "device": None, "resident": False, "fits": False,
                "reason": "no model assigned", "warnings": [],
            }
        elif resident or (weight is not None and weight <= 0):
            result = {
                "device": None,
                "resident": True,
                "fits": True,
                "reason": "already resident" if resident else "policy seat (no GPU)",
            }
            placed[model] = result
            plan = {
                "role": role, "model": model, "placement": policy,
                "device": None, "resident": True, "fits": True,
                "reason": result["reason"], "warnings": [],
            }
        elif weight is None:
            result = {
                "device": None, "resident": False, "fits": False,
                "reason": f"{label}: model size is unknown; fit cannot be proven",
            }
            placed[model] = result
            plan = {
                "role": role, "model": model, "placement": policy,
                "device": None, "resident": False, "fits": False,
                "reason": result["reason"], "warnings": [],
            }
        else:
            candidates = _policy_candidates(policy, devices)
            if not candidates:
                result = {
                    "device": None, "resident": False, "fits": False,
                    "reason": f"{label}: no card matches placement "
                              f"{policy or 'auto'}",
                }
                placed[model] = result
                plan = {
                    "role": role, "model": model, "placement": policy,
                    "device": None, "resident": False, "fits": False,
                    "reason": result["reason"], "warnings": [],
                }
            else:
                fitting = [d for d in candidates if free[int(d.index)] >= weight]
                if fitting:
                    chosen = max(fitting, key=lambda d: free[int(d.index)])
                    before = free[int(chosen.index)]
                    free[int(chosen.index)] = max(0.0, before - weight)
                    result = {
                        "device": int(chosen.index),
                        "resident": False,
                        "fits": True,
                        "reason": f"{label}: {weight:.1f} GB fits on card "
                                  f"#{chosen.index} ({before:.1f} GB free)",
                    }
                    if not policy:
                        warnings.append("no card policy; auto-placed on the spare card")
                else:
                    chosen = max(candidates, key=lambda d: free[int(d.index)])
                    result = {
                        "device": int(chosen.index),
                        "resident": False,
                        "fits": False,
                        "reason": f"{label}: needs {weight:.1f} GB, card "
                                  f"#{chosen.index} has "
                                  f"{free[int(chosen.index)]:.1f} GB free",
                    }
                placed[model] = result
                plan = {
                    "role": role, "model": model, "placement": policy,
                    "device": result["device"], "resident": False,
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
