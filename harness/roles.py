"""The live Hive role assignment.

One file is the source of truth for the console, the API and the agent.  A
source checkout keeps the historical ``harness/webui/roles.json`` path; an
installed wheel keeps a writable copy under ``HIVE_HOME``.  The environment
override ``HIVE_ROLES_FILE`` wins in both cases.

The values are *model keys* (the same keys the console model selector uses):
``controller17``, ``reader17``, ``scion35``, ``lfm3``, ``d1-2b`` and so on.
A role that is not a chat model (``risk``) is stored, but cannot be routed to a
chat completion; the API reports that explicitly rather than silently falling
back.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from harness import paths

DEFAULT_ROLES: dict[str, str] = {
    "planner": "scion35",
    "controller": "controller17",
    "router": "scion35",
    "face": "scion35",
    "agency": "toolmanager17",
    "mechanics": "reader17",
    "judge": "d1-2b",
    "rag": "reader17",
    "risk": "guards",
    "vision": "lfm3vl",
    "placement": "auditor17",
    "critic": "auditor17",
    "dispatcher": "dispatcher17",
}

DEFAULT_PLACEMENTS: dict[str, str] = {
    "planner": "non_display",
    "controller": "display",
    "router": "non_display",
    "face": "non_display",
    "agency": "non_display",
    "mechanics": "display",
    "judge": "display",
    "rag": "display",
    "risk": "",
    "vision": "display",
    "placement": "non_display",
    "critic": "non_display",
    "dispatcher": "non_display",
}

# Roles that are policies/guards rather than OpenAI-compatible chat models.
NON_CHAT_ROLES = frozenset({"risk"})

# Model keys that do not expose an OpenAI-compatible chat endpoint.  d1 speaks
# /v1/systemone; guards are a policy layer.
NON_CHAT_MODELS = frozenset({"d1-2b", "guards", "guards+d1-3b", "d1-3b-cli"})

# Provider registry names used by older installations.  New installations seed
# the short keys; the aliases keep a live registry working without a rewrite.
MODEL_PROVIDER_ALIASES: dict[str, str] = {
    "ds41": "ds41-local",
    "lfm3": "local-LFM3-1.7B-Q4_K_M",
    "lfm3vl": "local-LFM3-VL-2B-Q8_0",
    "d1-2b": "local-d1-2B-Q8_0",
}

# GGUF aliases.  A model loaded from the library is registered as
# ``local-<file-stem>``, not as the role key, so match the served model too.
# Old role files (and old live provider registries) used these keys.
LEGACY_MODELS: dict[str, str] = {
    "clerk17": "toolmanager17",
    "clerk-1.7b": "tool-manager-1.7b",
}

MODEL_ALIASES: dict[str, str] = {
    "controller17": "conductor-1.7b",
    "reader17": "reader-1.7b",
    "auditor17": "auditor-1.7b",
    "dispatcher17": "dispatcher-1.7b",
    "toolmanager17": "tool-manager-1.7b",
    "lfm3": "lfm3",
    "lfm3vl": "LFM3-VL-2B-Q8_0",
    "d1-2b": "d1-2B-Q8_0",
    "scion35": "scion-35b-a3b-pq2_0-corr",
    "scion176": "scion-flashnext-176b-a6b-q2_0",
    "ds41": "deepseek-v4.1-flash-gsq",
}


def roles_path(path: str | Path | None = None) -> Path:
    """Resolve the role file: explicit > env > checkout seed > HIVE_HOME."""
    if path:
        return Path(path).expanduser()
    env = os.environ.get("HIVE_ROLES_FILE", "").strip()
    if env:
        return Path(env).expanduser()
    seed = paths.SOURCE_ROOT / "harness" / "webui" / "roles.json"
    if paths.is_source_checkout() and seed.is_file():
        return seed
    return paths.home() / "roles.json"


def _clean(data: Any) -> tuple[dict[str, str], dict[str, str]]:
    if not isinstance(data, dict):
        return dict(DEFAULT_ROLES), dict(DEFAULT_PLACEMENTS)
    raw_roles = data.get("roles") if isinstance(data.get("roles"), dict) else {}
    raw_places = data.get("placements") if isinstance(data.get("placements"), dict) else {}
    roles = dict(DEFAULT_ROLES)
    placements = dict(DEFAULT_PLACEMENTS)
    roles.update({str(k): LEGACY_MODELS.get(str(v), str(v))
                  for k, v in raw_roles.items() if v})
    placements.update({str(k): str(v) for k, v in raw_places.items()})
    return roles, placements


def load_roles(path: str | Path | None = None) -> dict[str, Any]:
    target = roles_path(path)
    if not target.is_file():
        return {"roles": dict(DEFAULT_ROLES), "placements": dict(DEFAULT_PLACEMENTS)}
    try:
        roles, placements = _clean(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        roles, placements = dict(DEFAULT_ROLES), dict(DEFAULT_PLACEMENTS)
    return {"roles": roles, "placements": placements}


def save_roles(roles: dict[str, str], placements: dict[str, str],
               path: str | Path | None = None) -> Path:
    target = roles_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "roles": {str(k): str(v) for k, v in roles.items()},
        "placements": {str(k): str(v) for k, v in placements.items()},
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def provider_name(model: str) -> str:
    """The preferred provider registry name for a console model key."""
    return MODEL_PROVIDER_ALIASES.get(model, model)


def resolve_provider(registry, model: str):
    """Find the provider serving a role's model key.

    Exact registry names win.  A library-launched model is registered as
    ``local-<stem>``, so the second pass matches the served model alias
    (``conductor-1.7b``) and the third matches the role key in the provider
    name.  Returns ``None`` when the role is not configured yet.
    """
    wanted = provider_name(model).strip().lower()
    alias = MODEL_ALIASES.get(model, "").strip().lower()
    key = model.strip().lower()
    providers = list(getattr(registry, "providers", []) or [])
    for provider in providers:
        if str(provider.name).strip().lower() == wanted:
            return provider
    if alias:
        for provider in providers:
            served = str(getattr(provider, "model", "") or "").strip().lower()
            if served == alias or alias in str(provider.name).lower():
                return provider
    if key:
        for provider in providers:
            if key in str(provider.name).lower():
                return provider
    return None
