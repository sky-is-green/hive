"""The harness's controls, resolved for a run.

The Global tab stores five blocks.  This module is the one place that
knows what each block *means* when something actually runs:

``memory_defaults``   the curation and retrieval baseline a conversation
                      starts from (``SplinterConfig``'s own names)
``encoder``           the small model that does the retrieval and the
                      memory management, and the second scorer beside it
``runtime_defaults``  the sampling a reply is drawn with, and the launch
                      options a new engine is started from
``quality_defaults``  the cascade's judge, repair, gate and escalation
``mcp``               what an MCP client may read and write

Every block is a *baseline*: the more specific thing wins.  A
conversation's own overrides beat the baseline, an engine profile's
sampling beats the baseline's, and a run's own knobs beat the
cascade's baseline.  Nothing here reads or writes the settings - the
caller hands over the settings dict and gets the resolved baseline back.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

#: the curation knobs the harness understands.  They are the
#: ``SplinterConfig``'s own names, so a baseline key is a config key.
MEMORY_KEYS: tuple[str, ...] = (
    "max_context",
    "max_tokens",
    "max_chunks",
    "stale_threshold",
    "stale_factor",
    "dedup_threshold",
    "drift_threshold",
    "remembrance_threshold",
    "vocab_boost",
    "confidence_mode",
    "drift_prior",
    "sanitize_context",
    "strip_secrets",
    "filter_hedge_replies",
    "enable_medium",
    "comb_enabled",
    "comb_top_k",
    "comb_gate_threshold",
    "comb_max_records",
    "comb_relevant_only",
)

#: the runtime baseline's sampling keys - the OpenAI-compatible ones the
#: chat actually sends with the request.
SAMPLING_KEYS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "stop",
)

#: the runtime baseline's launch keys, with the aliases the engine
#: profiles use (the baseline says ``ctx_size``, a profile says
#: ``context``; both mean the same thing).
LAUNCH_KEYS: dict[str, tuple[str, ...]] = {
    "ctx_size": ("ctx_size", "context"),
    "ngl": ("ngl", "gpu_layers"),
    "threads": ("threads",),
}


def _block(settings: Mapping[str, Any] | None, name: str) -> dict:
    value = (settings or {}).get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def encoder_of(settings: Mapping[str, Any] | None) -> dict:
    """The encoder the retrieval and the memory management run on.

    ``model``       the sentence-transformers id the drone embeds with
    ``backend``     ``local`` (here, on the CPU) or ``served``
    ``url``         the embedding endpoint, when the backend is served
    ``medium_model`` the second, larger scorer
    """
    block = _block(settings, "encoder")
    backend = str(block.get("embedding_backend") or "local").strip().lower()
    if backend not in ("local", "served"):
        backend = "local"
    return {
        "model": str(block.get("embedding_model") or "").strip(),
        "backend": backend,
        "url": str(block.get("embedding_url") or "").strip(),
        "medium_model": str(block.get("medium_model") or "").strip(),
    }


def memory_of(settings: Mapping[str, Any] | None) -> dict:
    """The curation baseline: the knobs a conversation starts from."""
    block = _block(settings, "memory_defaults")
    out: dict[str, Any] = {}
    for key in MEMORY_KEYS:
        value = block.get(key)
        if value is None or value == "":
            continue
        out[key] = value
    return out


def sampling_of(settings: Mapping[str, Any] | None) -> dict:
    """The sampling baseline: what a reply is drawn with."""
    block = _block(settings, "runtime_defaults")
    return {key: block[key] for key in SAMPLING_KEYS
            if block.get(key) is not None and block.get(key) != ""}


def launch_of(settings: Mapping[str, Any] | None) -> dict:
    """The launch baseline: the context, the GPU layers, the threads."""
    block = _block(settings, "runtime_defaults")
    out: dict[str, Any] = {}
    for name, aliases in LAUNCH_KEYS.items():
        for alias in aliases:
            value = block.get(alias)
            if value is None or value == "":
                continue
            out[name] = value
            break
    return out


def quality_of(settings: Mapping[str, Any] | None) -> dict:
    """The cascade's baseline: the judge, the repair, the gate."""
    return _block(settings, "quality_defaults")


def mcp_of(settings: Mapping[str, Any] | None) -> dict:
    """What an MCP client may do with the memory."""
    block = _block(settings, "mcp")
    tools = block.get("tools")
    if not isinstance(tools, (list, tuple)):
        tools = ["remember", "search"]
    return {
        "enabled": bool(block.get("enabled", True)),
        "read_only": bool(block.get("read_only", False)),
        "tools": [str(t) for t in tools],
    }


def merge(base: Mapping[str, Any] | None,
          override: Mapping[str, Any] | None) -> dict:
    """The baseline first, the override on top.  ``None`` never wins."""
    out = dict(base or {})
    for key, value in (override or {}).items():
        if value is None:
            continue
        out[key] = value
    return out


def merge_sampling(base: Mapping[str, Any] | None,
                   override: Mapping[str, Any] | None) -> dict:
    """The sampling baseline, then the more specific sampling on top."""
    return merge(base, override)


def drone_choice(settings: Mapping[str, Any] | None,
                 config: Mapping[str, Any] | None = None) -> dict:
    """The full description of the drone a conversation needs.

    The encoder comes from the Global tab, but a conversation may name
    its own model (``ultra_model``) and its own scoring knobs
    (``vocab_boost``, ``confidence_mode``) - those win.  The result is
    the cache key for the drone, so a different choice gets its own
    drone and switching back is instant.
    """
    encoder = encoder_of(settings)
    config = config or {}
    memory = memory_of(settings)
    model = str(config.get("ultra_model") or encoder["model"] or "").strip()
    medium = str(config.get("medium_model")
                  or encoder["medium_model"] or "").strip()
    return {
        "model": model,
        "backend": encoder["backend"],
        "url": encoder["url"],
        "medium_model": medium,
        "vocab_boost": config.get("vocab_boost", memory.get("vocab_boost", 0.15)),
        "confidence_mode": str(
            config.get("confidence_mode")
            or memory.get("confidence_mode") or "off"),
    }


def effective_summary(settings: Mapping[str, Any] | None,
                      config: Mapping[str, Any] | None = None) -> dict:
    """A short report of what a turn is actually running with.

    This is what the chat shows under the reply, so the user can see
    that the Global tab's baseline is the one in force.
    """
    encoder = encoder_of(settings)
    config = config or {}
    memory = memory_of(settings)
    merged = merge(memory, config)
    keys = ("max_context", "stale_threshold", "dedup_threshold",
            "drift_threshold", "confidence_mode", "vocab_boost",
            "enable_medium", "comb_enabled")
    return {
        "encoder": str(config.get("ultra_model") or encoder["model"] or ""),
        "backend": encoder["backend"],
        "medium_model": str(config.get("medium_model")
                            or encoder["medium_model"] or ""),
        "memory": {key: merged[key] for key in keys if key in merged},
        "sampling": dict(config.get("sampling") or {}),
    }


def known_keys() -> Iterable[str]:
    """Every key the harness reads out of the settings (for the docs)."""
    return (*MEMORY_KEYS, *SAMPLING_KEYS, *LAUNCH_KEYS,
            "embedding_model", "embedding_backend", "embedding_url",
            "medium_model")
