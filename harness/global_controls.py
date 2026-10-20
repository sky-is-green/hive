"""The Global harness controls tab.

The Studio's controls were scattered: the memory knobs lived in the
Splinter tab (as per-conversation overrides), the cascade knobs on the
cascade page, the Studio's own toggles in the Settings index.  The Global
tab is the one place that holds the harness-wide baseline - what a
conversation, a cascade run and an agent turn all start from.

Each group maps to one stored block:

``memory_defaults``  the curation and retrieval baseline (RAG)
``quality_defaults`` the cascade's judge, repair, gate and escalation
``mcp``              the memory over MCP: the endpoint and its tools
``runtime_defaults`` the runtime baseline: the context and the sampling
"""

from __future__ import annotations

#: (key, label, kind, hint) — kind is "num", "text", "check" or "select".
#: The hint is the house style: how it works, what it does, what changes.
MEMORY_FIELDS: list[tuple[str, str, str, str]] = [
    ("max_context", "context window", "num",
     "How: the tokens the model keeps. Does: the ceiling on the curated context. "
     "Changing: bigger = more memory (the KV cache grows with it)."),
    ("max_tokens", "max answer tokens", "num",
     "How: the cap on one reply. Does: how long an answer may be. "
     "Changing: higher = longer answers, more time."),
    ("stale_threshold", "stale threshold", "num",
     "How: the turn age where a chunk starts to decay. Does: how quickly old "
     "context loses its pull. Changing: lower = the memory forgets faster."),
    ("dedup_threshold", "dedup threshold", "num",
     "How: the cosine above which two chunks are the same. Does: stops a "
     "repeated fact eating the budget. Changing: higher = more aggressive dedup."),
    ("drift_threshold", "drift threshold", "num",
     "How: how far the topic may move before the old context is dropped. "
     "Does: the conversation's focus. Changing: lower = it forgets on a new topic."),
    ("remembrance_threshold", "remembrance threshold", "num",
     "How: the relevance a chunk needs to be re-invoked. Does: whether the "
     "memory reaches for older facts. Changing: lower = more reach, more noise."),
    ("vocab_boost", "vocabulary boost", "num",
     "How: how much a rare word counts. Does: the retrieval's handling of "
     "unusual terms. Changing: higher = rarer words weigh more."),
    ("drift_prior", "numeric prior", "select",
     "How: how much a previous answer's numbers are trusted when the topic "
     "drifts. Does: the numeric continuity of a long thread. "
     "Changing: on = the numbers stick; off = they are re-derived."),
    ("confidence_mode", "confidence", "select",
     "How: whether the curator reports a confidence in its selection. "
     "Does: the extra work per turn. Changing: on = the cost, the signal."),
    ("sanitize_context", "sanitize the context", "check",
     "How: secrets and blobs are redacted before the context is stored. "
     "Does: keeps credentials out of the memory. "
     "Changing: off = the raw text is kept (and may leave the box)."),
    ("filter_hedge_replies", "drop hedged replies", "check",
     "How: a reply that only hedges is not stored. Does: keeps the memory "
     "worth having. Changing: off = every reply is kept."),
    ("enable_medium", "the medium drone", "check",
     "How: a second, larger scorer looks at the harder candidates. "
     "Does: the quality of the selection on a big context. "
     "Changing: off = cheaper turns, weaker selection."),
    ("comb_enabled", "the comb (the SSD archive)", "check",
     "How: the chunks that fall out of the budget are archived on disk and "
     "can be resurrected later. Does: the long memory. "
     "Changing: off = the memory is only what fits in the budget."),
    ("comb_top_k", "comb top-k", "num",
     "How: how many archived chunks are considered. Does: the reach of the "
     "archive. Changing: higher = more candidates, more work."),
    ("comb_gate_threshold", "comb gate", "num",
     "How: the confidence a resurrected chunk needs. Does: keeps the weak "
     "matches out. Changing: lower = more resurrections, more noise."),
    ("comb_max_records", "comb records", "num",
     "How: how many records the archive keeps. Does: the archive's size. "
     "Changing: higher = a longer memory, more disk."),
    ("comb_relevant_only", "comb: relevant only", "check",
     "How: only the chunks the curator judged relevant are archived. "
     "Does: keeps the archive small and clean. "
     "Changing: off = everything is archived."),
]

QUALITY_FIELDS: list[tuple[str, str, str, str]] = [
    ("judge", "judge", "select",
     "How: what decides an answer is wrong. Does: the gate in front of the "
     "repair. Changing: deterministic = the checks, no model; tiny-jev = a "
     "small model on the CPU."),
    ("repair", "repair", "select",
     "How: one greedy re-attempt with the finding quoted. Does: the second "
     "chance. Changing: auto = one round, always greedy; off = no repair."),
    ("gate", "learned gate", "select",
     "How: a small head as a third trigger. Does: catches what the "
     "deterministic checks miss. Changing: off is the default by rule, not "
     "by accident."),
    ("gate_threshold", "gate threshold", "num",
     "How: how sure the head must be. Does: the trade between misses and "
     "false alarms. Changing: lower = more repairs attempted."),
    ("human_gate", "human gate", "select",
     "How: a person is asked before the paid API. Does: keeps a rejection "
     "from becoming a bill. Changing: auto = only when a TTY is present."),
    ("api_fallback", "API fallback", "text",
     "How: the endpoint a rejected answer escalates to. Does: the paid path. "
     "Changing: empty = no escalation, the local answer stands."),
    ("router", "router", "select",
     "How: a route is chosen before the generation. Does: the cost/latency "
     "split. Changing: off = local first, always."),
    ("router_mode", "router mode", "select",
     "How: what the router is allowed to do. Does: whether it may escalate "
     "or only measure. Changing: measure = the router never spends money."),
]

MCP_FIELDS: list[tuple[str, str, str, str]] = [
    ("enabled", "expose the memory over MCP", "check",
     "How: the Studio answers POST /v1/mcp with the memory tools. "
     "Does: lets an MCP client (an agent, an editor) read and write the "
     "conversation's memory. Changing: off = the endpoint stops answering."),
    ("read_only", "read-only", "check",
     "How: the write tool (remember) is withheld. Does: an external client "
     "cannot plant facts in the memory. "
     "Changing: on = safe to expose to anything."),
]

RUNTIME_FIELDS: list[tuple[str, str, str, str]] = [
    ("ctx_size", "context size", "num",
     "How: the context the engine is launched with. Does: the window the "
     "model serves. Changing: bigger = more memory, more of it on the disk."),
    ("ngl", "GPU layers", "num",
     "How: how many layers sit on the GPU. Does: how much of the model is "
     "fast. Changing: lower = less VRAM, slower."),
    ("threads", "threads", "num",
     "How: the CPU parallelism. Does: the prompt read. "
     "Changing: more = faster on a long prompt."),
    ("temperature", "temperature", "num",
     "How: how random a reply is. Does: the shape of the answers. "
     "Changing: lower = more repeatable."),
    ("top_p", "top-p", "num",
     "How: the cumulative probability the sample is cut at. Does: the tail "
     "of the distribution. Changing: lower = more focused."),
    ("top_k", "top-k", "num",
     "How: how many tokens may be sampled from. Does: the tail. "
     "Changing: lower = more focused."),
]

#: The Studio's own switches that belong to the harness rather than to a
#: chat.  The sandbox, the tool use and the auto-compact stay in the chat
#: footer (they are contextual), so they are not repeated here.
TOGGLES: list[tuple[str, str, str]] = [
    ("dev_mode", "developer mode",
     "How: the developer surface appears (the deep-research queue, the "
     "Docker setup tab, the raw status). Does: nothing but show things. "
     "Changing: on = those panes are visible."),
]

#: the select options per field
CHOICES: dict[str, list[tuple[str, str]]] = {
    "drift_prior": [("", "off (re-derive)"), ("on", "on (keep the numbers)")],
    "confidence_mode": [("off", "off"), ("on", "on")],
    "judge": [("deterministic", "deterministic"), ("tiny-jev", "tiny-jev (CPU)")],
    "repair": [("auto", "auto (one round)"), ("off", "off")],
    "gate": [("off", "off (the default)"), ("intern-0.8b", "intern-0.8b (opt-in)")],
    "human_gate": [("auto", "auto (a TTY)"), ("on", "on"), ("off", "off")],
    "router": [("off", "off"), ("on", "on")],
    "router_mode": [("measure", "measure (never spends)"), ("apply", "apply")],
}


def _field_html(prefix: str, key: str, label: str, kind: str,
                hint: str) -> str:
    """One control: the label, the input, and the house-style tooltip."""
    ident = f"{prefix}-{key}"
    if kind == "check":
        control = (f'<input type="checkbox" id="{ident}" '
                   f'data-key="{key}" title="{hint}">')
    elif kind == "select":
        options = "".join(
            f'<option value="{value}">{text}</option>'
            for value, text in CHOICES.get(key, []))
        control = (f'<select id="{ident}" data-key="{key}" title="{hint}">'
                   f'{options}</select>')
    elif kind == "text":
        control = (f'<input type="text" id="{ident}" data-key="{key}" '
                   f'title="{hint}" style="min-width:180px">')
    else:
        control = (f'<input type="number" id="{ident}" data-key="{key}" '
                   f'title="{hint}" style="width:5.5rem">')
    return (f'<label class="inline" title="{hint}">{label} {control}</label>')


def _group(title: str, note: str, fields: list[tuple[str, str, str, str]],
            prefix: str) -> str:
    rows = ""
    for key, label, kind, hint in fields:
        rows += _field_html(prefix, key, label, kind, hint)
    return (f'<h3>{title}</h3><p class="note">{note}</p>'
            f'<div class="row" style="flex-wrap:wrap;gap:.35rem .8rem">{rows}</div>')


def render_global_pane() -> str:
    """The Global harness controls pane."""
    memory = _group(
        "Memory &amp; RAG",
        "The curation baseline: what a conversation gets when it overrides "
        "nothing. The Splinter tab layers this conversation's changes on top.",
        MEMORY_FIELDS, "gm")
    quality = _group(
        "Quality (the cascade)",
        "The defaults every cascade run starts from. The Cascade page shows "
        "them as the form's starting values.",
        QUALITY_FIELDS, "gq")
    mcp = _group(
        "MCP",
        "The memory over MCP: what an external client may read and write.",
        MCP_FIELDS, "gp")
    runtime = _group(
        "Runtime",
        "The baseline a launched engine starts from (the Engines tab keeps "
        "the saved profiles), plus the switches that belong to the harness "
        "rather than to one chat.",
        RUNTIME_FIELDS, "gr")
    toggles = "".join(
        f'<label class="inline" title="{hint}">{label} '
        f'<input type="checkbox" id="g-{key}" data-toggle="{key}" '
        f'onchange="saveStudioSetting(\'{key}\', this.checked)"></label>'
        for key, label, hint in TOGGLES)
    return f"""<div id="tab-global" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Global harness controls</h2>
<p class="note">The baseline the whole harness starts from: the memory a
conversation gets, the quality a cascade run asks for, what an MCP client may
touch, and the runtime a new engine is launched with. The Splinter tab layers
one conversation's overrides on top; nothing here changes a conversation you
have already started.</p>
{memory}
{runtime}
<div class="row" style="flex-wrap:wrap;gap:.35rem .8rem">{toggles}</div>
{quality}
{mcp}
<div class="row" style="margin-top:1rem">
<button onclick="saveGlobalControls(this)" title="How: PUT /v1/studio/settings with the four control blocks. Does: stores the baseline the harness starts from. Changing: takes effect for the next conversation or cascade run.">Save the baseline</button>
<span class="note" id="global-note"></span>
</div>
</section>
</div>"""
