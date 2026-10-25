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
     "How: SplinterConfig.max_context (8192 default) caps assembly.py focal "
     "budget vs drone budget (1-6k). Does: Token ceiling for curated prompt. "
     "Changing: Up → more chunks fit but higher token_count/latency; down → "
     "truncates even high-relevance facts."),
    ("max_tokens", "max answer tokens", "num",
     "How: SplinterConfig.max_tokens (None=backend default) → sampling "
     "max_tokens. Does: Caps reply length. Changing: Up longer answers; down "
     "≤256 on reasoning models → empty_reply_reasoning_starved."),
    ("stale_threshold", "stale wall", "num",
     "How: stale_threshold 20 → decay.py age>20 *0.5 and archive to comb. "
     "Does: Stale penalty + archiving gate. Changing: Up zombies linger; down "
     "faster forgetting, comb sooner, cleaner but lose mid-horizon."),
    ("dedup_threshold", "dedup", "num",
     "How: dedup_threshold 0.92 → ContextDeduplicator cosine>0.92 keeps "
     "denser info. Does: Filters duplicates pre-scoring. Changing: 0.98 keeps "
     "near-variants; 0.85 aggressively merges distinct facts."),
    ("drift_threshold", "drift", "num",
     "How: drift_threshold 0.6 → TopicDriftDetector 1-cosine(recent,history)>0.6 "
     "penalises old topics *0.1. Does: Detects shift. Changing: Up less "
     "sensitive (stale surfaces); down isolates recent but hurts cross-topic."),
    ("remembrance_threshold", "remem", "num",
     "How: remembrance_threshold 0.65 (currently unwired — RemembrancePass "
     "hardcoded). Does: Would save at deletion if relevant. Changing: No "
     "effect until wired; lower would save more, higher fewer."),
    ("vocab_boost", "vocab boost", "num",
     "How: vocab_boost 0.15 via UltraSmallDrone vocab match. Does: Keyword "
     "bonus on cosine. Changing: Up domain terms dominate; 0 pure semantic "
     "may miss code."),
    ("drift_prior", "numeric prior", "select",
     "How: how much a previous answer's numbers are trusted when the topic "
     "drifts. Does: the numeric continuity of a long thread. "
     "Changing: on = the numbers stick; off = they are re-derived."),
    ("confidence_mode", "confidence", "select",
     "How: confidence off/single/mcdropout via ultra_small 1 vs 3 passes "
     "(std). Does: Drives escalation to medium. Changing: off no escalation "
     "fastest; mcdropout enables medium re-score but 3× encode, needs medium "
     "drone for effect."),
    ("sanitize_context", "sanitize context", "check",
     "How: sanitize_context True → sanitize.py wraps <user_data> + "
     "neutralises ignore/system prompts. Does: Prevents injection. Changing: "
     "Off raw chunks to LLM (risk); on may alter legit code mentioning "
     "system:."),
    ("filter_hedge_replies", "filter hedge replies", "check",
     "How: filter_hedge True → hedges.py 90ch lead-anchored drops before "
     "add_chunk (also curate/observe). Does: Drops ~50% refusals. Changing: "
     "Off stores hedges → later retrieved loops; on may drop edge factual "
     "hedges."),
    ("enable_medium", "the medium drone", "check",
     "How: enable_medium False → MediumDrone graphcodebert 400MB vs stub 0.5. "
     "Does: Second-pass for score>=2 complex. Changing: On better recall "
     "+20-50ms, VRAM heavy; off ultra only."),
    ("comb_enabled", "the comb (the SSD archive)", "check",
     "How: comb_enabled False → CombStore JSONL at harness_comb, evicted not "
     "deleted. Does: Surplus tier for returns. Changing: On long-horizon "
     "recall, forces comb_dir set."),
    ("comb_top_k", "comb top-k", "num",
     "How: comb_top_k 5 → comb.retrieve k lexically ranked. Does: Competes "
     "vs store. Changing: Up more resurrect candidates but budget competition; "
     "down fewer faster."),
    ("comb_gate_threshold", "comb gate", "num",
     "How: comb_gate 0.85 → fires when top_raw<0.85 or echo. Does: Gate "
     "decides comb consult (~820ms). Changing: 0.70 fires often crowding; 0.95 "
     "rarely misses returns."),
    ("comb_max_records", "comb records", "num",
     "How: comb_max_records 2000 → prune LRU cap + 1000 turn prune. Does: Disk "
     "cap. Changing: Up longer horizon but disk; down sooner forgetting."),
    ("comb_relevant_only", "comb: curated-only", "check",
     "How: comb_relevant_only True → archive only once_curated "
     "(relevance_history). Does: Lean archive. Changing: Off archives every "
     "evicted (more noise/recall)."),
]

#: The encoder-sized models: the small LLM that does the retrieval and the
#: memory management.  The list is a suggestion - any sentence-transformers
#: id works.
ENCODER_CANDIDATES = [
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
    "Qwen/Qwen3-Embedding-0.6B",
    "nomic-ai/nomic-embed-text-v1.5",
]

ENCODER_FIELDS: list[tuple[str, str, str, str]] = [
    ("embedding_model", "the embedding model", "text",
     "How: the sentence-transformers model the drone embeds with. Does: the "
     "retrieval and the memory management - the ranking, the dedup, the "
     "decay. Changing: a different encoder changes what the memory finds; "
     "the dim size and the speed change with it. Takes effect when the "
     "Studio restarts (the drone is built once)."),
    ("embedding_backend", "the embedding backend", "select",
     "How: local runs the model here, served calls an embedding endpoint. "
     "Does: where the encoding happens. Changing: served moves the cost to "
     "another box (and needs the URL below)."),
    ("embedding_url", "the served embedding URL", "text",
     "How: the endpoint that serves the embeddings. Does: nothing unless the "
     "backend is served. Changing: the URL the drone posts to."),
    ("medium_model", "the medium drone's model", "text",
     "How: the second, larger scorer the curator consults on the hard "
     "candidates. Does: the recall on a big context. Changing: off (the "
     "toggle) leaves the small encoder alone; a different model trades VRAM "
     "for quality."),
    ("controller_idle_unload", "unload controller encoder when idle", "check",
     "How: after the controller's lazy encoder finishes a request, a daemon "
     "timer drops the model reference when it has been idle for the number of "
     "seconds below. Does: frees the encoder's CPU RAM; the next request "
     "reloads it. Changing: off keeps it resident for the life of the Studio. "
     "This never touches the serving engine or VRAM."),
    ("controller_idle_seconds", "controller encoder idle seconds", "num",
     "How: idle timeout for the controller encoder. Does: how long it stays "
     "loaded after the last use. Changing: lower frees RAM sooner but reloads "
     "more often; 0.1 is the floor, 300 is the default. Ignored while the "
     "unload toggle is off."),
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
    ("resident_gib", "resident experts (GiB)", "num",
     "How: the GiB of experts kept in ordinary RAM for the ones the VRAM "
     "cache cannot hold (the Strata engine's resident tier). Does: how "
     "much of the model is served from RAM instead of the disk. Changing: "
     "more = a faster first request, less RAM for everything else. "
     "Measured on the 176B: 8 GiB took the cold first request from 33.8 s "
     "to 17.3 s and the storage read on the second from 1,415 MB to "
     "204 MB, with the warm rate unchanged. 0 = off."),
]

#: The Studio's own switches that belong to the harness rather than to a
#: chat.  The sandbox, the tool use and the auto-compact stay in the chat
#: footer (they are contextual), so they are not repeated here.
TOGGLES: list[tuple[str, str, str]] = [
    ("dev_mode", "developer mode",
     "How: the developer surface appears (the deep-research queue, the "
     "Docker setup tab, the raw status). Does: nothing but show things. "
     "Changing: on = those panes are visible."),
    ("harvest_enabled", "harvest the turns",
     "How: every chat turn (the query, the reply, the chunks the curation "
     "selected) is appended to harness_state/harvest.jsonl. Does: builds the "
     "dataset the encoder is trained on. Changing: on = the file grows with "
     "your own words - it is the training data, so it is your call."),
]

#: the select options per field
CHOICES: dict[str, list[tuple[str, str]]] = {
    "drift_prior": [("", "off (re-derive)"), ("on", "on (keep the numbers)")],
    "confidence_mode": [("off", "off"), ("single", "single"), ("mcdropout", "mcdropout")],
    "embedding_backend": [("local", "local (here, on the CPU)"),
                          ("served", "served (an embedding endpoint)")],
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


CONTROLLER_PANEL = """
<h3>Hardware profile &amp; settings controller</h3>
<p class="note">The E0 controller is deterministic: it reads a local hardware
snapshot and proposes only allowlisted runtime settings. Nothing leaves this
machine. A scan needs the checkbox; applying needs a second confirmation, and
the last change can be reverted. If you include request text, only its
deterministic features are written to the trace; the raw text is not stored.
“Mark accepted/rejected” appends a label to that trace.</p>
<div class="row" style="flex-wrap:wrap;gap:.35rem .8rem;align-items:center">
  <label class="inline" title="Read CPU, RAM, GPU/VRAM/GTT and storage. No prompts, models or network.">
    <input type="checkbox" id="ctrl-consent"> allow hardware scan
  </label>
  <button onclick="scanHardware(this)" title="POST /v1/profiles/scan with consent=hardware. Saves a local snapshot; it does not change settings.">Scan this machine</button>
  <button onclick="loadControllerStatus()" title="Reload the saved profile and proposals.">Refresh</button>
</div>
<div id="ctrl-profile" class="note" style="margin:.5rem 0">No profile loaded.</div>
<div class="row" style="flex-wrap:wrap;gap:.35rem .8rem;align-items:center">
  <input id="ctrl-model-gib" type="number" step="0.1" placeholder="model GiB" title="Optional: model weight size for a fit estimate." style="width:8rem">
  <input id="ctrl-native-ctx" type="number" placeholder="native ctx" title="Optional: the model's native context length." style="width:8rem">
  <input id="ctrl-ctx" type="number" placeholder="max ctx" title="Optional: the largest context the proposal may choose." style="width:8rem">
  <input id="ctrl-request" type="text" placeholder="request text (optional)" title="Used only for deterministic features (intent, context pressure, risk). Raw text is not stored in the trace." style="min-width:16rem">
  <label class="inline" title="Load the configured sentence-transformer on CPU and trace a 384/768-dimensional vector. Raw text is still not stored. This is explicit and can be slow.">
    <input type="checkbox" id="ctrl-embed-request"> trace encoder vector
  </label>
  <button onclick="proposeController(this)" title="POST /v1/controller/propose. Creates a proposal only; it does not apply anything.">Propose</button>
  <button onclick="revertController(this)" title="POST /v1/controller/revert with confirm=true. Restores the last applied values.">Revert last</button>
</div>
<div id="ctrl-proposal" class="note" style="margin-top:.75rem"></div>
<div id="ctrl-note" class="note" style="margin-top:.35rem"></div>
"""

CONTROLLER_SCRIPT = """
<script>
function ctrlNote(text, bad) {
  const el = document.getElementById('ctrl-note');
  if (el) { el.textContent = text || ''; el.style.color = bad ? '#ff6b6b' : ''; }
}
function ctrlProfileText(p) {
  if (!p) return 'No hardware profile yet. Tick consent and scan.';
  const cpu = p.cpu || {}, ram = p.ram || {};
  const cards = (p.gpus || []).map(g =>
    'card ' + g.index + ': ' + (g.free_gb != null ? g.free_gb + ' GiB free' : '?')
    + (g.display ? ' (display)' : '')).join(' · ');
  return (p.profile_id || 'profile') + ' · ' + (cpu.model || 'CPU ?')
    + ' · ' + (ram.available_gb != null ? ram.available_gb : '?') + ' GiB RAM'
    + (cards ? ' · ' + cards : ' · no GPU');
}
function renderControllerProposal(p) {
  const box = document.getElementById('ctrl-proposal');
  if (!box) return;
  box.textContent = '';
  if (!p) { box.textContent = 'No proposal yet.'; return; }
  const head = document.createElement('div');
  head.textContent = (p.id || 'proposal') + ' · confidence ' + p.confidence
    + ' · ' + (p.reason || '');
  box.appendChild(head);
  const list = document.createElement('div');
  list.className = 'note';
  list.style.margin = '.35rem 0';
  for (const [key, value] of Object.entries(p.changes || {})) {
    const row = document.createElement('div');
    row.textContent = key + ' = ' + value;
    list.appendChild(row);
  }
  box.appendChild(list);
  for (const line of (p.evidence || [])) {
    const row = document.createElement('div');
    row.className = 'note';
    row.textContent = '· ' + line;
    box.appendChild(row);
  }
  for (const line of (p.warnings || [])) {
    const row = document.createElement('div');
    row.className = 'note';
    row.style.color = '#FFB703';
    row.textContent = 'warning: ' + line;
    box.appendChild(row);
  }
  const apply = document.createElement('button');
  apply.textContent = 'Apply with confirmation';
  apply.style.marginTop = '.5rem';
  apply.onclick = () => applyController(p.id);
  box.appendChild(apply);
  const accepted = document.createElement('button');
  accepted.textContent = 'Mark accepted';
  accepted.style.margin = '.5rem .35rem 0 0';
  accepted.onclick = () => outcomeController(p.id, 'accepted');
  box.appendChild(accepted);
  const rejected = document.createElement('button');
  rejected.textContent = 'Mark rejected';
  rejected.style.marginTop = '.5rem';
  rejected.onclick = () => outcomeController(p.id, 'rejected');
  box.appendChild(rejected);
}
async function loadControllerStatus() {
  const profile = document.getElementById('ctrl-profile');
  try {
    const st = await api('/v1/controller/status');
    if (profile) profile.textContent = ctrlProfileText(st.profile);
    renderControllerProposal(st.latest_proposal);
  } catch (e) {
    if (profile) profile.textContent = 'controller status unavailable: ' + e.message;
  }
}
async function scanHardware(btn) {
  const consent = document.getElementById('ctrl-consent');
  if (!consent || !consent.checked) {
    ctrlNote('Tick “allow hardware scan” first.', true);
    return;
  }
  if (btn) btn.disabled = true;
  try {
    await api('/v1/profiles/scan', 'POST', {consent: 'hardware'});
    ctrlNote('Hardware profile saved locally.');
    await loadControllerStatus();
  } catch (e) {
    ctrlNote('scan failed: ' + e.message, true);
  } finally { if (btn) btn.disabled = false; }
}
async function proposeController(btn) {
  if (btn) btn.disabled = true;
  try {
    const body = {};
    const gib = num('ctrl-model-gib');
    const nativeCtx = num('ctrl-native-ctx');
    const ctx = num('ctrl-ctx');
    const request = document.getElementById('ctrl-request')?.value || '';
    const embed = document.getElementById('ctrl-embed-request')?.checked || false;
    if (gib != null) body.model_gib = gib;
    if (nativeCtx != null) body.native_ctx = nativeCtx;
    if (ctx != null) body.desired_ctx = ctx;
    if (request.trim()) body.request = request;
    if (embed) body.embed_request = true;
    const r = await api('/v1/controller/propose', 'POST', body);
    renderControllerProposal(r.proposal);
    ctrlNote('Proposal created. Review the evidence before applying.');
  } catch (e) {
    ctrlNote('proposal failed: ' + e.message, true);
  } finally { if (btn) btn.disabled = false; }
}
async function outcomeController(id, outcome) {
  try {
    await api('/v1/controller/outcome', 'POST', {proposal_id: id, outcome: outcome});
    ctrlNote('Recorded ' + outcome + ' for ' + id + '.');
  } catch (e) {
    ctrlNote('outcome failed: ' + e.message, true);
  }
}
async function applyController(id) {
  if (!confirm('Apply this controller proposal? The previous values are saved for rollback.')) return;
  try {
    await api('/v1/controller/apply', 'POST', {proposal_id: id, confirm: true});
    ctrlNote('Applied. Revert last restores the previous values.');
    await loadControllerStatus();
  } catch (e) {
    ctrlNote('apply failed: ' + e.message, true);
  }
}
async function revertController(btn) {
  if (!confirm('Revert the last applied controller change?')) return;
  if (btn) btn.disabled = true;
  try {
    await api('/v1/controller/revert', 'POST', {confirm: true});
    ctrlNote('Reverted the last controller change.');
    await loadControllerStatus();
  } catch (e) {
    ctrlNote('revert failed: ' + e.message, true);
  } finally { if (btn) btn.disabled = false; }
}
</script>
"""


def render_global_pane() -> str:
    """The Global harness controls pane."""
    encoder = _group(
        "The encoder",
        "The small model that does the retrieval and the memory management: "
        "it embeds the chunks, ranks them, dedups them and ages them. Pick "
        "the one you want here - the list is a suggestion, any "
        "sentence-transformers id works.",
        ENCODER_FIELDS, "ge")
    candidates = "".join(
        f'<option value="{name}"></option>' for name in ENCODER_CANDIDATES)
    encoder = encoder.replace(
        '<input type="text" id="ge-embedding_model"',
        f'<input type="text" list="encoder-candidates" id="ge-embedding_model"')
    encoder += (f'<datalist id="encoder-candidates">{candidates}</datalist>')
    memory = _group(
        "Memory &amp; RAG",
        "The curation baseline: what a conversation gets when it overrides "
        "nothing.",
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
<p class="note">Everything the harness does, in one place: the encoder that
does the retrieval and the memory management, the curation it applies, the
quality a cascade run asks for, what an MCP client may touch, and the runtime
a new engine is launched with. There is no second menu for any of it - these
are the settings, and they apply to the next conversation or run.</p>
{encoder}
{memory}
{runtime}
<div class="row" style="flex-wrap:wrap;gap:.35rem .8rem">{toggles}</div>
{quality}
{mcp}
{CONTROLLER_PANEL}
{CONTROLLER_SCRIPT}
<div class="row" style="margin-top:1rem">
<button onclick="saveGlobalControls(this)" title="How: PUT /v1/studio/settings with the control blocks. Does: stores the baseline the harness starts from. Changing: takes effect for the next conversation or cascade run; the encoder takes effect when the Studio restarts.">Save the baseline</button>
<span class="note" id="global-note"></span>
</div>
</section>
</div>"""
