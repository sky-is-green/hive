"""Server-rendered report views (Seam B): runs, reports and Cascade.

Plain HTML, no JS dependencies. Renders one ``run_report.json`` bundle: the
post-run PES headline with its weighted components, the P1–P11 verdict table,
the deterministic P2 retrieval diagnostic, comb (P11) totals, and the
baselines comparison. Every dynamic value is HTML-escaped; missing blocks
render as em dashes rather than erroring, so partial bundles from in-flight
runs stay viewable.

This file is UTF-8 and must stay that way — it contains arrows, stars and
ellipses in the console UI. Never round-trip it through PowerShell text
cmdlets (they decode as ANSI on Windows and mojibake every non-ASCII char).
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from fastapi import HTTPException


# Paper weights for the PES composite (README §3.1).
_PES_COMPONENTS = [
    ("retrieval_precision", 0.30),
    ("routing_accuracy", 0.20),
    ("latency_health", 0.20),
    ("throughput_health", 0.15),
    ("context_utilization", 0.15),
]
_BAND_FLOORS = [(80, "band-green"), (60, "band-yellow"), (40, "band-red"),
                (0, "band-critical")]

_CSS = """
 body { font-family: Segoe UI, system-ui, sans-serif; margin: 2rem auto; max-width: 60rem;
        color: #1c2733; background: #f7f9fb; padding: 0 1rem; }
 h1 { font-size: 1.4rem; } h2 { font-size: 1.05rem; margin-top: 2rem;
      border-bottom: 1px solid #dde5ec; padding-bottom: .3rem; }
 table { border-collapse: collapse; width: 100%; background: #fff; }
 th, td { text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #e8edf2;
          font-size: .92rem; }
 th { background: #eef2f6; }
 .cards { display: flex; gap: 1rem; flex-wrap: wrap; }
 .card { background: #fff; border-radius: 8px; padding: 1rem 1.4rem;
         box-shadow: 0 1px 3px rgba(16,32,48,.08); min-width: 150px; }
 .card span { color: #5a6b7d; font-size: .85rem; }
 .card b { display: block; font-size: 1.6rem; }
 .kv-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
            gap: .4rem 1.4rem; background: #fff; padding: 1rem; border-radius: 8px; }
 .kv span { color: #5a6b7d; } .kv b { float: right; font-variant-numeric: tabular-nums; }
 .band-green { color: #157a3e; } .band-yellow { color: #9a6b00; }
 .band-red { color: #b3372c; } .band-critical { color: #8a1f1f; font-weight: 700; }
 .st-pass { color: #157a3e; font-weight: 600; } .st-fail { color: #b3372c; font-weight: 600; }
 .st-skip { color: #9a6b00; } .st-report { color: #456; }
 .note { color: #1c2733; font-size: .85rem; }
 code { background: #000; color: #FFDD00; padding: .15rem .4rem; border-radius: 4px; border: 1px solid #222; }
 .note code { background: #000; color: #FFDD00; border-color: #FFDD00; }
 ul.runs { list-style: none; padding: 0; } ul.runs li { margin: .35rem 0; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _fmt(value: object, nd: int = 1) -> str:
    if value is None or value == "":
        return "&mdash;"
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return _esc(value)


def _pes_band_class(score: object) -> str:
    try:
        s = float(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    for floor, cls in _BAND_FLOORS:
        if s >= floor:
            return cls
    return ""


def _status_cell(status: object) -> str:
    s = _esc(status or "")
    return f"<span class='st-{s.lower()}'>{s}</span>"


def resolve_run_dir(runs_root: Path, run_dir: str) -> Path:
    """Resolve + confine a run_dir under runs_root (path-traversal safe)."""
    root = runs_root.resolve()
    target = (root / run_dir).resolve()
    if target != root and root not in target.parents:
        raise HTTPException(400, "run_dir must stay inside the runs directory")
    return target


def _comb_totals(comb: dict) -> dict:
    totals = {"archived": 0, "resurrected": 0, "comb_hits": 0, "gate_fired": 0}
    conversations = comb.get("conversations")
    if isinstance(conversations, list):
        for entry in conversations:
            if not isinstance(entry, dict):
                continue
            for key in totals:
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
    return totals


def _baseline_rows(report: dict, composite: object) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = [("Splinter (post-run PES)", composite)]
    for label, key in (("LM-Studio rolling", "baseline_lm_studio"),
                       ("FIFO truncation", "baseline_fifo")):
        blob = report.get(key)
        if not isinstance(blob, dict):
            nested = report.get("baselines")
            blob = nested.get(key) if isinstance(nested, dict) else None
        if isinstance(blob, dict):
            aggregate = blob.get("aggregate")
            if isinstance(aggregate, dict):
                rows.append((label, aggregate.get("avg_pes")))
    return rows


def render_report_page(report: dict, run_name: str) -> str:
    """One run_report.json bundle as a standalone HTML page."""
    if isinstance(report, dict) and report.get("kind") == "training":
        return render_training_report(report, run_name)
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    post = report.get("post_run_pes") if isinstance(report.get("post_run_pes"), dict) else {}
    components = post.get("components") if isinstance(post.get("components"), dict) else {}
    diag = report.get("retrieval_diagnostic") \
        if isinstance(report.get("retrieval_diagnostic"), dict) else {}
    comb = report.get("comb") if isinstance(report.get("comb"), dict) else {}
    comb_t = _comb_totals(comb)
    # generate_data has used two shapes across versions: composite/components
    # and pes/breakdown — accept both.
    composite = post.get("composite", post.get("pes"))
    components = post.get("components")
    if not isinstance(components, dict):
        components = post.get("breakdown")
    if not isinstance(components, dict):
        components = {}
    band = str(post.get("band") or "")

    def _band_class() -> str:
        if band:
            for token in ("green", "yellow", "red", "critical"):
                if token in band.lower():
                    return f"band-{token}"
            return ""
        return _pes_band_class(composite)

    protocol_rows = "".join(
        "<tr>"
        f"<td>{_esc(p.get('id', ''))}</td>"
        f"<td>{_status_cell(p.get('status'))}</td>"
        f"<td>{_esc(p.get('title', ''))}</td>"
        f"<td class='note'>{_esc(p.get('note', ''))}</td>"
        "</tr>"
        for p in (report.get("protocol") or [])
        if isinstance(p, dict)
    ) or "<tr><td colspan='4' class='note'>no protocol block</td></tr>"

    component_rows = "".join(
        "<tr>"
        f"<td>{_esc(name.replace('_', ' ').title())}</td>"
        f"<td>{weight:.2f}</td>"
        f"<td>{_fmt(components.get(name))}</td>"
        "</tr>"
        for name, weight in _PES_COMPONENTS
    )

    baseline_rows = "".join(
        "<tr>"
        f"<td>{_esc(label)}</td>"
        f"<td class='{_pes_band_class(pes)}'><b>{_fmt(pes)}</b></td>"
        "</tr>"
        for label, pes in _baseline_rows(report, composite)
    )

    def kv(label: str, value: object, nd: int = 1) -> str:
        return f"<div class='kv'><span>{label}</span><b>{_fmt(value, nd)}</b></div>"

    avg_ms = aggregate.get("avg_total_ms")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><link rel="icon" href="/favicon.svg" type="image/svg+xml">
<title>HiveBench report &mdash; {_esc(run_name)}</title>
<style>{_get_server_css()}</style></head><body>
<h1>HiveBench report &mdash; <code>{_esc(run_name)}</code></h1>
<p class="note">mode <b>{_esc(report.get('mode') or '&mdash;')}</b> &middot;
backend <b>{_esc(report.get('backend') or '&mdash;')}</b> &middot; run
<b>{_esc(report.get('run_id') or '&mdash;')}</b></p>
<div class="cards">
<div class="card"><span>PES (post-run)</span><b class="{_band_class()}">{_fmt(composite)}</b></div>
<div class="card"><span>Avg per-turn PES</span><b>{_fmt(aggregate.get('avg_pes'))}</b></div>
<div class="card"><span>User turns</span><b>{_fmt(aggregate.get('user_turns'), 0)}</b></div>
<div class="card"><span>Avg turn</span><b>{_fmt(avg_ms / 1000 if isinstance(avg_ms, (int, float)) else None)}s</b></div>
</div>

<h2>PES components (post-run)</h2>
<table><tr><th>Component</th><th>Weight</th><th>Score</th></tr>
{component_rows}
</table>

<h2>P1&ndash;P11 verdicts</h2>
<table><tr><th>ID</th><th>Status</th><th>Prediction</th><th>Note</th></tr>
{protocol_rows}
</table>

<h2>Deterministic retrieval diagnostic (P2)</h2>
<div class="kv-grid">
{kv('Recall (all turns)', diag.get('retrieval_recall'))}
{kv('Recall (retrievable turns)', diag.get('retrieval_recall_retrievable'))}
{kv('Ingestion rate', diag.get('ingestion_rate'))}
{kv('Perfect-splinter ceiling', diag.get('perfect_hive_ceiling'))}
{kv('Precision (sentence proxy)', diag.get('retrieval_precision'))}
</div>

<h2>Comb (P11 surplus tier)</h2>
<div class="kv-grid">
{kv('Archived', comb_t['archived'], 0)}
{kv('Resurrected', comb_t['resurrected'], 0)}
{kv('Comb hits', comb_t['comb_hits'], 0)}
{kv('Gate fired', comb_t['gate_fired'], 0)}
</div>

<h2>Baselines comparison</h2>
<table><tr><th>System</th><th>PES</th></tr>
{baseline_rows}
</table>

<p class="note"><a href="/runs">&larr; all runs</a></p>
</body></html>"""



def render_nav(active: str = "") -> str:
    """Shared top nav bar for all Studio pages.

    Labels are short for humans; every link also carries a title that names
    the surface's job.  ``aria-current`` marks the active page for assistive
    tech and for deterministic page scraping.
    """
    items = [
        ("runs", "/runs", "Runs", "Browse saved run bundles and reports"),
        ("cascade", "/cascade", "Cascade", "Generate, detect, repair, and escalate answers"),
        ("senses", "/cascade#senses-panel", "Senses",
         "Upload a sensory input and preview its cascade route"),
        ("docs", "/docs", "API docs", "Read the HTTP API and console route map"),
    ]
    parts = []
    for key, href, label, title in items:
        cls = "active" if key == active else ""
        current = " aria-current='page'" if key == active else ""
        parts.append(
            f"<a href='{href}' title='{title}'{current}><button type='button' "
            f"class='{cls}'>{label}</button></a>"
        )
    return ("<nav style='display:flex;gap:.4rem;align-items:center;"
            "margin:.3rem 0 1rem;flex-wrap:wrap'>" + " ".join(parts) + "</nav>")

def render_runs_page(entries: list[dict]) -> str:
    """Index of available run bundles."""
    def _kind(e: dict) -> str:
        kind = e.get("kind") or "benchmark"
        return f" · <b>{_esc(kind)}</b>" if kind != "benchmark" else ""

    items = "".join(
        f"<li><a href='/view/{_esc(e['name'])}'><code>{_esc(e['name'])}</code></a> "
        f"<span class='note'>modified {_esc(e['modified'])}"
        f"{_kind(e)}{'' if e.get('has_report') else ' · no run_report.json yet'}</span></li>"
        for e in entries
    ) or "<li class='note'>no runs yet</li>"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><link rel="icon" href="/favicon.svg" type="image/svg+xml"><title>HiveBench runs</title>
<style>{_get_server_css()}</style></head><body>
<h1>HiveBench run bundles</h1>
{render_nav("runs")}
<ul class="runs">{items}</ul>
</body></html>"""


def _training_kv(label: str, value: object, nd: int = 2) -> str:
    return f"<div class='kv'><span>{label}</span><b>{_fmt(value, nd)}</b></div>"


def render_training_report(report: dict, run_name: str) -> str:
    """One training run bundle (``kind == 'training'``) as a standalone page."""
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    progress = report.get("progress") if isinstance(report.get("progress"), dict) else {}
    config = report.get("config") if isinstance(report.get("config"), dict) else {}
    regions = report.get("regions") if isinstance(report.get("regions"), list) else []

    ratio = aggregate.get("mean_ratio")
    retention = report.get("retention")
    if retention is None and isinstance(ratio, (int, float)) and ratio:
        retention = round(100.0 / float(ratio), 1)

    region_rows = "".join(
        "<tr>"
        f"<td>{_esc(r.get('name', ''))}</td>"
        f"<td>{_fmt(r.get('start_token'), 0)}</td>"
        f"<td>{_fmt(r.get('n_windows'), 0)}</td>"
        f"<td>{_fmt(r.get('ppl'), 3)}</td>"
        f"<td>{_fmt(r.get('teacher_ppl'), 3)}</td>"
        f"<td><b>{_fmt(r.get('ratio'), 4)}</b></td>"
        "</tr>"
        for r in regions if isinstance(r, dict)
    ) or "<tr><td colspan='6' class='note'>no evaluation yet &mdash; run still in progress</td></tr>"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><link rel="icon" href="/favicon.svg" type="image/svg+xml">
<title>Hive-Ternary run &mdash; {_esc(run_name)}</title>
<style>{_get_server_css()}</style></head><body>
<h1>Hive-Ternary &mdash; <code>{_esc(run_name)}</code></h1>
<p class="note">engine <b>{_esc(report.get('engine') or '&mdash;')}</b> &middot;
recipe <b>{_esc(report.get('method') or '&mdash;')}</b> &middot;
status <b>{_esc(report.get('status') or '&mdash;')}</b></p>
<div class="cards">
<div class="card"><span>Mean ratio</span><b>{_fmt(ratio, 4)}</b></div>
<div class="card"><span>Retention</span><b>{_fmt(retention, 1)}%</b></div>
<div class="card"><span>Best region</span><b>{_fmt(aggregate.get('min_ratio'), 4)}</b></div>
<div class="card"><span>Regions</span><b>{_fmt(aggregate.get('n_regions'), 0)}</b></div>
</div>

<h2>Per-region retention (lower ratio is better)</h2>
<table><tr><th>Region</th><th>Start</th><th>Windows</th><th>PPL</th><th>Teacher</th><th>Ratio</th></tr>
{region_rows}
</table>

<h2>Run</h2>
<div class="kv-grid">
{_training_kv('Step', progress.get('step'), 0)}
{_training_kv('Total steps', progress.get('total_steps'), 0)}
{_training_kv('Loss', progress.get('loss'))}
{_training_kv('Deployed ratio', progress.get('deployed_ratio'))}
{_training_kv('Init ratio', progress.get('init_ratio'))}
{_training_kv('Elapsed (s)', progress.get('elapsed_s'), 0)}
</div>

<h2>Config</h2>
<div class="kv-grid">
{_training_kv('Model', config.get('model_dir'))}
{_training_kv('Corpus', config.get('corpus'))}
{_training_kv('Steps', config.get('steps'), 0)}
{_training_kv('Seq', config.get('seq'), 0)}
{_training_kv('LR', config.get('lr'))}
{_training_kv('Device', config.get('device'))}
{_training_kv('Eval regions', config.get('eval_regions'), 0)}
{_training_kv('Eval windows', config.get('eval_windows'), 0)}
</div>

<p class="note"><a href="/runs">&larr; all runs</a></p>
</body></html>"""


_STUDIO_CSS_PATH = Path(__file__).with_name("studio.css")

def _get_server_css() -> str:
    """Studio CSS: base + studio overrides, re-read on each request so refresh picks up edits without restart."""
    try:
        extra = _STUDIO_CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        extra = ""
    return _CSS + extra



def render_cascade_page(task_files: list[str]) -> str:
    """Hive Cascade launch console: every knob the runner accepts, in one form."""
    opts = "".join(f"<option value='{_esc(t)}'>{_esc(t)}</option>" for t in task_files)
    if not opts:
        opts = "<option value='tasks-hard.json'>tasks-hard.json</option>"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><link rel="icon" href="/favicon.svg" type="image/svg+xml"><meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"><meta http-equiv="Pragma" content="no-cache"><meta http-equiv="Expires" content="0"><title>Hive Cascade</title>
<style>{_get_server_css()}</style></head><body>
<h1><img src="/bee.svg" alt="" style="height:2rem;vertical-align:middle;margin-right:.4rem">Hive Cascade <span class="note">generate → detect → repair → escalate</span></h1>
{render_nav("cascade")}
<p class="note">The cascade is <b>fully in your control</b>: every knob the runner accepts is exposed here. The learned gate is <b>off by default</b> (its own pre-registered rule); <code>--gate intern-0.8b</code> is an opt-in third trigger. The escalation lane harvests every ladder-resolved item to <code>escalation.jsonl</code> next to the report.</p>

<section id="senses-panel" style="max-width:72rem">
<h2>Senses: upload an input</h2>
<p class="note">Image and text uploads are classified and encoded here. An image with a routed sense goes to <b>P6</b>; audio and video are stored and encoded but have no cascade path yet. The panel stays disabled until a sensory encoder is attached — it does not substitute a fake embedding.</p>
<div class="grid-2col">
  <label title="One image, audio clip, video, or text/code document.">Input file <input id="senses-file" type="file" accept="image/*,audio/*,video/*,text/*,.md,.markdown,.json,.jsonl,.yaml,.yml,.toml,.py,.js,.ts,.c,.h,.cpp,.rs,.go,.java,.sh,.sql,.html,.css,.xml,.csv,.tsv"></label>
</div>
<div class="row" style="margin-top:.75rem">
  <button type="button" id="senses-button" onclick="uploadSense()" disabled>Ingest upload</button>
  <span id="senses-status" class="note" style="margin-left:1rem">checking encoder…</span>
</div>
<pre id="senses-result" style="display:none; margin-top:1rem; max-height:24rem; overflow:auto"></pre>
</section>

<form id="cascade-form" onsubmit="return launchCascade(event)" style="max-width:72rem">
<section>
<h2 style="margin-top:0">Run</h2>
<div class="grid-2col">
  <label title="Which task bundle to run.">Task file <select name="tasks">{opts}</select></label>
  <label title="Name for this run (used for the output directory).">Run name <input name="run_name" placeholder="cascade-YYYYMMDD-HHMMSS"></label>
  <label title="Run only the first N tasks (0 = all).">Limit (0 = all) <input name="limit" type="number" min="0" value="0"></label>
  <label title="Router model: local (default) or jev (Tiny-Jev).">Router <select name="router"><option value="local">local</option><option value="jev">jev</option></select></label>
  <label title="gate: router decisions skip local generation. measure: always generate locally.">Router mode <select name="router_mode"><option value="gate">gate</option><option value="measure">measure</option></select></label>
</div>
</section>

<details style="margin:1rem 0">
<summary style="cursor:pointer;font-weight:700;padding:.5rem 0;color:#000000">Advanced settings (judge, repair, gate, model, API) ▾</summary>
<section>
<h2>Role coverage</h2>
<p class="note">Which candidate models cover which measurement role, and what the router table costs. Read-only — the runner uses the local tier for generation and the API for escalation.</p>
<div id="cascade-roles" class="note" style="margin:.5rem 0">loading…</div>
</section>
<section>
<h2>Judge &amp; repair</h2>
<div class="grid-2col">
  <label title="Verifier: deterministic (consistency + code_exec, gold-free) or learned.">Judge <select name="judge"><option value="deterministic">deterministic (consistency + code_exec)</option><option value="learned">learned</option></select></label>
  <label title="Accept candidate when P(correct) >= this.">Judge threshold <input name="judge_threshold" type="number" step="0.01" value="0.5"></label>
  <label title="Repair rejected answers: auto (on with deterministic judge), on, or off.">Repair <select name="repair"><option value="auto">auto</option><option value="on">on</option><option value="off">off</option></select></label>
  <label title="Ask a human before API escalation. auto = on only with TTY + API fallback.">Human gate <select name="human_gate"><option value="auto">auto</option><option value="on">on</option><option value="off">off</option></select></label>
</div>
<p class="note" id="cascade-baseline-note"></p>
</section>

<section>
<h2>Learned gate (third trigger, blind-spot only)</h2>
<div class="grid-2col">
  <label title="Learned gate (third trigger, blind-spot only). off by default.">Gate <select name="gate"><option value="off">off (default)</option><option value="intern-0.8b">intern-0.8b</option><option value="scion">scion (needs serve restart)</option></select></label>
  <label title="Which readout head to use for the gate. L0.75 = precision-first default.">Gate head <select name="gate_head"><option value="L0.5">L0.5</option><option value="L0.75" selected>L0.75</option><option value="L1">L1</option><option value="ridge-concat">ridge-concat</option><option value="mlp-concat">mlp-concat</option></select></label>
  <label title="Fire the gate when P(correct) < this threshold.">Gate threshold <input name="gate_threshold" type="number" step="0.01" value="0.5"></label>
</div>
<p class="note">Scores a draft only when the deterministic detectors are silent. A fired gate is one more rejection finding (greedy repair quotes it); the repaired answer is re-scored.</p>
</section>

<section>
<h2>Local model</h2>
<div class="grid-2col">
  <label title="Sampling temperature for local drafts. Higher = more diverse.">Temperature <input name="local_temperature" type="number" step="0.1" value="0.7"></label>
  <label title="Token budget per draft.">Max tokens <input name="local_max_tokens" type="number" value="1024"></label>
  <label title="Reasoning effort for the local model. none = fastest.">Reasoning effort <select name="local_reasoning_effort"><option value="none">none</option><option value="low">low</option><option value="medium">medium</option><option value="high">high</option></select></label>
  <label title="Token budget for reasoning (0 = auto).">Reasoning budget tokens <input name="local_reasoning_budget_tokens" type="number" value="0"></label>
</div>
</section>

<section>
<h2>API fallback (optional)</h2>
<div class="grid-2col">
  <label title="OpenAI-compatible API endpoint for escalated items.">Base URL <input name="api_fallback" placeholder="http://127.0.0.1:4000/v1"></label>
  <label title="Model name for the API fallback.">Model <input name="api_fallback_model" placeholder="model name"></label>
  <label title="Bearer key for the API fallback.">Key <input name="api_fallback_key" type="password" placeholder="bearer key"></label>
</div>
</section>

<section>
<h2>Harvest</h2>
<label style="display:flex; align-items:center; gap:.5rem"><input name="no_escalation_lane" type="checkbox"> Disable escalation lane (lean smoke runs)</label>
<p class="note">By default every ladder-resolved item is written to <code>escalation.jsonl</code> next to the report (training-side only).</p>
</section>

</details>
<div class="row" style="margin-top:1rem">
  <button type="submit" style="background:#157a3e;color:#fff;font-weight:700;padding:.6rem 1.4rem">Launch cascade</button>
  <span id="cascade-status" class="note" style="margin-left:1rem"></span>
</div>
</form>

<pre id="cascade-result" style="display:none; margin-top:1rem"></pre>

<script>
async function launchCascade(ev) {{
  ev.preventDefault();
  const form = document.getElementById('cascade-form');
  const fd = new FormData(form);
  const body = {{}};
  for (const [k, v] of fd.entries()) {{
    if (k === 'no_escalation_lane') continue;
    if (v === '') continue;
    if (k === 'limit' || k === 'local_max_tokens' || k === 'local_reasoning_budget_tokens') body[k] = parseInt(v, 10);
    else if (k === 'gate_threshold' || k === 'judge_threshold' || k === 'local_temperature') body[k] = parseFloat(v);
    else body[k] = v;
  }}
  if (form.querySelector('[name=no_escalation_lane]').checked) body.no_escalation_lane = true;
  const status = document.getElementById('cascade-status');
  const result = document.getElementById('cascade-result');
  status.textContent = 'launching…';
  result.style.display = 'none';
  try {{
    const r = await fetch('/v1/cascade/run', {{ method: 'POST', headers: {{'Content-Type':'application/json'}}, body: JSON.stringify(body) }});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || JSON.stringify(data));
    status.textContent = 'launched: ' + data.run + ' (pid ' + data.pid + ')';
    result.textContent = JSON.stringify(data, null, 2);
    result.style.display = 'block';
  }} catch (e) {{
    status.textContent = 'error: ' + e.message;
    status.style.color = '#b3372c';
  }}
  return false;
}}

/* The form starts from the Global tab's Quality block: the harness-wide
   baseline every run begins with.  Change a knob here and this run uses
   yours; leave it and the baseline stands. */
(async () => {{
  try {{
    const r = await fetch('/v1/studio/settings');
    if (!r.ok) return;
    const st = await r.json();
    const q = (st && st.quality_defaults) || {{}};
    const form = document.getElementById('cascade-form');
    if (!form) return;
    let n = 0;
    for (const [k, v] of Object.entries(q)) {{
      if (v === '' || v === null || v === undefined) continue;
      const el = form.querySelector('[name="' + k + '"]');
      if (!el) continue;
      el.value = v;
      n++;
    }}
    const note = document.getElementById('cascade-baseline-note');
    if (note && n) {{
      note.textContent = n + ' knob(s) start from the Global tab\\'s baseline '
        + '(the Quality block) - change one here and this run uses yours.';
    }}
  }} catch (e) {{ /* the form still works without the baseline */ }}
}})();

async function refreshSenses() {{
  const status = document.getElementById('senses-status');
  const button = document.getElementById('senses-button');
  if (!status || !button) return;
  try {{
    const r = await fetch('/v1/cascade/senses/status');
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'status failed');
    if (!data.configured) {{
      status.textContent = 'no sensory encoder attached (model loading is deferred)';
      button.disabled = true;
      return;
    }}
    status.textContent = data.model + ' · ' + data.dimensions + 'd · ≤'
      + data.max_bytes + ' bytes';
    button.disabled = false;
  }} catch (e) {{
    status.textContent = 'senses unavailable: ' + e.message;
    button.disabled = true;
  }}
}}

async function uploadSense() {{
  const input = document.getElementById('senses-file');
  const status = document.getElementById('senses-status');
  const result = document.getElementById('senses-result');
  const button = document.getElementById('senses-button');
  if (!input || !input.files || !input.files.length) {{
    status.textContent = 'choose a file first';
    return;
  }}
  const body = new FormData();
  body.append('file', input.files[0]);
  status.textContent = 'ingesting…';
  status.style.color = '';
  result.style.display = 'none';
  button.disabled = true;
  try {{
    const r = await fetch('/v1/cascade/senses', {{method: 'POST', body: body}});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || JSON.stringify(data));
    status.textContent = data.modality + ' → ' + (data.path || 'no cascade path yet');
    result.textContent = JSON.stringify(data, null, 2);
    result.style.display = 'block';
  }} catch (e) {{
    status.textContent = 'error: ' + e.message;
    status.style.color = '#b3372c';
  }} finally {{
    button.disabled = false;
  }}
}}

refreshSenses();
</script>
</body></html>"""
