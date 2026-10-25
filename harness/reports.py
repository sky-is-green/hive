"""Server-rendered report views (Seam B) + the Studio console page.

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

from harness.global_controls import render_global_pane

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
        ("server", "/server", "Studio", "Configure engines, models, and the loaded chat"),
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


_TRAINING_PANE = r"""
<div id="tab-training" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Training <span class="note">Unsloth Core &middot; Hive-Ternary</span></h2>
<div class="row">
  <label class="inline">Engine <select id="tr-engine" onchange="trSyncMethods()"></select></label>
  <label class="inline">Recipe <select id="tr-method"></select></label>
  <label class="inline">Run name <input id="tr-name" size="16" placeholder="ste-rotate 20k"></label>
</div>
<div class="row"><label class="inline" style="flex:1">Model dir
  <input id="tr-model" size="34" placeholder="/path/to/hf base model"></label></div>
<div class="row"><label class="inline" style="flex:1">Corpus
  <input id="tr-corpus" size="34" placeholder="/path/to/corpus.txt"></label></div>
<div class="row">
  <label class="inline">steps <input id="tr-steps" type="number" value="20000" size="6"></label>
  <label class="inline">seq <input id="tr-seq" type="number" value="512" size="5"></label>
  <label class="inline">lr <input id="tr-lr" value="5e-5" size="7"></label>
  <label class="inline">device <input id="tr-device" value="cuda:1" size="7"></label>
  <label class="inline">seed <input id="tr-seed" type="number" value="1337" size="6"></label>
  <label class="inline">eval regions <input id="tr-regions" type="number" value="8" size="3"></label>
  <label class="inline">eval windows <input id="tr-windows" type="number" value="8" size="3"></label>
</div>
<div class="row">
  <button onclick="trLaunch(this)">Start training</button>
  <span id="tr-msg" class="note"></span>
</div>
<div id="tr-status" class="kv-grid"></div>
<pre id="tr-log" style="max-height:240px;overflow:auto"></pre>
<p class="note">One 512-token window per step &mdash; batch size is not a control here.
History lives in <a href="/runs">Runs</a>; ternary reports render at <code>/view/&lt;run&gt;</code>.</p>
</section>

<section>
<h2 style="margin-top:0">Preset trainer <span class="note">evidence-driven composition iteration</span></h2>
<p class="note">Your agent gets better at your workflow the more you use it: mine session logs for
successful tool-use patterns, draft a candidate preset, then compare it against the baseline.</p>
<div class="row">
  <button onclick="trMineEvidence(this)">Mine evidence</button>
  <label class="inline" style="flex:1">Baseline preset <input id="tr-trainer-baseline" size="34"
    placeholder="/path/to/baseline-preset.yml" title="How: POST /v1/trainer/draft baseline. Does: The preset the candidate is derived from. Changing: Point at the preset you want to improve."></label>
  <button onclick="trDraftCandidate(this)">Draft candidate</button>
</div>
<div id="tr-trainer-summary" class="note" style="margin:.5rem 0"></div>
<pre id="tr-trainer-log" style="max-height:200px;overflow:auto;display:none"></pre>
</section>
<script>
const TR_CATALOG = __CATALOG__;
let trRun = null, trTimer = null;
function trFillEngines(){
  const es = document.getElementById('tr-engine'); if(!es) return;
  es.innerHTML = '';
  for (const e of TR_CATALOG){
    const o = document.createElement('option'); o.value = e.id;
    o.textContent = e.label + (e.external ? ' (external)' : '');
    es.appendChild(o);
  }
  trSyncMethods();
}
function trSyncMethods(){
  const ev = document.getElementById('tr-engine').value;
  const ms = document.getElementById('tr-method'); if(!ms) return;
  ms.innerHTML = '';
  const engine = TR_CATALOG.find(e => e.id === ev); if(!engine) return;
  for (const m of engine.methods){
    const o = document.createElement('option'); o.value = m.id;
    o.textContent = m.label + (m.implemented ? '' : ' \u2014 not implemented');
    o.disabled = !m.implemented;
    ms.appendChild(o);
  }
}
function trFields(){
  const val = id => { const el = document.getElementById(id); return el && el.value.trim() !== '' ? el.value.trim() : undefined; };
  return {model_dir: val('tr-model'), corpus: val('tr-corpus'),
          steps: val('tr-steps'), seq: val('tr-seq'), lr: val('tr-lr'),
          device: val('tr-device'), seed: val('tr-seed'),
          eval_regions: val('tr-regions'), eval_windows: val('tr-windows')};
}
async function trLaunch(btn){
  const msg = document.getElementById('tr-msg');
  const method = document.getElementById('tr-method').value;
  if(!method){ msg.textContent = 'pick a recipe'; return; }
  msg.textContent = 'launching\u2026';
  try{
    const r = await api('/v1/training/launch', 'POST',
      {engine: document.getElementById('tr-engine').value, method,
       name: document.getElementById('tr-name').value, fields: trFields()});
    trRun = r.run_dir;
    msg.textContent = 'started ' + r.run_dir + ' (pid ' + r.pid + ')';
    trPoll();
    if(trTimer) clearInterval(trTimer);
    trTimer = setInterval(trPoll, 3000);
  }catch(e){ msg.textContent = 'launch failed: ' + e.message; }
}
async function trPoll(){
  if(!trRun) return;
  try{
    const s = await api('/v1/training/status/' + encodeURIComponent(trRun));
    const p = s.progress || {};
    const rows = [['state', s.state], ['step', (p.step||0) + ' / ' + (p.total_steps||'?')],
                  ['loss', p.loss], ['deployed ratio', p.deployed_ratio],
                  ['init ratio', p.init_ratio], ['elapsed s', p.elapsed_s]];
    document.getElementById('tr-status').innerHTML = rows.map(([k,v]) =>
      "<div class='kv'><span>" + k + "</span><b>" +
      (v === undefined || v === null ? '&mdash;' : v) + "</b></div>").join('');
    const logEl = document.getElementById('tr-log');
    if (logEl) logEl.textContent = s.log || '';
    if(s.state === 'finished' || s.state === 'stopped'){
      if(trTimer){ clearInterval(trTimer); trTimer = null; }
      document.getElementById('tr-msg').innerHTML =
        "done &mdash; <a href='/view/" + encodeURIComponent(trRun) + "'>open report</a>";
    }
  }catch(e){ /* keep polling while the run settles */ }
}
document.addEventListener('DOMContentLoaded', trFillEngines);
const trTabBtn = document.querySelector('[data-tab="tab-training"]');
if(trTabBtn) trTabBtn.addEventListener('click', () => setTimeout(trFillEngines, 10));

/* ------------------------- preset trainer -------------------------- */
function trTrainerLog(text) {
  const el = document.getElementById('tr-trainer-log');
  if (el) { el.style.display = 'block'; el.textContent = text; }
}
async function trMineEvidence(btn) {
  const out = document.getElementById('tr-trainer-summary');
  if (btn) btn.disabled = true;
  if (out) out.textContent = 'mining session logs…';
  try {
    const r = await api('/v1/trainer/evidence');
    const s = r.summary || {};
    const ev = r.evidence || [];
    if (out) out.innerHTML = 'Mined <b>' + ev.length + '</b> evidence item(s) from your sessions. '
      + Object.entries(s).map(([k,v]) => k + ': ' + (typeof v === 'object' ? JSON.stringify(v) : v)).join(' · ');
  } catch (e) {
    if (out) out.textContent = 'evidence mining failed: ' + e;
  } finally {
    if (btn) btn.disabled = false;
  }
}
async function trDraftCandidate(btn) {
  const baseline = (document.getElementById('tr-trainer-baseline')?.value || '').trim();
  const out = document.getElementById('tr-trainer-summary');
  if (!baseline) { if (out) out.textContent = 'set a baseline preset path first'; return; }
  if (btn) btn.disabled = true;
  if (out) out.textContent = 'drafting candidate…';
  try {
    const r = await api('/v1/trainer/draft', 'POST', {baseline: baseline});
    trTrainerLog(JSON.stringify(r, null, 2));
    if (out) out.textContent = 'Drafted ' + (r.candidate || r.name || 'candidate')
      + ' — evaluate it with POST /v1/trainer/evaluate, then promote it from the candidate directory.';
  } catch (e) {
    if (out) out.textContent = 'draft failed: ' + e;
  } finally {
    if (btn) btn.disabled = false;
  }
}
</script>
</div>
"""


def render_training_pane() -> str:
    """The Training tab pane (self-contained HTML + script)."""
    import json as _json

    from harness.training import engine_catalog

    return _TRAINING_PANE.replace("__CATALOG__", _json.dumps(engine_catalog()))


_STUDIO_CSS_PATH = Path(__file__).with_name("studio.css")

def _get_server_css() -> str:
    """Studio CSS: base + studio overrides, re-read on each request so refresh picks up edits without restart."""
    try:
        extra = _STUDIO_CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        extra = ""
    return _CSS + extra



SETTINGS_INDEX = [
  {
    "label": "Model (engine profile)",
    "desc": "GGUF weights for -m",
    "cat": "Model",
    "tab": "tab-engines",
    "id": "eng-model"
  },
  {
    "label": "Context length",
    "desc": "Caps prompt+reply, KV cost",
    "cat": "Model",
    "tab": "tab-engines",
    "id": "eng-ctxlen"
  },
  {
    "label": "GPU layers",
    "desc": "GPU offload (999=all)",
    "cat": "Model",
    "tab": "tab-engines",
    "id": "eng-gpu"
  },
  {
    "label": "Engine name",
    "desc": "Profile lookup key",
    "cat": "Model",
    "tab": "tab-engines",
    "id": "eng-name"
  },
  {
    "label": "Base URL",
    "desc": "Provider endpoint",
    "cat": "Model",
    "tab": "tab-engines",
    "id": "eng-url"
  },
  {
    "label": "API key",
    "desc": "Auth for endpoint",
    "cat": "Model",
    "tab": "tab-engines",
    "id": "eng-apikey"
  },
  {
    "label": "Temperature",
    "desc": "Logit scale (0=deterministic)",
    "cat": "Sampling",
    "tab": "tab-engines",
    "id": "s-temp"
  },
  {
    "label": "Top-p",
    "desc": "Nucleus mass cut",
    "cat": "Sampling",
    "tab": "tab-engines",
    "id": "s-topp"
  },
  {
    "label": "Top-k",
    "desc": "Keep K most likely",
    "cat": "Sampling",
    "tab": "tab-engines",
    "id": "s-topk"
  },
  {
    "label": "Min-p",
    "desc": "Drop below fraction*top",
    "cat": "Sampling",
    "tab": "tab-engines",
    "id": "s-minp"
  },
  {
    "label": "Repeat penalty",
    "desc": "Penalise present tokens",
    "cat": "Sampling",
    "tab": "tab-engines",
    "id": "s-rep"
  },
  {
    "label": "Presence penalty",
    "desc": "Once-per-token",
    "cat": "Sampling",
    "tab": "tab-engines",
    "id": "s-pres"
  },
  {
    "label": "Frequency penalty",
    "desc": "Per-repeat penalty",
    "cat": "Sampling",
    "tab": "tab-engines",
    "id": "s-freq"
  },
  {
    "label": "Seed",
    "desc": "Reproducible output",
    "cat": "Sampling",
    "tab": "tab-engines",
    "id": "s-seed"
  },
  {
    "label": "Flash attention",
    "desc": "Faster long context",
    "cat": "Performance",
    "tab": "tab-engines",
    "id": "eng-flash"
  },
  {
    "label": "Parallel slots",
    "desc": "Concurrent decode",
    "cat": "Performance",
    "tab": "tab-engines",
    "id": "eng-parallel"
  },
  {
    "label": "mlock",
    "desc": "Pin RAM, no swap",
    "cat": "Performance",
    "tab": "tab-engines",
    "id": "eng-mlock"
  },
  {
    "label": "no-mmap",
    "desc": "Full RAM vs mmap",
    "cat": "Performance",
    "tab": "tab-engines",
    "id": "eng-nommap"
  },
  {
    "label": "KV cache K",
    "desc": "KV quantization (K)",
    "cat": "Performance",
    "tab": "tab-engines",
    "id": "eng-ctk"
  },
  {
    "label": "KV cache V",
    "desc": "KV quantization (V)",
    "cat": "Performance",
    "tab": "tab-engines",
    "id": "eng-ctv"
  },
  {
    "label": "Batch size",
    "desc": "Tokens per step",
    "cat": "Performance",
    "tab": "tab-engines",
    "id": "eng-batch"
  },
  {
    "label": "Micro-batch",
    "desc": "Physical micro-batch",
    "cat": "Performance",
    "tab": "tab-engines",
    "id": "eng-ubatch"
  },
  {
    "label": "Threads",
    "desc": "CPU parallelism",
    "cat": "Performance",
    "tab": "tab-engines",
    "id": "eng-threads"
  },
  {
    "label": "Alias",
    "desc": "Model id for clients",
    "cat": "Model",
    "tab": "tab-engines",
    "id": "eng-alias"
  },
  {
    "label": "Providers",
    "desc": "OpenAI-compatible endpoints & keys",
    "cat": "Providers",
    "tab": "tab-providers",
    "id": "tab-providers"
  },
  {
    "label": "The encoder",
    "desc": "The small model that does the retrieval and the memory",
    "cat": "Splinter",
    "tab": "tab-global",
    "id": "tab-global"
  },
  {
    "label": "Memory & RAG baseline",
    "desc": "The curation every conversation starts from",
    "cat": "Splinter",
    "tab": "tab-global",
    "id": "tab-global"
  },
  {
    "label": "Cascade defaults",
    "desc": "Judge, repair, gate, human gate, escalation",
    "cat": "Cascade",
    "tab": "tab-global",
    "id": "tab-global"
  },
  {
    "label": "MCP controls",
    "desc": "The memory over MCP: the endpoint and its tools",
    "cat": "Cascade",
    "tab": "tab-global",
    "id": "tab-global"
  },
  {
    "label": "Chat mode",
    "desc": "Splinter vs Agent (dsh)",
    "cat": "Chat",
    "tab": "tab-home",
    "id": "chat-title"
  },
  {
    "label": "Cascade knobs",
    "desc": "Judge, repair, gate, temperature, etc.",
    "cat": "Cascade",
    "tab": None,
    "id": None,
    "href": "/cascade"
  },
  {
    "label": "Cascade runs",
    "desc": "View past run bundles",
    "cat": "Cascade",
    "tab": None,
    "id": None,
    "href": "/runs"
  },
  {
    "label": "Library path",
    "desc": "Local GGUF folder",
    "cat": "Setup",
    "tab": "tab-library",
    "id": "library-path"
  }
]

def render_server_page() -> str:
    """Model-management console: server + settings | loaded-model chat | hub.

    Discovery is live against the Hugging Face hub API — no model catalog is
    hardcoded here."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><link rel="icon" href="/favicon.svg" type="image/svg+xml"><meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"><meta http-equiv="Pragma" content="no-cache"><meta http-equiv="Expires" content="0"><title>Studio server &amp; models</title>
<style>{_get_server_css()}</style><style>input[type=number]::-webkit-inner-spin-button,input[type=number]::-webkit-outer-spin-button{{-webkit-appearance:none;margin:0}}input[type=number]{{-moz-appearance:textfield;appearance:textfield}}</style></head><body>
<h1>HiveBench Studio</h1>
<!-- the load bar: the bottom half of the screen while a model loads or unloads -->
<div id="loadbar" class="loadbar" style="display:none">
  <div class="loadbar-head">
    <div class="loadbar-title" id="loadbar-title">Loading&hellip;</div>
    <button class="loadbar-close" onclick="hideLoadBar()" title="Hide this bar (the work keeps going)">&#10005;</button>
  </div>
  <div class="loadbar-strip"><i></i></div>
  <div class="loadbar-detail" id="loadbar-detail"></div>
</div>
{render_nav("server")}

<div class="grid">

<!-- Command palette (Ctrl+K) -->
<div id="cmd-palette" style="display:none; position:fixed; inset:0; z-index:9999; background:rgba(0,0,0,.5); align-items:flex-start; justify-content:center; padding-top:15vh">
  <div style="background:#000; border:2px solid #FFB703; border-radius:10px; width:min(600px,90vw); max-height:60vh; display:flex; flex-direction:column; overflow:hidden">
    <input id="cmd-palette-input" type="text" placeholder="Search commands…" style="padding:.8rem 1rem; font-size:1rem; background:#FFB703; color:#000; border:none; border-bottom:1px solid #FFDD00; outline:none">
    <div id="cmd-palette-list" style="overflow-y:auto; flex:1; padding:.5rem"></div>
  </div>
</div>

<!-- ==================== LEFT: tabs ==================== -->
<div class="col" id="left-col">
<div class="tabs" id="main-tabs">
<button class="tab active" data-tab="tab-home" title="Start here: load a model, open the cascade, or run setup.">Home</button>
<button class="tab" data-tab="tab-settings" title="Models, providers, engines, and local library settings.">Settings</button>
<button class="tab" data-tab="tab-global" title="Global quality defaults inherited by conversations and runs.">Global</button>
<button class="tab" data-tab="tab-cascade" onclick="location.href='/cascade'" title="Verification, repair, escalation, and sensory upload.">Cascade</button>
<button id="adv-toggle" class="tab" onclick="toggleAdvanced()" style="margin-left:auto" title="Agent, engines, library, providers, Hub, inspector, and training.">More tools ▾</button>
</div>
<div class="tabs" id="adv-tabs" style="display:none">
<button class="tab" data-tab="tab-setup" style="display:none">Docker setup</button>
<button class="tab" data-tab="tab-agent">Agent</button>
<button class="tab" data-tab="tab-engines">Engines</button>
<button class="tab" data-tab="tab-library">Local Library</button>
<button class="tab" data-tab="tab-providers">Providers</button>
<button class="tab" data-tab="tab-hub">Hub</button>
<button class="tab" data-tab="tab-inspect">Inspector</button>
<button class="tab" data-tab="tab-training">Training</button>
</div>

<div id="tab-home" class="tabpane">
<section>
<h2 style="margin-top:0">Welcome to HiveBench Studio</h2>
<p class="note">Your local-first AI harness. Pick a starting point:</p>
<div class="cards" style="margin:1rem 0">
  <a href="/cascade" style="text-decoration:none"><div class="card"><span>Run a cascade</span><b style="font-size:1.1rem">Hive Cascade →</b><span class="note">generate → detect → repair → escalate</span></div></a>
  <a href="/runs" style="text-decoration:none"><div class="card"><span>Browse results</span><b style="font-size:1.1rem">Runs →</b><span class="note">view past run bundles</span></div></a>
  <button onclick="document.querySelector('[data-tab=\\'tab-settings\\']')?.click()" style="all:unset;cursor:pointer"><div class="card"><span>Configure</span><b style="font-size:1.1rem">Settings →</b><span class="note">models, providers, engines</span></div></button>
</div>
<p class="note">Advanced tools (agent presets, engines, library, inspector, training, hub) are under <b>More tools ▾</b> above. Click any tab to open it.</p>

<section style="margin-top:1.5rem">
<h2 style="margin-top:0">Setup <span class="note">let the Studio pick</span></h2>
<p class="note">Tell the Studio what you have. It ranks your local models by what actually fits
in VRAM and sets the context for you &mdash; no knob-fiddling required.</p>
<div class="row" style="gap:.5rem;align-items:center;flex-wrap:wrap">
  <button onclick="wizardDetect(this)">Detect hardware &amp; recommend</button>
  <span id="wizard-hw" class="note"></span>
</div>
<div id="wizard-backends" style="margin:.6rem 0"></div>
<div id="wizard-models" style="margin-top:.6rem"></div>
</section>

<section id="dev-research" style="margin-top:1.5rem; display:none">
<h2 style="margin-top:0">Deep research <span class="note">dev mode · queued for the background worker</span></h2>
<p class="note">Ask a question and it is queued for the deep-research worker. The worker is
master-only: the report lands in <code>RESEARCH/&lt;slug&gt;.md</code> and is summarised on its next
wake. Nothing runs in the foreground.</p>
<div class="row">
  <input id="researchq" placeholder="ask a research question…" style="flex:1;min-width:180px" autocomplete="off" title="How: researchAdd queues to RESEARCH-QUEUE.md. Does: Deep research task (Auditor only). Changing: Adds an entry - it does not run now." onkeydown="if (event.key === &quot;Enter&quot;) researchAdd(this)">
  <button id="researchsubmit" onclick="researchAdd(this)">Queue</button>
  <span id="researchcount" class="meta" title="Queues a deep-research question for the worker. Reports land in RESEARCH/&lt;slug&gt;.md."></span>
</div>
</section>

<section style="margin-top:1.5rem">
<h2 style="margin-top:0">Backend</h2>
<p class="note">Choose which AI backend the Studio and Cascade use. Only one is active at a time.</p>
<div class="row" style="gap:.6rem;flex-wrap:wrap;align-items:center;margin:.6rem 0">
  <label class="inline" style="cursor:pointer"><input type="radio" name="backend" value="strata" onchange="setBackend(this.value)"> <b>Strata</b> (local 176B, port 8081)</label>
  <label class="inline" style="cursor:pointer"><input type="radio" name="backend" value="llama" onchange="setBackend(this.value)"> <b>Llama.cpp</b> (local GGUF, Launch panel)</label>
  <label class="inline" style="cursor:pointer"><input type="radio" name="backend" value="api" onchange="setBackend(this.value)"> <b>API</b> (remote endpoint, Providers tab)</label>
</div>
<div id="backend-status" class="note" style="margin:.4rem 0"></div>
</section>
</section>
</div>

<div id="tab-setup" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Linux / Docker Setup <span class="note">(deprecated)</span></h2>
<p class="note">This setup was for running very large models (DeepSeek v4, 27B+) via a Linux Docker container with VHDX storage. It is no longer needed for most users — the Studio runs models directly via llama.cpp or strata.</p>
<p class="note">If you need it, the backend endpoints (<code>/v1/setup/*</code>) are still available. The UI has been simplified; see the README for large-model setup.</p>
</section>
</div>

<div id="tab-agent" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Agent profiles <span class="note">(presets)</span></h2>
<div class="row"><button onclick="loadAgentPresets()">Refresh</button><button onclick="createAgentPreset()">New</button></div>
<div id="agent-presets-list" class="liblist">loading…</div>
<div id="agent-preset-detail" style="display:none"><h3 id="agent-preset-name"></h3><pre id="agent-preset-content" style="max-height:300px;overflow:auto"></pre><div class="row"><button onclick="saveAgentPreset()">Save</button><button onclick="closeAgentPreset()">Close</button><span id="agent-preset-msg" class="note"></span></div></div>
<div class="note">Create your own, customise ones you have — like DSH <code>apps/cli/config/agent-presets</code> (<code>preset.yml</code> + <code>agent.cordis.yml</code>). Duplicate any system preset to <code>~/.dsh/.agent-presets/&lt;id&gt;/</code> and edit.</div>
</section>
</div>

<div id="tab-engines" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Engine Profiles</h2>
<form id="engine-profile-form" class="engine-grid" onsubmit="event.preventDefault(); saveEngineProfile();">
  <div class="row" style="gap:.6rem; flex-wrap:wrap; align-items:center">
    <label class="inline">Profile <select id="eng-select" style="min-width:180px" onchange="engineSelected()"><option value="">— select profile —</option></select></label>
    <label class="inline" title="Use this profile as the default for new conversations.">Set as default <input id="eng-default" type="checkbox" onchange="engDirty=true"></label>
    <button type="button" onclick="engineAdd()">+ New profile</button>
  </div>

  <!-- Model -->
  <div class="engine-group">
    <h3 title="How: GET /v1/models/local lists GGUFs from models_dir; resolve checks file else needs hf. Does: Sets weights for -m (size_gb for fit). Changing: Pick different GGUF changes VRAM need — Save+Load to restart.">Model </h3>
    <div class="grid-2col">
      <label title="How: GET /v1/models/local scans models_dir. Does: Selects GGUF for -m (advisory, on Start). Changing: Different file → different size_gb/KV fit — Save+Load required.">Model  <select id="eng-model" onchange="engDirty=true; updateFit(); updateEngModelDisplay();"><option value="">— choose local model —</option></select></label>
      <span class="note">Selected: <b id="eng-model-display">—</b> <span id="eng-model-note" class="note" style="margin-left:.5rem"></span></span>
    </div>
  </div>

  <!-- Endpoint -->
  <div class="engine-group">
    <h3>Endpoint</h3>
    <div class="grid-2col">
      <label title="How: Required EngineProfile.name, validated non-empty, key for registry.resolve. Does: Lookup for provider/engine param. Changing: Rename breaks saved refs.">name  <input id="eng-name" size="16" placeholder="local-bonsai" oninput="engDirty=true"></label>
      <label title="How: UI alias not in profile. Does: Cosmetic picker label. Changing: No backend effect.">displayName  <input id="eng-displayName" placeholder="local-bonsai" oninput="engDirty=true"></label>
      <label title="How: Select a provider (from Providers tab) to use its endpoint and key. Changing: Switches the backend this profile talks to.">Provider  <select id="eng-provider" onchange="engProviderChanged()" style="min-width:160px"><option value="">— none (use base URL below) —</option></select></label>
      <label title="Resolved from the selected provider. Read-only.">baseURL  <input id="eng-url" placeholder="http://localhost:1234/v1" readonly style="opacity:.7"></label>
      <label title="Resolved from the selected provider. Read-only.">apiKey  <input id="eng-apikey" type="password" placeholder="(none)" readonly style="opacity:.7"></label>
    </div>
  </div>

  <!-- Context -->
  <div class="engine-group">
    <h3>Context</h3>
    <div class="grid-2col">
      <label title="How: ctx_size → -c (8192 default), fit needs=size+ctx*0.25GB/1k. Does: Caps prompt+completion, KV cost. Changing: 8k→32k +6GB spill→NVMe/oom; >16k Auto forces q8_0.">contextLength  <input id="eng-ctxlen" type="number" value="8192" min="256" step="256" oninput="engDirty=true; updateFit();"></label>
      <div id="eng-fit" class="fit-panel" style="display:block">
        <div class="fit-grid">
          <span id="fit-needs" class="fit-needs">needs —</span>
          <span id="fit-has" class="fit-has">you have —</span>
          <span id="fit-pct" class="fit-pct"></span>
        </div>
        <div class="fit-bar"><div id="fit-fill" class="fit-fill" style="width:0%"></div></div>
        <div id="fit-plan" class="note fit-plan"></div>
        <div class="row fit-controls">
          <label class="inline">context <input id="fit-ctx" type="range" min="2048" max="131072" step="1024" value="8192"> <span id="fit-ctx-label" title="How: fit needs vs hardware via /v1/server/status. Does: Visual guide, disables Load if needs>has. Changing: Slider only visual — Auto/Save commits.">8k</span> → <b id="fit-needs-val">—</b> </label>
          <button type="button" id="eng-load-fit" onclick="engineLoadFromFit()" disabled>Load</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Sampling 4x2 -->
  <div class="engine-group">
    <h3>Sampling</h3>
    <div class="grid-2col sampling-grid">
      <label title="How: payload.temperature 0..2 via sampling, no range check. Does: Logit scale. Changing: 0.2→1.2 more creative/hallucination; 0 deterministic.">temp  <input id="s-temp" type="number" step="0.05" min="0" max="2" oninput="engDirty=true"></label>
      <label title="How: payload.top_p 0..1. Does: Nucleus mass cut. Changing: 1→0.7 tighter, less tail.">top_p  <input id="s-topp" type="number" step="0.05" min="0" max="1" oninput="engDirty=true"></label>
      <label title="How: payload.top_k int. Does: Keep K most likely. Changing: 0→40 prunes weird tokens.">top_k  <input id="s-topk" type="number" oninput="engDirty=true"></label>
      <label title="How: payload.min_p 0..1. Does: Drop <fraction*top. Changing: 0.05 prunes low prob without top_p.">min_p  <input id="s-minp" type="number" step="0.01" oninput="engDirty=true"></label>
      <label title="How: repeat_penalty via llama.cpp. Does: Penalise present. Changing: 1.0→1.2 less looping.">repeat_penalty  <input id="s-rep" type="number" step="0.05" oninput="engDirty=true"></label>
      <label title="How: presence_penalty OpenAI. Does: Once-per-token. Changing: 0→0.6 broader explore.">presence_penalty  <input id="s-pres" type="number" step="0.1" oninput="engDirty=true"></label>
      <label title="How: frequency_penalty growing. Does: Per-repeat. Changing: 0→0.5 curbs loops.">frequency_penalty  <input id="s-freq" type="number" step="0.1" oninput="engDirty=true"></label>
      <label title="How: payload.seed int, blank random. Does: Reproducible. Changing: Fixed → same output.">seed  <input id="s-seed" type="number" oninput="engDirty=true"></label>
    </div>
  </div>

  <!-- Load -->
  <div class="engine-group">
    <h3>Load</h3>
    <div class="grid-2col load-grid">
      <label title="How: -t <n> or omit auto. Does: CPU parallelism. Changing: 4→16 up tok/s, over→down.">threads  <input id="eng-threads" type="number" placeholder="auto" oninput="engDirty=true"></label>
      <label title="How: -ngl <n> 999=all, clamped to est_layers. Does: GPU VRAM linear. Changing: 999→28 on 8GB fits but slower; 999 oom→exit.">gpu_layers  <input id="eng-gpu" type="number" value="999" oninput="engDirty=true"></label>
      <label title="How: auto_fit → ngl from free VRAM + KV (GET /v1/server/status). Does: Offloads overflow layers to CPU/GTT instead of OOM. Changing: on overrides gpu_layers.">auto-fit <input id="eng-autofit" type="checkbox" onchange="engDirty=true"></label>
      <label title="How: visible_devices → HIP/CUDA_VISIBLE_DEVICES for this server. Does: Pins the model to specific card(s). Changing: e.g. 1 = second HIP device (PCI order). 'auto' takes the card that is NOT the display, so the desktop keeps its own VRAM.">GPU(s) <select id="eng-devices" onfocus="fillGpuDevices()" onchange="engDirty=true"><option value="">auto (the spare card)</option></select></label>
      <label title="How: -fa on if set, auto on when ctx>=8192. Does: Faster long ctx. Changing: off→on +10-30% at 32k, needs GPU.">flash_attn  <select id="eng-flash" onchange="engDirty=true"><option value="">off</option><option value="on">on</option><option value="auto">auto</option></select></label>
      <label title="How: -np <n> parallel slots. Does: Concurrent decode, ctx/slots. Changing: 1→4 throughput up, per-slot ctx down.">parallel  <input id="eng-parallel" type="number" placeholder="1" oninput="engDirty=true"></label>
      <label title="How: -b <n> 512 default. Does: Tokens/step RAM. Changing: 512→2048 prompt faster, more VRAM/spill.">batch  <input id="eng-batch" type="number" placeholder="512" oninput="engDirty=true"></label>
      <label title="How: -ub <n> 512. Does: Micro-batch physically. Changing: 512→128 lower peak but more steps; tune with batch.">ubatch  <input id="eng-ubatch" type="number" placeholder="512" oninput="engDirty=true"></label>
      <label title="How: --cache-type-k q8_0/q4_0 else f16, Auto q8_0 if ctx>16384. Does: KV quant saves 50%/75%. Changing: f16→q8 halves KV.">cache K  <select id="eng-ctk" onchange="engDirty=true"><option value="">f16</option><option>q8_0</option><option>q4_0</option></select></label>
      <label title="How: --cache-type-v same. Does: V cache quant. Changing: Same as K.">cache V  <select id="eng-ctv" onchange="engDirty=true"><option value="">f16</option><option>q8_0</option><option>q4_0</option></select></label>
    </div>
  </div>

  <!-- Advanced -->
  <div class="engine-group">
    <h3>Advanced</h3>
    <div class="grid-2col advanced-grid">
      <label title="How: mirostat 0..2 payload. Does: Perplexity target. Changing: 0 off →2 adaptive.">mirostat  <input id="s-miro" type="number" min="0" max="2" oninput="engDirty=true"></label>
      <label title="How: mirostat_tau float. Does: Entropy target. Changing: Higher more surprise.">mirostat_tau  <input id="s-tau" type="number" step="0.1" oninput="engDirty=true"></label>
      <label title="How: mirostat_eta float. Does: Adapt rate. Changing: 0.1 fast vs 0.01 slow.">mirostat_eta  <input id="s-eta" type="number" step="0.01" oninput="engDirty=true"></label>
      <label title="How: stop array payload. Does: Early stop. Changing: Add </s> to truncate.">stop  <input id="s-stop" placeholder="a,b" oninput="engDirty=true"></label>
      <label title="How: --alias <id>. Does: Model id vs path. Changing: Clients see alias.">alias  <input id="eng-alias" placeholder="model id" oninput="engDirty=true"></label>
      <label title="How: kind in llama_cpp,lmstudio,vllm,ollama,hosted validated. Does: Routing hint. Changing: Wrong→422; hosted skips launch.">kind  <select id="eng-kind" onchange="engDirty=true"><option>llama_cpp</option><option>lmstudio</option><option>vllm</option><option>ollama</option><option>hosted</option></select></label>
    </div>
  </div>

  <!-- A/B compare — grid cell with two profile selects, Bench button, winner badge -->
  <div class="engine-group" id="ab-compare">
    <h3>A/B compare</h3>
    <div class="grid-2col">
      <label title="How: POST /v1/engines/ab/bench profile_a,basePort reuses or loads. Does: Bench A vs B. Changing: Pick different to compare tok/s.">A profile  <select id="ab-select-a"></select></label>
      <label title="How: Second on basePort+1. Does: Bench vs A. Changing: Determines B load.">B profile  <select id="ab-select-b"></select></label>
      <label title="How: basePort 1024-65534, B=basePort+1 via _ab_ensure_started. Does: Ports for AB. Changing: 1234→5678 avoids clash.">basePort  <input id="ab-baseport" type="number" value="1234" min="1024" max="65534"></label>
      <div class="row" style="gap:.5rem; align-items:center">
        <button type="button" id="ab-bench-btn" onclick="benchAb()">Bench</button>
        <span id="ab-winner" class="winner-badge">—</span>
        <span id="ab-badge" style="display:none"></span>
      </div>
      <span id="ab-result-a" class="note"></span>
      <span id="ab-result-b" class="note"></span>
      <!-- aliases for alternative test ids -->
      <select id="ab-profile-a" style="display:none"></select>
      <select id="ab-profile-b" style="display:none"></select>
    </div>
  </div>

  <div class="row" style="margin-top:.6rem; gap:.5rem; align-items:center; flex-wrap:wrap"><span id="eng-msg" class="note"></span><button type="submit" id="eng-save">Save</button> <button type="button" id="eng-auto" onclick="engineAuto()" title="How: GET /v1/engines/preset → _auto_preset size_gb+hardware table + est_layers, writes load_options. Does: Sets gpu/context/flash/cache. Changing: Click to fit model to VRAM (e.g. 32b 8GB→12/4k, 4b 8GB→999/32k).">Auto</button> <span id="eng-auto-msg" class="note" style="margin-left:.5rem"></span> <button type="button" onclick="exportEngineProfiles()">Export</button><button type="button" onclick="document.getElementById('import-engine-profiles').click()">Import</button><input type="file" id="import-engine-profiles" accept=".json" style="display:none" onchange="importEngineProfiles(this)"><span id="import-export-msg" class="note"></span></div>
  <pre id="eng-loadopts" style="display:none"></pre>
</form>
</section>
</div>

<div id="tab-library" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Local Library <span class="note">models on disk &mdash; not loaded until you start one</span></h2>
<div class="lib-container">
<div class="tabs" style="margin-bottom:.6rem">
<div class="lib-tab active" data-rtab="lib-system" onclick="switchLibraryTab('lib-system', event)" role="button" tabindex="0">System</div>
</div>
<div id="lib-system" class="lib-tabpane">
<div class="note" style="color:#000">System models on Windows host (<span id="library-path-note"></span>)</div>
<div class="row"><input type="text" id="library-path" placeholder="C:/Users/you/.lmstudio/models" size="38" style="flex:1" title="How: POST /v1/models/local/path writes to harness_state/models_dir.txt, list_local scans there. Does: Sets local library root for GGUF dropdowns. Changing: Point to LM Studio folder to see your models; persists across restarts."> <button onclick="setLibraryPath()">Use this folder</button><span id="import-status" class="note" style="margin-left:.5rem; color:#000"></span></div>
<div class="row"><button onclick="document.getElementById('import-folder-input').click()" title="How: POST /v1/models/local/import (multipart). Does: Copies the chosen folder's .gguf files into the library. Changing: Use when the folder is not reachable as a path (e.g. a download).">Import folder…</button><input type="file" id="import-folder-input" webkitdirectory directory multiple accept=".gguf" style="display:none" onchange="importFolder(this)"><span class="note" style="margin-left:.5rem">copies .gguf files into the library</span></div>
<div class="row" style="margin-top:.6rem"><input type="text" id="library-filter" placeholder="Filter by name…" size="28" style="flex:1" oninput="filterLibrary()" title="How: Client filterLibrary() substring on filenames. Does: Filters list + dropdown options, shows visible/total. Changing: Type qwen to narrow; clear to show all."> <span class="note" id="library-filter-count" style="margin-left:.5rem; color:#000"></span></div>
<div id="local-system" class="liblist">loading…</div>
<div id="local" style="display:none"></div>
</div>
</div>
</section>
</div>

<div id="tab-providers" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Providers <span class="note">(OpenAI-compatible)</span></h2>
<div id="prov-list"></div>




<div class="row">
<button onclick="providerAdd()">+ Add provider</button>
<button onclick="saveProviders()">Save providers</button>
<span id="prov-msg" class="note"></span></div>
<div class="note">Keys echo as *** — leave untouched rows as-is to keep the
stored secret; type a new key to replace it. Saved to
providers.local.json (gitignored).</div>
</section>
</div>

<div id="tab-hub" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Hugging Face hub <span class="note">(live)</span></h2>
<div class="row"><span class="sugwrap" style="width:100%"><input id="q" placeholder="search gguf repos…" style="width:100%" oninput="searchHub()" title="Search the Hub for GGUF repos. Click a result to list its files.">
<div class="sugbox" id="sug-q"></div></span></div>
<div id="hub-cards" style="margin-top:.6rem"></div>
<h3 style="margin:1rem 0 .3rem; color:#FFDD00">Downloads</h3>
<div id="hub-downloads" class="note">no downloads yet</div>
</section>
</div>

<div id="tab-inspect" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Prompt inspector <span class="note">(last turn)</span></h2>
<div id="inspect-summary" class="kv-grid"></div>
<h2 style="font-size:.95rem;margin-top:1rem">Selected chunks (in the context)</h2>
<div id="inspect-selected" class="inspector-list"></div>
<h2 style="font-size:.95rem;margin-top:1rem">Dropped chunks <span class="note" id="inspect-dropped-n"></span></h2>
<div id="inspect-dropped" class="inspector-list"></div>
<h2 style="font-size:.95rem;margin-top:1rem">Assembled context (preview)</h2>
<pre id="inspect-assembled" style="max-height:200px"></pre>
<h2 style="font-size:.95rem;margin-top:1rem">Stage timings</h2>
<pre id="inspect-timings"></pre>
</section>
</div>

{render_global_pane()}

<div id="tab-settings" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Settings</h2>
<p class="note">Search all settings below, or browse by category. Each setting links to where it lives.</p>
<div class="row" style="margin:.6rem 0">
  <input id="settings-search" type="text" placeholder="Search settings…" style="flex:1;min-width:200px" oninput="filterSettings(this.value)">
</div>
<div id="settings-list"></div>
</section>
</div>

{render_training_pane()}

</div>

<!-- ==================== MIDDLE: loaded AI chat ==================== -->
<div class="col mid">
<section class="chatpane">
<div id="sess-tabs" class="sesstabs"></div>
<div class="chat-head">
<h2 id="chat-title">Loaded model</h2>
<div class="chat-controls">
<span class="note" id="chat-mode-summary"></span>
<details class="chat-menu">
<summary title="Session actions: name it, take it out, bring one back.">&#8942;</summary>
<div class="chat-menu-items">
<button type="button" onclick="saveSession()" title="How: saveSession prompts for a title and keeps it in localStorage. Does: names the current session tab. Changing: the name only; the transcript is untouched.">Save session</button>
<button type="button" onclick="exportSession()" title="How: exportSession builds a JSON file from this session and downloads it. Does: a copy you can open later. Changing: nothing in the session changes.">Export session</button>
<button type="button" onclick="document.getElementById('import-session-input').click()" title="How: picks a session JSON file and restores it as a new tab. Does: brings a session back. Changing: the imported copy is added; the original stays.">Import session</button>
<input type="file" id="import-session-input" accept=".json" style="display:none" onchange="importSession(this)">
</div>
</details>
</div>
</div>
<div class="chat-mode-note note">
  <b>Chat</b> = one curated reply (splinter picks the context). <b>Agent (dsh)</b> = a full
  agent loop with bash, files and code, multi-step turns, a durable session log.
</div>
<div id="chatlog" class="chatlog"></div>
<div class="composer">
<div class="chat-pick">
  <div id="pinned-commands" class="pinned-commands" style="display:none"></div>
  <div class="cmd-pick-anchor">
    <button type="button" id="cmd-picker-toggle" class="cmd-picker-toggle" onclick="toggleCmdPicker()" title="Pick the commands you want at hand above the chat bar. Click a command in the list to add or remove it.">Commands <span class="note" id="commands-count"></span> ▾</button>
    <div id="cmd-picker" class="cmd-picker" style="display:none">
      <div id="commands-list" class="note">loading…</div>
    </div>
  </div>
</div>
<div class="composer-box">
<div class="composer-toggles">
  <label class="inline" title="How: the encoder gets the Studio's own API map and reads it with bash. Does: It can report your hardware, models, engines and runs, and act on them. Changing: Off = the encoder is off, so Splinter and Agent cannot run either."><input type="checkbox" id="chat-assistant" onchange="applyEncoderToggle()"> App assistant</label>
  <label class="inline" title="How: POST /v1/splinter/stream → splinter.process_turn curates the context → one curated reply. Does: Fast, no tools. Changing: Needs the app assistant (the encoder drives it)."><input type="checkbox" id="cfg-splinter" onchange="saveChatToggles()"> Splinter</label>
  <label class="inline" title="How: POST /v1/agent/stream → the dsh agent loop with tools and a durable session log. Does: Full agent (bash, files, code, subagents). Changing: Needs the app assistant."><input type="checkbox" id="cfg-agent" onchange="saveChatToggles()"> Agent (dsh)</label>
  <span class="toggle-sep"></span>
  <label title="How: POST /v1/studio/settings sandbox_enabled. Does: lets the Studio run commands in the sandbox (Docker, network off) instead of refusing. Changing: off blocks the sandbox endpoint and the cascade's code_exec detector."><input type="checkbox" id="cfg-sandbox" onchange="saveStudioSetting('sandbox_enabled', this.checked)"> Sandbox</label>
  <label title="How: POST /v1/studio/settings tools_enabled. Does: lets the agent call tools (bash, files, code). Changing: off forces plain chat - the agent endpoint refuses."><input type="checkbox" id="cfg-tools" onchange="saveStudioSetting('tools_enabled', this.checked)"> Tools</label>
  <label title="How: POST /v1/studio/settings auto_compact. Does: compacts a conversation into a checkpoint when it outgrows the window, and offers an engine switch at the checkpoint. Changing: off compacts only when you ask."><input type="checkbox" id="cfg-compact" onchange="saveStudioSetting('auto_compact', this.checked)"> Auto-compact</label>
</div>
<div class="composer-row">
<span class="sugwrap composer-input"><textarea id="chatin" rows="1" placeholder="Talk to the loaded AI…  (/ for commands)" title="How: Enter → chatSubmit routes / → /v1/commands/run else sendChat/sendAgent via Splinter vs Agent. Does: Sends prompt through splinter curate+generate or agent loop. Changing: /command vs message decides path."
       onkeydown="if (event.key === 'Enter' && !event.shiftKey) {{ event.preventDefault(); chatSubmit(); }}" oninput="autoGrow(this)" autocomplete="off"></textarea>
<div class="sugbox" id="sug-chat"></div></span>
<button id="sendbtn" class="iconbtn" onclick="chatSubmit()" aria-label="Send" title="How: Enter sends, Shift+Enter for a new line. Does: the message goes to the model. Changing: while the model is replying, the square next to this stops the turn.">&#9654;</button>
<button id="stopbtn" class="iconbtn" onclick="cancelStream()" style="display:none" aria-label="Stop" title="Stop the model now. The provider connection is closed, so the engine stops generating (llama.cpp and strata both do) and the GPU is free at once. The turn ends where it is; the conversation stays.">&#9632;</button>
</div>
</div>
<div class="chat-hint">Enter sends · Shift+Enter for a new line · <kbd>/</kbd> commands · <kbd>Ctrl</kbd>+<kbd>K</kbd> palette <span id="cfg-note"></span></div>
</div>
<div id="surgery-panel" style="display:none;margin-top:.5rem"></div>
</section>
</div>

<!-- ==================== RIGHT: Server (standalone, no tabs) ==================== -->
<div class="col" id="right-col">
<section>
<h2 style="margin-top:0">Launch</h2>
<div class="row"><button onclick="api('/v1/server/status').then(s => show('status', s))">Refresh</button>
<button onclick="unloadLoaded(this)" title="How: POST /v1/server/unload-loaded -&gt; the local llama.cpp instances are unloaded and the Strata engine frees its VRAM. Does: releases the memory; the engine profiles and providers stay. Changing: Start works again right away. (Stopping a reply is the Stop button in the chat.)">Unload</button></div>
<!-- the engine tabs: what differs per engine is behind its own tab -->
<div class="tabs launch-tabs" id="launch-engine-tabs" style="margin:.5rem 0 .2rem">
<button class="tab active" data-engine="llamacpp" onclick="selectLaunchEngine('llamacpp')" title="Launch a local GGUF with llama.cpp (llama-server).">llama.cpp</button>
<button class="tab" data-engine="strata" onclick="selectLaunchEngine('strata')" title="Launch the Strata engine (the local 176B serve) from a config profile.">Strata</button>
</div>
<!-- shared: the context window and the port, whichever engine serves it -->
<div class="row">
<label class="inline" title="How: the context window in tokens. Does: how much the model keeps. Changing: bigger = more memory (the KV cache grows with it).">ctx <input id="ctx" type="number" value="8192" size="6"></label>
<label class="inline" title="How: the port the server listens on. Does: where the clients reach it. Changing: the next free port if left empty.">port <input id="l-port" type="number" size="5" placeholder="auto"></label>
</div>

<!-- ============ llama.cpp ============ -->
<div class="launch-pane" id="launch-pane-llamacpp">
<div class="row">
<label class="inline" style="flex:1">Model <select id="launch-model-select" style="flex:1;min-width:180px" title="Pick a local GGUF model to launch. Populated from your library (set the path in the Local Library tab)."><option value="">— choose local model —</option></select></label>
</div>
<div class="row">
<label class="inline" title="How: -ngl <n> GPU layers (999 = all). Does: how much of the model sits on the GPU. Changing: lower it when VRAM is tight.">gpu layers <input id="ngl" type="number" value="999" size="4"></label>
<label class="inline" title="How: --auto-fit. Does: computes the GPU layers from the free VRAM plus the KV cache. Changing: on = the Studio sizes it for you."><input id="l-autofit" type="checkbox"> auto-fit</label>
<label class="inline" title="How: HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES. Does: which GPUs this server may use. Changing: 0,1 uses both cards.">devices <input id="l-devices" size="6" placeholder="0,1"></label>
</div>
<div class="row">
<span class="sugwrap" style="flex:1"><input id="hfrepo" placeholder="HF repo (optional, downloads on start)" style="width:100%">
<div class="sugbox" id="sug-hfrepo"></div></span>
<input id="hffile" placeholder="HF file" size="12">
</div>
<details><summary>llama.cpp flags</summary>
<div class="row">
<label class="inline" title="How: -t <n> or omit auto. Does: CPU parallelism. Changing: 4->16 up tok/s, over->down.">threads  <input id="l-threads" type="number" size="3" placeholder="auto (from profile)"></label>
<label class="inline"><input id="l-fa" type="checkbox" title="How: -fa on if set, auto on when ctx>=8192. Does: Faster long ctx. Changing: off->on +10-30% at 32k, needs GPU."> flash-attn </label>
<label class="inline" title="How: -np <n> parallel slots. Does: Concurrent decode, ctx/slots. Changing: 1->4 throughput up, per-slot ctx down.">parallel  <input id="l-parallel" type="number" size="2" placeholder="1"></label>
<label class="inline"><input id="l-mlock" type="checkbox" title="How: mlock -> --mlock. Does: Pins RAM, no swap. Changing: Slower start, stable."> mlock </label>
<label class="inline"><input id="l-nommap" type="checkbox" title="How: no_mmap -> --no-mmap. Does: Full RAM vs mmap. Changing: More RSS, avoids faults."> no-mmap </label>
</div>
<div class="row">
<label class="inline" title="How: cache K -> --cache-type-k f16/q8_0/q4_0, Auto q8_0 >16k. Does: Halves KV. Changing: f16->q8 saves 50%.">kv-K  <select id="l-ctk"><option value="">profile default</option><option>q8_0</option><option>q4_0</option></select></label>
<label class="inline" title="How: --cache-type-v same. Does: V cache quant. Changing: Same as K.">kv-V  <select id="l-ctv"><option value="">profile default</option><option>q8_0</option><option>q4_0</option></select></label>
<label class="inline" title="How: -b <n> 512 default. Does: Tokens/step RAM. Changing: 512->2048 prompt faster, more VRAM/spill.">batch  <input id="l-batch" type="number" size="4" placeholder="profile"></label>
<label class="inline" title="How: -ub <n> 512. Does: Micro-batch physically. Changing: 512->128 lower peak but more steps; tune with batch.">ubatch  <input id="l-ubatch" type="number" size="4" placeholder="profile"></label>
<label class="inline" title="How: --alias <id>. Does: Model id vs path. Changing: Clients see alias.">alias  <input id="l-alias" size="14" placeholder="profile"></label>
<label class="inline" title="How: --api-key. Does: requires a bearer token on every request. Changing: on = only a client with the key gets in.">api key <input id="l-apikey" size="10" placeholder="none"></label>
</div>
</details>
</div>

<!-- ============ Strata ============ -->
<div class="launch-pane" id="launch-pane-strata" style="display:none">
<div class="row">
<label class="inline" style="flex:1">Config <select id="l-strata-config" style="flex:1;min-width:170px" title="The Strata config profile: the model pack, the engine and its defaults. Pick one from the checkout."><option value="">— choose a config —</option></select></label>
</div>
<div class="row">
<label class="inline" style="flex:1" title="How: the Strata checkout. Does: where the serve lives and which config it reads. Changing: point it at another checkout.">Repo <input id="l-strata-repo" style="width:100%" placeholder="the strata checkout"></label>
</div>
<div class="row">
<label class="inline" title="How: --vram-reserve-mib. Does: the VRAM left free for the desktop. Changing: raise it if the display card is used.">vram reserve <input id="l-strata-vram" type="number" value="3072" size="6"></label>
<label class="inline" title="How: --prefill. Does: the prompt chunk size. Changing: bigger = faster long prompts, more memory.">prefill <input id="l-strata-prefill" type="number" value="8192" size="6"></label>
<label class="inline" title="How: --spec. Does: the MTP draft depth. Changing: 0 disables the speculation.">spec <input id="l-strata-spec" type="number" value="4" size="3"></label>
</div>
<div class="row">
<label class="inline" style="flex:1" title="How: extra engine flags, passed straight through. Does: anything the profile does not cover. Changing: e.g. --ncmoe 12 --msc 256.">extra engine args <input id="l-strata-args" style="width:100%" placeholder="--ncmoe 12 --msc 256"></label>
</div>
<div class="row">
<label class="inline" title="How: how long the Studio waits for the engine to answer. Does: the first load takes minutes. Changing: raise it for a cold pack.">timeout s <input id="l-strata-timeout" type="number" value="600" size="5"></label>
</div>
</div>

<div class="row">
<button onclick="startServer(this)" title="How: starts the engine the selected tab names. Does: llama.cpp launches llama-server; Strata starts the serve and waits for it to answer. Changing: the engine tab decides which.">Start</button>
<span class="note" id="launch-note"></span>
</div>
<p class="note">The dials here are the ones this run needs. The Engines tab keeps the reusable profiles (with their provider and sampling).</p>
<div id="instances"></div>
<details style="margin-top:.6rem">
<summary style="cursor:pointer;font-weight:600;padding:.3rem 0;color:#FFDD00">Console logs &amp; processes ▾</summary>
<section id="proc-manager" style="margin-top:1rem">
<h3 style="margin:.6rem 0 .3rem">Process manager <span class="note">CPU/RAM · kill</span></h3>
<div class="row" style="gap:.5rem;align-items:center"><button onclick="refreshProcesses()">Refresh processes</button><span id="proc-msg" class="note"></span><span id="proc-sidecar" class="note" style="margin-left:.4rem"></span></div>
<table style="width:100%;margin-top:.4rem"><thead><tr><th>PID</th><th>Kind</th><th>Name</th><th title="CPU % (process)">CPU%</th><th title="Resident memory MB">RAM MB</th><th>Port</th><th>Status</th><th></th></tr></thead><tbody id="proc-table"><tr><td colspan="8" class="note">loading…</td></tr></tbody></table>
</section>
<details id="dev-raw-status" style="margin-top:1rem; display:none">
<summary style="cursor:pointer;font-weight:700;padding:.45rem 0;color:#FFDD00">Raw server status <span class="note">(dev mode)</span></summary>
<pre id="dev-raw-json" style="max-height:240px">…</pre>
</details>
<h3 style="margin:.6rem 0 .3rem; color:#FFDD00">Engine log (strata)</h3>
<pre id="srvlog" title="strata engine log tail">log: loading…</pre>
<details style="margin-top:.6rem">
<summary style="cursor:pointer;font-weight:600;padding:.3rem 0;color:#FFDD00;opacity:.8">Server status (raw) ▾</summary>
<h3 style="margin:.6rem 0 .3rem; color:#FFDD00">Server status</h3>
<pre id="status">loading…</pre>
</details>
</details>
</section>

</div></div>

<script>
let convId = localStorage.getItem('splinter-console-conv');
if (!convId) {{
  convId = 'console-' + crypto.randomUUID().slice(0, 8);
  localStorage.setItem('splinter-console-conv', convId);
}}
// engDirty declared in unified grid block above

async function api(path, method, body, signal) {{
  const headers = {{'content-type': 'application/json'}};
  const token = localStorage.getItem('splinter-token');
  if (token) headers['x-splinter-token'] = token;
  const opts={{method: method || 'GET', headers}};
  if(body !== undefined && body !== null) opts.body=JSON.stringify(body);
  else if(method === 'POST' && body === undefined) opts.body='{{}}';
  if(signal) opts.signal=signal;
  const r = await fetch(path, opts);
  if (r.status === 401) {{
    const t = prompt('This server requires an access token (HARNESS_TOKEN):');
    if (t !== null) {{ localStorage.setItem('splinter-token', t); }}
    throw new Error('unauthorized — token saved, retry');
  }}
  const t = await r.text();
  if (!r.ok) throw new Error(r.status + ': ' + t.slice(0, 400));
  if (!t) return t;
  const first = t.trimStart()[0];
  return (first === '{{' || first === '[') ? JSON.parse(t) : t;
}}
window.onerror=(msg, src, line, col, err)=>{{ console.error('ONERROR', msg, src+':'+line+':'+col, err); alert('JS ERROR: '+msg+'\\n'+src+':'+line); }};
window.onunhandledrejection=(e)=>{{ console.error('UNHANDLED REJECTION', e.reason); alert('UNHANDLED: '+String(e.reason).slice(0,300)); }};
function show(id, obj) {{ 
  try{{
    const el=document.getElementById(id);
    if(!el){{ console.warn('show missing', id); return; }}
    el.textContent=typeof obj === 'string' ? obj : JSON.stringify(obj, null, 1);
  }} catch(err){{ console.error('show failed', id, err); }}
}}
function val(id) {{ const el=document.getElementById(id); if(!el){{ console.warn('val missing', id); return ''; }} return el.value.trim(); }}
function num(id) {{ const v = val(id); return v === '' ? null : +v; }}
function fmtCap(n) {{ if(n==null||n==='-') return '-'; n=Number(n); if(n>=1000000) return (n/1000000).toFixed(1)+'M'; if(n>=1000) return (n/1000).toFixed(1)+'K'; return String(n); }}

function toggleAdvanced() {{
  const el = document.getElementById('adv-tabs');
  const btn = document.getElementById('adv-toggle');
  if (!el) return;
  const show = el.style.display === 'none' || el.style.display === '';
  el.style.display = show ? 'flex' : 'none';
  if (btn) btn.textContent = show ? 'Fewer tools ▴' : 'More tools ▾';
}}

/* --------------------------- setup wizard --------------------------- */
async function wizardDetect(btn) {{
  const hwEl = document.getElementById('wizard-hw');
  const listEl = document.getElementById('wizard-models');
  if (btn) btn.disabled = true;
  if (hwEl) hwEl.textContent = 'probing…';
  try {{
    const [st, strata] = await Promise.all([
      api('/v1/server/status'),
      api('/v1/studio/strata').catch(() => ({{alive: false, model: null, base_url: ''}})),
    ]);
    const hw = st.hardware || {{}};
    const has = Number(hw.available_gb) || 0;
    const ram = Number(hw.available_ram_gb) || 0;
    if (hwEl) hwEl.textContent = `${{(hw.vram_gb||0).toFixed(1)}} GB VRAM (free ${{has.toFixed(1)}}), ${{(hw.available_ram_gb||ram).toFixed(1)}} GB RAM free, ${{hw.cpu_count||'?'}} CPU`;
    // backend 1: strata (the local 176B serve)
    const backends = document.getElementById('wizard-backends');
    if (backends) {{
      backends.innerHTML = strata.alive
        ? `<div style="border:1.5px solid #000; border-radius:8px; padding:.6rem .8rem; margin:.4rem 0; background:#FFB703; color:#000; display:flex; justify-content:space-between; gap:.6rem; align-items:center; flex-wrap:wrap">
             <span><b style="color:#000">Strata</b> <span class="note" style="color:#333">${{strata.model || 'local 176B'}} · ${{strata.base_url}}</span></span>
             <button type="button" onclick="wizardUseStrata()">Use this backend</button>
           </div>`
        : `<div style="border:1.5px solid #555; border-radius:8px; padding:.6rem .8rem; margin:.4rem 0; color:#333; display:flex; justify-content:space-between; gap:.6rem; align-items:center; flex-wrap:wrap">
             <span><b>Strata</b> <span class="note">not running on ${{strata.base_url || '127.0.0.1:8081'}}</span></span>
             <span class="note">start the 176B serve to use it</span>
           </div>`;
    }}
    const KV = 0.25;               // GB of KV per 1k context tokens
    const models = (st.local_models || []).filter(m => (m.size_gb || 0) > 0.2);
    const ranked = models.map(m => {{
      const size = Number(m.size_gb) || 0;
      // the largest context that still fits: 8k is the safe default
      const ctx = 8192;
      const need = size + (ctx / 1024) * KV;
      return {{ file: m.file, name: (m.file || '').split('/').pop(), size: size, need: need, fits: need <= has }};
    }}).sort((a, b) => (b.fits - a.fits) || (b.size - a.size));
    const fitting = ranked.filter(m => m.fits).slice(0, 6);
    const show = fitting.length ? fitting : ranked.slice(0, 4);
    if (!listEl) return;
    if (!show.length) {{
      listEl.innerHTML = '<p class="note">No local models found. Set your library path in '
        + '<b>More tools &rsaquo; Local Library</b>, or download one from the Hub tab.</p>';
      return;
    }}
    listEl.innerHTML = '<p class="note">Click a model to create its engine profile and start it.</p>'
      + show.map(m => `<div style="display:flex;gap:.6rem;align-items:center;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid rgba(255,221,0,.12);cursor:pointer" onclick="wizardApply('${{m.file}}')">
          <span><b>${{m.name}}</b> <span class="note">${{m.size.toFixed(1)}} GB model · ${{m.need.toFixed(1)}} GB needed at 8k</span></span>
          <span class="note">${{m.fits ? 'fits ✓' : 'too large for VRAM'}}</span></div>`).join('');
  }} catch (e) {{
    if (hwEl) hwEl.textContent = 'probe failed: ' + e;
  }} finally {{
    if (btn) btn.disabled = false;
  }}
}}

async function wizardUseStrata() {{
  const hwEl = document.getElementById('wizard-hw');
  try {{
    const strata = await api('/v1/studio/strata');
    if (!strata.alive) throw new Error('the strata serve is not reachable');
    // register it as a provider and make it the default
    const cur = await api('/v1/provider/config');
    const providers = (cur.providers || []).filter(p => p.name !== 'strata');
    // a provider's base_url is the server ROOT: the clients append /v1
    let root = (strata.base_url || '').replace(/\\/+$/, '');
    if (root.endsWith('/v1')) root = root.slice(0, -3);
    providers.push({{ name: 'strata', base_url: root, api_key: '',
                      model: strata.model || '', headers: {{}} }});
    await api('/v1/provider/config', 'POST', {{
      providers: providers, default: 'strata',
      persist: true, expectedRevision: cur.revision,
    }});
    if (hwEl) hwEl.textContent = 'strata is now the default backend (' + strata.model + ')';
    setBackend('strata');
  }} catch (e) {{
    if (hwEl) hwEl.textContent = 'could not use strata: ' + e;
  }}
}}

async function wizardApply(file) {{
  const hwEl = document.getElementById('wizard-hw');
  const listEl = document.getElementById('wizard-models');
  if (hwEl) hwEl.textContent = 'setting up ' + file.split('/').pop() + '…';
  try {{
    const preset = await api('/v1/engines/preset?file=' + encodeURIComponent(file));
    const load = preset.load_options || preset.preset || {{}};
    const name = file.split('/').pop().replace('.gguf', '');
    // create (or update) the engine profile for this model
    const cur = await api('/v1/engines');
    const engines = (cur.engines || []).filter(e => e.name !== name);
    engines.push({{
      name: name, kind: 'llama_cpp', model: file,
      base_url: 'http://127.0.0.1:1234/v1',
      load_options: load, capabilities: ['streaming'], sampling: {{}},
    }});
    await api('/v1/engines', 'POST', {{
      engines: engines, default: cur.default || name,
      persist: true, expectedRevision: cur.revision,
    }});
    await api('/v1/server/start', 'POST', {{
      model: file, ctx_size: load.context || 8192,
      ngl: load.gpu_layers ?? 999, flash_attn: !!load.flash_attn,
      register_provider: true, claim_default: true,
    }});
    if (hwEl) hwEl.textContent = 'started ' + name + ' — talk to it in the chat pane';
    if (listEl) listEl.innerHTML = '';
  }} catch (e) {{
    if (hwEl) hwEl.textContent = 'setup failed: ' + e;
  }}
}}

function setBackend(val) {{
  localStorage.setItem('hive-backend', val);
  const st = document.getElementById('backend-status');
  if (val === 'strata') {{
    // check if strata is alive
    fetch('/v1/server/status').then(r => r.json()).then(s => {{
      if (st) st.textContent = s.running ? 'Strata is running and healthy.' : 'Strata is not running. Start it separately (port 8081).';
    }}).catch(() => {{ if (st) st.textContent = 'Could not reach strata.'; }});
  }} else if (val === 'llama') {{
    if (st) st.textContent = 'Use the Launch panel to start a local model.';
  }} else {{
    if (st) st.textContent = 'Configure endpoints in the Providers tab.';
  }}
}}
// restore backend selection on load
(function() {{
  const v = localStorage.getItem('hive-backend') || 'strata';
  const r = document.querySelector('input[name="backend"][value="' + v + '"]');
  if (r) r.checked = true;
  setBackend(v);
}})();

/* ------------------------------ settings search ----------------------- */
const SETTINGS_INDEX = {json.dumps(SETTINGS_INDEX)};

function buildSettingsList(filter) {{
  const q = (filter || '').toLowerCase().trim();
  const list = document.getElementById('settings-list');
  if (!list) return;
  const cats = {{}};
  for (const s of SETTINGS_INDEX) {{
    if (q && !s.label.toLowerCase().includes(q) && !(s.desc||'').toLowerCase().includes(q) && !s.cat.toLowerCase().includes(q)) continue;
    (cats[s.cat] = cats[s.cat] || []).push(s);
  }}
  const catNames = Object.keys(cats).sort();
  if (!catNames.length) {{ list.innerHTML = '<p class="note">No matching settings.</p>'; return; }}
  let html = '';
  for (const cat of catNames) {{
    html += `<details open style="margin:.6rem 0"><summary style="cursor:pointer;font-weight:700;padding:.4rem 0;color:#FFDD00">${{cat}} <span class="note">(${{cats[cat].length}})</span></summary>`;
    for (const s of cats[cat]) {{
      const go = s.href ? `location.href='${{s.href}}'` : `goSetting('${{s.tab}}','${{s.id||''}}')`;
      html += `<div style="padding:.3rem 0 .3rem 1rem;border-bottom:1px solid rgba(255,221,0,.12);cursor:pointer" onclick="${{go}}">` +
              `<b style="color:#FFDD00">${{s.label}}</b> <span class="note">${{s.desc}}</span></div>`;
    }}
    html += '</details>';
  }}
  list.innerHTML = html;
}}

function goSetting(tab, id) {{
  const btn = document.querySelector('.tab[data-tab="' + tab + '"]');
  if (btn) btn.click();
  if (id) setTimeout(() => {{
    const el = document.getElementById(id);
    if (el) {{ el.scrollIntoView({{behavior:'smooth',block:'center'}}); el.style.outline='2px solid #FFDD00'; setTimeout(()=>el.style.outline='',1500); }}
  }}, 100);
}}

function filterSettings(q) {{ buildSettingsList(q); }}

setTimeout(() => buildSettingsList(''), 500);
/* --------------------------- command palette ---------------------- */
let _cmdPaletteOpen = false;
async function openCmdPalette() {{
  const pal = document.getElementById('cmd-palette');
  const input = document.getElementById('cmd-palette-input');
  const list = document.getElementById('cmd-palette-list');
  if (!pal || !input || !list) return;
  pal.style.display = 'flex';
  input.value = '';
  input.focus();
  _cmdPaletteOpen = true;
  try {{
    const data = await api('/v1/commands');
    const cmds = data.commands || [];
    window._cmdList = cmds;
    renderCmdPalette('');
  }} catch (e) {{ list.textContent = 'Failed to load commands: ' + e; }}
}}
function closeCmdPalette() {{
  const pal = document.getElementById('cmd-palette');
  if (pal) pal.style.display = 'none';
  _cmdPaletteOpen = false;
}}
function renderCmdPalette(q) {{
  const list = document.getElementById('cmd-palette-list');
  if (!list) return;
  q = (q || '').toLowerCase();
  const cmds = (window._cmdList || []).filter(c =>
    !q || c.name.includes(q) || (c.description||'').toLowerCase().includes(q));
  list.innerHTML = cmds.map(c =>
    `<div style="padding:.5rem .8rem; cursor:pointer; border-bottom:1px solid rgba(255,221,0,.12)" onclick="runCmd('${{c.name}}')">
       <b style="color:#FFDD00">/${{c.name}}</b> <span class="note">${{c.description||''}}</span>
       ${{c.input_hint ? `<span class="note" style="float:right">${{c.input_hint}}</span>` : ''}}
     </div>`).join('') || '<p class="note" style="padding:.5rem">No matching commands.</p>';
}}
async function runCmd(name) {{
  closeCmdPalette();
  const input = document.getElementById('chatin');
  if (input) {{ input.value = '/' + name + ' '; input.focus(); }}
}}
document.getElementById('cmd-palette-input')?.addEventListener('input', e => renderCmdPalette(e.target.value));
document.getElementById('cmd-palette')?.addEventListener('click', e => {{ if (e.target.id === 'cmd-palette') closeCmdPalette(); }});
document.addEventListener('keydown', e => {{
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {{ e.preventDefault(); openCmdPalette(); }}
  if (e.key === 'Escape' && _cmdPaletteOpen) closeCmdPalette();
}});

for (const btn of document.querySelectorAll('.tab[data-tab]')) {{
  btn.addEventListener('click', (e) => {{
    try{{
      console.log('outer tab click', btn.dataset.tab);
      for (const b of document.querySelectorAll('.tab[data-tab]')) b.classList.remove('active');
      for (const p of document.querySelectorAll('.tabpane')) p.style.display = 'none';
      btn.classList.add('active');
      const pane=document.getElementById(btn.dataset.tab);
      if(pane) pane.style.display = '';
      // Defer library load to avoid blocking tab switch
      if (btn.dataset.tab === 'tab-library') setTimeout(loadLibrary, 10);
      else if (btn.dataset.tab === 'tab-agent') loadAgentPresets();
      else if (btn.dataset.tab === 'tab-engines') loadEngines();
      else if (btn.dataset.tab === 'tab-providers') loadProviders();
      else if (btn.dataset.tab === 'tab-hub') {{ pollHubDownloads(); searchHub(); }}
    }} catch(err){{ console.error('tab click failed', err); alert('tab error: '+err.message); }}
  }});
}}
let _libCache={{system: null, linux: null, ts: 0}};
function switchLibraryTab(tab, ev) {{
  if(ev) {{ ev.preventDefault(); ev.stopPropagation(); ev.stopImmediatePropagation(); }}
  // Ensure outer Local Library stays open
  const outer=document.getElementById('tab-library');
  if(outer) outer.style.display='';
  // Keep outer tab active
  for(const b of document.querySelectorAll('.tab[data-tab]')) b.classList.toggle('active', b.dataset.tab==='tab-library');
  const prevSel=window._selectedFile;
  for (const b of document.querySelectorAll('[data-rtab^="lib-"]')) b.classList.remove('active');
  for (const p of document.querySelectorAll('.lib-tabpane')) p.style.display = 'none';
  const btn=document.querySelector(`[data-rtab="${{tab}}"]`);
  const pane=document.getElementById(tab);
  if(btn) btn.classList.add('active');
  if(pane) pane.style.display = '';
  const now=Date.now();
  const fresh=now - _libCache.ts < 30000;
  if (tab === 'lib-system') {{
    if(_libCache.system && fresh) renderLibrary('local-system', _libCache.system, 'system');
    else loadLibrarySystem();
  }}
  if (tab === 'lib-linux') {{
    if(_libCache.linux && fresh) renderLibrary('local-linux', _libCache.linux, 'linux');
    else loadLibraryLinux();
  }}
  // Preserve selection across tab switches — restore after render
  setTimeout(()=>{{
    if(prevSel) {{
      window._selectedFile=prevSel;
      for(const id of ['local-system','local-linux']) {{
        const w=document.getElementById(id);
        if(!w) continue;
        for(const r of w.children) r.classList.toggle('selected', r.dataset.file===prevSel);
      }}
    }}
  }}, 10);
  return false;
}}


/* ---------------- typeahead (hub repos + local library) --------------- */
let suggestTimer = null;
function hideSuggestions() {{
  for (const id of ['sug-q', 'sug-drepo', 'sug-hfrepo']) {{
    const el = document.getElementById(id);
    if (el) el.innerHTML = '';
  }}
}}

async function suggestRepos(value, targetId) {{
  if (!value || value.length < 2) {{ hideSuggestions(); return; }}
  let results = [];
  try {{
    const res = await api('/v1/models/hub?q=' + encodeURIComponent(value) + '&limit=8');
    results = res.results || [];
  }} catch (e) {{ return; }}
  const dl = document.getElementById('repo-suggestions');
  dl.innerHTML = '';
  for (const r of results) {{
    const opt = document.createElement('option');
    opt.value = r.repo;
    dl.appendChild(opt);
  }}
  const box = document.getElementById('sug-' + targetId);
  if (!box) return;
  box.innerHTML = '';
  for (const r of results.slice(0, 8)) {{
    const item = document.createElement('div');
    item.className = 'sugitem';
    const name = document.createElement('span');
    name.textContent = r.repo;
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = '↓' + r.downloads;
    item.appendChild(name);
    item.appendChild(meta);
    item.addEventListener('mousedown', ev => {{
      ev.preventDefault();
      document.getElementById(targetId).value = r.repo;
      hideSuggestions();
      if (targetId === 'q') searchHub();
      else hubFiles(r.repo).catch(() => {{}});
    }});
    box.appendChild(item);
  }}
}}

async function hubFiles(repo) {{
  const res = await api('/v1/models/hub/files/' + encodeURIComponent(repo));
  document.getElementById('drepo').value = repo;
  document.getElementById('dfile').value = res.files.length ? res.files[0].file : '';
  const lines = res.files.map(f => `${{f.file}} — ${{f.size_gb}} GB`);
  show('hub', `files in ${{repo}}:\\n${{lines.join('\\n') || '(none)'}}`);
}}

async function suggestLocal() {{
  try {{
    let l;
    if(_libCache.system && (Date.now()-_libCache.ts < 30000)) l={{models: _libCache.system}};
    else l=await api('/v1/models/local');
    const dl = document.getElementById('local-suggestions');
    if (!dl) return;
    dl.innerHTML = '';
    for (const m of l.models) {{
      const opt = document.createElement('option');
      opt.value = m.file;
      opt.label = `${{m.size_gb}} GB`;
      dl.appendChild(opt);
    }}
  }} catch (e) {{ /* best-effort */ }}
}}

for (const id of ['q', 'drepo', 'hfrepo']) {{
  const el = document.getElementById(id);
  if (!el) continue;
  el.addEventListener('input',
    e => {{ clearTimeout(suggestTimer);
           suggestTimer = setTimeout(() => suggestRepos(e.target.value.trim(), id), 350); }});
  el.addEventListener('blur', () => setTimeout(hideSuggestions, 150));
}}
const _m = document.getElementById('model'); if (_m) _m.addEventListener('focus', suggestLocal);

/* --------------------- sessions (OpenCode-style tabs) ----------------- */
const SESS_KEY = 'splinter-console-sessions';
let sessions = {{}};
try {{ sessions = JSON.parse(localStorage.getItem(SESS_KEY) || '{{}}') || {{}}; }}
catch (e) {{ sessions = {{}}; }}
function persistSessions() {{
  localStorage.setItem(SESS_KEY, JSON.stringify(sessions));
}}
function ensureSession(id) {{
  if (!sessions[id]) sessions[id] = {{ title: null, updated: Date.now(), transcript: [] }};
  return sessions[id];
}}
function record(role, text, meta) {{
  const s = ensureSession(convId);
  s.transcript.push({{role: role, text: text, meta: meta || null}});
  if (s.transcript.length > 400) s.transcript.splice(0, s.transcript.length - 400);
  s.updated = Date.now();
  persistSessions();
}}
function restoreTranscript(id) {{
  const log = document.getElementById('chatlog');
  log.innerHTML = '';
  const s = sessions[id];
  if (!s) return;
  for (const m of s.transcript) {{
    if (m.role === 'tool') continue;   // tool steps are agent-run noise on replay
    bubble(m.role, m.text, m.meta);
  }}
}}
function tabLabel(id) {{
  const s = sessions[id];
  return (s && s.title) ? s.title : 'Session ' + id.slice(-5);
}}
function renderTabs() {{
  const wrap = document.getElementById('sess-tabs');
  wrap.innerHTML = '';
  const ids = Object.keys(sessions).sort((a, b) =>
    (sessions[b].updated || 0) - (sessions[a].updated || 0));
  for (const id of ids) {{
    const t = document.createElement('span');
    t.className = 'sesstab' + (id === convId ? ' active' : '')
      + (sessions[id].title ? '' : ' unsaved');
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = tabLabel(id);
    t.appendChild(name);
    const x = document.createElement('span');
    x.className = 'x';
    x.textContent = '\\u00d7';
    x.title = 'Close session';
    x.addEventListener('click', ev => {{ ev.stopPropagation(); closeSession(id); }});
    t.appendChild(x);
    t.addEventListener('click', () => switchSession(id));
    wrap.appendChild(t);
  }}
}}
async function switchSession(id) {{
  if (id === convId) return;
  convId = id;
  localStorage.setItem('splinter-console-conv', convId);
  restoreTranscript(id);
  renderTabs();
  try {{ await api('/v1/splinter/state?conversation_id=' + encodeURIComponent(id)); }} catch (e) {{}}
}}
function closeSession(id) {{
  delete sessions[id];
  persistSessions();
  api('/v1/splinter/reset', 'POST', {{conversation_id: id}}).catch(() => {{}});
  if (id !== convId) {{ renderTabs(); return; }}
  const rest = Object.keys(sessions).sort((a, b) =>
    (sessions[b].updated || 0) - (sessions[a].updated || 0));
  if (rest.length) switchSession(rest[0]);
  else newConversation();
}}
function saveSession() {{
  const cur = ensureSession(convId);
  const name = prompt('Name this session:', cur.title || '');
  if (name === null) return;
  cur.title = name.trim() || ('Session ' + new Date().toLocaleString());
  cur.updated = Date.now();
  persistSessions();
  renderTabs();
}}
/* The session file is the tab itself: id, title and the transcript.  Export
   writes it out as JSON, import brings it back as a new tab. */
function exportSession() {{
  const cur = ensureSession(convId);
  const payload = {{
    kind: 'hivebench-studio-session', version: 1,
    exported: new Date().toISOString(),
    session: {{ id: convId, title: cur.title, updated: cur.updated,
                transcript: cur.transcript }},
  }};
  const blob = new Blob([JSON.stringify(payload, null, 1)],
                        {{type: 'application/json'}});
  const slug = (cur.title || 'session').replace(/[^A-Za-z0-9_-]+/g, '-')
                .replace(/^-+|-+$/g, '') || 'session';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = slug + '.json';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}}
function importSession(input) {{
  const file = input.files && input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {{
    let data = null;
    try {{ data = JSON.parse(reader.result); }} catch (e) {{ /* not JSON */ }}
    const s = (data && data.session) || data;
    if (!s || !Array.isArray(s.transcript)) {{
      alert('That file has no session transcript in it.');
      input.value = '';
      return;
    }}
    const id = 'import-' + Date.now().toString(16);
    sessions[id] = {{
      title: (s.title ? s.title : 'Session') + ' (imported)',
      updated: Date.now(),
      transcript: s.transcript.slice(-400),
    }};
    persistSessions();
    renderTabs();
    switchSession(id);
    input.value = '';
  }};
  reader.readAsText(file);
}}
/* Per-conversation inference target: removed.  Model selection lives in the
   Engines tab (and the Launch panel); the chat header shows the loaded
   model read-only, so there is no second place to pick one. */

/* AFK mode removed (shelved). Revisit when we revisit the shelf. */
/* --------------------- compaction + engine surgery ------------------ */
async function maybeAutoCompact() {{
  // the Studio compacts a conversation that outgrows the model's window
  if (!document.getElementById('cfg-compact')?.checked) return;
  const s = sessions[convId];
  const transcript = (s && s.transcript) || [];
  const messages = transcript
    .filter(m => m.role === 'user' || m.role === 'assistant')
    .map(m => ({{ role: m.role, text: m.text }}));
  if (messages.length < 4) return;
  const approx = Math.ceil(messages.reduce((n, m) => n + (m.text || '').length, 0) / 4);
  const limit = await studioContextLimit();
  if (!limit || approx < Math.floor(limit * 0.85)) return;
  const note = document.getElementById('cfg-note');
  if (note) note.textContent = 'auto-compacting…';
  try {{
    const r = await api('/v1/compaction/run', 'POST', {{
      messages: messages, trigger: 'auto', keep: Math.floor(limit * 0.4), limit: limit,
    }});
    if (r.status !== 'completed') return;
    s.transcript = [{{ role: 'system', text: r.checkpoint.rendered }}];
    persistSessions();
    renderTabs();
    restoreTranscript(convId);
    if (note) note.textContent = 'auto-compacted';
    await offerEngineSurgery(r.checkpoint);
  }} catch (e) {{ /* compaction is best-effort */ }}
}}

async function studioContextLimit() {{
  // the active engine profile's context, else the provider's model default
  try {{
    const eng = await api('/v1/engines');
    const cur = eng.engines.find(e => e.name === eng.default) || eng.engines[0];
    const ctx = cur && cur.load_options && (cur.load_options.context || cur.load_options.contextLength);
    if (ctx) return Number(ctx) || 0;
  }} catch (e) {{}}
  return 8192;
}}

async function compactConversation() {{
  const s = sessions[convId];
  const transcript = (s && s.transcript) || [];
  const messages = transcript
    .filter(m => m.role === 'user' || m.role === 'assistant')
    .map(m => ({{ role: m.role, text: m.text }}));
  if (messages.length < 2) {{ alert('Nothing to compact yet - send a few messages first.'); return; }}
  const note = document.getElementById('cfg-note');
  if (note) note.textContent = 'compacting…';
  try {{
    const r = await api('/v1/compaction/run', 'POST', {{
      messages: messages, trigger: 'manual', keep: 2000,
    }});
    if (r.status !== 'completed') {{
      if (note) note.textContent = 'nothing to compact: ' + (r.error || r.status);
      return;
    }}
    // the checkpoint replaces the covered history
    s.transcript = [{{ role: 'system', text: r.checkpoint.rendered }}];
    persistSessions();
    renderTabs();
    restoreTranscript(convId);
    if (note) note.textContent = 'compacted';
    await offerEngineSurgery(r.checkpoint);
  }} catch (e) {{
    if (note) note.textContent = 'compaction failed: ' + e;
  }}
}}

async function offerEngineSurgery(checkpoint) {{
  let cfg;
  try {{ cfg = await api('/v1/provider/config'); }} catch (e) {{ return; }}
  const current = cfg.default || '';
  const candidates = (cfg.providers || []).map(p => p.name).filter(n => n !== current);
  if (!candidates.length) return;
  const r = await api('/v1/compaction/surgery', 'POST', {{
    checkpoint: checkpoint, current: current, candidates: candidates,
  }});
  const panel = document.getElementById('surgery-panel');
  if (!panel) return;
  const rows = (r.results || []).map(x =>
    `<div style="margin:.2rem 0">· <b>${{x.name}}</b> — ${{x.probe_ok ? 'reachable (' + (x.probe_model || '?') + ')' : 'unreachable'}}; ` +
    `${{x.gate_passed === null ? 'not gated' : (x.gate_passed ? 'gate passed' : 'gate failed')}}` +
    `${{x.detail ? ' — ' + x.detail : ''}}</div>`).join('');
  const head = r.status === 'committed'
    ? `<b style="color:#157a3e">switched to ${{r.engine}}</b> — the checkpoint was a safe switch point`
    : `<b>engine switch declined</b> — ${{r.reason || 'no candidate passed the gate'}}`;
  panel.style.display = '';
  panel.innerHTML = `<div style="border:1.5px solid #000; border-radius:8px; padding:.6rem .8rem; background:#FFB703; color:#000">
      <div>${{head}}</div>${{rows}}
      <div class="note" style="color:#333">at a checkpoint the local engine can be swapped safely - the summary and the recent context carry over</div>
    </div>`;
}}

/* ------------------- commands: list, pin, run ----------------------- */
let _commands = [];
const PINNED_COMMANDS_KEY = 'hive-pinned-commands';

function pinnedCommands() {{
  try {{ return JSON.parse(localStorage.getItem(PINNED_COMMANDS_KEY) || '[]') || []; }}
  catch (e) {{ return []; }}
}}

function togglePinnedCommand(name) {{
  const pins = pinnedCommands();
  const i = pins.indexOf(name);
  if (i >= 0) pins.splice(i, 1); else pins.push(name);
  localStorage.setItem(PINNED_COMMANDS_KEY, JSON.stringify(pins));
  renderCommandsList();
  renderPinnedCommands();
}}

function renderCommandsList() {{
  const box = document.getElementById('commands-list');
  if (!box) return;
  const pins = pinnedCommands();
  const count = document.getElementById('commands-count');
  if (count) count.textContent = `${{_commands.length}}`;
  if (!_commands.length) {{ box.textContent = 'no commands'; return; }}
  // this is a SELECTOR: click a command to add or remove it from the row
  // above the chat bar.  (The buttons there are what you run.)
  box.innerHTML = _commands.map(c => {{
    const pinned = pins.includes(c.name);
    const hint = c.input ? ` · ${{c.input.hint}}` : '';
    return `<label title="${{c.description}}${{hint}}" style="display:flex; gap:.45rem; align-items:center; padding:.2rem .1rem; border-bottom:1px solid rgba(255,221,0,.10); cursor:pointer">
      <input type="checkbox" ${{pinned ? 'checked' : ''}} onchange="togglePinnedCommand('${{c.name}}')" style="accent-color:#FFB703; margin:0">
      <code style="background:#000; color:#FFDD00; padding:0 .25rem; border-radius:3px">/${{c.name}}</code>
      <span class="note" style="flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap">${{c.description}}</span>
    </label>`;
  }}).join('');
}}

function toggleCmdPicker() {{
  const panel = document.getElementById('cmd-picker');
  const btn = document.getElementById('cmd-picker-toggle');
  if (!panel) return;
  const show = panel.style.display === 'none' || panel.style.display === '';
  panel.style.display = show ? 'block' : 'none';
  if (btn) btn.classList.toggle('open', show);
  if (show) {{ renderCommandsList(); placeCmdPicker(); }}
}}

/* The panel is viewport-anchored: the chat column clips its overflow, so a
   panel placed inside it would be cut off at the column's edge.  It opens
   upward (the chat bar is at the bottom) and to the right, into the open
   space beside the chat column — falling back to the left when the screen
   is too narrow for that. */
function placeCmdPicker() {{
  const panel = document.getElementById('cmd-picker');
  const btn = document.getElementById('cmd-picker-toggle');
  if (!panel || !btn) return;
  if (panel.style.display === 'none') return;
  const r = btn.getBoundingClientRect();
  const width = panel.offsetWidth || 320;
  const gap = 6;
  if (r.left + width + gap <= window.innerWidth) {{
    panel.style.left = Math.round(r.left) + 'px';
    panel.style.right = 'auto';
  }} else {{
    panel.style.left = 'auto';
    panel.style.right = Math.round(window.innerWidth - r.right) + 'px';
  }}
  panel.style.bottom = Math.round(window.innerHeight - r.top + gap) + 'px';
  /* never taller than the space above the button */
  const room = Math.max(160, r.top - 16);
  panel.style.maxHeight = Math.round(room) + 'px';
}}

/* the picker follows the button and closes on Escape / an outside click */
window.addEventListener('resize', placeCmdPicker);
window.addEventListener('scroll', placeCmdPicker, true);
document.addEventListener('click', (ev) => {{
  const panel = document.getElementById('cmd-picker');
  if (!panel || panel.style.display === 'none') return;
  if (panel.contains(ev.target) || ev.target.closest('#cmd-picker-toggle')) return;
  panel.style.display = 'none';
  document.getElementById('cmd-picker-toggle')?.classList.remove('open');
}});
document.addEventListener('keydown', (ev) => {{
  if (ev.key !== 'Escape') return;
  const panel = document.getElementById('cmd-picker');
  if (!panel || panel.style.display === 'none') return;
  panel.style.display = 'none';
  document.getElementById('cmd-picker-toggle')?.classList.remove('open');
}});

function renderPinnedCommands() {{
  const row = document.getElementById('pinned-commands');
  if (!row) return;
  const pins = pinnedCommands().filter(n => _commands.some(c => c.name === n));
  if (!pins.length) {{ row.style.display = 'none'; row.innerHTML = ''; return; }}
  row.style.display = '';
  // each command is its own container: the run button, plus a tiny red x
  // inside it so one click takes the command off the row.
  row.innerHTML = pins.map(n =>
    `<div class="pin-cmd" title="Run /${{n}}">
       <button type="button" class="pin-run" onclick="runStudioCommand('${{n}}')">/${{n}}</button>
       <button type="button" class="pin-x" title="Take /${{n}} off this row"
               onclick="togglePinnedCommand('${{n}}')">×</button>
     </div>`).join('');
}}

async function loadStudioCommands() {{
  try {{
    const data = await api('/v1/commands');
    _commands = data.commands || [];
    renderCommandsList();
    renderPinnedCommands();
  }} catch (e) {{ /* the palette still works */ }}
}}

async function runStudioCommand(name) {{
  const input = document.getElementById('chatin');
  if (input) {{ input.value = '/' + name + ' '; input.focus(); }}
  else {{ await api('/v1/commands/run', 'POST', {{line: '/' + name, conversation_id: convId}}); }}
}}

/* ------------------------- pinned tabs ------------------------------- */
const PINNED_TABS_KEY = 'hive-pinned-tabs';
const TAB_REGISTRY = [
  ['tab-home', 'Home'], ['tab-settings', 'Settings'], ['tab-global', 'Global'],
  ['tab-cascade', 'Cascade'],
  ['tab-setup', 'Docker setup'], ['tab-agent', 'Agent'], ['tab-engines', 'Engines'],
  ['tab-library', 'Local Library'],
  ['tab-providers', 'Providers'], ['tab-hub', 'Hub'], ['tab-inspect', 'Inspector'],
  ['tab-training', 'Training'],
];

function pinnedTabs() {{
  try {{ return JSON.parse(localStorage.getItem(PINNED_TABS_KEY) || '[]') || []; }}
  catch (e) {{ return []; }}
}}

function togglePinnedTab(id) {{
  const pins = pinnedTabs();
  const i = pins.indexOf(id);
  if (i >= 0) pins.splice(i, 1); else pins.push(id);
  localStorage.setItem(PINNED_TABS_KEY, JSON.stringify(pins));
  applyPinnedTabs();
}}

const DEFAULT_MAIN_TABS = ['tab-home', 'tab-settings', 'tab-global', 'tab-cascade'];

function applyPinnedTabs() {{
  const main = document.getElementById('main-tabs');
  const adv = document.getElementById('adv-tabs');
  if (!main || !adv) return;
  const pins = pinnedTabs();
  for (const [id] of TAB_REGISTRY) {{
    const btn = document.querySelector(`.tab[data-tab="${{id}}"]`);
    if (!btn) continue;
    // the main strip always holds the default tabs plus whatever is pinned
    const inMain = DEFAULT_MAIN_TABS.includes(id) || pins.includes(id);
    if (inMain) {{ if (btn.parentElement !== main) main.appendChild(btn); }}
    else if (btn.parentElement === main) adv.appendChild(btn);
  }}
  // a pin marker on every tab
  for (const [id, label] of TAB_REGISTRY) {{
    const btn = document.querySelector(`.tab[data-tab="${{id}}"]`);
    if (!btn) continue;
    let mark = btn.querySelector('.pin-mark');
    if (!mark) {{
      mark = document.createElement('span');
      mark.className = 'pin-mark';
      mark.title = 'Pin this tab to the top row';
      mark.onclick = (e) => {{ e.stopPropagation(); togglePinnedTab(id); }};
      btn.appendChild(mark);
    }}
    mark.textContent = pins.includes(id) ? '★' : '☆';
    mark.style.opacity = pins.includes(id) ? '1' : '.35';
  }}
}}

/* ------------------------- dev mode --------------------------------- */
function applyDevMode(on) {{
  const research = document.getElementById('dev-research');
  if (research) research.style.display = on ? '' : 'none';
  const setupTab = document.querySelector('.tab[data-tab="tab-setup"]');
  if (setupTab) setupTab.style.display = on ? '' : 'none';
  const raw = document.getElementById('dev-raw-status');
  if (raw) raw.style.display = on ? '' : 'none';
  document.body.classList.toggle('dev-mode', !!on);
}}

/* ------------------------- studio settings --------------------------- */
async function loadStudioSettings() {{
  try {{
    const st = await api('/v1/studio/settings');
    const set = (id, val) => {{ const el = document.getElementById(id); if (el) el.checked = !!val; }};
    set('cfg-sandbox', st.sandbox_enabled);
    set('cfg-tools', st.tools_enabled);
    set('cfg-compact', st.auto_compact);
    applyToolUseSetting(!!st.tools_enabled);
    applyDevMode(!!st.dev_mode);
    fillGlobalControls(st);
  }} catch (e) {{ /* settings are optional */ }}
}}

async function saveStudioSetting(key, value) {{
  try {{
    const st = await api('/v1/studio/settings', 'PUT', {{[key]: !!value}});
    if (key === 'tools_enabled') applyToolUseSetting(!!st.tools_enabled);
    const note = document.getElementById('cfg-note');
    if (note) note.textContent = 'saved';
    setTimeout(() => {{ if (note) note.textContent = ''; }}, 1200);
  }} catch (e) {{
    const note = document.getElementById('cfg-note');
    if (note) note.textContent = 'could not save: ' + e;
  }}
}}

function applyToolUseSetting(on) {{
  // with tools off the agent cannot run: the agent toggle is disabled
  const ag = document.getElementById('cfg-agent');
  if (!ag) return;
  ag.disabled = !on;
  if (!on) ag.checked = false;
}}

/* --------------------------- research queue -------------------------- */
async function researchAdd(btn) {{
  const inp = document.getElementById('researchq');
  const q = inp.value.trim();
  if (!q) return;
  btn.disabled = true;
  try {{
    await api('/v1/research/queue', 'POST', {{question: q}});
    inp.value = '';
    loadResearchCount();
  }} finally {{ btn.disabled = false; }}
}}
async function loadResearchCount() {{
  try {{
    const r = await api('/v1/research/queue');
    const b = document.getElementById('researchcount');
    if (b) b.textContent = r.items.length ? '(' + r.items.length + ' queued)' : '';
  }} catch (e) {{}}
}}
loadResearchCount();
/* ------------------------------ chat -------------------------------- */
/* every source that can print in the chat gets its own colour + label */
const CHAT_SPEAKERS = {{
  /* the user needs no label — the text says who wrote it */
  user:    {{cls: 'user',    label: ''}},
  ai:      {{cls: 'llm',     label: 'model'}},
  harness: {{cls: 'harness', label: 'studio'}},
  /* the console speaks for the commands: it says so */
  console: {{cls: 'console', label: 'Command Console'}},
  sys:     {{cls: 'sys',     label: ''}},
}};

function bubble(role, text, meta) {{
  const log = document.getElementById('chatlog');
  const speaker = CHAT_SPEAKERS[role] || CHAT_SPEAKERS.sys;
  // a message is a row: the speaker in its own gutter on the left, then
  // the box.  The box holds the message and nothing else, so its lines
  // stay uniform — and it is only as wide as the message needs.
  const row = document.createElement('div');
  row.className = 'msg ' + speaker.cls;
  if (speaker.label) {{
    const label = document.createElement('span');
    label.className = 'role';
    label.textContent = speaker.label;
    row.appendChild(label);
  }}
  const box = document.createElement('div');
  box.className = 'bubble';
  const body = document.createElement('span');
  body.className = 'body';
  body.textContent = text;
  box.appendChild(body);
  if (meta) {{
    const m = document.createElement('span');
    m.className = 'meta';
    m.textContent = meta;
    box.appendChild(m);
  }}
  row.appendChild(box);
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
  // Cap DOM growth: a long-lived tab must not accumulate every bubble.
  while (log.children.length > 200) log.removeChild(log.firstChild);
  return box;
}}

function consoleLine(text) {{
  // errors and logs print as the console voice, never as the model's answer
  return bubble('console', text);
}}

function setBubbleText(el, text) {{
  // keep the role label: write into the body span, not the bubble
  if (!el) return;
  const body = el.querySelector('.body');
  if (body) body.textContent = text;
  else el.textContent = text;
}}

function autoGrow(el) {{
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, window.innerHeight * 0.3) + 'px';
}}

function chatMode() {{
  // the agent wins when both are on; the encoder is the prerequisite for both
  return document.getElementById('cfg-agent')?.checked ? 'agent' : 'splinter';
}}

const CHAT_TOGGLE_KEY = 'hive-chat-toggles';

function saveChatToggles() {{
  try {{
    localStorage.setItem(CHAT_TOGGLE_KEY, JSON.stringify({{
      assistant: !!document.getElementById('chat-assistant')?.checked,
      splinter: !!document.getElementById('cfg-splinter')?.checked,
      agent: !!document.getElementById('cfg-agent')?.checked,
    }}));
  }} catch (e) {{}}
}}

function loadChatToggles() {{
  let saved = {{}};
  try {{ saved = JSON.parse(localStorage.getItem(CHAT_TOGGLE_KEY) || '{{}}') || {{}}; }} catch (e) {{}}
  const a = document.getElementById('chat-assistant');
  const sp = document.getElementById('cfg-splinter');
  const ag = document.getElementById('cfg-agent');
  if (!a || !sp || !ag) return;
  a.checked = saved.assistant !== false;      // on by default
  sp.checked = saved.splinter !== false;
  ag.checked = !!saved.agent;
  applyEncoderToggle();
}}

function applyEncoderToggle() {{
  // the encoder drives both chat modes: without it, neither can be turned on
  const a = document.getElementById('chat-assistant');
  const sp = document.getElementById('cfg-splinter');
  const ag = document.getElementById('cfg-agent');
  if (!a || !sp || !ag) return;
  const on = a.checked;
  sp.disabled = !on;
  ag.disabled = !on;
  if (!on) {{ sp.checked = false; ag.checked = false; }}
  const note = document.getElementById('cfg-note');
  if (note && !on) note.textContent = 'the app assistant is off - Splinter and Agent need it';
  else if (note) note.textContent = '';
  const composer = document.getElementById('chatin');
  if (composer) composer.placeholder = on
    ? 'Talk to the loaded AI…  (/ for commands)'
    : 'Turn on the app assistant to chat';
  const summary = document.getElementById('chat-mode-summary');
  if (summary) {{
    const parts = [];
    if (on) parts.push('encoder on');
    if (on && document.getElementById('cfg-agent')?.checked) parts.push('agent');
    else if (on) parts.push('splinter');
    summary.textContent = on ? parts.join(' · ') : 'the app assistant is off';
  }}
  saveChatToggles();
}}

/* ------------------------- slash commands ---------------------------- */
let chatCommands = [];

async function loadChatCommands() {{
  try {{
    chatCommands = (await api('/v1/commands')).commands;
  }} catch (e) {{ chatCommands = []; }}
}}

function chatSubmit() {{
  const input = document.getElementById('chatin');
  const text = input.value;
  if (text.trim().startsWith('/')) {{
    input.value = '';
    hideSuggestions();
    runCommand(text.trim());
    return;
  }}
  sendChat();
}}

/* An attached file waits for the next message: the model sees it for that
   turn, and the conversation's store keeps it for later turns. */
let pendingAttachment = null;

async function runCommand(line) {{
  bubble('user', line);
  record('user', line);
  try {{
    const r = await api('/v1/commands/run', 'POST',
                        {{line: line, conversation_id: convId}});
    bubble('console', (r.kind === 'error' ? '⚠ ' : '') + (r.text || r.kind));
    record('console', (r.kind === 'error' ? '⚠ ' : '') + (r.text || r.kind));
    if (r.attachment) {{
      pendingAttachment = r.attachment;
      bubble('console', `📎 ${{r.attachment.name}} is attached — it rides along with your next message.`);
    }}
    if (r.new_conversation_id) {{
      convId = r.new_conversation_id;
      localStorage.setItem('splinter-console-conv', convId);
      ensureSession(convId);
      persistSessions();
      renderTabs();
      document.getElementById('chatlog').innerHTML = '';
    }}
    if (r.mode) {{
      const el = document.getElementById(r.mode === 'agent' ? 'cfg-agent' : 'cfg-splinter');
      if (el) {{ el.checked = true; saveChatToggles(); }}
    }}
    if (r.compact_browser) {{
      /* the command compacted the durable log; the visible transcript is
         compacted here so the chat pane shows the checkpoint too */
      await compactConversation();
    }}
  }} catch (e) {{
    bubble('console', 'command failed: ' + e.message);
  }}
}}

document.getElementById('chatin').addEventListener('input', e => {{
  const v = e.target.value;
  const box = document.getElementById('sug-chat');
  if (!v.startsWith('/')) {{ box.innerHTML = ''; return; }}
  const query = v.slice(1).toLowerCase();
  const matches = chatCommands.filter(c => c.name.startsWith(query));
  box.innerHTML = '';
  for (const cmd of matches) {{
    const item = document.createElement('div');
    item.className = 'sugitem';
    const name = document.createElement('span');
    name.textContent = '/' + cmd.name;
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = cmd.description + (cmd.input ? ' ' + cmd.input.hint : '');
    item.appendChild(name);
    item.appendChild(meta);
    item.addEventListener('mousedown', ev => {{
      ev.preventDefault();
      e.target.value = '/' + cmd.name + ' ';
      box.innerHTML = '';
      e.target.focus();
    }});
    box.appendChild(item);
  }}
}});

loadChatCommands();

async function sendChat() {{
  if (streamBusy) return;              /* one stream at a time */
  const input = document.getElementById('chatin');
  const query = input.value.trim();
  if (!query) return;
  input.value = '';
  if (chatMode() === 'agent') return sendAgent(query);
  bubble('user', query);
  record('user', query);
  const ai = bubble('ai', '');
  ai.classList.add('typing');
  setBusy(true);
  const ctrl = new AbortController();
  streamAbort = ctrl;
  const body = {{query: query, conversation_id: convId}};
  // an engine that has to load the model first says so on the bar: the
  // 176B takes minutes to come up, and silence looks like a hang
  try {{
    const st = await api('/v1/server/status');
    const strata = st.strata || {{}};
    if (st.running) {{
      showLoadBar('Loading the model\u2026', 'The engine is loading the weights.');
    }} else if (strata.alive && !strata.loaded) {{
      showLoadBar('Loading the model into VRAM\u2026',
                  'The engine loads the weights on the first request; the reply takes a while.');
    }}
  }} catch (e) {{ /* the bar is a nicety, never a blocker */ }}
  if (pendingAttachment) {{
    /* the file's text goes in with the message, so the model really sees it
       (the store keeps it too, but retrieval is not guaranteed) */
    body.query = pendingAttachment.text + '\\n\\n' + query;
    pendingAttachment = null;
  }}
  const provSel = document.getElementById('chat-provider')?.value;
  if (provSel) body.provider = provSel;
  let text = '';
  let turn = '?';
  let metaText = '';
  let harness = null;   // what the turn ran with (the Global tab's baseline)
  try {{
    const r = await fetch('/v1/splinter/stream', {{method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify(body), signal: ctrl.signal}});
    if (!r.ok || !r.body) throw new Error('HTTP ' + r.status);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {{
      const {{done, value}} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {{stream: true}});
      let idx;
      while ((idx = buf.indexOf('\\n\\n')) >= 0) {{
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!frame.startsWith('data: ')) continue;
        const ev = JSON.parse(frame.slice(6));
        if (ev.type === 'delta') {{
          if (!text) ai.classList.remove('typing');
          text += ev.text;
          setBubbleText(ai, text);
          log_scroll();
        }} else if (ev.type === 'meta') {{
          turn = ev.turn;
          harness = ev.harness || null;
          metaText = `curated ${{ev.token_count}}/${{ev.budget}} tokens`;
        }} else if (ev.type === 'done') {{
          metaText = `splinter-curated · turn ${{turn}}`
            + (ev.tokens ? ` · ${{ev.tokens}} tok` : '')
            + (ev.tokens_per_sec ? ` · ${{ev.tokens_per_sec}} tok/s` : '')
            + (harness && harness.encoder
               ? ` · ${{shortModel(harness.encoder)}}` : '')
            + (ev.stored ? '' : ' · not stored');
        }} else if (ev.type === 'stopped') {{
          /* the turn was stopped: the provider connection was closed, so the
             engine stopped generating (the text so far is kept) */
          metaText = 'stopped';
        }} else if (ev.type === 'error') {{
          metaText = 'error: ' + friendlyError(ev.error);
        }}
      }}
    }}
    ai.classList.remove('typing');
    if (!text && metaText && metaText.startsWith('error:')) {{
      /* error-only turn: print it as the console voice, not as an answer */
      ai.remove();
      consoleLine(metaText);
      record('console', metaText);
    }} else if (!text && metaText) {{
      setBubbleText(ai, metaText);
    }} else {{
      setBubbleText(ai, text || '(empty reply)');
      const m = document.createElement('span');
      m.className = 'meta';
      m.textContent = metaText;
      const summary = harnessSummary(harness);
      if (summary) m.title = summary;
      ai.appendChild(m);
    }}
    if (body.inspection !== undefined && body.inspection) {{
      renderInspection(body.inspection);
    }} else {{
      fetchInspection();
    }}
  }} catch (e) {{
    ai.classList.remove('typing');
    const cancelled = e && e.name === 'AbortError';
    const msg = cancelled ? 'cancelled.'
      : 'request failed: ' + friendlyError(e.message);
    ai.remove();
    consoleLine(msg);
    record('console', msg);
    if (!input.value) input.value = query;   /* give the draft back */
  }} finally {{
    streamAbort = null;
    setBusy(false);
    hideLoadBar(true);          /* the load bar goes with the turn */
    maybeAutoCompact();
  }}
}}

let streamBusy = false;
let streamAbort = null;
function setBusy(b) {{
  streamBusy = b;
  document.getElementById('stopbtn').style.display = b ? '' : 'none';
  document.getElementById('sendbtn').disabled = b;
}}

/* "sentence-transformers/paraphrase-MiniLM-L3-v2" -> "MiniLM-L3-v2" */
function shortModel(name) {{
  if (!name) return '';
  const parts = String(name).split('/');
  return parts[parts.length - 1];
}}

/* The one-line report of what a turn ran with: the encoder the Global tab
   chose, the curation it asked for, the sampling the reply was drawn with.
   It is the tooltip of the meta line under the reply. */
function harnessSummary(h) {{
  if (!h) return '';
  const bits = [];
  if (h.encoder) bits.push('encoder: ' + h.encoder
    + (h.backend && h.backend !== 'local' ? ' (' + h.backend + ')' : ''));
  if (h.medium_model) bits.push('medium: ' + h.medium_model);
  const mem = h.memory || {{}};
  for (const [k, v] of Object.entries(mem)) bits.push(k + '=' + v);
  const samp = h.sampling || {{}};
  for (const [k, v] of Object.entries(samp)) bits.push(k + '=' + v);
  return bits.join(' · ');
}}

/* Map backend/sidecar failure text to something a user can act on. The
   SSE `error` frame carries raw Python exception strings today; this is
   the client-side translation layer until the server emits clean copy. */
function friendlyError(msg) {{
  const s = String(msg || '');
  const low = s.toLowerCase();
  if (low.includes(':1234') || low.includes('max retries') ||
      low.includes('failed to establish') || low.includes('connection refused'))
    return 'No backend reachable on :1234 — start a server from the Models '
         + 'tab, or pick another provider.';
  if (low.includes('failed to fetch') || low.includes('networkerror') ||
      low.includes('load failed'))
    return 'Sidecar unreachable — is the harness process still running?';
  if (s.length > 300) return s.slice(0, 300) + '…';
  return s;
}}

function cancelStream() {{
  if (chatMode() === 'agent') return cancelAgent();
  // close the provider connection: the engine stops generating (llama.cpp
  // and strata both stop when the client goes away), so the GPU is free
  // now - not after the model finishes a reply nobody is waiting for.
  api('/v1/chat/stop', 'POST', {{conversation_id: convId}}).catch(() => {{}});
  if (streamAbort) streamAbort.abort();
}}

async function cancelAgent() {{
  try {{ await api('/v1/agent/cancel', 'POST'); }} catch (e) {{}}
}}

async function sendAgent(query) {{
  bubble('user', query);
  record('user', query);
  bubble('sys', 'dsh agent starting…');
  const ai = bubble('ai', '');
  setBusy(true);
  let text = '';
  let finish = '';
  try {{
    const r = await fetch('/v1/agent/stream', {{method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify({{message: query, conversation_id: convId,
        assistant: !!document.getElementById('chat-assistant')?.checked}})}});
    if (!r.ok || !r.body) {{
      const t = await r.text();
      throw new Error('HTTP ' + r.status + ' ' + t.slice(0, 200));
    }}
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {{
      const {{done, value}} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {{stream: true}});
      let idx;
      while ((idx = buf.indexOf('\\n\\n')) >= 0) {{
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!frame.startsWith('data: ')) continue;
        const ev = JSON.parse(frame.slice(6));
        if (ev.type === 'assistant') {{
          text = ev.text;                       // committed per agent step
          setBubbleText(ai, text);
          log_scroll();
        }} else if (ev.type === 'tool') {{
          /* persistent collapsible step: stays in the transcript so a long
             agent run reads like dsh's own tool view instead of toasts */
          const det = document.createElement('details');
          det.className = 'toolstep';
          const sum = document.createElement('summary');
          sum.textContent = (ev.phase === 'end' ? '\\u2713 ' : '\\u25b6 ')
            + ev.tool;
          det.appendChild(sum);
          const pre = document.createElement('pre');
          pre.textContent = JSON.stringify(ev, null, 1).slice(0, 2000);
          det.appendChild(pre);
          document.getElementById('chatlog').appendChild(det);
          record('tool', ev.phase + ': ' + ev.tool);
          log_scroll();
        }} else if (ev.type === 'lifecycle') {{
          /* turn/step boundaries stay quiet */
        }} else if (ev.type === 'done') {{
          finish = ev.finish_reason || 'end';
          if (ev.final && ev.final !== text) {{
            text = ev.final;
            setBubbleText(ai, text);
          }}
        }} else if (ev.type === 'error') {{
          ai.remove();
          consoleLine('agent error: ' + ev.error);
          return;
        }}
      }}
    }}
    setBubbleText(ai, text || '(no response)');
    const m = document.createElement('span');
    m.className = 'meta';
    m.textContent = 'dsh agent · ' + finish;
    ai.appendChild(m);
    record('ai', text || '(no response)', 'dsh agent · ' + finish);
  }} catch (e) {{
    ai.remove();
    consoleLine('agent request failed: ' + e.message);
    record('console', 'agent request failed: ' + e.message);
  }} finally {{
    setBusy(false);
    maybeAutoCompact();
  }}
}}
function log_scroll() {{
  const log = document.getElementById('chatlog');
  log.scrollTop = log.scrollHeight;
}}

function newConversation() {{
  /* OpenCode semantics: the current session keeps its tab; this opens a
     fresh conversation id and a fresh (empty) tab beside it. */
  convId = 'console-' + crypto.randomUUID().slice(0, 8);
  localStorage.setItem('splinter-console-conv', convId);
  ensureSession(convId);
  persistSessions();
  document.getElementById('chatlog').innerHTML = '';
  bubble('ai', 'Fresh session started.');
  renderTabs();
}}

/* boot: rebuild the tab strip and the last session's transcript */
ensureSession(convId);
persistSessions();
renderTabs();
restoreTranscript(convId);
/* populate engine profile dropdown for the Launch panel */
loadEngines().catch(() => {{}});
/* populate model dropdowns from local library */
loadLibrarySystem().catch(() => {{}});
loadStudioSettings();
loadChatToggles();
loadStudioCommands();
applyPinnedTabs();
/* the model dropdown populates async from /v1/provider/config */

/* ----------------------- cascade role coverage ----------------------- */
async function loadCascadeRoles() {{
  const box = document.getElementById('cascade-roles');
  if (!box) return;
  try {{
    const [reg, roles] = await Promise.all([
      api('/v1/cascade/registry'), api('/v1/cascade/roles'),
    ]);
    const roleRows = (roles.roles || []).map(r => {{
      const cands = (reg.candidates || []).filter(c => (c.roles || []).includes(r.id));
      const names = cands.map(c => c.id).slice(0, 6).join(', ') || '<span class="note">no candidate</span>';
      return `<tr><td><code>${{r.id}}</code></td><td>${{names}}</td><td class="note">${{r.description || ''}}</td></tr>`;
    }}).join('');
    const missing = (reg.missing_roles || []);
    box.innerHTML =
      (missing.length
        ? `<p class="note" style="color:#FFDD00">Uncovered role(s): <b>${{missing.join(', ')}}</b> (the deterministic policy fills these by design).</p>`
        : '<p class="note">Every role has a candidate.</p>') +
      `<table style="width:100%;margin-top:.5rem"><thead><tr><th>Role</th><th>Candidates</th><th>Description</th></tr></thead><tbody>${{roleRows}}</tbody></table>`;
  }} catch (e) {{
    box.textContent = 'Could not load the role registry: ' + e;
  }}
}}
setTimeout(loadCascadeRoles, 300);

/* --------------------------- server panel ---------------------------- */
/* The launch console has one tab per engine: what the two share (the
   context window, the port) sits above the tabs, and what differs sits
   behind its own. */
let _launchEngine = 'llamacpp';

function selectLaunchEngine(engine) {{
  _launchEngine = engine;
  for (const btn of document.querySelectorAll('#launch-engine-tabs .tab')) {{
    btn.classList.toggle('active', btn.dataset.engine === engine);
  }}
  for (const pane of document.querySelectorAll('.launch-pane')) {{
    pane.style.display = pane.id === 'launch-pane-' + engine ? '' : 'none';
  }}
  const ctx = document.getElementById('ctx');
  if (ctx) {{
    const cur = String(ctx.value).trim();
    // only reset the context when the user has not touched it
    if (cur === '' || cur === '8192' || cur === '262144') {{
      ctx.value = engine === 'strata' ? 262144 : 8192;
    }}
  }}
  if (engine === 'strata') loadStrataConfigs();
}}

async function loadStrataConfigs() {{
  const sel = document.getElementById('l-strata-config');
  if (!sel) return;
  try {{
    const r = await api('/v1/strata/configs');
    const repo = document.getElementById('l-strata-repo');
    if (repo && !repo.value && r.repo) repo.value = r.repo;
    const keep = sel.value;
    sel.innerHTML = '<option value="">— choose a config —</option>';
    for (const c of (r.configs || [])) {{
      const o = document.createElement('option');
      o.value = c.file;
      o.textContent = c.file + (c.model_name ? ' — ' + c.model_name : '');
      sel.appendChild(o);
    }}
    if (keep) sel.value = keep;
  }} catch (e) {{ /* no checkout, no configs */ }}
}}

function strataLaunchBody() {{
  const args = [];
  const vram = val('l-strata-vram');
  if (vram) args.push('--vram-reserve-mib', String(vram));
  const prefill = val('l-strata-prefill');
  if (prefill) args.push('--prefill', String(prefill));
  const spec = val('l-strata-spec');
  if (spec !== '') args.push('--spec', String(spec));
  const extra = (val('l-strata-args') || '').trim().split(/\\s+/).filter(Boolean);
  args.push(...extra);
  return {{
    repo: val('l-strata-repo') || null,
    config: val('l-strata-config') || null,
    port: +(val('l-port') || 8081),
    ctx: +(val('ctx') || 262144),
    engine_args: args,
    timeout_s: +(val('l-strata-timeout') || 600),
  }};
}}

function launchBody() {{
  const lo = {{}};
  const override = (id) => {{
    const el = document.getElementById(id);
    if (!el) return null;
    if (el.type === 'checkbox') return el.checked ? true : null;
    const v = (el.value || '').trim();
    return v === '' ? null : (el.type === 'number' ? +v : v);
  }};
  return {{
    model: (document.getElementById('model')?.value?.trim() || document.getElementById('launch-model-select')?.value || null),
    hf_repo: val('hfrepo') || null,
    hf_file: val('hffile') || null,
    ctx_size: (+(document.getElementById('ctx')?.value) || 8192),
    ngl: (+(document.getElementById('ngl')?.value) || 999),
    api_key: null,
    threads: override('l-threads'),
    flash_attn: document.getElementById('l-fa')?.checked || false,
    parallel_slots: override('l-parallel'),
    cache_type_k: override('l-ctk'),
    cache_type_v: override('l-ctv'),
    batch_size: override('l-batch'),
    ubatch_size: override('l-ubatch'),
    alias: override('l-alias'),
    mlock: document.getElementById('l-mlock')?.checked || false,
    no_mmap: document.getElementById('l-nommap')?.checked || false,
    port: override('l-port'),
    auto_fit: document.getElementById('l-autofit')?.checked || false,
    visible_devices: override('l-devices'),
    api_key: override('l-apikey'),
  }};
}}

async function startServer(btn) {{
  if (btn) btn.disabled = true;          /* slow POST — no double submits */
  if (_launchEngine === 'strata') return startStrataEngine(btn);
  const wanted = document.getElementById('launch-model-select')?.value || 'the model';
  showLoadBar('Loading ' + wanted + '…',
              'The weights are loading; the engine answers when it is up.');
  try {{
    const r = await fetch('/v1/server/start', {{method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify(launchBody())}});
    const text = await r.text();
    show('status', text);
    if (!r.ok) {{
      // a failed load stays on the bar, with the reason on it
      showLoadBar('The load failed', text, 'error');
    }} else {{
      hideLoadBar(true);
    }}
    refresh();
  }} catch (e) {{
    showLoadBar('The load failed', String(e.message || e), 'error');
  }} finally {{ if (btn) btn.disabled = false; }}
}}

async function startStrataEngine(btn) {{
  // Strata is its own serve: the Studio starts it, waits for it to answer,
  // and owns the process from then on.
  const body = strataLaunchBody();
  if (!body.config) {{
    showLoadBar('No config profile', 'Pick the Strata config to launch.', 'error');
    if (btn) btn.disabled = false;
    return;
  }}
  showLoadBar('Starting Strata…',
              'The engine loads the pack; the first load takes minutes.');
  try {{
    const r = await fetch('/v1/strata/launch', {{method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify(body)}});
    const text = await r.text();
    show('status', text);
    let ok = false;
    try {{ ok = JSON.parse(text).started === true; }} catch (e) {{}}
    if (!r.ok || !ok) {{
      showLoadBar('The start failed', text, 'error');
    }} else {{
      hideLoadBar(true);
      showToast('Strata is up');
    }}
    refresh();
  }} catch (e) {{
    showLoadBar('The start failed', String(e.message || e), 'error');
  }} finally {{ if (btn) btn.disabled = false; }}
}}

async function unloadLoaded(btn) {{
  // Unload is not "stop": it releases the model itself (the local
  // llama.cpp instances and the Strata engine) and keeps the profiles.
  if (btn) btn.disabled = true;
  showLoadBar('Unloading\u2026', 'Stopping the servers and giving the VRAM back.');
  try {{
    const r = await api('/v1/server/unload-loaded', 'POST');
    const bits = [];
    if (r.instances && r.instances.length) bits.push('unloaded ' + r.instances.join(', '));
    if (r.retired && r.retired.length) bits.push('retired ' + r.retired.join(', '));
    if (r.strata && r.strata < 400) bits.push('the strata engine gave its VRAM back');
    if (!bits.length) bits.push('nothing was loaded');
    show('status', 'unloaded: ' + bits.join('; '));
    hideLoadBar();
    // the confirmation, for a couple of seconds
    showToast('Unload Successful');
    refresh();
  }} catch (e) {{
    showLoadBar('The unload failed', String(e.message || e), 'error');
    show('status', 'unload failed: ' + e.message);
  }} finally {{ if (btn) btn.disabled = false; }}
}}

async function searchHub() {{
  const box = document.getElementById('hub-cards');
  if (!box) return;
  const q = (document.getElementById('q')?.value || '').trim();
  if (q.length < 2) {{ box.innerHTML = '<p class="note">Type at least 2 characters.</p>'; return; }}
  box.innerHTML = '<p class="note">searching…</p>';
  try {{
    const res = await api('/v1/models/hub?q=' + encodeURIComponent(q) + '&limit=12');
    const rows = (res.results || []).map(r => `
      <div style="border:1.5px solid #000; border-radius:8px; padding:.6rem .8rem; margin:.4rem 0; background:#FFB703; color:#000">
        <div style="display:flex; justify-content:space-between; gap:.6rem; align-items:baseline">
          <b style="color:#000">${{r.repo}}</b>
          <span class="note" style="color:#333">↓${{r.downloads}} · ★${{r.likes}} · ${{r.last_modified}}</span>
        </div>
        ${{r.pipeline_tag ? `<div class="note" style="color:#333">${{r.pipeline_tag}}</div>` : ''}}
        ${{r.base_model ? `<div class="note" style="color:#333">from ${{r.base_model}}</div>` : ''}}
        ${{(r.tags||[]).length ? `<div style="margin-top:.3rem">${{r.tags.slice(0,5).map(t => `<span class="note" style="color:#333">#${{t}}</span>`).join(' ')}}</div>` : ''}}
        <button type="button" style="margin-top:.5rem" onclick="hubFiles('${{r.repo}}', this)">Show files</button>
        <span class="note" style="color:#333; margin-left:.5rem" id="files-of-${{r.repo.replace(/[^a-zA-Z0-9]/g,'_')}}"></span>
      </div>`).join('');
    box.innerHTML = rows || '<p class="note">no repos matched</p>';
  }} catch (e) {{
    box.innerHTML = '<p class="note">search failed: ' + e + '</p>';
  }}
}}

async function hubFiles(repo, btn) {{
  if (btn) btn.disabled = true;
  try {{
    const res = await api('/v1/models/hub/files/' + encodeURIComponent(repo));
    const files = res.files || [];
    const key = 'files-of-' + repo.replace(/[^a-zA-Z0-9]/g, '_');
    const target = document.getElementById(key);
    if (!target) return;
    if (!files.length) {{ target.textContent = 'no .gguf files in this repo'; return; }}
    target.innerHTML = files.map(f =>
      `<div style="display:flex; gap:.5rem; align-items:center; margin-top:.3rem; flex-wrap:wrap">
         <code style="background:#000; color:#FFDD00; padding:.1rem .35rem; border-radius:4px">${{f.file}}</code>
         <span class="note" style="color:#333">${{f.size_gb}} GB</span>
         <button type="button" onclick="download('${{repo}}','${{f.file}}', this)">Download</button>
       </div>`).join('');
  }} catch (e) {{
    const box = document.getElementById('hub-cards');
    if (box) box.insertAdjacentHTML('afterbegin', '<p class="note">file list failed: ' + e + '</p>');
  }} finally {{ if (btn) btn.disabled = false; }}
}}

async function download(repo, file, btn) {{
  if (btn) btn.disabled = true;
  try {{
    await api('/v1/models/hub/download', 'POST', {{repo: repo, file: file}});
    pollHubDownloads();
  }} catch (e) {{
    const box = document.getElementById('hub-downloads');
    if (box) box.textContent = 'download failed: ' + e;
  }} finally {{ if (btn) btn.disabled = false; }}
}}

let _hubPoll = null;
async function pollHubDownloads() {{
  const box = document.getElementById('hub-downloads');
  if (!box) return;
  try {{
    const res = await api('/v1/models/hub/downloads');
    const jobs = res.downloads || res.jobs || [];
    if (!jobs.length) {{ box.textContent = 'no downloads yet'; return; }}
    box.innerHTML = jobs.map(j => `
      <div style="display:flex; gap:.6rem; align-items:center; margin:.25rem 0; flex-wrap:wrap">
        <span style="color:${{j.state === 'done' ? '#157a3e' : (j.state === 'error' ? '#b3372c' : '#000')}}; font-weight:600">${{j.state}}</span>
        <code style="background:#000; color:#FFDD00; padding:.1rem .35rem; border-radius:4px">${{(j.repo||'') + '/' + (j.filename||'')}}</code>
        ${{j.error ? `<span class="note" style="color:#b3372c">${{j.error}}</span>` : ''}}
      </div>`).join('');
    if (jobs.some(j => j.state === 'queued' || j.state === 'downloading')) {{
      clearTimeout(_hubPoll);
      _hubPoll = setTimeout(pollHubDownloads, 1500);
    }}
  }} catch (e) {{
    box.textContent = 'download status failed: ' + e;
  }}
}}

async function importFolder(input) {{
  const status = document.getElementById('import-status');
  const files = Array.from(input.files || []).filter(f => f.name.toLowerCase().endsWith('.gguf'));
  if (!files.length) {{ if (status) status.textContent = 'no .gguf files in that folder'; input.value=''; return; }}
  const fd = new FormData();
  for (const f of files) fd.append('files', f, f.webkitRelativePath || f.name);
  if (status) status.textContent = `importing ${{files.length}} file(s)…`;
  try {{
    const r = await fetch('/v1/models/local/import', {{method: 'POST', body: fd}});
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || j.error || await r.text());
    if (status) status.textContent = `imported ${{(j.imported||[]).length}} model(s)`;
    input.value = '';
    await loadLibrarySystem();
  }} catch (e) {{
    if (status) status.textContent = 'import failed: ' + e;
    input.value = '';
  }}
}}

async function setLibraryPath() {{
  const input = document.getElementById('library-path');
  const folder = input.value.trim();
  const status = document.getElementById('import-status');
  if (!folder) {{ status.textContent = 'enter a folder path'; setTimeout(()=>status.textContent='',2000); return; }}
  status.textContent = `switching to ${{folder}}…`;
  try {{
    const r = await fetch('/v1/models/local/path', {{method: 'POST', headers: {{'content-type':'application/json'}}, body: JSON.stringify({{folder}})}});
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || j.error || await r.text());
    status.textContent = `now using ${{j.models_dir}} (${{j.models?.length || 0}} models)`;
    document.getElementById('library-path-note').textContent = j.models_dir;
    refresh();
  }} catch (e) {{
    status.textContent = 'failed: ' + e.message;
  }} finally {{
    setTimeout(()=>status.textContent='',4000);
  }}
}}

function filterLibrary() {{
  const q = (document.getElementById('library-filter').value || '').toLowerCase().trim();
  const wrap = document.getElementById('local');
  let visible = 0;
  for (const row of wrap.children) {{
    const name = (row.dataset.file || '').toLowerCase();
    const show = !q || name.includes(q);
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  }}
  const count = document.getElementById('library-filter-count');
  if (count) count.textContent = q ? `${{visible}} / ${{wrap.children.length}}` : '';
  const engSel = document.getElementById('eng-model');
  const launchSel = document.getElementById('launch-model-select');
  const filterSelect = (sel) => {{
    if (!sel) return;
    for (const opt of sel.options) {{
      if (!opt.value) continue;
      opt.style.display = !q || opt.value.toLowerCase().includes(q) ? '' : 'none';
      opt.hidden = !q || opt.value.toLowerCase().includes(q) ? false : true;
    }}
  }};
  filterSelect(engSel);
  filterSelect(launchSel);
  // also filter both new panes
  for (const id of ['local-system','local-linux']) {{
    const w = document.getElementById(id);
    if (!w) continue;
    let vis=0;
    for (const row of w.children) {{
      const name=(row.dataset.file||'').toLowerCase();
      const show=!q||name.includes(q);
      row.style.display=show?'':'none';
      if(show) vis++;
    }}
  }}
}}
/* The load bar: the bottom half of the screen while a model loads or
   unloads.  It also carries the error when a load fails - the log is in
   the server log, but a failed load is a dead end the user has to see. */
let _loadBar = null;

function showLoadBar(title, detail, kind) {{
  const bar = document.getElementById('loadbar');
  if (!bar) return;
  _loadBar = {{ kind: kind || 'load' }};
  const t = document.getElementById('loadbar-title');
  const d = document.getElementById('loadbar-detail');
  if (t) t.textContent = title;
  if (d) d.textContent = detail || '';
  bar.style.display = 'flex';
}}

function hideLoadBar(onlyIfLoading) {{
  const bar = document.getElementById('loadbar');
  if (!bar) return;
  if (onlyIfLoading && _loadBar && _loadBar.kind !== 'load') return;
  bar.style.display = 'none';
  _loadBar = null;
}}

/* ------------------- the global harness controls --------------------- */
/* The baseline the harness starts from: the memory a conversation gets,
   the quality a cascade run asks for, what an MCP client may touch, the
   runtime a new engine is launched with. */
const GLOBAL_BLOCKS = {{
  ge: 'encoder', gm: 'memory_defaults', gq: 'quality_defaults',
  gp: 'mcp', gr: 'runtime_defaults',
}};

function fillGlobalControls(st) {{
  /* A control that was never stored is not "false" - it is unset.  The
     distinction matters on save: an unset checkbox must not be written
     as false, or opening the tab and pressing Save would switch off
     every default the user never touched. */
  for (const [prefix, block] of Object.entries(GLOBAL_BLOCKS)) {{
    const values = (st && st[block]) || {{}};
    for (const el of document.querySelectorAll(`[id^="${{prefix}}-"]`)) {{
      const key = el.dataset.key;
      if (!key) continue;
      if (el.type === 'checkbox') {{
        if (values[key] === undefined) {{ el.dataset.unset = '1'; el.checked = false; }}
        else {{ el.dataset.unset = ''; el.checked = !!values[key]; }}
      }} else {{
        el.value = (values[key] === undefined || values[key] === null) ? '' : values[key];
      }}
    }}
  }}
  const dev = document.getElementById('g-dev_mode');
  if (dev) dev.checked = !!(st && st.dev_mode);
}}

function collectGlobalControls() {{
  const out = {{}};
  for (const [prefix, block] of Object.entries(GLOBAL_BLOCKS)) {{
    const values = {{}};
    for (const el of document.querySelectorAll(`[id^="${{prefix}}-"]`)) {{
      const key = el.dataset.key;
      if (!key) continue;
      if (el.type === 'checkbox') {{
        if (el.checked) values[key] = true;
        else if (el.dataset.unset !== '1') values[key] = false;
      }} else if (el.value !== '') {{
        values[key] = (el.type === 'number') ? +el.value : el.value;
      }}
    }}
    out[block] = values;
  }}
  return out;
}}

async function saveGlobalControls(btn) {{
  if (btn) btn.disabled = true;
  const note = document.getElementById('global-note');
  try {{
    const body = collectGlobalControls();
    const r = await api('/v1/studio/settings', 'PUT', body);
    if (note) note.textContent = 'the baseline is saved - it takes effect for the next conversation or run';
    fillGlobalControls(r);
  }} catch (e) {{
    if (note) note.textContent = 'save failed: ' + e.message;
  }} finally {{ if (btn) btn.disabled = false; }}
}}

function showToast(msg) {{
  let t=document.getElementById('toast');
  if(!t){{ t=document.createElement('div'); t.id='toast'; t.className='toast'; document.body.appendChild(t); }}
  t.textContent=msg;
  t.style.display='block';
  t.classList.remove('fade');
  t.onclick=()=>{{ t.style.display='none'; }};
  clearTimeout(t._timer);
  // long enough to be read: the confirmation comes at the moment the bar
  // closes, and the eye is on the bottom of the screen
  t._timer=setTimeout(()=>{{ t.classList.add('fade'); setTimeout(()=>{{ t.style.display='none'; }},300); }},4500);
}}
let _ctxMenu=null;
function hideContextMenu(){{ if(_ctxMenu){{ _ctxMenu.remove(); _ctxMenu=null; }} }}
function showContextMenu(x,y,file,location) {{
  hideContextMenu();
  const menu=document.createElement('div');
  menu.className='context-menu';
  _ctxMenu=menu;
  const addItem=(label, handler, danger=false)=>{{
    const it=document.createElement('div');
    it.className='context-menu-item'+(danger?' danger':'');
    it.textContent=label;
    it.onclick=async (e)=>{{ e.stopPropagation(); hideContextMenu(); await handler(); }};
    menu.appendChild(it);
  }};
  const hfUrl='https://huggingface.co/models?search=' + encodeURIComponent(file.split('/').pop().split('\\\\').pop().replace(/\\.gguf$/i,''));
  addItem('View on Hugging Face', ()=>{{ window.open(hfUrl, '_blank'); }});
  if(location==='system'){{
    addItem('Move to Linux (/mnt/dsh_storage/models)', async ()=>{{ await moveToLinux(file); }});
    addItem('Delete', async ()=>{{ await deleteModel(file,'system'); }}, true);
  }} else {{
    addItem('Send back to Windows', async ()=>{{ await moveToWindows(file); }});
    addItem('Delete', async ()=>{{ await deleteModel(file,'linux'); }}, true);
  }}
  menu.style.left=x+'px';
  menu.style.top=y+'px';
  document.body.appendChild(menu);
  // reposition if off-screen
  const r=menu.getBoundingClientRect();
  if(r.right>window.innerWidth) menu.style.left=(window.innerWidth-r.width-8)+'px';
  if(r.bottom>window.innerHeight) menu.style.top=(window.innerHeight-r.height-8)+'px';
  setTimeout(()=>{{ document.addEventListener('click', hideContextMenu, {{once:true}}); }},10);
}}
async function moveToLinux(file) {{
  if(!confirm('Move ' + file + ' to Linux (/mnt/dsh_storage/models)?\\n\\nThis copies the GGUF via WSL to the Docker volume.')) return;
  try{{
    const r=await api('/v1/models/move-to-linux','POST',{{file}});
    showToast('Moved to Linux: ' + (r.file||file));
    _libCache.ts=0; _lastRender.system=null; _lastRender.linux=null;
    // Only refresh the two library panes, not the entire app
    await Promise.all([loadLibrarySystem(), loadLibraryLinux()]);
  }} catch(e){{ alert('Move failed: '+String(e)); }}
}}
async function moveToWindows(file) {{
  if(!confirm('Send ' + file + ' back to Windows library?')) return;
  try{{
    const r=await api('/v1/models/move-to-windows','POST',{{file}});
    showToast('Sent to Windows: ' + (r.file||file));
    _libCache.ts=0; _lastRender.system=null; _lastRender.linux=null;
    await Promise.all([loadLibrarySystem(), loadLibraryLinux()]);
  }} catch(e){{ alert('Send back failed: '+String(e)); }}
}}
async function deleteModel(file, location) {{
  if(!confirm('Delete ' + file + ' from ' + location + '?')) return;
  try{{
    if(location==='system') await api('/v1/models/local?file='+encodeURIComponent(file),'DELETE');
    else await api('/v1/models/linux?file='+encodeURIComponent(file),'DELETE');
    showToast('Deleted ' + file);
    // UI-only update: remove the row, no full reload
    const wrapId=location==='system'?'local-system':'local-linux';
    const wrap=document.getElementById(wrapId);
    if(wrap) {{
      for(const row of [...wrap.children]){{
        if(row.dataset.file===file) row.remove();
      }}
      if(!wrap.children.length){{ wrap.textContent='(no .gguf files yet)'; wrap.style.color='#000'; }}
    }}
    // Update caches
    const key=location;
    if(_libCache[key]) _libCache[key]=_libCache[key].filter(m=>m.file!==file);
    if(_lastRender[key]) _lastRender[key]=_lastRender[key].filter(m=>m.file!==file);
    if(window._selectedFile===file) window._selectedFile=null;
    // Keep hidden old #local in sync
    const old=document.getElementById('local');
    if(old && location==='system') old.innerHTML=document.getElementById('local-system').innerHTML;
  }} catch(e){{ alert(String(e)); }}
}}
let _lastRender={{system: null, linux: null}};
function renderLibrary(wrapId, models, location) {{
  const wrap=document.getElementById(wrapId);
  if(!wrap) return;
  // Avoid flicker: if models are the same as last render, skip
  const cacheKey=location;
  const last=_lastRender[cacheKey];
  const same=last && last.length===models.length && last.every((m,i)=>m.file===models[i].file && m.size_gb===models[i].size_gb);
  if(same) {{
    // Just restore selection without re-rendering
    if(window._selectedFile) {{
      for(const r of wrap.children) r.classList.toggle('selected', r.dataset.file===window._selectedFile);
    }}
    return;
  }}
  _lastRender[cacheKey]=models.map(m=>({{file:m.file, size_gb:m.size_gb}}));
  wrap.innerHTML='';
  if(!models.length){{ wrap.textContent='(no .gguf files yet)'; wrap.style.color='#000'; return; }}
  for(const m of models){{
    const row=document.createElement('div');
    row.className='librow';
    row.dataset.file=m.file;
    const label=document.createElement('span');
    const short=m.file.split('/').pop().split('\\\\').pop();
    label.textContent=`${{short}} — ${{m.size_gb}} GB`;
    // tooltip
    const arch=m.architecture||m.gguf_metadata?.['general.architecture']||m.ggufMetadata?.['general.architecture']||'—';
    const quant=m.quantization||m.gguf_metadata?.quantization||m.ggufMetadata?.quantization||'—';
    const ctx=m.context_length??m.contextLength??m.gguf_metadata?.context_length??m.ggufMetadata?.context_length??(()=>{{ const g=m.gguf_metadata||m.ggufMetadata||{{}}; for(const k in g) if(k.endsWith('.context_length')) return g[k]; return '—'; }})();
    const size=m.size_gb!=null?`${{m.size_gb}} GB`:(m.sizeGb!=null?`${{m.sizeGb}} GB`:'—');
    const mod=m.modified||m.lastModified||'—';
    row.title=`${{arch}} · ${{quant}} · ${{ctx}} · ${{size}} · ${{mod}}`;
    row.addEventListener('click', (e)=>{{
      // left click selects
      if(e.button!==0) return;
      const launchSelect=document.getElementById('launch-model-select');
      if(launchSelect){{ launchSelect.value=m.file; for(const o of launchSelect.options) if(o.value===m.file){{o.hidden=false; o.style.display='';}} }}
      window._selectedFile=m.file;
      // update selection visuals for both panes
      for(const id of ['local-system','local-linux']) {{
        const w=document.getElementById(id);
        if(!w) continue;
        for(const r of w.children) r.classList.toggle('selected', r.dataset.file===m.file);
      }}
    }});
    row.addEventListener('contextmenu', (e)=>{{
      e.preventDefault();
      showContextMenu(e.clientX, e.clientY, m.file, location);
    }});
    // restore selection if this is the selected file
    if(window._selectedFile && window._selectedFile===m.file) row.classList.add('selected');
    // long-press visual
    let pt=null;
    const sp=(on)=>row.classList.toggle('pressed',on);
    row.addEventListener('mousedown',()=>{{ pt=setTimeout(()=>sp(true),400); }});
    row.addEventListener('mouseup',()=>{{ clearTimeout(pt); sp(false); }});
    row.addEventListener('mouseleave',()=>{{ clearTimeout(pt); sp(false); }});
    wrap.appendChild(row);
    row.appendChild(label);
    // keep row as flex with label only; context menu handles actions
  }}
}}
async function loadLibrarySystem() {{
  const t0=performance.now();
  const wrapSys=document.getElementById('local-system');
  const isCached = _libCache.system && (Date.now() - _libCache.ts < 30000);
  if(wrapSys && !isCached) wrapSys.textContent='loading…';
  console.log('loadLibrarySystem START', new Date().toISOString(), isCached?'cached':'fetch');
  try{{
    const ctrl=new AbortController(); const to=setTimeout(()=>ctrl.abort(), 8000);
    const l=await api('/v1/models/local', 'GET', null, ctrl.signal).catch(e=>{{ clearTimeout(to); throw e; }});
    clearTimeout(to);
    console.log('loadLibrarySystem FETCH', (performance.now()-t0).toFixed(0)+'ms', l.models?.length+' models');
    const pathInput=document.getElementById('library-path');
    const pathNote=document.getElementById('library-path-note');
    if(l.models_dir) {{
      if(pathInput && !pathInput.value) pathInput.value=l.models_dir;
      if(pathNote) pathNote.textContent=`(${{l.models_dir}})`;
    }}
    // also populate engine selects — preserve selection
    const launchSelect=document.getElementById('launch-model-select');
    const engSel=document.getElementById('eng-model');
    const prevSel=window._selectedFile || (launchSelect?launchSelect.value:'');
    if(launchSelect) launchSelect.innerHTML='<option value="">— choose local model —</option>';
    if(engSel) engSel.innerHTML='<option value="">— choose local model —</option>';
    for(const m of l.models) {{
      const short=m.file.split('/').pop().split('\\\\').pop();
      if(launchSelect) {{
        const o=document.createElement('option'); o.value=m.file; o.textContent=short; launchSelect.appendChild(o);
        if(engSel){{ const oe=document.createElement('option'); oe.value=m.file; oe.textContent=short; engSel.appendChild(oe); }}
      }}
    }}
    // restore previous selection if still valid
    if(prevSel && l.models.some(m=>m.file===prevSel)) {{
      if(launchSelect) launchSelect.value=prevSel;
      window._selectedFile=prevSel;
    }} else if(window._selectedFile && !l.models.some(m=>m.file===window._selectedFile)) {{
      // selected file no longer exists (deleted) — clear
      window._selectedFile=null;
      if(launchSelect) launchSelect.value='';
    }}
    _libCache.system=l.models; _libCache.ts=Date.now();
    renderLibrary('local-system', l.models, 'system');
    filterLibrary();
    // keep hidden old #local in sync for filter
    const old=document.getElementById('local');
    if(old){{ old.innerHTML=document.getElementById('local-system').innerHTML; }}
  }} catch(e){{ console.error('system fetch failed',e); const w=document.getElementById('local-system'); if(w) w.textContent='load failed: '+String(e).slice(0,200); }}
}}
async function loadLibraryLinux() {{
  const t0=performance.now();
  const wrap=document.getElementById('local-linux');
  const isCachedLinux = _libCache.linux && (Date.now() - _libCache.ts < 30000);
  if(wrap && !isCachedLinux) wrap.textContent='loading…';
  console.log('loadLibraryLinux START', new Date().toISOString(), isCachedLinux?'cached':'fetch');
  try{{
    const ctrl=new AbortController(); const to=setTimeout(()=>ctrl.abort(), 8000);
    const l=await api('/v1/models/linux', 'GET', null, ctrl.signal).catch(e=>{{ clearTimeout(to); throw e; }});
    clearTimeout(to);
    console.log('loadLibraryLinux FETCH', (performance.now()-t0).toFixed(0)+'ms', (l.models||[]).length+' models', l.mounted?'mounted':'not mounted');
    _libCache.linux=l.models||[]; _libCache.ts=Date.now();
    renderLibrary('local-linux', l.models||[], 'linux');
    // Keep filter in sync
    filterLibrary();
    if(l.error) console.log('linux models', l.error);
  }} catch(e){{ console.error('linux fetch failed',e); if(wrap) wrap.textContent='load failed: '+String(e).slice(0,200); }}
}}
async function loadLibrary() {{
  await Promise.all([loadLibrarySystem(), loadLibraryLinux()]);
}}

let _currentAgentPreset = null;
async function loadAgentPresets() {{
  const wrap = document.getElementById('agent-presets-list');
  const detail = document.getElementById('agent-preset-detail');
  if (wrap) wrap.textContent = 'loading…';
  try {{
    // Try DSH apiproxy first, fallback to local file list
    let presets = [];
    try {{ const r = await api('/v1/agent-presets/list'); presets = r.presets || r; }} catch(e) {{
      // Fallback: list from /v1/provider/config + shipped presets
      const r = await api('/v1/provider/config'); presets = (r.providers||[]).map(p=>({{id:p.name, name:p.displayName||p.name, trust:'user'}}));
    }}
    if (!wrap) return;
    wrap.innerHTML = '';
    if (!presets.length) {{ wrap.textContent = '(no presets)'; return; }}
    for (const p of presets) {{
      const row = document.createElement('div');
      row.className = 'librow';
      const label = document.createElement('span');
      label.textContent = `${{p.name||p.id}} — ${{p.trust||'system'}}${{p.broken?' — broken':''}}`;
      const open = document.createElement('button');
      open.textContent = p.trust==='user' ? 'Edit' : 'View';
      open.onclick = async () => {{
        _currentAgentPreset = p.id;
        document.getElementById('agent-preset-name').textContent = p.name||p.id;
        const contentEl = document.getElementById('agent-preset-content');
        try {{
          const r = await api('/v1/agent-presets/read', 'POST', {{agentPreset:p.id}});
          contentEl.textContent = r.content || JSON.stringify(r,null,2);
        }} catch(e) {{ contentEl.textContent = 'cannot read: '+e.message; }}
        detail.style.display='';
      }};
      const dup = document.createElement('button');
      dup.textContent = 'Duplicate';
      dup.onclick = async () => {{
        const nid = prompt('New preset id (a-z0-9-):', p.id+'-copy');
        if (!nid) return;
        try {{ await api('/v1/agent-presets/copy', 'POST', {{from:p.id, agentPreset:nid}}); loadAgentPresets(); }} catch(e){{ alert(String(e)); }}
      }};
      row.appendChild(label);
      row.appendChild(open);
      row.appendChild(dup);
      if (p.trust==='user') {{
        const del = document.createElement('button'); del.textContent='Delete'; del.onclick=async()=>{{ if(!confirm('Delete '+p.id+'?'))return; try{{ await api('/v1/agent-presets/remove','POST',{{agentPreset:p.id}}); loadAgentPresets();}}catch(e){{alert(String(e));}} }}; row.appendChild(del);
      }}
      if (p.description) row.title = p.description;
      wrap.appendChild(row);
    }}
  }} catch(e) {{ if(wrap) wrap.textContent='load failed: '+e.message; }}
}}
async function createAgentPreset() {{
  const id = prompt('New preset id (a-z0-9-):', 'my-preset');
  if (!id) return;
  const from = prompt('Copy from preset id (blank for empty):', 'standard') || 'standard';
  try {{ await api('/v1/agent-presets/copy','POST',{{from, agentPreset:id}}); loadAgentPresets(); }} catch(e){{ alert(String(e)); }}
}}
async function saveAgentPreset() {{
  const contentEl = document.getElementById('agent-preset-content');
  const msg = document.getElementById('agent-preset-msg');
  if (!_currentAgentPreset || !contentEl) return;
  msg.textContent='saving…';
  try {{
    // For now, just show that editing is via files; direct save not yet wired in Studio
    msg.textContent='editing via files: save agent.cordis.yml in ~/.dsh/.agent-presets/'+_currentAgentPreset+'/';
    // Future: POST to /v1/agent-presets/write
  }} catch(e){{ msg.textContent='save failed: '+e.message; }}
  setTimeout(()=>{{msg.textContent='';}},3000);
}}
function closeAgentPreset() {{
  document.getElementById('agent-preset-detail').style.display='none';
  _currentAgentPreset=null;
}}

/* --------------------------- engines tab ----------------------------- */
let engines = [];
let engDefault = '';

function samplingToForm(s) {{
  const set = (id, v) => document.getElementById(id).value =
    (v === undefined || v === null) ? '' : v;
  set('s-temp', s.temperature); set('s-topp', s.top_p); set('s-topk', s.top_k);
  set('s-minp', s.min_p); set('s-rep', s.repeat_penalty);
  set('s-pres', s.presence_penalty); set('s-freq', s.frequency_penalty);
  set('s-seed', s.seed); set('s-miro', s.mirostat);
  set('s-tau', s.mirostat_tau); set('s-eta', s.mirostat_eta);
  set('s-stop', Array.isArray(s.stop) ? s.stop.join(',') : (s.stop || ''));
}}

function formToSampling() {{
  const out = {{}};
  const put = (key, id, parse) => {{
    const raw = val(id);
    if (raw !== '') out[key] = parse ? +raw : raw;
  }};
  put('temperature', 's-temp', true); put('top_p', 's-topp', true);
  put('top_k', 's-topk', true); put('min_p', 's-minp', true);
  put('repeat_penalty', 's-rep', true); put('presence_penalty', 's-pres', true);
  put('frequency_penalty', 's-freq', true); put('seed', 's-seed', true);
  put('mirostat', 's-miro', true); put('mirostat_tau', 's-tau', true);
  put('mirostat_eta', 's-eta', true);
  const stop = val('s-stop');
  if (stop) out.stop = stop.includes(',') ? stop.split(',').map(x => x.trim())
                                          : stop;
  return out;
}}

function currentEngineIndex() {{
  return +document.getElementById('eng-select').value;
}}

async function loadProviders() {{
  try {{
    const pc = await api('/v1/provider/config');
    window._providers = pc.providers || [];
    window._provDefault = pc.default || '';
  }} catch (err) {{
    window._providers = [];
    window._provDefault = '';
  }}
  return window._providers;
}}

function fillProviderSelect(sel, selected) {{
  if (!sel) return;
  sel.innerHTML = '<option value="">— none (use base URL below) —</option>';
  for (const p of (window._providers || [])) {{
    const o = document.createElement('option');
    o.value = p.name;
    o.textContent = p.name + (p.model ? ' — ' + p.model.split('/').pop() : '');
    sel.appendChild(o);
  }}
  sel.value = selected || '';
}}

function engineSelected() {{
  const e = engines[currentEngineIndex()];
  if (!e) return;
  document.getElementById('eng-name').value = e.name;
  document.getElementById('eng-kind').value = e.kind;
  document.getElementById('eng-url').value = e.base_url || '';
  document.getElementById('eng-apikey').value = '';
  // populate provider dropdown (providers are cached separately — this
  // function stays synchronous so callers can rely on it)
  const provSel = document.getElementById('eng-provider');
  if (provSel) {{
    if (!window._providers) {{ loadProviders().then(() => fillProviderSelect(provSel, e.provider)); }}
    else fillProviderSelect(provSel, e.provider);
  }}
  document.getElementById('eng-default').checked =
    e.name.toLowerCase() === engDefault.toLowerCase();
  samplingToForm(e.sampling || {{}});
  show('eng-loadopts', e.load_options && Object.keys(e.load_options).length
    ? e.load_options : '(none recorded)');
  engDirty = false;
  
  // merged grid: populate uniform fields
  const mSel = document.getElementById('eng-model');
  if (mSel && e.model) mSel.value = e.model;
  const ctxLenInput = document.getElementById('eng-ctxlen');
  if (ctxLenInput) ctxLenInput.value = String(e.load_options?.context ?? e.load_options?.contextLength ?? 8192);
  const fitCtx = document.getElementById('fit-ctx');
  if (fitCtx) fitCtx.value = String(e.load_options?.context ?? 8192);
  const thr = document.getElementById('eng-threads');
  if (thr) thr.value = e.load_options?.threads ?? '';
  const gpu = document.getElementById('eng-gpu');
  if (gpu) gpu.value = String(e.load_options?.gpu_layers ?? 999);
  const flash = document.getElementById('eng-flash');
  if (flash) flash.value = e.load_options?.flash_attn ? 'on' : '';
  const par = document.getElementById('eng-parallel');
  if (par) par.value = e.load_options?.parallel_slots ?? '';
  const bat = document.getElementById('eng-batch');
  if (bat) bat.value = e.load_options?.batch_size ?? '';
  const ubat = document.getElementById('eng-ubatch');
  if (ubat) ubat.value = e.load_options?.ubatch_size ?? '';
  const ctk = document.getElementById('eng-ctk');
  if (ctk) ctk.value = e.load_options?.cache_type_k ?? '';
  const ctv = document.getElementById('eng-ctv');
  if (ctv) ctv.value = e.load_options?.cache_type_v ?? '';
  const alias = document.getElementById('eng-alias');
  if (alias) alias.value = e.load_options?.alias ?? '';
  const disp = document.getElementById('eng-displayName');
  if (disp) disp.value = e.displayName || e.name || '';
  const apik = document.getElementById('eng-apikey');
  if (apik) apik.value = '';
document.getElementById('eng-msg').textContent = '';
}}

function engProviderChanged() {{
  const provSel = document.getElementById('eng-provider');
  const urlEl = document.getElementById('eng-url');
  const keyEl = document.getElementById('eng-apikey');
  if (!provSel) return;
  const name = provSel.value;
  const prov = (window._providers || []).find(p => p.name === name);
  if (prov) {{
    if (urlEl) urlEl.value = prov.base_url || '';
    if (keyEl) keyEl.value = prov.api_key || '';
  }} else {{
    if (urlEl) urlEl.value = '';
    if (keyEl) keyEl.value = '';
  }}
  engDirty = true;
}}

function engineAdd() {{
  engines.push({{name: 'engine-' + (engines.length + 1), kind: 'llama_cpp',
                base_url: '', load_options: {{}}, capabilities: ['streaming'],
                sampling: {{}}}});
  renderEngineSelect(engines.length - 1);
  engineSelected();
  engDirty = true;
}}

function renderEngineSelect(selectIndex) {{
  const sel = document.getElementById('eng-select');
  sel.innerHTML = '';
  engines.forEach((e, i) => {{
    const o = document.createElement('option');
    o.value = i;
    o.textContent = e.name + (e.name.toLowerCase() === engDefault.toLowerCase()
                              ? '  (default)' : '');
    sel.appendChild(o);
  }});
  sel.value = selectIndex;
  refreshAbSelects();
}}

function refreshAbSelects() {{
  const ids = ['ab-select-a','ab-select-b','ab-profile-a','ab-profile-b'];
  for (const id of ids) {{
    const sel = document.getElementById(id);
    if (!sel) continue;
    const cur = sel.value;
    sel.innerHTML = '';
    engines.forEach(e => {{
      const o = document.createElement('option');
      o.value = e.name;
      o.textContent = e.name;
      sel.appendChild(o);
    }});
    if (cur && [...sel.options].some(o => o.value === cur)) sel.value = cur;
    else if (sel.options.length) {{
      if (id.endsWith('-a') && sel.options.length >= 1) sel.value = sel.options[0].value;
      if (id.endsWith('-b') && sel.options.length >= 2) sel.value = sel.options[1].value;
      else if (id.endsWith('-b') && sel.options.length >= 1) sel.value = sel.options[0].value;
    }}
  }}
}}

async function benchAb() {{
  const a = document.getElementById('ab-select-a')?.value || document.getElementById('ab-profile-a')?.value || '';
  const b = document.getElementById('ab-select-b')?.value || document.getElementById('ab-profile-b')?.value || '';
  const basePort = parseInt(document.getElementById('ab-baseport')?.value || '1234', 10);
  const btn = document.getElementById('ab-bench-btn');
  const winnerEl = document.getElementById('ab-winner');
  const badgeEl = document.getElementById('ab-badge');
  const ra = document.getElementById('ab-result-a');
  const rb = document.getElementById('ab-result-b');
  if (!a || !b) {{ if (winnerEl) winnerEl.textContent = 'select two profiles'; return; }}
  if (btn) btn.disabled = true;
  if (winnerEl) winnerEl.textContent = 'benching…';
  if (badgeEl) badgeEl.textContent = 'benching…';
  if (ra) ra.textContent = '';
  if (rb) rb.textContent = '';
  try {{
    let r;
    try {{
      r = await api('/v1/engines/ab/bench', 'POST', {{profile_a: a, profile_b: b, basePort: basePort, base_port: basePort}});
    }} catch (e) {{
      // fallback aliases
      try {{ r = await api('/v1/engines/bench', 'POST', {{profile_a: a, profile_b: b, basePort: basePort}}); }}
      catch (e2) {{ r = await api('/v1/ab/bench', 'POST', {{a: a, b: b, basePort: basePort}}); }}
    }}
    const winner = r.winner || (r.a_tok_per_sec > r.b_tok_per_sec ? 'A' : r.b_tok_per_sec > r.a_tok_per_sec ? 'B' : 'tie');
    const atps = (r.a_tok_per_sec ?? r.a?.tok_per_sec ?? r.tokens_per_sec_a ?? 0).toFixed(1);
    const btps = (r.b_tok_per_sec ?? r.b?.tok_per_sec ?? r.tokens_per_sec_b ?? 0).toFixed(1);
    const txt = `winner: ${{winner}} — A ${{atps}} tok/s vs B ${{btps}} tok/s`;
    if (winnerEl) {{ winnerEl.textContent = txt; winnerEl.className = 'winner-badge winner-' + winner.toLowerCase(); }}
    if (badgeEl) {{ badgeEl.textContent = txt; badgeEl.style.display = ''; badgeEl.className = 'winner-badge'; }}
    if (ra) ra.textContent = `A (${{a}}) ${{atps}} tok/s`;
    if (rb) rb.textContent = `B (${{b}}) ${{btps}} tok/s`;
  }} catch (e) {{
    const msg = 'error: ' + (e.message || String(e));
    if (winnerEl) winnerEl.textContent = msg;
    if (badgeEl) {{ badgeEl.textContent = msg; badgeEl.style.display = ''; }}
  }} finally {{
    if (btn) btn.disabled = false;
  }}
}}

function collectEngines() {{
  const e = engines[currentEngineIndex()];
  if (e) {{
    e.name = val('eng-name') || e.name;
    e.kind = document.getElementById('eng-kind').value;
    e.base_url = val('eng-url');
    e.sampling = formToSampling();
  }}
  return engines;
}}

async function loadEngines() {{
  try {{
    const data = await api('/v1/engines');
    engines = data.engines.map(e => ({{...e}}));
    engDefault = data.default || '';
    engRevision = data.revision ?? 0;
    // also fetch provider revision for atomic guard
    try {{ const p = await api('/v1/provider/config'); provRevision = p.revision ?? 0; provDefault = p.default || provDefault; }} catch(e) {{}}
    renderEngineSelect(0);
    engineSelected();
  }} catch (e) {{ show('eng-loadopts', String(e)); }}
}}

// unified Engine profiles grid — single form edits both namespaces atomically (ctx.models lifecycle + ctx.llm provider profile)
let engRevision = 0;
let provRevision = 0;
let engDirty = false;

// settings façade mirroring dsh-settings: mutate(ns, payload, expectedRevision)
const settings = {{
  mutate: async (ns, payload, expectedRevision) => {{
    const path = ns === 'llm' ? '/v1/provider/config' : '/v1/engines';
    const body = ns === 'llm'
      ? {{ providers: payload.providers, default: payload.default, persist: true, expectedRevision }}
      : {{ engines: payload.engines, default: payload.default, persist: true, expectedRevision }};
    const r = await api(path, 'POST', body);
    if (ns === 'llm') provRevision = r.revision ?? provRevision + 1;
    else engRevision = r.revision ?? engRevision + 1;
    return r;
  }}
}};

async function requestLoad(opts) {{
  // ctx.models lifecycle: spawn /v1/server/start with the profile's load_options
  return api('/v1/server/start', 'POST', opts);
}}

async function saveEngines() {{
  // legacy alias — now delegates to the unified saver
  return saveEngineProfile();
}}

async function saveEngineProfile() {{
  const form = document.getElementById('engine-profile-form');
  if (!form) return saveEnginesLegacy();
  const idx = currentEngineIndex();
  const eng = engines[idx];
  if (!eng) return;
  // collect Endpoint + Model + Context into the two namespaces
  eng.name = val('eng-name') || eng.name;
  eng.kind = document.getElementById('eng-kind').value;
  eng.base_url = val('eng-url');
  eng.provider = document.getElementById('eng-provider')?.value || '';
  eng.displayName = val('eng-displayName') || eng.name;
  // Model dropdown from local library
  const selModel = val('eng-model') || document.getElementById('launch-model-select')?.value || '';
  eng.model = selModel;
  // Context
  const ctxLen = parseInt(val('eng-ctxlen') || document.getElementById('fit-ctx')?.value || '8192', 10);
  eng.load_options = eng.load_options || {{}};
  eng.load_options.context = ctxLen;
  eng.load_options.contextLength = ctxLen;
  // Load grid
  const threadsVal = val('eng-threads'); if (threadsVal) eng.load_options.threads = parseInt(threadsVal,10); else delete eng.load_options.threads;
  const gpuVal = val('eng-gpu'); if (gpuVal) eng.load_options.gpu_layers = parseInt(gpuVal,10);
  const flashVal = document.getElementById('eng-flash')?.value; if (flashVal) eng.load_options.flash_attn = flashVal === 'on'; else delete eng.load_options.flash_attn;
  const parVal = val('eng-parallel'); if (parVal) eng.load_options.parallel_slots = parseInt(parVal,10); else delete eng.load_options.parallel_slots;
  const batchVal = val('eng-batch'); if (batchVal) eng.load_options.batch_size = parseInt(batchVal,10); else delete eng.load_options.batch_size;
  const ubatchVal = val('eng-ubatch'); if (ubatchVal) eng.load_options.ubatch_size = parseInt(ubatchVal,10); else delete eng.load_options.ubatch_size;
  eng.load_options.cache_type_k = val('eng-ctk') || undefined;
  eng.load_options.cache_type_v = val('eng-ctv') || undefined;
  eng.load_options.alias = val('eng-alias') || undefined;
  // Sampling 4x2
  eng.sampling = formToSampling();
  // Advanced (mirostat etc already in sampling)
  // persist expectedRevision guard — 409 conflict if stale
  const msgEl = document.getElementById('eng-msg');
  const saveBtn = document.getElementById('eng-save');
  if (saveBtn) saveBtn.disabled = true;
  if (msgEl) msgEl.textContent = 'saving…';
  // build llm provider profile (ctx.llm) from Endpoint section
  const llmPayload = {{
    providers: [{{ name: eng.name, base_url: eng.base_url, api_key: val('eng-apikey') || '', model: selModel || eng.model || '', displayName: eng.displayName || eng.name }}],
    default: document.getElementById('eng-default')?.checked ? eng.name : provDefault,
  }};
  const modelsPayload = {{
    engines: engines,
    default: document.getElementById('eng-default')?.checked ? eng.name : engDefault,
  }};
  const loadOpts = {{
    model: selModel || eng.model || null,
    ctx_size: ctxLen,
    ngl: eng.load_options.gpu_layers ?? 999,
    auto_fit: document.getElementById('eng-autofit')?.checked ?? false,
    visible_devices: val('eng-devices') || null,
    threads: eng.load_options.threads ?? null,
    flash_attn: eng.load_options.flash_attn ?? false,
    parallel_slots: eng.load_options.parallel_slots ?? null,
    cache_type_k: eng.load_options.cache_type_k ?? null,
    cache_type_v: eng.load_options.cache_type_v ?? null,
    batch_size: eng.load_options.batch_size ?? null,
    ubatch_size: eng.load_options.ubatch_size ?? null,
    alias: eng.load_options.alias ?? null,
    mlock: eng.load_options.mlock ?? false,
    no_mmap: eng.load_options.no_mmap ?? false,
  }};
  try {{
    // one Save mutates both namespaces + the lifecycle, with the
    // expectedRevision guard on each (409 = someone else changed it first)
    await Promise.all([
      settings.mutate('llm', llmPayload, provRevision),
      settings.mutate('models', modelsPayload, engRevision),
      requestLoad(loadOpts),
    ]);
    if (msgEl) msgEl.textContent = 'saved + loaded';
    engDirty = false;
  }} catch (e) {{
    const txt = String(e.message || e);
    if (txt.includes('409') || txt.toLowerCase().includes('revision')) {{
      if (msgEl) msgEl.textContent = 'conflict: stale revision — reload and retry (expectedRevision guard)';
    }} else {{
      if (msgEl) msgEl.textContent = 'save failed: ' + txt;
    }}
    throw e;
  }} finally {{
    if (saveBtn) saveBtn.disabled = false;
    setTimeout(() => {{ if (msgEl && msgEl.textContent.startsWith('saved')) msgEl.textContent = ''; }}, 3000);
  }}
}}

async function saveEnginesLegacy() {{

  const list = collectEngines();
  const defName = document.getElementById('eng-default').checked
    ? (val('eng-name') || (engines[currentEngineIndex()] || {{}}).name || '')
    : engDefault;
  try {{
    const r = await api('/v1/engines', 'POST',
      {{engines: list, default: defName, persist: true}});
    engDefault = r.default;
    renderEngineSelect(currentEngineIndex());
    document.getElementById('eng-msg').textContent =
      'saved to ' + r.persisted_to;
    engDirty = false;
  }} catch (e) {{ document.getElementById('eng-msg').textContent = String(e); }}
}}

/* ---------------- Auto presets: hardware + model size + gguf-metadata → load_options ---------------- */
function computeAutoPreset(size_gb, hw, file, ggufMeta) {{
  const avail = Number(hw.available_gb ?? hw.vram_gb ?? hw.total_ram_gb ?? 8) || 8;
  const name = (file || '').toLowerCase();
  const meta = ggufMeta || {{}};
  // gguf-metadata: prefer declared parameter count / block_count when available
  const paramsB = Number(meta.parameter_count) ? Number(meta.parameter_count)/1e9
    : Number(meta['general.parameter_count']) ? Number(meta['general.parameter_count'])/1e9
    : null;
  const blockCount = Number(meta.block_count ?? meta['llama.block_count'] ?? meta['qwen.block_count']) || null;
  // Heuristic layers for known families when metadata missing
  let estLayers = blockCount;
  if (!estLayers) {{
    if (name.includes('32b') || name.includes('30b') || (paramsB && paramsB>=30) || size_gb>15) estLayers = 62;
    else if (name.includes('14b') || name.includes('13b') || (paramsB && paramsB>=13)) estLayers = 40;
    else if (name.includes('7b') || name.includes('8b') || (paramsB && paramsB>=7)) estLayers = 32;
    else if (name.includes('4b') || name.includes('3b') || (paramsB && paramsB>=3)) estLayers = 36;
    else if (name.includes('qwen3-4b')) estLayers = 36;
    else if (name.includes('qwen3-32b')) estLayers = 64;
    else estLayers = 32;
  }}
  let gpu_layers, ctx, threads, flash_attn, cache_k, cache_v;
  const low = name;
  const is32 = low.includes('32b') || low.includes('30b') || (paramsB && paramsB>=30) || size_gb>15;
  const is4 = low.includes('4b') || low.includes('3b') || low.includes('qwen3-4b') || (paramsB && paramsB>=3 && paramsB<6) || (size_gb>=2 && size_gb<6);
  if (is32) {{
    // qwen3-32b example: on 8GB → offload 12 + 4k
    if (avail <= 9) {{ gpu_layers = 12; ctx = 4096; }}
    else if (avail <= 16) {{ gpu_layers = 20; ctx = 8192; }}
    else if (avail <= 24) {{ gpu_layers = Math.min(40, estLayers); ctx = 8192; }}
    else {{ gpu_layers = 999; ctx = 16384; }}
  }} else if (is4) {{
    // qwen3-4b example: on 8GB → gpu_layers 28 + 8k ctx
    if (avail <= 9) {{ gpu_layers = 28; ctx = 8192; }}
    else if (avail <= 16) {{ gpu_layers = 999; ctx = 16384; }}
    else {{ gpu_layers = 999; ctx = 32768; }}
  }} else {{
    if (size_gb + 2.0 <= avail * 0.9) {{ gpu_layers = 999; ctx = 8192; }}
    else if (size_gb + 1.0 <= avail * 1.1) {{ gpu_layers = Math.min(28, estLayers); ctx = 8192; }}
    else {{ gpu_layers = Math.min(12, estLayers); ctx = 4096; }}
  }}
  // Clamp to real layer count
  if (gpu_layers !== 999) gpu_layers = Math.min(gpu_layers, estLayers);
  // Other load options tuned with context
  threads = null;
  flash_attn = ctx >= 8192;
  cache_k = null; cache_v = null;
  if (ctx > 16384) {{ cache_k = 'q8_0'; cache_v = 'q8_0'; }}
  return {{ gpu_layers, context: ctx, ctx_size: ctx, threads, flash_attn, cache_type_k: cache_k, cache_type_v: cache_v, block_count: estLayers, gguf_metadata: meta }};
}}
async function fillGpuDevices() {{
  const sel = document.getElementById('eng-devices');
  if (!sel || sel.dataset.filled === '1') return;
  try {{
    const st = await api('/v1/server/status');
    const devs = (st.hardware && st.hardware.devices) || [];
    for (const d of devs) {{
      const opt = document.createElement('option');
      opt.value = String(d.index);
      opt.textContent = `${{d.index}}: ${{d.name || d.bdf || 'gpu'}} ${{d.free_gb ?? '?'}}GB free${{d.display ? ' (display)' : ''}}`;
      sel.appendChild(opt);
    }}
    // say which card "auto" will take
    const spare = devs.find(d => !d.display);
    const auto = sel.querySelector('option[value=""]');
    if (auto && spare) {{
      auto.textContent = `auto (the spare card: ${{spare.index}})`;
      auto.title = `GPU ${{spare.index}} is not the display, so the desktop keeps its own VRAM`;
    }}
    sel.dataset.filled = '1';
  }} catch (e) {{}}
}}
async function engineAuto() {{
  const btn = document.getElementById('eng-auto');
  const msg = document.getElementById('eng-auto-msg');
  const selFile = window._selectedFile || (document.getElementById('launch-model-select') ? document.getElementById('launch-model-select').value : '');
  if (!selFile) {{
    if (msg) msg.textContent = 'select a model first';
    setTimeout(()=>{{ if(msg) msg.textContent=''; }}, 2500);
    return;
  }}
  if (btn) btn.disabled = true;
  if (msg) msg.textContent = 'computing preset…';
  try {{
    // Prefer server-side preset API, fallback to client-side hardware/size_gb/gguf-metadata
    let preset = null;
    try {{
      const data = await api('/v1/engines/preset?file=' + encodeURIComponent(selFile));
      preset = data.load_options || data.preset || data;
      if (preset.gpu_layers == null && preset.gpu_layers !== 0) throw new Error('no preset');
    }} catch (_e) {{
      // Client-side: use cached _libCache if fresh, else fetch
      let local;
      if(_libCache.system && (Date.now()-_libCache.ts < 30000)) local={{models: _libCache.system}};
      else local=await api('/v1/models/local');
      const status=await api('/v1/server/status');
      const hardware = status.hardware || status;
      const entry = (local.models||[]).find(m=>m.file===selFile);
      const size_gb = entry ? Number(entry.size_gb)||0 : 0;
      const gguf_metadata = entry ? (entry.gguf_metadata || entry.metadata || {{}}) : {{}};
      preset = computeAutoPreset(size_gb, hardware, selFile, gguf_metadata);
    }}
    const ctxVal = preset.context ?? preset.ctx_size ?? 8192;
    const nglVal = preset.gpu_layers ?? 999;
    // Update Engine profile load_options and display
    const idx = currentEngineIndex();
    if (engines[idx]) {{
      engines[idx].load_options = {{ ...(engines[idx].load_options||{{}}), context: ctxVal, gpu_layers: nglVal }};
      if (preset.threads) engines[idx].load_options.threads = preset.threads;
      if (preset.flash_attn) engines[idx].load_options.flash_attn = true;
      if (preset.cache_type_k) engines[idx].load_options.cache_type_k = preset.cache_type_k;
      if (preset.cache_type_v) engines[idx].load_options.cache_type_v = preset.cache_type_v;
      show('eng-loadopts', engines[idx].load_options);
    }}
    // Directly set values in the form (Launch panel + Engine fit slider) and then save
    const ctxInput = document.getElementById('ctx');
    const nglInput = document.getElementById('ngl');
    if (ctxInput) ctxInput.value = String(ctxVal);
    if (nglInput) nglInput.value = String(nglVal);
    const fitCtx = document.getElementById('fit-ctx');
    if (fitCtx) {{ fitCtx.value = String(ctxVal); try{{ updateFit(); }}catch(e){{}} }}
    // Optional: reflect other load options in Launch panel
    if (preset.threads != null) {{ const el=document.getElementById('l-threads'); if(el) el.value=preset.threads; }}
    if (preset.flash_attn != null) {{ const el=document.getElementById('l-fa'); if(el) el.checked=!!preset.flash_attn; }}
    if (preset.cache_type_k) {{ const el=document.getElementById('l-ctk'); if(el) el.value=preset.cache_type_k; }}
    if (preset.cache_type_v) {{ const el=document.getElementById('l-ctv'); if(el) el.value=preset.cache_type_v; }}
    engDirty = true;
    if (msg) msg.textContent = `preset: ${{nglVal}} layers / ${{ctxVal}} ctx — saving…`;
    await saveEngines();
    if (msg) msg.textContent = `Auto: ${{nglVal}} layers / ${{ctxVal}} ctx — saved`;
  }} catch (e) {{
    if (msg) msg.textContent = 'auto failed: ' + e.message;
  }} finally {{
    if (btn) btn.disabled = false;
    setTimeout(()=>{{ if(msg && msg.textContent.startsWith('Auto:')) msg.textContent=''; }}, 4000);
  }}
}}

async function exportEngineProfiles() {{
  const msg = document.getElementById('import-export-msg');
  try {{
    const data = {{ engines, exported_at: new Date().toISOString(), version: 1 }};
    const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'engine-profiles.json';
    a.click();
    URL.revokeObjectURL(url);
    if (msg) msg.textContent = `exported ${{engines.length}} profiles`;
  }} catch (e) {{
    if (msg) msg.textContent = 'export failed: ' + e.message;
  }} finally {{
    setTimeout(()=>{{ if(msg) msg.textContent=''; }}, 3000);
  }}
}}

async function importEngineProfiles(input) {{
  const file = input.files[0];
  if (!file) return;
  const msg = document.getElementById('import-export-msg');
  try {{
    const text = await file.text();
    const data = JSON.parse(text);
    const incoming = data.engines || data.profiles || data;
    if (!Array.isArray(incoming)) throw new Error('invalid bundle: missing engines array');
    for (const eng of incoming) {{
      if (!eng.name) throw new Error('engine missing name');
      const idx = engines.findIndex(e => e.name === eng.name);
      if (idx >= 0) engines[idx] = eng;
      else engines.push(eng);
    }}
    renderEngineSelect(0);
    engineSelected();
    if (msg) msg.textContent = `imported ${{incoming.length}} profiles — save to persist`;
  }} catch (e) {{
    if (msg) msg.textContent = 'import failed: ' + e.message;
  }} finally {{
    input.value = '';
    setTimeout(()=>{{ if(msg) msg.textContent=''; }}, 4000);
  }}
}}

/* ---------------- Fit estimator S2: VRAM needs vs has + context slider ---------------- */
let _fitHardware = null;
let _fitHardwarePromise = null;
let _fitModelsCache = null;
async function getFitHardware() {{
  if (_fitHardware) return _fitHardware;
  if (_fitHardwarePromise) return _fitHardwarePromise;
  _fitHardwarePromise = api('/v1/server/status').then(s => {{
    const hw = s.hardware || s;
    const has = hw.available_gb ?? hw.available_ram_gb ?? hw.total_ram_gb ?? hw.totalRamGb ?? 8;
    const total = hw.total_ram_gb ?? hw.totalRamGb ?? has;
    return {{ available_gb: Number(has) || 8, total_ram_gb: Number(total) || 8, vram_gb: hw.vram_gb ?? null, devices: hw.devices || [] }};
  }}).catch(()=>({{available_gb:8, total_ram_gb:8, vram_gb:null, devices:[]}})).then(h=>{{_fitHardware=h; return h;}});
  return _fitHardwarePromise;
}}
async function getFitModels() {{
  if (_fitModelsCache && Date.now()-_fitModelsCache.ts < 5000) return _fitModelsCache.data;
  // Use _libCache if fresh (avoid extra fetch)
  if(_libCache.system && (Date.now()-_libCache.ts < 30000)) {{
    const data={{models_dir: document.getElementById('library-path')?.value||'', models: _libCache.system}};
    _fitModelsCache={{data, ts: Date.now()}};
    return data;
  }}
  const data = await api('/v1/models/local');
  _fitModelsCache = {{data, ts: Date.now()}};
  return data;
}}
function formatCtx(v) {{
  const n = Number(v);
  if (n>=1024) return (n/1024)|0 + 'k';
  return String(n);
}}
function updateEngModelDisplay() {{
  const sel = document.getElementById('eng-model');
  const disp = document.getElementById('eng-model-display');
  const note = document.getElementById('eng-model-note');
  if (!sel) return;
  const file = sel.value;
  if (disp) disp.textContent = file ? file.split('/').pop().split('\\\\').pop() : '—';
  if (!note) return;
  const local = (window._fitModels && window._fitModels.models) || [];
  const entry = local.find(m => m.file === file);
  if (!entry) {{ note.textContent = ''; return; }}
  const bits = [];
  if (entry.size_gb != null) bits.push(entry.size_gb + ' GB');
  if (entry.architecture) bits.push(entry.architecture);
  if (entry.quantization) bits.push(entry.quantization);
  if (entry.context_length) bits.push(entry.context_length + ' ctx');
  note.textContent = bits.join(' · ');
}}

async function updateFit() {{
  const selFile = window._selectedFile || (document.getElementById('launch-model-select') ? document.getElementById('launch-model-select').value : '');
  const box = document.getElementById('eng-fit');
  const ctxEl = document.getElementById('fit-ctx');
  if (!box || !ctxEl) return;
  if (!selFile) {{ box.style.display='none'; return; }}
  box.style.display='';
  let size_gb = 0;
  try {{
    const local = await getFitModels();
    window._fitModels = local;
    const entry = (local.models||[]).find(m=>m.file===selFile);
    if (entry && entry.size_gb!=null) size_gb = Number(entry.size_gb)||0;
  }} catch(e) {{}}
  updateEngModelDisplay();
  const ctx = parseInt(ctxEl.value,10)||8192;
  const label = document.getElementById('fit-ctx-label');
  if (label) label.textContent = formatCtx(ctx);
  let hw;
  try {{ hw = await getFitHardware(); }} catch(e) {{ hw = {{available_gb:8}}; }}
  const has_gb = Number(hw.available_gb)||8;
  // the server owns the estimate: the tiers (VRAM, RAM, disk), the disk
  // guard and the context that fits
  let plan = null;
  try {{
    const kv = document.getElementById('eng-ctk')?.value || 'q8_0';
    plan = await api('/v1/server/fit?model_gb=' + encodeURIComponent(size_gb) +
                     '&ctx=' + encodeURIComponent(ctx) + '&kv=' + encodeURIComponent(kv));
  }} catch (e) {{ plan = null; }}
  const kv_gb = plan ? plan.metrics.kv_gb : (ctx/1024)*0.07;
  const needs_gb = plan ? plan.metrics.total_gb : size_gb + kv_gb;
  const ratio = has_gb>0 ? needs_gb/has_gb : 1;
  const needsEl = document.getElementById('fit-needs');
  const hasEl = document.getElementById('fit-has');
  const needsVal = document.getElementById('fit-needs-val');
  const pctEl = document.getElementById('fit-pct');
  const fill = document.getElementById('fit-fill');
  const btn = document.getElementById('eng-load-fit');
  if (needsEl) needsEl.textContent = `needs ${{needs_gb.toFixed(1)}}GB`;
  if (hasEl) hasEl.textContent = `/ ${{has_gb.toFixed(1)}}GB`;
  if (needsVal) needsVal.textContent = `${{needs_gb.toFixed(1)}}GB`;
  if (pctEl) {{
    const fits = needs_gb <= has_gb;
    pctEl.textContent = fits ? (ratio>0.9 ? 'Tight' : 'Fits') : 'Too large';
    pctEl.className = 'fit-pct ' + (fits ? (ratio>0.7 ? 'warn' : 'ok') : 'bad');
  }}
  if (fill) {{
    fill.style.width = Math.min(100, ratio*100).toFixed(1)+'%';
    fill.className = 'fit-fill ' + (ratio>1 ? 'red' : ratio>0.7 ? 'amber' : 'green');
  }}
  const planEl = document.getElementById('fit-plan');
  if (planEl) {{
    if (!plan) {{
      planEl.textContent = '';
    }} else {{
      const m = plan.metrics, f = plan.flags;
      const tiers = `vram ${{m.tier1_vram_gb.toFixed(1)}}GB · ram ${{m.tier2_ram_gb.toFixed(1)}}GB · disk ${{m.tier3_disk_gb.toFixed(1)}}GB`;
      const notes = [];
      if (f.recommend_cap) notes.push('fits up to ' + formatCtx(f.recommend_cap));
      if (f.io_latency_warning) notes.push('the disk serves part of it - long prompts get slower');
      if (f.disk_full) notes.push('the disk is nearly full');
      planEl.textContent = tiers + (notes.length ? ' — ' + notes.join('; ') : '');
      planEl.style.color = f.disk_full ? '#FF9B9B' : (f.io_latency_warning ? '#FFD54F' : '#FFDD00');
    }}
  }}
  if (btn) {{
    const fits = needs_gb <= has_gb;
    btn.disabled = !fits;
    btn.title = fits ? `Fits — load ${{selFile.split('/').pop().split('\\\\').pop()}} at ${{formatCtx(ctx)}}` : `needs ${{needs_gb.toFixed(1)}}GB but you have ${{has_gb.toFixed(1)}}GB — reduce context or free VRAM`;
  }}
}}
function engineLoadFromFit() {{
  const selFile = window._selectedFile || (document.getElementById('launch-model-select') ? document.getElementById('launch-model-select').value : '');
  const ctx = parseInt(document.getElementById('fit-ctx')?.value||'8192',10);
  if (!selFile) return;
  const modelInput = document.getElementById('eng-model');
  const ctxInput = document.getElementById('ctx');
  if (modelInput) modelInput.value = selFile;
  if (ctxInput) ctxInput.value = String(ctx);
  const sel = document.getElementById('launch-model-select');
  if (sel) sel.value = selFile;
  startServer(document.querySelector('button[onclick="startServer(this)"]') || null);
}}
(function initFitHooks(){{
  const ctxEl = document.getElementById('fit-ctx');
  if (ctxEl) ctxEl.addEventListener('input', ()=>{{ updateFit(); }});
  let lastSel = null;
  setInterval(()=>{{
    const cur = window._selectedFile || (document.getElementById('launch-model-select') ? document.getElementById('launch-model-select').value : '');
    if (cur !== lastSel) {{ lastSel = cur; updateFit(); }}
  }}, 400);
  const launchSel = document.getElementById('launch-model-select');
  if (launchSel) launchSel.addEventListener('change', ()=>{{ setTimeout(updateFit, 50); }});
  setTimeout(()=>{{ getFitHardware().then(()=>updateFit()); }}, 600);
}})();

// Library tab loads on demand — not on initial refresh to keep UI responsive
console.log('studio ready, tabs active');
document.getElementById('chatin').addEventListener('blur',
  () => setTimeout(() => {{ document.getElementById('sug-chat').innerHTML = ''; }}, 150));
document.getElementById('chatin').focus();
// Refresh status when tab becomes visible again after idle (library only on click)
document.addEventListener('visibilitychange', ()=>{{
  if(document.visibilityState==='visible'){{
    refresh().catch(e=>console.error(e));
    refreshProcesses().catch(e=>console.error(e));
  }}
}});
</script>
</body></html>"""


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
