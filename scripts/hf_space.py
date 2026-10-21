"""Render and publish the HF Space page for the cascade paper.

Six tabs (the memory, the move, the controller, what is measured, the
checker, the paper). Above the tab's content sits a **stage**: the
example for that tab, which changes with it. The demos are
dependency-free, respect ``prefers-reduced-motion``, and the page works
without JavaScript (all panels are present, the tabs are anchors).

A static Space serves ``index.html``; the ``README.md`` is only the repo
card. This builds both from ``docs/CASCADE-PAPER.md``.

    python scripts/hf_space.py            # render to /tmp/opencode/hf-space
    python scripts/hf_space.py --upload   # publish to the Space
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPACE = "SkyIsNotGreen/hive-cascade-paper"
OUT = Path("/tmp/opencode/hf-space")
GH = "https://github.com/sky-is-green/hivebench"

CSS = """
:root {
  --bg: #fbfaf8; --fg: #171614; --muted: #6b675f; --line: #e4e0d8;
  --card: #ffffff; --accent: #b45309; --accent2: #0f766e; --code: #f3efe7;
  --bad: #b91c1c; --good: #15803d; --hero: linear-gradient(160deg, #fef3c7 0%, #fbfaf8 70%);
  --c-prompt: #2563eb; --c-model: #b45309; --c-judge: #7c3aed;
  --c-mem: #0f766e; --c-exec: #64748b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12110f; --fg: #ece8e1; --muted: #a29c90; --line: #2c2a26;
    --card: #1a1815; --accent: #fbbf24; --accent2: #5eead4; --code: #22201c;
    --bad: #f87171; --good: #4ade80; --hero: linear-gradient(160deg, #2a2113 0%, #12110f 70%);
    --c-prompt: #60a5fa; --c-model: #fbbf24; --c-judge: #c084fc;
    --c-mem: #5eead4; --c-exec: #94a3b8;
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 17px/1.65 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 920px; margin: 0 auto; padding: 0 24px; }

.hero { background: var(--hero); border-bottom: 1px solid var(--line); padding: 56px 0 34px; }
.eyebrow { color: var(--accent); font-size: 13px; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase; }
.eyebrow a { color: inherit; text-decoration: none;
  border-bottom: 1px dotted color-mix(in srgb, var(--accent) 55%, transparent); }
.eyebrow a:hover { border-bottom-style: solid; }
.hero h1 { font-size: clamp(2.1rem, 5vw, 3.2rem); line-height: 1.08;
  letter-spacing: -.03em; margin: .3em 0 .2em; }
.lede { font-size: 1.16rem; max-width: 48ch; color: var(--muted); margin: 0; }
.meta { font-size: .84rem; color: var(--muted); margin-top: 1.1rem; }
.meta .dot { display: inline-block; width: .55em; height: .55em; background: var(--good);
  border-radius: 50%; margin-right: .45em; vertical-align: .06em; }
.flow { display: flex; align-items: center; gap: 6px; margin-top: 20px; flex-wrap: wrap;
  font-size: .78rem; }
.flow .node { border: 1px solid var(--line); background: var(--card); border-radius: 8px;
  padding: 5px 9px; font-family: ui-monospace, monospace; }
.flow .node.hot { border-color: var(--accent); color: var(--accent); }
.flow .link { flex: 1 1 40px; min-width: 30px; height: 2px; background: var(--line);
  position: relative; overflow: hidden; }
.flow .link i { position: absolute; top: -2px; left: -10px; width: 8px; height: 6px;
  border-radius: 3px; background: var(--accent); animation: travel 3.2s linear infinite; }
.flow .link:nth-of-type(4) i { animation-delay: .8s; }
.flow .link:nth-of-type(6) i { animation-delay: 1.6s; }
.flow .link:nth-of-type(8) i { animation-delay: 2.4s; }
@keyframes travel { 0% { left: -10px; opacity: 0; } 12% { opacity: 1; }
  88% { opacity: 1; } 100% { left: 100%; opacity: 0; } }

.tabs { position: sticky; top: 0; z-index: 20;
  background: color-mix(in srgb, var(--bg) 90%, transparent);
  backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); }
.tabs .wrap { display: flex; gap: 4px; overflow-x: auto; padding-top: 10px; padding-bottom: 10px; }
.tabs a { padding: .38rem .72rem; border-radius: 999px; color: var(--muted);
  text-decoration: none; font-size: .82rem; white-space: nowrap; }
.tabs a:hover { color: var(--fg); }
.tabs a[aria-selected="true"] { background: var(--accent); color: #fff; }
@media (prefers-color-scheme: dark) { .tabs a[aria-selected="true"] { color: #1a1815; } }

.stage-wrap { padding: 22px 0 6px; }
.stage { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 16px 18px 18px; }
.stage-head { font-size: .72rem; text-transform: uppercase; letter-spacing: .09em;
  color: var(--muted); margin-bottom: 10px; }
.stage-head strong { color: var(--accent); font-weight: 700; }
.stage-panel[hidden] { display: none; }
.tab-panel[hidden] { display: none; }
main { padding: 10px 0 24px; }
h2 { font-size: 1.45rem; letter-spacing: -.01em; margin: 2rem 0 .5rem;
  padding-bottom: .3rem; border-bottom: 1px solid var(--line); }
h3 { font-size: 1.08rem; margin: 1.4rem 0 .3rem; }
p { margin: .75rem 0; }
a { color: var(--accent); text-underline-offset: 3px; }
.sub { color: var(--muted); margin-top: -.1rem; }
.small { font-size: .85rem; color: var(--muted); }
table { width: 100%; border-collapse: collapse; margin: .9rem 0 1.2rem;
  font-size: .91rem; display: block; overflow-x: auto; }
th, td { text-align: left; padding: .5rem .65rem; border-bottom: 1px solid var(--line);
  vertical-align: top; }
th { color: var(--muted); font-size: .74rem; text-transform: uppercase;
  letter-spacing: .06em; font-weight: 600; }
code { background: var(--code); padding: .12em .38em; border-radius: 4px;
  font-size: .9em; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
pre { background: var(--code); padding: 1rem 1.1rem; border-radius: 8px;
  overflow-x: auto; font-size: .85rem; line-height: 1.5; }
pre code { background: none; padding: 0; }
a.ref { text-decoration: none; border-bottom: 1px dotted var(--accent); }
a.ref code { color: inherit; }
blockquote { margin: 1.1rem 0; padding: .1rem 0 .1rem 1rem;
  border-left: 3px solid var(--accent); color: var(--muted); }
hr { border: 0; border-top: 1px solid var(--line); margin: 2.2rem 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px; margin: 1rem 0 1.2rem; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 16px 12px; }
.card h3 { margin: 0 0 .25rem; font-size: .95rem; }
.card p { margin: 0; font-size: .87rem; color: var(--muted); }
.note { background: var(--code); border-left: 3px solid var(--accent);
  padding: .75rem .95rem; border-radius: 0 6px 6px 0; font-size: .89rem; }
.note.good { border-left-color: var(--good); }
.pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px;
  padding: .08rem .5rem; font-size: .74rem; color: var(--muted); margin-right: .3rem; }

/* stage demos */
.demo-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.demo-head h3 { margin: 0; }
button { font: inherit; font-size: .78rem; padding: .25rem .65rem; border-radius: 999px;
  border: 1px solid var(--line); background: var(--bg); color: var(--fg); cursor: pointer; }
button:hover { border-color: var(--accent); color: var(--accent); }
.caption { min-height: 2.9em; margin: .7rem 0 0; font-size: .88rem; color: var(--muted); }
.caption strong { color: var(--fg); }
.chunks { display: grid; gap: 6px; margin-top: 10px; }
.chunk { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center;
  border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font-size: .84rem;
  transition: opacity .4s, border-color .4s; background: var(--bg); }
.chunk .tag { font-size: .68rem; color: var(--muted); font-family: ui-monospace, monospace; }
.chunk.merged { opacity: .28; border-style: dashed; }
.chunk.kept { border-color: var(--good); }
.chunk.dimmed { opacity: .4; border-color: var(--accent); }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 999px;
  overflow: hidden; }
.seg button { border: 0; border-radius: 0; padding: .24rem .62rem;
  background: transparent; color: var(--muted); }
.seg button[aria-pressed="true"] { background: var(--accent); color: #fff; }
@media (prefers-color-scheme: dark) { .seg button[aria-pressed="true"] { color: #1a1815; } }
.ledger { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 10px; }
.fact { font-size: .7rem; font-family: ui-monospace, monospace;
  border: 1px solid var(--line); border-radius: 999px; padding: .08rem .48rem;
  color: var(--good); transition: color .3s, border-color .3s, opacity .3s; }
.fact.lost { color: var(--bad); border-color: var(--bad);
  text-decoration: line-through; opacity: .8; }
.budget { height: 8px; background: var(--line); border-radius: 999px; overflow: hidden;
  margin-top: 10px; }
.budget i { display: block; height: 100%; background: var(--accent2); width: 100%;
  transition: width .7s ease; }
.route { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-top: 12px; }
@media (max-width: 760px) { .route { grid-template-columns: 1fr; } }
.station { border: 1px solid var(--line); border-radius: 8px; padding: 8px; min-height: 72px;
  background: var(--bg); transition: border-color .3s, box-shadow .3s; }
.station .name { font-family: ui-monospace, monospace; font-size: .72rem; color: var(--muted); }
.station .val { font-size: .92rem; margin-top: 4px; }
.station.active { border-color: var(--accent);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 18%, transparent); }
.station.pass { border-color: var(--good); }
.station.fail { border-color: var(--bad); }
.q { font-size: .84rem; color: var(--muted); border-left: 2px solid var(--line);
  padding-left: 9px; margin: 8px 0 0; }
.map { display: grid; gap: 5px; }
.map-row { display: grid; grid-template-columns: 1fr 22px 1fr; align-items: center;
  gap: 8px; padding: 6px 9px; border-radius: 7px; font-size: .85rem;
  transition: background .35s; }
.map-row .arrow { text-align: center; color: var(--muted); }
.map-row.active { background: var(--code); }
.bars { display: grid; gap: 7px; margin-top: 10px; }
.bar-row { display: grid; grid-template-columns: 96px 1fr 58px; gap: 9px;
  align-items: center; font-size: .84rem; }
.bar { height: 9px; background: var(--line); border-radius: 999px; overflow: hidden; }
.bar i { display: block; height: 100%; width: 0; background: var(--accent2);
  border-radius: 999px; transition: width .85s ease; }
.bars.animate .bar i { width: var(--v); }
.bar-val { font-family: ui-monospace, monospace; color: var(--muted); font-size: .78rem; }
.runs { display: grid; gap: 5px; margin: 9px 0; }
.run { display: flex; justify-content: space-between; gap: 10px; border: 1px solid var(--line);
  border-radius: 7px; padding: 6px 10px; font-size: .83rem; }
.verdict { font-family: ui-monospace, monospace; }
.verdict.pass { color: var(--good); } .verdict.fail { color: var(--bad); }
.outcome { font-size: .86rem; padding: .55rem .75rem; border-radius: 7px; background: var(--code); }
.flowmap { width: 100%; height: auto; display: block; margin: 4px 0 2px; }
.flowmap .edge { fill: none; stroke: var(--line); stroke-width: 2;
  transition: stroke .25s, opacity .25s; }
.flowmap .edge.dashed { stroke-dasharray: 5 5; }
.flowmap .edge.active { stroke-width: 3; }
.flowmap .edge.struck { stroke: var(--bad) !important; opacity: .3; }
.flowmap .node rect { fill: var(--bg); stroke: var(--line); stroke-width: 1.4;
  transition: stroke .25s, stroke-width .25s; }
.flowmap .node text { font: 600 12px ui-monospace, SFMono-Regular, Menlo, monospace;
  text-anchor: middle; fill: var(--fg); }
.flowmap .node .val { font-size: 11px; font-weight: 500; fill: var(--muted);
  opacity: .2; transition: opacity .3s; }
.flowmap .node.revealed .val { opacity: 1; }
.flowmap .node .val.fail { fill: var(--bad); }
.flowmap .node .val.pass { fill: var(--good); }
.flowmap .node.active rect { stroke: var(--accent); stroke-width: 2.4; }
.flowmap .node.fail rect { stroke: var(--bad); }
.flowmap .node.pass rect { stroke: var(--good); }
.flowmap .node[data-node="mem"] text { fill: var(--c-mem); }
.flowmap .node[data-node="prompt"] text { fill: var(--c-prompt); }
.flowmap .node[data-node="35b"] text,
.flowmap .node[data-node="27b"] text { fill: var(--c-model); }
.flowmap .node[data-node="judge"] text { fill: var(--c-judge); }
.flowmap .node[data-node="sandbox"] text { fill: var(--c-exec); }
.legend { display: flex; gap: 12px; flex-wrap: wrap; font-size: .72rem;
  color: var(--muted); margin-top: 4px; }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.legend i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.paper-card { border-left: 3px solid var(--accent); padding: .1rem 0 .1rem 1rem; }
.paper-card h3 { margin: 0 0 .3rem; font-size: 1.15rem; }
.paper-card p { color: var(--muted); }
footer { border-top: 1px solid var(--line); padding: 24px 0 44px; color: var(--muted);
  font-size: .85rem; }
footer a { color: inherit; }
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
}
"""

HERO = """
<header class="hero"><div class="wrap">
  <div class="eyebrow"><a href="https://github.com/sky-is-green/hivebench">HiveBench Studio</a> · research preview</div>
  <h1>The Hive Cascade</h1>
  <p class="lede">One loop, two levels: curate the memory, and route the
    harness. The controller is the current work.</p>
  <p class="meta"><span class="dot"></span>not a product yet ·
    <span class="pill">MIT</span><span class="pill">updated 2026-10-21</span></p>
  <div class="flow" aria-hidden="true">
    <span class="node">35B</span><span class="link"><i></i></span>
    <span class="node hot">verify</span><span class="link"><i></i></span>
    <span class="node">judge</span><span class="link"><i></i></span>
    <span class="node">27B</span><span class="link"><i></i></span>
    <span class="node">API</span>
  </div>
</div></header>
"""

TABS = """
<nav class="tabs" aria-label="sections"><div class="wrap" role="tablist">
  <a role="tab" href="#memory" data-tab="memory" aria-selected="true">the memory</a>
  <a role="tab" href="#move" data-tab="move" aria-selected="false">the move</a>
  <a role="tab" href="#controller" data-tab="controller" aria-selected="false">the controller</a>
  <a role="tab" href="#measured" data-tab="measured" aria-selected="false">what is measured</a>
  <a role="tab" href="#checker" data-tab="checker" aria-selected="false">the checker</a>
  <a role="tab" href="#paper" data-tab="paper" aria-selected="false">the paper</a>
</div></nav>
"""

STAGE_OPEN = """
<section class="stage-wrap"><div class="wrap"><div class="stage">
  <div class="stage-head">example · <strong id="stage-name">the memory</strong></div>
"""

STAGE_MEMORY = """
  <div class="stage-panel" data-stage="memory">
    <div class="demo-head"><h3>The same pair, two thresholds</h3>
      <div class="seg" role="group" aria-label="dedup threshold">
        <button type="button" data-threshold="0.92" aria-pressed="true">shipped 0.92</button>
        <button type="button" data-threshold="0.67" aria-pressed="false">tuned 0.67</button>
      </div></div>
    <div class="chunks">
      <div class="chunk" data-near="keep"><span>Order schema: 3NF with a denormalized read model.</span><span class="tag">fact A</span></div>
      <div class="chunk" data-near="merge"><span>Order schema: 3NF with a denormalized read model, soft deletes via deleted_at.</span><span class="tag">fact A + fact D</span></div>
      <div class="chunk"><span>Indexes: composite on (customer_id, created_at).</span><span class="tag">fact B</span></div>
      <div class="chunk"><span>Migrations: Alembic, forward-only.</span><span class="tag">fact C</span></div>
      <div class="chunk dimmed"><span>Kubernetes deploys the sidecar on port 8765.</span><span class="tag">other topic — penalised, never deleted</span></div>
    </div>
    <div class="ledger" aria-label="facts kept">
      <span class="fact" data-fact="A">A</span>
      <span class="fact" data-fact="B">B</span>
      <span class="fact" data-fact="C">C</span>
      <span class="fact" data-fact="D">D · soft deletes</span>
    </div>
    <div class="budget" title="context budget"><i></i></div>
    <p class="caption" aria-live="polite"></p>
  </div>
"""

STAGE_MOVE = """
  <div class="stage-panel" data-stage="move" hidden>
    <div class="map">
      <div class="map-row" data-i="0"><span>the router scores every chunk</span><span class="arrow">→</span><span><code>ask(model, prompt)</code></span></div>
      <div class="map-row" data-i="1"><span>dedup and drift decide</span><span class="arrow">→</span><span><code>verify</code> decides</span></div>
      <div class="map-row" data-i="2"><span>the judge is a readout</span><span class="arrow">→</span><span><code>score</code> is the fallback</span></div>
      <div class="map-row" data-i="3"><span>a false merge deletes a fact</span><span class="arrow">→</span><span>a false accept ships a wrong answer</span></div>
    </div>
    <p class="caption" data-move-caption aria-live="polite"></p>
  </div>
"""

STAGE_CONTROLLER = """
  <div class="stage-panel" data-stage="controller" hidden>
    <div class="demo-head"><h3>Prompt to execution, one task</h3>
      <button type="button" data-route-play>Play</button></div>
    <svg class="flowmap" viewBox="0 0 880 330" role="img"
         aria-label="The routed task: prompt and memory into the 35B, the candidate to the judge and to execution, the failure to the 27B, the new candidate to the checker, the answer.">
      <path id="e-mem" class="edge dashed" d="M120,45 C150,60 150,110 165,130"/>
      <path id="e-prompt" class="edge" d="M120,162 L165,162"/>
      <path id="e-35b-judge" class="edge" d="M275,135 C300,120 310,70 330,55"/>
      <path id="e-35b-verify" class="edge" d="M275,160 C300,175 310,205 330,215"/>
      <path id="e-judge-answer" class="edge dashed" d="M440,50 L720,50"/>
      <path id="e-verify-loop" class="edge dashed" d="M391,246 L391,294 L379,294 L379,246"/>
      <path id="e-verify-27b" class="edge" d="M440,218 L500,218"/>
      <path id="e-27b-verify2" class="edge" d="M600,218 L660,218"/>
      <path id="e-verify2-answer" class="edge" d="M770,200 C800,160 795,100 785,75"/>
      <circle id="packet" r="6" fill="#3b82f6" opacity="0"/>
      <g class="node" data-node="mem"><rect x="10" y="20" width="110" height="38" rx="8"/><text x="65" y="43">memory</text></g>
      <g class="node" data-node="prompt"><rect x="10" y="140" width="110" height="44" rx="8"/><text x="65" y="167">prompt</text></g>
      <g class="node" data-node="35b"><rect x="165" y="115" width="110" height="64" rx="8"/><text x="220" y="140">35B</text><text class="val" x="220" y="160" data-val="35b">24.50</text></g>
      <g class="node" data-node="judge"><rect x="330" y="25" width="110" height="50" rx="8"/><text x="385" y="45">judge</text><text class="val" x="385" y="62" data-val="judge">p = 0.973</text></g>
      <g class="node" data-node="answer"><rect x="720" y="25" width="130" height="50" rx="8"/><text x="785" y="45">answer</text><text class="val pass" x="785" y="62" data-val="answer">accepted</text></g>
      <g class="node" data-node="verify1"><rect x="330" y="190" width="110" height="56" rx="8"/><text x="385" y="211">verify</text><text class="val fail" x="385" y="230" data-val="verify1">FAIL</text></g>
      <g class="node" data-node="sandbox"><rect x="330" y="275" width="110" height="38" rx="8"/><text x="385" y="298">execution</text></g>
      <g class="node" data-node="27b"><rect x="500" y="190" width="100" height="56" rx="8"/><text x="550" y="211">27B</text><text class="val" x="550" y="230" data-val="27b">245</text></g>
      <g class="node" data-node="verify2"><rect x="660" y="190" width="110" height="56" rx="8"/><text x="715" y="211">verify</text><text class="val pass" x="715" y="230" data-val="verify2">PASS</text></g>
    </svg>
    <div class="legend" aria-hidden="true">
      <span><i style="background: var(--c-prompt)"></i>prompt</span>
      <span><i style="background: var(--c-model)"></i>model output</span>
      <span><i style="background: var(--c-judge)"></i>judgement</span>
      <span><i style="background: var(--bad)"></i>failed check</span>
      <span><i style="background: var(--good)"></i>passed check</span>
      <span><i style="background: var(--c-mem)"></i>memory / context</span>
      <span><i style="background: var(--c-exec)"></i>execution</span>
    </div>
    <p class="caption" data-route-caption aria-live="polite"></p>
  </div>
"""

STAGE_MEASURED = """
  <div class="stage-panel" data-stage="measured" hidden>
    <div class="bars" data-bars>
      <div class="bar-row"><span>35B alone</span><div class="bar"><i style="--v:75%"></i></div><span class="bar-val">21/28</span></div>
      <div class="bar-row"><span>judge gate + 27B</span><div class="bar"><i style="--v:82%"></i></div><span class="bar-val">23/28</span></div>
      <div class="bar-row"><span>27B alone</span><div class="bar"><i style="--v:93%"></i></div><span class="bar-val">26/28</span></div>
      <div class="bar-row"><span>oracle gate + 27B</span><div class="bar"><i style="--v:96%"></i></div><span class="bar-val">27/28</span></div>
    </div>
    <p class="caption">The same 28 tasks. The judge alone sends up 3 of 7
      errors and 0 of 21 correct answers; the oracle gate is the ceiling.</p>
  </div>
"""

STAGE_CHECKER = """
  <div class="stage-panel" data-stage="checker" hidden>
    <div class="demo-head"><h3>Is the check falsifiable?</h3>
      <div class="seg" role="group" aria-label="check quality">
        <button type="button" data-check="real" aria-pressed="true">a real check</button>
        <button type="button" data-check="vacuous" aria-pressed="false">print("PASS")</button>
      </div></div>
    <div class="runs">
      <div class="run"><span>on the candidate <code>245</code></span><span class="verdict pass" data-run-a>PASS</span></div>
      <div class="run"><span>on the mutation <code>246</code></span><span class="verdict fail" data-run-b>FAIL</span></div>
    </div>
    <div class="outcome" data-outcome>accepted — the check discriminates between the answer and its mutation</div>
    <p class="caption">A check that passes the mutation is not evidence. It is
      <code>unknown</code>, and the controller must not accept on it.</p>
  </div>
"""

STAGE_PAPER = """
  <div class="stage-panel" data-stage="paper" hidden>
    <div class="paper-card">
      <h3>The Hive Cascade: detect → repair → escalate</h3>
      <p>A gold-free verification-and-repair loop for local language models.
        A model that is wrong is rarely uncertain - it is confidently wrong,
        and the confidence is indistinguishable from a right answer's.</p>
    </div>
    <p class="caption">The full text is below. <a href="https://github.com/sky-is-green/hivebench/blob/main/docs/CASCADE-PAPER.md">Also in the repository.</a></p>
  </div>
"""

STAGE_CLOSE = "</div></div></section>\n"

PANEL_MEMORY = """
<section class="tab-panel" data-panel="memory" id="memory" role="tabpanel">
  <div class="wrap">
    <h2>The memory: what the drone learned</h2>
    <p>The first harness curated <em>within</em> a conversation. A 17M
      encoder - the drone - scores every stored chunk; the membrane makes two
      decisions on every turn: <strong>dedup</strong> (is this chunk already
      said?) and <strong>drift</strong> (did the topic change?), and the
      assembler keeps the context under budget.</p>
    <p>The stage above is the argument for the conservative dedup. At 0.92
      both schema chunks survive and the soft-delete fact survives with them.
      At 0.67 the near-duplicate merges and the fact goes. The drift chunk
      dims in both states: it is only penalised, never deleted.</p>
    <div class="grid">
      <div class="card"><h3>the shipped point</h3><p><strong>60.51%</strong> of
        the facts kept in <strong>374</strong> tokens. Precision 1.00: it never
        deletes a fact.</p></div>
      <div class="card"><h3>the tuned dedup at 0.67</h3><p><strong>32.66%</strong>
        of the facts in <strong>139</strong> tokens. The decision improved; the
        curation nearly halved.</p></div>
      <div class="card"><h3>the lesson</h3><p>A true duplicate merged is a fact
        <em>deleted</em>. The threshold and the model are two dials on the same
        curve, and the constraint fixes where the system sits.</p></div>
    </div>
    <p>The drift threshold is the free dial: it only penalises, it never
      deletes. The dedup threshold is the constrained one, because a false
      merge deletes a fact - and that is what fixes where the system sits on
      the curve.</p>
  </div>
</section>
"""

PANEL_MOVE = """
<section class="tab-panel" data-panel="move" id="move" role="tabpanel">
  <div class="wrap">
    <h2>The move: the same loop, one level up</h2>
    <p>The membrane answers "what should stay in the context?" The controller
      answers "what should handle this task, and when is it done?" They are the
      same question at different scope, so the machinery carries over:</p>
    <table>
      <thead><tr><th>the memory</th><th>the controller</th></tr></thead>
      <tbody>
        <tr><td>the router scores every chunk</td><td><code>ask(model, prompt)</code> routes to the tier</td></tr>
        <tr><td>dedup and drift are decisions with consequences</td><td><code>verify</code> is the decision with a consequence</td></tr>
        <tr><td>the judge is a readout, not the gate</td><td><code>score</code> is consulted only when <code>verify</code> is unknown</td></tr>
        <tr><td>the assembler keeps the facts under budget</td><td>the loop keeps the answer under a step budget</td></tr>
        <tr><td>a false merge deletes a fact</td><td>a false accept ships a wrong answer</td></tr>
      </tbody>
    </table>
    <p>The move is not a different mechanism. It is the same trade - what to
      keep, what to check, when to stop - with the consequence moved from the
      context to the answer.</p>
  </div>
</section>
"""

PANEL_CONTROLLER = """
<section class="tab-panel" data-panel="controller" id="controller" role="tabpanel">
  <div class="wrap">
    <h2>The controller: four tools, one table</h2>
    <p>The controller is a <strong>router and a verifier, not a knower</strong>.
      That is why it can be small: the tools do the work, and the table says
      what each tier is worth.</p>
    <div class="grid">
      <div class="card"><h3><code>ask(model, prompt)</code></h3><p>Route the
        task to a tier: the 35B (bulk), the 27B (repair), the API (residue).</p></div>
      <div class="card"><h3><code>verify(task, answer)</code></h3><p>Execution or
        claim-level grounding. <code>pass / fail / unknown</code>; a check that
        cannot fail a mutation is <code>unknown</code>.</p></div>
      <div class="card"><h3><code>score(task, answer)</code></h3><p>The judge's
        <code>p(correct)</code>. Only when <code>verify</code> is unknown; it
        never overrides a check.</p></div>
      <div class="card"><h3><code>ask_human(...)</code></h3><p>The last local
        tier: solve, escalate or reject - with the evidence.</p></div>
    </div>
    <p>The stage above is the loop on one task. The judge read the answer and
      wanted to keep it (<strong>p = 0.973</strong>); the checker recomputed it:
      700&times;5 + 1,400&times;15 = <strong>24,500&cent; = $245</strong>. The
      judge has a blind spot exactly where the answer is <em>plausible</em>;
      the checker does not. This is one of the four errors at
      <code>p &ge; 0.97</code>.</p>
    <p class="note good"><strong>The loop is bounded.</strong> After a configured
      number of steps the harness removes the tools and requires a final answer:
      <code>accepted</code>, <code>escalated</code> or <code>rejected</code>. The
      invariants are enforced by the tool surface and the policy layer, not by
      the prompt.</p>
  </div>
</section>
"""

PANEL_MEASURED = """
<section class="tab-panel" data-panel="measured" id="measured" role="tabpanel">
  <div class="wrap">
    <h2>What is measured</h2>
    <p class="sub">The same 28 organic tasks throughout. Costs are wall-clock
      per item on this box.</p>
    <table>
      <thead><tr><th>the arm</th><th>the result</th><th>the cost</th></tr></thead>
      <tbody>
        <tr><td>the 35B alone</td><td>21/28 (0.750)</td><td>the cheapest local</td></tr>
        <tr><td><strong>Bonsai 2 27B</strong></td>
            <td><strong>26/28 (0.929); +5 net</strong> (6 fixed, 1 broken)</td>
            <td>8.92 s/item</td></tr>
        <tr><td>the 176B generating</td><td>net 0 (2 fixed, 2 broken)</td>
            <td>20.68 s/item</td></tr>
        <tr><td>the 176B head judging the 35B</td>
            <td>AUC 0.864; <strong>3 of 7 errors, 0 of 21 correct</strong> sent up</td>
            <td>2.52 s/verdict</td></tr>
        <tr><td>judge gate + 27B</td><td>23/28 (0.821); 3 escalations, all errors</td>
            <td>3 escalations</td></tr>
        <tr><td>oracle gate + 27B</td><td>27/28 (0.964)</td><td>7 escalations</td></tr>
        <tr><td>the checker (execution + grounding)</td>
            <td><em>unmeasured: the current build</em></td><td>a sandbox call</td></tr>
      </tbody>
    </table>
    <p class="small">The 176B is the wrong shape for escalation: it is a
      judgement model, and its generation is no better than the 35B's. A 27B
      dense one is +5 net. The judge is the 176B's head; the checker is the
      missing tier.</p>
  </div>
</section>
"""

PANEL_CHECKER = """
<section class="tab-panel" data-panel="checker" id="checker" role="tabpanel">
  <div class="wrap">
    <h2>The checker: two verifiers, one verdict</h2>
    <p><strong>Execution</strong> where a program can settle it (run the code,
      recompute the arithmetic); <strong>grounding</strong> where only evidence
      can (claims against the retrieved context, each cited). They sit behind
      one <code>verify</code>; the log records the method.</p>
    <p>The stage above is the falsifiability rule. A check that passes the
      candidate and fails the mutation is evidence. A check that passes both is
      <code>print("PASS")</code> - and it must be <code>unknown</code>, or a
      small checker is worse than none.</p>
    <div class="grid">
      <div class="card"><h3>the false-positive bar</h3><p>The 176B head set it:
        <strong>0 of 21</strong> correct answers rejected. The checker is
        scored on the 21 correct first, then on the 7 errors.</p></div>
      <div class="card"><h3>grounding</h3><p>For the factual remainder,
        decompose the answer into claims and check each against the store.
        A check that cannot cite is <code>unknown</code>.</p></div>
      <div class="card"><h3>the cost</h3><p>A sandbox call, not a model
        verdict. The check runs once per answer and once per mutation; the
        decoder is only paid on the repair.</p></div>
    </div>
    <p class="small">This is the last unmeasured tier. The contract and the
      prior art are in <code>docs/HARNESS-CONTROLLER.md</code> and
      <code>docs/CONTROLLER-PRIOR-ART.md</code>.</p>
  </div>
</section>
"""

PANEL_PAPER_OPEN = """
<section class="tab-panel" data-panel="paper" id="paper" role="tabpanel">
  <div class="wrap">
    <h2>The paper</h2>
"""

PANEL_PAPER_CLOSE = "</div></section>\n"

LINKS = """
<section id="links"><div class="wrap">
  <h2>Where the rest is</h2>
  <table>
    <tr><td>the code, the full results log and the corrections</td>
        <td><a href="https://github.com/sky-is-green/hivebench">github.com/sky-is-green/hivebench</a></td></tr>
    <tr><td>the status: measured / invalid / unmeasured / blocked</td>
        <td><a href="https://github.com/sky-is-green/hivebench/blob/main/docs/STATUS.md">docs/STATUS.md</a></td></tr>
    <tr><td>the controller contract</td>
        <td><a href="https://github.com/sky-is-green/hivebench/blob/main/docs/HARNESS-CONTROLLER.md">docs/HARNESS-CONTROLLER.md</a></td></tr>
    <tr><td>the task sets (100 bench + 664 community)</td>
        <td><a href="https://huggingface.co/datasets/SkyIsNotGreen/hive-cascade">SkyIsNotGreen/hive-cascade</a></td></tr>
    <tr><td>the memory system</td>
        <td><a href="https://huggingface.co/SkyIsNotGreen/splinter-memory">SkyIsNotGreen/splinter-memory</a></td></tr>
    <tr><td>the encoder (research checkpoint)</td>
        <td><a href="https://huggingface.co/SkyIsNotGreen/splinter-encoder">SkyIsNotGreen/splinter-encoder</a></td></tr>
  </table>
</div></section>
"""

FOOT = """
<footer><div class="wrap">HiveBench Studio contributors · MIT · the paper is
  also in the repository at <a class="ref" href="https://github.com/sky-is-green/hivebench/blob/main/docs/CASCADE-PAPER.md"><code>docs/CASCADE-PAPER.md</code></a>.
</div></footer>
"""

JS = """
<script>
(function () {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NAMES = {memory: "the memory", move: "the move", controller: "the controller",
                 measured: "what is measured", checker: "the checker", paper: "the paper"};
  let moveTimer = null, routeTimer = null;

  // ---- memory: the threshold toggle
  const CURATION = {
    "0.92": "<strong>0.92:</strong> both stay. The pair is close but not the same fact, so the context can still say how deletes work.",
    "0.67": "<strong>0.67:</strong> the near-duplicate merges into fact A. <strong>D is gone</strong> - the context shrinks and can no longer answer the delete question. This is the mechanism behind the 60.51% -> 32.66% result."
  };
  function setCuration(demo, threshold) {
    const merge = threshold === "0.67";
    demo.dataset.threshold = threshold;
    demo.querySelectorAll("[data-near]").forEach(function (el) {
      el.classList.toggle("merged", merge && el.dataset.near === "merge");
      el.classList.toggle("kept", !merge && el.dataset.near === "keep");
      const tag = el.querySelector(".tag");
      if (tag && el.dataset.near === "merge")
        tag.textContent = merge ? "merged - fact D lost" : "fact A + fact D";
    });
    const d = demo.querySelector('[data-fact="D"]');
    if (d) { d.classList.toggle("lost", merge); d.textContent = merge ? "D - lost" : "D - soft deletes"; }
    const bar = demo.querySelector(".budget i");
    if (bar) bar.style.width = merge ? "55%" : "100%";
    const cap = demo.querySelector(".caption");
    if (cap) cap.innerHTML = CURATION[threshold];
    demo.querySelectorAll(".seg button").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.threshold === threshold));
    });
  }

  // ---- move: highlight the mapping, one row at a time
  const MOVE_CAPTIONS = [
    "The chunk router. The controller does the same, one level up.",
    "The memory's two decisions become the controller's single verification decision.",
    "A readout is never the gate - in either place.",
    "The consequence moves from the context (a deleted fact) to the answer (a wrong one)."
  ];
  function setMove(index) {
    const value = index % 4;
    document.querySelectorAll("[data-stage='move'] .map-row").forEach(function (row) {
      row.classList.toggle("active", Number(row.dataset.i) === value);
    });
    const cap = document.querySelector("[data-move-caption]");
    if (cap) cap.innerHTML = MOVE_CAPTIONS[value];
  }
  function startMove() {
    stopMove(); setMove(0);
    moveTimer = setInterval(function () {
      const active = document.querySelector("[data-stage='move'] .map-row.active");
      const next = active ? Number(active.dataset.i) + 1 : 0;
      setMove(next);
    }, 1800);
  }
  function stopMove() { if (moveTimer) { clearInterval(moveTimer); moveTimer = null; } }

  // ---- controller: the moving diagram
  const STEPS = [
    {edge: "e-prompt", color: "var(--c-prompt)", node: "35b", reveal: "35b",
     caption: "The prompt enters the 35B. The memory (teal) has already chosen the context."},
    {edge: "e-35b-judge", color: "var(--c-model)", node: "judge", reveal: "judge",
     keep: "e-judge-answer",
     caption: "The candidate <strong>24.50</strong> reaches the judge: <strong>p = 0.973</strong>. The plausible answer would be kept (purple, dashed)."},
    {edge: "e-verify-loop", color: "var(--c-exec)", node: "verify1", reveal: "verify1",
     state: "fail", strikeKeep: true, sandbox: true,
     caption: "The candidate also reaches execution: 700&times;5 + 1,400&times;15 = <strong>24,500&cent; = $245</strong>. 24.50 &ne; 245: <strong>FAIL</strong>. The judge's keep is overridden."},
    {edge: "e-verify-27b", color: "var(--bad)", node: "27b",
     caption: "The failure and the evidence go to the 27B: escalate with the check, not just the answer."},
    {edge: "e-27b-verify2", color: "var(--c-model)", node: "verify2", reveal: "27b",
     caption: "The 27B answers <strong>245</strong>."},
    {edge: "e-verify2-answer", color: "var(--good)", node: "answer", reveal: "verify2",
     state: "pass", answer: true,
     caption: "The checker passes <strong>245</strong>: <strong>accepted</strong>. The judge never had the last word."}
  ];
  let packetFrame = null;

  function resetRoute() {
    const stage = document.querySelector("[data-stage='controller']");
    if (!stage) return;
    stage.querySelectorAll(".node").forEach(function (n) {
      n.classList.remove("active", "revealed", "fail", "pass");
    });
    stage.querySelectorAll(".edge").forEach(function (e) {
      e.classList.remove("active", "struck");
      e.style.stroke = "";
    });
    stage.querySelectorAll(".node .val").forEach(function (v) { v.style.opacity = ""; });
    const packet = document.getElementById("packet");
    if (packet) { packet.style.opacity = "0"; }
  }

  function travel(path, color, duration) {
    const packet = document.getElementById("packet");
    if (!packet) return;
    if (reduced) { packet.style.opacity = "0"; return; }
    if (packetFrame) cancelAnimationFrame(packetFrame);
    const len = path.getTotalLength();
    const t0 = performance.now();
    packet.style.fill = color;
    packet.style.opacity = "1";
    function frame(now) {
      const p = Math.min((now - t0) / duration, 1);
      const pt = path.getPointAtLength(len * p);
      packet.setAttribute("cx", pt.x);
      packet.setAttribute("cy", pt.y);
      if (p < 1) { packetFrame = requestAnimationFrame(frame); }
      else { packet.style.opacity = "0"; packetFrame = null; }
    }
    packetFrame = requestAnimationFrame(frame);
  }

  function setRouteStep(i) {
    const stage = document.querySelector("[data-stage='controller']");
    if (!stage) return;
    const step = STEPS[i];
    if (!step) return;
    stage.querySelectorAll(".node").forEach(function (n) { n.classList.remove("active"); });
    stage.querySelectorAll(".edge").forEach(function (e) {
      if (!e.classList.contains("struck")) e.classList.remove("active");
    });
    const node = stage.querySelector('[data-node="' + step.node + '"]');
    if (node) {
      node.classList.add("active");
      if (step.reveal) node.classList.add("revealed");
    }
    const edge = document.getElementById(step.edge);
    if (edge) {
      edge.classList.add("active");
      edge.style.stroke = step.color;
      travel(edge, step.color, step.sandbox ? 1250 : 750);
    }
    if (step.keep) {
      const keep = document.getElementById(step.keep);
      if (keep) { keep.classList.add("active"); keep.style.stroke = "var(--c-judge)"; }
    }
    if (step.strikeKeep) {
      const keep = document.getElementById("e-judge-answer");
      if (keep) { keep.classList.remove("active"); keep.classList.add("struck"); keep.style.stroke = ""; }
    }
    if (step.state === "fail") {
      const v = stage.querySelector('[data-node="verify1"]');
      if (v) v.classList.add("fail");
    }
    if (step.state === "pass") {
      const v = stage.querySelector('[data-node="verify2"]');
      if (v) v.classList.add("pass");
      const a = stage.querySelector('[data-node="answer"]');
      if (a) { a.classList.add("revealed"); }
    }
    const cap = document.querySelector("[data-route-caption]");
    if (cap) cap.innerHTML = step.caption;
  }

  function stopRoute() {
    if (routeTimer) { clearInterval(routeTimer); routeTimer = null; }
  }

  function playRoute() {
    stopRoute();
    resetRoute();
    let i = 0;
    setRouteStep(0);
    if (reduced) { setRouteStep(STEPS.length - 1); return; }
    routeTimer = setInterval(function () {
      i += 1;
      if (i >= STEPS.length) { stopRoute(); return; }
      setRouteStep(i);
    }, 2600);
  }

  // ---- measured: animate the bars
  function animateBars() {
    const bars = document.querySelector("[data-bars]");
    if (!bars) return;
    bars.classList.remove("animate");
    void bars.offsetWidth;
    bars.classList.add("animate");
  }

  // ---- checker: real vs vacuous
  function setCheck(kind) {
    const real = kind === "real";
    const stage = document.querySelector("[data-stage='checker']");
    if (!stage) return;
    stage.querySelectorAll(".seg button").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.check === kind));
    });
    const a = stage.querySelector("[data-run-a]");
    const b = stage.querySelector("[data-run-b]");
    if (a) { a.textContent = "PASS"; a.className = "verdict pass"; }
    if (b) {
      b.textContent = real ? "FAIL" : "PASS";
      b.className = "verdict " + (real ? "fail" : "pass");
    }
    const out = stage.querySelector("[data-outcome]");
    if (out) out.textContent = real
      ? "accepted - the check discriminates between the answer and its mutation"
      : "unknown - the check passes the mutation, so it is not evidence";
  }

  // ---- tabs
  function activate(name, push) {
    if (!NAMES[name]) name = "memory";
    document.querySelectorAll(".tabs a").forEach(function (a) {
      a.setAttribute("aria-selected", String(a.dataset.tab === name));
    });
    document.querySelectorAll(".tab-panel").forEach(function (p) {
      p.hidden = p.dataset.panel !== name;
    });
    document.querySelectorAll(".stage-panel").forEach(function (p) {
      p.hidden = p.dataset.stage !== name;
    });
    const label = document.getElementById("stage-name");
    if (label) label.textContent = NAMES[name];
    if (name !== "move") stopMove(); else startMove();
    if (name !== "controller") stopRoute();
    if (name === "measured") animateBars();
    if (name === "controller" && !reduced) playRoute();
    if (push !== false) history.replaceState(null, "", "#" + name);
  }

  // ---- wire
  const curation = document.querySelector("[data-demo='curation'], [data-stage='memory']");
  if (curation) {
    setCuration(curation, "0.92");
    curation.querySelectorAll(".seg button").forEach(function (b) {
      b.addEventListener("click", function () { setCuration(curation, b.dataset.threshold); });
    });
  }
  document.querySelectorAll(".seg button[data-check]").forEach(function (b) {
    b.addEventListener("click", function () { setCheck(b.dataset.check); });
  });
  setCheck("real");
  const play = document.querySelector("[data-route-play]");
  if (play) play.addEventListener("click", playRoute);

  document.querySelectorAll(".tabs a").forEach(function (a) {
    a.addEventListener("click", function (e) {
      e.preventDefault();
      activate(a.dataset.tab);
      document.querySelector(".tabs").scrollIntoView({ block: "start" });
    });
  });

  const start = (location.hash || "#memory").slice(1);
  activate(NAMES[start] ? start : "memory", false);
  if (reduced) {
    document.querySelectorAll("[data-route-play]").forEach(function (b) { b.hidden = true; });
    stopRoute();
    setRouteStep(STEPS.length - 1);
  }
})();
</script>
"""


_REPO = "https://github.com/sky-is-green/hivebench"
_REF = re.compile(
    r"<code>((?:docs|experiments|harness|scripts|tests|tools|configs|stacks)/[A-Za-z0-9_./-]+)</code>")


def link_refs(html: str) -> str:
    """Link every inline code span that names a file that exists here."""
    def repl(match):
        path = match.group(1)
        target = REPO / path
        if not target.exists():
            return match.group(0)
        kind = "tree" if target.is_dir() else "blob"
        url = f"{_REPO}/{kind}/main/{path}"
        return f'<a class="ref" href="{url}"><code>{path}</code></a>'
    html = _REF.sub(repl, html)
    html = html.replace(
        "<code>SkyIsNotGreen/decision-heads</code>",
        '<a class="ref" href="https://huggingface.co/datasets/SkyIsNotGreen/decision-heads">'
        "<code>SkyIsNotGreen/decision-heads</code></a>")
    for name in ("results/runner-cascade-sampled-20261018/",
                 "results/runner-cascade-sampled-nothink-20261018/"):
        html = html.replace(
            f"<code>{name}</code>",
            f'<a class="ref" href="https://huggingface.co/datasets/SkyIsNotGreen/hive-cascade/tree/main/results">'
            f"<code>{name}</code></a>")
    return html


def build() -> tuple[str, str]:
    import markdown

    paper = (REPO / "docs" / "CASCADE-PAPER.md").read_text(encoding="utf-8")
    parts = paper.split("---\n", 2)
    body_md = parts[2] if len(parts) == 3 else paper
    lines = body_md.lstrip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    body_md = "\n".join(lines)
    body_html = link_refs(markdown.markdown(
        body_md, extensions=["tables", "fenced_code", "sane_lists"]))
    html = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>The Hive Cascade - from memory to a global controller</title>\n"
        '<meta name="description" content="A gold-free verification loop for local language models: the memory, the move to a global controller, and what is measured.">\n'
        '<link rel="stylesheet" href="style.css">\n</head>\n<body>\n'
        + HERO + TABS + STAGE_OPEN + STAGE_MEMORY + STAGE_MOVE + STAGE_CONTROLLER
        + STAGE_MEASURED + STAGE_CHECKER + STAGE_PAPER + STAGE_CLOSE
        + "<main>\n" + PANEL_MEMORY + PANEL_MOVE + PANEL_CONTROLLER
        + PANEL_MEASURED + PANEL_CHECKER + PANEL_PAPER_OPEN + body_html
        + PANEL_PAPER_CLOSE + "</main>\n" + LINKS + FOOT + JS
        + "\n</body>\n</html>\n"
    )
    return html, CSS


def main(argv: list[str]) -> int:
    html, css = build()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / "style.css").write_text(css, encoding="utf-8")
    print(f"wrote {OUT/'index.html'} ({len(html)} chars) and style.css")
    if "--upload" in argv:
        from huggingface_hub import HfApi

        api = HfApi()
        for name in ("index.html", "style.css"):
            api.upload_file(path_or_fileobj=str(OUT / name), path_in_repo=name,
                            repo_id=SPACE, repo_type="space",
                            commit_message="Tabs with a per-tab example stage")
        print(f"uploaded to https://huggingface.co/spaces/{SPACE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
