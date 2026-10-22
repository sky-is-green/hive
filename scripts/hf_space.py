"""Render and publish the HF Space page for the cascade paper.

Seven tabs (the premise, the memory, the routing, the controller, what is measured, the
checker, the paper). Above the tab's content sits a **stage**: the
example for that tab, which changes with it. The demos are
dependency-free, respect ``prefers-reduced-motion``, and the page works
without JavaScript (all panels are present, the tabs are anchors).

A static Space serves ``index.html``; the ``README.md`` is only the repo
card. The paper itself lives in the repository (``docs/CASCADE-PAPER.md``);
this script builds the page and the card, and points at it.

    python scripts/hf_space.py            # render to /tmp/opencode/hf-space
    python scripts/hf_space.py --upload   # publish to the Space
"""

from __future__ import annotations

import sys
from pathlib import Path

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
.flow { display: flex; align-items: center; gap: 5px; margin-top: 20px;
  flex-wrap: wrap; font-size: .74rem; }
.flow .node { border: 1px solid var(--line); background: var(--card); border-radius: 7px;
  padding: 5px 8px; font-family: ui-monospace, monospace; color: var(--muted);
  transition: border-color .3s, color .3s, box-shadow .3s; }
.flow .node[data-flow-node="memory"] { color: var(--c-mem); }
.flow .node[data-flow-node="prompt"] { color: var(--c-prompt); }
.flow .node[data-flow-node="35b"],
.flow .node[data-flow-node="27b"] { color: var(--c-model); }
.flow .node[data-flow-node="judge"] { color: var(--c-judge); }
.flow .node[data-flow-node="verify"],
.flow .node[data-flow-node="answer"] { color: var(--good); }
.flow .node.api { color: var(--c-exec); opacity: .55; }
.flow .node.active { border-color: var(--packet, var(--accent));
  color: var(--packet, var(--accent));
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--packet, var(--accent)) 18%, transparent); }
.flow .link { flex: 1 1 22px; min-width: 16px; height: 2px; background: var(--line);
  position: relative; overflow: hidden; }
.flow .link.dashed { background: none; border-top: 2px dashed var(--line); height: 0; }
.flow .link i { position: absolute; top: -3px; left: -8px; width: 7px; height: 8px;
  border-radius: 4px; background: var(--packet, var(--accent)); opacity: 0; }
.flow .link.dashed i { top: -5px; }
.flow .link.active i { animation: travel 1.15s linear; opacity: 1; }
.flow .link .p { position: absolute; width: 5px; height: 5px; border-radius: 50%;
  opacity: 0; pointer-events: none; }
.flow .link .p.ltr { background: var(--c-mem); top: -5px; }
.flow .link .p.rtl { background: var(--c-prompt); top: 3px; }
.flow .link.exchange { overflow: visible; }
.flow .link.exchange .p.ltr { animation: ex-ltr 1.25s linear infinite; }
.flow .link.exchange .p.rtl { animation: ex-rtl 1.25s linear infinite; }
.flow .link.exchange .p.d1 { animation-delay: .62s; }
@keyframes ex-ltr { 0% { left: -6px; opacity: 0; } 12% { opacity: 1; }
  88% { opacity: 1; } 100% { left: 100%; opacity: 0; } }
@keyframes ex-rtl { 0% { left: 100%; opacity: 0; } 12% { opacity: 1; }
  88% { opacity: 1; } 100% { left: -6px; opacity: 0; } }
@keyframes travel { 0% { left: -8px; opacity: 0; } 15% { opacity: 1; }
  85% { opacity: 1; } 100% { left: 100%; opacity: 0; } }
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

.lead { font-size: 1.12rem; }
.thesis-q { font-size: 1.4rem; letter-spacing: -.01em; margin: 0 0 .3rem; }
.path { display: flex; gap: 8px; align-items: stretch; flex-wrap: wrap; margin-top: 14px; }
.path button { flex: 1 1 150px; text-align: left; border: 1px solid var(--line);
  border-radius: 9px; padding: 10px 12px; background: var(--bg); cursor: pointer;
  font: inherit; font-size: .9rem; color: var(--fg); }
.path button small { display: block; color: var(--muted); font-size: .74rem;
  margin-top: 2px; }
.path button:hover { border-color: var(--accent); }
.path .arrow { align-self: center; color: var(--muted); }
@media (max-width: 700px) { .path .arrow { display: none; } }
/* stage demos */
.demo-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.demo-head h3 { margin: 0; }
button { font: inherit; font-size: .78rem; padding: .25rem .65rem; border-radius: 999px;
  border: 1px solid var(--line); background: var(--bg); color: var(--fg); cursor: pointer; }
button:hover { border-color: var(--accent); color: var(--accent); }
.caption { min-height: 2.9em; margin: .7rem 0 0; font-size: .88rem; color: var(--muted); }
.caption strong { color: var(--fg); }
.labflow { margin-top: 6px; }
.flow .node b { font-weight: 700; }
.lab { display: grid; grid-template-columns: 1.1fr .9fr; gap: 18px; margin-top: 12px; }
@media (max-width: 760px) { .lab { grid-template-columns: 1fr; } }
.lab-row { display: grid; grid-template-columns: 44px 1fr 52px 38px 16px; gap: 7px;
  align-items: center; padding: 4px 0; border-bottom: 1px solid var(--line);
  font-size: .8rem; }
.lab-row .lab-text { font-size: .77rem; color: var(--muted); overflow: hidden;
  white-space: nowrap; text-overflow: ellipsis; }
.lab-row.selected .pick { color: var(--good); }
.lab-row .pick { color: var(--muted); text-align: center; }
.lab-row.merge .lab-text { color: var(--bad); text-decoration: line-through; }
@media (max-width: 720px) { .lab-row { grid-template-columns: 40px 1fr 38px 14px; }
  .lab-row .vec { display: none; } }
.query { margin-top: 10px; font-size: .9rem; color: var(--muted); }
.query strong { color: var(--fg); }
.context-box { border: 1px solid var(--line); border-radius: 8px; padding: 7px;
  display: grid; gap: 5px; margin-top: 5px; }
.ctx-item { font-size: .76rem; background: var(--code); border-radius: 5px;
  padding: 5px 7px; transition: color .3s, opacity .3s; }
.ctx-item.lost { color: var(--bad); text-decoration: line-through; opacity: .55; }
.lab-row .lab-name { font-family: ui-monospace, monospace; color: var(--muted); }
.lab-row .cos { text-align: right; font-family: ui-monospace, monospace;
  color: var(--muted); font-size: .74rem; }
.lab-row.merge .lab-name, .lab-row.merge .cos { color: var(--bad); }
.lab-row.dim { opacity: .45; }
.vec { display: flex; gap: 2px; align-items: flex-end; height: 20px; }
.vec i { flex: 1; background: var(--c-mem); border-radius: 2px 2px 0 0;
  opacity: .8; min-height: 1px; }
.lab-decision { min-height: 3em; font-size: .84rem; margin: .6rem 0 0; color: var(--muted); }
.lab-decision strong { color: var(--fg); }
.slider { display: flex; align-items: center; gap: 10px; margin-top: 12px;
  font-size: .82rem; color: var(--muted); }
.slider input { flex: 1; accent-color: var(--accent); }
.slider output { font-family: ui-monospace, monospace; color: var(--fg);
  min-width: 3.2em; text-align: right; }
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
  transition: background .35s; position: relative; cursor: help; }
.map-row:focus { outline: 1px solid var(--accent); outline-offset: 1px; }
.map-row::after { content: attr(data-tip); position: absolute; left: 0; right: 0;
  top: calc(100% + 5px); background: var(--card); border: 1px solid var(--accent);
  border-radius: 8px; padding: 8px 10px; font-size: .78rem; line-height: 1.45;
  color: var(--fg); opacity: 0; pointer-events: none; transition: opacity .15s;
  z-index: 10; box-shadow: 0 8px 22px rgba(0, 0, 0, .18); }
.map-row:hover::after, .map-row:focus::after, .map-row.show-tip::after { opacity: 1; }
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
.check-code { margin: 8px 0 2px; }
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
.flowmap .node.hub rect { stroke: var(--accent); stroke-width: 2; }
.flowmap .node .sub { font-size: 10px; font-weight: 500; fill: var(--muted); }
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
.timing { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px;
  font-size: .76rem; color: var(--muted); }
.timing span { border: 1px solid var(--line); border-radius: 999px;
  padding: 2px 8px; transition: border-color .3s, color .3s; }
.timing b { color: var(--fg); font-weight: 600; }
.timing span.active { border-color: var(--accent); color: var(--accent); }
.timing .total { border-color: var(--accent); color: var(--accent); }
.timing-note { font-size: .78rem; color: var(--muted); margin-top: 6px; }
.paper-card { border-left: 3px solid var(--accent); padding: .1rem 0 .1rem 1rem; }
.paper-actions { display: flex; gap: 8px; flex-wrap: wrap; margin: 1rem 0 1.25rem; }
.cta { display: inline-block; background: var(--accent); color: var(--bg);
  padding: 8px 14px; border-radius: 8px; text-decoration: none;
  font-weight: 600; font-size: .9rem; }
.cta:hover { filter: brightness(1.08); }
.cta.ghost { background: transparent; color: var(--accent);
  border: 1px solid var(--accent); }
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
  <p class="meta"><span class="dot"></span>Not a product yet ·
    <span class="pill">MIT</span><span class="pill">updated 2026-10-21</span></p>
  <div class="flow" data-flow aria-hidden="true">
    <span class="node" data-flow-node="memory">memory</span>
    <span class="link dashed" data-to="prompt"><i></i><b class="p ltr d0"></b><b class="p ltr d1"></b><b class="p rtl d0"></b><b class="p rtl d1"></b></span>
    <span class="node" data-flow-node="prompt">prompt</span>
    <span class="link" data-to="35b"><i></i></span>
    <span class="node" data-flow-node="35b">35B</span>
    <span class="link" data-to="judge"><i></i></span>
    <span class="node" data-flow-node="judge">judge</span>
    <span class="link" data-to="verify"><i></i></span>
    <span class="node" data-flow-node="verify">verify</span>
    <span class="link" data-to="27b"><i></i></span>
    <span class="node" data-flow-node="27b">27B</span>
    <span class="link" data-to="answer"><i></i></span>
    <span class="node" data-flow-node="answer">answer</span>
    <span class="link dashed" data-to="api"><i></i></span>
    <span class="node api" data-flow-node="api">API · residue</span>
  </div>
</div></header>
"""

TABS = """
<nav class="tabs" aria-label="sections"><div class="wrap" role="tablist">
  <a role="tab" href="#premise" data-tab="premise" aria-selected="true">the premise</a>
  <a role="tab" href="#memory" data-tab="memory" aria-selected="false">the memory</a>
  <a role="tab" href="#routing" data-tab="routing" aria-selected="false">the routing</a>
  <a role="tab" href="#controller" data-tab="controller" aria-selected="false">the controller</a>
  <a role="tab" href="#measured" data-tab="measured" aria-selected="false">what is measured</a>
  <a role="tab" href="#checker" data-tab="checker" aria-selected="false">the checker</a>
  <a role="tab" href="#paper" data-tab="paper" aria-selected="false">the paper</a>
</div></nav>
"""

STAGE_OPEN = """
<section class="stage-wrap"><div class="wrap"><div class="stage">
  <div class="stage-head">stage · <strong id="stage-name">the premise</strong></div>
"""

STAGE_PREMISE = """
  <div class="stage-panel" data-stage="premise">
    <div class="thesis">
      <p class="thesis-q">A model that is wrong is rarely uncertain.</p>
      <p>The 176B's judge gave the 35B's wrong answer <strong>p = 0.973</strong>.
        Its mean confidence on the 21 correct answers was <strong>0.988</strong>.
        Confidence does not separate them; execution does.</p>
    </div>
    <div class="path" aria-label="the research path">
      <button type="button" data-goto="memory">the memory<small>curate the context</small></button>
      <span class="arrow">→</span>
      <button type="button" data-goto="controller">the controller<small>route and verify</small></button>
      <span class="arrow">→</span>
      <button type="button" data-goto="checker">the checker<small>the last tier</small></button>
      <span class="arrow">→</span>
      <button type="button" data-goto="measured">the evidence<small>28 tasks</small></button>
    </div>
  </div>
"""


STAGE_MEMORY = """
  <div class="stage-panel" data-stage="memory">
    <div class="demo-head"><h3>The drone at work</h3></div>
    <div class="flow labflow" data-labflow aria-hidden="true">
      <span class="node" data-lab-node="store">store · 5</span><span class="link"><i></i></span>
      <span class="node" data-lab-node="drone">drone · 17M</span><span class="link"><i></i></span>
      <span class="node" data-lab-node="cosine">cosine</span><span class="link"><i></i></span>
      <span class="node" data-lab-node="membrane">membrane · <b data-lab-membrane>2</b> kept</span><span class="link"><i></i></span>
      <span class="node" data-lab-node="context">context · <b data-lab-context-count>2</b></span>
    </div>
    <div class="query">the prompt: <strong>“How are order deletions handled?”</strong></div>
    <div class="lab">
      <div>
        <div class="lab-row selected">
          <span class="lab-name">A</span>
          <span class="lab-text">3NF with a denormalized read model</span>
          <span class="vec" data-vec="a"></span>
          <span class="cos">0.74</span><span class="pick">✓</span>
        </div>
        <div class="lab-row selected" data-lab-row="ad">
          <span class="lab-name">A + D</span>
          <span class="lab-text" data-lab-text>… soft deletes via deleted_at</span>
          <span class="vec" data-vec="ad"></span>
          <span class="cos">0.91</span><span class="pick">✓</span>
        </div>
        <div class="lab-row">
          <span class="lab-name">B</span>
          <span class="lab-text">indexes on (customer_id, created_at)</span>
          <span class="vec" data-vec="b"></span>
          <span class="cos">0.22</span><span class="pick">—</span>
        </div>
        <div class="lab-row">
          <span class="lab-name">C</span>
          <span class="lab-text">Alembic, forward-only</span>
          <span class="vec" data-vec="c"></span>
          <span class="cos">0.18</span><span class="pick">—</span>
        </div>
        <div class="lab-row dim">
          <span class="lab-name">K</span>
          <span class="lab-text">Kubernetes sidecar :8765</span>
          <span class="vec" data-vec="k"></span>
          <span class="cos">0.05</span><span class="pick">—</span>
        </div>
        <p class="lab-decision" data-lab-decision></p>
      </div>
      <div>
        <div class="small">the prompt's context</div>
        <div class="context-box">
          <div class="ctx-item">A · 3NF with a denormalized read model</div>
          <div class="ctx-item" data-ctx-d>D · soft deletes via deleted_at</div>
        </div>
        <div class="budget" title="context budget: 2 of 2 chunks" aria-label="context budget used"><i data-lab-budget></i></div>
        <p class="small" data-lab-context></p>
      </div>
    </div>
    <label class="slider">dedup threshold
      <input type="range" min="0.60" max="0.95" step="0.01" value="0.92" data-lab-threshold>
      <output data-lab-value>0.92</output>
    </label>
    <p class="caption" data-lab-caption aria-live="polite"></p>
  </div>
"""

STAGE_ROUTING = """
  <div class="stage-panel" data-stage="routing" hidden>
    <div class="map" data-map>
      <div class="map-row" data-i="0" tabindex="0"
           data-tip="The memory's router scores every stored chunk against the query and keeps the top ones under the budget. The controller's ask does the same at the tier level: the table scores what each tier can settle, and the task goes to the one that fits.">
        <span>the router scores every chunk</span><span class="arrow">→</span><span><code>ask(model, prompt)</code></span></div>
      <div class="map-row" data-i="1" tabindex="0"
           data-tip="Dedup is the hard decision: a merge removes a chunk, so a wrong merge deletes a fact. Drift is the soft one: it only demotes old-topic chunks, so a wrong drift is cheap. The controller mirrors this: verify can reject, while score only advises. The threshold matters most on the hard side.">
        <span>dedup decides; drift only penalises</span><span class="arrow">→</span><span><code>verify</code> decides; <code>score</code> only advises</span></div>
      <div class="map-row" data-i="2" tabindex="0"
           data-tip="A similarity score never deletes; it only ranks. In the controller, score never overrides a check; it is consulted only when verify is unknown. In both places the readout is evidence, not the gate.">
        <span>the judge is a readout</span><span class="arrow">→</span><span><code>score</code> is the fallback</span></div>
      <div class="map-row" data-i="3" tabindex="0"
           data-tip="Both failures are irreversible consequences behind an agreeable decision: the merge looks fine until the fact is gone; the answer looks plausible until it ships. That is why both thresholds are conservative.">
        <span>a false merge deletes a fact</span><span class="arrow">→</span><span>a false accept ships a wrong answer</span></div>
    </div>
    <p class="caption" data-routing-caption aria-live="polite"></p>
  </div>
"""

STAGE_CONTROLLER = """
  <div class="stage-panel" data-stage="controller" hidden>
    <div class="demo-head"><h3>The controller, calling its tools</h3>
      <button type="button" data-route-play>Play</button></div>
    <svg class="flowmap" viewBox="0 0 880 380" role="img"
         aria-label="The controller at the centre: it calls the 35B, the judge, execution and the 27B, and receives each result back. The models never talk to each other.">
      <path id="e-prompt-ctl" class="edge" d="M130,142 C210,142 280,175 340,185"/>
      <path id="e-mem-ctl" class="edge dashed" d="M130,270 C220,270 280,230 340,215"/>
      <path id="e-ctl-35b" class="edge" d="M390,165 C360,130 320,100 290,94"/>
      <path id="e-35b-ctl" class="edge" d="M290,94 C330,115 370,150 395,168"/>
      <path id="e-ctl-judge" class="edge" d="M490,165 C530,130 570,100 590,94"/>
      <path id="e-judge-ctl" class="edge" d="M590,100 C555,115 525,145 495,168"/>
      <path id="e-ctl-verify" class="edge" d="M540,200 C565,200 585,208 610,210"/>
      <path id="e-verify-ctl" class="edge" d="M610,222 C585,222 560,212 540,212"/>
      <path id="e-ctl-27b" class="edge" d="M500,240 C540,275 570,305 590,315"/>
      <path id="e-27b-ctl" class="edge" d="M590,340 C555,320 520,270 505,245"/>
      <path id="e-ctl-answer" class="edge" d="M340,240 C300,265 280,295 300,305"/>
      <path id="e-ctl-answer-keep" class="edge dashed" d="M350,225 C285,235 260,285 295,300"/>
      <path id="e-ctl-api" class="edge dashed" d="M540,230 C620,265 700,305 770,315"/>
      <circle id="packet" r="6" fill="#3b82f6" opacity="0"/>
      <g id="packets"></g>
      <g class="node" data-node="prompt"><rect x="30" y="120" width="100" height="44" rx="8"/><text x="80" y="147">prompt</text></g>
      <g class="node" data-node="memory"><rect x="30" y="250" width="100" height="40" rx="8"/><text x="80" y="274">memory</text></g>
      <g class="node hub" data-node="controller"><rect x="340" y="165" width="200" height="80" rx="10"/><text x="440" y="200">controller</text><text class="sub" x="440" y="220">the loop</text></g>
      <g class="node" data-node="35b"><rect x="170" y="40" width="120" height="54" rx="8"/><text x="230" y="63">35B</text><text class="val" x="230" y="80" data-val="35b">24.50</text></g>
      <g class="node" data-node="judge"><rect x="590" y="40" width="120" height="54" rx="8"/><text x="650" y="63">judge</text><text class="val" x="650" y="80" data-val="judge">p = 0.973</text></g>
      <g class="node" data-node="verify"><rect x="610" y="180" width="140" height="70" rx="8"/><text x="680" y="205">verify</text><text class="val" x="680" y="226" data-val="verify">—</text><rect x="635" y="255" width="90" height="24" rx="6" fill="none" stroke-dasharray="3 3"/><text class="sub" x="680" y="271">sandbox</text></g>
      <g class="node" data-node="27b"><rect x="590" y="305" width="120" height="54" rx="8"/><text x="650" y="328">27B</text><text class="val" x="650" y="345" data-val="27b">245</text></g>
      <g class="node" data-node="answer"><rect x="170" y="290" width="130" height="54" rx="8"/><text x="235" y="313">answer</text><text class="val pass" x="235" y="330" data-val="answer">accepted</text></g>
      <g class="node" data-node="api"><rect x="770" y="296" width="90" height="44" rx="8"/><text x="815" y="316">API</text><text class="sub" x="815" y="330">the slot</text></g>
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
    <div class="timing" aria-label="stage times">
      <span data-time-step="1"><b>35B</b> 6.68 s</span>
      <span data-time-step="2"><b>judge</b> 2.52 s</span>
      <span data-time-step="3"><b>checker</b> &mdash;</span>
      <span data-time-step="4"><b>27B</b> 8.92 s</span>
      <span data-time-step="5"><b>checker</b> &mdash;</span>
      <span class="total"><b>total</b> 18.1 s</span>
    </div>
    <p class="caption" data-route-caption aria-live="polite"></p>
    <p class="timing-note">Means over the 28 tasks: 35B 6.68 s, judge 2.52 s,
      27B 8.92 s. The checker is unmeasured, so it is not in the 18.1 s
      total.</p>
    <p class="timing-note"><strong>It is conditional.</strong> This is the
      escalated path, not the normal one. On the 28-task run the 35B answered
      all 28; the judge sent up <strong>3</strong> - all of them errors, and
      <strong>0 of the 21 correct</strong> - and the 27B ran on those 3. The
      other 25 never left the first tier. The mean over the run was
      <strong>10.2 s</strong> per task (35B and judge on every task, the 27B on
      three); the expensive tier is the exception, and the gate is what keeps
      it one.</p>
    <p class="timing-note"><strong>The 27B is a slot, not a requirement.</strong>
      It can be the API: swap it for one API call and the loop is unchanged.
      Or keep it and put the API <em>behind</em> it - escalate locally first,
      and call the API only on the case the 27B cannot settle. The controller's
      table decides which; the diagram draws the API as the residue because
      that is its role. The 27B is the measured tier (<strong>+5 net</strong>);
      the API arm is the unmeasured one (the key is expired).</p>
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
        <button type="button" data-check="real" aria-pressed="true">A real check</button>
        <button type="button" data-check="vacuous" aria-pressed="false">A vacuous check</button>
      </div></div>
    <pre class="check-code"><code data-check-code>assert float(answer) == (700*5 + 1400*15) / 100</code></pre>
    <div class="runs">
      <div class="run"><span>On the candidate <code>245</code></span><span class="verdict pass" data-run-a>PASS</span></div>
      <div class="run"><span>On the mutation <code>246</code></span><span class="verdict fail" data-run-b>FAIL</span></div>
    </div>
    <div class="outcome" data-outcome>Accepted - the check discriminates between the answer and its mutation</div>
    <p class="caption">A check that passes the mutation is not evidence. It is
      <code>unknown</code>, and the controller must not accept on it.</p>
  </div>
"""

STAGE_PAPER = f"""
  <div class="stage-panel" data-stage="paper" hidden>
    <div class="paper-card">
      <h3>The Hive Cascade: detect → repair → escalate</h3>
      <p>A gold-free verification-and-repair loop for local language models.
        The full text lives in the repository, with the code and the run
        bundles.</p>
    </div>
    <p class="caption">The paper is the record.
      <a href="{GH}/blob/main/docs/CASCADE-PAPER.md">Read it on GitHub.</a></p>
  </div>
"""

STAGE_CLOSE = "</div></div></section>\n"

PANEL_PREMISE = """
<section class="tab-panel" data-panel="premise" id="premise" role="tabpanel">
  <div class="wrap">
    <h2>Why I did this</h2>
    <p class="lead">A local model is cheap, private, and wrong in ways that
      look exactly like being right. The question is not whether the model is
      good; it is whether the system around it can tell.</p>
    <p>The research began with a memory system: a 17M encoder curating a long
      conversation into a bounded context. It became a study of
      <em>consequences</em> - which decision, when wrong, actually costs
      something. That thread runs through every tab here: a false merge deletes
      a fact; a false accept ships a wrong answer.</p>

    <h3>The hypotheses that drove it</h3>
    <div class="grid">
      <div class="card"><h3>H1 · The harness is the lever</h3>
        <p>A small model with verification can beat a bigger model alone. The
          27B reaches <strong>26/28</strong> against the 35B's 21/28; the
          176B's own generation is net zero. Supported - for the harness, not
          for "bigger".</p></div>
      <div class="card"><h3>H2 · Consequence, not accuracy</h3>
        <p>Optimise what a wrong decision costs. A better dedup decision at
          0.67 halves the facts (<strong>60.51% &rarr; 32.66%</strong>); the
          judge's four misses at <code>p &ge; 0.97</code> are plausible
          answers. Supported.</p></div>
      <div class="card"><h3>H3 · Execution, not judgement</h3>
        <p>A reader cannot see a plausible wrong answer; a check can. The
          checker is the last unmeasured tier. Supported by the failures, not
          yet by the fix.</p></div>
      <div class="card"><h3>H4 · One loop, two levels</h3>
        <p>The membrane and the controller make the same trade - what to keep,
          what to check, when to stop. The memory level is measured; the controller
          is the current work.</p></div>
    </div>

    <p class="note"><strong>What follows.</strong> The next tabs are the
      evidence: the memory's encoder at work, the routing, the controller's
      loop, the measured table, the missing checker, and the paper. The first
      one is interactive.</p>
  </div>
</section>
"""


PANEL_MEMORY = """
<section class="tab-panel" data-panel="memory" id="memory" role="tabpanel">
  <div class="wrap">
    <h2>The memory: what the drone learned</h2>
    <p>The first harness curated <em>within</em> a conversation. A 17M
      encoder, “the drone”, scores every stored chunk; the membrane makes two
      decisions on every turn: <strong>dedup</strong> (is this chunk already
      said?) and <strong>drift</strong> (did the topic change?), and the
      assembler keeps the context under budget.</p>
    <p>The stage above is the argument for the conservative dedup, on a prompt
      that asks <strong>how order deletions are handled</strong>. At 0.92 the
      two schema chunks are both selected - one of them carries the
      <code>deleted_at</code> fact - and the prompt can answer. At 0.67 the
      near-duplicate merges before selection, and the only evidence for deletes
      goes with it. The drift chunk dims in both states: it is only penalised,
      never deleted.</p>
    <p class="small">The slider is there to be explored: the pair's cosine is
      0.89, so the boundary is discoverable, and either side has a visible cost.
      But the shipped 0.92 is not derived from this pair - it is set by the
      corpus replay in the cards below (60.51% of the facts at 0.92, 32.66% at
      0.67). A threshold is a population decision; the lab is one case.</p>
    <div class="grid">
      <div class="card"><h3>The shipped point</h3><p><strong>60.51%</strong> of
        the facts kept in <strong>374</strong> tokens. Precision 1.00: it never
        deletes a fact.</p></div>
      <div class="card"><h3>The tuned dedup at 0.67</h3><p><strong>32.66%</strong>
        of the facts in <strong>139</strong> tokens. The decision improved; the
        curation nearly halved.</p></div>
      <div class="card"><h3>The lesson</h3><p>A true duplicate merged is a fact
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

PANEL_ROUTING = """
<section class="tab-panel" data-panel="routing" id="routing" role="tabpanel">
  <div class="wrap">
    <h2>The routing: the same loop, one level up</h2>
    <p>The membrane answers "what should stay in the context?" The controller
      answers "what should handle this task, and when is it done?" They are the
      same question at different scope, so the machinery carries over:</p>
    <table>
      <thead><tr><th>the memory</th><th>the controller</th></tr></thead>
      <tbody>
        <tr><td>the router scores every chunk</td><td><code>ask(model, prompt)</code> routes to the tier</td></tr>
        <tr><td>dedup decides; drift only penalises</td><td><code>verify</code> decides; <code>score</code> only advises</td></tr>
        <tr><td>the judge is a readout, not the gate</td><td><code>score</code> is consulted only when <code>verify</code> is unknown</td></tr>
        <tr><td>the assembler keeps the facts under budget</td><td>the loop keeps the answer under a step budget</td></tr>
        <tr><td>a false merge deletes a fact</td><td>a false accept ships a wrong answer</td></tr>
      </tbody>
    </table>
    <p>The routing is not a different mechanism. It is the same trade - what to
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
      <div class="card"><h3>The false-positive bar</h3><p>The 176B head set it:
        <strong>0 of 21</strong> correct answers rejected. The checker is
        scored on the 21 correct first, then on the 7 errors.</p></div>
      <div class="card"><h3>Grounding</h3><p>For the factual remainder,
        decompose the answer into claims and check each against the store.
        A check that cannot cite is <code>unknown</code>.</p></div>
      <div class="card"><h3>The status</h3><p><strong>Unmeasured.</strong> The
        sandbox, the check-writer and the grounding verifier are the current
        build. The check runs once per answer and once per mutation; a decoder
        is paid only on the repair. Its time is the one row missing from the
        controller total.</p></div>
    </div>
    <p class="small">The contract and the prior art are in
      <code>docs/HARNESS-CONTROLLER.md</code> and
      <code>docs/CONTROLLER-PRIOR-ART.md</code>.</p>
  </div>
</section>
"""

PANEL_PAPER = f"""
<section class="tab-panel" data-panel="paper" id="paper" role="tabpanel">
  <div class="wrap">
    <h2>The paper</h2>
    <p class="lead">The full paper lives in the repository, versioned with
      the code and the results it describes.</p>
    <p>This page is the shop window; the paper is the record. It carries the
      method, the gold-free contract, the measured tier results and their
      corrections, the negative results I kept, and the controller table.</p>
    <p class="paper-actions">
      <a class="cta" href="{GH}/blob/main/docs/CASCADE-PAPER.md">Read the paper on GitHub</a>
      <a class="cta ghost" href="https://raw.githubusercontent.com/sky-is-green/hivebench/main/docs/CASCADE-PAPER.md">Raw markdown</a>
    </p>
    <div class="grid">
      <div class="card"><h3>The claim</h3><p>On this class, a 35B with a
        check-and-repair loop serves <strong>67/84</strong> over three draws
        against <strong>64/84</strong> without it; a 27B escalation tier adds
        <strong>+5 net</strong>; the 12&times; larger 176B's generation adds
        nothing.</p></div>
      <div class="card"><h3>What it covers</h3><p>The method and the
        gold-free contract; the taxonomy; the measured tier results; the
        learned gate's negative result; the harvest and the reporting; the
        routing table; and what is not claimed.</p></div>
      <div class="card"><h3>The record</h3><p>Every number has a run bundle,
        and every correction stays in
        <a href="{GH}/blob/main/docs/AB-RESULTS.md"><code>docs/AB-RESULTS.md</code></a>.
        The current state is in
        <a href="{GH}/blob/main/docs/STATUS.md"><code>docs/STATUS.md</code></a>.</p></div>
    </div>
  </div>
</section>
"""

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

FOOT = f"""
<footer><div class="wrap"><a href="https://github.com/sky-is-green">sky-is-green</a> · MIT · The paper lives at
  <a class="ref" href="{GH}/blob/main/docs/CASCADE-PAPER.md"><code>docs/CASCADE-PAPER.md</code></a>.
</div></footer>
"""

README = f"""---
title: "The Hive Cascade: detect -> repair -> escalate"
emoji: "🐝"
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
license: mit
---
**A gold-free verification-and-repair loop for local language models.**

This Space is the visual companion. **The paper lives in the repository:**
[docs/CASCADE-PAPER.md]({GH}/blob/main/docs/CASCADE-PAPER.md).

- [The page](https://skyisnotgreen-hive-cascade-paper.static.hf.space/)
- [The code and the results]({GH})
- [The status]({GH}/blob/main/docs/STATUS.md)
- [The task sets](https://huggingface.co/datasets/SkyIsNotGreen/hive-cascade)
"""


JS = """
<script>
(function () {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NAMES = {premise: "the premise", memory: "the memory", routing: "the routing",
                 controller: "the controller",
                 measured: "what is measured", checker: "the checker", paper: "the paper"};
  let mapTimer = null, routeTimer = null;

  // ---- memory: the drone lab
  const VECTORS = {
    a:  [.95, .80, .70, .40, .25, .15, .35, .60],
    ad: [.95, .80, .70, .40, .25, .15, .35, .90],
    b:  [.20, .35, .80, .15, .40, .60, .20, .30],
    c:  [.30, .15, .20, .70, .50, .30, .75, .20],
    k:  [.15, .60, .20, .10, .85, .30, .15, .25]
  };
  function renderVectors() {
    document.querySelectorAll("[data-vec]").forEach(function (el) {
      const vals = VECTORS[el.dataset.vec] || [];
      el.innerHTML = vals.map(function (v) {
        return '<i style="height:' + Math.round(v * 100) + '%"></i>';
      }).join("");
    });
  }
  function setMemory(threshold) {
    const stage = document.querySelector("[data-stage='memory']");
    if (!stage) return;
    const value = Number(threshold);
    const merge = 0.89 >= value;
    const out = stage.querySelector("[data-lab-value]");
    if (out) out.textContent = value.toFixed(2);
    const row = stage.querySelector('[data-lab-row="ad"]');
    if (row) {
      row.classList.toggle("merge", merge);
      row.classList.toggle("selected", !merge);
      const pick = row.querySelector(".pick");
      if (pick) pick.textContent = merge ? "—" : "✓";
    }
    const rowText = stage.querySelector("[data-lab-text]");
    if (rowText) rowText.textContent = merge
      ? "merged into A - the fact is gone"
      : "… soft deletes via deleted_at";
    const d = stage.querySelector("[data-ctx-d]");
    if (d) d.classList.toggle("lost", merge);
    const bar = stage.querySelector("[data-lab-budget]");
    if (bar) {
      bar.style.width = merge ? "55%" : "100%";
      bar.style.background = merge ? "var(--bad)" : "var(--accent2)";
      const budget = bar.parentElement;
      budget.title = merge ? "context budget: 1 of 2 chunks" : "context budget: 2 of 2 chunks";
      budget.setAttribute("aria-label", budget.title);
    }
    const mem = stage.querySelector("[data-lab-membrane]");
    if (mem) mem.textContent = merge ? "1" : "2";
    const cc = stage.querySelector("[data-lab-context-count]");
    if (cc) cc.textContent = merge ? "1" : "2";
    const dec = stage.querySelector("[data-lab-decision]");
    if (dec) dec.innerHTML = merge
      ? "<strong>Merged.</strong> 0.89 &ge; " + value.toFixed(2) + ". A+D merges into A; the delete question can no longer be answered."
      : "<strong>Distinct.</strong> 0.89 &lt; " + value.toFixed(2) + ". Both are selected, and the context can answer the delete question.";
    const ctx = stage.querySelector("[data-lab-context]");
    if (ctx) ctx.textContent = merge
      ? "1 of 2 chunks. The delete question is unanswerable."
      : "2 of 2 chunks. The delete question can be answered.";
    const cap = stage.querySelector("[data-lab-caption]");
    if (cap) cap.innerHTML = merge
      ? "The merge removes A+D before it can be selected: the prompt's only evidence for deletes is gone. The mechanism behind 60.51% &rarr; 32.66%."
      : "The query selects A (0.74) and A+D (0.91); the budget keeps both. The prompt's context contains the soft-delete fact.";
  }

  // ---- hero: the process strip (driven by the example on the controller tab)
  const FLOW_COLORS = {
    memory: "var(--c-mem)", prompt: "var(--c-prompt)", "35b": "var(--c-model)",
    judge: "var(--c-judge)", verify: "var(--good)", "27b": "var(--c-model)",
    answer: "var(--good)", api: "var(--c-exec)"
  };
  const FLOW_ORDER = ["prompt", "35b", "judge", "verify", "27b", "answer"];
  let heroTimer = null;

  function heroSet(nodeName, color) {
    const flow = document.querySelector("[data-flow]");
    if (!flow) return;
    const value = color || FLOW_COLORS[nodeName] || "var(--accent)";
    flow.querySelectorAll(".node").forEach(function (n) {
      n.classList.toggle("active", n.dataset.flowNode === nodeName);
    });
    flow.querySelectorAll(".link").forEach(function (l) {
      const on = l.dataset.to === nodeName;
      l.classList.toggle("active", on);
      if (on) l.style.setProperty("--packet", value);
    });
    const node = flow.querySelector('[data-flow-node="' + nodeName + '"]');
    if (node) node.style.setProperty("--packet", value);
    if (nodeName === "prompt") heroExchange(2400);
  }
  function heroExchange(duration) {
    const link = document.querySelector('[data-to="prompt"]');
    if (!link) return;
    link.classList.remove("exchange");
    void link.offsetWidth;
    link.classList.add("exchange");
    if (link._exTimer) clearTimeout(link._exTimer);
    link._exTimer = setTimeout(function () { link.classList.remove("exchange"); },
                               duration || 2600);
  }
  function stopHero() { if (heroTimer) { clearInterval(heroTimer); heroTimer = null; } }
  function startHero() {
    stopHero();
    if (reduced) { heroSet("answer"); return; }
    let i = 0;
    heroSet(FLOW_ORDER[0]);
    heroTimer = setInterval(function () {
      i = (i + 1) % FLOW_ORDER.length;
      heroSet(FLOW_ORDER[i]);
    }, 1900);
  }

  // ---- routing: highlight the correspondence, one row at a time
  const MAP_CAPTIONS = [
    "The chunk router. The controller does the same, one level up.",
    "Dedup removes; drift only demotes. Verify rejects; score only advises. The hard side is the one whose threshold is constrained.",
    "A readout is never the gate - in either place.",
    "The consequence moves from the context (a deleted fact) to the answer (a wrong one)."
  ];
  function setMap(index) {
    const value = index % 4;
    document.querySelectorAll("[data-stage='routing'] .map-row").forEach(function (row) {
      row.classList.toggle("active", Number(row.dataset.i) === value);
    });
    const cap = document.querySelector("[data-routing-caption]");
    if (cap) cap.innerHTML = MAP_CAPTIONS[value];
  }
  function startMap() {
    stopMap(); setMap(0);
    mapTimer = setInterval(function () {
      const active = document.querySelector("[data-stage='routing'] .map-row.active");
      const next = active ? Number(active.dataset.i) + 1 : 0;
      setMap(next);
    }, 1800);
  }
  function stopMap() { if (mapTimer) { clearInterval(mapTimer); mapTimer = null; } }
  const mapStage = document.querySelector("[data-map]");
  if (mapStage) {
    mapStage.querySelectorAll(".map-row").forEach(function (row) {
      function focusRow() {
        stopMap(); setMap(Number(row.dataset.i));
        mapStage.querySelectorAll(".map-row").forEach(function (r) { r.classList.remove("show-tip"); });
        row.classList.add("show-tip");
      }
      row.addEventListener("mouseenter", focusRow);
      row.addEventListener("focus", focusRow);
      row.addEventListener("mouseleave", function () { row.classList.remove("show-tip"); });
      row.addEventListener("blur", function () { row.classList.remove("show-tip"); });
    });
    mapStage.addEventListener("mouseleave", function () {
      mapStage.querySelectorAll(".map-row").forEach(function (r) { r.classList.remove("show-tip"); });
      startMap();
    });
    mapStage.addEventListener("focusout", function (e) {
      if (!mapStage.contains(e.relatedTarget)) startMap();
    });
  }

  // ---- controller: the hub and its tool calls
  const STEPS = [
    {ins: [["e-prompt-ctl", "var(--c-prompt)"], ["e-mem-ctl", "var(--c-mem)"]],
     node: "controller",
     caption: "The controller takes the prompt and the memory context. Every call below starts and ends here - the models never talk to each other."},
    {out: "e-ctl-35b", back: "e-35b-ctl", color: "var(--c-model)", reply: "var(--c-model)",
     node: "35b", reveal: "35b",
     caption: "It calls the 35B with the task and the context. The candidate comes back: <strong>24.50</strong>."},
    {out: "e-ctl-judge", back: "e-judge-ctl", color: "var(--c-model)", reply: "var(--c-judge)",
     node: "judge", reveal: "judge", keep: true,
     caption: "It calls the judge. The score comes back: <strong>p = 0.973</strong>. The judge would keep it (dashed)."},
    {out: "e-ctl-verify", back: "e-verify-ctl", color: "var(--c-exec)", reply: "var(--bad)",
     node: "verify", reveal: "verify", state: "fail", strikeKeep: true,
     caption: "It calls execution. 700&times;5 + 1,400&times;15 = <strong>24,500&cent; = $245</strong>; 24.50 fails. The keep path is struck."},
    {out: "e-ctl-27b", back: "e-27b-ctl", color: "var(--c-model)", reply: "var(--c-model)",
     node: "27b", reveal: "27b",
     caption: "It escalates: the task, the answer and the failure go to the 27B. <strong>245</strong> comes back."},
    {out: "e-ctl-verify", back: "e-verify-ctl", color: "var(--c-exec)", reply: "var(--good)",
     node: "verify", reveal: "verify", state: "pass", accept: true,
     caption: "It calls execution again: 245 passes. The controller accepts - the judge never had the last word."}
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
    const val = stage.querySelector('[data-val="verify"]');
    if (val) { val.textContent = "\u2014"; val.classList.remove("fail", "pass"); }
    const packet = document.getElementById("packet");
    if (packet) packet.style.opacity = "0";
    const group = document.getElementById("packets");
    if (group) group.replaceChildren();
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

  function spawnParticle(path, color, reverse, delay) {
    const group = document.getElementById("packets");
    if (!group || reduced) return;
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("r", "3");
    dot.setAttribute("fill", color);
    dot.setAttribute("opacity", "0");
    group.appendChild(dot);
    const len = path.getTotalLength();
    const t0 = performance.now() + delay;
    function frame(now) {
      if (now < t0) { requestAnimationFrame(frame); return; }
      const p = Math.min((now - t0) / 950, 1);
      const pt = path.getPointAtLength(len * (reverse ? 1 - p : p));
      dot.setAttribute("cx", pt.x);
      dot.setAttribute("cy", pt.y);
      dot.setAttribute("opacity", p > 0.05 && p < 0.95 ? "0.9" : "0");
      if (p < 1) requestAnimationFrame(frame); else dot.remove();
    }
    requestAnimationFrame(frame);
  }

  function burst(specs) {
    const group = document.getElementById("packets");
    if (group) group.replaceChildren();
    (specs || []).forEach(function (spec) {
      const path = document.getElementById(spec.edge);
      if (!path) return;
      for (let k = 0; k < (spec.count || 2); k++) {
        spawnParticle(path, spec.color, !!spec.reverse, 120 + k * 170);
      }
    });
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
    const hub = stage.querySelector('[data-node="controller"]');
    if (hub) hub.classList.add("active");
    const node = stage.querySelector('[data-node="' + step.node + '"]');
    if (node) {
      node.classList.add("active");
      if (step.reveal) node.classList.add("revealed");
    }
    if (step.ins) {
      step.ins.forEach(function (pair) {
        const e = document.getElementById(pair[0]);
        if (e) { e.classList.add("active"); e.style.stroke = pair[1]; }
      });
      const first = document.getElementById(step.ins[0][0]);
      if (first) travel(first, step.ins[0][1], 850);
      burst(step.ins.map(function (pair) {
        return {edge: pair[0], color: pair[1], count: 2};
      }));
    }
    if (step.out) {
      const out = document.getElementById(step.out);
      if (out) {
        out.classList.add("active");
        out.style.stroke = step.color;
        travel(out, step.color, 850);
      }
      burst([{edge: step.out, color: step.color, count: 2},
             {edge: step.back, color: step.reply, count: 2}]);
    }
    if (step.keep) {
      const keep = document.getElementById("e-ctl-answer-keep");
      if (keep) { keep.classList.add("active"); keep.style.stroke = "var(--c-judge)"; }
    }
    if (step.strikeKeep) {
      const keep = document.getElementById("e-ctl-answer-keep");
      if (keep) { keep.classList.remove("active"); keep.classList.add("struck"); keep.style.stroke = ""; }
    }
    if (step.state === "fail") {
      const v = stage.querySelector('[data-node="verify"]');
      if (v) {
        v.classList.add("fail");
        const val = v.querySelector(".val");
        if (val) { val.textContent = "FAIL"; val.classList.add("fail"); }
      }
    }
    if (step.state === "pass") {
      const v = stage.querySelector('[data-node="verify"]');
      if (v) {
        v.classList.remove("fail");
        v.classList.add("pass");
        const val = v.querySelector(".val");
        if (val) { val.textContent = "PASS"; val.classList.remove("fail"); val.classList.add("pass"); }
      }
    }
    if (step.accept) {
      const a = stage.querySelector('[data-node="answer"]');
      if (a) { a.classList.add("revealed"); a.classList.add("pass"); }
      const e = document.getElementById("e-ctl-answer");
      if (e) {
        e.classList.add("active");
        e.style.stroke = "var(--good)";
        travel(e, "var(--good)", 850);
      }
      burst([{edge: "e-ctl-answer", color: "var(--good)", count: 3}]);
    }
    const timing = stage.querySelector(".timing");
    if (timing) {
      timing.querySelectorAll("[data-time-step]").forEach(function (s) {
        s.classList.toggle("active", Number(s.dataset.timeStep) === i);
      });
    }
    const cap = document.querySelector("[data-route-caption]");
    if (cap) cap.innerHTML = step.caption;
    const heroMap = [["prompt", "var(--c-prompt)"], ["35b", "var(--c-model)"],
                     ["judge", "var(--c-judge)"], ["verify", "var(--bad)"],
                     ["27b", "var(--c-model)"], ["answer", "var(--good)"]];
    const h = heroMap[i];
    if (h) heroSet(h[0], h[1]);
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
    }, 2800);
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
    const code = stage.querySelector("[data-check-code]");
    if (code) code.textContent = real
      ? "assert float(answer) == (700*5 + 1400*15) / 100"
      : 'print("PASS")';
    const a = stage.querySelector("[data-run-a]");
    const b = stage.querySelector("[data-run-b]");
    if (a) { a.textContent = "PASS"; a.className = "verdict pass"; }
    if (b) {
      b.textContent = real ? "FAIL" : "PASS";
      b.className = "verdict " + (real ? "fail" : "pass");
    }
    const out = stage.querySelector("[data-outcome]");
    if (out) out.textContent = real
      ? "Accepted - the check discriminates between the answer and its mutation"
      : "Unknown - the check passes the mutation, so it is not evidence";
  }

  // ---- tabs
  function activate(name, push) {
    if (!NAMES[name]) name = "premise";
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
    if (name !== "routing") stopMap(); else startMap();
    if (name !== "controller") stopRoute();
    stopHero();
    if (name !== "controller") startHero();
    if (name === "measured") animateBars();
    if (name === "controller" && !reduced) playRoute();
    if (push !== false) history.replaceState(null, "", "#" + name);
  }

  // ---- wire
  renderVectors();
  setMemory(0.92);
  const labSlider = document.querySelector("[data-lab-threshold]");
  if (labSlider) {
    labSlider.addEventListener("input", function () { setMemory(labSlider.value); });
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
  document.querySelectorAll("[data-goto]").forEach(function (b) {
    b.addEventListener("click", function () {
      activate(b.dataset.goto);
      document.querySelector(".tabs").scrollIntoView({ block: "start" });
    });
  });

  const start = (location.hash || "#premise").slice(1);
  activate(NAMES[start] ? start : "premise", false);
  if (reduced) {
    document.querySelectorAll("[data-route-play]").forEach(function (b) { b.hidden = true; });
    stopRoute();
    setRouteStep(STEPS.length - 1);
  }
})();
</script>
"""


def build() -> tuple[str, str, str]:
    html = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>The Hive Cascade - from memory to a global controller</title>\n"
        '<meta name="description" content="A gold-free verification loop for local language models: the memory, the path to a global controller, and what is measured.">\n'
        '<link rel="stylesheet" href="style.css">\n</head>\n<body>\n'
        + HERO + TABS + STAGE_OPEN + STAGE_PREMISE + STAGE_MEMORY + STAGE_ROUTING + STAGE_CONTROLLER
        + STAGE_MEASURED + STAGE_CHECKER + STAGE_PAPER + STAGE_CLOSE
        + "<main>\n" + PANEL_PREMISE + PANEL_MEMORY + PANEL_ROUTING + PANEL_CONTROLLER
        + PANEL_MEASURED + PANEL_CHECKER + PANEL_PAPER
        + "</main>\n" + LINKS + FOOT + JS
        + "\n</body>\n</html>\n"
    )
    return html, CSS, README


def main(argv: list[str]) -> int:
    html, css, readme = build()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / "style.css").write_text(css, encoding="utf-8")
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {OUT/'index.html'} ({len(html)} chars), style.css and README.md")
    if "--upload" in argv:
        from huggingface_hub import HfApi

        api = HfApi()
        for name in ("index.html", "style.css", "README.md"):
            api.upload_file(path_or_fileobj=str(OUT / name), path_in_repo=name,
                            repo_id=SPACE, repo_type="space",
                            commit_message="The page; the paper lives in the repository")
        print(f"uploaded to https://huggingface.co/spaces/{SPACE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
