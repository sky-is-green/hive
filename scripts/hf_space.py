"""Render and publish the HF Space page for the cascade paper.

The page is the narrative, not just the paper: the memory system and its
measured lesson, the move to the global controller, two illustrative
demos (a curated turn and a routed task), the measured table, and the
paper itself in a collapsible section.

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

CSS = """
:root {
  --bg: #fbfaf8; --fg: #171614; --muted: #6b675f; --line: #e4e0d8;
  --card: #ffffff; --accent: #b45309; --accent2: #0f766e; --code: #f3efe7;
  --bad: #b91c1c; --good: #15803d; --hero: linear-gradient(160deg, #fef3c7 0%, #fbfaf8 70%);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12110f; --fg: #ece8e1; --muted: #a29c90; --line: #2c2a26;
    --card: #1a1815; --accent: #fbbf24; --accent2: #5eead4; --code: #22201c;
    --bad: #f87171; --good: #4ade80; --hero: linear-gradient(160deg, #2a2113 0%, #12110f 70%);
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 17px/1.65 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 920px; margin: 0 auto; padding: 0 24px; }
.nav { position: sticky; top: 0; z-index: 20; background: color-mix(in srgb, var(--bg) 88%, transparent);
  backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); }
.nav .wrap { display: flex; gap: 18px; overflow-x: auto; padding-top: 12px; padding-bottom: 12px;
  font-size: .84rem; }
.nav a { color: var(--muted); text-decoration: none; white-space: nowrap; }
.nav a:hover { color: var(--fg); }
.hero { background: var(--hero); border-bottom: 1px solid var(--line); padding: 64px 0 40px; }
.eyebrow { color: var(--accent); font-size: 13px; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase; }
.eyebrow a { color: inherit; text-decoration: none;
  border-bottom: 1px dotted color-mix(in srgb, var(--accent) 55%, transparent); }
.eyebrow a:hover { border-bottom-style: solid; }
.hero h1 { font-size: clamp(2.2rem, 5vw, 3.4rem); line-height: 1.08;
  letter-spacing: -.03em; margin: .35em 0 .25em; }
.lede { font-size: 1.22rem; max-width: 46ch; color: var(--muted); margin: 0; }
.meta { font-size: .85rem; color: var(--muted); margin-top: 1.3rem; }
.meta .dot { display: inline-block; width: .55em; height: .55em; background: var(--good);
  border-radius: 50%; margin-right: .45em; vertical-align: .06em; }
.flow { display: flex; align-items: center; gap: 6px; margin-top: 26px; flex-wrap: wrap;
  font-size: .8rem; }
.flow .node { border: 1px solid var(--line); background: var(--card); border-radius: 8px;
  padding: 6px 10px; font-family: ui-monospace, monospace; }
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
main { padding: 28px 0 24px; }
h2 { font-size: 1.5rem; letter-spacing: -.01em; margin: 2.8rem 0 .5rem;
  padding-bottom: .35rem; border-bottom: 1px solid var(--line); }
h3 { font-size: 1.12rem; margin: 1.6rem 0 .35rem; }
p { margin: .8rem 0; }
a { color: var(--accent); text-underline-offset: 3px; }
.sub { color: var(--muted); margin-top: -.15rem; }
.small { font-size: .86rem; color: var(--muted); }
table { width: 100%; border-collapse: collapse; margin: 1rem 0 1.4rem;
  font-size: .92rem; display: block; overflow-x: auto; }
th, td { text-align: left; padding: .55rem .7rem; border-bottom: 1px solid var(--line);
  vertical-align: top; }
th { color: var(--muted); font-size: .76rem; text-transform: uppercase;
  letter-spacing: .06em; font-weight: 600; }
code { background: var(--code); padding: .12em .38em; border-radius: 4px;
  font-size: .9em; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
pre { background: var(--code); padding: 1rem 1.1rem; border-radius: 8px;
  overflow-x: auto; font-size: .86rem; line-height: 1.5; }
pre code { background: none; padding: 0; }
a.ref { text-decoration: none; border-bottom: 1px dotted var(--accent); }
a.ref code { color: inherit; }
blockquote { margin: 1.2rem 0; padding: .1rem 0 .1rem 1rem;
  border-left: 3px solid var(--accent); color: var(--muted); }
hr { border: 0; border-top: 1px solid var(--line); margin: 2.4rem 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 14px; margin: 1.1rem 0 1.4rem; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px 16px 14px; }
.card h3 { margin: 0 0 .3rem; font-size: .98rem; }
.card p { margin: 0; font-size: .88rem; color: var(--muted); }
.note { background: var(--code); border-left: 3px solid var(--accent);
  padding: .8rem 1rem; border-radius: 0 6px 6px 0; font-size: .9rem; }
.note.bad { border-left-color: var(--bad); }
.note.good { border-left-color: var(--good); }
.pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px;
  padding: .1rem .55rem; font-size: .76rem; color: var(--muted); margin-right: .35rem; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 760px) { .two { grid-template-columns: 1fr; } }

/* demos */
.demo { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 18px; margin: 1.2rem 0 1.6rem; }
.demo-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.demo-head h3 { margin: 0; }
.demo button { font: inherit; font-size: .8rem; padding: .28rem .7rem; border-radius: 999px;
  border: 1px solid var(--line); background: var(--bg); color: var(--fg); cursor: pointer; }
.demo button:hover { border-color: var(--accent); color: var(--accent); }
.caption { min-height: 3.2em; margin: .8rem 0 0; font-size: .9rem; color: var(--muted); }
.caption strong { color: var(--fg); }
.chunks { display: grid; gap: 8px; margin-top: 12px; }
.chunk { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center;
  border: 1px solid var(--line); border-radius: 8px; padding: 9px 11px; font-size: .86rem;
  transition: opacity .4s, transform .4s, border-color .4s, max-height .4s;
  background: var(--bg); }
.chunk .tag { font-size: .7rem; color: var(--muted); font-family: ui-monospace, monospace; }
.chunk.merged { opacity: .28; border-style: dashed; }
.chunk.kept { border-color: var(--good); }
.chunk.dimmed { opacity: .4; border-color: var(--accent); }
.chunk.dropped { opacity: .18; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 999px;
  overflow: hidden; }
.seg button { border: 0; border-radius: 0; padding: .26rem .68rem;
  background: transparent; color: var(--muted); }
.seg button[aria-pressed="true"] { background: var(--accent); color: #fff; }
@media (prefers-color-scheme: dark) { .seg button[aria-pressed="true"] { color: #1a1815; } }
.ledger { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 12px; }
.fact { font-size: .72rem; font-family: ui-monospace, monospace;
  border: 1px solid var(--line); border-radius: 999px; padding: .1rem .5rem;
  color: var(--good); transition: color .3s, border-color .3s, opacity .3s; }
.fact.lost { color: var(--bad); border-color: var(--bad);
  text-decoration: line-through; opacity: .8; }
.budget { height: 8px; background: var(--line); border-radius: 999px; overflow: hidden;
  margin-top: 12px; }
.budget i { display: block; height: 100%; background: var(--accent2); width: 100%;
  transition: width .8s ease; }
.route { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-top: 14px; }
@media (max-width: 760px) { .route { grid-template-columns: 1fr; } }
.station { border: 1px solid var(--line); border-radius: 9px; padding: 10px; min-height: 84px;
  background: var(--bg); transition: border-color .3s, box-shadow .3s; }
.station .name { font-family: ui-monospace, monospace; font-size: .76rem; color: var(--muted); }
.station .val { font-size: .95rem; margin-top: 6px; }
.station.active { border-color: var(--accent); box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 18%, transparent); }
.station.pass { border-color: var(--good); }
.station.fail { border-color: var(--bad); }
.q { font-size: .86rem; color: var(--muted); border-left: 2px solid var(--line);
  padding-left: 10px; margin: 10px 0 0; }
details { border: 1px solid var(--line); border-radius: 10px; padding: 12px 18px;
  background: var(--card); margin-top: 1rem; }
details summary { cursor: pointer; font-weight: 600; }
details[open] summary { margin-bottom: .8rem; }
footer { border-top: 1px solid var(--line); padding: 26px 0 48px; color: var(--muted);
  font-size: .86rem; }
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

NAV = """
<nav class="nav"><div class="wrap">
  <a href="#memory">the memory</a>
  <a href="#move">the move</a>
  <a href="#controller">the controller</a>
  <a href="#measured">what is measured</a>
  <a href="#checker">the checker</a>
  <a href="#paper">the paper</a>
</div></nav>
"""

MEMORY = """
<section id="memory"><div class="wrap">
  <h2>The memory: what the drone learned</h2>
  <p>The first harness curated <em>within</em> a conversation. A 17M
    encoder - the drone - scores every stored chunk; the membrane makes two
    decisions on every turn: <strong>dedup</strong> (is this chunk already
    said?) and <strong>drift</strong> (did the topic change?), and the
    assembler keeps the context under budget.</p>

  <div class="demo" data-demo="curation" data-threshold="0.92">
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

  <p class="small">The example is illustrative. The numbers below are the
    measured replay (<a class="ref" href="https://github.com/sky-is-green/hivebench/blob/main/docs/AB-RESULTS.md"><code>docs/AB-RESULTS.md</code></a> §2C-quater).</p>

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
</div></section>
"""

MOVE = """
<section id="move"><div class="wrap">
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
</div></section>
"""

CONTROLLER = """
<section id="controller"><div class="wrap">
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

  <div class="demo" data-demo="routing">
    <div class="demo-head"><h3>A routed task, from the measured run</h3>
      <button type="button">Play</button></div>
    <p class="q">Anthony sends 700 small coupons at 5&cent; and twice as many
      big ones at 15&cent;. What is the total postage?</p>
    <div class="route">
      <div class="station" data-station="0"><div class="name">35B</div><div class="val">24.50</div></div>
      <div class="station" data-station="1"><div class="name">judge</div><div class="val">p = 0.973</div></div>
      <div class="station" data-station="2"><div class="name">verify</div><div class="val">FAIL</div></div>
      <div class="station" data-station="3"><div class="name">27B</div><div class="val">245</div></div>
      <div class="station" data-station="4"><div class="name">verify</div><div class="val">PASS</div></div>
    </div>
    <p class="caption" aria-live="polite">The 35B answers <strong>24.50</strong>. Press Play.</p>
  </div>

  <p>The judge read the answer and wanted to keep it (<strong>p = 0.973</strong>).
    The checker recomputed it: 700&times;5 + 1,400&times;15 = <strong>24,500&cent;
    = $245</strong>. The judge has a blind spot exactly where the answer is
    <em>plausible</em>; the checker does not. This is one of the four errors
    at <code>p &ge; 0.97</code>.</p>

  <p class="note good"><strong>The loop is bounded.</strong> After a configured
    number of steps the harness removes the tools and requires a final answer:
    <code>accepted</code>, <code>escalated</code> or <code>rejected</code>. The
    invariants are enforced by the tool surface and the policy layer, not by
    the prompt.</p>
</div></section>
"""

MEASURED = """
<section id="measured"><div class="wrap">
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
</div></section>
"""

CHECKER = """
<section id="checker"><div class="wrap">
  <h2>The checker: two verifiers, one verdict</h2>
  <p><strong>Execution</strong> where a program can settle it (run the code,
    recompute the arithmetic); <strong>grounding</strong> where only evidence
    can (claims against the retrieved context, each cited). They sit behind
    one <code>verify</code>; the log records the method.</p>
  <div class="grid">
    <div class="card"><h3>falsifiability</h3><p>A check that passes a
      deliberately mutated answer is rejected as <code>unknown</code>. That
      is the anti-<code>print("PASS")</code> rule - and what makes a small
      checker safe.</p></div>
    <div class="card"><h3>the false-positive bar</h3><p>The 176B head set it:
      <strong>0 of 21</strong> correct answers rejected. The checker is
      scored on the 21 correct first, then on the 7 errors.</p></div>
    <div class="card"><h3>grounding</h3><p>For the factual remainder,
      decompose the answer into claims and check each against the store.
      A check that cannot cite is <code>unknown</code>.</p></div>
  </div>
  <p class="small">This is the last unmeasured tier. The contract and the
    prior art are in <a class="ref" href="https://github.com/sky-is-green/hivebench/blob/main/docs/HARNESS-CONTROLLER.md"><code>docs/HARNESS-CONTROLLER.md</code></a> and
    <a class="ref" href="https://github.com/sky-is-green/hivebench/blob/main/docs/CONTROLLER-PRIOR-ART.md"><code>docs/CONTROLLER-PRIOR-ART.md</code></a>.</p>
</div></section>
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

FOOT = """
<footer><div class="wrap">HiveBench Studio contributors · MIT · the paper is
  also in the repository at <a class="ref" href="https://github.com/sky-is-green/hivebench/blob/main/docs/CASCADE-PAPER.md"><code>docs/CASCADE-PAPER.md</code></a>.
</div></footer>
"""

JS = """
<script>
(function () {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const timers = new WeakMap();

  const CURATION_CAPTIONS = {
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
    if (d) {
      d.classList.toggle("lost", merge);
      d.textContent = merge ? "D - lost" : "D - soft deletes";
    }
    const bar = demo.querySelector(".budget i");
    if (bar) bar.style.width = merge ? "55%" : "100%";
    const cap = demo.querySelector(".caption");
    if (cap) cap.innerHTML = CURATION_CAPTIONS[threshold];
    demo.querySelectorAll(".seg button").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.threshold === threshold));
    });
  }

  const ROUTING = [
    ["The 35B answers <strong>24.50</strong>. The task looks settled.", 0],
    ["The judge reads it: <strong>p = 0.973</strong>. Plausible - so it keeps it. This is one of the four errors at p &ge; 0.97.", 1],
    ["The checker computes 700&times;5 + 1,400&times;15 = <strong>24,500&cent; = $245</strong>. The candidate says 24.50: <strong>FAIL</strong>.", 2],
    ["The 27B answers <strong>245</strong>.", 3],
    ["The checker recomputes: <strong>PASS</strong>. Two tiers, one verdict, and the judge never had the last word.", 4],
  ];

  function setRouting(demo, step) {
    demo.querySelectorAll(".station").forEach(function (el) {
      const i = Number(el.dataset.station);
      el.classList.toggle("active", i === step);
      el.classList.remove("pass", "fail");
      if (step >= 2 && i === 2) el.classList.add("fail");
      if (step >= 4 && i === 4) el.classList.add("pass");
      if (step >= 4 && i === 2) el.classList.remove("fail");
    });
  }

  function renderRouting(demo, step) {
    const caption = demo.querySelector(".caption");
    if (caption) caption.innerHTML = ROUTING[step][0];
    setRouting(demo, step);
    demo.dataset.step = String(step);
  }

  function stop(demo) {
    const t = timers.get(demo);
    if (t) { clearInterval(t); timers.delete(demo); }
  }

  function play(demo) {
    stop(demo);
    let i = Number(demo.dataset.step || 0);
    if (i >= ROUTING.length - 1) i = -1;
    const tick = function () {
      i = (i + 1) % ROUTING.length;
      renderRouting(demo, i);
      if (i === ROUTING.length - 1) stop(demo);
    };
    tick();
    timers.set(demo, setInterval(tick, 2400));
  }

  document.querySelectorAll(".demo").forEach(function (demo) {
    if (demo.dataset.demo === "curation") {
      setCuration(demo, "0.92");
      demo.querySelectorAll(".seg button").forEach(function (b) {
        b.addEventListener("click", function () { setCuration(demo, b.dataset.threshold); });
      });
      return;
    }
    renderRouting(demo, 0);
    const button = demo.querySelector("button");
    if (!button) return;
    if (reduced) { renderRouting(demo, ROUTING.length - 1); button.hidden = true; return; }
    button.addEventListener("click", function () { play(demo); });
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { if (!e.isIntersecting) stop(demo); });
      }, { threshold: 0.2 });
      io.observe(demo);
    }
  });
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
    # the hero carries the title; drop the paper's own H1 so the page has one
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
        + HERO + NAV + "<main>\n" + MEMORY + MOVE + CONTROLLER + MEASURED
        + CHECKER
        + '<section id="paper"><div class="wrap"><h2>The paper</h2>'
        + '<details><summary>The full text (also in <a class="ref" href="https://github.com/sky-is-green/hivebench/blob/main/docs/CASCADE-PAPER.md"><code>docs/CASCADE-PAPER.md</code></a>)</summary>'
        + body_html + "</details></div></section>\n"
        + LINKS + "</main>\n" + FOOT + JS + "\n</body>\n</html>\n"
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
                            commit_message="The narrative page: memory -> controller, with the curated-turn and routed-task demos")
        print(f"uploaded to https://huggingface.co/spaces/{SPACE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
