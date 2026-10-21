"""Render and publish the HF Space page for the cascade paper.

The Hub's long-form article format is a Blog Article, which needs PRO;
a static Space is the free equivalent, but a static Space serves
``index.html`` - the ``README.md`` alone renders as the repo card, not as
the page. This builds the page from ``docs/CASCADE-PAPER.md``: a hero,
the measured table, the four controller tools, and the paper.

    python scripts/hf_space.py            # render to /tmp/opencode/hf-space
    python scripts/hf_space.py --upload   # publish to the Space
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPACE = "SkyIsNotGreen/hive-cascade-paper"
OUT = Path("/tmp/opencode/hf-space")

CSS = """
:root {
  --bg: #fbfaf8; --fg: #171614; --muted: #6b675f; --line: #e4e0d8;
  --card: #ffffff; --accent: #b45309; --code: #f3efe7;
  --hero: linear-gradient(160deg, #fef3c7 0%, #fbfaf8 70%);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12110f; --fg: #ece8e1; --muted: #a29c90; --line: #2c2a26;
    --card: #1a1815; --accent: #fbbf24; --code: #22201c;
    --hero: linear-gradient(160deg, #2a2113 0%, #12110f 70%);
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 17px/1.65 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 880px; margin: 0 auto; padding: 0 24px; }
.hero { background: var(--hero); border-bottom: 1px solid var(--line); padding: 72px 0 48px; }
.eyebrow { color: var(--accent); font-size: 13px; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase; }
.hero h1 { font-size: clamp(2.2rem, 5vw, 3.4rem); line-height: 1.08;
  letter-spacing: -.03em; margin: .35em 0 .25em; }
.lede { font-size: 1.25rem; max-width: 42ch; color: var(--muted); margin: 0; }
.meta { font-size: .85rem; color: var(--muted); margin-top: 1.6rem; }
.meta .dot { display: inline-block; width: .55em; height: .55em; background: #22c55e;
  border-radius: 50%; margin-right: .45em; vertical-align: .06em; }
main { padding: 48px 0 24px; }
h2 { font-size: 1.55rem; letter-spacing: -.01em; margin: 2.8rem 0 .6rem;
  padding-bottom: .35rem; border-bottom: 1px solid var(--line); }
h3 { font-size: 1.15rem; margin: 1.8rem 0 .4rem; }
p { margin: .8rem 0; }
a { color: var(--accent); text-underline-offset: 3px; }
.sub { color: var(--muted); margin-top: -.2rem; }
table { width: 100%; border-collapse: collapse; margin: 1rem 0 1.4rem;
  font-size: .93rem; display: block; overflow-x: auto; }
th, td { text-align: left; padding: .55rem .7rem; border-bottom: 1px solid var(--line);
  vertical-align: top; }
th { color: var(--muted); font-size: .78rem; text-transform: uppercase;
  letter-spacing: .06em; font-weight: 600; }
code { background: var(--code); padding: .12em .38em; border-radius: 4px;
  font-size: .9em; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
pre { background: var(--code); padding: 1rem 1.1rem; border-radius: 8px;
  overflow-x: auto; font-size: .87rem; line-height: 1.5; }
pre code { background: none; padding: 0; }
blockquote { margin: 1.2rem 0; padding: .1rem 0 .1rem 1rem;
  border-left: 3px solid var(--accent); color: var(--muted); }
hr { border: 0; border-top: 1px solid var(--line); margin: 2.5rem 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 14px; margin: 1.2rem 0 1.6rem; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 18px 18px 16px; }
.card h3 { margin: 0 0 .35rem; font-size: 1rem; }
.card h3 code { font-size: .95rem; }
.card p { margin: 0; font-size: .9rem; color: var(--muted); }
.note { background: var(--code); border-left: 3px solid var(--accent);
  padding: .8rem 1rem; border-radius: 0 6px 6px 0; font-size: .92rem; }
footer { border-top: 1px solid var(--line); padding: 28px 0 48px;
  color: var(--muted); font-size: .88rem; }
footer a { color: inherit; }
.pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px;
  padding: .12rem .6rem; font-size: .78rem; color: var(--muted); margin-right: .4rem; }
"""

HERO = """
<header class="hero">
  <div class="wrap">
    <div class="eyebrow">HiveBench Studio · research preview</div>
    <h1>The Hive Cascade</h1>
    <p class="lede">Detect → repair → escalate. A gold-free verification loop
      for local language models.</p>
    <p class="meta"><span class="dot"></span>not a product yet ·
      <span class="pill">MIT</span><span class="pill">updated 2026-10-21</span></p>
  </div>
</header>
"""

MEASURED = """
<section id="measured">
  <div class="wrap">
    <h2>What is measured</h2>
    <p class="sub">The same 28 organic tasks throughout.</p>
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
        <tr><td>the checker (execution + grounding)</td>
            <td>unmeasured: the next build</td><td>a sandbox call</td></tr>
      </tbody>
    </table>
    <p class="note">The head's four misses are plausible answers at
      <code>p &gt;= 0.97</code> - arithmetic and code. That is why the next
      tier is a checker, not a better judge.</p>
  </div>
</section>
"""

TOOLS = """
<section id="tools">
  <div class="wrap">
    <h2>The four tools</h2>
    <p class="sub">The drop-in surface. Any controller can call these; the
      harness enforces the invariants, not the prompt.</p>
    <div class="grid">
      <div class="card"><h3><code>ask(model, prompt)</code></h3>
        <p>Route the task to a tier: the 35B, the 27B, or the API.</p></div>
      <div class="card"><h3><code>verify(task, answer)</code></h3>
        <p>Execution or claim-level grounding. <code>pass / fail / unknown</code>.
        A check that cannot fail a mutation is <code>unknown</code>.</p></div>
      <div class="card"><h3><code>score(task, answer)</code></h3>
        <p>The judge's <code>p(correct)</code>. Consulted only when
        <code>verify</code> is <code>unknown</code>; never overrides a check.</p></div>
      <div class="card"><h3><code>ask_human(question, answer, evidence)</code></h3>
        <p>The last local tier: solve, escalate or reject - with the evidence.</p></div>
    </div>
    <p>The loop is bounded: after a configured number of steps the harness
      removes the tools and requires a final answer. The controller is a
      router and a verifier, not a knower - which is why it can be a small
      model.</p>
  </div>
</section>
"""

LINKS = """
<section id="links">
  <div class="wrap">
    <h2>Where the rest is</h2>
    <table>
      <tr><td>the code, the full results log and the corrections</td>
          <td><a href="https://github.com/sky-is-green/hivebench">github.com/sky-is-green/hivebench</a></td></tr>
      <tr><td>the status: what is measured, invalid, unmeasured, blocked</td>
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
    <p class="note"><strong>Corrections are part of the record.</strong> The
      drift numbers in the encoder sections were measured on a corpus that
      included an ingested attachment and are withdrawn; the tuned dedup
      threshold is rejected by the replay. The log keeps both, and
      <code>docs/STATUS.md</code> says which is which.</p>
  </div>
</section>
"""

FOOT = """
<footer><div class="wrap">
  HiveBench Studio contributors · MIT · the paper is also in the repository
  at <code>docs/CASCADE-PAPER.md</code>.
</div></footer>
"""


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
    body_html = markdown.markdown(
        body_md, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Hive Cascade - detect, repair, escalate</title>
<meta name="description" content="A gold-free verification loop for local language models.">
<link rel="stylesheet" href="style.css">
</head>
<body>
{HERO}
<main>
{MEASURED}
{TOOLS}
<section id="paper"><div class="wrap">
<h2>The paper</h2>
{body_html}
</div></section>
{LINKS}
</main>
{FOOT}
</body>
</html>
"""
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
                            commit_message="The page: hero, the measured table, the four tools, the paper")
        print(f"uploaded to https://huggingface.co/spaces/{SPACE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
