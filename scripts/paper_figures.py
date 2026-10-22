"""Render the paper's measured figures as standalone SVG.

The numbers are from ``docs/AB-RESULTS.md``:

- §2B: the 10-task latency (35B 5.36 s, 176B 229.67 s).
- §3A-3E: the 84-item A/B arms (64, 63, 67, 70 served-correct).
- §1B-sexies: the 28-task tier results (35B 21, 176B 19, 27B 26,
  head -> Bonsai 23, oracle -> Bonsai 27).

Run with ``python scripts/paper_figures.py``. The SVGs are committed next
to the paper in ``docs/figures`` so GitHub renders them without a build.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "cascade" / "figures"

BG = "#fbfaf8"
INK = "#171614"
MUTED = "#6b675f"
LINE = "#e4e0d8"
GRID = "#ece7de"
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, "
        "Arial, sans-serif")


def text(x: float, y: float, value: str, size: float = 12.5,
         fill: str = INK, weight: int = 400, anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
            f'font-size="{size}" fill="{fill}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{escape(value)}</text>')


def card(width: int, height: int, title: str, subtitle: str,
         desc: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title>
<desc id="desc">{escape(desc)}</desc>
<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="14" fill="{BG}" stroke="{LINE}"/>
<text x="28" y="37" font-family="{FONT}" font-size="17" font-weight="700" fill="{INK}">{escape(title)}</text>
<text x="28" y="59" font-family="{FONT}" font-size="12" fill="{MUTED}">{escape(subtitle)}</text>
<line x1="28" y1="74" x2="{width - 28}" y2="74" stroke="{LINE}"/>
'''


def footer(width: int, y: int, source: str) -> str:
    return (f'<line x1="28" y1="{y - 17}" x2="{width - 28}" y2="{y - 17}" '
            f'stroke="{LINE}"/>\n'
            f'<text x="28" y="{y}" font-family="{FONT}" font-size="10.5" '
            f'fill="{MUTED}">{escape(source)}</text>\n')


def grid(x0: float, y0: float, y1: float, tick: float,
         ticks: list[int], label_y: float, suffix: str = "") -> str:
    out = []
    for value in ticks:
        x = x0 + value * tick
        out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" '
                   f'stroke="{GRID}" stroke-dasharray="3 4"/>')
        out.append(text(x, label_y, f"{value}{suffix}", 10.5, MUTED, anchor="middle"))
    return "\n".join(out)


def value_text(x: float, y: float, n: int, total: int) -> str:
    pct = 100 * n / total
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
            f'font-size="12.5" fill="{INK}" font-weight="600">'
            f'{n}/{total}<tspan fill="{MUTED}" font-weight="400">'
            f' · {pct:.1f}%</tspan></text>')


def arms_svg() -> str:
    width, height = 800, 338
    x0, tick = 260.0, 420.0 / 84
    total = 84
    rows = [
        ("No verifier, no repair", 64, "#e9e4da", "#b8b0a3", "",
         "the baseline", MUTED),
        ("The verifier alone", 63, "#f7d9c4", "#c2410c", "",
         "a tax: -1.2", "#b91c1c"),
        ("The verifier + the repair leg", 67, "#0f766e", "#0f766e", "",
         "+3.6 over no verification", "#0f766e"),
        ("The learned gate (off)", 70, "#fef3c7", "#b45309", ' stroke-dasharray="5 3"',
         "3 false flags > budget of 2", "#b45309"),
    ]
    centers = [110, 162, 214, 266]
    svg = [card(
        width, height,
        "The A/B arms, 84 items each",
        "Served-correct. A verifier alone is a tax; the repair leg is what pays.",
        "Horizontal bar chart of four arms over 84 items each: no verifier "
        "64/84, verifier alone 63/84, verifier plus repair 67/84, and the "
        "learned gate 70/84. The gate is off because its false flags exceed "
        "the pre-registered budget.",
    )]
    svg.append(grid(x0, 92, 288, tick, [0, 21, 42, 63, 84], 300))
    base = x0 + 64 * tick
    svg.append(f'<line x1="{base:.1f}" y1="92" x2="{base:.1f}" y2="288" '
               f'stroke="#b8b0a3" stroke-dasharray="4 4"/>')
    svg.append(text(base, 88, "64 baseline", 9.5, MUTED, anchor="middle"))
    for (label, n, fill, stroke, dash, note, note_fill), cy in zip(rows, centers):
        y = cy - 14
        w = n * tick
        svg.append(f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="28" '
                   f'rx="6" fill="{fill}" stroke="{stroke}"{dash}/>')
        svg.append(text(28, cy + 1, label, 12.5, INK, 500))
        svg.append(text(28, cy + 19, note, 9.5, note_fill))
        svg.append(value_text(x0 + w + 9, cy + 4, n, total))
    svg.append(footer(width, 324,
                      "Source: docs/AB-RESULTS.md §3A-3E · 3 draws × 28 tasks · 84 items per arm"))
    svg.append("</svg>\n")
    return "\n".join(svg)


def tiers_svg() -> str:
    width, height = 800, 372
    x0, tick = 280.0, 400.0 / 28
    total = 28
    rows = [
        ("The 176B alone", 19, "#e9e4da", "#a8a29e", "", "head-to-head", MUTED),
        ("The 35B alone", 21, "#d6d0c4", "#8a8378", "", "the baseline", MUTED),
        ("The 176B head → Bonsai", 23, "#dbeafe", "#2563eb", "",
         "3 escalations, all errors", "#2563eb"),
        ("The 27B alone (+5 net)", 26, "#0f766e", "#0f766e", "",
         "6 fixed, 1 broken", "#0f766e"),
        ("The oracle → Bonsai", 27, "#134e4a", "#134e4a", "",
         "7 escalations, all errors", "#134e4a"),
    ]
    centers = [104, 152, 200, 248, 296]
    svg = [card(
        width, height,
        "The 28-task tier results",
        "Correct answers, one draw per task. The 27B is the generator; the head is the gate.",
        "Horizontal bar chart of five 28-task results: 176B alone 19/28, "
        "35B alone 21/28, 176B head to Bonsai 23/28, 27B alone 26/28, and "
        "oracle to Bonsai 27/28.",
    )]
    svg.append(grid(x0, 90, 316, tick, [0, 7, 14, 21, 28], 328))
    base = x0 + 21 * tick
    svg.append(f'<line x1="{base:.1f}" y1="90" x2="{base:.1f}" y2="316" '
               f'stroke="#b8b0a3" stroke-dasharray="4 4"/>')
    svg.append(text(base, 86, "35B: 21/28", 9.5, MUTED, anchor="middle"))
    for (label, n, fill, stroke, dash, note, note_fill), cy in zip(rows, centers):
        y = cy - 13
        w = n * tick
        svg.append(f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="26" '
                   f'rx="6" fill="{fill}" stroke="{stroke}"{dash}/>')
        svg.append(text(28, cy - 1, label, 12.5, INK, 500))
        svg.append(text(28, cy + 15, note, 9.5, note_fill))
        svg.append(value_text(x0 + w + 9, cy + 4, n, total))
    svg.append(footer(width, 358,
                      "Source: docs/AB-RESULTS.md §1B-sexies · the 176B alone is the head-to-head; its generation arm is net 0"))
    svg.append("</svg>\n")
    return "\n".join(svg)




def latency_svg() -> str:
    width, height = 800, 272
    x0, tick = 260.0, 420.0 / 229.67
    rows = [
        ("The 35B-A3B", 5.36, "#0f766e", "#0f766e", "the small tier"),
        ("The 176B-A6B", 229.67, "#e9e4da", "#a8a29e", "the large tier"),
    ]
    centers = [120, 190]
    svg = [card(
        width, height,
        "The same answer, 42.5× the wall clock",
        "Ten organic tasks, co-hosted; served-correct 0.800 for both. Mean seconds per item.",
        "Horizontal bar chart of mean seconds per item: the 35B-A3B at 5.36 seconds, "
        "the 176B-A6B at 229.67 seconds, a factor of 42.5.",
    )]
    svg.append(grid(x0, 82, 216, tick, [0, 50, 100, 150, 200], 232))
    for (label, value, fill, stroke, note), cy in zip(rows, centers):
        w = value * tick
        svg.append(f'<rect x="{x0}" y="{cy - 15}" width="{w:.1f}" height="30" '
                   f'rx="6" fill="{fill}" stroke="{stroke}"/>')
        svg.append(text(28, cy - 2, label, 12.5, INK, 500))
        svg.append(text(28, cy + 16, note, 9.5, MUTED))
        svg.append(f'<text x="{x0 + w + 9:.1f}" y="{cy + 4:.1f}" font-family="{FONT}" '
                   f'font-size="12.5" fill="{INK}" font-weight="600">{value:.2f} s</text>')
    svg.append(footer(width, 254,
                      "Source: docs/AB-RESULTS.md §2B · 10 organic tasks · mean seconds per item"))
    svg.append("</svg>\n")
    return "\n".join(svg)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in (("arms.svg", arms_svg()), ("latency.svg", latency_svg()),
                       ("tiers.svg", tiers_svg())):
        (OUT / name).write_text(body, encoding="utf-8")
        print(f"wrote {OUT / name} ({len(body)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
