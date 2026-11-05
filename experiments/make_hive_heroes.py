#!/usr/bin/env python
"""Generate the Hive 1.7B pixel hero badges.

The five cards share one visual system: a dark honeycomb hexagon, a thin
role-coloured inner frame, a small HIVE wordmark, a role vignette, and a
two-line plaque (name + 1.7B).  The logical canvas is 128x128; the delivered
PNG is 768x768 with nearest-neighbour scaling.

    python experiments/make_hive_heroes.py
    python experiments/make_hive_heroes.py --preview
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "assets" / "heroes"
S = 6
SIZE = 128
PX = SIZE * S

# Reuse the 5x7 bitmap alphabet from the existing hero generator.
from make_model_heroes import FONT  # noqa: E402

PALETTES = {
    "conductor": {
        "bg": (11, 16, 39), "bg2": (17, 24, 55),
        "accent": (255, 183, 3), "accent2": (78, 132, 255),
        "light": (246, 241, 227), "dark": (5, 8, 18),
        "glow": (78, 132, 255),
    },
    "reader": {
        "bg": (7, 26, 24), "bg2": (11, 40, 36),
        "accent": (127, 227, 176), "accent2": (43, 179, 163),
        "light": (246, 241, 227), "dark": (3, 12, 10),
        "glow": (127, 227, 176),
    },
    "auditor": {
        "bg": (32, 20, 9), "bg2": (50, 31, 14),
        "accent": (255, 179, 71), "accent2": (224, 90, 71),
        "light": (250, 239, 220), "dark": (13, 7, 3),
        "glow": (255, 179, 71),
    },
    "dispatcher": {
        "bg": (7, 24, 43), "bg2": (12, 37, 64),
        "accent": (90, 210, 255), "accent2": (44, 116, 217),
        "light": (236, 247, 255), "dark": (3, 10, 18),
        "glow": (90, 210, 255),
    },
    "toolmanager": {
        "bg": (25, 14, 38), "bg2": (40, 22, 60),
        "accent": (192, 139, 255), "accent2": (255, 111, 181),
        "light": (248, 240, 255), "dark": (10, 4, 17),
        "glow": (255, 111, 181),
    },
}

LABELS = {
    "conductor": "CONDUCTOR",
    "reader": "READER",
    "auditor": "AUDITOR",
    "dispatcher": "DISPATCHER",
    "toolmanager": "TOOL MANAGER",
}

PARAMS = {k: "1.7B" for k in LABELS}


def hex_points(inset: int = 0):
    return [(32 + inset, 4 + inset), (96 - inset, 4 + inset),
            (124 - inset, 64), (104 - inset, 120 - inset),
            (24 + inset, 120 - inset), (4 + inset, 64)]


def text(draw, label: str, y: int, color, *, x: int = 64,
         scale: int = 1, spacing: int = 1) -> None:
    pitch = 5 + spacing
    width = (len(label) * pitch - spacing) * scale
    start = x - width // 2
    for i, char in enumerate(label):
        glyph = FONT.get(char.upper(), FONT[" "])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit == "1":
                    px = start + (i * pitch + col) * scale
                    py = y + row * scale
                    draw.rectangle((px, py, px + scale - 1, py + scale - 1),
                                   fill=color)


def bitmap_width(label: str, *, scale: int = 1, spacing: int = 1) -> int:
    return (len(label) * (5 + spacing) - spacing) * scale


def bitmap_text(draw, label: str, x: int, y: int, color, *,
                scale: int = 1, spacing: int = 1) -> None:
    """Left-aligned 5x7 text, for the word shown inside the lens."""
    pitch = (5 + spacing) * scale
    for i, char in enumerate(label):
        glyph = FONT.get(char.upper(), FONT[" "])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit == "1":
                    px = x + i * pitch + col * scale
                    py = y + row * scale
                    draw.rectangle((px, py, px + scale - 1, py + scale - 1),
                                   fill=color)


def honeycomb(draw, p, mask):
    """A faint cell field behind the scene, clipped to the inner hex."""
    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    r = 7
    h = int(round(math.sqrt(3) * r))
    for row, cy in enumerate(range(18, 108, h)):
        for col, cx in enumerate(range(10, 126, 3 * r)):
            ox = (1.5 * r) if row % 2 else 0
            pts = [(cx + ox + r * math.cos(math.pi / 3 * i),
                    cy + r * math.sin(math.pi / 3 * i)) for i in range(6)]
            d.polygon(pts, outline=(*p["bg2"], 255))
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), mask))
    return layer


def node(draw, x, y, p, r=3, fill=None):
    fill = fill or p["accent2"]
    draw.rectangle((x - r, y - r, x + r, y + r), fill=p["dark"])
    draw.rectangle((x - r + 1, y - r + 1, x + r - 1, y + r - 1), fill=fill)
    draw.point((x, y), fill=p["light"])


def arrow(draw, x0, y0, x1, y1, color, width=1):
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    if abs(x1 - x0) >= abs(y1 - y0):
        sx = 1 if x1 >= x0 else -1
        draw.point((x1, y1), fill=color)
        draw.point((x1 - sx, y1 - 1), fill=color)
        draw.point((x1 - sx, y1 + 1), fill=color)
    else:
        sy = 1 if y1 >= y0 else -1
        draw.point((x1, y1), fill=color)
        draw.point((x1 - 1, y1 - sy), fill=color)
        draw.point((x1 + 1, y1 - sy), fill=color)


def draw_conductor(draw, p):
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    # central hive core
    cx, cy, r = 64, 47, 13
    pts = [(cx + r * math.cos(math.pi / 3 * i),
            cy + r * math.sin(math.pi / 3 * i)) for i in range(6)]
    draw.polygon(pts, fill=dark, outline=a)
    for dx, dy in [(0, 0), (-7, -4), (7, -4), (-7, 5), (7, 5)]:
        rr = 4
        cell = [(cx + dx + rr * math.cos(math.pi / 3 * i),
                 cy + dy + rr * math.sin(math.pi / 3 * i)) for i in range(6)]
        draw.polygon(cell, fill=a if (dx, dy) == (0, 0) else a2)
    # worker cells on a circuit
    workers = [(29, 29), (99, 29), (29, 66), (99, 66)]
    for x, y in workers:
        draw.line((64, 47, x, y), fill=a2, width=1)
    for i, (x, y) in enumerate(workers):
        draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill=dark)
        draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=a2)
        draw.rectangle((x - 2, y - 2, x + 2, y + 2), fill=light if i == 0 else a)
    # baton
    draw.line((40, 75, 88, 24), fill=dark, width=4)
    draw.line((41, 74, 87, 25), fill=a, width=2)
    draw.line((86, 24, 90, 27), fill=light, width=1)
    draw.rectangle((37, 73, 43, 79), fill=dark)
    draw.rectangle((38, 74, 42, 78), fill=a)
    draw.point((40, 76), fill=light)
    # small sparkles
    for x, y in [(20, 48), (108, 50), (64, 18), (64, 78)]:
        draw.point((x, y), fill=light)


PAGE_LINES = (43, 49, 55, 61, 67, 73)
LENS_WORD = "DOC"
_WORD_LENGTHS = (7, 5, 8, 4, 6, 3)


def _word_runs(x0: int, x1: int, seed: int):
    """Deterministic word-like dashes, so the page reads as text."""
    runs = []
    x = x0
    i = seed
    while x <= x1 - 2:
        w = _WORD_LENGTHS[i % len(_WORD_LENGTHS)]
        end = min(x + w, x1)
        if end - x >= 2:
            runs.append((x, end))
        x = end + 2
        i += 1
    return runs


def draw_reader_book(draw, p):
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    # book
    draw.rectangle((18, 34, 110, 84), fill=dark)
    draw.rectangle((20, 36, 62, 82), fill=light)
    draw.rectangle((66, 36, 108, 82), fill=light)
    draw.rectangle((62, 36, 65, 82), fill=dark)
    draw.rectangle((18, 32, 110, 36), fill=a2)
    for i, y in enumerate(PAGE_LINES):
        for x0, x1 in _word_runs(25, 56, i):
            draw.line((x0, y, x1, y), fill=a2)
        for x0, x1 in _word_runs(72, 103, i + 2):
            draw.line((x0, y, x1, y), fill=a2)
    # bookmark, kept short so the answer card can sit below it
    draw.rectangle((88, 32, 94, 41), fill=a)
    draw.polygon([(88, 41), (91, 38), (94, 41)], fill=dark)


def _glass_colour(p):
    light, a2 = p["light"], p["accent2"]
    return tuple(round(light[i] * 0.86 + a2[i] * 0.14) for i in range(3))


def draw_reader_lens(draw, p, cx=41, cy=55, *, selected=None, blank=None,
                     handle_len: int = 9):
    """A glass lens that actually shows the page text through the glass."""
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    r_out, r_rim, r_glass = 14, 13, 11
    # handle first, so it sits under the rim
    hx, hy = cx + 11 + handle_len, cy + 11 + handle_len
    draw.line((cx + 11, cy + 11, hx, hy), fill=dark, width=5)
    draw.line((cx + 11, cy + 11, hx - 1, hy - 1), fill=a, width=2)
    # rim and glass
    draw.ellipse((cx - r_out, cy - r_out, cx + r_out, cy + r_out), fill=dark)
    draw.ellipse((cx - r_rim, cy - r_rim, cx + r_rim, cy + r_rim), fill=a)
    draw.ellipse((cx - r_glass, cy - r_glass, cx + r_glass, cy + r_glass),
                 fill=_glass_colour(p))
    # the page's word dashes continue through the glass, redrawn bolder;
    # the line under the lens is replaced by an actual magnified word
    for i, y in enumerate(PAGE_LINES):
        if y == selected:
            continue
        dy = y - cy
        if abs(dy) > r_glass - 1:
            continue
        half = int(math.sqrt((r_glass - 1) ** 2 - dy ** 2))
        lo, hi = cx - half, cx + half
        pieces = []
        for x0, x1 in _word_runs(25, 56, i):
            if blank is not None and y == blank[0]:
                bx0, bx1 = blank[1], blank[2]
                if x1 <= bx0 or x0 >= bx1:
                    pieces.append((x0, x1))
                else:
                    if x0 < bx0:
                        pieces.append((x0, bx0))
                    if x1 > bx1:
                        pieces.append((bx1, x1))
            else:
                pieces.append((x0, x1))
        for x0, x1 in pieces:
            sx, ex = max(x0, lo), min(x1, hi)
            if ex <= sx:
                continue
            draw.line((sx, y, ex, y), fill=a2)
    if selected is not None:
        w = bitmap_width(LENS_WORD)
        bitmap_text(draw, LENS_WORD, cx - w // 2, selected - 3, dark)
    # small glass glint, kept off the centre line
    draw.point((cx - 7, cy - 8), fill=light)
    draw.point((cx - 6, cy - 8), fill=light)
    draw.point((cx - 7, cy - 7), fill=light)


def draw_reader(draw, p):
    draw_reader_book(draw, p)
    draw_reader_lens(draw, p, cx=41, cy=55, selected=55)


def draw_auditor(draw, p):
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    # clipboard
    draw.rectangle((16, 26, 72, 86), fill=dark)
    draw.rectangle((19, 29, 69, 83), fill=a)
    draw.rectangle((23, 34, 65, 78), fill=light)
    draw.rectangle((34, 22, 54, 31), fill=a2)
    # checklist
    for i, y in enumerate((41, 52, 63)):
        draw.rectangle((27, y, 34, y + 7), outline=dark)
        if i < 2:
            draw.line((27, y + 4, 30, y + 7), fill=dark)
            draw.line((30, y + 7, 35, y + 1), fill=dark)
        draw.line((39, y + 3, 61, y + 3), fill=a2)
    # seal / approval hex
    sx, sy, rr = 92, 41, 13
    seal = [(sx + rr * math.cos(math.pi / 3 * i),
             sy + rr * math.sin(math.pi / 3 * i)) for i in range(6)]
    draw.polygon(seal, fill=dark, outline=a)
    draw.polygon([(sx - 7, sy), (sx - 2, sy + 5), (sx + 8, sy - 6)],
                 fill=a2)
    # lens
    draw.ellipse((75, 53, 103, 81), fill=dark)
    draw.ellipse((78, 56, 100, 78), fill=(*p["bg"], 255), outline=a, width=2)
    draw.line((99, 77, 112, 90), fill=dark, width=5)
    draw.line((99, 77, 111, 89), fill=a, width=2)
    draw.point((84, 64), fill=light)


def draw_dispatcher(draw, p):
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    # clock
    draw.ellipse((48, 18, 80, 50), fill=dark, outline=a, width=2)
    draw.ellipse((51, 21, 77, 47), outline=a2, width=1)
    draw.line((64, 24, 64, 36), fill=light, width=1)
    draw.line((64, 36, 72, 39), fill=light, width=1)
    draw.point((64, 36), fill=light)
    # queues
    for i, y in enumerate((38, 56, 74)):
        draw.rectangle((13, y, 34, y + 11), fill=dark)
        draw.rectangle((15, y + 2, 32, y + 9), fill=a2)
        draw.point((18, y + 5), fill=light)
        draw.line((22, y + 5, 30, y + 5), fill=dark)
    # workers
    for i, y in enumerate((32, 54, 76)):
        x, r = 98, 9
        pts = [(x + r * math.cos(math.pi / 3 * j),
                y + r * math.sin(math.pi / 3 * j)) for j in range(6)]
        draw.polygon(pts, fill=dark, outline=a)
        draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=a)
        draw.point((x - 1, y - 1), fill=dark)
        draw.point((x + 2, y - 1), fill=dark)
    # lanes
    for i, y in enumerate((40, 62, 84)):
        arrow(draw, 36, y, 86, y, a2)
    # packets
    for x, y in [(48, 40), (64, 62), (78, 84)]:
        draw.rectangle((x - 2, y - 2, x + 2, y + 2), fill=light)
    # dependency ticks
    for x, y in [(44, 49), (60, 71)]:
        draw.line((x, y, x + 3, y), fill=a)
        draw.point((x + 3, y), fill=a)


def draw_toolmanager(draw, p):
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    # a 2x3 honeycomb tool rack
    cells = [
        (34, 30, "file"), (64, 30, "terminal"), (94, 30, "wrench"),
        (34, 58, "image"), (64, 58, "cursor"), (94, 58, "toolbox"),
    ]
    for x, y, kind in cells:
        r = 12
        pts = [(x + r * math.cos(math.pi / 3 * i),
                y + r * math.sin(math.pi / 3 * i)) for i in range(6)]
        draw.polygon(pts, fill=dark, outline=a2)
        if kind == "file":
            draw.rectangle((x - 5, y - 6, x + 5, y + 6), fill=light)
            for yy in (y - 3, y, y + 3):
                draw.line((x - 3, yy, x + 3, yy), fill=a2)
        elif kind == "terminal":
            draw.rectangle((x - 6, y - 5, x + 6, y + 5), fill=dark, outline=a)
            draw.line((x - 3, y - 2, x, y + 1), fill=a)
            draw.line((x, y + 1, x - 3, y + 4), fill=a)
            draw.line((x + 2, y + 3, x + 5, y + 3), fill=light)
        elif kind == "wrench":
            draw.line((x - 5, y + 5, x + 4, y - 4), fill=light, width=2)
            draw.ellipse((x - 8, y - 8, x - 2, y - 2), outline=light, width=1)
            draw.ellipse((x + 2, y + 2, x + 8, y + 8), outline=light, width=1)
        elif kind == "image":
            draw.rectangle((x - 6, y - 5, x + 6, y + 5), fill=light)
            draw.ellipse((x - 4, y - 3, x - 1, y), fill=a2)
            draw.polygon([(x - 6, y + 5), (x - 1, y), (x + 2, y + 3),
                          (x + 6, y - 1), (x + 6, y + 5)], fill=a)
        elif kind == "cursor":
            draw.polygon([(x - 4, y - 7), (x - 4, y + 5), (x - 1, y + 2),
                          (x + 2, y + 8), (x + 4, y + 7), (x + 1, y + 1),
                          (x + 5, y + 1)], fill=light, outline=dark)
        elif kind == "toolbox":
            draw.rectangle((x - 7, y - 3, x + 7, y + 6), fill=dark, outline=a)
            draw.line((x - 4, y - 6, x + 4, y - 6), fill=light, width=2)
            draw.line((x - 4, y - 6, x - 4, y - 3), fill=light, width=2)
            draw.line((x + 4, y - 6, x + 4, y - 3), fill=light, width=2)
            draw.line((x - 7, y + 1, x + 7, y + 1), fill=a2, width=1)
            draw.rectangle((x - 2, y + 2, x + 2, y + 5), fill=light)
    # ordered flow and a small tool belt
    draw.line((28, 82, 100, 82), fill=a2, width=1)
    for x in (28, 46, 64, 82, 100):
        draw.rectangle((x - 2, 80, x + 2, 84), fill=a)
    for x, y in [(44, 88), (64, 88), (84, 88)]:
        draw.point((x, y), fill=light)


ICONS = {
    "conductor": draw_conductor,
    "reader": draw_reader,
    "auditor": draw_auditor,
    "dispatcher": draw_dispatcher,
    "toolmanager": draw_toolmanager,
}


def render(name: str, frame: int = 0, *, lens: bool = True) -> Image.Image:
    p = PALETTES[name]
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # outer rim, inner rim, body
    d.polygon(hex_points(0), fill=(*p["dark"], 255))
    d.polygon(hex_points(1), fill=(*p["accent"], 255))
    d.polygon(hex_points(3), fill=(*p["bg2"], 255))
    d.polygon(hex_points(4), fill=(*p["bg"], 255))

    inner = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(inner).polygon(hex_points(5), fill=255)
    img.alpha_composite(honeycomb(d, p, inner))

    # role-coloured hairline just inside the gold rim
    d.line(hex_points(3)[0:2], fill=p["accent2"], width=1)

    # header
    d.rectangle((52, 9, 76, 10), fill=p["accent2"])
    text(d, "HIVE", 13, p["light"])

    # role art
    if name == "reader" and not lens:
        draw_reader_book(d, p)
    else:
        ICONS[name](d, p)

    # footer plaque
    d.polygon([(15, 87), (113, 87), (102, 116), (26, 116)],
              fill=(*p["dark"], 245))
    d.line((15, 87, 113, 87), fill=p["accent"], width=1)
    if name == "toolmanager":
        text(d, LABELS[name], 91, p["light"])
        text(d, PARAMS[name], 102, p["accent"])
    else:
        text(d, LABELS[name], 91, p["light"])
        text(d, PARAMS[name], 102, p["accent"])
    # small family diamond
    d.polygon([(64, 118), (66, 121), (64, 124), (62, 121)],
              fill=p["accent2"])
    return img.resize((PX, PX), Image.Resampling.NEAREST)


def preview():
    from PIL import Image as I
    names = list(PALETTES)
    sheet = I.new("RGBA", (PX * len(names), PX), (18, 18, 20, 255))
    for i, name in enumerate(names):
        sheet.alpha_composite(render(name), (i * PX, 0))
    path = OUT / "hive-heroes-preview.png"
    sheet.save(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in PALETTES:
        img = render(name)
        path = OUT / f"{name}.png"
        img.save(path)
        print(path)
    if args.preview:
        print(preview())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
