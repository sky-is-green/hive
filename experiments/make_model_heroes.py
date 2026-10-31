#!/usr/bin/env python
"""Generate pixel-art hexagon heroes for the Hive model cards.

Each hero is a 96x96 logical pixel drawing scaled 8x with nearest-neighbour:
a dark hexagon badge, role palette, icon, and a 5x7 bitmap label.

    python experiments/make_model_heroes.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "assets" / "heroes"
SCALE = 8

FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    " ": ["00000"] * 7,
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
}

PALETTES = {
    "conductor": {"bg": (13, 18, 48), "accent": (242, 193, 78), "accent2": (79, 124, 255),
                  "light": (246, 241, 227), "dark": (7, 10, 26)},
    "reader": {"bg": (8, 28, 24), "accent": (127, 227, 176), "accent2": (43, 179, 163),
               "light": (246, 241, 227), "dark": (5, 15, 13)},
    "auditor": {"bg": (35, 22, 9), "accent": (255, 179, 71), "accent2": (224, 90, 71),
                "light": (250, 239, 220), "dark": (15, 9, 3)},
    "dispatcher": {"bg": (8, 27, 42), "accent": (90, 210, 255), "accent2": (44, 116, 217),
                   "light": (236, 247, 255), "dark": (4, 11, 18)},
    "clerk": {"bg": (28, 15, 43), "accent": (192, 139, 255), "accent2": (255, 111, 181),
              "light": (248, 240, 255), "dark": (12, 5, 20)},
    "scion35": {"bg": (8, 32, 24), "accent": (94, 224, 138), "accent2": (255, 209, 102),
                "light": (240, 255, 244), "dark": (4, 14, 10)},
    "scion176": {"bg": (42, 13, 13), "accent": (255, 138, 76), "accent2": (255, 209, 102),
                 "light": (255, 244, 232), "dark": (18, 5, 5)},
}


def hex_points(inset: int = 0):
    # flat-top hexagon
    return [(24 + inset, 4 + inset), (72 - inset, 4 + inset), (92 - inset, 48),
            (72 - inset, 92 - inset), (24 + inset, 92 - inset), (4 + inset, 48)]


def text(draw: ImageDraw.ImageDraw, label: str, y: int, color, x: int = 48) -> None:
    width = len(label) * 6 - 1
    start = x - width // 2
    for i, char in enumerate(label):
        glyph = FONT.get(char.upper(), FONT[" "])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit == "1":
                    draw.rectangle((start + i * 6 + col, y + row,
                                    start + i * 6 + col, y + row), fill=color)


def node(draw, x, y, color, light):
    draw.rectangle((x - 1, y - 1, x + 6, y + 6), fill=light)
    draw.rectangle((x + 1, y + 1, x + 4, y + 4), fill=color)


def arrow(draw, x0, y0, x1, y1, color, width=1):
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    # tiny pixel arrowhead
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
    # a small control graph
    nodes = [(22, 18), (44, 13), (66, 22), (60, 38)]
    for x0, y0 in nodes[:3]:
        for x1, y1 in nodes[1:]:
            if (x0, y0) != (x1, y1):
                draw.line((x0 + 2, y0 + 2, x1 + 2, y1 + 2), fill=a2, width=1)
    for x, y in nodes:
        node(draw, x, y, a2, light)
    # baton
    draw.line((28, 55, 62, 24), fill=dark, width=4)
    draw.line((30, 54, 61, 25), fill=a, width=2)
    draw.rectangle((60, 21, 64, 25), fill=light)
    # handle
    draw.rectangle((26, 53, 32, 58), fill=dark)
    draw.rectangle((27, 54, 31, 57), fill=a)
    draw.point((28, 55), fill=light)


def draw_reader(draw, p):
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    # book
    draw.rectangle((18, 20, 78, 58), fill=dark)
    draw.rectangle((21, 23, 46, 55), fill=light)
    draw.rectangle((49, 23, 75, 55), fill=light)
    draw.rectangle((47, 23, 48, 55), fill=dark)
    draw.rectangle((18, 20, 78, 22), fill=a2)
    for y in (29, 34, 39, 44, 49):
        draw.line((25, y, 42, y), fill=a2, width=1)
        draw.line((53, y, 69, y), fill=a2, width=1)
    # bookmark
    draw.rectangle((58, 20, 62, 32), fill=a)
    draw.point((59, 30), fill=dark)
    draw.point((61, 30), fill=dark)
    # magnifier
    draw.ellipse((54, 30, 74, 50), outline=dark, width=3)
    draw.ellipse((56, 32, 72, 48), outline=a, width=1)
    draw.line((68, 46, 78, 56), fill=dark, width=4)
    draw.line((68, 46, 77, 55), fill=a, width=2)
    draw.point((60, 37), fill=light)
    draw.point((62, 37), fill=light)


def draw_auditor(draw, p):
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    # clipboard
    draw.rectangle((16, 14, 46, 58), fill=dark)
    draw.rectangle((18, 17, 44, 56), fill=a)
    draw.rectangle((22, 22, 40, 52), fill=light)
    draw.rectangle((27, 11, 37, 17), fill=a2)
    # checkbox list: clear checks, not red squares
    for y in (27, 35, 43):
        draw.rectangle((24, y, 29, y + 5), outline=dark)
        draw.line((24, y + 3, 26, y + 5), fill=dark, width=1)
        draw.line((26, y + 5, 30, y + 1), fill=dark, width=1)
        draw.line((33, y + 2, 40, y + 2), fill=a2, width=1)
    # card grid / magnifier
    draw.rectangle((52, 22, 80, 46), fill=dark)
    draw.rectangle((54, 24, 78, 44), fill=a2)
    draw.line((66, 24, 66, 44), fill=dark, width=1)
    draw.line((54, 34, 78, 34), fill=dark, width=1)
    # a cream check in the top-left cell
    draw.line((57, 29, 59, 32), fill=light, width=1)
    draw.line((59, 32, 64, 26), fill=light, width=1)
    # magnifier outline only, so the card grid stays visible
    draw.ellipse((58, 32, 78, 52), outline=dark, width=3)
    draw.ellipse((60, 34, 76, 50), outline=a, width=1)
    draw.line((74, 48, 82, 56), fill=dark, width=4)
    draw.line((74, 48, 81, 55), fill=a, width=2)
    draw.point((62, 37), fill=light)
    draw.point((64, 37), fill=light)


def draw_dispatcher(draw, p):
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    # job queue
    for i, y in enumerate((18, 30, 42)):
        draw.rectangle((12, y, 24, y + 8), fill=dark)
        draw.rectangle((14, y + 2, 22, y + 6), fill=a2)
        draw.point((16, y + 4), fill=light)
    # workers
    for i, y in enumerate((16, 32, 48)):
        draw.rectangle((64, y, 82, y + 8), fill=dark)
        draw.rectangle((66, y + 2, 80, y + 6), fill=a)
        draw.point((68, y + 4), fill=dark)
        draw.point((71, y + 4), fill=dark)
        arrow(draw, 26, 22 + i * 12, 63, y + 4, a2, 1)
    # clock
    draw.ellipse((40, 8, 56, 24), outline=dark, width=3)
    draw.ellipse((42, 10, 54, 22), outline=a2, width=1)
    draw.line((48, 12, 48, 17), fill=light, width=1)
    draw.line((48, 17, 52, 19), fill=light, width=1)


def draw_clerk(draw, p):
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    boxes = [(16, 14), (48, 14), (16, 36), (48, 36)]
    for x, y in boxes:
        draw.rectangle((x, y, x + 28, y + 18), fill=dark)
        draw.rectangle((x + 2, y + 2, x + 26, y + 16), fill=(30, 18, 44))
        draw.rectangle((x, y, x + 28, y + 18), outline=a2, width=1)
    # file
    draw.rectangle((23, 18, 37, 29), fill=light)
    for y in (21, 24, 27):
        draw.line((25, y, 34, y), fill=a2, width=1)
    # terminal
    draw.line((55, 20, 59, 24), fill=a, width=1)
    draw.line((59, 24, 55, 28), fill=a, width=1)
    draw.line((62, 28, 68, 28), fill=light, width=1)
    # image
    draw.rectangle((22, 42, 36, 51), outline=light)
    draw.ellipse((24, 43, 27, 46), fill=a)
    draw.polygon([(22, 51), (28, 45), (32, 49), (36, 44), (36, 51)], fill=a2)
    # wrench
    draw.line((56, 40, 64, 50), fill=light, width=2)
    draw.ellipse((52, 38, 58, 44), outline=light, width=1)
    draw.ellipse((61, 48, 69, 54), outline=light, width=1)
    # cursor
    draw.polygon([(56, 36), (56, 46), (59, 43), (62, 48), (64, 47), (61, 42), (65, 42)],
                 fill=light, outline=dark)
    # sequence
    draw.line((38, 58, 42, 58), fill=a2, width=1)
    draw.point((42, 57), fill=a2)
    draw.point((42, 59), fill=a2)
    draw.line((46, 58, 50, 58), fill=a2, width=1)
    draw.point((50, 57), fill=a2)
    draw.point((50, 59), fill=a2)


def draw_scion(draw, p, big=False):
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    # tree / graft
    draw.rectangle((45, 26, 51, 60), fill=dark)
    draw.rectangle((46, 28, 50, 59), fill=a)
    branches = [((48, 42), (30, 28)), ((48, 38), (66, 24)),
                ((48, 50), (27, 46)), ((48, 50), (69, 44))]
    if big:
        branches += [((48, 34), (38, 18)), ((48, 34), (58, 18))]
    for (x0, y0), (x1, y1) in branches:
        draw.line((x0, y0, x1, y1), fill=dark, width=3)
        draw.line((x0, y0, x1, y1), fill=a, width=1)
        draw.rectangle((x1 - 2, y1 - 2, x1 + 2, y1 + 2), fill=a2)
    # graft band
    draw.rectangle((42, 50, 54, 53), fill=dark)
    draw.rectangle((43, 51, 53, 52), fill=a2)
    # crown
    draw.polygon([(36, 8), (40, 15), (44, 8), (48, 15), (52, 8), (56, 15), (60, 8),
                  (58, 18), (38, 18)], fill=a2, outline=dark)
    draw.point((42, 12), fill=light)
    draw.point((48, 12), fill=light)
    draw.point((54, 12), fill=light)
    if big:
        # lightning bolt through the graft
        draw.polygon([(60, 20), (48, 38), (56, 38), (42, 58), (52, 40), (45, 40)],
                     fill=a2, outline=dark)
    # roots
    draw.line((44, 59, 36, 62), fill=a, width=1)
    draw.line((52, 59, 60, 62), fill=a, width=1)


ICONS = {
    "conductor": draw_conductor,
    "reader": draw_reader,
    "auditor": draw_auditor,
    "dispatcher": draw_dispatcher,
    "clerk": draw_clerk,
    "scion35": lambda d, p: draw_scion(d, p, big=False),
    "scion176": lambda d, p: draw_scion(d, p, big=True),
}

LABELS = {
    "conductor": "CONDUCTOR",
    "reader": "READER",
    "auditor": "AUDITOR",
    "dispatcher": "DISPATCHER",
    "clerk": "CLERK",
    "scion35": "SCION 35B",
    "scion176": "SCION 176B",
}


def make(name: str) -> Path:
    p = PALETTES[name]
    bg, accent, accent2 = p["bg"], p["accent"], p["accent2"]
    light, dark = p["light"], p["dark"]
    img = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # outer glow
    d.polygon([(24, 2), (72, 2), (94, 48), (72, 94), (24, 94), (2, 48)],
              fill=(*accent, 45))
    # border and body
    d.polygon(hex_points(0), fill=(*dark, 255))
    d.polygon(hex_points(2), fill=(*accent, 255))
    d.polygon(hex_points(4), fill=(*bg, 255))
    # pixel sparkles / dither
    for x, y in [(20, 12), (76, 16), (14, 60), (82, 54), (48, 56)]:
        d.point((x, y), fill=(*light, 90))
    # icon
    ICONS[name](d, p)
    # label banner
    d.rectangle((16, 64, 80, 78), fill=(*dark, 240))
    d.line((16, 64, 80, 64), fill=accent, width=1)
    d.line((16, 78, 80, 78), fill=accent2, width=1)
    text(d, LABELS[name], 67, light)
    # base diamond
    d.polygon([(48, 80), (51, 84), (48, 88), (45, 84)], fill=accent2)
    # corner brackets
    for x, y, sx, sy in [(8, 40, 1, 0), (88, 40, -1, 0), (18, 6, 0, 1), (78, 6, 0, 1)]:
        d.line((x, y, x + 4 * sx if sx else x, y + 4 * sy if sy else y), fill=accent, width=1)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    img.resize((96 * SCALE, 96 * SCALE), Image.Resampling.NEAREST).save(path)
    return path


def main() -> int:
    for name in PALETTES:
        print(make(name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
