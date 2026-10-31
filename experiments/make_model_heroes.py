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

MINI_FONT = {
    "D": ["1110", "1001", "1001", "1001", "1110"],
    "R": ["1110", "1001", "1110", "1010", "1001"],
    "A": ["0110", "1001", "1111", "1001", "1001"],
    "F": ["1111", "1000", "1110", "1000", "1000"],
    "T": ["1111", "0110", "0110", "0110", "0110"],
    "E": ["1111", "1000", "1110", "1000", "1111"],
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
    "scion35": {"bg": (6, 26, 17), "accent": (86, 220, 132), "accent2": (44, 166, 96),
                "light": (232, 255, 238), "dark": (3, 12, 8),
                "root": (118, 144, 82), "root_dark": (50, 70, 38),
                "soil": (28, 38, 22), "band": (228, 196, 104)},
    "scion176": {"bg": (7, 30, 17), "accent": (138, 238, 112), "accent2": (58, 180, 96),
                 "light": (240, 255, 226), "dark": (3, 14, 7),
                 "root": (132, 158, 88), "root_dark": (56, 78, 42),
                 "soil": (30, 40, 24), "band": (240, 208, 110)},
    "scion_drafter": {"bg": (6, 26, 17), "accent": (86, 220, 132), "accent2": (44, 166, 96),
                      "light": (232, 255, 238), "dark": (3, 12, 8),
                      "root": (118, 144, 82), "root_dark": (50, 70, 38),
                      "soil": (28, 38, 22), "band": (228, 196, 104),
                      "gold": (240, 204, 110), "gold_dark": (164, 124, 48)},
}


def hex_points(inset: int = 0):
    # flat-top hexagon, wider across the caption end so long titles fit
    return [(24 + inset, 4 + inset), (72 - inset, 4 + inset), (94 - inset, 48),
            (78 - inset, 92 - inset), (18 + inset, 92 - inset),
            (2 + inset, 48)]


def text(draw: ImageDraw.ImageDraw, label: str, y: int, color,
         x: int = 48, scale: int = 1, spacing: int = 1) -> None:
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
                    draw.rectangle((px, py, px + scale - 1, py + scale - 1), fill=color)


def mini_text(draw: ImageDraw.ImageDraw, label: str, y: int, color,
              x: int = 48) -> None:
    """Four-by-five pixel caption, for a third line in the badge."""
    pitch = 5
    width = len(label) * pitch - 1
    start = x - width // 2
    for i, char in enumerate(label):
        glyph = MINI_FONT.get(char.upper(), ["0000"] * 5)
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit == "1":
                    draw.point((start + i * pitch + col, y + row), fill=color)


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


def leaf(draw, box, p, pointing_right: bool = True, tip_dy: int = 0,
         blade=None, ink=None) -> None:
    """A full pixel leaf with a rounded base, tapered tip and short vein."""
    blade = blade or p["accent"]
    ink = ink or p["accent2"]
    x0, y0, x1, y1 = box
    cy = (y0 + y1) // 2
    if pointing_right:
        tip = (x1, cy + tip_dy)
        edge = [(x0, cy), (x0 + 1, y0 + 1), (x0 + 4, y0),
                (x0 + 7, y0 + 1), tip, (x0 + 7, y1 - 1),
                (x0 + 4, y1), (x0 + 1, y1 - 1)]
        vein = (x0 + 3, cy, x1 - 3, cy + tip_dy)
        shoulder = (x0 + 4, y0 + 1)
    else:
        tip = (x0, cy + tip_dy)
        edge = [(x1, cy), (x1 - 1, y0 + 1), (x1 - 4, y0),
                (x1 - 7, y0 + 1), tip, (x1 - 7, y1 - 1),
                (x1 - 4, y1), (x1 - 1, y1 - 1)]
        vein = (x1 - 3, cy, x0 + 3, cy + tip_dy)
        shoulder = (x1 - 4, y0 + 1)
    draw.polygon(edge, fill=blade, outline=p["dark"])
    draw.line(vein, fill=ink, width=1)
    draw.point(shoulder, fill=p["light"])


def draw_scion(draw, p, big=False, drafter=False):
    """A young scion shoot grafted onto an old, established rootstock.

    The rootstock is a stump with a deep root fan and a cut side branch; the
    bright scion is a different green, joined at a pale grafting band. The
    176B hero gets a taller shoot, more leaves and a denser root fan.
    """
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    root, root_dark = p["root"], p["root_dark"]
    soil, band = p["soil"], p["band"]

    # ground line and a soil cross-section for the established roots
    draw.polygon([(13, 61), (13, 49), (20, 46), (30, 44), (42, 44),
                  (54, 44), (66, 44), (76, 46), (83, 49), (83, 61)],
                 fill=soil)
    draw.line((13, 49, 20, 46, 30, 44, 42, 44, 54, 44, 66, 44,
               76, 46, 83, 49), fill=root_dark, width=1)
    for x in (18, 24, 72, 78):
        draw.line((x, 47, x - 1, 44), fill=a2, width=1)
        draw.line((x, 47, x + 1, 44), fill=a2, width=1)
    for x, y in [(22, 52), (30, 56), (67, 53), (74, 57), (58, 59)]:
        draw.point((x, y), fill=root_dark)

    # established root fan: thick mains, tapered laterals, a few rootlets
    if big:
        roots = [
            ([(48, 47), (42, 50), (35, 53), (28, 56), (24, 59)], 4),
            ([(48, 47), (54, 50), (61, 53), (68, 56), (72, 59)], 4),
            ([(42, 50), (39, 55), (37, 59)], 3),
            ([(54, 50), (57, 55), (59, 59)], 3),
            ([(28, 56), (26, 60)], 2),
            ([(68, 56), (70, 60)], 2),
            ([(48, 47), (48, 54), (47, 60)], 3),
            ([(35, 53), (32, 58)], 2),
            ([(61, 53), (64, 58)], 2),
        ]
    else:
        roots = [
            ([(48, 47), (43, 50), (37, 53), (31, 55), (26, 58)], 4),
            ([(48, 47), (53, 50), (59, 53), (65, 55), (70, 58)], 4),
            ([(43, 50), (40, 54), (38, 58), (36, 60)], 3),
            ([(53, 50), (56, 54), (58, 58), (60, 60)], 3),
            ([(31, 55), (29, 59)], 2),
            ([(65, 55), (67, 59)], 2),
            ([(48, 47), (48, 53), (47, 59)], 3),
            ([(37, 53), (35, 57)], 1),
            ([(59, 53), (61, 57)], 1),
        ]
    for pts, width in roots:
        segs = len(pts) - 1
        for i in range(segs):
            w = max(1, round(width * (segs - i) / segs))
            draw.line((pts[i], pts[i + 1]), fill=root_dark, width=w + 2)
        for i in range(segs):
            w = max(1, round(width * (segs - i) / segs))
            draw.line((pts[i], pts[i + 1]), fill=root, width=w)

    # rootstock stump: older, darker wood, wider than the scion
    draw.polygon([(42, 49), (44, 33), (52, 33), (54, 49)], fill=dark)
    draw.polygon([(44, 48), (45, 35), (51, 35), (52, 48)], fill=root)
    draw.line((46, 37, 46, 40), fill=root_dark, width=1)
    draw.line((50, 42, 50, 46), fill=root_dark, width=1)
    draw.point((47, 44), fill=root_dark)
    # old cut side branch
    draw.line((43, 40, 39, 37), fill=dark, width=3)
    draw.line((43, 40, 39, 37), fill=root, width=1)

    # scion shoot: new growth, brighter green, slimmer than the stump
    top = 10 if big else (9 if drafter else 13)
    draw.line((48, top, 48, 34), fill=dark, width=3)
    draw.line((48, top + 1, 48, 33), fill=a, width=1)
    # new growth tip
    draw.point((48, top), fill=light)
    draw.point((47, top + 1), fill=light)

    # 35B-A3B: three active leaves; the drafter adds one gold leaf.
    leaves = [(50, 14, 60, 19), (36, 20, 46, 25),
              (50, 25, 61, 30)]
    if big:
        leaves = [(50, 9, 60, 14), (36, 11, 46, 16),
                  (51, 16, 62, 21), (34, 19, 45, 24),
                  (50, 23, 61, 28), (33, 25, 43, 30)]
    for box in leaves:
        # short petiole joins the leaf to the shoot
        x0, y0, x1, y1 = box
        cy = (y0 + y1) // 2
        if x0 > 48:
            draw.line((48, cy, x0, cy), fill=a2, width=1)
        else:
            draw.line((x1, cy, 48, cy), fill=a2, width=1)
        leaf(draw, box, p, pointing_right=x0 > 48,
             tip_dy=-1 if cy <= 20 else 1)

    if drafter:
        # the side-car drafter: one extra leaf, in graft-gold, on its own shoot
        draw.line((44, 11, 48, 11), fill=p["gold_dark"], width=1)
        leaf(draw, (34, 9, 44, 14), p, pointing_right=False, tip_dy=-1,
             blade=p["gold"], ink=p["gold_dark"])

    # grafting band over the union: the visual join between old and new
    draw.rectangle((41, 31, 55, 35), fill=dark)
    draw.rectangle((42, 32, 54, 34), fill=band)
    draw.line((42, 32, 54, 32), fill=light, width=1)
    draw.line((42, 34, 54, 34), fill=root_dark, width=1)
    for x in (44, 47, 50, 53):
        draw.point((x, 33), fill=dark)


ICONS = {
    "conductor": draw_conductor,
    "reader": draw_reader,
    "auditor": draw_auditor,
    "dispatcher": draw_dispatcher,
    "clerk": draw_clerk,
    "scion35": lambda d, p: draw_scion(d, p, big=False),
    "scion176": lambda d, p: draw_scion(d, p, big=True),
    "scion_drafter": lambda d, p: draw_scion(d, p, big=False, drafter=True),
}

LABELS = {
    "conductor": "CONDUCTOR",
    "reader": "READER",
    "auditor": "AUDITOR",
    "dispatcher": "DISPATCHER",
    "clerk": "CLERK",
    "scion35": "SCION",
    "scion176": "SCION",
    "scion_drafter": "SCION",
}

PARAMS = {"scion35": "35B-A3B", "scion176": "176B-A6B",
          "scion_drafter": "35B-A3B"}


def make(name: str) -> Path:
    p = PALETTES[name]
    bg, accent, accent2 = p["bg"], p["accent"], p["accent2"]
    light, dark = p["light"], p["dark"]
    img = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # border and body
    d.polygon(hex_points(0), fill=(*dark, 255))
    d.polygon(hex_points(2), fill=(*accent, 255))
    d.polygon(hex_points(4), fill=(*bg, 255))
    # icon
    ICONS[name](d, p)
    if name == "scion_drafter":
        # three-line banner: title, params, and the gold side-car caption
        banner = [(14, 61), (82, 61), (70, 86), (26, 86)]
        d.polygon(banner, fill=(*dark, 245))
        d.line((14, 61, 82, 61), fill=accent, width=1)
        text(d, LABELS[name], 63, accent)
        text(d, PARAMS[name], 72, light)
        mini_text(d, "DRAFTER", 81, p["gold"])
    elif name in PARAMS:
        # two-line banner, inset from the wider hexagon
        banner = [(14, 61), (82, 61), (70, 86), (26, 86)]
        d.polygon(banner, fill=(*dark, 245))
        d.line((14, 61, 82, 61), fill=accent, width=1)
        text(d, LABELS[name], 64, accent)
        text(d, PARAMS[name], 75, light)
    else:
        # single-line label banner, inset from the wider hexagon
        d.polygon([(14, 64), (82, 64), (76, 80), (20, 80)],
                  fill=(*dark, 240))
        d.line((14, 64, 82, 64), fill=accent, width=1)
        d.line((20, 80, 76, 80), fill=accent2, width=1)
        text(d, LABELS[name], 67, light)
        # base diamond
        d.polygon([(48, 82), (51, 86), (48, 90), (45, 86)], fill=accent2)
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
