#!/usr/bin/env python
"""Hive 1.7B heroes at 256 logical pixels.

The 128px badge could not carry the retrieval pass, so the Reader is drawn on
a 256px canvas: a task label, three scored chunks, a lens that reports the
match state, a context tray, and a completed result card.

    python experiments/make_hive_heroes_256.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

import make_hive_heroes as h

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "assets" / "heroes"
SIZE = 256
S = 3
PX = SIZE * S

PALETTES = dict(h.PALETTES)
PALETTES["reader"] = {
    "bg": (20, 14, 7),
    "bg2": (38, 26, 12),
    "accent": (255, 183, 3),       # hive honey
    "accent2": (196, 120, 45),     # warm copper for stored content
    "hive": (105, 75, 18),          # dim honeycomb lines
    "encode1": (30, 58, 138),      # deep blue reading path
    "encode2": (30, 58, 138),
    "low": (210, 60, 50),          # no-match red
    "mid": (120, 170, 255),        # close light blue
    "high": (70, 200, 110),        # found green
    "high_text": (35, 135, 70),    # completion label
    "light": (246, 241, 227),
    "dark": (10, 6, 2),
    "glow": (255, 183, 3),
}

PALETTES["conductor"] = {
    "bg": (10, 14, 34),
    "bg2": (22, 30, 60),
    "accent": (255, 183, 3),
    "accent2": (78, 132, 255),
    "hive": (105, 75, 18),
    "light": (246, 241, 227),
    "dark": (5, 8, 18),
    "glow": (78, 132, 255),
    "high": (70, 200, 110),
    "high_text": (35, 135, 70),
    "low": (78, 132, 255),          # worker blue replaces red for blocked
    "mid": (120, 170, 255),
}

PALETTES["auditor"] = {
    "bg": (20, 14, 8),
    "bg2": (38, 26, 14),
    "accent": (255, 183, 3),        # hive honey
    "accent2": (78, 132, 255),      # scan / active blue
    "hive": (105, 75, 18),
    "copper": (196, 120, 45),       # model blocks
    "glass": (14, 22, 38),          # lens glass
    "low": (210, 60, 50),           # reject / over capacity
    "mid": (120, 170, 255),
    "high": (70, 200, 110),         # pass / placed
    "high_text": (35, 135, 70),
    "light": (246, 241, 227),
    "dark": (8, 5, 2),
    "glow": (255, 183, 3),
}

PALETTES["toolmanager"] = {
    "bg": (22, 12, 34),
    "bg2": (40, 22, 60),
    "accent": (255, 183, 3),        # hive honey
    "accent2": (78, 132, 255),      # active blue
    "hive": (95, 60, 120),          # dim purple honeycomb
    "mid": (160, 130, 255),         # idle lavender
    "high": (70, 200, 110),         # selected / done
    "high_text": (35, 135, 70),
    "low": (210, 60, 50),
    "light": (246, 241, 227),
    "dark": (8, 4, 14),
    "glow": (192, 139, 255),
}

PALETTES["dispatcher"] = {
    "bg": (7, 18, 30),
    "bg2": (14, 34, 54),
    "accent": (255, 183, 3),        # hive honey / jobs
    "accent2": (90, 210, 255),      # cyan timeline / playhead
    "wire": (176, 128, 255),        # violet dependency wires
    "hive": (40, 90, 110),          # dim cyan honeycomb
    "mid": (120, 180, 220),
    "high": (70, 200, 110),         # completed / on time
    "high_text": (35, 135, 70),
    "low": (210, 60, 50),
    "light": (236, 247, 255),
    "dark": (3, 8, 14),
    "glow": (176, 128, 255),
}

SEARCH_TASK = "SEARCH TASK"
TASK_COMPLETE = "TASK COMPLETE"
PLAN_TASK = "PLAN TASK"
PLAN_READY = "PLAN READY"
EXECUTING = "EXECUTING"
COMPLETE = "COMPLETE"
AUDIT_TASK = "AUDIT TASK"
PASS = "PASS"
TOOL_TASK = "TOOL TASK"
DISPATCH_TASK = "DISPATCH TASK"
LENS_WORD = "DOC"


def hex_points(inset: int = 0):
    return [(64 + inset, 8 + inset), (192 - inset, 8 + inset),
            (248 - inset, 128), (208 - inset, 240 - inset),
            (48 + inset, 240 - inset), (8 + inset, 128)]


def label(draw, text: str, x: int, y: int, color, *, scale: int = 2,
          spacing: int = 1) -> None:
    h.bitmap_text(draw, text, x, y, color, scale=scale, spacing=spacing)


def _honeycomb(p, mask) -> Image.Image:
    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    r = 15
    hh = int(round(math.sqrt(3) * r))
    for row, cy in enumerate(range(30, 228, hh)):
        for col, cx in enumerate(range(12, 248, 3 * r)):
            ox = (1.5 * r) if row % 2 else 0
            pts = [(cx + ox + r * math.cos(math.pi / 3 * i),
                    cy + r * math.sin(math.pi / 3 * i)) for i in range(6)]
            d.polygon(pts, outline=(*p.get("hive", p["bg2"]), 255))
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), mask))
    return layer


def _honeycomb_uniform(p, mask, r: int = 18) -> Image.Image:
    """A true flat-top honeycomb: every cell shares its edges."""
    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    height = math.sqrt(3) * r
    for col in range(-1, 16):
        cx = col * 1.5 * r
        for row in range(-1, 15):
            cy = row * height + (height / 2 if col % 2 else 0)
            pts = [(cx + r * math.cos(math.pi / 3 * i),
                    cy + r * math.sin(math.pi / 3 * i)) for i in range(6)]
            d.polygon(pts, outline=(*p.get("hive", p["bg2"]), 255))
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), mask))
    return layer


def dispatch_hex_points(inset: int = 0):
    """The Dispatcher badge is 8% larger than the shared 256px badge."""
    return [(128 + (x - 128) * 1.08, 128 + (y - 128) * 1.08)
            for x, y in hex_points(inset)]


def badge_base(name: str = "reader") -> Image.Image:
    p = PALETTES[name]
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    points = dispatch_hex_points if name == "dispatcher" else hex_points
    d.polygon(points(0), fill=(*p["dark"], 255))
    d.polygon(points(2), fill=(*p["accent"], 255))
    d.polygon(points(6), fill=(*p["bg2"], 255))
    d.polygon(points(8), fill=(*p["bg"], 255))

    inner = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(inner).polygon(points(10), fill=255)
    pattern = _honeycomb_uniform if name == "dispatcher" else _honeycomb
    img.alpha_composite(pattern(p, inner))

    d.rectangle((104, 18, 152, 21), fill=p["accent"])
    label(d, "HIVE", 105, 26, p["light"])

    d.polygon([(30, 174), (226, 174), (204, 232), (52, 232)],
              fill=p["dark"])
    d.line((30, 174, 226, 174), fill=p["accent"], width=2)
    role_label = h.LABELS[name]
    role_w = h.bitmap_width(role_label, scale=2)
    label(d, role_label, 128 - role_w // 2, 182, p["light"])
    label(d, "1.7B", 105, 204, p["accent"])
    return img


def _dash_runs(x0: int, x1: int, seed: int):
    lengths = (14, 10, 16, 8, 12, 6)
    runs = []
    x = x0
    i = seed
    while x <= x1 - 4:
        w = lengths[i % len(lengths)]
        end = min(x + w, x1)
        if end - x >= 4:
            runs.append((x, end))
        x = end + 4
        i += 1
    return runs


def draw_book(img: Image.Image, p) -> None:
    d = ImageDraw.Draw(img)
    d.rectangle((36, 64, 220, 168), fill=p["dark"])
    d.rectangle((40, 72, 124, 164), fill=p["light"])
    d.rectangle((132, 72, 216, 164), fill=p["light"])
    d.rectangle((124, 72, 132, 164), fill=p["dark"])
    d.rectangle((36, 64, 220, 72), fill=p["accent"])
    # only the search page keeps a little paper texture; the result page is clean
    texture = tuple(round(p["light"][i] * 0.78 + p["accent2"][i] * 0.22)
                    for i in range(3))
    for i in range(6):
        y = 82 + i * 14
        for run in _dash_runs(44, 68, i):
            d.rectangle((run[0], y, run[1], y + 1), fill=texture)


def draw_centered_label(img: Image.Image, p, text: str) -> None:
    d = ImageDraw.Draw(img)
    w = h.bitmap_width(text, scale=1)
    x = 128 - w // 2
    y = 44
    label(d, text, x, y, p["accent"], scale=1)
    d.line((x, y + 8, x + w - 1, y + 8), fill=p["accent"], width=1)


def draw_task_label(img: Image.Image, p) -> None:
    """The global task, centred, replacing the old query box."""
    draw_centered_label(img, p, SEARCH_TASK)


def draw_chunk_card(img: Image.Image, p, rect, seed: int, *, selected=False,
                    dim=False, edge=None) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    edge = edge or (p["accent"] if selected else p["accent2"])
    fill = h._glass_colour(p) if selected else p["light"]
    d.rectangle((x0, y0, x1, y1), fill=fill, outline=edge, width=2)
    ink = p["accent2"]
    if dim:
        ink = tuple(round(p["accent2"][i] * 0.55 + p["light"][i] * 0.45)
                    for i in range(3))
    for yy in (y0 + 7, y0 + 14):
        for run in _dash_runs(x0 + 6, x1 - 6, seed):
            d.rectangle((run[0], yy, run[1], yy + 1), fill=ink)
    if selected:
        d.line((x0, y0, x0, y1), fill=edge, width=4)


def draw_score_bar(img: Image.Image, p, rect, score: float, *,
                   fill_color, tip_color=None, tip_height: int = 1) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    d.rectangle((x0, y0, x1, y1), fill=p["dark"])
    d.rectangle((x0, y0, x1, y1), outline=p["accent2"], width=1)
    height = round((y1 - y0 - 2) * score)
    if height:
        d.rectangle((x0 + 1, y1 - height, x1 - 1, y1 - 1), fill=fill_color)
        if tip_color:
            d.rectangle((x0 + 1, y1 - height, x1 - 1,
                         y1 - height + tip_height - 1), fill=tip_color)
    d.point((x0 + 1, y0 + 1), fill=p["accent2"])


def _draw_status(d, p, cx: int, cy: int, status: str) -> None:
    if status == "low":
        col = p["low"]
        d.line((cx - 7, cy - 7, cx + 7, cy + 7), fill=col, width=3)
        d.line((cx + 7, cy - 7, cx - 7, cy + 7), fill=col, width=3)
    elif status == "mid":
        col = p["encode1"]
        # a blue tick only; the blue score colour carries "close"
        d.line((cx - 6, cy + 2, cx - 2, cy + 6), fill=col, width=3)
        d.line((cx - 2, cy + 6, cx + 7, cy - 3), fill=col, width=3)
    elif status == "high":
        col = p["high"]
        d.line((cx - 7, cy + 1, cx - 2, cy + 6), fill=col, width=3)
        d.line((cx - 2, cy + 6, cx + 8, cy - 5), fill=col, width=3)


def draw_lens(img: Image.Image, p, cx: int, cy: int, *,
              handle_len: int = 16, card_lines: bool = False,
              status: str | None = None) -> None:
    d = ImageDraw.Draw(img)
    r_out, r_rim, r_glass = 24, 22, 20
    hx, hy = cx + 16 + handle_len, cy + 16 + handle_len
    d.line((cx + 16, cy + 16, hx, hy), fill=p["dark"], width=10)
    d.line((cx + 16, cy + 16, hx - 2, hy - 2), fill=p["accent"], width=4)
    d.ellipse((cx - r_out, cy - r_out, cx + r_out, cy + r_out),
              fill=p["dark"])
    d.ellipse((cx - r_rim, cy - r_rim, cx + r_rim, cy + r_rim),
              fill=p["accent"])
    d.ellipse((cx - r_glass, cy - r_glass, cx + r_glass, cy + r_glass),
              fill=h._glass_colour(p), outline=p["accent2"], width=1)
    if status:
        _draw_status(d, p, cx, cy, status)
    elif card_lines:
        for yy in (cy - 4, cy + 3):
            d.rectangle((cx - 14, yy, cx + 14, yy + 1), fill=p["accent2"])
    d.rectangle((cx - 11, cy - 13, cx - 9, cy - 11), fill=p["light"])


def draw_context_card(img: Image.Image, p, rect, seed: int, *,
                      edge) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    d.rectangle((x0, y0, x1, y1), fill=h._glass_colour(p),
                outline=p["accent2"], width=2)
    d.line((x0, y0, x0, y1), fill=edge, width=4)
    for yy in (y0 + 7, y0 + 17):
        for run in _dash_runs(x0 + 8, x1 - 8, seed):
            d.rectangle((run[0], yy, run[1], yy + 1), fill=p["accent2"])


def draw_context_beam(img: Image.Image, p, rect, *, pulse=1.0) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    cy = y0 + 12
    left, right = x0 + 5, x1 - 5
    cx = (left + right) // 2
    blue = p["encode1"]
    glass = h._glass_colour(p)
    edge = tuple(round(glass[i] * (1.0 - pulse) + blue[i] * pulse)
                 for i in range(3))
    d.line((left, cy, cx, cy), fill=edge, width=2)
    d.line((cx, cy, right, cy), fill=edge, width=2)
    d.polygon([(left, cy), (left + 6, cy - 3), (left + 6, cy + 3)],
              fill=edge)
    d.polygon([(right, cy), (right - 6, cy - 3), (right - 6, cy + 3)],
              fill=edge)
    # solid blue reads: no cream pips punched through the arrow


def draw_answer(img: Image.Image, p, dy: int = 0) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 136, 140 + dy, 212, 162 + dy
    d.rectangle((x0, y0, x1, y1), fill=p["dark"], outline=p["high"],
                width=2)
    label(d, LENS_WORD, x0 + 5, y0 + 4, p["light"])
    d.line((x0 + 50, y0 + 12, x0 + 54, y0 + 16), fill=p["high"], width=3)
    d.line((x0 + 54, y0 + 16, x0 + 64, y0 + 4), fill=p["high"], width=3)


def draw_task_complete(img: Image.Image, p) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 136, 73, 212, 82
    d.rectangle((x0, y0, x1, y1), fill=p["high_text"], outline=p["high"],
                width=1)
    w = h.bitmap_width(TASK_COMPLETE, scale=1, spacing=0)
    label(d, TASK_COMPLETE, x0 + (76 - w) // 2, y0 + 1, p["light"],
          scale=1, spacing=0)


def render_reader() -> Image.Image:
    p = PALETTES["reader"]
    img = badge_base("reader")
    draw_book(img, p)
    draw_task_label(img, p)

    cards = [(42, 78, 112, 100), (42, 104, 112, 126), (42, 130, 112, 152)]
    bars = [(114, 80, 121, 100), (114, 106, 121, 126), (114, 132, 121, 152)]
    scores = (0.32, 0.72, 0.92)
    edges = (p["accent2"], p["mid"], p["high"])
    fills = (p["low"], p["mid"], p["high"])
    for i, (card, bar) in enumerate(zip(cards, bars)):
        draw_chunk_card(img, p, card, seed=i, selected=(i in (1, 2)),
                        dim=(i == 0), edge=edges[i])
        draw_score_bar(img, p, bar, scores[i], fill_color=fills[i],
                       tip_color=p["accent"], tip_height=2)

    draw_lens(img, p, 77, 141, status="high")

    draw_task_complete(img, p)
    draw_context_card(img, p, (136, 84, 212, 110), seed=0, edge=p["high"])
    draw_context_beam(img, p, (136, 84, 212, 110))
    draw_context_card(img, p, (136, 112, 212, 138), seed=1, edge=p["mid"])
    draw_context_beam(img, p, (136, 112, 212, 138))
    draw_answer(img, p)
    return img.resize((PX, PX), Image.Resampling.NEAREST)


def _blend(c1, c2, u: float):
    return tuple(round(c1[i] * (1.0 - u) + c2[i] * u) for i in range(3))


def _hex_pts(cx: int, cy: int, r: int):
    return [(cx + r * math.cos(math.pi / 3 * i),
             cy + r * math.sin(math.pi / 3 * i)) for i in range(6)]


def draw_task_card(img: Image.Image, p, rect, letter: str, state: str,
                   *, done_color=None) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    done = done_color or p["high"]
    edge = {
        "blocked": p["low"],
        "ready": p["accent"],
        "active": p["accent2"],
        "done": done,
    }[state]
    fill = p["bg2"]
    d.rectangle((x0, y0, x1, y1), fill=fill, outline=edge, width=2)
    d.line((x0, y0, x0, y1), fill=edge, width=4)
    ink = p["light"] if state != "blocked" else _blend(p["light"], p["bg2"], 0.55)
    d.rectangle((x0 + 6, y0 + 8, x0 + 8, y0 + 10), fill=ink)
    h.bitmap_text(d, letter, x0 + 12, y0 + 3, ink, scale=2)
    if state == "done":
        d.line((x1 - 12, y0 + 8, x1 - 8, y0 + 12), fill=done, width=2)
        d.line((x1 - 8, y0 + 12, x1 - 2, y0 + 3), fill=done, width=2)
    elif state == "blocked":
        d.rectangle((x1 - 10, y0 + 6, x1 - 5, y0 + 11), fill=p["low"])
        d.point((x1 - 8, y0 + 8), fill=p["dark"])
    elif state == "ready":
        d.rectangle((x1 - 9, y0 + 7, x1 - 6, y0 + 10), fill=p["accent"])
    else:
        d.rectangle((x1 - 9, y0 + 7, x1 - 6, y0 + 10), fill=p["accent2"])


def draw_dep_arrow(img: Image.Image, p, start, end, state: str) -> None:
    d = ImageDraw.Draw(img)
    color = {"blocked": p["low"], "ready": p["accent"],
             "done": p["high"]}[state]
    sx, sy = start
    ex, ey = end
    d.line((sx, sy, ex, ey), fill=color, width=2)
    ang = math.atan2(ey - sy, ex - sx)
    bx = ex - 5 * math.cos(ang)
    by = ey - 5 * math.sin(ang)
    px = 3 * -math.sin(ang)
    py = 3 * math.cos(ang)
    d.polygon([(ex, ey), (bx + px, by + py), (bx - px, by - py)],
              fill=color)


def draw_spinner(img: Image.Image, p, rect, phase: float, *,
                 dot: int = 3, spread: int = 3) -> None:
    """A four-dot loading spinner, one bright head rotating clockwise."""
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    points = ((0, -spread), (spread, 0), (0, spread), (-spread, 0))
    head = int(phase) % 4
    for i, (dx, dy) in enumerate(points):
        color = p["light"] if i == head else p["accent2"]
        d.rectangle((cx + dx - dot + 1, cy + dy - dot + 1,
                     cx + dx, cy + dy), fill=color)


def draw_worker_card(img: Image.Image, p, rect, label_text: str,
                     tasks: list[str], *, slot_colors=None) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    d.rectangle((x0, y0, x1, y1), fill=p["bg2"], outline=p["accent2"],
                width=2)
    d.line((x0, y0, x0, y1), fill=p["accent2"], width=4)
    # centre the full worker label and the slot row inside the card
    label(d, label_text, x0 + 6, y0 + 4, p["light"], scale=1)
    for i in range(3):
        sx = x0 + 7 + i * 16
        sy = y0 + 13
        filled = i < len(tasks)
        if filled:
            color = slot_colors[i] if slot_colors else p["high"]
            fill, edge = color, color
        else:
            fill = p["dark"]
            edge = p["mid"]
        d.rectangle((sx, sy, sx + 13, sy + 10), fill=fill, outline=edge,
                    width=1)
        if filled:
            h.bitmap_text(d, tasks[i], sx + 4, sy + 2, p["dark"], scale=1)


def draw_plan_card(img: Image.Image, p, *, mode: str = "ready",
                   spin_phase: float = 0.0) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 156, 144, 216, 168
    d.rectangle((x0, y0, x1, y1), fill=p["dark"], outline=p["high"],
                width=2)
    if mode == "ready":
        for text, yy in (("PLAN", y0 + 4), ("READY", y0 + 12)):
            w = h.bitmap_width(text, scale=1)
            label(d, text, x0 + (60 - w) // 2, yy, p["high"], scale=1)
    elif mode == "executing":
        w = h.bitmap_width(EXECUTING, scale=1)
        label(d, EXECUTING, x0 + (60 - w) // 2, y0 + 4, p["mid"],
              scale=1)
        draw_spinner(img, p, (x0 + 21, y0 + 12, x0 + 39, y0 + 20),
                     spin_phase)
    else:
        w = h.bitmap_width(COMPLETE, scale=1)
        label(d, COMPLETE, x0 + (60 - w) // 2, y0 + 4, p["high"], scale=1)
        cx = x0 + 30
        d.line((cx - 5, y0 + 15, cx - 1, y0 + 19), fill=p["high"], width=2)
        d.line((cx - 1, y0 + 19, cx + 6, y0 + 12), fill=p["high"], width=2)


def draw_conductor_core(img: Image.Image, p, *, pulse: float = 0.0,
                        pupil=(0, 0)) -> None:
    d = ImageDraw.Draw(img)
    cx, cy = 124, 118
    # white hexagon core: the conductor's eye
    d.polygon(_hex_pts(cx, cy, 18), fill=p["light"], outline=p["accent"],
              width=2)
    # dark iris/pupil, offset toward the worker being routed to
    px, py = cx + pupil[0], cy + pupil[1]
    rows = ("..###..", ".#####.", "#######", "#######",
            "#######", ".#####.", "..###..")
    for j, row in enumerate(rows):
        for i, ch in enumerate(row):
            if ch == "#":
                d.point((px - 3 + i, py - 3 + j), fill=p["dark"])
    if pulse > 0:
        # blue ring = the core is routing, not idle
        d.polygon(_hex_pts(cx, cy, 22), outline=p["accent2"], width=2)


def render_conductor() -> Image.Image:
    p = PALETTES["conductor"]
    img = badge_base("conductor")
    draw_centered_label(img, p, PLAN_TASK)

    tasks = [(52, 66, 108, 86, "1"), (32, 104, 88, 124, "2"),
             (52, 142, 108, 162, "3")]
    for x0, y0, x1, y1, letter in tasks:
        done_color = p["accent"] if letter == "3" else p["high"]
        draw_task_card(img, p, (x0, y0, x1, y1), letter, "done",
                       done_color=done_color)
    draw_dep_arrow(img, p, (80, 86), (60, 104), "done")
    draw_dep_arrow(img, p, (60, 124), (80, 142), "done")

    draw_conductor_core(img, p)
    draw_worker_card(img, p, (156, 76, 216, 104), "WORKER 1", ["1", "2"],
                     slot_colors=[p["high"], p["high"]])
    draw_worker_card(img, p, (156, 110, 216, 138), "WORKER 2", ["3"],
                     slot_colors=[p["accent"]])
    draw_plan_card(img, p, mode="complete")
    return img.resize((PX, PX), Image.Resampling.NEAREST)


DISPATCH_LANES = ((66, 94), (98, 126), (130, 158))
DISPATCH_X0, DISPATCH_X1 = 92, 202
DISPATCH_TMAX = 12
DISPATCH_DEADLINE = 12
DISPATCH_MAKESPAN = 12
DISPATCH_JOBS = (
    ("BUILD", 0, 4, 0),
    ("TEST", 4, 7, 0),
    ("RELEASE", 7, 12, 0),
    ("WRITE", 0, 4, 1),
    ("CHECK", 4, 7, 2),
)


def dispatch_x(t: float) -> int:
    return DISPATCH_X0 + round(t * (DISPATCH_X1 - DISPATCH_X0) / DISPATCH_TMAX)


def draw_dispatch_grid(img: Image.Image, p) -> None:
    d = ImageDraw.Draw(img)
    d.rectangle((44, 58, 91, 65), fill=p["dark"])
    label(d, "WORKER", 48, 60, p["light"], scale=1)
    for i, (y0, y1) in enumerate(DISPATCH_LANES):
        d.rectangle((DISPATCH_X0, y0, DISPATCH_X1, y1), fill=p["bg2"],
                    outline=p["mid"], width=1)
        label(d, str(i + 1), 64, y0 + 7, p["light"], scale=2)
    for t in (0, 4, 8, 12):
        x = dispatch_x(t)
        colour = p["low"] if t == DISPATCH_DEADLINE else p["mid"]
        text = str(t)
        w = h.bitmap_width(text, scale=1)
        label(d, text, x - w // 2, 54, colour, scale=1)
        d.line((x, 60, x, 64), fill=colour, width=1)
    deadline = dispatch_x(DISPATCH_DEADLINE)
    d.line((deadline, 60, deadline, DISPATCH_LANES[2][1]),
           fill=p["low"], width=2)
    d.polygon([(deadline - 3, 60), (deadline + 3, 60), (deadline, 65)],
              fill=p["low"])


def draw_dispatch_job(img: Image.Image, p, name: str, start: float,
                      end: float, lane: int, *, dim: bool = False) -> None:
    d = ImageDraw.Draw(img)
    x0, x1 = dispatch_x(start), dispatch_x(end) - 1
    y0, y1 = DISPATCH_LANES[lane]
    top, bot = y0 + 3, y1 - 3
    edge = p["hive"] if dim else p["accent"]
    ink = p["mid"] if dim else p["light"]
    d.rectangle((x0, top, x1, bot), fill=p["dark"], outline=edge,
                width=2)
    width = x1 - x0
    if width >= 34:
        d.rectangle((x0 + 1, top + 2, x0 + 2, bot - 2), fill=edge)
    pad = 4 if width >= 30 else 2
    spacing = 0 if name == "CHECK" and width < 34 else 1
    label(d, name, x0 + pad, top + 3, ink, scale=1, spacing=spacing)
    label(d, str(end - start), x0 + pad, top + 11, p["light"], scale=1)


def draw_dispatch_deps(img: Image.Image, p,
                       visible: set[str] | None = None) -> None:
    names = (set(visible) if visible is not None
             else {job[0] for job in DISPATCH_JOBS})
    d = ImageDraw.Draw(img)
    # same-lane precedence marks
    for left, right, at in (("BUILD", "TEST", 4), ("TEST", "RELEASE", 7)):
        if {left, right} <= names:
            x = dispatch_x(at)
            d.polygon([(x - 2, 77), (x + 2, 81), (x - 2, 85)],
                      fill=p["wire"])
    # WRITE -> CHECK: two side jobs form the second chain
    if {"WRITE", "CHECK"} <= names:
        wx = (dispatch_x(0) + dispatch_x(4)) // 2
        wy = DISPATCH_LANES[1][1] - 3
        lx = (dispatch_x(4) + dispatch_x(7)) // 2
        gap = DISPATCH_LANES[1][1] + 2
        d.line((wx, wy, wx, gap), fill=p["wire"], width=2)
        d.line((wx, gap, lx, gap), fill=p["wire"], width=2)
        d.line((lx, gap, lx, DISPATCH_LANES[2][0] + 1),
               fill=p["wire"], width=2)
        d.polygon([(lx - 3, DISPATCH_LANES[2][0] + 1),
                   (lx + 3, DISPATCH_LANES[2][0] + 1),
                   (lx, DISPATCH_LANES[2][0] + 3)], fill=p["wire"])
    # CHECK -> RELEASE: the chain folds back into the final step
    if {"CHECK", "RELEASE"} <= names:
        cx = dispatch_x(7) - 1
        cy = (DISPATCH_LANES[2][0] + DISPATCH_LANES[2][1]) // 2
        jx = cx + 6
        top = DISPATCH_LANES[1][0] - 2
        relx = (dispatch_x(7) + dispatch_x(12)) // 2
        bottom = DISPATCH_LANES[0][1] - 3
        d.line((cx, cy, jx, cy), fill=p["wire"], width=2)
        d.line((jx, cy, jx, top), fill=p["wire"], width=2)
        d.line((jx, top, relx, top), fill=p["wire"], width=2)
        d.line((relx, top, relx, bottom + 2), fill=p["wire"], width=2)
        d.polygon([(relx - 3, bottom + 2), (relx + 3, bottom + 2),
                   (relx, bottom)], fill=p["wire"])


def draw_dispatch_summary(img: Image.Image, p, *,
                          makespan: int | None, on_time: bool,
                          status: str = "", status_colour=None) -> None:
    d = ImageDraw.Draw(img)
    edge = p["high"] if on_time else p["mid"]
    d.rectangle((36, 160, 220, 171), fill=p["dark"], outline=edge,
                width=1)
    label(d, "MAKESPAN", 47, 163, p["mid"], scale=1)
    if makespan is not None:
        label(d, str(makespan), 99, 163, p["light"], scale=1)
    label(d, "MINUTES", 115, 163, p["mid"], scale=1)
    ink = status_colour or p["accent"]
    if on_time:
        label(d, "ON TIME", 168, 163, ink, scale=1)
    elif status:
        w = h.bitmap_width(status, scale=1)
        label(d, status, 209 - w, 163, ink, scale=1)


def render_dispatcher() -> Image.Image:
    p = PALETTES["dispatcher"]
    img = badge_base("dispatcher")
    draw_centered_label(img, p, DISPATCH_TASK)
    draw_dispatch_grid(img, p)
    for name, start, end, lane in DISPATCH_JOBS:
        draw_dispatch_job(img, p, name, start, end, lane)
    draw_dispatch_deps(img, p)
    draw_dispatch_summary(img, p, makespan=DISPATCH_MAKESPAN,
                          on_time=True)
    return img.resize((PX, PX), Image.Resampling.NEAREST)


TOOL_TILES = (
    ((66, 66, 106, 98), "READ", "read"),
    ((108, 66, 148, 98), "EDIT", "edit"),
    ((150, 66, 190, 98), "BASH", "bash"),
    ((66, 102, 106, 134), "TEST", "test"),
    ((108, 102, 148, 134), "SEARCH", "search"),
    ((150, 102, 190, 134), "IMAGE", "image"),
)


def draw_tool_icon(d, p, kind: str, x0: int, y0: int, ink) -> None:
    if kind == "read":
        d.rectangle((x0 + 13, y0 + 5, x0 + 25, y0 + 19), outline=ink,
                    width=1)
        for yy in (y0 + 9, y0 + 12, y0 + 15):
            d.line((x0 + 15, yy, x0 + 23, yy), fill=ink, width=1)
    elif kind == "edit":
        # document outline, open at the top-right with room for the pencil
        d.line((x0 + 13, y0 + 5, x0 + 19, y0 + 5), fill=ink, width=1)
        d.line((x0 + 12, y0 + 6, x0 + 12, y0 + 18), fill=ink, width=1)
        d.line((x0 + 13, y0 + 19, x0 + 24, y0 + 19), fill=ink, width=1)
        d.line((x0 + 25, y0 + 13, x0 + 25, y0 + 18), fill=ink, width=1)
        d.line((x0 + 12, y0 + 6, x0 + 13, y0 + 5), fill=ink, width=1)
        d.line((x0 + 12, y0 + 18, x0 + 13, y0 + 19), fill=ink, width=1)
        d.line((x0 + 24, y0 + 19, x0 + 25, y0 + 18), fill=ink, width=1)
        # pencil: sharpened tip, body, small eraser outside
        d.polygon([(x0 + 14, y0 + 16), (x0 + 16, y0 + 12),
                   (x0 + 18, y0 + 14)], fill=ink)
        d.line((x0 + 17, y0 + 13, x0 + 25, y0 + 5), fill=ink, width=2)
        d.line((x0 + 24, y0 + 4, x0 + 26, y0 + 6), fill=ink, width=2)
    elif kind == "bash":
        d.rectangle((x0 + 12, y0 + 5, x0 + 26, y0 + 19), outline=ink,
                    width=1)
        d.line((x0 + 15, y0 + 9, x0 + 18, y0 + 12), fill=ink, width=1)
        d.line((x0 + 18, y0 + 12, x0 + 15, y0 + 15), fill=ink, width=1)
        d.line((x0 + 20, y0 + 15, x0 + 24, y0 + 15), fill=ink, width=1)
    elif kind == "test":
        d.rectangle((x0 + 13, y0 + 5, x0 + 25, y0 + 19), outline=ink,
                    width=1)
        d.line((x0 + 15, y0 + 11, x0 + 17, y0 + 14), fill=ink, width=1)
        d.line((x0 + 17, y0 + 14, x0 + 21, y0 + 8), fill=ink, width=1)
        d.line((x0 + 15, y0 + 16, x0 + 23, y0 + 16), fill=ink, width=1)
    elif kind == "search":
        d.ellipse((x0 + 12, y0 + 5, x0 + 22, y0 + 15), outline=ink,
                  width=1)
        d.line((x0 + 21, y0 + 14, x0 + 26, y0 + 19), fill=ink, width=2)
    else:
        d.rectangle((x0 + 12, y0 + 5, x0 + 26, y0 + 19), outline=ink,
                    width=1)
        d.ellipse((x0 + 14, y0 + 7, x0 + 17, y0 + 10), fill=ink)
        d.line((x0 + 13, y0 + 17, x0 + 18, y0 + 12), fill=ink, width=1)
        d.line((x0 + 18, y0 + 12, x0 + 21, y0 + 15), fill=ink, width=1)
        d.line((x0 + 21, y0 + 15, x0 + 23, y0 + 13), fill=ink, width=1)
        d.line((x0 + 23, y0 + 13, x0 + 25, y0 + 17), fill=ink, width=1)


def draw_tool_tile(img: Image.Image, p, rect, name: str, kind: str,
                   state: str) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    edge = {"idle": p["mid"], "active": p["accent"],
            "selected": p["high"]}[state]
    ink = p["light"] if state != "idle" else _blend(p["light"], p["bg2"], 0.45)
    d.rectangle((x0, y0, x1, y1), fill=p["bg2"], outline=edge, width=2)
    draw_tool_icon(d, p, kind, x0, y0, ink)
    w = h.bitmap_width(name, scale=1)
    h.bitmap_text(d, name, x0 + ((x1 - x0) - w) // 2, y0 + 22, ink,
                  scale=1)
    if state == "selected":
        d.line((x1 - 11, y0 + 6, x1 - 8, y0 + 9), fill=p["high"], width=2)
        d.line((x1 - 8, y0 + 9, x1 - 4, y0 + 4), fill=p["high"], width=2)


def draw_tool_pipeline(img: Image.Image, p, selected, *, done: bool) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 68, 140, 188, 162
    edge = p["high"] if done else p["accent2"]
    d.rectangle((x0, y0, x1, y1), fill=p["dark"], outline=edge, width=2)
    dim = _blend(p["light"], p["bg2"], 0.55)
    for name, xx in (("READ", x0 + 6), ("EDIT", x0 + 38),
                     ("TEST", x0 + 70)):
        col = p["light"] if name in selected else dim
        h.bitmap_text(d, name, xx, y0 + 7, col, scale=1)
    for ax, prev in ((x0 + 31, "READ"), (x0 + 63, "EDIT")):
        col = p["accent"] if prev in selected else dim
        d.line((ax, y0 + 8, ax + 3, y0 + 10), fill=col, width=1)
        d.line((ax + 3, y0 + 10, ax, y0 + 12), fill=col, width=1)
    if done:
        cx = x0 + 100
        d.line((cx, y0 + 10, cx + 3, y0 + 13), fill=p["high"], width=2)
        d.line((cx + 3, y0 + 13, cx + 8, y0 + 7), fill=p["high"], width=2)


def draw_tool_cursor(img: Image.Image, p, x: int, y: int) -> None:
    d = ImageDraw.Draw(img)
    pts = [(x, y), (x, y + 13), (x + 3, y + 10), (x + 5, y + 15),
           (x + 8, y + 13), (x + 6, y + 9), (x + 10, y + 9)]
    d.polygon(pts, fill=p["light"], outline=p["dark"])


def render_toolmanager() -> Image.Image:
    p = PALETTES["toolmanager"]
    img = badge_base("toolmanager")
    draw_centered_label(img, p, TOOL_TASK)
    selected = {"READ", "EDIT", "TEST"}
    for rect, name, kind in TOOL_TILES:
        state = "selected" if name in selected else "idle"
        draw_tool_tile(img, p, rect, name, kind, state)
    draw_tool_pipeline(img, p, selected, done=True)
    draw_tool_cursor(img, p, 98, 118)
    return img.resize((PX, PX), Image.Resampling.NEAREST)


MODELS = (
    ((36, 76, 92, 104), "MODEL 1", "12 GB"),
    ((36, 108, 92, 136), "MODEL 2", "4 GB"),
    ((36, 140, 92, 168), "MODEL 3", "10 GB"),
)
RACKS = (
    ((156, 76, 216, 108), "C1", "18G", 18),
    ((156, 112, 216, 144), "C2", "14G", 14),
)
VERDICT_RECT = (156, 148, 216, 168)
LENS_CENTER = (124, 122)


def draw_model_card(img: Image.Image, p, rect, name: str, size: str,
                    state: str) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    edge = {
        "queued": p["accent2"],
        "scan": p["accent"],
        "placing": p["accent2"],
        "placed": p["high"],
        "rejected": p["low"],
    }[state]
    d.rectangle((x0, y0, x1, y1), fill=p["bg2"], outline=edge, width=2)
    d.line((x0, y0, x0, y1), fill=edge, width=4)
    ink = p["light"] if state != "queued" else _blend(p["light"], p["bg2"], 0.45)
    nw = h.bitmap_width(name, scale=1)
    sw = h.bitmap_width(size, scale=1)
    h.bitmap_text(d, name, x0 + (56 - nw) // 2, y0 + 6, ink, scale=1)
    h.bitmap_text(d, size, x0 + (56 - sw) // 2, y0 + 15, p["accent"], scale=1)


def draw_rack(img: Image.Image, p, rect, name: str, capacity: str,
              used: int, chips: list[str], *, over: bool = False) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    edge = p["low"] if over else p["accent2"]
    d.rectangle((x0, y0, x1, y1), fill=p["bg2"], outline=edge, width=2)
    d.line((x0, y0, x0, y1), fill=edge, width=4)
    label(d, name, x0 + 6, y0 + 4, p["light"], scale=1)
    cw = h.bitmap_width(capacity, scale=1)
    label(d, capacity, x1 - 6 - cw, y0 + 4, p["accent"], scale=1)
    bx0, bx1 = x0 + 6, x1 - 6
    by0, by1 = y0 + 12, y0 + 16
    d.rectangle((bx0, by0, bx1, by1), fill=p["dark"], outline=p["mid"],
                width=1)
    cap = int(capacity.rstrip("G"))
    ratio = min(used / cap, 1.0)
    fw = round((bx1 - bx0 - 2) * ratio)
    fill = p["low"] if over else p["accent"]
    if fw > 0:
        d.rectangle((bx0 + 1, by0 + 1, bx0 + fw, by1 - 1), fill=fill)
    for i in range(3):
        sx = x0 + 6 + i * 16
        sy = y0 + 19
        if i < len(chips):
            d.rectangle((sx, sy, sx + 13, sy + 9), fill=p["copper"],
                        outline=p["high"], width=1)
            h.bitmap_text(d, chips[i], sx + 1, sy + 1, p["dark"], scale=1)
        else:
            d.rectangle((sx, sy, sx + 13, sy + 9), fill=p["dark"],
                        outline=p["mid"], width=1)
    if over:
        # the next empty slot rejects the model
        idx = len(chips)
        if idx < 3:
            sx = x0 + 6 + idx * 16
            sy = y0 + 19
            d.rectangle((sx, sy, sx + 13, sy + 9), fill=p["low"],
                        outline=p["dark"], width=1)


def draw_auditor_lens(img: Image.Image, p, *, scan=(0, 0), ring=None) -> None:
    d = ImageDraw.Draw(img)
    cx, cy = LENS_CENTER
    # handle behind the lens
    d.line((cx + 12, cy + 12, cx + 26, cy + 26), fill=p["dark"], width=8)
    d.line((cx + 12, cy + 12, cx + 26, cy + 26), fill=p["accent"], width=4)
    # honey rim and dark glass
    d.ellipse((cx - 19, cy - 19, cx + 19, cy + 19), fill=p["dark"])
    d.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=p["accent"])
    d.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=p["glass"])
    d.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), outline=p["accent2"],
              width=1)
    for dx, dy in ((-9, -8), (-8, -9), (-7, -10), (-10, -7), (-10, -6)):
        d.point((cx + dx, cy + dy), fill=p["light"])
    sx, sy = cx + scan[0], cy + scan[1]
    d.rectangle((sx - 2, sy - 2, sx + 2, sy + 2), fill=p["accent"])
    d.point((sx, sy), fill=p["light"])
    if ring:
        d.ellipse((cx - 22, cy - 22, cx + 22, cy + 22), outline=ring, width=2)


def draw_verdict(img: Image.Image, p) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = VERDICT_RECT
    d.rectangle((x0, y0, x1, y1), fill=p["dark"], outline=p["high"],
                width=2)
    w = h.bitmap_width(PASS, scale=1)
    label(d, PASS, x0 + (60 - w) // 2, y0 + 4, p["high"], scale=1)
    cx = x0 + 30
    d.line((cx - 4, y0 + 14, cx - 1, y0 + 17), fill=p["high"], width=2)
    d.line((cx - 1, y0 + 17, cx + 5, y0 + 11), fill=p["high"], width=2)


def render_auditor() -> Image.Image:
    p = PALETTES["auditor"]
    img = badge_base("auditor")
    draw_centered_label(img, p, AUDIT_TASK)
    for rect, name, size in MODELS:
        draw_model_card(img, p, rect, name, size, "placed")
    draw_auditor_lens(img, p)
    draw_rack(img, p, RACKS[0][0], "C1", "18G", 16, ["M1", "M2"])
    draw_rack(img, p, RACKS[1][0], "C2", "14G", 10, ["M3"])
    draw_verdict(img, p)
    return img.resize((PX, PX), Image.Resampling.NEAREST)


def main() -> int:
    for name, render in (("reader", render_reader),
                         ("conductor", render_conductor),
                         ("auditor", render_auditor),
                         ("toolmanager", render_toolmanager),
                         ("dispatcher", render_dispatcher)):
        img = render()
        path = OUT / f"{name}-256.png"
        img.save(path)
        print(path, path.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
