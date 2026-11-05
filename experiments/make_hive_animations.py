#!/usr/bin/env python
"""Animate the Hive Reader hero as a compact RAG pass.

The loop makes the retrieval step visible instead of just showing a lens on a
page:

    query -> score three stored chunks -> select the top two -> assemble
    context -> read both selected chunks -> answer

The page texture is the memory; the lens is the encoder; the score bars in the
left margin are the similarity scores; the right page is the assembled context
and the extracted answer.

    python experiments/make_hive_animations.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

import make_hive_heroes as h

OUT = h.OUT
SIZE = h.SIZE
PX = h.PX
FRAMES = 96
DURATION = 83  # ms, ~12 fps, 8 s loop

LENS_CX = 41
LENS_REST = 46
QUERY = (27, 22, 55, 30)

# The three stored chunks are two page lines each.
CHUNK_LINES = ((43, 49), (55, 61), (67, 73))
CHUNK_RECTS = ((23, 40, 56, 52), (23, 52, 56, 64), (23, 64, 56, 76))
BAR_RECTS = ((57, 42, 61, 51), (57, 54, 61, 63), (57, 66, 61, 75))
SCORES = (0.32, 0.72, 0.92)

# The top two chunks lift into the context tray on the right page.
STRIP_SRC = ((23, 64, 56, 76), (23, 52, 56, 64))
STRIP_DST = ((69, 39, 105, 53), (69, 55, 105, 69))
ANSWER = (69, 70, 105, 81)


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _clamp01(t: float) -> float:
    return min(1.0, max(0.0, t))


def _lens_y(t: float) -> int:
    if t < 0.14:
        return 46
    if t < 0.20:
        return round(_lerp(46, 58, _ease((t - 0.14) / 0.06)))
    if t < 0.28:
        return 58
    if t < 0.34:
        return round(_lerp(58, 70, _ease((t - 0.28) / 0.06)))
    if t < 0.44:
        return 70
    if t < 0.48:
        return round(_lerp(70, 46, _ease((t - 0.44) / 0.04)))
    return 46


def _fill(t: float, idx: int) -> float:
    starts = (0.04, 0.20, 0.34)
    ends = (0.14, 0.28, 0.42)
    return _ease(_clamp01((t - starts[idx]) / (ends[idx] - starts[idx])))


def _lift(t: float, idx: int):
    start = 0.48 + idx * 0.01
    u = _ease(_clamp01((t - start) / 0.12))
    if t < start:
        return None
    sx0, sy0, sx1, sy1 = STRIP_SRC[idx]
    dx0, dy0, dx1, dy1 = STRIP_DST[idx]
    arc = math.sin(math.pi * u) * 3
    return (round(_lerp(sx0, dx0, u)), round(_lerp(sy0, dy0, u) - arc),
            round(_lerp(sx1, dx1, u)), round(_lerp(sy1, dy1, u) - arc))


def _answer_progress(t: float) -> float:
    return _ease(_clamp01((t - 0.78) / 0.06))


def _draw_query(d, p) -> None:
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    x0, y0, x1, y1 = QUERY
    d.rectangle((x0, y0, x1, y1), fill=dark, outline=a)
    d.line((x0 + 1, y0 + 1, x1 - 1, y0 + 1), fill=(*a2, 140))
    h.bitmap_text(d, "Q", x0 + 3, y0 + 2, light)
    for i in range(3):
        d.rectangle((x0 + 15 + i * 3, y0 + 3, x0 + 16 + i * 3, y0 + 4),
                    fill=a)
    # a small encoded-query underline, then a dotted feed into the lens
    d.line((x0 + 3, y1 - 1, x0 + 14, y1 - 1), fill=(*a2, 180))
    for yy in range(y1 + 1, 34, 2):
        d.point((41, yy), fill=(*a, 170))


def _draw_bars(d, p, t: float) -> None:
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    for i, (x0, y0, x1, y1) in enumerate(BAR_RECTS):
        d.rectangle((x0, y0, x1, y1), fill=(*dark, 175))
        progress = _fill(t, i)
        height = round((y1 - y0 - 1) * SCORES[i] * progress)
        if height > 0:
            d.rectangle((x0 + 1, y1 - height, x1 - 1, y1 - 1), fill=a)
            d.rectangle((x0 + 1, y1 - height, x1 - 1, y1 - height),
                        fill=light)
        # a tiny tick at the top of the track
        d.point((x0 + 1, y0), fill=(*a2, 160))


def _draw_chunk_outlines(d, p, t: float) -> None:
    a, a2, light = p["accent"], p["accent2"], p["light"]
    for i, (x0, y0, x1, y1) in enumerate(CHUNK_RECTS):
        if i > 0 and t >= 0.48:
            continue  # lifted chunks leave a clean gap
        progress = _fill(t, i)
        selected = t >= 0.44 and i in (1, 2)
        alpha = int(45 + 190 * progress)
        colour = (*a, alpha) if selected or progress >= 0.99 else (*a2, alpha)
        if selected:
            d.rectangle((x0, y0, x1, y1), fill=(*a, 24), outline=colour, width=2)
        else:
            d.rectangle((x0, y0, x1, y1), outline=colour, width=2)
        # score ticks at the chunk's left edge
        for yy in range(y0 + 3, y1 - 2, 3):
            d.point((x0 - 1, yy), fill=(*a2, 90 if selected else 55))


def _draw_strip(d, p, rect, seed: int) -> None:
    a, a2, light = p["accent"], p["accent2"], p["light"]
    x0, y0, x1, y1 = rect
    d.rectangle((x0, y0, x1, y1), fill=h._glass_colour(p), outline=a2, width=2)
    d.line((x0, y0, x0, y1), fill=a, width=3)
    for yy in (y0 + 4, y0 + 10):
        for run in h._word_runs(x0 + 2, x1 - 3, seed):
            d.line((run[0], yy, run[1], yy), fill=a2)
    # a small score notch in the top-right corner
    d.rectangle((x1 - 3, y0 + 2, x1 - 2, y0 + 3), fill=a)


def _draw_context_beam(d, p, t: float, rect, idx: int) -> None:
    a, a2, light = p["accent"], p["accent2"], p["light"]
    x0, y0, x1, y1 = rect
    cy = y0 + 7
    left, right = x0 + 2, x1 - 2
    cx = (left + right) // 2
    if 0.62 <= t <= 0.80:
        d.line((left, cy, right, cy), fill=(*a2, 90), width=2)
        d.polygon([(left, cy), (left + 4, cy - 3), (left + 4, cy + 3)],
                  fill=(*a2, 240))
        d.polygon([(right, cy), (right - 4, cy - 3), (right - 4, cy + 3)],
                  fill=(*a2, 240))
    for start, end in ((0.62 + idx * 0.02, 0.69 + idx * 0.02),
                       (0.71 + idx * 0.02, 0.78 + idx * 0.02)):
        if not (start <= t <= end):
            continue
        k = (t - start) / (end - start)
        u = _ease(k / 0.58) if k < 0.58 else 1.0
        fade = 1.0 if k < 0.78 else max(0.0, (1.0 - k) / 0.22)
        if fade <= 0:
            continue
        hl = round(_lerp(cx, left + 1, u))
        hr = round(_lerp(cx, right - 1, u))
        fi = int(245 * fade)
        d.line((hl, cy, hl + 5, cy), fill=(*light, fi), width=2)
        d.line((hr - 5, cy, hr, cy), fill=(*light, fi), width=2)
        if u > 0.82:
            fl = int(255 * fade * (u - 0.82) / 0.18)
            d.polygon([(left + 1, cy), (left + 4, cy - 2), (left + 4, cy + 2)],
                      fill=(*light, fl))
            d.polygon([(right - 1, cy), (right - 4, cy - 2), (right - 4, cy + 2)],
                      fill=(*light, fl))


def _draw_answer(d, p, t: float) -> None:
    progress = _answer_progress(t)
    if progress <= 0:
        return
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    dy = round((1.0 - progress) * 2)
    x0, y0, x1, y1 = ANSWER
    y0 += dy
    y1 += dy
    d.rectangle((x0, y0, x1, y1), fill=dark, outline=a, width=2)
    h.bitmap_text(d, h.LENS_WORD, x0 + 3, y0 + 2, light)
    # verified check
    d.line((x0 + 22, y0 + 6, x0 + 25, y0 + 9), fill=a, width=2)
    d.line((x0 + 25, y0 + 9, x0 + 32, y0 + 2), fill=a, width=2)


def frame(t: float) -> Image.Image:
    base = h.render("reader", lens=False)
    p = h.PALETTES["reader"]
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]

    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    _draw_query(d, p)
    _draw_bars(d, p, t)
    _draw_chunk_outlines(d, p, t)

    # the lens is the encoder scanning the memory chunks
    cy = _lens_y(t)
    word_line = 67 if 0.38 <= t < 0.44 else None
    h.draw_reader_lens(d, p, cx=LENS_CX, cy=cy, selected=word_line,
                       handle_len=6)

    # erase the two selected chunks when they lift, then draw the flight
    if t >= 0.48:
        for rect in STRIP_SRC:
            d.rectangle(rect, fill=light)
    if t < 0.60:
        for i in range(2):
            rect = _lift(t, i)
            if rect is not None:
                _draw_strip(d, p, rect, seed=3 - i)
                x0, y0, x1, y1 = rect
                d.line((x0 - 6, (y0 + y1) // 2, x0 - 2, (y0 + y1) // 2),
                       fill=(*a, 80))
    else:
        # the assembled context reads in both directions
        for i, rect in enumerate(STRIP_DST):
            _draw_strip(d, p, rect, seed=3 - i)
            _draw_context_beam(d, p, t, rect, i)

    _draw_answer(d, p, t)

    overlay = layer.resize((PX, PX), Image.Resampling.NEAREST)
    out = base.copy()
    out.alpha_composite(overlay)
    return out


def main() -> int:
    frames = [frame(i / FRAMES) for i in range(FRAMES)]
    path = OUT / "reader.webp"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=DURATION, loop=0, quality=90, method=6)
    print(path, path.stat().st_size, "bytes", len(frames), "frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
