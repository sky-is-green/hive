#!/usr/bin/env python
"""Animate the 256px Hive Reader hero as a scored retrieval pass.

    search task -> score three chunks -> read each one -> keep the matches ->
    read both directions -> extract DOC -> task complete

    python experiments/make_hive_animations_256.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_hive_heroes_256 as m  # noqa: E402

OUT = m.OUT
FRAMES = 96
DURATION = 83  # ms, ~12 fps, 8 s loop

CARDS = ((42, 78, 112, 100), (42, 104, 112, 126), (42, 130, 112, 152))
BARS = ((114, 80, 121, 100), (114, 106, 121, 126), (114, 132, 121, 152))
SCORES = (0.32, 0.72, 0.92)
EDGES = ("accent2", "mid", "high")
FILLS = ("low", "mid", "high")

STRIP_SRC = (CARDS[2], CARDS[1])
STRIP_EDGE = ("high", "mid")
STRIP_DST = ((136, 84, 212, 110), (136, 112, 212, 138))


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _clamp01(t: float) -> float:
    return min(1.0, max(0.0, t))


def _lens_y(t: float) -> int:
    if t < 0.14:
        return 89
    if t < 0.20:
        return round(_lerp(89, 115, _ease((t - 0.14) / 0.06)))
    if t < 0.30:
        return 115
    if t < 0.36:
        return round(_lerp(115, 141, _ease((t - 0.30) / 0.06)))
    if t < 0.46:
        return 141
    if t < 0.52:
        return round(_lerp(141, 89, _ease((t - 0.46) / 0.06)))
    return 89


def _fill(t: float, idx: int) -> float:
    starts = (0.05, 0.21, 0.37)
    ends = (0.13, 0.29, 0.45)
    return _ease(_clamp01((t - starts[idx]) / (ends[idx] - starts[idx])))


def _status(t: float):
    if 0.08 <= t < 0.14:
        return "low"
    if 0.24 <= t < 0.30:
        return "mid"
    if 0.40 <= t < 0.46:
        return "high"
    return None


def _lift(t: float, idx: int):
    start = 0.52
    u = _ease(_clamp01((t - start) / 0.14))
    if t < start:
        return None
    sx0, sy0, sx1, sy1 = STRIP_SRC[idx]
    dx0, dy0, dx1, dy1 = STRIP_DST[idx]
    arc = math.sin(math.pi * u) * 5
    return (round(_lerp(sx0, dx0, u)), round(_lerp(sy0, dy0, u) - arc),
            round(_lerp(sx1, dx1, u)), round(_lerp(sy1, dy1, u) - arc))


def _draw_flight_card(img, p, rect, edge) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    d.rectangle((x0, y0, x1, y1), fill=m.h._glass_colour(p),
                outline=edge, width=2)
    d.line((x0, y0, x0, y1), fill=edge, width=4)
    for yy in (y0 + 7, y1 - 9):
        for run in m._dash_runs(x0 + 8, x1 - 8, 0):
            d.rectangle((run[0], yy, run[1], yy + 1), fill=p["accent2"])


def _draw_beam_pulse(img, p, rect, t: float) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    cy = y0 + 12
    left, right = x0 + 5, x1 - 5
    cx = (left + right) // 2
    glass = m.h._glass_colour(p)
    for start, end in ((0.66, 0.74), (0.76, 0.84)):
        if not (start <= t <= end):
            continue
        k = (t - start) / (end - start)
        u = _ease(k / 0.58) if k < 0.58 else 1.0
        fade = 1.0 if k < 0.78 else max(0.0, (1.0 - k) / 0.22)
        if fade <= 0:
            continue
        hl = round(_lerp(cx, left + 1, u))
        hr = round(_lerp(cx, right - 1, u))
        pulse = tuple(round(glass[i] * (1.0 - fade) + p["accent"][i] * fade)
                      for i in range(3))
        # yellow pulses travel toward the blue arrowheads, then fill them
        d.line((hl, cy, hl + 12, cy), fill=pulse, width=2)
        d.line((hr - 12, cy, hr, cy), fill=pulse, width=2)
        if u > 0.82:
            # fill the whole blue arrowhead and close the last shaft gap
            d.polygon([(left, cy), (left + 6, cy - 3), (left + 6, cy + 3)],
                      fill=pulse)
            d.polygon([(right, cy), (right - 6, cy - 3), (right - 6, cy + 3)],
                      fill=pulse)
            d.line((left, cy, hl + 12, cy), fill=pulse, width=2)
            d.line((hr - 12, cy, right, cy), fill=pulse, width=2)


def _base() -> Image.Image:
    p = m.PALETTES["reader"]
    img = m.badge_base("reader")
    m.draw_book(img, p)
    m.draw_task_label(img, p)
    return img


BASE = _base()


def frame(t: float) -> Image.Image:
    p = m.PALETTES["reader"]
    img = BASE.copy()
    d = ImageDraw.Draw(img)

    lifted = t >= 0.52
    selected = t >= 0.48

    for i, card in enumerate(CARDS):
        if lifted and i in (1, 2):
            continue
        edge = p[EDGES[i]]
        m.draw_chunk_card(img, p, card, seed=i,
                          selected=(selected and i in (1, 2)),
                          dim=(i == 0), edge=edge)
        progress = _fill(t, i)
        m.draw_score_bar(img, p, BARS[i], SCORES[i] * progress,
                         fill_color=p[FILLS[i]],
                         tip_color=p["accent"], tip_height=2)

    if lifted:
        for card in STRIP_SRC:
            d.rectangle(card, fill=p["light"])

    cy = _lens_y(t)
    status = _status(t)
    m.draw_lens(img, p, 77, cy, handle_len=16,
                card_lines=(status is None), status=status)

    if 0.52 <= t < 0.66:
        for i in range(2):
            rect = _lift(t, i)
            if rect is not None:
                _draw_flight_card(img, p, rect, p[STRIP_EDGE[i]])
                x0, y0, x1, y1 = rect
                d.line((x0 - 10, (y0 + y1) // 2, x0 - 3, (y0 + y1) // 2),
                       fill=p[STRIP_EDGE[i]], width=2)

    if t >= 0.66:
        for i, rect in enumerate(STRIP_DST):
            m.draw_context_card(img, p, rect, seed=i, edge=p[STRIP_EDGE[i]])
            m.draw_context_beam(img, p, rect)
            _draw_beam_pulse(img, p, rect, t)

    if t >= 0.84:
        u = _ease(_clamp01((t - 0.84) / 0.06))
        m.draw_answer(img, p, dy=round((1.0 - u) * 3))
    if t >= 0.86:
        m.draw_task_complete(img, p)

    return img.resize((m.PX, m.PX), Image.Resampling.NEAREST)


def main() -> int:
    frames = [frame(i / FRAMES) for i in range(FRAMES)]
    path = OUT / "reader-256.webp"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=DURATION, loop=0, quality=90, method=6)
    print(path, path.stat().st_size, "bytes", len(frames), "frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
