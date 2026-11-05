#!/usr/bin/env python
"""Animate the 256px Hive Tool Manager hero.

    tool task -> select READ -> select EDIT -> select TEST -> tools ready

The rack is the agency surface; the cursor is the tool manager choosing and
ordering the calls; the bottom card records the sequence.

    python experiments/make_hive_animations_toolmanager_256.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_hive_heroes_256 as m  # noqa: E402

OUT = m.OUT
FRAMES = 120
DURATION = 83  # ms, ~12 fps, 10 s loop

START = (56, 70)
READ = (98, 82)
EDIT = (140, 82)
TEST = (98, 118)


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _lerp_pt(a, b, u: float):
    return (round(_lerp(a[0], b[0], u)), round(_lerp(a[1], b[1], u)))


def _tile_state(t: float, name: str) -> str:
    if name == "READ":
        if t < 0.10:
            return "idle"
        if t < 0.22:
            return "active"
        return "selected"
    if name == "EDIT":
        if t < 0.26:
            return "idle"
        if t < 0.40:
            return "active"
        return "selected"
    if name == "TEST":
        if t < 0.44:
            return "idle"
        if t < 0.60:
            return "active"
        return "selected"
    return "idle"


def _selected(t: float):
    chosen = set()
    if t >= 0.22:
        chosen.add("READ")
    if t >= 0.40:
        chosen.add("EDIT")
    if t >= 0.60:
        chosen.add("TEST")
    return chosen


def _cursor(t: float):
    if t < 0.06:
        return START
    if t < 0.10:
        return _lerp_pt(START, READ, _ease((t - 0.06) / 0.04))
    if t < 0.22:
        return READ
    if t < 0.26:
        return _lerp_pt(READ, EDIT, _ease((t - 0.22) / 0.04))
    if t < 0.40:
        return EDIT
    if t < 0.415:
        return _lerp_pt(EDIT, (140, 100), _ease((t - 0.40) / 0.015))
    if t < 0.43:
        return _lerp_pt((140, 100), (98, 100),
                        _ease((t - 0.415) / 0.015))
    if t < 0.44:
        return _lerp_pt((98, 100), TEST, _ease((t - 0.43) / 0.01))
    return TEST


def _base() -> Image.Image:
    p = m.PALETTES["toolmanager"]
    img = m.badge_base("toolmanager")
    m.draw_centered_label(img, p, m.TOOL_TASK)
    return img


BASE = _base()


def frame(t: float) -> Image.Image:
    p = m.PALETTES["toolmanager"]
    img = BASE.copy()

    for rect, name, kind in m.TOOL_TILES:
        m.draw_tool_tile(img, p, rect, name, kind, _tile_state(t, name))

    chosen = _selected(t)
    m.draw_tool_pipeline(img, p, chosen, done=(t >= 0.68))

    x, y = _cursor(t)
    m.draw_tool_cursor(img, p, x, y)
    return img.resize((m.PX, m.PX), Image.Resampling.NEAREST)


def main() -> int:
    frames = [frame(i / FRAMES) for i in range(FRAMES)]
    path = OUT / "toolmanager-256.webp"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=DURATION, loop=0, quality=90, method=6)
    print(path, path.stat().st_size, "bytes", len(frames), "frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
