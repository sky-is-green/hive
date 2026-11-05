#!/usr/bin/env python
"""Animate the 256px Hive Dispatcher hero.

    jobs placed on workers -> deadline scan -> makespan 11, on time

The schedule is one representative dispatcher use case: BUILD -> TEST ->
RELEASE on Worker 1, WRITE on Worker 2, and CHECK on Worker 3. TEST, WRITE
and CHECK all finish before RELEASE. The deadline is 12 minutes; the plan
finishes at 12.

    python experiments/make_hive_animations_dispatcher_256.py
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

APPEAR = {
    "BUILD": 0.06,
    "WRITE": 0.12,
    "TEST": 0.18,
    "CHECK": 0.24,
    "RELEASE": 0.30,
}
SCAN_START = 0.38
SCAN_END = 0.78


def _base() -> Image.Image:
    p = m.PALETTES["dispatcher"]
    img = m.badge_base("dispatcher")
    m.draw_centered_label(img, p, m.DISPATCH_TASK)
    m.draw_dispatch_grid(img, p)
    return img


BASE = _base()


def frame(t: float) -> Image.Image:
    p = m.PALETTES["dispatcher"]
    img = BASE.copy()
    names: set[str] = set()

    for name, start, end, lane in m.DISPATCH_JOBS:
        if t < APPEAR[name]:
            continue
        names.add(name)
        m.draw_dispatch_job(img, p, name, start, end, lane)
        if t - APPEAR[name] < 0.025:
            x0, x1 = m.dispatch_x(start), m.dispatch_x(end) - 1
            y0, y1 = m.DISPATCH_LANES[lane]
            ImageDraw.Draw(img).rectangle(
                (x0 + 1, y0 + 4, x1 - 1, y1 - 4), outline=p["light"],
                width=1)

    m.draw_dispatch_deps(img, p, names)

    if SCAN_START <= t < SCAN_END:
        u = (t - SCAN_START) / (SCAN_END - SCAN_START)
        x = m.dispatch_x(min(u, 1.0) * m.DISPATCH_MAKESPAN)
        d = ImageDraw.Draw(img)
        d.line((x, 60, x, m.DISPATCH_LANES[2][1]), fill=p["accent2"],
               width=2)
        d.polygon([(x - 3, 60), (x + 3, 60), (x, 65)],
                  fill=p["accent2"])

    done = t >= SCAN_END
    if done:
        status, since = "ON TIME", SCAN_END
    elif t >= SCAN_START:
        status, since = "LOADING", SCAN_START
    else:
        status, since = "PLAN", 0.0
    status_colour = (p["light"] if t - since < 0.025
                     else p["accent"])
    m.draw_dispatch_summary(
        img, p,
        makespan=m.DISPATCH_MAKESPAN if done else None,
        on_time=done,
        status=status,
        status_colour=status_colour)
    if not done:
        m.draw_spinner(img, p, (96, 161, 112, 171),
                       phase=(t - APPEAR["BUILD"]) * 60.0, dot=2, spread=2)

    return img.resize((m.PX, m.PX), Image.Resampling.NEAREST)


def main() -> int:
    frames = [frame(i / FRAMES) for i in range(FRAMES)]
    path = OUT / "dispatcher-256.webp"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=DURATION, loop=0, quality=90, method=6)
    print(path, path.stat().st_size, "bytes", len(frames), "frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
