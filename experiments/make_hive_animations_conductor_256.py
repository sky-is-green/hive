#!/usr/bin/env python
"""Animate the 256px Hive Conductor hero.

    plan task -> dependency unlocks -> route A/B/C to workers -> fill capacity
    -> plan ready -> executing -> complete

The left column is the dependency chain; the centre is the conductor/baton;
the right column is worker capacity and the final plan contract.

    python experiments/make_hive_animations_conductor_256.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_hive_heroes_256 as m  # noqa: E402

OUT = m.OUT
FRAMES = 120
DURATION = 83  # ms, ~12 fps, 10 s loop

TASKS = (
    ((52, 66, 108, 86), "1"),
    ((32, 104, 88, 124), "2"),
    ((52, 142, 108, 162), "3"),
)
WORKERS = (
    ((156, 76, 216, 104), "WORKER 1"),
    ((156, 110, 216, 138), "WORKER 2"),
)
CORE = (124, 118)
RAIL_TASKS = ((108, 76), (88, 114), (108, 152))
RAIL_WORKERS = ((156, 90), (156, 90), (156, 124))


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _clamp01(t: float) -> float:
    return min(1.0, max(0.0, t))


# Each task gets a ready beat, a routed beat, then completes. PLAN READY
# holds for 2 s before the plan executes.
READY_WINDOWS = ((0.00, 0.04), (0.13, 0.17), (0.26, 0.30))
ROUTE_WINDOWS = ((0.04, 0.13), (0.17, 0.26), (0.30, 0.39))


def _task_state(t: float, idx: int) -> str:
    ready_start, active_start = READY_WINDOWS[idx]
    active_end = ROUTE_WINDOWS[idx][1]
    if t < ready_start:
        return "blocked"
    if t < active_start:
        return "ready"
    if t < active_end:
        return "active"
    return "done"


def _core_pulse(t: float) -> float:
    for start, end in ROUTE_WINDOWS:
        if start <= t <= end:
            u = (t - start) / (end - start)
            return math.sin(math.pi * u)
    return 0.0


def _lerp_pt(a, b, u: float):
    return (round(_lerp(a[0], b[0], u)), round(_lerp(a[1], b[1], u)))


def _path_point(start, mid, end, progress: float):
    seg1 = math.dist(start, mid)
    seg2 = math.dist(mid, end)
    dist = (seg1 + seg2) * progress
    if dist <= seg1:
        u = dist / seg1 if seg1 else 0
        return (round(_lerp(start[0], mid[0], u)),
                round(_lerp(start[1], mid[1], u)))
    u = (dist - seg1) / seg2 if seg2 else 1
    return (round(_lerp(mid[0], end[0], u)),
            round(_lerp(mid[1], end[1], u)))


def _packet_offset(idx: int, progress: float):
    x, y = _path_point(RAIL_TASKS[idx], CORE, RAIL_WORKERS[idx], progress)
    return (round((x - CORE[0]) * 0.25), round((y - CORE[1]) * 0.25))


def _pupil(t: float):
    """The eye looks at the task, follows its packet, and lands on the worker."""
    center = (0, 0)
    t1 = _packet_offset(0, 0)
    w1 = _packet_offset(0, 1)
    t2 = _packet_offset(1, 0)
    w2 = _packet_offset(1, 1)
    t3 = _packet_offset(2, 0)
    w3 = _packet_offset(2, 1)
    if t < 0.02:
        return center
    if t < 0.04:
        return _lerp_pt(center, t1, _ease((t - 0.02) / 0.02))
    if t < 0.13:
        return _packet_offset(0, _ease((t - 0.04) / 0.09))
    if t < 0.15:
        return _lerp_pt(w1, t2, _ease((t - 0.13) / 0.02))
    if t < 0.17:
        return t2
    if t < 0.26:
        return _packet_offset(1, _ease((t - 0.17) / 0.09))
    if t < 0.28:
        return _lerp_pt(w2, t3, _ease((t - 0.26) / 0.02))
    if t < 0.30:
        return t3
    if t < 0.39:
        return _packet_offset(2, _ease((t - 0.30) / 0.09))
    if t < 0.41:
        return _lerp_pt(w3, center, _ease((t - 0.39) / 0.02))
    return center


def _worker_tasks(t: float):
    w1 = []
    if t >= ROUTE_WINDOWS[0][1]:
        w1.append("1")
    if t >= ROUTE_WINDOWS[1][1]:
        w1.append("2")
    w2 = ["3"] if t >= ROUTE_WINDOWS[2][1] else []
    return w1, w2



def _rail_idx(t: float):
    for idx, (ready_start, _) in enumerate(READY_WINDOWS):
        if ready_start <= t <= ROUTE_WINDOWS[idx][1]:
            return idx
    return None


def _rail_progress(t: float, idx: int):
    ready_start, active_start = READY_WINDOWS[idx]
    if t < ready_start or t > ROUTE_WINDOWS[idx][1]:
        return None
    if t < active_start:
        return _ease((t - ready_start) / (active_start - ready_start))
    return 1.0


def _draw_rail(img, p, idx: int, progress: float) -> None:
    start = RAIL_TASKS[idx]
    mid = CORE
    end = RAIL_WORKERS[idx]
    seg1 = math.dist(start, mid)
    seg2 = math.dist(mid, end)
    dist = (seg1 + seg2) * progress
    d = ImageDraw.Draw(img)
    if dist <= seg1:
        u = dist / seg1 if seg1 else 0
        point = (round(_lerp(start[0], mid[0], u)),
                 round(_lerp(start[1], mid[1], u)))
        d.line((start[0], start[1], point[0], point[1]),
               fill=p["accent"], width=2)
        return
    d.line((start[0], start[1], mid[0], mid[1]), fill=p["accent"], width=2)
    u = (dist - seg1) / seg2 if seg2 else 1
    point = (round(_lerp(mid[0], end[0], u)),
             round(_lerp(mid[1], end[1], u)))
    d.line((mid[0], mid[1], point[0], point[1]), fill=p["accent"], width=2)


def _route_pulse(img, p, t: float, idx: int) -> None:
    start, end = ROUTE_WINDOWS[idx]
    if not (start <= t <= end):
        return
    u = _ease((t - start) / (end - start))
    x, y = _path_point(RAIL_TASKS[idx], CORE, RAIL_WORKERS[idx], u)
    d = ImageDraw.Draw(img)
    # dark rim keeps the honey packet readable over the honey core
    d.rectangle((x - 4, y - 4, x + 4, y + 4), fill=p["dark"])
    d.rectangle((x - 3, y - 3, x + 3, y + 3), fill=p["accent"])
    d.point((x, y), fill=p["light"])


def _base() -> Image.Image:
    p = m.PALETTES["conductor"]
    img = m.badge_base("conductor")
    m.draw_centered_label(img, p, m.PLAN_TASK)
    return img


BASE = _base()


def frame(t: float) -> Image.Image:
    p = m.PALETTES["conductor"]
    img = BASE.copy()

    for i, (rect, letter) in enumerate(TASKS):
        done_color = p["accent"] if letter == "3" else p["high"]
        m.draw_task_card(img, p, rect, letter, _task_state(t, i),
                         done_color=done_color)

    # dependency 1 -> 2 -> 3
    m.draw_dep_arrow(img, p, (80, 86), (60, 104),
                     "done" if t >= ROUTE_WINDOWS[0][1] else "blocked")
    m.draw_dep_arrow(img, p, (60, 124), (80, 142),
                     "done" if t >= ROUTE_WINDOWS[1][1] else "blocked")

    rail_idx = _rail_idx(t)
    if rail_idx is not None:
        progress = _rail_progress(t, rail_idx)
        if progress:
            _draw_rail(img, p, rail_idx, progress)

    m.draw_conductor_core(img, p, pulse=_core_pulse(t), pupil=_pupil(t))

    w1, w2 = _worker_tasks(t)
    m.draw_worker_card(img, p, WORKERS[0][0], WORKERS[0][1], w1,
                       slot_colors=[p["high"]] * len(w1))
    m.draw_worker_card(img, p, WORKERS[1][0], WORKERS[1][1], w2,
                       slot_colors=[p["accent"]] * len(w2))

    for idx in range(3):
        _route_pulse(img, p, t, idx)

    if t >= 0.90:
        m.draw_plan_card(img, p, mode="complete")
    elif t >= 0.60:
        m.draw_plan_card(img, p, mode="executing",
                         spin_phase=(t - 0.60) * 60.0)
    elif t >= 0.40:
        m.draw_plan_card(img, p, mode="ready")
    return img.resize((m.PX, m.PX), Image.Resampling.NEAREST)


def main() -> int:
    frames = [frame(i / FRAMES) for i in range(FRAMES)]
    path = OUT / "conductor-256.webp"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=DURATION, loop=0, quality=90, method=6)
    print(path, path.stat().st_size, "bytes", len(frames), "frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
