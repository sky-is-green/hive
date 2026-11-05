#!/usr/bin/env python
"""Animate the 256px Hive Auditor hero.

    place M1 on C1 -> place M2 on C1 -> M3 overloads C1 and is rejected
    -> M3 moves to C2 -> PASS

The left column is the model queue; the centre is the inspection lens; the
right column is card capacity and the final verdict.

    python experiments/make_hive_animations_auditor_256.py
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

MODEL_ANCHORS = ((92, 90), (92, 122), (92, 154))
CARD_ANCHORS = ((156, 92), (156, 128))
LENS = m.LENS_CENTER

M1_PROPOSE = (0.04, 0.10)
M1_PLACE = (0.10, 0.16)
M2_PROPOSE = (0.20, 0.26)
M2_PLACE = (0.26, 0.32)
M3_BAD_PROPOSE = (0.36, 0.40)
M3_BAD_END = 0.46
M3_GOOD_PROPOSE = (0.54, 0.58)
M3_GOOD_PLACE = (0.58, 0.66)
VERDICT = 0.72


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _clamp01(t: float) -> float:
    return min(1.0, max(0.0, t))


def _lerp_pt(a, b, u: float):
    return (round(_lerp(a[0], b[0], u)), round(_lerp(a[1], b[1], u)))


def _model_state(t: float, idx: int) -> str:
    if idx == 0:
        if t < 0.04:
            return "queued"
        if t < 0.10:
            return "scan"
        if t < 0.16:
            return "placing"
        return "placed"
    if idx == 1:
        if t < 0.20:
            return "queued"
        if t < 0.26:
            return "scan"
        if t < 0.32:
            return "placing"
        return "placed"
    if t < 0.36:
        return "queued"
    if t < 0.40:
        return "scan"
    if t < 0.52:
        return "rejected"
    if t < 0.58:
        return "scan"
    if t < 0.66:
        return "placing"
    return "placed"


def _packet_offset(model_idx: int, card_idx: int, progress: float):
    x, y = _path_point(MODEL_ANCHORS[model_idx], LENS,
                       CARD_ANCHORS[card_idx], progress)
    return (round((x - LENS[0]) * 0.25), round((y - LENS[1]) * 0.25))


def _lens_focus(t: float):
    """The lens follows the packet, arriving as the result lands."""
    center = (0, 0)
    m1 = _packet_offset(0, 0, 0)
    m2 = _packet_offset(1, 0, 0)
    m3 = _packet_offset(2, 0, 0)
    c1 = _packet_offset(0, 0, 1)
    c2 = _packet_offset(2, 1, 1)
    if t < 0.02:
        return center
    if t < 0.04:
        return _lerp_pt(center, m1, _ease((t - 0.02) / 0.02))
    if t < 0.10:
        return m1
    if t < 0.16:
        return _packet_offset(0, 0, _ease((t - 0.10) / 0.06))
    if t < 0.17:
        return c1
    if t < 0.20:
        return _lerp_pt(c1, m2, _ease((t - 0.17) / 0.03))
    if t < 0.26:
        return m2
    if t < 0.32:
        return _packet_offset(1, 0, _ease((t - 0.26) / 0.06))
    if t < 0.36:
        return _lerp_pt(c1, m3, _ease((t - 0.32) / 0.04))
    if t < 0.58:
        return m3
    if t < 0.66:
        return _packet_offset(2, 1, _ease((t - 0.58) / 0.08))
    if t < 0.68:
        return c2
    if t < 0.72:
        return _lerp_pt(c2, center, _ease((t - 0.68) / 0.04))
    return center


def _lens_ring(t: float):
    if (0.04 <= t < 0.16 or 0.20 <= t < 0.32 or
            0.36 <= t < 0.40 or 0.54 <= t < 0.66):
        return m.PALETTES["auditor"]["accent2"]
    if 0.40 <= t < 0.52:
        return m.PALETTES["auditor"]["low"]
    return None


def _rack_used(t: float):
    c1 = 0.0
    if t >= M1_PLACE[0]:
        c1 = _lerp(0, 12, _clamp01((t - M1_PLACE[0]) /
                                   (M1_PLACE[1] - M1_PLACE[0])))
    if t >= M2_PLACE[0]:
        c1 = _lerp(12, 16, _clamp01((t - M2_PLACE[0]) /
                                    (M2_PLACE[1] - M2_PLACE[0])))
    if M3_BAD_PROPOSE[1] <= t < 0.52:
        c1 = 26.0
    elif t >= 0.52:
        c1 = 16.0
    c2 = 0.0
    if t >= M3_GOOD_PLACE[0]:
        c2 = _lerp(0, 10, _clamp01((t - M3_GOOD_PLACE[0]) /
                                   (M3_GOOD_PLACE[1] - M3_GOOD_PLACE[0])))
    return c1, c2


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


def _draw_rail(img, p, start, end, progress: float, color) -> None:
    mid = LENS
    seg1 = math.dist(start, mid)
    seg2 = math.dist(mid, end)
    dist = (seg1 + seg2) * progress
    d = ImageDraw.Draw(img)
    if dist <= seg1:
        u = dist / seg1 if seg1 else 0
        point = (round(_lerp(start[0], mid[0], u)),
                 round(_lerp(start[1], mid[1], u)))
        d.line((start[0], start[1], point[0], point[1]), fill=color,
               width=2)
        return
    d.line((start[0], start[1], mid[0], mid[1]), fill=color, width=2)
    u = (dist - seg1) / seg2 if seg2 else 1
    point = (round(_lerp(mid[0], end[0], u)),
             round(_lerp(mid[1], end[1], u)))
    d.line((mid[0], mid[1], point[0], point[1]), fill=color, width=2)


def _draw_packet(img, p, x: int, y: int, color) -> None:
    d = ImageDraw.Draw(img)
    d.rectangle((x - 5, y - 5, x + 5, y + 5), fill=p["dark"])
    d.rectangle((x - 4, y - 4, x + 4, y + 4), fill=color)
    d.point((x, y), fill=p["light"])


def _rail(t: float):
    if M1_PROPOSE[0] <= t <= M1_PLACE[1]:
        if t < M1_PROPOSE[1]:
            u = _ease((t - M1_PROPOSE[0]) /
                      (M1_PROPOSE[1] - M1_PROPOSE[0]))
        else:
            u = 1.0
        return 0, 0, u, m.PALETTES["auditor"]["accent"]
    if M2_PROPOSE[0] <= t <= M2_PLACE[1]:
        if t < M2_PROPOSE[1]:
            u = _ease((t - M2_PROPOSE[0]) /
                      (M2_PROPOSE[1] - M2_PROPOSE[0]))
        else:
            u = 1.0
        return 1, 0, u, m.PALETTES["auditor"]["accent"]
    if M3_BAD_PROPOSE[0] <= t <= M3_BAD_END:
        if t < M3_BAD_PROPOSE[1]:
            u = _ease((t - M3_BAD_PROPOSE[0]) /
                      (M3_BAD_PROPOSE[1] - M3_BAD_PROPOSE[0]))
        else:
            u = 1.0
        color = (m.PALETTES["auditor"]["low"]
                 if t >= M3_BAD_PROPOSE[1]
                 else m.PALETTES["auditor"]["accent"])
        return 2, 0, u, color
    if M3_GOOD_PROPOSE[0] <= t <= M3_GOOD_PLACE[1]:
        if t < M3_GOOD_PROPOSE[1]:
            u = _ease((t - M3_GOOD_PROPOSE[0]) /
                      (M3_GOOD_PROPOSE[1] - M3_GOOD_PROPOSE[0]))
        else:
            u = 1.0
        return 2, 1, u, m.PALETTES["auditor"]["accent"]
    return None


def _packet(t: float):
    if M1_PLACE[0] <= t <= M1_PLACE[1]:
        u = _ease((t - M1_PLACE[0]) / (M1_PLACE[1] - M1_PLACE[0]))
        return 0, 0, u, m.PALETTES["auditor"]["copper"]
    if M2_PLACE[0] <= t <= M2_PLACE[1]:
        u = _ease((t - M2_PLACE[0]) / (M2_PLACE[1] - M2_PLACE[0]))
        return 1, 0, u, m.PALETTES["auditor"]["copper"]
    if M3_GOOD_PLACE[0] <= t <= M3_GOOD_PLACE[1]:
        u = _ease((t - M3_GOOD_PLACE[0]) /
                  (M3_GOOD_PLACE[1] - M3_GOOD_PLACE[0]))
        return 2, 1, u, m.PALETTES["auditor"]["copper"]
    return None


def _base() -> Image.Image:
    p = m.PALETTES["auditor"]
    img = m.badge_base("auditor")
    m.draw_centered_label(img, p, m.AUDIT_TASK)
    return img


BASE = _base()


def frame(t: float) -> Image.Image:
    p = m.PALETTES["auditor"]
    img = BASE.copy()

    for i, (rect, name, size) in enumerate(m.MODELS):
        m.draw_model_card(img, p, rect, name, size, _model_state(t, i))

    rail = _rail(t)
    if rail is not None:
        model_idx, card_idx, progress, color = rail
        _draw_rail(img, p, MODEL_ANCHORS[model_idx],
                   CARD_ANCHORS[card_idx], progress, color)

    m.draw_auditor_lens(img, p, scan=_lens_focus(t), ring=_lens_ring(t))

    c1_used, c2_used = _rack_used(t)
    c1_chips = []
    if t >= M1_PLACE[1]:
        c1_chips.append("M1")
    if t >= M2_PLACE[1]:
        c1_chips.append("M2")
    c2_chips = ["M3"] if t >= M3_GOOD_PLACE[1] else []
    c1_over = M3_BAD_PROPOSE[1] <= t < 0.52
    m.draw_rack(img, p, m.RACKS[0][0], "C1", "18G", c1_used, c1_chips,
                over=c1_over)
    m.draw_rack(img, p, m.RACKS[1][0], "C2", "14G", c2_used, c2_chips)

    packet = _packet(t)
    if packet is not None:
        model_idx, card_idx, progress, color = packet
        x, y = _path_point(MODEL_ANCHORS[model_idx], LENS,
                           CARD_ANCHORS[card_idx], progress)
        _draw_packet(img, p, x, y, color)

    if t >= VERDICT:
        m.draw_verdict(img, p)

    return img.resize((m.PX, m.PX), Image.Resampling.NEAREST)


def main() -> int:
    frames = [frame(i / FRAMES) for i in range(FRAMES)]
    path = OUT / "auditor-256.webp"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=DURATION, loop=0, quality=90, method=6)
    print(path, path.stat().st_size, "bytes", len(frames), "frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
