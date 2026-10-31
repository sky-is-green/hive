#!/usr/bin/env python3
"""Animate the approved Scion pixel heroes as WebP loops.

The approved raster art is preserved: the base scene is re-rendered without
leaves, and each leaf is a separate sprite. The shoot sways as one piece while
the leaves add a smaller individual sway; the grass and dirt are separate
layers. A burst starts at the leaves, travels down the stem and through the
roots, returns faintly, then the drafter fires a gold token burst in the gap.
The loop is one burst per 8 seconds.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_model_heroes as m  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "assets" / "heroes"
OUT.mkdir(parents=True, exist_ok=True)
S = 8
SIZE = 96 * S
FRAMES = 96
DURATION = 83  # ms, ~12 fps, 8 s loop: one burst, then a long idle
INSET = [(29, 9), (67, 9), (88, 48), (72, 86), (24, 86), (8, 48)]  # inside the ring


def scene_data(name):
    p = m.PALETTES[name]
    big = name == "scion176"
    drafter = name == "scion_drafter"
    top = 10 if big else (9 if drafter else 13)
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
        leaves = [(50, 9, 60, 14), (36, 11, 46, 16),
                  (51, 16, 62, 21), (34, 19, 45, 24),
                  (50, 23, 61, 28), (33, 25, 43, 30)]
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
        leaves = [(50, 14, 60, 19), (36, 20, 46, 25),
                  (50, 25, 61, 30)]
    return p, big, drafter, top, roots, leaves


SOIL_TOP = [(8, 49), (20, 46), (30, 44), (42, 44), (54, 44),
            (66, 44), (76, 46), (88, 49)]


def soil_top(x):
    for (x0, y0), (x1, y1) in zip(SOIL_TOP, SOIL_TOP[1:]):
        if x0 <= x <= x1:
            f = (x - x0) / (x1 - x0)
            return y0 + (y1 - y0) * f
    return SOIL_TOP[-1][1]


def draw_soil(d, p):
    """The dirt reaches the inner border; the roots and caption sit on it."""
    root, root_dark = p["root"], p["root_dark"]
    soil = p["soil"]
    d.polygon([(6, 61), (6, 49), (20, 46), (30, 44), (42, 44),
               (54, 44), (66, 44), (76, 46), (90, 49), (90, 61)],
              fill=soil)
    d.line((6, 49, 20, 46, 30, 44, 42, 44, 54, 44, 66, 44,
            76, 46, 90, 49), fill=root_dark, width=1)
    # loosely scattered clods and specks
    for x, y in [(12, 57), (17, 52), (23, 58), (29, 55), (35, 60),
                 (41, 54), (47, 58), (53, 52), (59, 57), (64, 60),
                 (69, 54), (75, 58), (81, 52), (85, 57)]:
        d.point((x, y), fill=root_dark)
    for x, y in [(15, 55), (26, 59), (38, 57), (50, 55), (62, 59),
                 (73, 55), (83, 60)]:
        d.point((x, y), fill=root)
    d.rectangle((21, 56, 22, 56), fill=root_dark)
    d.rectangle((57, 53, 58, 53), fill=root_dark)
    d.rectangle((78, 60, 79, 60), fill=root)
    # small disturbed patches around the four grass blades
    for x, y in [(12, 56), (17, 56), (19, 53), (30, 49), (34, 49),
                 (29, 45), (59, 51), (63, 51), (60, 47), (79, 58),
                 (83, 58), (77, 54)]:
        d.point((x, y), fill=root_dark)
    for x, y in [(13, 52), (31, 47), (62, 52), (80, 54)]:
        d.point((x, y), fill=root)
    d.rectangle((16, 55, 17, 55), fill=root_dark)
    d.rectangle((33, 48, 34, 48), fill=root)
    d.rectangle((62, 50, 63, 50), fill=root_dark)
    d.rectangle((80, 57, 81, 57), fill=root)


def draw_base_scene(d, p, big, drafter, top, roots):
    """The approved scene, minus the leaf blades (petioles stay)."""
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    root, root_dark = p["root"], p["root_dark"]
    band = p["band"]

    for pts, width in roots:
        segs = len(pts) - 1
        for i in range(segs):
            w = max(1, round(width * (segs - i) / segs))
            d.line((pts[i], pts[i + 1]), fill=root_dark, width=w + 2)
        for i in range(segs):
            w = max(1, round(width * (segs - i) / segs))
            d.line((pts[i], pts[i + 1]), fill=root, width=w)

    d.polygon([(42, 49), (44, 33), (52, 33), (54, 49)], fill=dark)
    d.polygon([(44, 48), (45, 35), (51, 35), (52, 48)], fill=root)
    d.line((46, 37, 46, 40), fill=root_dark, width=1)
    d.line((50, 42, 50, 46), fill=root_dark, width=1)
    d.point((47, 44), fill=root_dark)
    d.line((43, 40, 39, 37), fill=dark, width=3)
    d.line((43, 40, 39, 37), fill=root, width=1)

    d.rectangle((41, 31, 55, 35), fill=dark)
    d.rectangle((42, 32, 54, 34), fill=band)
    d.line((42, 32, 54, 32), fill=light, width=1)
    d.line((42, 34, 54, 34), fill=root_dark, width=1)
    for x in (44, 47, 50, 53):
        d.point((x, 33), fill=dark)


def render_base(name):
    p, big, drafter, top, roots, _ = scene_data(name)
    accent = p["accent"]
    light, dark = p["light"], p["dark"]
    img = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    frame = ImageDraw.Draw(img)
    frame.polygon(m.hex_points(0), fill=(*dark, 255))
    frame.polygon(m.hex_points(2), fill=(*accent, 255))
    frame.polygon(m.hex_points(4), fill=(*p["bg"], 255))

    # dirt reaches the inner edge of the ring; the rest stays inset
    soil_layer = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    draw_soil(ImageDraw.Draw(soil_layer), p)
    soil_clip = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(soil_clip).polygon(m.hex_points(4), fill=255)
    soil_layer.putalpha(ImageChops.multiply(soil_layer.getchannel("A"), soil_clip))
    img.alpha_composite(soil_layer)

    art = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    d = ImageDraw.Draw(art)
    draw_base_scene(d, p, big, drafter, top, roots)
    if drafter:
        banner = [(14, 61), (82, 61), (70, 86), (26, 86)]
        d.polygon(banner, fill=(*dark, 245))
        d.line((14, 61, 82, 61), fill=accent, width=1)
        m.text(d, "SCION", 63, accent)
        m.text(d, "35B-A3B", 72, light)
        m.mini_text(d, "DRAFTER", 81, p["gold"])
    else:
        banner = [(14, 61), (82, 61), (70, 86), (26, 86)]
        d.polygon(banner, fill=(*dark, 245))
        d.line((14, 61, 82, 61), fill=accent, width=1)
        m.text(d, "SCION", 64, accent)
        m.text(d, "35B-A3B" if not big else "176B-A6B", 75, light)

    # Everything except the frame stays inside the accent ring.
    clip = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(clip).polygon(INSET, fill=255)
    art.putalpha(ImageChops.multiply(art.getchannel("A"), clip))
    img.alpha_composite(art)
    return img.resize((SIZE, SIZE), Image.Resampling.NEAREST)


def leaf_sprite(name, box, gold=False):
    p, big, drafter, _, _, _ = scene_data(name)
    img = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cy = (box[1] + box[3]) // 2
    m.leaf(d, box, p, pointing_right=box[0] > 48,
           tip_dy=-1 if cy <= 20 else 1,
           blade=p["gold"] if gold else None,
           ink=p["gold_dark"] if gold else None)
    return img.resize((SIZE, SIZE), Image.Resampling.NEAREST)


def leaf_meta(name):
    p, big, drafter, top, roots, leaves = scene_data(name)
    out = []
    for i, box in enumerate(leaves):
        cy = (box[1] + box[3]) // 2
        base = (box[0] if box[0] > 48 else box[2], cy)
        sprite = leaf_sprite(name, box, False)
        out.append({
            "box": box, "gold": False, "base": base,
            "phase": i * 0.13, "angle": 2.0,
            "sprite": sprite, "rim": make_rim(sprite, p["accent"], 9),
            "color": p["accent"],
        })
    if drafter:
        sprite = leaf_sprite(name, (34, 9, 44, 14), True)
        out.append({
            "box": (34, 9, 44, 14), "gold": True, "base": (44, 11),
            "phase": 0.1, "angle": 3.0,
            "sprite": sprite, "rim": make_rim(sprite, p["gold"], 9),
            "color": p["gold"],
        })
    return out


def rotate_about(img, center, angle):
    return img.rotate(angle, resample=Image.Resampling.NEAREST,
                      center=center, expand=False)


def make_rim(sprite, color, radius=11):
    alpha = sprite.getchannel("A")
    dilated = alpha.filter(ImageFilter.MaxFilter(radius))
    rim = ImageChops.subtract(dilated, alpha)
    out = Image.new("RGBA", sprite.size, (*color, 0))
    out.putalpha(rim)
    return out


def body_mask():
    mask = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(mask)
    d.polygon([(x * S, y * S) for x, y in INSET], fill=255)
    return mask


def branch_angle(t):
    return 0.8 * math.sin(2 * math.pi * (t + 0.08))


def rot_point(x, y, angle, pivot=(48, 31)):
    a = math.radians(angle)
    px, py = pivot
    dx, dy = x - px, y - py
    return (px + math.cos(a) * dx + math.sin(a) * dy,
            py - math.sin(a) * dx + math.cos(a) * dy)


def stem_point(y, angle):
    if y >= 31:
        return 48.0, y
    return rot_point(48, y, angle)


def grass_meta(name):
    p, big, drafter, top, roots, leaves = scene_data(name)
    # four blades only, spread across open dirt and clear of the roots
    blades = [(15, 54, 4, 1, 0.13), (32, 47, 3, -1, 0.67),
              (61, 49, 4, 1, 0.31), (81, 56, 3, -1, 0.88)]
    out = []
    for x, y, h, lean, phase in blades:
        img = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.line((x, y, x + lean, y - 2), fill=p["accent2"], width=1)
        d.line((x + lean, y - 2, x + 2 * lean, y - h),
               fill=p["accent2"], width=1)
        out.append((img.resize((SIZE, SIZE), Image.Resampling.NEAREST),
                    x, y, phase))
    return out


def build_shoot(name, t, ta, leaves_meta):
    p, big, drafter, top, roots, leaves = scene_data(name)
    a, a2, light, dark = p["accent"], p["accent2"], p["light"], p["dark"]
    shoot = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(shoot)

    # branch stem and new-growth tip
    d.line((48 * S, top * S, 48 * S, 31 * S), fill=dark, width=3 * S)
    d.line((48 * S, (top + 1) * S, 48 * S, 31 * S), fill=a, width=S)
    d.rectangle((48 * S, top * S, 48 * S + S - 1, top * S + S - 1), fill=light)
    d.rectangle((47 * S, (top + 1) * S,
                 47 * S + S - 1, (top + 1) * S + S - 1), fill=light)

    # petioles are part of the branch so they sway with it
    for box in (lm["box"] for lm in leaves_meta):
        x0, y0, x1, y1 = box
        cy = (y0 + y1) // 2
        if x0 > 48:
            d.line((48 * S, cy * S, x0 * S, cy * S), fill=a2, width=S)
        else:
            d.line((x1 * S, cy * S, 48 * S, cy * S), fill=a2, width=S)
    if drafter:
        d.line((44 * S, 11 * S, 48 * S, 11 * S), fill=p["gold_dark"], width=S)

    n_leaves = len(leaves_meta)
    for i, lm in enumerate(leaves_meta):
        center = (lm["base"][0] * S, lm["base"][1] * S)
        angle = lm["angle"] * math.sin(2 * math.pi * (t + lm["phase"]))
        sprite = rotate_about(lm["sprite"], center, angle)
        shoot.alpha_composite(sprite)

        if lm["gold"]:
            pulse = max(0.0, 1.0 - abs(ta - 0.90) / 0.04)
        else:
            t_leaf = 0.05 + 0.13 * (i / max(1, n_leaves - 1))
            pulse = max(0.0, 1.0 - abs(ta - t_leaf) / 0.05)
            ret = max(0.0, 1.0 - abs(ta - 0.84) / 0.03) * 0.22
            pulse = max(pulse, ret)
        if pulse > 0.01:
            rim = rotate_about(lm["rim"], center, angle)
            rim.putalpha(rim.getchannel("A").point(
                lambda v: int(v * min(1.0, pulse * 0.75))))
            shoot.alpha_composite(rim)
            tint = Image.new("RGBA", (SIZE, SIZE), (*lm["color"], 0))
            tint.putalpha(sprite.getchannel("A").point(
                lambda v: int(v * pulse * 0.16)))
            shoot.alpha_composite(tint)

    return shoot.rotate(branch_angle(t), resample=Image.Resampling.NEAREST,
                        center=(48 * S, 31 * S), expand=False)


def point_along(points, u):
    """Point at normalized distance u along a polyline (pixel coords)."""
    segs = []
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        d = math.hypot(x2 - x1, y2 - y1)
        segs.append((total, total + d, (x1, y1), (x2, y2)))
        total += d
    target = u * total
    for a, b, (x1, y1), (x2, y2) in segs:
        if a <= target <= b or b == total:
            f = 0 if b == a else (target - a) / (b - a)
            return x1 + (x2 - x1) * f, y1 + (y2 - y1) * f
    return points[-1]


def draw_dot(d, x, y, color, alpha, r=5, glow=2.4):
    if alpha <= 0:
        return
    ga = int(alpha * 0.28)
    d.ellipse((x - r * glow, y - r * glow, x + r * glow, y + r * glow),
              fill=(*color, ga))
    d.ellipse((x - r, y - r, x + r, y + r), fill=(*color, int(alpha)))


def draw_pip(d, x, y, color, light, alpha):
    if alpha <= 0:
        return
    r = 10
    d.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)],
              fill=(*color, int(alpha)))
    r2 = 5
    d.polygon([(x, y - r2), (x + r2, y), (x, y + r2), (x - r2, y)],
              fill=(*light, int(alpha)))


def draw_gold_link(anim, p, t, active, pts):
    segs = []
    total = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        d = math.hypot(x2 - x1, y2 - y1)
        segs.append((total, total + d, x1, y1, x2, y2))
        total += d
    d = ImageDraw.Draw(anim)
    if not active:
        # the graft is still visible as a faint connection
        d.line(pts, fill=(*p["gold"], 45), width=4)
        return
    offset = (t * 90) % 22
    pos = -22 + offset
    while pos < total:
        for a, b, x1, y1, x2, y2 in segs:
            s, e = max(a, pos), min(b, pos + 13)
            if e > s:
                f1, f2 = (s - a) / (b - a), (e - a) / (b - a)
                xa, ya = x1 + (x2 - x1) * f1, y1 + (y2 - y1) * f1
                xb, yb = x1 + (x2 - x1) * f2, y1 + (y2 - y1) * f2
                d.line((xa, ya, xb, yb), fill=(*p["gold"], 80), width=18)
                d.line((xa, ya, xb, yb), fill=(*p["gold"], 255), width=8)
        pos += 22


def compose(name):
    p, big, drafter, top, roots, leaves = scene_data(name)
    base = render_base(name)
    leaves_meta = leaf_meta(name)
    grass = grass_meta(name)
    mask = body_mask()
    frames = []

    for f in range(FRAMES):
        t = f / FRAMES
        ta = min(1.0, t * 2.0)  # one activation burst in the first half
        ba = branch_angle(t)
        anim = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        d = ImageDraw.Draw(anim)

        # main generation pulse: leaves -> stem -> roots
        root_env = (math.sin(math.pi * (ta - 0.40) / 0.26)
                    if 0.40 < ta < 0.66 else 0.0)
        if root_env > 0:
            for ri, (pts, width) in enumerate(roots):
                pix = [(x * S, y * S) for x, y in pts]
                for k in range(2):
                    u = (((ta - 0.40) / 0.26) * 1.5 + k * 0.5 + ri * 0.07) % 1.0
                    x, y = point_along(pix, u)
                    draw_dot(d, x, y, p["accent"], 210 * root_env, r=6)

        # faint return: roots -> stem -> leaves
        if 0.64 < ta < 0.76:
            prog = (ta - 0.64) / 0.12
            env = math.sin(math.pi * prog)
            for ri, (pts, width) in enumerate(roots):
                pix = [(x * S, y * S) for x, y in pts]
                x, y = point_along(pix, max(0.0, 1.0 - prog))
                draw_dot(d, x, y, p["light"], 160 * env, r=6)

        # scattered grass blades sway on the painted ground
        for sprite, gx, gy, phase in grass:
            ga = 4.5 * math.sin(2 * math.pi * (t + phase))
            gs = sprite.rotate(ga, resample=Image.Resampling.NEAREST,
                               center=(gx * S, gy * S), expand=False)
            anim.alpha_composite(gs)

        # the branch, petioles and leaves sway as one shoot
        anim.alpha_composite(build_shoot(name, t, ta, leaves_meta))

        # graft band lights as the forward pulse passes through
        if 0.34 < ta < 0.48:
            be = math.sin(math.pi * (ta - 0.34) / 0.14)
            d.rectangle((42 * S, 32 * S, 54 * S, 34 * S),
                        fill=(*p["light"], int(160 * be)))

        # forward stem pulse follows the swaying branch
        if 0.16 < ta < 0.40:
            u = (ta - 0.16) / 0.24
            sy = top + u * (47 - top)
            sx, sy = stem_point(sy, ba)
            env = math.sin(math.pi * u)
            draw_dot(d, sx * S, sy * S, p["light"], 220 * env, r=7)

        # faint return up the stem
        if 0.72 < ta < 0.84:
            u = (ta - 0.72) / 0.12
            sy = 47 - u * (47 - top)
            sx, sy = stem_point(sy, ba)
            env = math.sin(math.pi * u)
            draw_dot(d, sx * S, sy * S, p["light"], 150 * env, r=6)

        # drafter: faint link always; a burst of draft pips in the gap
        if drafter:
            link_pix = []
            for lx, ly in [(44, 11), (48, 11), (48, 31)]:
                rx, ry = rot_point(lx, ly, ba)
                link_pix.append((rx * S, ry * S))
            draw_gold_link(anim, p, t, active=(0.86 < ta < 0.99),
                           pts=link_pix)
            for i in range(3):
                u = (ta - (0.86 + i * 0.025)) / 0.06
                if 0.0 <= u <= 1.0:
                    x, y = point_along(link_pix, u)
                    ux, uy = point_along(link_pix, max(0.0, u - 0.12))
                    d.line((ux, uy, x, y), fill=(*p["gold"], 100), width=6)
                    draw_pip(d, x, y, p["gold"], p["light"],
                             245 * (1.0 - 0.10 * u))
            if 0.92 < ta < 0.995:
                ge = math.sin(math.pi * (ta - 0.92) / 0.075)
                d.rectangle((40 * S, 30 * S, 56 * S, 36 * S),
                            fill=(*p["light"], int(50 * ge)))
                d.rectangle((42 * S, 32 * S, 54 * S, 34 * S),
                            fill=(*p["light"], int(210 * ge)))
            if 0.93 < ta < 0.995:
                prog = (ta - 0.93) / 0.065
                cx, cy = rot_point(48, 33, ba)
                cx, cy = cx * S, cy * S
                rad = (4 + 7 * prog) * S
                d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad),
                          outline=(*p["accent"], int(240 * (1 - prog))),
                          width=3)
                rad2 = max(1 * S, rad - 2 * S)
                d.ellipse((cx - rad2, cy - rad2, cx + rad2, cy + rad2),
                          outline=(*p["light"], int(190 * (1 - prog))),
                          width=1)

        anim.putalpha(ImageChops.multiply(anim.getchannel("A"), mask))
        frame = base.copy()
        frame.alpha_composite(anim)
        frames.append(frame)
    return frames


def main():
    for name in ("scion35", "scion176", "scion_drafter"):
        print("rendering", name, flush=True)
        frames = compose(name)
        path = OUT / f"{name}.webp"
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=DURATION, loop=0, quality=90, method=6)
        png = OUT / f"{name}.png"
        frames[0].save(png)
        print(path, path.stat().st_size, "bytes", flush=True)
        print(png, flush=True)
    css = """
body { margin:0; background:#f7f7f4; font:18px system-ui,sans-serif; color:#1c2a22; }
.row { display:flex; gap:20px; padding:24px; }
.card { text-align:center; }
img { width:360px; height:360px; display:block; }
.label { margin-top:8px; }
"""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head>
<body><div class="row">
<div class="card"><img src="scion35.webp"><div class="label">Scion-35B-A3B</div></div>
<div class="card"><img src="scion_drafter.webp"><div class="label">Scion-35B-A3B-mtp-drafter</div></div>
<div class="card"><img src="scion176.webp"><div class="label">Scion-FlashNext-176B-A6B</div></div>
</div></body></html>"""
    (OUT / "scion-animations-preview.html").write_text(html)
    print(OUT / "scion-animations-preview.html")


if __name__ == "__main__":
    main()
