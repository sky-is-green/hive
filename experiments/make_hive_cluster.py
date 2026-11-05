#!/usr/bin/env python
"""Compose the five Hive 1.7B heroes into one connected hive banner.

    python experiments/make_hive_cluster.py

Writes assets/heroes/hive-heroes.webp (10 s loop) and hive-heroes.png
(static fallback). The Conductor sits at the centre; Reader, Auditor,
Dispatcher and Tool Manager sit around it, joined by honey-coloured links.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "assets" / "heroes"

W, H = 1100, 820
TILE = 260
FRAMES = 100
DURATION = 100  # ms, 10 s loop
GIF_FRAMES = 60
GIF_SIZE = (900, 671)

BG = (8, 14, 24, 255)
HIVE_LINE = (88, 66, 25, 255)
HIVE_GLOW = (35, 28, 12, 255)
HIVE_EDGE = (150, 105, 25, 255)
PULSE = (255, 183, 3, 255)
PULSE_CORE = (255, 241, 200, 255)

CENTER = (550, 410)
HEROES = (
    ("conductor", CENTER),
    ("reader", (270, 230)),
    ("auditor", (830, 230)),
    ("dispatcher", (270, 590)),
    ("toolmanager", (830, 590)),
)


def hex_points(cx: float, cy: float, r: float):
    return [(cx + r * math.cos(math.pi / 3 * i),
             cy + r * math.sin(math.pi / 3 * i)) for i in range(6)]


def background() -> Image.Image:
    img = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(img)
    r = 55
    height = math.sqrt(3) * r
    for col in range(-1, int(W / (1.5 * r)) + 2):
        cx = col * 1.5 * r
        for row in range(-1, int(H / height) + 2):
            cy = row * height + (height / 2 if col % 2 else 0)
            d.polygon(hex_points(cx, cy, r), outline=(23, 45, 58, 255))
    return img


def draw_links(img: Image.Image) -> None:
    d = ImageDraw.Draw(img)
    # outer hive ring
    ring = ("reader", "auditor", "toolmanager", "dispatcher")
    for a, b in zip(ring, ring[1:] + ring[:1]):
        pa = dict(HEROES)[a]
        pb = dict(HEROES)[b]
        d.line((*pa, *pb), fill=HIVE_LINE, width=2)
    # spokes into the controller
    for name in ("reader", "auditor", "dispatcher", "toolmanager"):
        p = dict(HEROES)[name]
        d.line((*CENTER, *p), fill=HIVE_GLOW, width=5)
        d.line((*CENTER, *p), fill=HIVE_EDGE, width=2)


def _pulse(d, start, end, u: float, size: int = 5) -> None:
    for k in range(3):
        uu = u - k * 0.09
        if not 0 <= uu <= 1:
            continue
        x = start[0] + (end[0] - start[0]) * uu
        y = start[1] + (end[1] - start[1]) * uu
        if k == 0:
            d.rectangle((x - size, y - size, x + size, y + size), fill=HIVE_EDGE)
            d.rectangle((x - size + 2, y - size + 2,
                         x + size - 2, y + size - 2), fill=PULSE)
            d.rectangle((x - 1, y - 1, x + 1, y + 1), fill=PULSE_CORE)
        else:
            r = max(1, size - 3)
            d.rectangle((x - r, y - r, x + r, y + r), fill=PULSE)


def draw_pulses(img: Image.Image, t: float) -> None:
    d = ImageDraw.Draw(img)
    pos = dict(HEROES)
    # centre -> role
    for i, name in enumerate(("reader", "auditor", "dispatcher", "toolmanager")):
        px, py = pos[name]
        dx, dy = px - CENTER[0], py - CENTER[1]
        dist = math.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        start = (CENTER[0] + ux * 145, CENTER[1] + uy * 145)
        end = (px - ux * 145, py - uy * 145)
        _pulse(d, start, end, (t * 3 + i * 0.25) % 1.0)
    # outer hive circuit
    ring = (("reader", "auditor"), ("auditor", "toolmanager"),
            ("toolmanager", "dispatcher"), ("dispatcher", "reader"))
    for j, (a, b) in enumerate(ring):
        pa, pb = pos[a], pos[b]
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        dist = math.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        start = (pa[0] + ux * 145, pa[1] + uy * 145)
        end = (pb[0] - ux * 145, pb[1] - uy * 145)
        _pulse(d, start, end, (t * 2 + j * 0.25) % 1.0, size=3)


def load_animation(name: str):
    im = Image.open(OUT / f"{name}-256.webp")
    frames, durations = [], []
    while True:
        frames.append(im.convert("RGBA").resize((TILE, TILE),
                                                Image.Resampling.LANCZOS))
        durations.append(int(im.info.get("duration", DURATION)))
        try:
            im.seek(im.tell() + 1)
        except EOFError:
            break
    return frames, durations


def sample(frames, durations, t: float):
    total = sum(durations)
    if total <= 0:
        return frames[0]
    at = (t * total) % total
    acc = 0
    for frame, duration in zip(frames, durations):
        acc += duration
        if at < acc:
            return frame
    return frames[-1]


def compose(t: float, animations) -> Image.Image:
    img = background()
    draw_links(img)
    draw_pulses(img, t)
    for name, (cx, cy) in HEROES:
        frame = sample(*animations[name], t)
        img.alpha_composite(frame, (cx - TILE // 2, cy - TILE // 2))
    return img


def compose_static() -> Image.Image:
    img = background()
    draw_links(img)
    draw_pulses(img, 0.15)
    for name, (cx, cy) in HEROES:
        tile = Image.open(OUT / f"{name}-256.png").convert("RGBA")
        tile = tile.resize((TILE, TILE), Image.Resampling.LANCZOS)
        img.alpha_composite(tile, (cx - TILE // 2, cy - TILE // 2))
    return img


def save_gif(frames, path: Path) -> None:
    picks = [round(i * (len(frames) - 1) / (GIF_FRAMES - 1))
             for i in range(GIF_FRAMES)]
    selected = [frames[i].convert("RGB").resize(GIF_SIZE,
                                                Image.Resampling.LANCZOS)
                for i in picks]
    base = selected[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=base, dither=Image.Dither.NONE)
                 for frame in selected]
    quantized[0].save(path, save_all=True, append_images=quantized[1:],
                      duration=round(10000 / GIF_FRAMES), loop=0,
                      optimize=True, disposal=2)


def main() -> int:
    animations = {name: load_animation(name) for name, _ in HEROES}
    frames = [compose(i / FRAMES, animations) for i in range(FRAMES)]
    # Start on the completed state so the README poster is legible before
    # the loop begins.
    frames = frames[-1:] + frames[:-1]

    webp = OUT / "hive-heroes.webp"
    frames[0].save(webp, save_all=True, append_images=frames[1:],
                   duration=DURATION, loop=0, quality=88, method=6)
    png = OUT / "hive-heroes.png"
    compose_static().save(png)
    gif = OUT / "hive-heroes.gif"
    save_gif(frames, gif)
    print(webp, webp.stat().st_size, "bytes", len(frames), "frames")
    print(png, png.stat().st_size, "bytes")
    print(gif, gif.stat().st_size, "bytes", GIF_FRAMES, "frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
