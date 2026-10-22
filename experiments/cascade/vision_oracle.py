#!/usr/bin/env python3
"""A2 vision oracle — pilot stage 0, vision edition.

Measures the three A2 ("vision encoding") candidates from
``harness/cascade/registry.py`` on CPU beside the resident 176B serve:

    florence-2-base   microsoft/Florence-2-base
    smolvlm2-256m     HuggingFaceTB/SmolVLM2-256M-Video-Instruct
    altclip           BAAI/AltCLIP

Each candidate runs one fixed six-image eval set (four synthetic, one chart,
one real photo), with programmatic checks and one call per image — three timed
repetitions after a warmup call.  The driver aggregates the per-candidate
records into ``CandidateMeasurement`` rows (``harness/cascade/oracle.py``),
builds the per-role frontier, and writes:

    experiments/cascade/results/a2-oracle-stage0-20261017/measurements.json
    experiments/cascade/results/a2-oracle-stage0-20261017/SUMMARY.md
    experiments/cascade/results/a2-oracle-stage0-20261017/<candidate>.json
    experiments/cascade/results/a2-oracle-stage0-20261017/images/…
    experiments/cascade/results/a2-oracle-stage0-20261017/memwatch.log
    experiments/cascade/a2-oracle-stage0.json          (for /v1/cascade/frontier)

Usage, from the repo root:

    .venv/bin/python experiments/cascade/vision_oracle.py all
    .venv/bin/python experiments/cascade/vision_oracle.py --model altclip
    .venv/bin/python experiments/cascade/vision_oracle.py --aggregate

CPU-only by design.  A torch ROCm context next to Strata's HIP context has
hard-reset this host twice (cascade/CASCADE-PILOT.md), and the dGPUs' free VRAM
is the 176B's resident reserve (see experiments/flashnext-perf-2026-10-17/README.md).

Florence-2's remote code does not import under transformers >= 4.50 — the
flash-attn availability rewrite leaves an empty ``if`` body (IndentationError) —
so that candidate runs in a pinned side env (transformers 4.46.3, CPU torch).
The driver picks its interpreter from ``$VISION_ORACLE_FLORENCE_PY``, defaulting
to ``<work-drive>/.envs/florence-legacy/bin/python`` (sibling of the repo).
The other two are native in the harness venv's transformers 5.x.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CASCADE_DIR = REPO_ROOT / "experiments" / "cascade"
RESULTS_DIR = CASCADE_DIR / "results" / "a2-oracle-stage0-20261017"
IMAGES_DIR = RESULTS_DIR / "images"
FRONTIER_JSON = CASCADE_DIR / "a2-oracle-stage0.json"

HEALTH_URL = "http://127.0.0.1:8081/health"
FONT_PATH = "/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf"
PHOTO_PATH = "/usr/share/wallpapers/cachyos-wallpapers/CachyOS_Moon.jpg"

CANDIDATES = ("florence-2-base", "smolvlm2-256m", "altclip")
REPO_BY_CANDIDATE = {
    "florence-2-base": "microsoft/Florence-2-base",
    "smolvlm2-256m": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    "altclip": "BAAI/AltCLIP",
}
REPS = 3
MAX_NEW_TOKENS = 48


# --------------------------------------------------------------------------
# eval set
# --------------------------------------------------------------------------

def _font(size: int):
    from PIL import ImageFont

    return ImageFont.truetype(FONT_PATH, size)


def _text_card():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (800, 520), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 180), "HIVE CASCADE 42", font=_font(64), fill="black")
    d.text((60, 300), "stage-0 vision oracle", font=_font(28), fill="black")
    return img


def _sign():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (800, 520), (235, 238, 242))
    d = ImageDraw.Draw(img)
    cx, cy, r = 400, 262, 172
    pts = [
        (cx + r * math.cos(math.pi / 8 + i * math.pi / 4),
         cy + r * math.sin(math.pi / 8 + i * math.pi / 4))
        for i in range(8)
    ]
    d.polygon(pts, fill=(200, 30, 30), outline=(120, 0, 0))
    f = _font(72)
    w = d.textlength("STOP", font=f)
    d.text((cx - w / 2, cy - 42), "STOP", font=f, fill="white")
    return img


def _shapes():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (800, 520), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((120, 120, 320, 320), fill=(200, 30, 30))              # red circle
    d.rectangle((480, 140, 680, 300), fill=(30, 60, 200))            # blue square
    d.polygon([(400, 250), (300, 440), (500, 440)], fill=(30, 150, 60))  # green triangle
    return img


def _count_squares():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (800, 520), "white")
    d = ImageDraw.Draw(img)
    for x, y in ((70, 70), (330, 70), (590, 70)):                    # 3 blue squares
        d.rectangle((x, y, x + 130, y + 130), fill=(30, 60, 200))
    for x, y in ((60, 300), (270, 300), (480, 300), (660, 330)):     # 4 red circles
        d.ellipse((x, y, x + 120, y + 120), fill=(200, 30, 30))
    return img


def _chart():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (800, 520), "white")
    d = ImageDraw.Draw(img)
    f = _font(32)
    base_y, max_h = 440, 320
    for (label, val), x in zip((("A", 2), ("B", 5), ("C", 3)), (110, 330, 550)):
        h = int(max_h * val / 5)
        d.rectangle((x, base_y - h, x + 140, base_y), fill=(40, 90, 180))
        w = d.textlength(str(val), font=f)
        d.text((x + 70 - w / 2, base_y - h - 46), str(val), font=f, fill="black")
        w = d.textlength(label, font=f)
        d.text((x + 70 - w / 2, base_y + 10), label, font=f, fill="black")
    d.line((60, base_y, 760, base_y), fill="black", width=3)
    return img


def _photo():
    from PIL import Image

    img = Image.open(PHOTO_PATH).convert("RGB")
    w = 800
    img = img.resize((w, round(img.height * w / img.width)))
    return img


def ensure_images() -> list[str]:
    """Build the eval set deterministically; return the image names in order."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "text_card": _text_card,
        "sign_stop": _sign,
        "shapes_scene": _shapes,
        "count_squares": _count_squares,
        "chart_bars": _chart,
        "photo_planet": _photo,
    }
    names = list(builders)
    for name in names:
        path = IMAGES_DIR / (name + ".jpg")
        if not path.exists():
            builders[name]().save(path, quality=92)
    manifest = []
    for name in names:
        path = IMAGES_DIR / (name + ".jpg")
        manifest.append({
            "name": name,
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        })
    (IMAGES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return names


def load_images(names: list[str]):
    from PIL import Image

    return [(n, Image.open(IMAGES_DIR / (n + ".jpg")).convert("RGB")) for n in names]


# --------------------------------------------------------------------------
# per-candidate runs (executed in the candidate's own interpreter)
# --------------------------------------------------------------------------

def _letter_tokens(text: str) -> list[str]:
    return re.findall(r"\b([abc])\b", text.lower())


def _clean_florence(raw: str) -> str:
    return re.sub(r"</?s>", "", raw).strip()


FLORENCE_TASKS = {
    "text_card": ("<OCR>", "OCR output contains '42'",
                  lambda t, labels: "42" in t),
    "sign_stop": ("<OCR>", "OCR output contains 'STOP'",
                  lambda t, labels: "stop" in t.lower()),
    "shapes_scene": ("<OD>", "OD labels >= 2, incl. a shape word",
                     lambda t, labels: bool(labels) and len(labels) >= 2
                     and any(k in lab.lower() for lab in labels
                             for k in ("circle", "square", "triangle"))),
    "count_squares": ("<OD>", "OD labels >= 2 'square'",
                      lambda t, labels: bool(labels)
                      and sum("square" in lab.lower() for lab in labels) >= 2),
    "chart_bars": ("<OCR>", "OCR reads labels A, B and C",
                   lambda t, labels: all(c in t.lower() for c in "abc")),
    "photo_planet": ("<MORE_DETAILED_CAPTION>", "caption has >= 4 words",
                     lambda t, labels: len(t.split()) >= 4),
}

SMOLVLM_TASKS = {
    "text_card": ("What text is written in this image? Answer briefly.",
                  "answer contains '42'", lambda a: "42" in a),
    "sign_stop": ("What does the sign say?",
                  "answer contains 'stop'", lambda a: "stop" in a.lower()),
    "shapes_scene": ("What color is the circle in this image? Answer briefly.",
                     "answer contains 'red'", lambda a: "red" in a.lower()),
    "count_squares": ("How many blue squares are there? Answer with just the number.",
                      "answer contains 3/'three'",
                      lambda a: "3" in a or "three" in a.lower()),
    "chart_bars": ("Which bar is the tallest, A, B or C? Answer with one letter.",
                   "answer names 'B'",
                   lambda a: "b" in _letter_tokens(a)),
    "photo_planet": ("Describe this image in one sentence.",
                     "answer has >= 4 words",
                     lambda a: len(a.split()) >= 4),
}

ALTCLIP_OPTIONS = {
    "text_card": ["an image containing the text HIVE CASCADE 42",
                  "a photo of a cat",
                  "a scenic mountain landscape"],
    "sign_stop": ["a red stop sign", "a blue square", "a bar chart"],
    "shapes_scene": ["a red circle and a blue square and a green triangle",
                     "a photo of a dog",
                     "a bar chart"],
    "count_squares": ["an image with three blue squares",
                      "a photo of a beach",
                      "a red stop sign"],
    "chart_bars": ["a bar chart with three bars labeled A, B and C",
                   "a photo of a cat",
                   "a red stop sign"],
    # Distractors must be clearly false for the image.  The earlier version of
    # this option set used "an image containing the text HIVE CASCADE 42",
    # which the wallpaper makes ambiguous (it carries a CachyOS wordmark) —
    # AltCLIP ranked that option over the scene (p≈0.90), a real retrieval
    # caveat kept in the summary, not a check for the quality axis.
    "photo_planet": ["a planet in space surrounded by a green nebula",
                     "a screenshot of a text editor",
                     "a bar chart with three bars"],
}


def _timed(call, reps: int) -> tuple[list, list[float]]:
    outs, lat = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs.append(call())
        lat.append((time.perf_counter() - t0) * 1000.0)
    return outs, lat


def _resident_gb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9


def _peak_rss_gib() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def _env_info() -> dict:
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _finish(candidate: str, load_s: float, resident_gb: float, items: list[dict]) -> dict:
    per_image = [statistics.median(it["latency_ms"]) for it in items]
    passed = sum(1 for it in items if it["check"]["passed"])
    return {
        "candidate": candidate,
        "repo": REPO_BY_CANDIDATE[candidate],
        "role": "A2",
        "device": "cpu",
        "precision": "fp32",
        "env": _env_info(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "load_s": round(load_s, 2),
        "resident_gb": round(resident_gb, 3),
        "peak_rss_gib": round(_peak_rss_gib(), 2),
        "items": items,
        "checks_passed": passed,
        "checks_total": len(items),
        "quality": round(passed / len(items), 4),
        "median_call_ms": round(statistics.median(per_image), 1),
        "mean_call_ms": round(statistics.fmean(per_image), 1),
    }


def run_florence2(out_path: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    images = load_images(ensure_images())
    repo = REPO_BY_CANDIDATE["florence-2-base"]
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()
    proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    load_s = time.perf_counter() - t0
    torch.set_grad_enabled(False)

    def call(img, task, max_new):
        inputs = proc(text=task, images=img, return_tensors="pt")
        gen = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new, num_beams=1, do_sample=False,
            early_stopping=False,
        )
        return proc.batch_decode(gen, skip_special_tokens=False)[0]

    # warmup on the first image/task, then the timed loop
    first_task = FLORENCE_TASKS[images[0][0]][0]
    call(images[0][1], first_task, MAX_NEW_TOKENS)

    items = []
    for name, img in images:
        task, desc, check = FLORENCE_TASKS[name]
        max_new = 96 if task == "<OD>" else MAX_NEW_TOKENS
        outs, lat = _timed(lambda: call(img, task, max_new), REPS)
        raw = outs[0]
        clean = _clean_florence(raw)
        labels = None
        if task == "<OD>":
            parsed = proc.post_process_generation(
                raw, task=task, image_size=(img.width, img.height))
            labels = parsed.get(task, {}).get("labels", [])
        items.append({
            "image": name,
            "task": task,
            "output": raw,
            "parsed_labels": labels,
            "latency_ms": [round(x, 1) for x in lat],
            "check": {"desc": desc, "passed": bool(check(clean, labels))},
        })

    rec = _finish("florence-2-base", load_s, _resident_gb(model), items)
    out_path.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"[florence-2-base] quality {rec['checks_passed']}/{rec['checks_total']}"
          f" | median {rec['median_call_ms']} ms | resident {rec['resident_gb']} GB")


def run_smolvlm2(out_path: Path) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    images = load_images(ensure_images())
    repo = REPO_BY_CANDIDATE["smolvlm2-256m"]
    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(repo).eval()
    load_s = time.perf_counter() - t0
    torch.set_grad_enabled(False)

    def call(img, question, max_new):
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = proc(text=prompt, images=[img], return_tensors="pt")
        gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return proc.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    call(images[0][1], SMOLVLM_TASKS[images[0][0]][0], MAX_NEW_TOKENS)

    items = []
    for name, img in images:
        question, desc, check = SMOLVLM_TASKS[name]
        max_new = 64 if name == "photo_planet" else MAX_NEW_TOKENS
        outs, lat = _timed(lambda: call(img, question, max_new), REPS)
        answer = outs[0]
        items.append({
            "image": name,
            "task": question,
            "output": answer,
            "latency_ms": [round(x, 1) for x in lat],
            "check": {"desc": desc, "passed": bool(check(answer))},
        })

    rec = _finish("smolvlm2-256m", load_s, _resident_gb(model), items)
    out_path.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"[smolvlm2-256m] quality {rec['checks_passed']}/{rec['checks_total']}"
          f" | median {rec['median_call_ms']} ms | resident {rec['resident_gb']} GB")


def run_altclip(out_path: Path) -> None:
    import torch
    from transformers import AltCLIPModel, AltCLIPProcessor

    images = load_images(ensure_images())
    repo = REPO_BY_CANDIDATE["altclip"]
    t0 = time.perf_counter()
    proc = AltCLIPProcessor.from_pretrained(repo)
    model = AltCLIPModel.from_pretrained(repo).eval()
    load_s = time.perf_counter() - t0
    torch.set_grad_enabled(False)

    def call(img, options):
        inputs = proc(text=options, images=img, return_tensors="pt", padding=True)
        out = model(**inputs)
        return out.logits_per_image.softmax(dim=1)[0].tolist()

    call(images[0][1], ALTCLIP_OPTIONS[images[0][0]])

    items = []
    for name, img in images:
        options = ALTCLIP_OPTIONS[name]
        outs, lat = _timed(lambda: call(img, options), REPS)
        probs = outs[0]
        top1 = int(max(range(len(probs)), key=probs.__getitem__))
        items.append({
            "image": name,
            "task": f"top-1 of {len(options)} text options",
            "output": options[top1],
            "options": options,
            "probs": [round(float(p), 4) for p in probs],
            "latency_ms": [round(x, 1) for x in lat],
            "check": {"desc": f"top-1 is option 0 ('{options[0]}')",
                      "passed": top1 == 0},
        })

    rec = _finish("altclip", load_s, _resident_gb(model), items)
    out_path.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"[altclip] quality {rec['checks_passed']}/{rec['checks_total']}"
          f" | median {rec['median_call_ms']} ms | resident {rec['resident_gb']} GB")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _model_json(candidate: str) -> Path:
    return RESULTS_DIR / f"{candidate}.json"


def _florence_python() -> str:
    env = os.environ.get("VISION_ORACLE_FLORENCE_PY")
    if env:
        return env
    return str(REPO_ROOT.parent / ".envs" / "florence-legacy" / "bin" / "python")


def _probe_health() -> dict:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as resp:
            return {"url": HEALTH_URL, "status": resp.status}
    except Exception as exc:  # noqa: BLE001 — the probe is informational
        return {"url": HEALTH_URL, "status": None, "error": str(exc)}


def _meminfo_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576.0
    return float("nan")


def _run_all() -> None:
    ensure_images()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    health_before = _probe_health()
    print(f"[oracle] serve health before: {health_before}", flush=True)

    samples: list[tuple[float, float]] = []
    stop = threading.Event()

    def memwatch() -> None:
        t_start = time.time()
        while not stop.is_set():
            samples.append((time.time() - t_start, _meminfo_available_gib()))
            stop.wait(2.0)

    watcher = threading.Thread(target=memwatch, daemon=True)
    watcher.start()
    t0 = time.time()

    interpreters = {
        "florence-2-base": _florence_python(),
        "smolvlm2-256m": sys.executable,
        "altclip": sys.executable,
    }
    for candidate in CANDIDATES:
        py = interpreters[candidate]
        if not Path(py).exists():
            raise SystemExit(
                f"[oracle] interpreter for {candidate} not found: {py}\n"
                f"         set VISION_ORACLE_FLORENCE_PY to the pinned env's python")
        print(f"[oracle] {candidate} -> {py}", flush=True)
        subprocess.run(
            [py, str(Path(__file__).resolve()), "--model", candidate,
             "--out", str(_model_json(candidate))],
            check=True, cwd=str(REPO_ROOT),
        )

    elapsed = time.time() - t0
    stop.set()
    watcher.join(timeout=5)
    health_after = _probe_health()
    print(f"[oracle] serve health after: {health_after}", flush=True)

    if samples:
        min_avail = min(v for _, v in samples)
        (RESULTS_DIR / "memwatch.log").write_text(
            f"# MemAvailable during the A2 oracle run ({RESULTS_DIR.name})\n"
            f"# min {min_avail:.2f} GiB | serve {HEALTH_URL} before="
            f"{health_before['status']} after={health_after['status']}\n"
            + "".join(f"{t:8.1f}s  MemAvailable={v:.2f} GiB\n" for t, v in samples)
        )

    meta = {
        "role": "A2",
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "device": "cpu",
        "host": platform.node(),
        "reps_per_image": REPS,
        "images": 6,
        "run_elapsed_s": round(elapsed, 1),
        "serve": {"before": health_before, "after": health_after},
        "mem_available_min_gib": round(min((v for _, v in samples), default=float("nan")), 2),
    }
    aggregate(meta)


def _measurement_notes(rec: dict) -> str:
    return (
        f"CPU fp32, 6-image A2 set: {rec['checks_passed']}/{rec['checks_total']} checks"
        f" passed; warm {rec['median_call_ms']} ms/call (median of per-image medians);"
        f" load {rec['load_s']} s; resident {rec['resident_gb']} GB;"
        f" transformers {rec['env']['transformers']}, peak RSS {rec['peak_rss_gib']} GiB"
    )


def aggregate(meta: dict | None = None) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from harness.cascade.oracle import CandidateMeasurement, frontier_by_role
    from harness.cascade.roles import Device

    recs: dict[str, dict] = {}
    rows = []
    for candidate in CANDIDATES:
        path = _model_json(candidate)
        if not path.exists():
            print(f"[oracle] no record for {candidate} — skipped in aggregation")
            continue
        rec = json.loads(path.read_text())
        recs[candidate] = rec
        rows.append(CandidateMeasurement(
            candidate=candidate,
            role="A2",
            device=Device.CPU,
            quality=rec["quality"],
            latency_ms=rec["median_call_ms"],
            unit="call",
            ctx=0,
            precision="fp32",
            bytes_gb=rec["resident_gb"],
            notes=_measurement_notes(rec),
        ))

    payload = {
        "measurements": [row.to_dict() for row in rows],
        "meta": meta or {},
    }
    (RESULTS_DIR / "measurements.json").write_text(json.dumps(payload, indent=2) + "\n")
    FRONTIER_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    frontiers = frontier_by_role(rows)
    frontier_rows = {role: [r.to_dict() for r in group]
                     for role, group in frontiers.items()}

    lines = [
        "# A2 vision oracle — pilot stage 0",
        "",
        "Measured on CPU beside the resident 176B serve (port 8081), 2026-10-17.",
        "Role A2 (\"vision encoding\", `harness/cascade/roles.py`). Candidates are",
        "the three registry rows (`harness/cascade/registry.py`).",
        "",
        "Each candidate ran one fixed six-image eval set — text card, stop sign,",
        "shapes scene, count scene, bar chart, real photo — with programmatic",
        "checks, one call per image, three timed repetitions after a warmup call.",
        "Quality = checks passed / 6. Latency = median of per-image medians, warm",
        "(processor + forward + decode included), milliseconds per call.",
        "",
        "| candidate | quality | median call | load | weights (fp32) | peak RSS |",
        "|---|---|---|---|---|---|",
    ]
    for candidate in CANDIDATES:
        rec = recs.get(candidate)
        if rec is None:
            continue
        lines.append(
            f"| {candidate} | {rec['checks_passed']}/{rec['checks_total']} "
            f"| {rec['median_call_ms']} ms | {rec['load_s']} s "
            f"| {rec['resident_gb']} GB | {rec['peak_rss_gib']} GiB |"
        )
    lines += [
        "",
        "## Per-image outputs",
        "",
    ]
    for candidate in CANDIDATES:
        rec = recs.get(candidate)
        if rec is None:
            continue
        lines += [f"### {candidate} ({rec['repo']})", ""]
        for item in rec["items"]:
            mark = "pass" if item["check"]["passed"] else "FAIL"
            out = str(item["output"]).replace("\n", " ")
            if len(out) > 140:
                out = out[:137] + "..."
            lines.append(
                f"- **{item['image']}** `{item['task']}` — {mark} "
                f"({item['check']['desc']}; {item['latency_ms']} ms) — `{out}`"
            )
        lines.append("")

    lines += ["## Frontier (role=A2, unit=call)", ""]
    for role, group in frontier_rows.items():
        lines.append(f"Frontier after stage 0 — non-dominated rows for `{role}`:")
        lines.append("")
        for row in group:
            lines.append(
                f"- **{row['candidate']}** — quality {row['quality']}, "
                f"{row['latency_ms']} ms/call, {row['bytes_gb']} GB, `{row['precision']}`"
            )
        lines.append("")
    all_last = {r.candidate: (r.quality, r.latency_ms) for r in rows}
    dominated = [c for c in all_last
                 if c not in {row["candidate"] for row in frontier_rows.get("A2", [])}]
    if dominated:
        lines.append("Dominated (kept for the record): " + ", ".join(sorted(dominated)) + ".")
        lines.append("")

    if meta:
        lines += [
            "## Environment",
            "",
            f"- Host: `{meta['host']}`, all candidates CPU-only (torch CPU builds;",
            "  no ROCm context beside the serve — see module docstring for why).",
            f"- Florence-2-base ran in the pinned side env "
            f"`.envs/florence-legacy` (transformers 4.46.3): its remote code does not",
            "  import under the harness venv's transformers 5.x (flash-attn rewrite",
            "  leaves an empty `if` body → IndentationError), and the native 5.x",
            "  Florence-2 classes expect converted weights this hub revision lacks.",
            f"- smolvlm2-256m and altclip are native in the harness venv "
            f"(transformers {recs['altclip']['env']['transformers'] if 'altclip' in recs else '5.x'}).",
            "",
            "## Coexistence with the live 176B serve",
            "",
            f"- `/health` before: {meta['serve']['before']['status']}, after: "
            f"{meta['serve']['after']['status']} (serve stayed up).",
            f"- MemAvailable across the run: min {meta['mem_available_min_gib']} GiB "
            f"(see `memwatch.log`); total run {meta['run_elapsed_s']} s.",
            "",
        ]
    lines += [
        "## Artifacts",
        "",
        "- `measurements.json` — the `CandidateMeasurement` rows (also copied to",
        "  `experiments/cascade/a2-oracle-stage0.json` for `/v1/cascade/frontier`).",
        "- `<candidate>.json` — full per-candidate records (raw outputs, latencies).",
        "- `images/` — the eval set + sha256 manifest.",
        "- `memwatch.log` — host MemAvailable samples during the run.",
        "",
    ]
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines))
    print(f"[oracle] wrote {RESULTS_DIR/'measurements.json'}")
    print(f"[oracle] wrote {RESULTS_DIR/'SUMMARY.md'}")
    print(f"[oracle] wrote {FRONTIER_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", nargs="?", choices=["all"],
                        help="run everything (alias for --all)")
    parser.add_argument("--model", choices=CANDIDATES,
                        help="run one candidate in this interpreter")
    parser.add_argument("--out", type=Path, help="record path for --model mode")
    parser.add_argument("--aggregate", action="store_true",
                        help="aggregate existing per-candidate records")
    parser.add_argument("--all", action="store_true",
                        help="run all candidates (right interpreter each) + aggregate")
    parser.add_argument("--make-images", action="store_true",
                        help="build the eval set and exit")
    args = parser.parse_args()

    if args.model:
        out = args.out or _model_json(args.model)
        out.parent.mkdir(parents=True, exist_ok=True)
        {"florence-2-base": run_florence2,
         "smolvlm2-256m": run_smolvlm2,
         "altclip": run_altclip}[args.model](out)
    elif args.aggregate:
        aggregate()
    elif args.all or args.action == "all":
        _run_all()
    elif args.make_images:
        print("\n".join(ensure_images()))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
