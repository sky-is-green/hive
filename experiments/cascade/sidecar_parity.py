#!/usr/bin/env python
"""Engine-vs-client parity for the decision-head sidecar.

Takes one sample from a community run, rebuilds its judge prompt through the
model server's /apply-template, runs the fork's `test-decision-head` harness
(engine-loaded sidecar + real model forward) and compares p_correct with the
client-side JSON applied to the stored hidden state.

Usage:
    sidecar_parity.py --server http://127.0.0.1:8897 \
        --model /path/Ternary-Bonsai-2-27B-PQ2_0.gguf \
        --sidecar /path/bonsai-2-27b-decision-head.gguf \
        --head-json .../bonsai-2-27b-decision-head.json \
        --prefix bonsai-community --task gsm8k-0001 --sample 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from community_eval import SYSTEM, load_jsonl, order_body  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"
DEFAULT_HARNESS = (Path.home() / "llama.cpp" / "build" / "bin"
                   / "test-decision-head")


def post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--head-json", type=Path, required=True)
    ap.add_argument("--prefix", default="bonsai-community")
    ap.add_argument("--task", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    ap.add_argument("--out-dir", type=Path, default=R)
    ap.add_argument("--tolerance", type=float, default=2e-3)
    args = ap.parse_args()

    tasks = {t["id"]: t for t in json.loads(
        (CASCADE / "tasks-community.json").read_text())["tasks"]}
    rows = load_jsonl(args.out_dir / f"{args.prefix}-samples.jsonl")
    if not args.task:
        args.task = rows[0]["task_id"]
    rec = next(r for r in rows
               if r["task_id"] == args.task and r["sample"] == args.sample)

    prompt = post(args.server, "/apply-template", {
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": order_body(
                         {"prompt": tasks[args.task]["prompt"],
                          "candidate": rec["content"]}, False)}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False}})["prompt"]

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     dir="/tmp/opencode") as fh:
        fh.write(prompt)
        prompt_path = fh.name

    proc = subprocess.run(
        [str(args.harness), "-m", args.model, "--head", str(args.sidecar),
         "--prompt-file", prompt_path],
        capture_output=True, text=True, timeout=600)
    print(proc.stdout.strip() or proc.stderr.strip())
    if proc.returncode != 0:
        return 1
    p_engine = float(proc.stdout.split("p_correct=")[1].split()[0])

    head = json.loads(args.head_json.read_text())
    z = np.load(args.out_dir / f"{args.prefix}-states.npz", allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    idx = keys.index(f"{args.task}|sample|{args.sample}")
    h = z["H"][idx].astype(np.float64)
    w = np.asarray(head["weights"]["w"])
    mu = np.asarray(head["weights"]["mu"])
    sd = np.asarray(head["weights"]["sd"])
    s = float(((h - mu) / sd) @ w + head["weights"]["b"])
    a, c = head["weights"]["platt"]
    p_client = 1.0 / (1.0 + np.exp(-(a * s + c)))

    delta = abs(p_engine - p_client)
    out = {"task": args.task, "sample": args.sample,
           "p_engine": p_engine, "p_client": p_client, "delta": delta,
           "prompt_file": prompt_path}
    print(json.dumps(out, indent=1))
    ok = delta <= args.tolerance
    print("PARITY", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
