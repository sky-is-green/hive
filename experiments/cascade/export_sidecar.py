#!/usr/bin/env python
"""Export a decision-head JSON to an engine-loadable sidecar GGUF.

The sidecar carries exactly what the runtime cannot derive from the target
model: the ridge direction and standardisation (w / mu / sd), the bias, the
Platt pair, the hidden dim, and the sha256 of the serving prompt format it was
fitted on.  The engine refuses to apply a head whose format hash does not match
the prompt it is asked to score (the format-lock from the attribution audit).

GGUF layout (arch "decision-head"):
  metadata: decision_head.version/dim/b/platt_a/platt_c/format_sha256/
            model/candidate/name
  tensors:  dh.w, dh.mu, dh.sd   (F32, [dim])

Usage:
    export_sidecar.py --head bonsai-2-27b-decision-head.json \
        --out bonsai-2-27b-decision-head.gguf --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CASCADE = REPO / "experiments" / "cascade"
R = CASCADE / "results" / "jev-readout-20261006"
DEFAULT_GGUF_PY = Path("/home/penis/llama.cpp/gguf-py")

ARCH = "decision-head"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def format_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--head", type=Path,
                    default=R / "bonsai-2-27b-decision-head.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gguf-py", type=Path, default=DEFAULT_GGUF_PY)
    ap.add_argument("--prompt", default="",
                    help="override the prompt-format string hashed into the "
                         "sidecar (default: the head's own `prompt` field)")
    ap.add_argument("--verify", action="store_true",
                    help="reload the sidecar and check scores vs the JSON")
    args = ap.parse_args()
    if args.out is None:
        args.out = args.head.with_suffix(".gguf")

    sys.path.insert(0, str(args.gguf_py))
    from gguf import GGUFReader, GGUFWriter  # noqa: E402

    head = json.loads(args.head.read_text())
    w = np.asarray(head["weights"]["w"], dtype=np.float32)
    mu = np.asarray(head["weights"]["mu"], dtype=np.float32)
    sd = np.asarray(head["weights"]["sd"], dtype=np.float32)
    b = float(head["weights"]["b"])
    pa, pc = (float(x) for x in head["weights"]["platt"])
    dim = int(head["readout"]["dim"])
    assert w.shape == mu.shape == sd.shape == (dim,), (w.shape, mu.shape, sd.shape)

    prompt = args.prompt or head.get("prompt", "")
    fsha = format_sha(prompt)

    writer = GGUFWriter(str(args.out), ARCH)
    writer.add_uint32("decision_head.version", 1)
    writer.add_uint32("decision_head.dim", dim)
    writer.add_float32("decision_head.b", b)
    writer.add_float32("decision_head.platt_a", pa)
    writer.add_float32("decision_head.platt_c", pc)
    writer.add_string("decision_head.format_sha256", fsha)
    writer.add_string("decision_head.model", str(head.get("model", "")))
    writer.add_string("decision_head.candidate",
                      str(head.get("readout", {}).get("candidate", "answer")))
    writer.add_string("decision_head.name", str(head.get("name", "")))
    writer.add_tensor("dh.w", w)
    writer.add_tensor("dh.mu", mu)
    writer.add_tensor("dh.sd", sd)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} ({args.out.stat().st_size/1024:.0f} KB, dim {dim}, "
          f"format sha {fsha[:12]})")

    if args.verify:
        reader = GGUFReader(str(args.out))
        meta = {k: reader.fields[k].contents() for k in (
            "decision_head.dim", "decision_head.b", "decision_head.platt_a",
            "decision_head.platt_c", "decision_head.format_sha256")}
        fsha_read = meta["decision_head.format_sha256"]
        if isinstance(fsha_read, bytes):
            fsha_read = fsha_read.decode()
        by_name = {t.name: t for t in reader.tensors}
        rw = np.asarray(by_name["dh.w"].data).astype(np.float64)
        rmu = np.asarray(by_name["dh.mu"].data).astype(np.float64)
        rsd = np.asarray(by_name["dh.sd"].data).astype(np.float64)
        rng = np.random.default_rng(7)
        h = rng.normal(size=dim).astype(np.float64)
        p_json = sigmoid(pa * (((h - np.asarray(head["weights"]["mu"]))
                                / np.asarray(head["weights"]["sd"]))
                               @ np.asarray(head["weights"]["w"]) + b) + pc)
        p_side = sigmoid(float(meta["decision_head.platt_a"])
                         * (((h - rmu) / rsd) @ rw
                            + float(meta["decision_head.b"]))
                         + float(meta["decision_head.platt_c"]))
        ok = (int(meta["decision_head.dim"]) == dim
              and fsha_read == fsha
              and abs(p_json - p_side) < 1e-4)
        print(f"verify: json p={p_json:.6f} sidecar p={p_side:.6f} "
              f"delta={abs(p_json-p_side):.2e} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
