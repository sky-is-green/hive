#!/usr/bin/env python3
"""flashnext-176B serve benchmark v2 - same script before/after every tuning window.

Per run: unique prompt (no prefix reuse), then read the API timings AND the engine
log's own lines for that request (prompt read ms/"reused", decode ms/tok, cache hit,
file tier I/O).  Sizes: ~300 / ~1000 / ~3000 estimated tokens (max 4096 context),
plus two 256-token decode runs.  Results -> results.json beside this script.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8081/v1/chat/completions"
LOG = "<work>/hivebench/logs/strata_q6k.engine.log"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

SENT = ("The routing layer keeps the busiest experts resident on the cards while the rest wait in RAM. "
        "A prompt window borrows cache slots and hands them back; the profile ranks every expert pair. ")
LEAD = "Summarize the passage below in one sentence. Passage #{nonce}, variation {v}:\n"


def call(tag: str, est_tokens: int, max_tokens: int) -> dict:
    reps = max(2, est_tokens // 40)
    prompt = LEAD.format(nonce=tag, v=time.time_ns() % 100000) + SENT * reps
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        return {"name": tag, "error": f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"}
    wall = time.time() - t0
    u, t = resp.get("usage", {}), resp.get("timings", {})
    dec = t.get("predicted_per_second")
    return {
        "name": tag,
        "wall_s": round(wall, 2),
        "prompt_n": u.get("prompt_tokens"),
        "read_n": t.get("prompt_n"),
        "reused_n": t.get("cache_n"),
        "completion_n": u.get("completion_tokens"),
        "prompt_tps": round(t.get("prompt_per_second", 0), 1),
        "decode_tps": round(dec, 2) if dec else None,
        "decode_ms_per_tok": round(1000.0 / dec, 2) if dec else None,
    }


def log_tail(offset: int) -> tuple:
    try:
        with open(LOG, "r", errors="replace") as f:
            f.seek(offset)
            return f.read().splitlines(), f.tell()
    except OSError:
        return [], offset


def pick(lines: list) -> list:
    return [ln for ln in lines if ln.startswith(("strata serve:", "strata mtp", "strata decode"))]


def main() -> None:
    log_off = os.path.getsize(LOG) if os.path.exists(LOG) else 0
    runs = []
    plan = [("warmup", 60, 16),
            ("p300a", 300, 64), ("p300b", 300, 64),
            ("p1000a", 1000, 64), ("p1000b", 1000, 64),
            ("p3000a", 3000, 64), ("p3000b", 3000, 64),
            ("d256a", 40, 256), ("d256b", 40, 256)]
    for name, pt, mt in plan:
        r = call(name, pt, mt)
        time.sleep(0.6)
        fresh, log_off = log_tail(log_off)
        r["engine"] = pick(fresh)
        runs.append(r)
        if "error" in r:
            print(f"{name:8s} {r['error']}", flush=True)
            continue
        print(f"{name:8s} read {r['read_n']:>5}(+{r['reused_n']} reused) @ {r['prompt_tps']:>7.1f} t/s | "
              f"gen {r['completion_n']:>4} tok @ {r['decode_tps'] or 0:>6.2f} t/s "
              f"({r['decode_ms_per_tok'] or 0:>5.2f} ms/tok) | wall {r['wall_s']:>6.1f}s", flush=True)
    with open(OUT, "w") as f:
        json.dump({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": runs}, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
