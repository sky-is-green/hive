#!/usr/bin/env python3
"""Long-context check for the 262144 max-context window: one fresh ~16K-token prompt
(impossible under the old 4096 cap), 128-token decode, plus a follow-up turn on the
same conversation (exercises checkpoint reuse). Writes results_longctx.json.
"""
import json
import os
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8081/v1/chat/completions"
LOG = "/home/penis/Desktop/work/hivebench/logs/strata_q6k.engine.log"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_longctx.json")

SENT = ("The routing layer keeps the busiest experts resident on the cards while the rest wait in RAM. "
        "A prompt window borrows cache slots and hands them back; the profile ranks every expert pair. ")
LEAD = "Read the passage below and answer the question at the end. Passage #{nonce}:\n"
TAIL = "\n\nQuestion: In one sentence, what does the routing layer keep resident?\n"


def call(tag, prompt, max_tokens, messages=None):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    body = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        return {"name": tag, "error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"}
    wall = time.time() - t0
    u, t = resp.get("usage", {}), resp.get("timings", {})
    dec = t.get("predicted_per_second")
    txt = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
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
        "answer": (txt or "").strip()[:200],
    }


def log_from(offset):
    try:
        with open(LOG, "r", errors="replace") as f:
            f.seek(offset)
            return f.read().splitlines(), f.tell()
    except OSError:
        return [], offset


def pick(lines):
    return [ln for ln in lines if ln.startswith(("strata serve:", "strata mtp", "strata decode", "strata verify"))]


def main():
    log_off = os.path.getsize(LOG) if os.path.exists(LOG) else 0
    reps = 420  # ~16K tokens at ~38 tokens per sentence
    prompt = LEAD.format(nonce=time.time_ns() % 100000) + SENT * reps + TAIL
    runs = []

    r = call("p16k_fresh", prompt, 128)
    time.sleep(0.6)
    fresh, log_off = log_from(log_off)
    r["engine"] = pick(fresh)
    runs.append(r)
    print(f"{r['name']:12s} read {r.get('read_n')}(+{r.get('reused_n')} reused) @ {r.get('prompt_tps')} t/s | "
          f"gen {r.get('completion_n')} @ {r.get('decode_tps')} t/s | wall {r.get('wall_s')}s | {r.get('answer','')[:80]}",
          flush=True)

    # Follow-up on the same conversation: should reuse checkpoints (cache_n > 0).
    msgs = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": r.get("answer", "ok")},
            {"role": "user", "content": "And in one more sentence, where do the rest wait?"}]
    r2 = call("p16k_followup", None, 96, messages=msgs)
    time.sleep(0.6)
    fresh, log_off = log_from(log_off)
    r2["engine"] = pick(fresh)
    runs.append(r2)
    print(f"{r2['name']:12s} read {r2.get('read_n')}(+{r2.get('reused_n')} reused) @ {r2.get('prompt_tps')} t/s | "
          f"gen {r2.get('completion_n')} @ {r2.get('decode_tps')} t/s | wall {r2.get('wall_s')}s",
          flush=True)

    with open(OUT, "w") as f:
        json.dump({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": runs}, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
