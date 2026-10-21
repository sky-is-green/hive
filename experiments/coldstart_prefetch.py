#!/usr/bin/env python3
"""Does a whole-file page-cache prefetch help the 176B's cold start?

The engine already warms itself: at startup it pre-fills the VRAM expert
cache *in profile order* (the experts ranked by routing frequency), which
is the targeted warm-up.  A whole-file ``posix_fadvise(WILLNEED)`` reads
the 60 GB file in *file* order into a 13 GB page cache - so the question is
whether that helps the first request or evicts the hot set the engine just
placed.

The test evicts the file's own pages between the arms with
``posix_fadvise(DONTNEED)`` (per-file, no root needed), so both arms start
from the same cold state:

  arm A: evict the file's pages, start the engine, time the first request
  arm B: evict the file's pages, prefetch the file, start the engine,
         time the first request

Usage: coldstart_prefetch.py [--arm a|b] [--out JSON]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STRATA = REPO.parent / "strata"
NATIVE = Path("/home/penis/models/scion-strata/Scion-FlashNext-176B-A6B-Q6K.gguf")
CONFIG = "strata-q6k-176b.json"
PORT = 8081
LIB = ctypes.CDLL("libc.so.6", use_errno=True)
LIB = ctypes.CDLL("libc.so.6", use_errno=True)
LIB.syscall.restype = ctypes.c_long
#: __NR_fadvise64 on x86_64 - glibc does not export posix_fadvise
NR_FADVISE64 = 221
DONTNEED, WILLNEED = 4, 3


def fadvise(path: Path, advice: int) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        rc = LIB.syscall(ctypes.c_long(NR_FADVISE64), ctypes.c_long(fd),
                         ctypes.c_long(0), ctypes.c_long(0), ctypes.c_long(advice))
        if rc != 0:
            raise OSError(ctypes.get_errno(), "fadvise64 failed")
    finally:
        os.close(fd)


def cached_share(path: Path, sample: int = 64) -> float:
    """mincore over a sample of the file's pages (the kernel's own answer)."""
    size = path.stat().st_size
    page = os.sysconf("SC_PAGE_SIZE")
    step = max(page, size // sample)
    resident = probed = 0
    fd = os.open(str(path), os.O_RDONLY)
    try:
        for off in range(0, size, step):
            offset = (off // page) * page
            if offset + page > size:
                break
            addr = LIB.mmap(None, page, 0x1, 0x1, fd, offset)
            if not addr or addr == ctypes.c_void_p(-1).value:
                break
            vec = (ctypes.c_ubyte * 1)()
            try:
                if LIB.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(page), vec, 1) == 0:
                    probed += 1
                    if vec[0] & 1:
                        resident += 1
            finally:
                LIB.munmap(ctypes.c_void_p(addr), page)
    finally:
        os.close(fd)
    return round(resident / max(1, probed), 4)


def stop_engine() -> None:
    """Stop the Strata serve and its engine (targeted PIDs, never a broad pkill)."""
    me = os.getpid()
    for pat in ("serve/server.py --engine strata", "serve --pack"):
        out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        for pid in out.stdout.split():
            try:
                pid_i = int(pid)
            except ValueError:
                continue
            if pid_i == me:
                continue
            try:
                os.kill(pid_i, 15)
            except OSError:
                pass
    time.sleep(6)


def start_engine(config: str = CONFIG) -> float:
    t0 = time.time()
    subprocess.Popen(
        [".venv/bin/python", "serve/server.py", "--engine", "strata",
         "--config", config, "--port", str(PORT)],
        cwd=str(STRATA), stdout=open("/tmp/opencode/coldstart-engine.log", "w"),
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    while time.time() - t0 < 900:
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("the engine did not come up in 15 minutes")


def first_request(prompt: str) -> dict:
    import urllib.request
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 256, "temperature": 0.0, "seed": 7}).encode()
    t0 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    el = time.time() - t0
    n = d.get("usage", {}).get("completion_tokens", 0)
    return {"s": round(el, 2), "tok": n, "tok_s": round(n / el, 2) if el else 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("a", "b"), required=True)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    task = json.loads((REPO / "experiments/cascade/tasks-organic-28.json").read_text())["tasks"][0]
    prompt = task["prompt"] + "\n\nSolve the problem, showing your work. End your answer with 'Final answer: <number>'."

    print(f"[arm {args.arm}] evicting the file's pages ...", flush=True)
    stop_engine()
    fadvise(NATIVE, DONTNEED)
    time.sleep(2)
    before = cached_share(NATIVE)
    print(f"[arm {args.arm}] the file is {before * 100:.1f}% cached before the start", flush=True)

    if args.arm == "b":
        print(f"[arm {args.arm}] prefetching the whole file ...", flush=True)
        t0 = time.time()
        fadvise(NATIVE, WILLNEED)
        print(f"[arm {args.arm}] the prefetch was asked for ({time.time() - t0:.1f}s)", flush=True)
        time.sleep(5)
        print(f"[arm {args.arm}] the file is {cached_share(NATIVE) * 100:.1f}% cached after it", flush=True)

    boot = start_engine(args.config)
    print(f"[arm {args.arm}] the engine came up in {boot:.1f}s", flush=True)

    reqs = [first_request(prompt)]
    for _ in range(2):
        reqs.append(first_request(prompt))

    out = {"arm": args.arm, "boot_s": round(boot, 1),
           "cached_before": before, "requests": reqs}
    print(json.dumps(out, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
