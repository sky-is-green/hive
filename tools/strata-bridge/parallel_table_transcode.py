#!/usr/bin/env python3
"""Parallel PLE table transcode: worker processes encode row ranges.

Each worker writes its output slice at ``start_row * row_out`` in a
preallocated file, so the result is byte-identical to the sequential
``table_only_transcode.py`` run.

usage: parallel_table_transcode.py <in.gguf> <out.bin> q3x32|q2x32|q4x160 [procs]
"""
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repack_experts_q2_0 as R  # noqa: E402


def worker(job):
    src, dst, ttype, base, start_row, end_row, chunk_rows = job
    row_in = 90
    per_out, size_out = R.BLOCKS[ttype]
    row_out = size_out * (160 // per_out)
    done = start_row
    with open(src, "rb") as f, open(dst, "r+b") as o:
        f.seek(base + start_row * row_in)
        o.seek(start_row * row_out)
        while done < end_row:
            nr = min(chunk_rows, end_row - done)
            data = f.read(nr * row_in)
            if len(data) != nr * row_in:
                raise SystemExit(f"short read at row {done}")
            o.write(R.transcode_q4_0_to_table(data, ttype))
            done += nr
    return end_row - start_row


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    ttype = R.TABLE_TARGETS[sys.argv[3]]
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else min(16, os.cpu_count() or 8)

    idx = R.read_index(src)
    t = next(t for t in idx["infos"] if t.name == R.TABLE_NAME)
    n_rows = R.n_elements(t.dims) // 160
    base = idx["data_start"] + t.offset
    per_out, size_out = R.BLOCKS[ttype]
    row_out = size_out * (160 // per_out)

    with open(dst, "wb") as f:
        f.truncate(n_rows * row_out)
    per = (n_rows + procs - 1) // procs
    jobs = [(str(src), str(dst), ttype, base, i * per, min((i + 1) * per, n_rows), 200_000)
            for i in range(procs) if i * per < n_rows]
    print(f"{src.name}: {n_rows} rows -> {R.TYPE_NAMES[ttype]}, {procs} workers", flush=True)

    t0 = time.time()
    total = 0
    with mp.Pool(len(jobs)) as pool:
        for k, done in enumerate(pool.imap_unordered(worker, jobs), 1):
            total += done
            el = time.time() - t0
            print(f"  worker {k}/{len(jobs)} done, {total/1e6:.0f}M rows, "
                  f"{total/el/1e3:.0f}k rows/s agg, {el/60:.1f} min", flush=True)
    print(f"wrote {dst} ({dst.stat().st_size/1e9:.3f} GB)")


if __name__ == "__main__":
    main()
