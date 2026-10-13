#!/usr/bin/env python3
"""Encode only the PLE table of a GGUF into the new block stream.

Local prebuild for the upload path: produces the table bytes that
``repack_experts_q2_0.py --table-bytes`` streams in on the pod, so the pod
never runs the CPU-heavy encoder.

usage: table_only_transcode.py <in.gguf> <out.bin> q3x32|q2x32|q4x160 [chunk_rows]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repack_experts_q2_0 as R  # noqa: E402


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    ttype = R.TABLE_TARGETS[sys.argv[3]]
    chunk_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 2_000_000

    idx = R.read_index(src)
    t = next(t for t in idx["infos"] if t.name == R.TABLE_NAME)
    per, size = R.BLOCKS[t.ttype]
    assert per == 32 and size == 18, f"{R.TABLE_NAME} is not Q4_0"
    n_rows = R.n_elements(t.dims) // 160
    row_bytes = 90

    print(f"{src.name}: {n_rows} rows -> {R.TYPE_NAMES[ttype]} ({dst.name})")
    done = 0
    t0 = time.time()
    with src.open("rb") as f, dst.open("wb") as o:
        f.seek(idx["data_start"] + t.offset)
        while done < n_rows:
            nr = min(chunk_rows, n_rows - done)
            data = f.read(nr * row_bytes)
            if len(data) != nr * row_bytes:
                raise SystemExit("short read on the table")
            o.write(R.transcode_q4_0_to_table(data, ttype))
            done += nr
            if done % (chunk_rows * 5) == 0 or done == n_rows:
                el = time.time() - t0
                print(f"  {done/1e6:.0f}M/{n_rows/1e6:.0f}M rows, "
                      f"{done/el/1e3:.0f}k rows/s, {el/60:.1f} min", flush=True)
    print(f"wrote {dst} ({dst.stat().st_size/1e9:.3f} GB)")


if __name__ == "__main__":
    main()
