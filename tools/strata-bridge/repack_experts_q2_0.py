#!/usr/bin/env python3
"""Repack Scion expert tensors (PTQ1_0 / PQ2_0) into Q2_0 for Strata.

Why this exists
---------------
Strata's native expert kernels and pack format take the legacy **Q2_0**
container (ggml type 42: 64-element blocks, one fp16 scale per block, 2-bit
codes LSB-first, symbol = code - 1 in {-1, 0, +1, +2}).  Scion releases store
the same ternary values in **PTQ1_0** (type 143, base-3 packed trits, block
128) or **PQ2_0** (type 142, 2-bit codes, block 128).  This tool rewrites only
the expert tensors into Q2_0 and streams everything else byte-for-byte, so it
never holds a whole tensor (or the 28.8 GB PLE table) in memory.

The transcode is **lossless at the decoded-value level**: both containers
store ``value = (code - 1) * fp16_scale`` for the same ternary codes; a
128-block becomes two 64-blocks with the scale duplicated.  PTQ1_0 trit
unpack reuses ``scion/moe/ptq1_0.py`` (the tested 1:1 transcription of the
C++ ``dequantize_row_ptq1_0``), so the encoder sees exactly the shipped
trits.

Usage
-----
    # census + plan, no writes
    python repack_experts_q2_0.py --in Q6K.gguf --dry-run

    # full rewrite + sampled bit-equality verification
    python repack_experts_q2_0.py --in Q6K.gguf --out Q2_0.gguf

    # codec self-test (synthetic trits through pack -> repack -> decode)
    python repack_experts_q2_0.py --self-test

Run with a python that has torch (the scion codec uses it), e.g.
``work/.venv-rocm/bin/python``.  ``--scion-moe`` defaults to the sibling
checkout ``~/Desktop/work/scion/moe``.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# ggml type table (only what this file needs to size tensors)
# ---------------------------------------------------------------------------
TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 40: "NVFP4",
    41: "Q1_0", 42: "Q2_0", 142: "PQ2_0", 143: "PTQ1_0",
}

# type id -> (elements per block, bytes per block)
BLOCKS = {
    0: (1, 4), 1: (1, 2), 30: (1, 2),
    2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36),
    10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
    14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50),
    20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
    29: (256, 56), 34: (256, 54), 35: (256, 66),
    41: (128, 18), 42: (64, 18), 142: (128, 34), 143: (128, 28),
}

EXPERT_SOURCES = (143, 142)  # PTQ1_0, PQ2_0 -> transcoded
TARGET = 42                  # Q2_0

# Strata compatibility conversions (--strata-compat).  The engine's embedding
# path takes only `is_iq(t) || BF16` (Q6_K is not in it) and its PLE table
# reader takes only IQ4_NL / Q5_0 / FP8 (Q4_0 is not).  Both conversions are
# validated on real tensors: Q4_0 -> Q5_0 is exactly lossless (the Q4_0 grid
# is a subset of Q5_0's), Q6_K -> Q8_0 is a 2.4e-4 max-abs re-round.
CONVERT = {
    "token_embd.weight": (14, 8),             # Q6_K -> Q8_0
    "per_layer_token_embd.weight": (2, 6),    # Q4_0 -> Q5_0
}


def n_elements(dims: tuple[int, ...]) -> int:
    out = 1
    for d in dims:
        out *= int(d)
    return out


def tensor_nbytes(ttype: int, dims: tuple[int, ...]) -> int:
    elems = n_elements(dims)
    if ttype in BLOCKS:
        per, size = BLOCKS[ttype]
        if elems % per:
            raise ValueError(f"type {TYPE_NAMES.get(ttype, ttype)}: {elems} "
                             f"elements not a multiple of block {per}")
        return elems // per * size
    raise ValueError(f"unsupported tensor type {ttype}")


# ---------------------------------------------------------------------------
# GGUF index (parse header + kv + tensor infos; data is streamed later)
# ---------------------------------------------------------------------------
@dataclass
class Tensor:
    name: str
    dims: tuple[int, ...]
    ttype: int
    offset: int
    index: int = 0
    new_type: int = 0
    new_offset: int = 0
    new_size: int = 0
    old_size: int = 0
    transcoded: bool = False
    converted: bool = False


_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read(f, fmt: str):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def _skip_kv_value(f, vtype: int) -> None:
    if vtype in _SZ:
        f.read(_SZ[vtype])
    elif vtype == 8:                      # string
        (n,) = _read(f, "<Q")
        f.read(n)
    elif vtype == 9:                      # array
        (et,) = _read(f, "<I")
        (n,) = _read(f, "<Q")
        if et in _SZ:
            f.read(_SZ[et] * n)
        elif et == 8:
            for _ in range(n):
                (l,) = _read(f, "<Q")
                f.read(l)
        else:
            raise ValueError(f"unsupported array element type {et}")
    else:
        raise ValueError(f"unsupported kv value type {vtype}")


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def read_index(path: Path):
    """Parse header/kv/tensor infos.  Returns a dict; no tensor data read."""
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"not a GGUF file: {path}")
        (version,) = _read(f, "<I")
        (n_tensors,) = _read(f, "<Q")
        (n_kv,) = _read(f, "<Q")
        kv_start = f.tell()
        alignment = 32
        for _ in range(n_kv):
            (klen,) = _read(f, "<Q")
            key = f.read(klen)
            (vtype,) = _read(f, "<I")
            if key == b"general.alignment" and vtype == 4:
                (alignment,) = _read(f, "<I")
            else:
                _skip_kv_value(f, vtype)
        kv_end = f.tell()
        infos: list[Tensor] = []
        for i in range(n_tensors):
            (nlen,) = _read(f, "<Q")
            name = f.read(nlen).decode("utf-8")
            (nd,) = _read(f, "<I")
            dims = tuple(_read(f, f"<{nd}Q"))
            (ttype,) = _read(f, "<I")
            (off,) = _read(f, "<Q")
            infos.append(Tensor(name=name, dims=dims, ttype=ttype,
                                offset=off, index=i))
        header_end = f.tell()
        f.seek(kv_start)
        kv_bytes = f.read(kv_end - kv_start)
    return {
        "version": version,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
        "kv_bytes": kv_bytes,
        "alignment": alignment,
        "data_start": align_up(header_end, alignment),
        "infos": infos,
        "file_size": path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# codecs
# ---------------------------------------------------------------------------
def scion_unpack(scion_moe: Path):
    sys.path.insert(0, str(scion_moe))
    from ptq1_0 import unpack_trits_ptq1_0  # noqa: E402
    return unpack_trits_ptq1_0


def transcode_ptq1_0(chunk: bytes, unpack, torch) -> bytes:
    """PTQ1_0 block stream (28 B / 128 elems) -> Q2_0 block stream (36 B)."""
    n_blocks = len(chunk) // 28
    raw = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
    raw = raw[: n_blocks * 28].reshape(n_blocks, 28)
    trits = unpack(raw[:, 0:24], raw[:, 24:26])          # int8 [n, 128]
    codes = (trits.to(torch.int64) + 1).to(torch.uint8).reshape(n_blocks, 2, 64)
    c = codes.reshape(n_blocks, 2, 16, 4).cpu().numpy()
    packed = (c[:, :, :, 0] | (c[:, :, :, 1] << 2)
              | (c[:, :, :, 2] << 4) | (c[:, :, :, 3] << 6))
    scale = raw[:, 26:28].cpu().numpy().reshape(n_blocks, 1, 2)
    out = np.empty((n_blocks, 2, 18), dtype=np.uint8)
    out[:, :, 0:2] = scale
    out[:, :, 2:18] = packed
    return out.tobytes()


def transcode_pq2_0(chunk: bytes) -> bytes:
    """PQ2_0 block stream (34 B / 128 elems) -> Q2_0 block stream (36 B).

    Codes are already 2-bit LSB-first in both containers; a 128-block splits
    into two 64-blocks with the fp16 scale duplicated.
    """
    n_blocks = len(chunk) // 34
    raw = np.frombuffer(chunk, dtype=np.uint8)[: n_blocks * 34].reshape(n_blocks, 34)
    out = np.empty((n_blocks, 2, 18), dtype=np.uint8)
    out[:, :, 0:2] = raw[:, 32:34].reshape(n_blocks, 1, 2)
    out[:, :, 2:18] = raw[:, 0:32].reshape(n_blocks, 2, 16)
    return out.tobytes()


def decode_q2_0(chunk: bytes) -> np.ndarray:
    n_blocks = len(chunk) // 18
    raw = np.frombuffer(chunk, dtype=np.uint8)[: n_blocks * 18].reshape(n_blocks, 18)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    codes = raw[:, 2:18].reshape(n_blocks, 16, 1) >> (2 * np.arange(4, dtype=np.uint8))
    codes = (codes & 3).reshape(n_blocks, 64).astype(np.int16) - 1
    return codes.astype(np.float32) * d


def decode_pq2_0(chunk: bytes) -> np.ndarray:
    n_blocks = len(chunk) // 34
    raw = np.frombuffer(chunk, dtype=np.uint8)[: n_blocks * 34].reshape(n_blocks, 34)
    d = raw[:, 32:34].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    codes = raw[:, 0:32].reshape(n_blocks, 32, 1) >> (2 * np.arange(4, dtype=np.uint8))
    codes = (codes & 3).reshape(n_blocks, 128).astype(np.int16) - 1
    return codes.astype(np.float32) * d


def gguf_py_tools(path: Path):
    sys.path.insert(0, str(path))
    from gguf.quants import quantize, dequantize  # noqa: E402
    from gguf.constants import GGMLQuantizationType as QT  # noqa: E402
    return quantize, dequantize, QT


def convert_q4_0_to_q5_0_fast(chunk: bytes) -> bytes:
    """Q4_0 -> Q5_0, exact and bit-level (no float round trip).

    Q4_0 stores ``value = (q4 - 8) * d``; Q5_0 stores ``(q5 - 16) * d5``.
    With ``d5 = d4`` and ``q5 = q4 + 8`` the decoded values are identical
    (Q4_0's grid is a subset of Q5_0's), so only the high-bit plane is new:
    ``qh`` bit k is the high bit of element k, and the nibble order matches
    Q4_0's (low nibbles = elements 0..15, high nibbles = 16..31).
    """
    nb = len(chunk) // 18
    raw = np.frombuffer(chunk, dtype=np.uint8)[: nb * 18].reshape(nb, 18)
    qs = raw[:, 2:18]
    bits = np.concatenate([(qs & 0x0F) >= 8, (qs >> 4) >= 8], axis=1)  # [nb,32]
    qh = np.packbits(bits, axis=1, bitorder="little")                  # [nb,4]
    out = np.empty((nb, 22), dtype=np.uint8)
    out[:, 0:2] = raw[:, 0:2]
    out[:, 2:6] = qh
    out[:, 6:22] = qs ^ np.uint8(0x88)   # q5 = q4 + 8: each nibble flips bit 3
    return out.tobytes()


def convert_chunk(src_type: int, dst_type: int, chunk: bytes,
                  quantize, dequantize, QT) -> tuple[bytes, float]:
    """One chunk of a --strata-compat conversion; returns (bytes, max|err|)."""
    if src_type == 14 and dst_type == 8:          # Q6_K -> Q8_0
        nb = len(chunk) // 210
        raw = np.frombuffer(chunk, dtype=np.uint8)[: nb * 210].reshape(nb, 210)
        f = dequantize(raw, QT.Q6_K)
        b = quantize(f, QT.Q8_0)
        err = float(np.abs(dequantize(b, QT.Q8_0) - f).max())
        return b.tobytes(), err
    if src_type == 2 and dst_type == 6:           # Q4_0 -> Q5_0 (exact)
        return convert_q4_0_to_q5_0_fast(chunk), 0.0
    raise ValueError(f"no converter for {src_type} -> {dst_type}")


def decode_ptq1_0(chunk: bytes, unpack, torch) -> np.ndarray:
    n_blocks = len(chunk) // 28
    raw = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)[: n_blocks * 28]
    raw = raw.reshape(n_blocks, 28)
    trits = unpack(raw[:, 0:24], raw[:, 24:26]).to(torch.int16)
    scale = raw[:, 26:28].contiguous().view(torch.float16).to(torch.float32)
    return trits.cpu().numpy().astype(np.float32) * scale.cpu().numpy().reshape(n_blocks, 1)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def self_test(scion_moe: Path) -> None:
    import torch
    unpack = scion_unpack(scion_moe)
    from ptq1_0 import pack_trits_ptq1_0

    rng = np.random.default_rng(0)
    for trial in range(3):
        n = 257
        trits = rng.integers(-1, 2, size=(n, 128)).astype(np.int8)
        scales = (rng.random(n).astype(np.float32) * 0.02 + 0.001).astype(np.float16)
        qs, qh = pack_trits_ptq1_0(torch.from_numpy(trits))
        d16 = torch.from_numpy(scales).view(torch.uint8).reshape(n, 2)
        block = torch.cat([qs, qh, d16], dim=-1).contiguous().numpy().tobytes()
        out = transcode_ptq1_0(block, unpack, torch)
        assert len(out) == n * 36, (len(out), n * 36)
        ref = decode_ptq1_0(block, unpack, torch)
        got = decode_q2_0(out).reshape(ref.shape)
        assert np.array_equal(ref, got), f"trial {trial}: decoded values differ"
    print("self-test: PTQ1_0 -> Q2_0 decoded values bit-equal on 3x257 blocks")

    n = 129
    codes = rng.integers(0, 3, size=(n, 32)).astype(np.uint8)
    scales = (rng.random(n).astype(np.float32) * 0.02 + 0.001).astype(np.float16)
    pq = np.concatenate([codes, scales.view(np.uint8).reshape(n, 2)], axis=1)
    out = transcode_pq2_0(pq.tobytes())
    ref = decode_pq2_0(pq.tobytes())
    got = decode_q2_0(out).reshape(ref.shape)
    assert np.array_equal(ref, got), "PQ2_0 path: decoded values differ"
    print("self-test: PQ2_0 -> Q2_0 decoded values bit-equal on 1x129 blocks")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", default="")
    ap.add_argument("--out", dest="dst", default="")
    ap.add_argument("--scion-moe", default=os.path.expanduser("~/Desktop/work/scion/moe"))
    ap.add_argument("--gguf-py", default=os.path.expanduser(
        "~/Desktop/work/llama-qwen4exp/gguf-py"))
    ap.add_argument("--strata-compat", action="store_true",
                    help="also convert token_embd Q6_K->Q8_0 and the PLE table "
                         "Q4_0->Q5_0 (the engine's embedding/table paths)")
    ap.add_argument("--keep-table", action="store_true",
                    help="with --strata-compat: leave the PLE table as-is "
                         "(for engines that read Q4_0 tables natively)")
    ap.add_argument("--chunk-mb", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--verify-tensors", type=int, default=4)
    args = ap.parse_args()

    scion_moe = Path(args.scion_moe)
    if args.self_test:
        self_test(scion_moe)
        return
    if not args.src:
        raise SystemExit("--in is required (or --self-test)")
    src = Path(args.src)
    if not src.is_file():
        raise SystemExit(f"missing input: {src}")

    idx = read_index(src)
    infos: list[Tensor] = idx["infos"]
    alignment: int = idx["alignment"]
    old_data_start: int = idx["data_start"]
    print(f"{src.name}: GGUF v{idx['version']}, {len(infos)} tensors, "
          f"{len(idx['kv_bytes'])} B kv, alignment {alignment}, "
          f"data at {old_data_start}")

    counts: dict[int, list[Tensor]] = {}
    for t in infos:
        t.old_size = tensor_nbytes(t.ttype, t.dims)
        counts.setdefault(t.ttype, []).append(t)

    print("type census:")
    for ttype in sorted(counts):
        ts = counts[ttype]
        total = sum(t.old_size for t in ts)
        print(f"  {TYPE_NAMES.get(ttype, ttype):8s} id={ttype:3d}  "
              f"{len(ts):4d} tensors  {total/1e9:8.3f} GB")

    order = sorted(infos, key=lambda t: t.offset)
    cursor = 0
    for t in order:
        if t.offset != cursor:
            print(f"  note: offset gap at {t.name}: {t.offset} vs {cursor}")
        cursor = t.offset + t.old_size
    tail = idx["file_size"] - old_data_start - cursor
    print(f"data: {cursor/1e9:.3f} GB, tail padding {tail} B")

    cum = 0
    for t in order:
        conv = CONVERT.get(t.name) if args.strata_compat else None
        if conv and args.keep_table and t.name == "per_layer_token_embd.weight":
            conv = None
        if conv and t.ttype == conv[0]:
            t.new_type = conv[1]
            t.converted = True
        elif t.ttype in EXPERT_SOURCES:
            t.new_type = TARGET
            t.transcoded = True
        else:
            t.new_type = t.ttype
        per, size = BLOCKS[t.new_type]
        t.new_size = n_elements(t.dims) // per * size
        t.new_offset = align_up(cum, alignment)
        cum = t.new_offset + t.new_size
    n_experts = sum(1 for t in order if t.transcoded)
    n_conv = sum(1 for t in order if t.converted)
    print(f"expert tensors: {n_experts}; strata-compat conversions: {n_conv}; "
          f"data {cursor/1e9:.3f} -> {cum/1e9:.3f} GB")
    for t in order:
        if t.converted:
            print(f"  convert {t.name}: {TYPE_NAMES[t.ttype]} -> "
                  f"{TYPE_NAMES[t.new_type]}  {t.old_size/1e9:.3f} -> "
                  f"{t.new_size/1e9:.3f} GB")

    if args.dry_run:
        for t in order:
            if t.transcoded:
                print(f"  {t.name}: {TYPE_NAMES[t.ttype]} -> Q2_0  "
                      f"{t.old_size/1e6:.1f} -> {t.new_size/1e6:.1f} MB")
            elif t.converted:
                print(f"  {t.name}: {TYPE_NAMES[t.ttype]} -> "
                      f"{TYPE_NAMES[t.new_type]}  {t.old_size/1e9:.3f} -> "
                      f"{t.new_size/1e9:.3f} GB")
        print("dry run: nothing written")
        return
    if not args.dst:
        raise SystemExit("--out is required unless --dry-run")
    dst = Path(args.dst)
    tmp = dst.with_suffix(dst.suffix + ".part")

    unpack = scion_unpack(scion_moe)
    import torch

    quantize = dequantize = QT = None
    if args.strata_compat:
        quantize, dequantize, QT = gguf_py_tools(Path(args.gguf_py))

    chunk_in = args.chunk_mb * 1024 * 1024
    new_data_start = 0
    with src.open("rb") as fin, tmp.open("wb") as out:
        out.write(b"GGUF" + struct.pack("<IQQ", idx["version"], len(infos), idx["n_kv"]))
        out.write(idx["kv_bytes"])
        for t in infos:                       # info table in original order
            nb = t.name.encode("utf-8")
            out.write(struct.pack("<Q", len(nb)) + nb)
            out.write(struct.pack("<I", len(t.dims)))
            for d in t.dims:
                out.write(struct.pack("<Q", d))
            out.write(struct.pack("<IQ", t.new_type, t.new_offset))
        pos = out.tell()
        out.write(b"\0" * (align_up(pos, alignment) - pos))
        new_data_start = out.tell()

        for i, t in enumerate(order):
            pad = new_data_start + t.new_offset - out.tell()
            if pad < 0:
                raise SystemExit(f"offset overlap at {t.name}")
            out.write(b"\0" * pad)
            fin.seek(old_data_start + t.offset)
            if t.transcoded:
                src_block = 28 if t.ttype == 143 else 34
                remaining = t.old_size
                while remaining > 0:
                    n = min(chunk_in // src_block * src_block, remaining)
                    data = fin.read(n)
                    if len(data) != n:
                        raise SystemExit(f"short read on {t.name}")
                    out.write(transcode_ptq1_0(data, unpack, torch) if t.ttype == 143
                              else transcode_pq2_0(data))
                    remaining -= n
            elif t.converted:
                src_block = BLOCKS[t.ttype][1]
                remaining = t.old_size
                max_err = 0.0
                while remaining > 0:
                    n = min(chunk_in // src_block * src_block, remaining)
                    data = fin.read(n)
                    if len(data) != n:
                        raise SystemExit(f"short read on {t.name}")
                    payload, err = convert_chunk(t.ttype, t.new_type, data,
                                                 quantize, dequantize, QT)
                    out.write(payload)
                    max_err = max(max_err, err)
                    remaining -= n
                print(f"[{i+1}/{len(order)}] {t.name}: "
                      f"{TYPE_NAMES[t.ttype]} -> {TYPE_NAMES[t.new_type]}, "
                      f"max|err| {max_err:.3e}", flush=True)
            else:
                remaining = t.old_size
                while remaining > 0:
                    n = min(chunk_in, remaining)
                    data = fin.read(n)
                    if len(data) != n:
                        raise SystemExit(f"short read on {t.name}")
                    out.write(data)
                    remaining -= n
            if t.transcoded:
                print(f"[{i+1}/{len(order)}] {t.name}: "
                      f"{t.old_size/1e6:.1f} -> {t.new_size/1e6:.1f} MB "
                      f"({out.tell()/1e9:.1f} GB written)", flush=True)
            elif i % 100 == 0:
                print(f"[{i+1}/{len(order)}] {t.name} "
                      f"({out.tell()/1e9:.1f} GB written)", flush=True)
        pos = out.tell()
        out.write(b"\0" * (align_up(pos, alignment) - pos))
    os.replace(tmp, dst)
    print(f"wrote {dst} ({dst.stat().st_size/1e9:.3f} GB, "
          f"data {new_data_start}..{new_data_start + cum})")

    # sampled bit-equality verification on expert tensors
    picks = [t for t in order if t.transcoded]
    if picks and args.verify_tensors > 0:
        step = max(1, len(picks) // args.verify_tensors)
        sample = picks[::step][:args.verify_tensors]
        with src.open("rb") as fin, dst.open("rb") as fout:
            for t in sample:
                src_block = 28 if t.ttype == 143 else 34
                nbytes = min(4 * 1024 * 1024, t.old_size)
                nbytes = nbytes // src_block * src_block
                fin.seek(old_data_start + t.offset)
                inb = fin.read(nbytes)
                fout.seek(new_data_start + t.new_offset)
                outb = fout.read(nbytes // src_block * 36)
                ref = (decode_ptq1_0(inb, unpack, torch) if t.ttype == 143
                       else decode_pq2_0(inb))
                got = decode_q2_0(outb).reshape(ref.shape)
                ok = np.array_equal(ref, got)
                print(f"verify {t.name}: {nbytes/1e6:.1f} MB in, "
                      f"{len(outb)/1e6:.1f} MB out, "
                      f"{'bit-equal' if ok else 'MISMATCH'}")
                if not ok:
                    raise SystemExit("verification failed")
    print("done")


if __name__ == "__main__":
    main()
