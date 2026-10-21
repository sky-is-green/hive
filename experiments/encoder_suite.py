"""Which encoder should the harness use?  A measurement of the three jobs it does.

The B-avenue measurement (2026-08-23) compared encoders on *ranking*: given
a query, which stored chunk is the relevant one.  That is one of the three
jobs the harness gives its encoder, and the least important of them:

``rank``   the sieve scores every chunk against the query and keeps the
           best ones (this is what the prior measurement measured)
``dedup``  the membrane drops a chunk that says what another chunk says
           (cosine > 0.92) and keeps the denser one
``drift``  the membrane notices that the conversation changed topic
           (1 - cosine(recent, older) > 0.6) and penalises the old chunks

``dedup`` and ``drift`` are *decisions*, not rankings: the encoder is asked
"are these the same?" and "is this the same topic?".  A cosine that
separates relevant from irrelevant chunks says nothing about either.

This module measures all three on the harness's own traffic - the
conversations the Studio and the sibling actually served - with labels
that are deterministic and independent of the encoder being measured:

  dedup  a pair is a duplicate iff the two chunks share most of their
         content words (the same fact, said twice)
  drift  a turn is a topic change iff the recent window and the older
         half of the store share almost no content words

Usage::

    python -m experiments.encoder_suite --encoders l3 l6 bge-small
    python -m experiments.encoder_suite --encoders l3 --out /tmp/suite.json

The result is a table: one row per encoder, the dedup and drift quality,
the ranking quality, and what the encoder costs (size, dimensions, the
time to embed a turn's worth of chunks).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

#: the harness's operating points (the defaults it ships with)
DEDUP_THRESHOLD = 0.92
DRIFT_THRESHOLD = 0.6

#: the lexical labels' operating points
DUP_JACCARD = 0.5       # a duplicate: the same content words
DRIFT_JACCARD = 0.05    # a topic change: almost no shared content words

#: The text handed to the encoder is capped.  The encoders truncate at
#: 256 tokens anyway, so the cap changes nothing that is measured - but
#: the drift detector's own input is the older half of a whole
#: conversation, which is megabytes of text per turn, and building those
#: strings for thousands of cases costs gigabytes of RAM.  (That the hot
#: path embeds the whole older half is a finding in its own right: the
#: drift check's cost grows with the conversation.)
MAX_EMBED_CHARS = 8000

WORD_RE = re.compile(r"[a-z0-9]{4,}")
STOPWORDS = {
    "about", "after", "again", "against", "because", "been", "before",
    "being", "below", "between", "both", "cannot", "could", "does", "doing",
    "down", "during", "each", "from", "further", "have", "having", "here",
    "into", "just", "more", "most", "once", "only", "other", "over", "same",
    "should", "some", "such", "than", "that", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "under", "until",
    "very", "were", "what", "when", "where", "which", "while", "will",
    "with", "would", "your", "also", "been", "were", "will", "shall",
    "using", "used", "use", "one", "two", "three", "get", "got", "make",
    "made", "way", "like", "need", "know", "see", "see", "yes", "yeah",
}


def content_terms(text: str) -> set[str]:
    """The distinctive words of a text (the lexical ground truth's atoms)."""
    return {w for w in WORD_RE.findall((text or "").lower())
            if w not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# The corpus: the harness's own conversations
# ---------------------------------------------------------------------------
def load_conversations(dirs: Iterable[str]) -> list[dict]:
    """The conversations the Studio and the sibling actually served.

    Each conversation is ``{"id": str, "chunks": [{"turn": int, "text": str}]}``
    in the order they were stored.
    """
    out: list[dict] = []
    for raw in dirs:
        base = Path(raw).expanduser()
        if not base.is_dir():
            print(f"  (no such directory: {base})", file=sys.stderr)
            continue
        files = sorted(base.glob("conv-*.json"))
        for path in files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            chunks = []
            for c in (data.get("store") or {}).get("chunks") or []:
                text = (c.get("content") or "").strip()
                if text:
                    chunks.append({"turn": int(c.get("turn") or 0), "text": text})
            if len(chunks) >= 2:
                out.append({"id": path.stem, "chunks": chunks})
    return out


# ---------------------------------------------------------------------------
# Task 1: the dedup decision
# ---------------------------------------------------------------------------
def build_dedup_pairs(conversations: list[dict], per_conv: int,
                      seed: int = 20261020) -> list[dict]:
    """The pairs the membrane's duplicate test sees, with lexical labels.

    A pair is a duplicate iff the two chunks share most of their content
    words - the same fact, said twice.  The label is lexical on purpose:
    it is independent of the encoder being measured.
    """
    rng = random.Random(seed)
    pairs: list[dict] = []
    for conv in conversations:
        chunks = conv["chunks"]
        terms = [content_terms(c["text"]) for c in chunks]
        n = len(chunks)
        # all the pairs, scored lexically (a conversation is at most a few
        # hundred chunks, so the pair matrix is cheap)
        scored: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                jac = jaccard(terms[i], terms[j])
                if jac >= DUP_JACCARD:
                    scored.append((jac, i, j))
        # the positives: the pairs the lexical prefilter found, capped
        rng.shuffle(scored)
        pos = scored[:per_conv]
        # the negatives: random pairs that are not duplicates
        neg: list[tuple[float, int, int]] = []
        tries = 0
        while len(neg) < per_conv and tries < per_conv * 40 and n > 1:
            tries += 1
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            jac = jaccard(terms[i], terms[j])
            if jac < DUP_JACCARD:
                neg.append((jac, min(i, j), max(i, j)))
        for jac, i, j in pos + neg:
            pairs.append({
                "conversation_id": conv["id"],
                "a": chunks[i]["text"],
                "b": chunks[j]["text"],
                "duplicate": bool(jac >= DUP_JACCARD),
                "jaccard": round(jac, 4),
            })
    return pairs


# ---------------------------------------------------------------------------
# Task 2: the drift decision
# ---------------------------------------------------------------------------
def build_drift_cases(conversations: list[dict], window: int = 3,
                      min_history: int = 5) -> list[dict]:
    """The turns the membrane's topic-change test sees, with lexical labels.

    A turn is a topic change iff the recent window and the older half of
    the store share almost no content words.  The signal the harness
    computes is 1 - cosine(recent_text, older_text).
    """
    cases: list[dict] = []
    for conv in conversations:
        chunks = conv["chunks"]
        if len(chunks) < min_history + window:
            continue
        terms = [content_terms(c["text"]) for c in chunks]
        older_half = chunks[:len(chunks) // 2]
        older_terms = set().union(*terms[:len(chunks) // 2]) if older_half else set()
        for cut in range(min_history, len(chunks) - window + 1):
            recent = chunks[cut:cut + window]
            if not older_half:
                continue
            jac = jaccard(set().union(*terms[cut:cut + window]), older_terms)
            cases.append({
                "conversation_id": conv["id"],
                "recent": " ".join(c["text"] for c in recent)[:MAX_EMBED_CHARS],
                "older": " ".join(c["text"] for c in older_half)[:MAX_EMBED_CHARS],
                "drift": bool(jac < DRIFT_JACCARD),
                "jaccard": round(jac, 4),
            })
    return cases


# ---------------------------------------------------------------------------
# The encoders
# ---------------------------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(a @ b / denom)


def make_encoder(name: str) -> tuple[Callable[[list[str]], np.ndarray], dict]:
    """An encoder: a function texts -> vectors, and its description.

    The models are used the way the harness uses them: **one** embedding
    function for both sides (the query and the chunk).  The asymmetric
    retrieval models (e5, bge, nomic, gte, qwen3) want a prefix on the
    query side only; the harness has no notion of a side, so they are
    measured as the harness would use them (no prefix) and, where it
    matters, with the prefix as well.
    """
    if name == "l3":
        return _sentence_transformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
    if name == "l6":
        return _sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
    if name == "bge-small":
        return _sentence_transformer("BAAI/bge-small-en-v1.5")
    if name == "e5-small":
        return _sentence_transformer("intfloat/e5-small-v2")
    if name == "gte-small":
        return _sentence_transformer("thenlper/gte-small")
    if name == "nomic":
        return _sentence_transformer("nomic-ai/nomic-embed-text-v1.5")
    if name == "bge-m3":
        return _sentence_transformer("BAAI/bge-m3")
    if name == "qwen3-0.6b":
        return _sentence_transformer("Qwen/Qwen3-Embedding-0.6B")
    if name == "qwen3-0.6b-p":
        # the model's own recommendation: a prefix on the query side
        return _sentence_transformer("Qwen/Qwen3-Embedding-0.6B",
                                     query_prefix="query: ")
    if name == "qwen3-0.6b-p":
        return _sentence_transformer("Qwen/Qwen3-Embedding-0.6B",
                                     query_prefix="query: ")
    if name == "e5-small-p":
        return _sentence_transformer("intfloat/e5-small-v2",
                                     query_prefix="query: ")
    if name.startswith("checkpoint:"):
        # a trained SentenceTransformer checkpoint
        return _sentence_transformer(name.split(":", 1)[1])
    if name.startswith("qwen-student:"):
        # our own Qwen-family bi-encoder (experiments/qwen_encoder.py): a
        # Qwen3 decoder with the Qwen3-Embedding pooling (last token), unit
        # length, measured the way the harness uses every encoder - one
        # function for both sides, on the CPU, like the rest.
        from experiments.qwen_encoder import student_encoder
        return student_encoder(name.split(":", 1)[1])
    if "/" in name or name.endswith(".gguf"):
        # any sentence-transformers id (a local path counts)
        return _sentence_transformer(name)
    raise SystemExit(f"unknown encoder: {name!r}")


def _sentence_transformer(model_id: str, query_prefix: str = ""
                          ) -> tuple[Callable[[list[str]], np.ndarray], dict]:
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(model_id, device="cpu")
    model.eval()
    load_s = time.time() - t0
    dim = int(model.get_embedding_dimension() or 0)
    params = sum(p.numel() for p in model.parameters())

    def embed_many(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, dim), dtype=float)
        vecs = model.encode(list(texts), normalize_embeddings=True,
                             batch_size=16, show_progress_bar=False)
        return np.asarray(vecs, dtype=float)

    return embed_many, {
        "model": model_id,
        "params": params,
        "params_m": round(params / 1e6, 1),
        "dim": dim,
        "load_s": round(load_s, 1),
        "query_prefix": query_prefix,
    }


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------
def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def measure_dedup(pairs: list[dict], embed) -> dict:
    """The duplicate test at the harness's threshold, and the best one."""
    texts = [p["a"] for p in pairs] + [p["b"] for p in pairs]
    vectors = embed_many(embed, texts)
    n = len(pairs)
    sims = [_cosine(vectors[i], vectors[n + i]) for i in range(n)]
    labels = [p["duplicate"] for p in pairs]
    out: dict = {"n": len(pairs), "positives": sum(1 for l in labels if l)}
    for name, thresh in (("at_0.92", DEDUP_THRESHOLD),):
        tp = sum(1 for s, l in zip(sims, labels) if l and s > thresh)
        fp = sum(1 for s, l in zip(sims, labels) if not l and s > thresh)
        fn = sum(1 for s, l in zip(sims, labels) if l and s <= thresh)
        out[name] = _prf(tp, fp, fn)
    # the separation: how far the two populations are apart
    pos_s = [s for s, l in zip(sims, labels) if l]
    neg_s = [s for s, l in zip(sims, labels) if not l]
    out["pos_p50"] = round(statistics.median(pos_s), 4) if pos_s else None
    out["neg_p50"] = round(statistics.median(neg_s), 4) if neg_s else None
    out["separation"] = (round(out["pos_p50"] - out["neg_p50"], 4)
                         if pos_s and neg_s else None)
    # the best threshold (the oracle the harness cannot have)
    best = {"f1": 0.0}
    for thresh in np.arange(0.30, 0.99, 0.01):
        tp = sum(1 for s, l in zip(sims, labels) if l and s > thresh)
        fp = sum(1 for s, l in zip(sims, labels) if not l and s > thresh)
        fn = sum(1 for s, l in zip(sims, labels) if l and s <= thresh)
        m = _prf(tp, fp, fn)
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "precision": m["precision"],
                    "recall": m["recall"], "threshold": round(float(thresh), 2)}
    out["best"] = best
    return out


def _auc(scores: list[float], labels: list[bool]) -> float:
    """The ROC AUC (the rank-based one: ties count as half)."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def measure_drift(cases: list[dict], embed) -> dict:
    """The topic-change test: the AUC, and the accuracy at 0.6."""
    texts = [c["recent"] for c in cases] + [c["older"] for c in cases]
    vectors = embed_many(embed, texts)
    n = len(cases)
    signals = [1.0 - _cosine(vectors[i], vectors[n + i]) for i in range(n)]
    labels = [c["drift"] for c in cases]
    out: dict = {"n": len(cases), "positives": sum(1 for l in labels if l)}
    out["auc"] = round(_auc(signals, labels), 4)
    tp = sum(1 for s, l in zip(signals, labels) if l and s > DRIFT_THRESHOLD)
    fp = sum(1 for s, l in zip(signals, labels) if not l and s > DRIFT_THRESHOLD)
    fn = sum(1 for s, l in zip(signals, labels) if l and s <= DRIFT_THRESHOLD)
    out["at_0.6"] = _prf(tp, fp, fn)
    # the best threshold (the oracle the harness does not have) - the
    # shipped 0.6 was calibrated for another encoder, and the signal
    # itself is usually good
    best = {"f1": 0.0}
    for thresh in np.arange(0.30, 0.995, 0.005):
        tp = sum(1 for s, l in zip(signals, labels) if l and s > thresh)
        fp = sum(1 for s, l in zip(signals, labels) if not l and s > thresh)
        fn = sum(1 for s, l in zip(signals, labels) if l and s <= thresh)
        m = _prf(tp, fp, fn)
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "precision": m["precision"],
                    "recall": m["recall"], "threshold": round(float(thresh), 3)}
    out["best"] = best
    if signals:
        out["signal_p50"] = round(statistics.median(signals), 4)
        out["signal_min"] = round(min(signals), 4)
        out["signal_max"] = round(max(signals), 4)
    return out


def measure_cost(embed, texts: list[str]) -> dict:
    """What the encoder costs: the time to embed a turn's worth of chunks.

    The first call loads the model, so the measurement is warmed up
    first - otherwise the number is the load time, not the cost.
    """
    if not texts:
        return {}
    sample = texts[:32]
    embed(sample[:2])          # the warm-up: the model load is not the cost
    t0 = time.perf_counter()
    embed(sample)
    per = (time.perf_counter() - t0) / len(sample)
    return {"ms_per_chunk": round(per * 1000, 2),
            "ms_per_100_chunks": round(per * 100, 1)}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def run(encoders: list[str], corpus_dirs: list[str], per_conv: int,
        out_path: Optional[Path], rank_pairs: int,
        max_pairs: int = 0, max_cases: int = 0) -> dict:
    conversations = load_conversations(corpus_dirs)
    chunks_total = sum(len(c["chunks"]) for c in conversations)
    print(f"corpus: {len(conversations)} conversations, {chunks_total} chunks")
    if not conversations:
        raise SystemExit("no conversations found; pass --corpus")
    dedup_pairs = build_dedup_pairs(conversations, per_conv=per_conv)
    drift_cases = build_drift_cases(conversations)
    if max_pairs:
        dedup_pairs = dedup_pairs[:max_pairs]
    if max_cases:
        drift_cases = drift_cases[:max_cases]
    print(f"dedup pairs: {len(dedup_pairs)} "
          f"({sum(1 for p in dedup_pairs if p['duplicate'])} positive)")
    print(f"drift cases: {len(drift_cases)} "
          f"({sum(1 for c in drift_cases if c['drift'])} topic changes)")

    # the ranking task: the prior measurement's pairs (the fixtures)
    from experiments.encoder_probe import compute_topk, load_pairs

    rank_pairs_list = load_pairs("fixture", rank_pairs, seed=20260822)

    results: dict = {
        "corpus": {"dirs": corpus_dirs, "conversations": len(conversations),
                   "chunks": chunks_total},
        "dedup_pairs": len(dedup_pairs),
        "drift_cases": len(drift_cases),
        "encoders": {},
    }
    for name in encoders:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            embed, info = make_encoder(name)
        except Exception as exc:  # noqa: BLE001 - a model may not download
            print(f"  failed to load: {exc}")
            results["encoders"][name] = {"error": str(exc)}
            continue
        print(f"  {info['model']} ({info['params_m']}M params, {info['dim']} dim)")
        ded = measure_dedup(dedup_pairs, embed)
        print(f"  dedup  at 0.92: {ded['at_0.92']}  best: {ded['best']}")
        drf = measure_drift(drift_cases, embed)
        print(f"  drift  auc={drf['auc']}  at 0.6: {drf['at_0.6']}")
        rank = compute_topk(rank_pairs_list, lambda q, cs: _scores(embed, q, cs),
                            [1, 3, 5, 8])
        print(f"  rank   {rank}")
        cost = measure_cost(embed, [c["text"] for c in conversations[0]["chunks"]])
        results["encoders"][name] = {
            "info": info,
            "dedup": ded,
            "drift": drf,
            "rank": rank,
            "cost": cost,
            "seconds": round(time.time() - t0, 1),
        }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"\nwrote {out_path}")
    return results


def embed_many(embed, texts: list[str]) -> np.ndarray:
    """The batch interface (the encoders already take a list)."""
    return embed(list(texts))


def _scores(embed, query: str, chunks: list[str]) -> list[float]:
    if not chunks:
        return []
    vecs = embed([query] + list(chunks))
    q = vecs[0]
    return [_cosine(q, vecs[i + 1]) for i in range(len(chunks))]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", nargs="+",
                        default=["l3", "l6", "bge-small"],
                        help="the encoders to measure")
    parser.add_argument("--corpus", nargs="+",
                        default=[str(REPO_ROOT / "harness_state"),
                                 str(REPO_ROOT.parent / "splinter-memory"
                                     / "harness_state")],
                        help="the directories holding the conversations")
    parser.add_argument("--per-conv", type=int, default=40,
                        help="the duplicate pairs sampled per conversation")
    parser.add_argument("--rank-pairs", type=int, default=400,
                        help="the ranking pairs (the fixtures)")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="cap the duplicate pairs (0 = all) - the big "
                             "encoders need a subset on the CPU")
    parser.add_argument("--max-cases", type=int, default=0,
                        help="cap the drift cases (0 = all)")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the JSON result")
    args = parser.parse_args(argv)
    run(args.encoders, args.corpus, args.per_conv, args.out, args.rank_pairs,
        max_pairs=args.max_pairs, max_cases=args.max_cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
