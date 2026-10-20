"""Hypothesis H: do the membrane's operating points matter?

**H.** With the shipped drone (``paraphrase-MiniLM-L3-v2``), running the
membrane's duplicate test at 0.57 and its topic-change test at 0.77 -
instead of the shipped 0.92 and 0.6 - keeps the same facts in the
assembled context, at the same context size, while the two *decisions*
the membrane makes stop being near-dead.

The measurements behind the hypothesis (``docs/ENCODER-NOTES.md``): at
the shipped points the duplicate test catches 4.6% of the duplicate pairs
it should, and the topic-change test fires on almost every turn.  At the
encoder's own best points the same decisions are 8.6x and 3.7x better.

This is the *cheap half* of the test: the curation is deterministic, so
the question "does the recalibrated threshold lose a fact?" needs no
generation at all.  The fixture corpus is replayed through the real
pipeline - the real router, the real assembly, the real dedup, the real
drift detector, the real store - with the two threshold settings, and
the assembled contexts are compared.

What is measured, per turn:

``recall``       the ground-truth answer's facts present in the assembled
                 context (the P2 diagnostic's own notion, unchanged)
``precision``    the sentence-level proxy the diagnostic reports
``tokens``       the assembled context's size
``churn``        how many of the selected chunks differ between the arms
                 (the dedup and the drift changed the selection)

Falsification: if the recalibrated arm's recall drops, or its context
inflates, H is false and the shipped points stand.

Usage::

    python -m experiments.threshold_ab
    python -m experiments.threshold_ab --fixtures tests/fixtures/generated
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The arms.  The two thresholds are factored, because they do different
#: things and their costs are not symmetric: a duplicate test that fires
#: wrongly *deletes a chunk* (a fact is gone), while a topic-drift test
#: that fires wrongly only *penalises* the old chunks (the context is
#: smaller, the facts are still in it).  A single "both recalibrated"
#: arm cannot tell you which one cost the facts.
SHIPPED = {"dedup": 0.92, "drift": 0.6}
DEDUP_ONLY = {"dedup": 0.57, "drift": 0.6}
DRIFT_ONLY = {"dedup": 0.92, "drift": 0.77}
RECALIBRATED = {"dedup": 0.57, "drift": 0.77}


def load_fixture_conversations(fixtures: Path) -> list[dict]:
    """The fixture conversations, each with its query and its answer."""
    from cortex.baselines.runner import load_conversations

    return load_conversations(str(fixtures))


def answer_map(conversations: list[dict]) -> dict[str, dict[str, str]]:
    """conversation_id -> {query: the fixture's answer}."""
    out: dict[str, dict[str, str]] = {}
    for conv in conversations:
        cid = conv.get("conversation_id", "unknown")
        per: dict[str, str] = {}
        for turn in conv.get("turns", []):
            if turn.get("role") != "user":
                continue
            per.setdefault(turn.get("content", ""), "")
        # the assistant turn that follows each query
        turns = conv.get("turns", [])
        for i, turn in enumerate(turns):
            if turn.get("role") != "user":
                continue
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            if nxt and nxt.get("role") == "assistant":
                per[turn.get("content", "")] = nxt.get("content", "")
        out[cid] = per
    return out


def replay(conversations: list[dict], answers: dict, points: dict,
           encoder: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",
           max_turns: int = 0) -> list[dict]:
    """Replay the corpus through the real pipeline at one threshold set."""
    from focal.budget import AdaptiveBudget
    from cortex.routing import DroneRouter, EscalationHandler
    from focal.assembly import ContextAssembler
    from membrane.dedup import ContextDeduplicator
    from membrane.drift import TopicDriftDetector
    from retention.store import ContextStore
    from sieve.ultra_small import UltraSmallDrone

    drone = UltraSmallDrone(model_name=encoder, confidence_mode="off",
                             device="cpu")
    router = DroneRouter()
    budget = AdaptiveBudget()
    assembler = ContextAssembler()
    escalation = EscalationHandler()
    dedup = ContextDeduplicator(threshold=points["dedup"])
    drift = TopicDriftDetector(threshold=points["drift"])

    records: list[dict] = []
    for conv in conversations:
        cid = conv.get("conversation_id", "unknown")
        store = ContextStore(embed_fn=drone.embed)
        turns = conv.get("turns", [])
        queries = [t for t in turns if t.get("role") == "user"]
        limit = len(queries) if not max_turns else min(len(queries), max_turns)
        turn_no = 0
        for turn in queries[:limit]:
            turn_no += 1
            query = turn.get("content", "")
            answer = (answers.get(cid) or {}).get(query, "")
            # the fixture's answer stands in for the model's reply: the
            # curation's input is the query, and the store's content is
            # what a model would have said
            assembled = assembler.assemble(
                query=query, current_turn=turn_no, store=store,
                router=router, ultra_small=drone, medium=None,
                escalation=escalation, dedup=dedup, drift_detector=drift,
                budget=budget, max_context=8192,
            )
            content = getattr(assembled, "content", "") or ""
            records.append({
                "conversation_id": cid,
                "query": query,
                "answer": answer,
                "content": content,
                "token_count": int(getattr(assembled, "token_count", 0) or 0),
                "selected": list(getattr(assembled, "selected_chunk_ids", []) or []),
            })
            if answer:
                store.add_chunk(turn_no, answer)
    return records


def score(records: list[dict]) -> dict:
    """The P2 diagnostic's own measures over the replayed turns."""
    from experiments.retrieval_diagnostic import (_answer_fact_terms,
                                                  _content_terms,
                                                  _turn_precision)

    hits: list[float] = []
    precisions: list[float] = []
    tokens: list[int] = []
    measurable = 0
    for record in records:
        answer = record.get("answer") or ""
        query = record.get("query") or ""
        content = record.get("content") or ""
        if not answer:
            continue
        facts = _answer_fact_terms(query, answer)
        if not facts:
            continue
        measurable += 1
        found = facts & _content_terms(content)
        hits.append(len(found) / len(facts))
        precision = _turn_precision(query, content)
        if precision is not None:
            precisions.append(precision)
        tokens.append(int(record.get("token_count") or 0))
    return {
        "measurable_turns": measurable,
        "recall": round(100.0 * statistics.mean(hits), 2) if hits else None,
        "recall_median": round(100.0 * statistics.median(hits), 2) if hits else None,
        "precision": round(100.0 * statistics.mean(precisions), 2) if precisions else None,
        "mean_tokens": round(statistics.mean(tokens), 1) if tokens else None,
    }


def compare(arm_a: list[dict], arm_b: list[dict]) -> dict:
    """How much the selection moved between the two arms."""
    by_key = {(r["conversation_id"], r["query"]): r for r in arm_b}
    churn: list[float] = []
    for record in arm_a:
        other = by_key.get((record["conversation_id"], record["query"]))
        if other is None:
            continue
        a = set(record.get("selected") or [])
        b = set(other.get("selected") or [])
        union = a | b
        if union:
            churn.append(len(a ^ b) / len(union))
    return {
        "turns_compared": len(churn),
        "mean_selection_churn": round(100.0 * statistics.mean(churn), 2)
        if churn else None,
        "turns_with_a_different_selection": sum(1 for c in churn if c > 0),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", default=str(REPO_ROOT / "tests" / "fixtures"
                                                   / "generated"))
    parser.add_argument("--max-turns", type=int, default=0,
                        help="cap the turns per conversation (0 = all)")
    parser.add_argument("--encoder", default="sentence-transformers/paraphrase-MiniLM-L3-v2")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    conversations = load_fixture_conversations(Path(args.fixtures))
    if not conversations:
        raise SystemExit(f"no fixture conversations in {args.fixtures}")
    answers = answer_map(conversations)
    print(f"corpus: {len(conversations)} conversations, "
          f"{sum(1 for c in conversations for t in c.get('turns', []) if t.get('role') == 'user')} turns")

    results: dict = {"fixtures": args.fixtures, "encoder": args.encoder,
                     "arms": {}}
    for name, points in (("shipped", SHIPPED), ("dedup_only", DEDUP_ONLY),
                         ("drift_only", DRIFT_ONLY),
                         ("recalibrated", RECALIBRATED)):
        print(f"\n=== {name} (dedup {points['dedup']}, drift {points['drift']})",
              flush=True)
        records = replay(conversations, answers, points,
                         encoder=args.encoder, max_turns=args.max_turns)
        summary = score(records)
        results["arms"][name] = {"points": points, "summary": summary,
                                  "records": records}
        print(f"  recall {summary['recall']}%  precision {summary['precision']}%  "
              f"mean tokens {summary['mean_tokens']}  "
              f"({summary['measurable_turns']} measurable turns)")

    base = results["arms"]["shipped"]["summary"]
    results["selection_churn"] = {
        name: compare(results["arms"]["shipped"]["records"],
                      arm["records"])
        for name, arm in results["arms"].items() if name != "shipped"
    }
    for name, churn in results["selection_churn"].items():
        print(f"selection churn ({name}): {churn}")
    # The verdict is per arm: a threshold that costs facts is not a
    # threshold to ship, whatever its decision F1 says.
    results["verdict"] = {
        name: {
            "recall_kept": (arm["summary"]["recall"] is not None
                            and base["recall"] is not None
                            and arm["summary"]["recall"] >= base["recall"] - 0.5),
            "context_not_inflated": (
                arm["summary"]["mean_tokens"] is not None
                and base["mean_tokens"] is not None
                and arm["summary"]["mean_tokens"] <= base["mean_tokens"] * 1.05),
        }
        for name, arm in results["arms"].items() if name != "shipped"
    }
    for name, verdict in results["verdict"].items():
        verdict["h_survives"] = all(verdict.values())
    print(f"verdict: {results['verdict']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1, default=str),
                            encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
