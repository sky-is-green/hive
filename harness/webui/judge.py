"""Tiny online judge for the A/B preference loop.

No dependencies: four normalized difference features and logistic regression.
The model is deliberately transparent — with a handful of votes the useful
output is the profile (longer/shorter, faster/slower, tier), not a black box.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

FEATURES = ("length", "latency", "tokens", "tier")
TIER_ORDER = {"simple": 0, "medium": 1, "complex": 2}
MODEL_PATH = Path(__file__).with_name("judge_model.json")


def _diff(a: float, b: float) -> float:
    denom = abs(a) + abs(b)
    return 0.0 if denom == 0 else (a - b) / denom


def features(pair: dict[str, Any]) -> dict[str, float]:
    len_a = len(str(pair.get("answer_a") or ""))
    len_b = len(str(pair.get("answer_b") or ""))
    return {
        "length": _diff(len_a, len_b),
        "latency": _diff(float(pair.get("latency_a") or 0), float(pair.get("latency_b") or 0)),
        "tokens": _diff(float(pair.get("tokens_a") or 0), float(pair.get("tokens_b") or 0)),
        "tier": float(TIER_ORDER.get(pair.get("tier_a"), 1) - TIER_ORDER.get(pair.get("tier_b"), 1)),
    }


def _answer_text(value: Any) -> str:
    """Normalize answer text for an equivalence check, not for scoring."""
    return " ".join(str(value or "").split()).strip().casefold().rstrip(".!?。！？")


def _answers_equivalent(pair: dict[str, Any]) -> bool:
    return _answer_text(pair.get("answer_a")) == _answer_text(pair.get("answer_b"))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _fit(rows: Iterable[dict[str, Any]], epochs: int = 500, lr: float = 0.08,
         l2: float = 0.01) -> dict[str, Any]:
    data = []
    for row in rows:
        vote = row.get("vote")
        if vote not in ("a", "b"):
            continue
        data.append((features(row), 1.0 if vote == "a" else 0.0))
    weights = {name: 0.0 for name in FEATURES}
    bias = 0.0
    n = len(data)
    for _ in range(epochs if n else 0):
        grads = {name: 0.0 for name in FEATURES}
        gb = 0.0
        for x, y in data:
            p = _sigmoid(bias + sum(weights[k] * x[k] for k in FEATURES))
            err = p - y
            for k in FEATURES:
                grads[k] += err * x[k]
            gb += err
        for k in FEATURES:
            weights[k] -= lr * (grads[k] / n + l2 * weights[k])
        bias -= lr * gb / n
    return {"weights": weights, "bias": bias, "n": n, "features": list(FEATURES)}


def _predict(model: dict[str, Any], feats: dict[str, float]) -> float:
    return _sigmoid(float(model.get("bias", 0.0)) + sum(
        float(model["weights"].get(k, 0.0)) * feats.get(k, 0.0) for k in FEATURES
    ))


def _profile_rule(pair: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Predict from the human's observed trait preferences, not a fitted model.

    This is the safe fallback while a logistic model is data-starved: it can
    only express "you have preferred shorter answers so far", which is exactly
    the profile the Judge page reports.  It is deliberately coarse.
    """
    if _answers_equivalent(pair):
        return {
            "choice": "tie",
            "source": "baseline (equivalent answers)",
            "confidence": None,
            "p_a": None,
        }
    votes = [r for r in rows if r.get("vote") in ("a", "b")]
    longer = shorter = 0
    for row in votes:
        ca, cb = str(row.get("answer_a") or ""), str(row.get("answer_b") or "")
        chosen, rejected = (ca, cb) if row["vote"] == "a" else (cb, ca)
        if len(chosen) > len(rejected):
            longer += 1
        elif len(chosen) < len(rejected):
            shorter += 1
    len_a = len(str(pair.get("answer_a") or ""))
    len_b = len(str(pair.get("answer_b") or ""))
    decided = longer + shorter
    if len_a != len_b and decided >= 3 and abs(longer - shorter) >= max(1, decided // 5):
        if shorter > longer:
            choice = "a" if len_a < len_b else "b"
            why = f"shorter answer ({shorter}/{decided} votes)"
        else:
            choice = "a" if len_a > len_b else "b"
            why = f"longer answer ({longer}/{decided} votes)"
        return {"choice": choice, "source": f"profile ({why})", "confidence": None, "p_a": None}

    faster = slower = 0
    for row in votes:
        la, lb = float(row.get("latency_a") or 0), float(row.get("latency_b") or 0)
        chosen, rejected = (la, lb) if row["vote"] == "a" else (lb, la)
        if chosen < rejected:
            faster += 1
        elif chosen > rejected:
            slower += 1
    lat_a = float(pair.get("latency_a") or 0)
    lat_b = float(pair.get("latency_b") or 0)
    decided = faster + slower
    if lat_a != lat_b and decided >= 3 and abs(faster - slower) >= max(1, decided // 3):
        if faster > slower:
            choice = "a" if lat_a < lat_b else "b"
            why = f"faster answer ({faster}/{decided} votes)"
        else:
            choice = "a" if lat_a > lat_b else "b"
            why = f"slower answer ({slower}/{decided} votes)"
        return {"choice": choice, "source": f"profile ({why})", "confidence": None, "p_a": None}

    # A tie is a real judgment: equal answers should not force a side just
    # because the baseline needs a label.  Tolerances keep formatting and
    # timing noise from turning a near-equal pair into a preference.
    len_tol = max(8, int(0.03 * max(len_a, len_b, 1)))
    lat_tol = max(0.15, 0.10 * max(abs(lat_a), abs(lat_b), 0.001))
    if abs(len_a - len_b) <= len_tol and abs(lat_a - lat_b) <= lat_tol:
        return {
            "choice": "tie",
            "source": "baseline (equivalent answers)",
            "confidence": None,
            "p_a": None,
        }
    if len_a != len_b:
        choice = "a" if len_a > len_b else "b"
        why = "longer answer"
    else:
        choice = "a" if lat_a <= lat_b else "b"
        why = "faster answer"
    return {"choice": choice, "source": f"baseline ({why})", "confidence": None, "p_a": None}


def predict(pair: dict[str, Any], model: dict[str, Any] | None,
            rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    feats = features(pair)
    if _answers_equivalent(pair):
        return {
            "choice": "tie",
            "source": "baseline (equivalent answers)",
            "confidence": None,
            "p_a": None,
            "features": feats,
        }
    if model and int(model.get("n") or 0) >= 5:
        p = _predict(model, feats)
        confidence = abs(p - 0.5) * 2.0
        # A six-vote logistic fit can be confidently wrong.  Only trust it
        # when it has a real margin; otherwise fall back to the profile rule.
        if confidence >= 0.12:
            return {
                "choice": "a" if p >= 0.5 else "b",
                "confidence": round(confidence, 3),
                "p_a": round(p, 4),
                "source": "model",
                "features": feats,
            }
    fallback = _profile_rule(pair, rows or [])
    return {**fallback, "features": feats}


def train(rows: list[dict[str, Any]]) -> dict[str, Any]:
    model = _fit(rows)
    model["loo_accuracy"] = loo_accuracy(rows)
    MODEL_PATH.write_text(json.dumps(model, indent=2) + "\n")
    return model


def loo_accuracy(rows: list[dict[str, Any]]) -> float | None:
    data = [r for r in rows if r.get("vote") in ("a", "b")]
    if len(data) < 3:
        return None
    # Online retraining happens on every vote; leave-one-out is O(n^2).  Cap it
    # to the most recent window so a long-running profile cannot stall a save.
    if len(data) > 60:
        data = data[-60:]
    hits = 0
    for index in range(len(data)):
        train_rows = data[:index] + data[index + 1:]
        model = _fit(train_rows)
        p = _predict(model, features(data[index]))
        predicted = "a" if p >= 0.5 else "b"
        hits += int(predicted == data[index]["vote"])
    return round(hits / len(data), 4)


def profile(rows: list[dict[str, Any]], rationales: dict[str, dict[str, Any]]) -> dict[str, Any]:
    votes = [r for r in rows if r.get("vote") in ("a", "b")]
    ties = sum(1 for r in rows if r.get("vote") == "tie")
    n = len(votes)
    longer = shorter = equal_len = 0
    faster = slower = equal_lat = 0
    higher = lower = equal_tier = 0
    chosen_len = rejected_len = 0
    chosen_lat = rejected_lat = 0
    tags: dict[str, int] = {}
    agreement = agree_n = 0
    by_model: dict[str, dict[str, int]] = {}
    for row in votes:
        vote = row["vote"]
        ca, cb = str(row.get("answer_a") or ""), str(row.get("answer_b") or "")
        la, lb = float(row.get("latency_a") or 0), float(row.get("latency_b") or 0)
        ta = TIER_ORDER.get(row.get("tier_a"), 1)
        tb = TIER_ORDER.get(row.get("tier_b"), 1)
        cl, rl = (len(ca), len(cb)) if vote == "a" else (len(cb), len(ca))
        clat, rlat = (la, lb) if vote == "a" else (lb, la)
        chosen_len += cl; rejected_len += rl
        chosen_lat += clat; rejected_lat += rlat
        longer += int(cl > rl); shorter += int(cl < rl); equal_len += int(cl == rl)
        faster += int(clat < rlat); slower += int(clat > rlat); equal_lat += int(clat == rlat)
        ct, rt = (ta, tb) if vote == "a" else (tb, ta)
        higher += int(ct > rt); lower += int(ct < rt); equal_tier += int(ct == rt)
        model = row.get("model_a" if vote == "a" else "model_b") or "?"
        bucket = by_model.setdefault(model, {"wins": 0, "losses": 0})
        bucket["wins"] += 1
        other = row.get("model_b" if vote == "a" else "model_a") or "?"
        by_model.setdefault(other, {"wins": 0, "losses": 0})["losses"] += 1
        rationale = rationales.get(str(row.get("id") or "")) or {}
        for tag in rationale.get("tags") or []:
            tags[tag] = tags.get(tag, 0) + 1
    for row in rows:
        vote = row.get("vote")
        auto = row.get("auto_choice")
        if vote in ("a", "b", "tie") and auto in ("a", "b", "tie"):
            agree_n += 1
            agreement += int(auto == vote)
    pct = lambda x: round(x / n, 3) if n else None
    return {
        "n": n,
        "ties": ties,
        "tie_rate": round(ties / len(rows), 3) if rows else None,
        "longer": pct(longer), "shorter": pct(shorter), "equal_length": pct(equal_len),
        "faster": pct(faster), "slower": pct(slower), "equal_latency": pct(equal_lat),
        "higher_tier": pct(higher), "lower_tier": pct(lower), "equal_tier": pct(equal_tier),
        "mean_chosen_len": round(chosen_len / n, 1) if n else None,
        "mean_rejected_len": round(rejected_len / n, 1) if n else None,
        "mean_chosen_latency": round(chosen_lat / n, 2) if n else None,
        "mean_rejected_latency": round(rejected_lat / n, 2) if n else None,
        "auto_agreement": round(agreement / agree_n, 3) if agree_n else None,
        "auto_agreement_n": agree_n,
        "tags": tags,
        "by_model": by_model,
    }
