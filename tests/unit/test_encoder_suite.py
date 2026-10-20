"""The encoder suite: the measurement's foundations.

The suite measures the three jobs the harness gives its encoder (rank,
dedup, drift) on the harness's own traffic.  The labels are lexical, so
they do not come from the encoder being measured - that is the property
these tests pin down, because a label that depends on the model under
test measures nothing.
"""

from __future__ import annotations

import pytest

from experiments.encoder_suite import (DUP_JACCARD, build_dedup_pairs,
                                       build_drift_cases, content_terms,
                                       jaccard, load_conversations)


@pytest.fixture()
def conversations():
    return [
        {"id": "c1", "chunks": [
            {"turn": 1, "text": "The build uses an RX 7900 XTX graphics card."},
            {"turn": 2, "text": "The graphics card is an RX 7900 XTX."},
            {"turn": 3, "text": "The power supply is a 850W unit."},
            {"turn": 4, "text": "Python runs the memory tests overnight."},
            {"turn": 5, "text": "The memory tests run in Python at night."},
            {"turn": 6, "text": "Tomorrow we will benchmark the encoder."},
        ]},
        {"id": "c2", "chunks": [
            {"turn": 1, "text": "Unrelated: the kitchen needs new tiles."},
            {"turn": 2, "text": "Also unrelated: the train leaves at nine."},
            {"turn": 3, "text": "Slightly different: the garden fence is old."},
        ]},
    ]


def test_the_content_terms_drop_the_filler():
    assert "graphics" in content_terms("The graphics card")
    assert "the" not in content_terms("The graphics card")
    assert "and" not in content_terms("cats and dogs")


def test_the_jaccard_is_the_lexical_ground_truth():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"a", "b", "c"}) == pytest.approx(2 / 3)
    assert jaccard(set(), set()) == 0.0


def test_the_duplicate_pairs_are_labelled_lexically(conversations):
    """A positive shares most of its content words; a negative does not."""
    pairs = build_dedup_pairs(conversations, per_conv=10)
    assert pairs
    for pair in pairs:
        if pair["duplicate"]:
            assert pair["jaccard"] >= DUP_JACCARD, pair
        else:
            assert pair["jaccard"] < DUP_JACCARD, pair


def test_the_duplicate_pairs_carry_both_texts(conversations):
    """The pair is what the membrane's test sees: two chunks."""
    pairs = build_dedup_pairs(conversations, per_conv=10)
    for pair in pairs:
        assert pair["a"] and pair["b"]
        assert pair["a"] != pair["b"]


def test_the_drift_cases_label_the_topic_change(conversations):
    """A topic change is a lexical fact, not a model's opinion."""
    cases = build_drift_cases(conversations, min_history=2)
    assert cases
    for case in cases:
        if case["drift"]:
            assert case["jaccard"] < 0.05, case
        else:
            assert case["jaccard"] >= 0.05, case


def test_the_drift_text_is_bounded(conversations):
    """The embedded text is capped: the encoders truncate anyway, but the
    strings themselves would cost gigabytes at full length."""
    from experiments.encoder_suite import MAX_EMBED_CHARS

    cases = build_drift_cases(conversations, min_history=2)
    for case in cases:
        assert len(case["recent"]) <= MAX_EMBED_CHARS
        assert len(case["older"]) <= MAX_EMBED_CHARS


def test_the_corpus_loader_skips_what_it_cannot_read(tmp_path):
    """A broken conversation file does not stop the measurement."""
    (tmp_path / "conv-broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "conv-ok.json").write_text(
        '{"store": {"chunks": [{"turn": 1, "content": "one"},'
        ' {"turn": 2, "content": "two"}]}}', encoding="utf-8")
    out = load_conversations([str(tmp_path)])
    assert len(out) == 1
    assert [c["text"] for c in out[0]["chunks"]] == ["one", "two"]


def test_the_finetune_examples_carry_a_hard_negative(conversations):
    """Every training example is anchor/positive/negative."""
    from experiments.encoder_finetune import build_examples

    examples = build_examples(conversations)
    assert examples
    for example in examples:
        assert example["anchor"]
        assert example["positive"]
        assert example["negative"]
        assert example["task"] in ("dedup", "drift", "rank")
