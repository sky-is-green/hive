"""The model-free request features are deterministic and inspectable."""

from __future__ import annotations

from harness.controller.features import text_features


def test_code_is_detected():
    features = text_features("def f(x):\n    return x + 1\n")
    assert features.intent == "code"
    assert features.context_pressure == "short"
    assert features.risk == "medium"


def test_questions_are_qa():
    features = text_features("How do I rotate a TLS certificate?")
    assert features.intent == "qa"
    assert features.is_question is True


def test_math_is_reasoning():
    features = text_features("Prove that the integral of x is x squared over two.")
    assert features.intent == "reasoning"


def test_risk_and_length_are_conservative():
    features = text_features("delete production and run sudo rm -rf /")
    assert features.risk == "high"
    long_text = "word " * 5000
    assert text_features(long_text).context_pressure == "long"


def test_features_do_not_claim_to_be_an_embedding():
    features = text_features("hello")
    payload = features.as_dict()
    assert "vector" not in payload
    assert payload["intent"] in ("chat", "qa", "code", "reasoning")
