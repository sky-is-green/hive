"""Unit tests for ``experiments.cascade.stream_capture``'s request builder.

The capture protocol is data (a chat-completions body); the Scion runs and the
Strata runs must differ only in the explicit knobs, so the payload is pinned
with and without them.  The reliable wrapper's recovery policy is pinned too:
engine wedges retry, client errors do not.
"""

from __future__ import annotations

import pytest

import experiments.cascade.stream_capture as sc
from experiments.cascade.stream_capture import SCION_SYSTEM, completion_payload


def test_default_payload_keeps_the_recorded_scion_protocol():
    payload = completion_payload("solve x", temperature=0.7, max_tokens=1536)

    assert payload["model"] == "scion"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["messages"][0] == {"role": "system", "content": SCION_SYSTEM}
    assert payload["messages"][1] == {"role": "user", "content": "solve x"}
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 1536
    # no reasoning key when unset: the engine's own default stays in charge
    assert "reasoning_effort" not in payload
    assert "reasoning_budget_tokens" not in payload


def test_strata_payload_carries_model_and_reasoning_controls():
    payload = completion_payload(
        "q", model="strata", temperature=0.7, max_tokens=3072,
        reasoning_effort="high", reasoning_budget_tokens=2048,
    )

    assert payload["model"] == "strata"
    assert payload["reasoning_effort"] == "high"
    assert payload["reasoning_budget_tokens"] == 2048
    assert payload["max_tokens"] == 3072


def test_stream_reliable_recovers_from_an_engine_wedge(monkeypatch):
    calls = {"n": 0, "restarted": False}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(
                'HTTP 400: {"message": "verify: #267; restart the engine"}')
        return {"ok": True}

    monkeypatch.setattr(sc, "stream_once", flaky)
    monkeypatch.setattr(sc, "force_restart",
                        lambda base: calls.__setitem__("restarted", True))

    out = sc.stream_reliable("http://127.0.0.1:8080/v1", "p",
                             temperature=0.7, max_tokens=8)

    assert out == {"ok": True}
    assert calls == {"n": 2, "restarted": True}


def test_stream_reliable_raises_real_client_errors(monkeypatch):
    def bad(*args, **kwargs):
        raise RuntimeError("HTTP 400: context length exceeded")

    monkeypatch.setattr(sc, "stream_once", bad)
    monkeypatch.setattr(sc, "force_restart",
                        lambda base: pytest.fail("must not recover a client error"))

    with pytest.raises(RuntimeError, match="context length"):
        sc.stream_reliable("http://127.0.0.1:8080/v1", "p",
                           temperature=0.7, max_tokens=8)
