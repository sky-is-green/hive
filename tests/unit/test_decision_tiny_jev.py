"""Unit tests for ``TinyJevJudge`` — the local validation role.

No torch/transformers in this venv: the model and tokenizer are injected, and
the test pins exactly what the judge sends to the checkpoint's ``noul`` head.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from experiments.cascade.decision import TinyJevJudge


def test_verdict_passes_state_and_instruction_to_noul():
    calls: dict = {}

    class FakeModel:
        def noul(self, tok, state, instruction):
            calls.update(tok=tok, state=state, instruction=instruction)
            return 0.87

    judge = TinyJevJudge(
        _model=FakeModel(),
        _tokenizer="tok",
        _torch=SimpleNamespace(no_grad=lambda: nullcontext()),
    )

    p, ms = judge.verdict({"prompt": "2+2?"}, "4")

    assert p == 0.87
    assert ms >= 0.0
    assert calls["tok"] == "tok"
    assert calls["state"] == {"task": "2+2?", "candidate_answer": "4"}
    assert calls["instruction"] == TinyJevJudge.INSTRUCTION
    assert judge.device == "cpu"
