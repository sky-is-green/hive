"""Unit tests for the pre-API human gate.

No model, network, or terminal: the prompt's input and output are injected.
"""

from __future__ import annotations

from harness.cascade.judgement import result_for, Judgement
from experiments.cascade.run_cascade import HumanGate


def _verdict(reason: str = "the unit test fails"):
    judgement = Judgement(
        id="verify-t1", request_id="t1", generation_id=0,
        role="D3", kind="verify",
    )
    return result_for(judgement, "reject", 0.91, payload={"reason": reason})


def _gate(answers: list[str]):
    lines: list[str] = []
    remaining = list(answers)

    def read(_prompt: str) -> str:
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    gate = HumanGate(
        api_target="http://api.example/v1",
        api_model="frontier-test",
        input_fn=read,
        print_fn=lines.append,
    )
    return gate, lines


def _ask(gate: HumanGate, *, checker_ok=False, verdict=None):
    return gate.ask(
        task={"id": "t1", "bucket": "code", "prompt": "write it"},
        local_answer="print('wrong')",
        verdict=verdict or _verdict(),
        checker_ok=checker_ok,
    )


def test_first_trigger_shows_task_answer_verifier_checker_and_api():
    gate, lines = _gate(["e", ""])
    decision = _ask(gate)

    output = "\n".join(lines)
    assert decision["choice"] == "escalate"
    assert decision["mode"] == "escalate"
    assert "first trigger" in output
    assert "t1" in output and "code" in output
    assert "write it" in output
    assert "print('wrong')" in output
    assert "the unit test fails" in output
    assert "FAILED" in output
    assert "http://api.example/v1" in output
    assert "frontier-test" in output
    assert "[s]olve" in output and "[e]scalate" in output


def test_solve_with_a_hint_returns_the_hint_and_keeps_the_local_first():
    gate, _ = _gate(["s", "hint: check the empty-list boundary", ""])
    decision = _ask(gate)

    assert decision["choice"] == "solve"
    assert decision["mode"] == "hint"
    assert decision["hint"] == "check the empty-list boundary"
    assert gate.asked == 1


def test_solve_with_a_corrected_answer_records_the_reason():
    gate, _ = _gate(["s", "42", "the local model dropped the carry"])
    decision = _ask(gate)

    assert decision["choice"] == "solve"
    assert decision["mode"] == "answer"
    assert decision["answer"] == "42"
    assert decision["reason"] == "the local model dropped the carry"


def test_keep_shortcut_is_only_offered_when_the_checker_already_passes():
    gate, lines = _gate(["s", "keep"])
    decision = _ask(gate, checker_ok=True)

    assert decision["mode"] == "keep"
    assert "PASSED" in "\n".join(lines)
    assert "false reject" in "\n".join(lines)


def test_invalid_choice_reprompts_instead_of_adding_a_third_option():
    gate, lines = _gate(["maybe", "e", ""])
    decision = _ask(gate)

    assert decision["choice"] == "escalate"
    assert "Please answer 's'" in "\n".join(lines)


def test_eof_escalates_with_a_reason_instead_of_hanging():
    gate, _ = _gate([])
    decision = _ask(gate)

    assert decision["choice"] == "escalate"
    assert decision["reason"] == "human gate reached EOF"


def test_later_triggers_are_compact():
    gate, lines = _gate(["e", "", "e", ""])
    _ask(gate)
    _ask(gate)

    output = "\n".join(lines)
    assert output.count("This is the first trigger") == 1
    assert "still unresolved" in output
    assert gate.asked == 2
