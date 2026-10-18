"""Unit tests for the deterministic verifier and the repair leg.

No model, network, or terminal: the local tier is a scripted stub, and the
findings come from the same gold-free detectors the loop uses.  The point of
these tests is the contract the wiring promised: the answer key never reaches
the prompt, repairs stay greedy and one round, and a truncated or frameless
repair is never accepted as the answer.
"""

from __future__ import annotations

import experiments.cascade.run_cascade as rc
from experiments.cascade.cascade import detect, runtime_spec
from harness.cascade import Judgement
from harness.cascade.policy import Policy, verifier_rejects

NUMBER_TASK = {
    "id": "t1",
    "prompt": "What is 12 + 12 + 16?",
    "bucket": "reasoning",
    "checker": {"type": "number", "expect": "40"},
}
WRONG = "We add them: $12 + 12 = 26$, so the total is 26.\nFinal answer: 26"
RIGHT = "We add them: $12 + 12 = 24$, so the total is 24.\nFinal answer: 24"

CODE_TASK = {
    "id": "t2",
    "prompt": "write double(x)",
    "bucket": "code",
    "checker": {
        "type": "code",
        "expect": "def double(x): return 2 * x",
        "tests": "assert double(2) == 4\nassert double(0) == 0\n",
    },
}
BROKEN_CODE = "```python\ndef double(x):\n    return x\n```"


class ScriptedTier:
    """A ScionTier stand-in: returns scripted answers and records the calls."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def chat(self, messages, *, max_tokens=None, temperature=None):
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        )
        answer = self.answers.pop(0)
        if isinstance(answer, str):
            answer = {"content": answer, "finish_reason": "stop", "usage": {}}
        return {"latency_ms": 1.0, "reasoning": "", "usage": {}, **answer}


def _judgement(candidate: str, task: dict = NUMBER_TASK) -> Judgement:
    return Judgement(
        id=f"verify-{task['id']}", request_id=task["id"], generation_id=0,
        role="D3", kind="verify", payload={"task": task, "candidate": candidate},
    )


# ---------------------------------------------------------------- the verdict

def test_runtime_spec_keeps_the_answer_key_out_of_the_loop():
    rs = runtime_spec(NUMBER_TASK)

    assert rs == {"kind": "number"}
    assert "40" not in str(rs)


def test_deterministic_verdict_rejects_a_self_inconsistent_draft():
    verdict = rc.deterministic_verdict(
        _judgement(WRONG), NUMBER_TASK, WRONG, runtime_spec(NUMBER_TASK)
    )

    assert verifier_rejects(verdict, Policy())
    assert verdict.payload["detectors"] == ["consistency"]
    assert "12 + 12 = 26 is false" in verdict.payload["reason"]


def test_deterministic_verdict_accepts_a_consistent_draft():
    verdict = rc.deterministic_verdict(
        _judgement(RIGHT), NUMBER_TASK, RIGHT, runtime_spec(NUMBER_TASK)
    )

    assert verdict.decision == "accept"
    assert not verifier_rejects(verdict, Policy())


# ---------------------------------------------------------------- prompts

def test_repair_prompt_quotes_the_finding_and_never_the_answer():
    findings = detect(WRONG, runtime_spec(NUMBER_TASK))

    detail, text = rc.repair_user_text(NUMBER_TASK, WRONG, findings)

    assert detail == "consistency"
    assert "12 + 12 = 26" in text          # the finding's equation
    assert "40" not in text                # the answer key never leaks


def test_repair_prompt_falls_back_to_generic_without_findings():
    detail, text = rc.repair_user_text(NUMBER_TASK, WRONG, [])

    assert detail == "generic"
    assert "Recheck the draft" in text
    assert WRONG in text


def test_code_repair_prompt_quotes_the_traceback_not_the_answer():
    findings = detect(BROKEN_CODE, runtime_spec(CODE_TASK))

    detail, text = rc.repair_user_text(CODE_TASK, BROKEN_CODE, findings)

    assert detail == "code_exec"
    assert "AssertionError" in text
    assert "return 2 * x" not in text      # the answer key never leaks


# ---------------------------------------------------------------- the round

def test_repair_clears_and_is_reframed_to_the_task_interface():
    rs = runtime_spec(NUMBER_TASK)
    tier = ScriptedTier([RIGHT])

    record = rc.attempt_repair(tier, NUMBER_TASK, WRONG, detect(WRONG, rs), rs,
                               max_tokens=512)

    assert record["cleared"] is True
    assert record["canonical"] == "Final answer: 24"
    assert record["still_flagged"] == []
    assert record["generations"] == 1
    assert tier.calls[0]["temperature"] == 0.0     # repairs stay greedy
    assert "40" not in tier.calls[0]["messages"][1]["content"]


def test_repair_repeating_the_draft_is_flagged_not_accepted():
    rs = runtime_spec(NUMBER_TASK)
    tier = ScriptedTier([WRONG])

    record = rc.attempt_repair(tier, NUMBER_TASK, WRONG, detect(WRONG, rs), rs,
                               max_tokens=512)

    assert record["cleared"] is False
    assert record["same_as_draft"] is True
    assert record["still_flagged"] == ["consistency"]


def test_truncated_repair_retries_bigger_and_is_never_accepted():
    rs = runtime_spec(NUMBER_TASK)
    tier = ScriptedTier([
        {"content": "12 + 12 = 24 and then", "finish_reason": "length",
         "usage": {"completion_tokens": 512}},
        {"content": "12 + 12 = 24 and still cut", "finish_reason": "length",
         "usage": {"completion_tokens": 1024}},
    ])

    record = rc.attempt_repair(tier, NUMBER_TASK, WRONG, detect(WRONG, rs), rs,
                               max_tokens=512)

    assert record["truncated"] is True
    assert record["cleared"] is False
    assert record["generations"] == 2
    assert tier.calls[1]["max_tokens"] == 2048     # the runner's doubled-budget floor
    assert record["stopped"] == "the repair was truncated"


def test_repair_without_a_consumable_frame_is_not_accepted():
    rs = runtime_spec(NUMBER_TASK)
    tier = ScriptedTier(["12 + 12 = 24, so the total is twenty-four."])

    record = rc.attempt_repair(tier, NUMBER_TASK, WRONG, detect(WRONG, rs), rs,
                               max_tokens=512)

    assert record["canonical"] is None
    assert record["cleared"] is False
    assert "no consumable" in record["stopped"]


# ---------------------------------------------------------------- the gate

def test_gate_shows_the_repair_attempt_on_the_first_trigger():
    lines: list[str] = []
    gate = rc.HumanGate(
        api_target="http://api.example/v1", api_model="frontier-test",
        input_fn=lambda _prompt: "e", print_fn=lines.append,
    )
    repair = {
        "text": RIGHT, "detail": "consistency", "still_flagged": ["consistency"],
        "finding": "consistency: 12 + 12 = 26 is false",
    }

    gate.ask(
        task={"id": "t1", "bucket": "reasoning", "prompt": NUMBER_TASK["prompt"]},
        local_answer=WRONG,
        verdict=rc.deterministic_verdict(
            _judgement(WRONG), NUMBER_TASK, WRONG, runtime_spec(NUMBER_TASK)
        ),
        checker_ok=False,
        repair=repair,
    )

    output = "\n".join(lines)
    assert "automatic repair" in output
    assert "repair still flagged by: consistency" in output
    assert "12 + 12 = 24" in output


# ---------------------------------------------------------------- the flags

def test_repair_flag_defaults_and_judge_coupling():
    assert rc.repair_enabled(rc.build_parser().parse_args([])) is False
    assert rc.repair_enabled(
        rc.build_parser().parse_args(["--judge", "deterministic"])
    ) is True
    assert rc.repair_enabled(
        rc.build_parser().parse_args(["--repair", "off", "--judge", "deterministic"])
    ) is False
    assert rc.repair_enabled(rc.build_parser().parse_args(["--repair", "on"])) is True


def test_local_temperature_is_greedy_by_default():
    args = rc.build_parser().parse_args([])

    assert args.local_temperature == 0.0
    assert rc.build_parser().parse_args(["--local-temperature", "0.7"]).local_temperature == 0.7
