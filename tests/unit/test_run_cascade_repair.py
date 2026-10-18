"""Unit tests for the deterministic verifier and the repair leg.

No model, network, or terminal: the local tier is a scripted stub, and the
findings come from the same gold-free detectors the loop uses.  The point of
these tests is the contract the wiring promised: the answer key never reaches
the prompt, repairs stay greedy and one round, and a truncated or frameless
repair is never accepted as the answer.
"""

from __future__ import annotations

import io

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


def test_truncation_verdict_rejects_even_when_the_detectors_accepted():
    accepted = rc.deterministic_verdict(
        _judgement(RIGHT), NUMBER_TASK, RIGHT, runtime_spec(NUMBER_TASK)
    )

    verdict = rc.truncation_verdict(accepted, "t1", "deterministic", 2048)

    assert verdict.decision == "reject"
    assert verifier_rejects(verdict, Policy())
    assert verdict.payload["detectors"] == ["truncated"]
    assert "2048 tokens" in verdict.payload["reason"]


def test_truncation_verdict_keeps_the_detector_findings():
    rejected = rc.deterministic_verdict(
        _judgement(WRONG), NUMBER_TASK, WRONG, runtime_spec(NUMBER_TASK)
    )

    verdict = rc.truncation_verdict(rejected, "t1", "deterministic", 2048)

    assert verdict.payload["detectors"] == ["consistency", "truncated"]
    assert "12 + 12 = 26 is false" in verdict.payload["reason"]


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


# ---------------------------------------------------------------- learned gate

class StubGate:
    """A LearnedGate stand-in: a scripted p per score, calls recorded."""

    def __init__(self, p: float):
        self.p = p
        self.calls: list[tuple[str, str]] = []

    def score(self, question, candidate):
        self.calls.append((question, candidate))
        return self.p, 12.5


def _gate_finding(p=0.074):
    return rc.gate_finding(p, 12.5, 0.5, "L0.75")


def test_gate_never_scores_a_already_rejected_draft():
    verdict = rc.deterministic_verdict(
        _judgement(WRONG), NUMBER_TASK, WRONG, runtime_spec(NUMBER_TASK)
    )
    gate = StubGate(0.05)

    out, row = rc.apply_learned_gate(
        gate, "t1", NUMBER_TASK, WRONG, verdict, 0.5
    )

    assert out is verdict                    # untouched
    assert row["scored"] is False            # blind-spot-only: never even scored
    assert gate.calls == []


def test_gate_fires_below_threshold_and_rejects_with_its_finding():
    verdict = rc.deterministic_verdict(
        _judgement(RIGHT), NUMBER_TASK, RIGHT, runtime_spec(NUMBER_TASK)
    )
    gate = StubGate(0.0738)

    fresh, row = rc.apply_learned_gate(
        gate, "t1", NUMBER_TASK, RIGHT, verdict, 0.5
    )

    assert row["scored"] is True and row["fired"] is True and row["p"] == 0.0738
    assert verifier_rejects(fresh, Policy())
    assert fresh.payload["detectors"] == ["learned_gate"]
    assert fresh.payload["gate"]["p"] == 0.0738       # the D3 row carries p
    assert "learned gate" in fresh.payload["reason"]
    assert "0.074" in fresh.payload["reason"]
    assert "40" not in str(fresh.payload["findings"])  # gold-free by construction
    assert float(fresh.latency_ms) > float(verdict.latency_ms)


def test_gate_stays_silent_above_threshold():
    verdict = rc.deterministic_verdict(
        _judgement(RIGHT), NUMBER_TASK, RIGHT, runtime_spec(NUMBER_TASK)
    )
    gate = StubGate(0.93)

    out, row = rc.apply_learned_gate(
        gate, "t1", NUMBER_TASK, RIGHT, verdict, 0.5
    )

    assert out.decision == "accept"          # unchanged in meaning
    assert out.confidence == verdict.confidence
    assert out.payload.get("detectors") == []
    assert out.payload["gate"]["p"] == 0.93     # silent scores are recorded too
    assert float(out.latency_ms) > float(verdict.latency_ms)  # its ms is counted
    assert row["scored"] is True and row["fired"] is False and row["p"] == 0.93


def test_gate_finding_digest_line():
    digest = rc.finding_digest([_gate_finding()])

    assert "learned gate" in digest
    assert "0.074" in digest
    assert "0.50" in digest


def test_repair_prompt_for_the_gate_rejection():
    detail, text = rc.repair_user_text(NUMBER_TASK, WRONG, [_gate_finding()])

    assert detail == "learned-gate"
    assert "0.074" in text and "0.50" in text      # p vs threshold quoted
    assert "inconsistency" in text                 # the ask: an internal recheck
    assert WRONG in text                           # the draft is quoted
    assert "40" not in text                        # never the answer key


def test_repair_rescored_strictly_when_the_gate_fired():
    rs = runtime_spec(NUMBER_TASK)
    tier = ScriptedTier([RIGHT])
    rescored: list[str] = []

    def rescore(text):
        rescored.append(text)
        return {"p": 0.93, "ms": 9.0, "threshold": 0.5}

    record = rc.attempt_repair(
        tier, NUMBER_TASK, WRONG, [_gate_finding()], rs,
        max_tokens=512, rescore=rescore,
    )

    assert record["cleared"] is True
    assert record["gate_rescored"] is True
    assert record["gate_cleared"] is True and record["gate_p2"] == 0.93
    assert len(rescored) == 1


def test_gate_fired_repair_that_still_scores_low_stays_blocked():
    rs = runtime_spec(NUMBER_TASK)
    tier = ScriptedTier([RIGHT])   # the repair passes the detectors ...
    rescore = lambda text: {"p": 0.1, "ms": 9.0, "threshold": 0.5}  # noqa: E731

    record = rc.attempt_repair(
        tier, NUMBER_TASK, WRONG, [_gate_finding()], rs,
        max_tokens=512, rescore=rescore,
    )

    assert record["cleared"] is False
    assert record["gate_cleared"] is False
    assert record["still_flagged"] == []           # the detectors are happy
    assert record["canonical"] == "Final answer: 24"  # ... but the gate is not
    assert "learned gate still scores" in record["stopped"]


def test_repair_rescore_is_skipped_when_detectors_still_flag():
    rs = runtime_spec(NUMBER_TASK)
    tier = ScriptedTier([WRONG])   # repeat the draft: consistency flags again
    rescored: list[str] = []

    def rescore(text):
        rescored.append(text)
        return {"p": 0.93, "ms": 9.0, "threshold": 0.5}

    record = rc.attempt_repair(
        tier, NUMBER_TASK, WRONG, [_gate_finding()], rs,
        max_tokens=512, rescore=rescore,
    )

    assert record["cleared"] is False
    assert record["still_flagged"] == ["consistency"]
    assert rescored == []  # the detectors remain the objection; no score needed


def test_open_gate_defaults_to_off():
    assert rc.open_gate(rc.build_parser().parse_args([])) is None


def test_open_gate_flags_and_construction():
    args = rc.build_parser().parse_args(
        ["--gate", "intern-0.8b", "--judge", "deterministic"]
    )
    assert args.gate_head == "L0.75"
    assert args.gate_threshold == 0.5

    err = io.StringIO()
    gate = rc.open_gate(args, err=err)
    if gate is None:
        pytest.skip("heads npz not present (head load failed -> gate off)")
    assert gate.head.name == "L0.75"
    assert gate.model_id == "internlm/Intern-Decision-0.8B"
    assert "learned gate on" in err.getvalue()


def test_open_gate_refuses_scion_and_the_wrong_judge():
    scion_args = rc.build_parser().parse_args(
        ["--gate", "scion", "--judge", "deterministic"]
    )
    err = io.StringIO()
    assert rc.open_gate(scion_args, err=err) is None
    assert "not wired yet" in err.getvalue()

    tiny_args = rc.build_parser().parse_args(["--gate", "intern-0.8b"])
    err = io.StringIO()   # default judge is tiny-jev -> refused
    assert rc.open_gate(tiny_args, err=err) is None
    assert "deterministic" in err.getvalue()
