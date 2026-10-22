"""The execution checker must fail wrong work, pass right work, and test itself."""

import shutil

import pytest

from experiments.cascade.verifier import (
    boxed,
    check_grounding,
    find_equations,
    run_sandboxed,
    runtime_spec,
    verify,
)

HAS_BWRAP = shutil.which("bwrap") is not None


def number_task(answer):
    spec = {"checker": {"type": "number", "expect": 12.0}}
    return verify(spec, answer)


def test_arithmetic_passes_and_mutation_is_caught():
    verdict = number_task("2 * $6 = $12\n\nFinal answer: 12")
    assert verdict["verdict"] == "pass"
    assert verdict["mutated_ok"] is True
    assert verdict["latency_ms"] >= 0


def test_unit_conversion_error_is_caught():
    verdict = number_task(
        "700 * 5 cents = 3500 cents\n"
        "1400 * 15 cents = 21000 cents\n"
        "3500 + 21000 = 24500 cents = $24.50\n"
        "\nFinal answer: 24.50"
    )
    assert verdict["verdict"] == "fail"
    assert "24500" in verdict["evidence"]
    assert verdict["mutated_ok"] is None


def test_rate_label_is_not_a_false_reject():
    # "15 - 4 = 11 boxes * $15" is a count times a rate, not an equality.
    verdict = number_task(
        "15 - 4 = 11 boxes * $15 = $165\n"
        "Final answer: 165"
    )
    assert verdict["verdict"] == "pass"


def test_no_checkable_equation_is_unknown_not_pass():
    verdict = number_task("I rounded the value to 6,290,000.\n\nFinal answer: 6290000")
    assert verdict["verdict"] == "unknown"
    assert verdict["mutated_ok"] is None


def test_fraction_product_is_caught():
    spec = {"checker": {"type": "boxed", "expect": "40/243"}}
    verdict = verify(
        spec,
        r"10 \cdot \frac{4}{9} \cdot \frac{1}{27} = \frac{40}{81}"
        "\n\n\\boxed{\\frac{40}{81}}",
    )
    assert verdict["verdict"] == "fail"
    assert "40/243" in verdict["evidence"] or "false" in verdict["evidence"]


def test_runtime_spec_drops_the_answer_key():
    public = runtime_spec({"checker": {"type": "number", "expect": 12,
                                       "tests": "assert True"}})
    assert "expect" not in public
    assert public == {"kind": "number", "tests": "assert True"}


def test_grounding_requires_a_citation():
    assert check_grounding("The sky is blue [1]. Water is wet [2].")["verdict"] == "pass"
    assert check_grounding("The sky is blue.")["verdict"] == "unknown"


def test_grounding_pass_has_a_mutation():
    verdict = verify(
        {"kind": "text"},
        "The sky is blue [1]. Water is wet [2].",
        grounding=True,
    )
    assert verdict["verdict"] == "pass"
    assert verdict["method"] == "grounding"
    assert verdict["mutated_ok"] is True


@pytest.mark.skipif(not HAS_BWRAP, reason="bwrap is needed for the code sandbox")
def test_sandbox_blocks_network():
    run = run_sandboxed(
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(1)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 80))\n"
        "    print('NET-OK')\n"
        "except Exception as exc:\n"
        "    print('NET-BLOCKED', type(exc).__name__)\n"
    )
    assert run["available"] is True
    assert "NET-BLOCKED" in run["stdout"]


@pytest.mark.skipif(not HAS_BWRAP, reason="bwrap is needed for the code sandbox")
def test_code_pass_and_mutation():
    spec = {
        "checker": {
            "type": "code",
            "tests": "assert double(2) == 4\nprint('PASS')",
            "footer": "print('PASS')",
        }
    }
    verdict = verify(spec, "```python\ndef double(x):\n    return x * 2\n```")
    assert verdict["verdict"] == "pass"
    assert verdict["mutated_ok"] is True


@pytest.mark.skipif(not HAS_BWRAP, reason="bwrap is needed for the code sandbox")
def test_code_failure_is_reported():
    spec = {
        "checker": {
            "type": "code",
            "tests": "assert double(2) == 4\nprint('PASS')",
            "footer": "print('PASS')",
        }
    }
    verdict = verify(spec, "```python\ndef double(x):\n    return x + 1\n```")
    assert verdict["verdict"] == "fail"
    assert verdict["mutated_ok"] is None


@pytest.mark.skipif(not HAS_BWRAP, reason="bwrap is needed for the code sandbox")
def test_weak_tests_are_unknown_not_pass():
    spec = {
        "checker": {
            "type": "code",
            "tests": "print('PASS')",
            "footer": "print('PASS')",
        }
    }
    verdict = verify(spec, "```python\ndef f(x):\n    return x\n```")
    assert verdict["verdict"] == "unknown"
    assert verdict["mutated_ok"] is False


def test_boxed_parser_handles_thousands_separators():
    assert boxed(r"\boxed{6{,}290{,}000}") == r"6{,}290{,}000"


def test_find_equations_skips_prose_fragments():
    equations = find_equations("On day 3 he saw 0 birds.\n3 + 4 = 7")
    assert any(equation["equation"].startswith("3 + 4") for equation in equations)
    assert all("day" not in equation["equation"].lower() for equation in equations)
