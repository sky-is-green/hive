"""The task checker must not reject a correct answer's formatting."""

from experiments.cascade.checkers import check


def test_boxed_number_with_latex_thousands_separators():
    task = {"checker": {"type": "boxed", "expect": "6290000"}}
    answer = r"The value rounds to \(6{,}290{,}000\).\n\n\[\boxed{6{,}290{,}000}\]"
    assert check(task, answer)


def test_boxed_fraction_is_unchanged():
    task = {"checker": {"type": "boxed", "expect": "-40/153"}}
    answer = r"\[\boxed{-\frac{40}{153}}\]"
    assert check(task, answer)
