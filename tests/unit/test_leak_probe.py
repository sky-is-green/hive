"""The leak probe's judgement (no model, no I/O).

The probe itself loads a real model, so it is not part of the suite; the
judgement it uses to call a leak is pure and is tested here.
"""

from __future__ import annotations

from experiments.leak_probe import detect_leak, is_reclaimed


def test_reclaimed_within_the_tolerance():
    before = {"vramMb": 1000, "ramMb": 2000, "timestamp": 1}
    after = {"vramMb": 1020, "ramMb": 2050, "timestamp": 2}
    assert is_reclaimed(before, after) is True


def test_a_real_leak_is_not_reclaimed():
    before = {"vramMb": 1000, "ramMb": 2000, "timestamp": 1}
    after = {"vramMb": 2000, "ramMb": 3000, "timestamp": 2}
    assert is_reclaimed(before, after) is False


def test_the_slack_covers_the_small_numbers():
    """A box with a little memory gets the slack, not a false alarm."""
    before = {"vramMb": 40, "ramMb": 60, "timestamp": 1}
    after = {"vramMb": 60, "ramMb": 80, "timestamp": 2}
    assert is_reclaimed(before, after) is True


def test_detect_leak_over_a_history():
    history = [
        {"vramMb": 1000, "ramMb": 2000, "timestamp": 1},
        {"vramMb": 1100, "ramMb": 2100, "timestamp": 2},
        {"vramMb": 1200, "ramMb": 2200, "timestamp": 3},
    ]
    assert detect_leak(history) is True


def test_a_clean_history_is_not_a_leak():
    history = [
        {"vramMb": 1000, "ramMb": 2000, "timestamp": 1},
        {"vramMb": 1010, "ramMb": 2010, "timestamp": 2},
    ]
    assert detect_leak(history) is False


def test_one_sample_is_not_a_leak():
    assert detect_leak([{"vramMb": 1000, "ramMb": 2000, "timestamp": 1}]) is False
