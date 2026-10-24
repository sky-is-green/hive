"""The interactive dashboard must never take over a pipe or a test capture."""
from harness import tui


def test_run_falls_back_when_disabled(monkeypatch):
    monkeypatch.setenv("HIVE_NO_TUI", "1")
    assert tui.run() == 1


def test_run_falls_back_without_a_tty(monkeypatch):
    monkeypatch.delenv("HIVE_NO_TUI", raising=False)
    # pytest captures stdin/stdout, so this is the pipe case
    assert tui.run() == 1
