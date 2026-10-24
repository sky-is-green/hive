"""The terminal presentation layer: plain in pipes, coloured only on request."""
import pytest

from harness import termui


@pytest.fixture(autouse=True)
def _reset_ui(monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    termui.set_color(None)
    yield
    termui.set_color(None)


def test_pipes_get_no_ansi(capsys):
    assert termui.color_enabled() is False
    text = termui.ok("done")
    assert "done" in text
    assert "\x1b" not in text


def test_forced_colour_and_forced_off(capsys):
    termui.set_color(True)
    assert "\x1b" in termui.accent("hive")
    termui.set_color(False)
    assert "\x1b" not in termui.ok("hive")


def test_table_is_plain_without_a_tty(capsys):
    termui.table(["name", "size"], [["a", "1"]])
    out = capsys.readouterr().out
    assert "NAME" in out and "a" in out and "1" in out
    assert "\x1b" not in out


def test_spinner_is_silent_without_a_tty(capsys):
    with termui.spinner("working"):
        pass
    assert capsys.readouterr().err == ""


def test_banner_is_plain_without_a_tty(capsys):
    termui.banner("0.1.0", [("models", "9"), ("serves", "none")],
              ["hive doctor", "hive status"])
    out = capsys.readouterr().out
    assert "hive" in out and "0.1.0" in out
    assert "hive doctor" in out
    assert "\x1b" not in out


def test_symbols_fall_back_to_ascii_when_the_encoding_cannot_hold_them():
    class _Ascii:
        encoding = "ascii"

    assert termui.symbols(_Ascii())["ok"] == "[ok]"
    assert termui.symbols(_Ascii())["hex"] == "#"
