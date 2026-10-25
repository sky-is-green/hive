"""The interactive dashboard must never take over a pipe or a test capture."""
from harness import tui


def test_run_falls_back_when_disabled(monkeypatch):
    monkeypatch.setenv("HIVE_NO_TUI", "1")
    assert tui.run() == 1


def test_run_falls_back_without_a_tty(monkeypatch):
    monkeypatch.delenv("HIVE_NO_TUI", raising=False)
    # pytest captures stdin/stdout, so this is the pipe case
    assert tui.run() == 1


def test_help_names_the_actions_and_the_resource_strip():
    help_text = "\n".join(tui.HELP_LINES)
    for token in ("switch tab", "ENTER", "supervisor log", "stop selected",
                  "confirmation", "refresh", "RAM free", "VRAM free",
                  "GTT used", "quit"):
        assert token in help_text


def test_fit_text_keeps_the_suffix_for_long_model_names():
    assert tui.fit_text("short", 8) == "short   "
    assert tui.fit_text("Qwen3.8-Flash-Next-00001-of-00002", 24) == (
        "Qwen3.8-Fla...1-of-00002"
    )
    assert tui.fit_text("abc", 2) == "ab"


def test_format_resources_is_compact_and_omits_missing_probes():
    line = tui.format_resources({
        "available_ram_gb": 24.0,
        "total_ram_gb": 30.0,
        "vram_free_gb": 1.4,
        "vram_gb": 21.5,
        "gtt_used_gb": 0.17,
        "gtt_total_gb": 29.4,
    })
    assert line == (
        "RAM 24.0/30.0 GiB free · VRAM 1.4/21.5 GiB free · GTT 0.2/29.4 GiB"
    )
    assert tui.format_resources({}) == ""
    assert tui.format_resources({"vram_gb": 21.5}) == ""
