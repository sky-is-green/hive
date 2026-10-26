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
                  "confirmation", "tool approvals", "refresh", "RAM free",
                  "VRAM free", "GTT used", "quit"):
        assert token in help_text


def test_approval_lines_show_the_exact_action_and_scope():
    lines = tui.approval_lines({
        "request_id": "req-1", "session_id": "conv-1",
        "tool_identity": "dsh:bash@1.0.0", "action": "bash: rm -rf build",
        "reason": "clean the build directory", "risk": "high",
        "network": "none", "filesystem": "workspace",
        "sandbox": "required", "expires_at": 9_999_999_999,
    }, width=60)
    text = "\n".join(lines)
    assert "req-1" in text
    assert "dsh:bash@1.0.0" in text
    assert "bash: rm -rf build" in text
    assert "clean the build directory" in text
    assert "high risk" in text
    assert "sandbox required" in text
    assert all(len(line) <= 60 for line in lines)


def _bare_app():
    app = object.__new__(tui.App)
    app.mode = "main"
    app.approvals = []
    app.approval_index = 0
    app.approval_error = ""
    app.message = ""
    return app


def test_tui_refresh_and_decide_use_the_live_studio(monkeypatch):
    app = _bare_app()
    calls = []

    def fake_call(path, *, method="GET", body=None, timeout=5.0):
        calls.append((path, method, body))
        if path == "/v1/tools/pending":
            return {"requests": [{
                "request_id": "req-1", "session_id": "conv-1",
                "tool_identity": "dsh:bash@1.0.0", "action": "bash: true",
                "reason": "verify the hook", "risk": "high",
                "network": "none", "filesystem": "workspace",
                "sandbox": "required", "expires_at": 9_999_999_999,
            }]}
        return {"status": "approved"}

    monkeypatch.setattr(tui.studio_api, "call", fake_call)
    app._open_approvals()
    assert app.mode == "approvals"
    assert app.approvals[0]["request_id"] == "req-1"
    app._decide_approval("once", "session")
    assert ("/v1/tools/decide", "POST",
            {"session_id": "conv-1", "request_id": "req-1",
             "choice": "once", "scope": "session"}) in calls
    assert "allowed once" in app.message


def test_tui_reports_a_missing_studio_without_crashing(monkeypatch):
    app = _bare_app()

    def fail(*args, **kwargs):
        raise tui.studio_api.StudioAPIError("Studio gateway unreachable")

    monkeypatch.setattr(tui.studio_api, "call", fail)
    app._open_approvals()
    assert app.approvals == []
    assert "unreachable" in app.approval_error


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
