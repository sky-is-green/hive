"""The optional browser adapter is reachable from the terminal."""

from __future__ import annotations

import json

from harness import browser_adapter, cli


def test_browser_status_reports_availability(capsys):
    assert cli.main(["browser", "status", "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert "available" in body


def test_browser_browse_renders_json_and_errors(monkeypatch, capsys, tmp_path):
    state = str(tmp_path / "state")

    def fake_browse(url, **kwargs):
        assert kwargs["allow_private"] is True
        assert kwargs["session_id"] == "s1"
        return {"ok": True, "url": url, "title": "Title", "text": "hello",
                "screenshot": None, "blocked": [], "requests": []}

    monkeypatch.setattr(browser_adapter, "browse", fake_browse)
    assert cli.main(["browser", "browse", "http://127.0.0.1:9/",
                     "--allow-private", "--session", "s1",
                     "--state-dir", state, "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["title"] == "Title"

    def fail(url, **kwargs):
        from harness.web_fetch import WebFetchError

        raise WebFetchError("private origin needs a grant",
                            code="private_origin", status=403)

    monkeypatch.setattr(browser_adapter, "browse", fail)
    assert cli.main(["browser", "browse", "http://127.0.0.1:9/",
                     "--state-dir", state]) == 1
    assert "private origin" in capsys.readouterr().err
