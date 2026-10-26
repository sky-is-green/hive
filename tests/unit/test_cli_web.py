"""The secure fetch policy and grants are usable from the terminal."""

from __future__ import annotations

import json

from harness import cli, web_fetch


def test_web_status_and_policy_round_trip(tmp_path, capsys):
    state = str(tmp_path / "state")
    assert cli.main(["web", "status", "--state-dir", state, "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["policy"]["allow_downloads"] is False

    assert cli.main(["web", "policy", "--allow-downloads",
                     "--state-dir", state, "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["allow_downloads"] is True

    assert cli.main(["web", "policy", "--state-dir", state, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["allow_downloads"] is True


def test_web_grants_and_revoke(tmp_path, capsys):
    state = str(tmp_path / "state")
    assert cli.main(["web", "grant", "https://example.com",
                     "--access", "private", "--scope", "global",
                     "--state-dir", state, "--json"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["origin"] == "https://example.com"
    assert row["access"] == "private"

    assert cli.main(["web", "grants", "--state-dir", state, "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["origin"] == "https://example.com"

    assert cli.main(["web", "revoke", "https://example.com",
                     "--state-dir", state, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["global"] == 1


def test_web_fetch_renders_json_and_errors(tmp_path, capsys, monkeypatch):
    state = str(tmp_path / "state")

    def fake_fetch(url, **kwargs):
        assert kwargs["allow_private"] is True
        assert kwargs["session_id"] == "s1"
        return {"ok": True, "status": 200, "url": url,
                "content_type": "text/plain", "body": "hello", "bytes": 5,
                "headers": {}, "upload": None}

    monkeypatch.setattr(web_fetch, "fetch", fake_fetch)
    assert cli.main(["web", "fetch", "http://127.0.0.1:9/",
                     "--allow-private", "--session", "s1",
                     "--state-dir", state, "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["body"] == "hello"

    def fail(url, **kwargs):
        raise web_fetch.WebFetchError("private origin needs a grant",
                                      code="private_origin", status=403)

    monkeypatch.setattr(web_fetch, "fetch", fail)
    assert cli.main(["web", "fetch", "http://127.0.0.1:9/",
                     "--state-dir", state]) == 1
    assert "private origin" in capsys.readouterr().err
