"""The PreToolUse hook is fail-closed and maps DSH tool names clearly."""

from __future__ import annotations

import json

import pytest

from harness import tool_hook


def _payload(name="bash", **extra):
    data = {
        "hook_event_name": "PreToolUse",
        "session_id": "conv-1",
        "tool_name": name,
        "tool_input": {"command": "rg -n TODO .", "reason": "find TODOs"},
    }
    data.update(extra)
    return data


def test_payload_maps_to_a_concrete_request():
    request = tool_hook._request(_payload())
    assert request["session_id"] == "conv-1"
    assert request["tool"]["source"] == "dsh"
    assert request["tool"]["side_effect"] is True
    assert request["action"].startswith("bash: rg -n TODO")
    assert request["reason"] == "find TODOs"


def test_read_only_tools_are_marked_read_only():
    request = tool_hook._request(_payload("read_file", tool_input={"path": "a.txt"}))
    assert request["tool"]["read_only"] is True
    assert request["tool"]["side_effect"] is False


def _web(url):
    return tool_hook._request({
        "hook_event_name": "PreToolUse", "session_id": "conv-web",
        "tool_name": "web_fetch", "tool_input": {"url": url},
    })


def test_public_web_fetch_is_public_and_read_only():
    request = _web("https://example.com/docs")
    assert request["tool"]["network"] == "public"
    assert request["tool"]["read_only"] is True
    assert request["resource"] == "https://example.com"


def test_the_resource_is_the_exact_origin():
    request = _web("http://127.0.0.1:8765/server?x=1")
    assert request["resource"] == "http://127.0.0.1:8765"
    ipv6 = _web("http://[::1]:8765/")
    assert ipv6["resource"] == "http://[::1]:8765"


def test_a_multi_url_call_gets_the_exact_origin_set():
    request = tool_hook._request(_payload("web_fetch", tool_input={
        "urls": ["https://example.com/a", "https://example.org/b"]}))
    assert request["resource"] == "https://example.com,https://example.org"


def test_loopback_literal_is_private():
    request = _web("http://127.0.0.1:8765/server")
    assert request["tool"]["network"] == "private"
    assert request["tool"]["risk"] == "high"


def test_private_and_metadata_hosts_are_private():
    for url in (
        "http://localhost:8080",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data",
        "http://10.1.2.3/",
        "http://192.168.1.1/",
        "http://metadata.google.internal/",
        "file:///etc/passwd",
    ):
        assert _web(url)["tool"]["network"] == "private", url


def test_a_hostname_resolving_to_loopback_is_private(monkeypatch):
    monkeypatch.setattr(
        tool_hook.socket, "getaddrinfo",
        lambda host, port: [(2, 1, 6, "", ("127.0.0.1", 0))])
    assert _web("https://rebind.example/")["tool"]["network"] == "private"


def test_a_hostname_resolving_to_a_public_address_is_public(monkeypatch):
    monkeypatch.setattr(
        tool_hook.socket, "getaddrinfo",
        lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))])
    assert _web("https://example.com/")["tool"]["network"] == "public"


def test_web_fetch_without_a_destination_is_private():
    request = tool_hook._request(_payload("web_fetch", tool_input={}))
    assert request["tool"]["network"] == "private"


def test_a_url_list_is_private_if_any_member_is_private():
    request = tool_hook._request(_payload("web_fetch", tool_input={
        "urls": ["https://example.com", "http://127.0.0.1:8765"]}))
    assert request["tool"]["network"] == "private"


def test_web_search_with_a_query_is_public():
    request = tool_hook._request(_payload("web_search",
                                          tool_input={"query": "hive docs"}))
    assert request["tool"]["network"] == "public"


def test_approved_call_exits_zero():
    out = tool_hook.decide(
        _payload(),
        post=lambda url, body: {"status": "approved"},
        get=lambda url: {},
        sleep=lambda _: None,
    )
    assert out["status"] == "approved"


def test_pending_then_approved_waits_for_decision():
    calls = []

    def get(url):
        calls.append(url)
        return {"status": "approved"}

    out = tool_hook.decide(
        _payload(),
        post=lambda url, body: {"status": "pending", "request_id": "req-1"},
        get=get,
        sleep=lambda _: None,
        timeout_s=2,
    )
    assert out["status"] == "approved"
    assert calls and "session_id=conv-1" in calls[0]


def test_denied_decision_blocks():
    out = tool_hook.decide(
        _payload(),
        post=lambda url, body: {"status": "pending", "request_id": "req-1"},
        get=lambda url: {"status": "denied", "reason": "user denied"},
        sleep=lambda _: None,
        timeout_s=2,
    )
    assert out["status"] == "denied"
    assert out["reason"] == "user denied"


def test_missing_gateway_is_fail_closed(capsys, monkeypatch):
    class Input:
        def read(self):
            return json.dumps(_payload())

    monkeypatch.setattr("sys.stdin", Input())
    monkeypatch.setattr(tool_hook, "decide",
                        lambda payload: (_ for _ in ()).throw(
                            tool_hook.HookError("Hive gateway is unreachable")))
    assert tool_hook.main([]) == 2
    assert "blocked:" in capsys.readouterr().err
