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
