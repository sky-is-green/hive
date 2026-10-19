"""``/import`` — showing the LLM a file.

The command attaches a text file to the conversation: the text goes into the
conversation's context store (Splinter mode) and rides along with the next
agent turn (dsh mode).  Binary files and missing files are refused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.app import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        providers_file=tmp_path / "providers.local.json",
        engines_file=tmp_path / "engines.local.json",
        runs_root=tmp_path / "runs",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
    )
    return TestClient(app)


def _conv_file(state_dir: Path, conversation_id: str) -> Path:
    digest = hashlib.md5(conversation_id.encode("utf-8")).hexdigest()[:16]
    return state_dir / f"conv-{digest}.json"


def test_import_is_discoverable(client):
    names = {c["name"] for c in client.get("/v1/commands").json()["commands"]}
    assert "import" in names


def test_import_without_a_path_explains_itself(client):
    r = client.post("/v1/commands/run", json={"line": "/import", "conversation_id": "c"})
    body = r.json()
    assert body["kind"] == "error"
    assert "/import <path>" in body["text"]


def test_import_refuses_a_missing_file(client):
    r = client.post("/v1/commands/run",
                    json={"line": "/import /nope/missing.txt", "conversation_id": "c"})
    body = r.json()
    assert body["kind"] == "error"
    assert "no such file" in body["text"]


def test_import_refuses_a_binary_file(client, tmp_path):
    blob = tmp_path / "thing.bin"
    blob.write_bytes(b"\x00\x01\x02\x03binary")
    r = client.post("/v1/commands/run",
                    json={"line": f"/import {blob}", "conversation_id": "c"})
    body = r.json()
    assert body["kind"] == "error"
    assert "binary" in body["text"]


def test_import_attaches_the_text_to_the_conversation(client, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("the quick brown fox jumps over the lazy dog\n", encoding="utf-8")
    cid = "import-test"
    r = client.post("/v1/commands/run",
                    json={"line": f"/import {note}", "conversation_id": cid})
    body = r.json()
    assert body["kind"] == "success", body
    assert "note.txt" in body["text"]

    # the store holds the file's text
    state_dir = Path(client.app.state.harness.state_dir)
    conv = json.loads(_conv_file(state_dir, cid).read_text(encoding="utf-8"))
    contents = [c["content"] for c in conv["store"]["chunks"]]
    joined = "\n".join(contents)
    assert "quick brown fox" in joined
    assert "attached file: note.txt" in joined


def test_import_queues_the_context_for_the_next_agent_turn(client, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("context for the agent\n", encoding="utf-8")
    cid = "import-agent"
    client.post("/v1/commands/run",
                json={"line": f"/import {note}", "conversation_id": cid})
    agent = client.app.state.agent
    queued = agent._pending_context.get(cid) or []
    assert any("context for the agent" in q for q in queued)


def test_queued_context_rides_along_with_the_next_turn(tmp_path):
    """The agent service hands the queued context to the next turn."""
    from harness.agent import DshAgentService

    service = DshAgentService(default_cwd=tmp_path, session_root=tmp_path / "sessions")
    service.queue_context("c1", "file contents here")
    assert service._take_context("c1") .startswith("file contents here")
    # it is taken once: the next turn does not repeat it
    assert service._take_context("c1") == ""
    # another conversation is unaffected
    assert service._take_context("c2") == ""


def test_import_redacts_secrets(client, tmp_path):
    secret = tmp_path / "creds.txt"
    secret.write_text("key sk-ABCDEFGHIJKLMNOPQRSTUVWX here\n", encoding="utf-8")
    cid = "import-secret"
    client.post("/v1/commands/run",
                json={"line": f"/import {secret}", "conversation_id": cid})
    state_dir = Path(client.app.state.harness.state_dir)
    conv = json.loads(_conv_file(state_dir, cid).read_text(encoding="utf-8"))
    joined = "\n".join(c["content"] for c in conv["store"]["chunks"])
    assert "sk-ABCDEFGHIJKLMNOPQRSTUVWX" not in joined
    assert "[redacted-secret]" in joined
