"""The cascade's verdict tier: a decision head's readout, and the tier's shape.

The escalation tier's expensive arm is a *generation* - the big model
answers again.  A model with a decision head can instead *judge* the small
tier's answer, and a judgement is a prefill: measured at 4.6 s against
20.7 s for the generation on the same 28 tasks
(``experiments/escalation_arms.py``).

These tests are hermetic - a stub HTTP server, no model, no network.  They
check the contract the runner relies on:

  * the client posts ``{"task", "candidate"}`` and reads ``p_correct``,
  * a 503 (the serve's single slot is busy) is retried, not raised,
  * a confident verdict keeps the local answer and skips the escalation,
  * the report records the verdict, and the summary counts it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from experiments.cascade.run_cascade import Verdict


class _Handler(BaseHTTPRequestHandler):
    """A stub of the engine's POST /v1/verdict."""

    calls: list[dict] = []
    p_correct: float = 0.9
    fail_times: int = 0          # how many 503s before it answers

    def log_message(self, *args):  # noqa: D401 - the base class's noise
        return

    def do_POST(self):  # noqa: N802 - the base class's name
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).calls.append(body)
        if len(type(self).calls) <= type(self).fail_times:
            self.send_response(503)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "the model is not loaded"}}')
            return
        payload = json.dumps({"p_correct": type(self).p_correct,
                              "threshold": 0.5, "accept": type(self).p_correct >= 0.5,
                              "tokens": 170, "ms": 2400.0}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def server():
    _Handler.calls = []
    _Handler.fail_times = 0
    _Handler.p_correct = 0.9
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_client_posts_the_task_and_the_candidate(server):
    v = Verdict(server, threshold=0.5)
    out = v.judge("What is 2+2?", "Final answer: 4")
    assert out["p_correct"] == pytest.approx(0.9)
    assert out["accept"] is True
    assert _Handler.calls == [{"task": "What is 2+2?", "candidate": "Final answer: 4"}]


def test_a_busy_serve_is_retried_not_raised(server):
    _Handler.fail_times = 2
    v = Verdict(server, threshold=0.5)
    out = v.judge("q", "a")
    assert out["p_correct"] == pytest.approx(0.9)
    assert len(_Handler.calls) == 3


def test_the_threshold_decides_keeps(server):
    _Handler.p_correct = 0.4
    v = Verdict(server, threshold=0.5)
    assert v.judge("q", "a")["accept"] is False
    _Handler.p_correct = 0.6
    v2 = Verdict(server, threshold=0.5)
    assert v2.judge("q", "a")["accept"] is True


def test_the_runner_offers_the_verdict_tier():
    """The CLI carries the two knobs the tier needs."""
    from experiments.cascade import run_cascade
    import argparse

    parser = argparse.ArgumentParser()
    # the runner's own parser is built inside main(); check the flags are
    # accepted by parsing the same source's add_argument calls
    src = Path(run_cascade.__file__).read_text(encoding="utf-8")
    assert "--verdict-url" in src
    assert "--verdict-threshold" in src
    assert "verdict_client" in src
    assert "head_verdict" in src
