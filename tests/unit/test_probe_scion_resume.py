"""probe_scion.py must survive an engine restart and resume, not lose the run.

Window B lost 93 minutes of embedding when the decision-head server SIGTERMed and auto-restarted a
frozen engine: the probe held every state in RAM and exited on the dropped connection, writing
nothing.  These tests pin the three behaviours that make the window survivable -- retry after a
dropped connection (waiting for /health), a partial checkpoint on disk, and resuming from it -- by
running the real CLI against a fake engine that drops connections on demand.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
CASCADE = REPO / "experiments/cascade"
DIM = 2560
NITEMS = 12


class _Engine:
    """Minimal /health + /apply-template + /embedding server with injectable failures."""

    def __init__(self, questions: list[str], labels: list[int]) -> None:
        self.questions = questions
        self.labels = labels
        self.drop: set[int] = set()
        self.delay = 0.0
        self.healthy = True
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: ANN002 - silence the test output
                pass

            def _send(self, code: int, body) -> None:
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                self._send(200 if engine.healthy else 503, {"loaded": engine.healthy})

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/apply-template":
                    text = body["messages"][1]["content"]
                    # longest match wins: a short answer can be a substring of a longer one
                    idx = max((i for i, q in enumerate(engine.questions) if q in text),
                              key=lambda i: len(engine.questions[i]), default=-1)
                    self._send(200, {"prompt": f"idx::{idx}"})
                elif self.path == "/embedding":
                    idx = int(body["content"].split("::")[1])
                    if engine.delay:
                        time.sleep(engine.delay)
                    if idx in engine.drop:
                        engine.drop.discard(idx)
                        engine.healthy = False
                        threading.Timer(4.0, lambda: setattr(engine, "healthy", True)).start()
                        self.close_connection = True     # abrupt drop, like the real restart
                        return
                    vec = [float(((engine.labels[idx] + 1) * 100 + (j % 13)) / 200.0)
                           for j in range(DIM)]
                    self._send(200, [{"embedding": [vec]}])
                else:
                    self._send(404, {})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """A tiny judge bench with both splits and both labels in each (long answers only)."""
    items = []
    for split in ("train", "test"):
        for label in (0, 1):
            picked, seen = [], set()
            for line in (CASCADE / "results/partner-bonsai-organic-20261018/"
                                 "bonsai-transfer-bench.jsonl").read_text().splitlines():
                row = json.loads(line)
                # a one-letter multiple-choice answer ("A") is a substring of every body and
                # cannot identify its item, so keep the long free-form answers
                if (row["split"] == split and row["label"] == label
                        and len(row["candidate"]) > 200 and row["candidate"] not in seen):
                    seen.add(row["candidate"])
                    picked.append(row)
                if len(picked) == 3:
                    break
            items += picked
    path = tmp_path / "mini-bench.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in items) + "\n")
    assert len(items) == NITEMS
    return path


def _cli(port: int, bench: Path, out: Path, *extra: str) -> list[str]:
    return [sys.executable, str(CASCADE / "probe_scion.py"),
            "--server", f"http://127.0.0.1:{port}", "--bench", str(bench),
            "--out", str(out), "--name", "resumetest", *extra]


def _engine_for(bench: Path) -> _Engine:
    rows = [json.loads(l) for l in bench.read_text().splitlines() if l.strip()]
    return _Engine([r["candidate"] for r in rows], [r["label"] for r in rows])


def test_dropped_connection_is_retried_after_the_engine_comes_back(bench, tmp_path):
    engine = _engine_for(bench)
    try:
        engine.drop = {2, 5}
        proc = subprocess.run(_cli(engine.port, bench, tmp_path, "--checkpoint-every", "3"),
                              capture_output=True, text=True, timeout=180)
    finally:
        engine.close()
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads((tmp_path / "probe-resumetest.json").read_text())
    assert report["items_retried_after_engine_restart"] == 2
    states = np.load(tmp_path / "probe-states-resumetest.npz", allow_pickle=True)
    assert len(states["item_ids"]) == NITEMS
    assert not (tmp_path / "probe-states-resumetest.partial.npz").exists(), \
        "the partial checkpoint must be cleaned up once the run completes"


def test_killed_run_resumes_from_the_checkpoint(bench, tmp_path):
    engine = _engine_for(bench)
    partial = tmp_path / "probe-states-resumetest.partial.npz"
    try:
        engine.delay = 0.12                      # slow enough that there is a run left to kill
        proc = subprocess.Popen(_cli(engine.port, bench, tmp_path, "--checkpoint-every", "1"),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline, saved = time.time() + 20, 0
        while time.time() < deadline:
            if partial.exists():
                saved = len(np.load(partial, allow_pickle=True)["item_ids"])
                if saved >= 4:
                    break
            time.sleep(0.05)
        proc.kill()
        proc.wait(timeout=10)
        assert 0 < saved < NITEMS, "expected a partial checkpoint to survive the kill"

        # a clean reference run, with no failures injected
        engine.delay = 0.0
        clean = subprocess.run(_cli(engine.port, bench, tmp_path / "clean",
                                    "--checkpoint-every", "3"),
                               capture_output=True, text=True, timeout=180)
        assert clean.returncode == 0, clean.stderr[-2000:]

        resumed = subprocess.run(_cli(engine.port, bench, tmp_path, "--checkpoint-every", "3"),
                                 capture_output=True, text=True, timeout=180)
        assert resumed.returncode == 0, resumed.stderr[-2000:]
        assert "resumed" in resumed.stdout
    finally:
        engine.close()
    a = np.load(tmp_path / "clean/probe-states-resumetest.npz", allow_pickle=True)
    b = np.load(tmp_path / "probe-states-resumetest.npz", allow_pickle=True)
    assert list(b["item_ids"]) == list(a["item_ids"])
    assert np.array_equal(b["L1"], a["L1"]), "resumed vectors differ from the clean run"


def test_no_resume_ignores_the_checkpoint(bench, tmp_path):
    engine = _engine_for(bench)
    try:
        engine.drop = {1}
        first = subprocess.run(_cli(engine.port, bench, tmp_path, "--checkpoint-every", "1"),
                               capture_output=True, text=True, timeout=180)
        assert first.returncode == 0, first.stderr[-2000:]
        # forge a partial checkpoint: --no-resume must ignore it and re-embed everything
        np.savez_compressed(tmp_path / "probe-states-resumetest.partial.npz",
                            item_ids=[json.loads(bench.read_text().splitlines()[0])["item_id"]],
                            labels=[0], splits=["train"], L1=np.zeros((1, DIM), dtype=np.float32))
        second = subprocess.run(_cli(engine.port, bench, tmp_path, "--no-resume"),
                                capture_output=True, text=True, timeout=180)
    finally:
        engine.close()
    assert second.returncode == 0, second.stderr[-2000:]
    states = np.load(tmp_path / "probe-states-resumetest.npz", allow_pickle=True)
    assert not np.allclose(states["L1"][0], 0.0), "--no-resume kept the forged checkpoint row"