#!/usr/bin/env python3
"""Disposable serve for the manual Hive UI release gate.

This does not start an engine.  It creates one real serve record, one live
process, a tiny OpenAI-compatible ``/models`` endpoint, and a log file under a
temporary HIVE_STATE_DIR.  The TUI can then exercise Status, tier health,
details, logs, stop confirmation and stop without loading a model.

  python scripts/ui_gate_fixture.py --state /tmp/hive-ui-gate
  # in another terminal:
  HIVE_STATE_DIR=/tmp/hive-ui-gate hive tui

Press Ctrl-C (or let the TUI stop it) to clean up.  The fixture removes only
its own record.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ModelsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib spelling
        if self.path.rstrip("/") == "/models":
            body = b'{"data":[{"id":"ui-gate-demo"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args) -> None:
        return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default="/tmp/hive-ui-gate",
                    help="temporary HIVE_STATE_DIR (default: /tmp/hive-ui-gate)")
    ap.add_argument("--port", type=int, default=0,
                    help="fake endpoint port (default: a free port)")
    ap.add_argument("--name", default="ui-gate-demo",
                    help="serve name shown in the TUI")
    args = ap.parse_args()

    state = Path(args.state).expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    os.environ["HIVE_STATE_DIR"] = str(state)
    from harness import serve_state

    port = args.port or _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    log = serve_state.log_path(args.name)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} supervisor: UI gate fixture starting\n"
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} tier face: listening on 127.0.0.1:{port}\n"
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} tier face: ready (fake endpoint)\n",
        encoding="utf-8",
    )
    pid = os.getpid()
    try:
        cmdline = (Path(f"/proc/{pid}/cmdline").read_bytes()
                   .replace(b"\0", b" ").decode().strip())
    except OSError:
        cmdline = ""  # Windows: pid_matches falls back to "alive"

    record = {
        "name": args.name,
        "pid": pid,
        "mode": "fixture",
        "stack": "ui-gate-fixture",
        "started_at": time.time(),
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cmdline": cmdline,
        "host": "127.0.0.1",
        "log_path": str(log),
        "tiers": [{
            "role": "face",
            "model": "ui-gate-demo",
            "port": port,
            "ctx": 8192,
            "backend": "fixture",
            "base_url": f"http://127.0.0.1:{port}",
        }],
    }
    serve_state.write_record(record)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_args: stop.set())

    print(f"HIVE_STATE_DIR={state}")
    print(f"fixture: {args.name} pid={pid} endpoint=http://127.0.0.1:{port}")
    print("open:   HIVE_STATE_DIR=%s hive tui" % state)
    print("stop:   Ctrl-C here, or press x then y in the TUI")
    try:
        while not stop.wait(0.5):
            pass
    finally:
        current = serve_state.read_record(args.name)
        if current and int(current.get("pid") or 0) == pid:
            serve_state.remove_record(args.name)
        server.shutdown()
        server.server_close()
        print("fixture stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
