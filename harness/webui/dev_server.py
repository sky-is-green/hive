#!/usr/bin/env python3
"""Preview server for the new Hive console.

Serves harness/webui on :8170 and proxies:

- /api/*      -> http://127.0.0.1:8765/*
- /view/*    -> http://127.0.0.1:8765/view/*
- /probe/P/*  -> http://127.0.0.1:P/*

This exists only so the new shell can be reviewed without restarting live Hive
or adding CORS. Production will serve the same files from FastAPI and use the
real /v1 endpoints directly.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HIVE = "http://127.0.0.1:8765"
ROLES = ROOT / "roles.json"
TRACES = ROOT / "routing.jsonl"
PREFERENCES = ROOT / "preferences.jsonl"
RATIONALES = ROOT / "preference_rationales.jsonl"
JUDGE_MODEL = ROOT / "judge_model.json"
BENCH_DIR = ROOT.parents[1] / "experiments" / "model-matrix"
ROUTE_TARGETS = ROOT / "route_targets.json"
DEFAULT_ROUTE_TARGETS = {
    "simple": {"model": "lfm3", "engine": "lfm3-1.7b"},
    "medium": {"model": "scion35", "engine": "scion-35b"},
    "complex": {"model": "ds41", "engine": ""},
}
sys.path.insert(0, str(ROOT))
import judge as judge_lib
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "splinter" / "splinter"))
TIER = {"ultra_small": "simple", "medium": "medium", "escalation": "complex"}
MODEL_ENDPOINTS = {
    "lfm3": "http://127.0.0.1:8155",
    "ds41": "http://127.0.0.1:8150",
    "scion35": "http://127.0.0.1:8082",
    "scion176": "http://127.0.0.1:8081",
    "lfm3vl": "http://127.0.0.1:8160",
    "controller17": "http://127.0.0.1:8156",
    "reader17": "http://127.0.0.1:8159",
    "auditor17": "http://127.0.0.1:8162",
    "dispatcher17": "http://127.0.0.1:8163",
    "clerk17": "http://127.0.0.1:8164",
}
DEFAULT_ROLES = {
    "planner": "scion35",
    "controller": "controller17",
    "router": "scion35",
    "face": "scion35",
    "agency": "clerk17",
    "mechanics": "reader17",
    "judge": "d1-2b",
    "rag": "reader17",
    "risk": "guards",
    "vision": "lfm3vl",
    "placement": "auditor17",
    "critic": "auditor17",
    "dispatcher": "dispatcher17",
}
DEFAULT_PLACEMENTS = {
    "planner": "non_display",
    "controller": "display",
    "router": "non_display",
    "face": "non_display",
    "agency": "non_display",
    "mechanics": "display",
    "judge": "display",
    "rag": "display",
    "risk": "",
    "vision": "display",
    "placement": "non_display",
    "critic": "non_display",
    "dispatcher": "non_display",
}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _proxy(self, url: str, data: bytes | None = None) -> None:
        try:
            if data is None:
                length = int(self.headers.get("content-length") or 0)
                data = self.rfile.read(length) if length else None
            req = urllib.request.Request(
                url, data=data, method=self.command,
                headers={"content-type": self.headers.get("content-type", "application/json")},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("content-type", resp.headers.get("content-type", "application/json"))
                self.send_header("cache-control", "no-store")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            self.send_response(exc.code)
            self.send_header("content-type", exc.headers.get("content-type", "application/json"))
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:  # noqa: BLE001
            body = ('{"error":"proxy failed","detail":"%s"}' % str(exc).replace('"', "'")).encode()
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _proxy_stream(self, url: str, data: bytes | None = None) -> None:
        try:
            if data is None:
                length = int(self.headers.get("content-length") or 0)
                data = self.rfile.read(length) if length else None
            req = urllib.request.Request(
                url, data=data, method=self.command,
                headers={"content-type": self.headers.get("content-type", "application/json")},
            )
            resp = urllib.request.urlopen(req, timeout=3600)
            self.send_response(resp.status)
            self.send_header("content-type", resp.headers.get("content-type", "text/event-stream"))
            self.send_header("cache-control", "no-cache")
            # No content-length on a streamed response: close the connection
            # when the generator ends so fetch()/EventSource see EOF.
            self.send_header("connection", "close")
            self.end_headers()
            while True:
                line = resp.readline()
                if not line:
                    break
                self.wfile.write(line)
                self.wfile.flush()
                # OpenAI-style SSE ends with [DONE] but the upstream may keep
                # the HTTP connection alive; stop instead of waiting forever.
                if b"[DONE]" in line:
                    break
        except Exception as exc:  # noqa: BLE001
            try:
                body = ('{"error":"stream failed","detail":"%s"}' % str(exc).replace('"', "'")).encode()
                self.send_response(502)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                pass

    def _proxy_model(self, url: str) -> None:
        length = int(self.headers.get("content-length") or 0)
        data = self.rfile.read(length) if length else None
        try:
            streaming = bool(json.loads(data or b"{}").get("stream"))
        except (ValueError, AttributeError):
            streaming = False
        if streaming:
            self._proxy_stream(url, data=data)
        else:
            self._proxy(url, data=data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self) -> None:
        try:
            from cortex.routing import DroneRouter
            payload = self._read_json()
            query = str(payload.get("query") or "")
            d = DroneRouter().route(query)
            self._send_json({
                "tier": TIER.get(d.route_to, "complex"),
                "route_to": d.route_to,
                "confidence": d.confidence,
                "reason": d.reason,
                "router": "drone-heuristic",
            })
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    def _trace_post(self) -> None:
        try:
            payload = self._read_json()
            payload["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            payload["id"] = payload.get("id") or f"tr-{uuid.uuid4().hex[:10]}"
            with TRACES.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._send_json({"ok": True, "id": payload["id"]})
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _traces_get(self) -> None:
        limit = 50
        try:
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            limit = max(1, min(500, int(q.get("limit", ["50"])[0])))
        except Exception:
            pass
        rows = []
        if TRACES.is_file():
            for line in TRACES.read_text(encoding="utf-8").splitlines()[-limit:]:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        order = {"simple": 0, "medium": 1, "complex": 2}
        summary = {"n": len(rows), "over": 0, "under": 0, "match": 0, "by_suggested": {}, "by_actual": {}}
        for row in rows:
            s = row.get("suggested_tier") or "?"
            a = row.get("actual_tier") or "?"
            summary["by_suggested"][s] = summary["by_suggested"].get(s, 0) + 1
            summary["by_actual"][a] = summary["by_actual"].get(a, 0) + 1
            if s in order and a in order:
                if order[a] > order[s]:
                    summary["over"] += 1
                elif order[a] < order[s]:
                    summary["under"] += 1
                else:
                    summary["match"] += 1
        self._send_json({"traces": rows, "summary": summary})

    def _preference_post(self) -> None:
        try:
            payload = self._read_json()
            payload["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            payload["id"] = payload.get("id") or f"pref-{uuid.uuid4().hex[:10]}"
            with PREFERENCES.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            model = None
            try:
                rows = [json.loads(x) for x in PREFERENCES.read_text().splitlines() if x.strip()]
                model = judge_lib.train(rows)
            except Exception:
                pass
            self._send_json({"ok": True, "id": payload["id"], "model": model})
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _rationale_post(self) -> None:
        try:
            payload = self._read_json()
            payload["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with RATIONALES.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._send_json({"ok": True})
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _judge_predict(self) -> None:
        try:
            body = self._read_json()
            model = json.loads(JUDGE_MODEL.read_text()) if JUDGE_MODEL.is_file() else None
            rows = []
            if PREFERENCES.is_file():
                for line in PREFERENCES.read_text().splitlines():
                    try: rows.append(json.loads(line))
                    except ValueError: pass
            self._send_json(judge_lib.predict(body, model, rows))
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    def _route_targets_get(self) -> None:
        targets = DEFAULT_ROUTE_TARGETS
        if ROUTE_TARGETS.is_file():
            try:
                saved = json.loads(ROUTE_TARGETS.read_text())
                targets = {**DEFAULT_ROUTE_TARGETS, **(saved.get("targets") or {})}
            except ValueError:
                pass
        self._send_json({"targets": targets})

    def _route_targets_post(self) -> None:
        try:
            payload = self._read_json()
            targets = {**DEFAULT_ROUTE_TARGETS, **(payload.get("targets") or {})}
            ROUTE_TARGETS.write_text(json.dumps({"targets": targets}, indent=2) + "\n")
            self._send_json({"ok": True, "targets": targets})
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _preference_reset(self) -> None:
        try:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            archive = ROOT / "archive"
            archive.mkdir(exist_ok=True)
            moved = []
            for path in (PREFERENCES, RATIONALES):
                if path.is_file():
                    target = archive / f"{path.stem}-{stamp}{path.suffix}"
                    path.replace(target)
                    moved.append(str(target.relative_to(ROOT)))
            if JUDGE_MODEL.is_file():
                JUDGE_MODEL.unlink()
            self._send_json({"ok": True, "archived": moved})
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, 500)

    def _benchmark_get(self) -> None:
        try:
            runs = sorted(BENCH_DIR.glob("test-*.json"))
            if not runs:
                self._send_json({"error": "no benchmark runs"}, 404)
                return
            path = runs[-1]
            data = json.loads(path.read_text())
            self._send_json({
                "path": str(path.relative_to(ROOT.parents[1])),
                "file": path.name,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
                "tasks": len(data.get("tasks") or []),
                "samples": data.get("samples", 1),
                "models": data.get("models") or {},
            })
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    def _judge_status(self) -> None:
        rows = []
        if PREFERENCES.is_file():
            for line in PREFERENCES.read_text().splitlines():
                try: rows.append(json.loads(line))
                except ValueError: pass
        rationales = {}
        if RATIONALES.is_file():
            for line in RATIONALES.read_text().splitlines():
                try:
                    item = json.loads(line)
                    rationales[str(item.get("id") or "")] = item
                except ValueError:
                    pass
        model = json.loads(JUDGE_MODEL.read_text()) if JUDGE_MODEL.is_file() else None
        for row in rows:
            rat = rationales.get(str(row.get("id") or ""))
            if rat:
                row["rationale"] = rat.get("rationale") or ""
                row["tags"] = rat.get("tags") or []
        self._send_json({
            "model": model,
            "profile": judge_lib.profile(rows, rationales),
            "recent": rows[-20:],
        })

    def _preference_get(self) -> None:
        limit = 50
        try:
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            limit = max(1, min(500, int(q.get("limit", ["50"])[0])))
        except Exception:
            pass
        rows = []
        if PREFERENCES.is_file():
            for line in PREFERENCES.read_text(encoding="utf-8").splitlines()[-limit:]:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        summary = {"n": len(rows), "votes": {}, "by_model": {}}
        for row in rows:
            vote = row.get("vote") or "?"
            summary["votes"][vote] = summary["votes"].get(vote, 0) + 1
            for side in ("a", "b"):
                model = row.get(f"model_{side}") or "?"
                bucket = summary["by_model"].setdefault(model, {"wins": 0, "losses": 0, "ties": 0})
                if vote == side:
                    bucket["wins"] += 1
                elif vote in ("a", "b"):
                    bucket["losses"] += 1
                elif vote == "tie":
                    bucket["ties"] += 1
        self._send_json({"preferences": rows, "summary": summary})

    def _roles_get(self) -> None:
        import json
        roles = DEFAULT_ROLES
        placements = DEFAULT_PLACEMENTS
        if ROLES.is_file():
            try:
                saved = json.loads(ROLES.read_text())
                roles = {**DEFAULT_ROLES, **(saved.get("roles") or {})}
                placements = {**DEFAULT_PLACEMENTS, **(saved.get("placements") or {})}
            except ValueError:
                pass
        body = json.dumps({"roles": roles, "placements": placements}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _roles_post(self) -> None:
        import json
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
            roles = {**DEFAULT_ROLES, **(payload.get("roles") or {})}
            placements = {**DEFAULT_PLACEMENTS, **(payload.get("placements") or {})}
            ROLES.write_text(json.dumps({"roles": roles, "placements": placements}, indent=2) + "\n")
            body = json.dumps({"ok": True, "roles": roles, "placements": placements}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:  # noqa: BLE001
            body = json.dumps({"ok": False, "error": str(exc)}).encode()
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _suggestions_get(self) -> None:
        try:
            from urllib.parse import parse_qs, urlsplit
            from harness.hive_build import suggestions

            query = parse_qs(urlsplit(self.path).query)
            limit = int(query.get("limit", ["50"])[0])
            rows = suggestions.list(limit=limit)
            self._send_json({"suggestions": rows, "summary": suggestions.summary()})
        except (TypeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    def _suggestions_post(self) -> None:
        try:
            from harness.hive_build import suggestions

            record = suggestions.add(self._read_json())
            self._send_json({"ok": True, "suggestion": record}, 201)
        except (TypeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    def _suggestion_status_patch(self, suggestion_id: str) -> None:
        try:
            from harness.hive_build import suggestions

            payload = self._read_json()
            record = suggestions.set_status(suggestion_id, payload.get("status"))
            self._send_json({"ok": True, "suggestion": record})
        except KeyError as exc:
            self._send_json({"error": str(exc)}, 404)
        except (TypeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    def _placement_plan_post(self) -> None:
        try:
            from math import isfinite
            from harness.hive_build.placement import Device, plan_roles

            payload = self._read_json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            required = ("roles", "placements", "models", "devices")
            if any(key not in payload for key in required):
                raise ValueError("roles, placements, models, and devices are required")
            if not all(isinstance(payload[key], dict) for key in required[:3]):
                raise ValueError("roles, placements, and models must be objects")
            if not isinstance(payload["devices"], list):
                raise ValueError("devices must be an array")
            devices = []
            for item in payload["devices"]:
                if not isinstance(item, dict):
                    raise ValueError("each device must be an object")
                if not {"index", "card", "display", "total_gb", "free_gb"}.issubset(item):
                    raise ValueError("each device needs index, card, display, total_gb, and free_gb")
                if not isinstance(item["display"], bool):
                    raise ValueError("device display must be a boolean")
                total_gb = float(item["total_gb"])
                free_gb = float(item["free_gb"])
                if not isfinite(total_gb) or not isfinite(free_gb):
                    raise ValueError("device capacities must be finite numbers")
                devices.append(Device(
                    index=int(item["index"]),
                    card=str(item["card"]),
                    display=item["display"],
                    total_gb=total_gb,
                    free_gb=free_gb,
                ))
            self._send_json(plan_roles(
                payload["roles"], payload["placements"], payload["models"], devices,
            ))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, 400)

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import urlsplit
        if urlsplit(self.path).path == "/api/hive_build/suggestions":
            self._suggestions_get()
            return
        if self.path == "/api/roles":
            self._roles_get()
            return
        if self.path.startswith("/api/routing/traces"):
            self._traces_get()
            return
        if self.path.startswith("/api/preferences"):
            self._preference_get()
            return
        if self.path == "/api/judge/status":
            self._judge_status()
            return
        if self.path == "/api/benchmark":
            self._benchmark_get()
            return
        if self.path == "/api/route-targets":
            self._route_targets_get()
            return
        if self.path.startswith("/api/") and self.path.endswith("/stream"):
            self._proxy_stream(HIVE + self.path[4:])
            return
        if self.path.startswith("/api/"):
            self._proxy(HIVE + self.path[4:])
            return
        if self.path.startswith("/model/"):
            parts = self.path.split("/", 3)
            model = parts[2]
            rest = parts[3] if len(parts) > 3 else ""
            base = MODEL_ENDPOINTS.get(model)
            if not base:
                self.send_error(404, "unknown model")
                return
            self._proxy_model(f"{base}/{rest}")
            return
        if self.path.startswith("/view/"):
            self._proxy(HIVE + self.path)
            return
        if self.path.startswith("/probe/"):
            parts = self.path.split("/", 3)
            port = parts[2]
            rest = parts[3] if len(parts) > 3 else ""
            self._proxy(f"http://127.0.0.1:{port}/{rest}")
            return
        if self.path in ("/", ""):
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        from urllib.parse import urlsplit
        if urlsplit(self.path).path == "/api/placement/plan":
            self._placement_plan_post()
            return
        if urlsplit(self.path).path == "/api/hive_build/suggestions":
            self._suggestions_post()
            return
        if self.path == "/api/roles":
            self._roles_post()
            return
        if self.path == "/api/route":
            self._route()
            return
        if self.path == "/api/routing/traces":
            self._trace_post()
            return
        if self.path == "/api/preferences":
            self._preference_post()
            return
        if self.path == "/api/preferences/rationale":
            self._rationale_post()
            return
        if self.path == "/api/preferences/reset":
            self._preference_reset()
            return
        if self.path == "/api/route-targets":
            self._route_targets_post()
            return
        if self.path == "/api/decision":
            self._proxy("http://127.0.0.1:8161/v1/systemone")
            return
        if self.path == "/api/judge/predict":
            self._judge_predict()
            return
        if self.path.startswith("/api/") and self.path.endswith("/stream"):
            self._proxy_stream(HIVE + self.path[4:])
            return
        if self.path.startswith("/api/"):
            self._proxy(HIVE + self.path[4:])
            return
        if self.path.startswith("/model/"):
            parts = self.path.split("/", 3)
            model = parts[2]
            rest = parts[3] if len(parts) > 3 else ""
            base = MODEL_ENDPOINTS.get(model)
            if not base:
                self.send_error(404, "unknown model")
                return
            self._proxy_model(f"{base}/{rest}")
            return
        self.send_error(404)

    def do_PATCH(self) -> None:  # noqa: N802
        from urllib.parse import unquote, urlsplit

        prefix = "/api/hive_build/suggestions/"
        path = urlsplit(self.path).path
        if path.startswith(prefix) and path[len(prefix):] and "/" not in path[len(prefix):]:
            self._suggestion_status_patch(unquote(path[len(prefix):]))
            return
        self.send_error(404)

    def do_PUT(self) -> None:  # noqa: N802
        if self.path.startswith("/api/"):
            self._proxy(HIVE + self.path[4:])
            return
        self.send_error(404)

    def do_DELETE(self) -> None:  # noqa: N802
        if self.path.startswith("/api/"):
            self._proxy(HIVE + self.path[4:])
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"{self.address_string()} {fmt % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8170)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    print(f"serving {ROOT} on http://{args.bind}:{args.port}", flush=True)
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
