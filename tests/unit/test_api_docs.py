"""``docs/API.md`` must describe the API the Studio actually serves.

The endpoint map is hand-maintained, which means it drifts the moment an
endpoint is added or removed.  This test walks the real app's routes and
compares them with the map, so a new endpoint without a doc line (or a
doc line for an endpoint that no longer exists) fails the suite.

The doc rows look like::

    | `GET` | `/v1/commands` | List console commands (name, description, input hint). |
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.app import create_app

DOC = Path(__file__).resolve().parents[2] / "docs" / "API.md"
ROW = re.compile(r"^\| `([A-Z]+)` \| `([^`]+)` \|", re.M)
#: the methods the map lists
METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH")


@pytest.fixture(scope="module")
def routes(tmp_path_factory):
    base = tmp_path_factory.mktemp("apidocs")
    app = create_app(
        providers_file=base / "providers.local.json",
        engines_file=base / "engines.local.json",
        runs_root=base / "runs",
        log_dir=str(base / "logs"),
        state_dir=str(base / "state"),
    )
    client = TestClient(app)
    spec = client.get("/openapi.json").json()
    out: dict[str, set[str]] = {}
    for path, methods in spec.get("paths", {}).items():
        for method in methods:
            if method.upper() in METHODS:
                out.setdefault(path, set()).add(method.upper())
    return out


def _doc_map() -> dict[str, set[str]]:
    text = DOC.read_text(encoding="utf-8")
    out: dict[str, set[str]] = {}
    for method, path in ROW.findall(text):
        out.setdefault(path, set()).add(method)
    return out


def test_api_doc_covers_every_endpoint(routes):
    doc = _doc_map()
    missing = sorted(p for p in routes if p not in doc)
    assert not missing, (
        "docs/API.md is missing endpoints the Studio serves: "
        + ", ".join(missing)
        + " — add a row for each (or the map is a lie)"
    )


def test_api_doc_has_no_stale_endpoints(routes):
    doc = _doc_map()
    stale = sorted(p for p in doc if p not in routes)
    assert not stale, (
        "docs/API.md documents endpoints that no longer exist: "
        + ", ".join(stale)
        + " — delete those rows"
    )


def test_api_doc_methods_match(routes):
    doc = _doc_map()
    wrong = {
        p: (sorted(routes[p]), sorted(doc[p]))
        for p in sorted(set(routes) & set(doc))
        if routes[p] != doc[p]
    }
    assert not wrong, (
        "docs/API.md has the wrong method for some endpoints "
        "(live vs documented): " + repr(wrong)
    )


def test_api_doc_rows_have_a_purpose():
    """Every row says what the endpoint is for — a bare path is not a map."""
    text = DOC.read_text(encoding="utf-8")
    bare = [
        line for line in text.splitlines()
        if re.match(r"^\| `[A-Z]+` \| `[^`]+` \|\s*\|", line)
    ]
    assert not bare, "these rows have no description:\n" + "\n".join(bare)
