"""The README's numbers must be the real ones.

A README that says "97 operations" when the app serves 95 is a small lie
that ages badly, and nobody notices until someone builds against it.  These
tests read the claims out of the README and check them against the running
app: the endpoint counts, the settings index, and the command list.

They fail with a message that says what to change.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.app import create_app
from harness.reports import SETTINGS_INDEX

README = Path(__file__).resolve().parents[2] / "README.md"
METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    base = tmp_path_factory.mktemp("readme")
    return create_app(
        providers_file=base / "providers.local.json",
        engines_file=base / "engines.local.json",
        runs_root=base / "runs",
        log_dir=str(base / "logs"),
        state_dir=str(base / "state"),
    )


def _text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_endpoint_counts_are_true(app):
    client = TestClient(app)
    spec = client.get("/openapi.json").json()
    paths = spec.get("paths", {})
    operations = sum(
        1 for methods in paths.values() for m in methods if m.upper() in METHODS
    )
    v1_operations = sum(
        1 for path, methods in paths.items()
        if path.startswith("/v1/")
        for m in methods if m.upper() in METHODS
    )
    v1_paths = sum(1 for path in paths if path.startswith("/v1/"))
    text = _text()

    match = re.search(r"\*\*(\d+) REST operations\*\* across (\d+) paths", text)
    assert match, "the README no longer states the operation/path counts"
    assert int(match.group(1)) == operations, (
        f"the README says {match.group(1)} operations; the app serves "
        f"{operations} — update the README"
    )
    assert int(match.group(2)) == len(paths), (
        f"the README says {match.group(2)} paths; the app serves {len(paths)}"
    )

    match = re.search(r"\((\d+) operations under `/v1/`\)", text)
    assert match, "the README no longer states the /v1/ operation count"
    assert int(match.group(1)) == v1_operations, (
        f"the README says {match.group(1)} operations under /v1/; the app "
        f"serves {v1_operations}"
    )

    match = re.search(r"### API surface \((\d+) operations across (\d+) paths under `/v1/`\)", text)
    assert match, "the README's API surface heading lost its counts"
    assert int(match.group(1)) == v1_operations, (
        f"the API surface heading says {match.group(1)}; the app serves "
        f"{v1_operations} operations under /v1/"
    )
    assert int(match.group(2)) == v1_paths, (
        f"the API surface heading says {match.group(2)} paths; the app serves "
        f"{v1_paths} under /v1/"
    )


def test_readme_settings_counts_are_true():
    text = _text()
    match = re.search(r"(\d+) settings across (\d+)\s*\n?\s*categories", text)
    assert match, "the README no longer states the settings counts"
    categories = {s.get("cat") for s in SETTINGS_INDEX}
    assert int(match.group(1)) == len(SETTINGS_INDEX), (
        f"the README says {match.group(1)} settings; the index has "
        f"{len(SETTINGS_INDEX)}"
    )
    assert int(match.group(2)) == len(categories), (
        f"the README says {match.group(2)} categories; the index has "
        f"{len(categories)}: {sorted(categories)}"
    )


def test_readme_command_list_is_the_real_one(app):
    client = TestClient(app)
    names = {c["name"] for c in client.get("/v1/commands").json()["commands"]}
    text = _text()
    match = re.search(r"commands ship today: (.+?)\.", text, re.S)
    assert match, "the README no longer lists the console commands"
    listed = set(re.findall(r"`([a-z_]+)`", match.group(1)))
    assert listed == names, (
        "the README's command list is out of date: "
        f"missing {sorted(names - listed)}, extra {sorted(listed - names)}"
    )
