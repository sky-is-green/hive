"""The /cascade page must render, not 500.

The route once passed a ``defaults=`` argument the renderer did not accept;
the page itself already loads the Global quality baseline from
/v1/studio/settings. This guards the route contract.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cortex.e2e import FakeUltraSmall
from harness.app import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        ultra_factory=FakeUltraSmall,
        providers_file=tmp_path / "providers.local.json",
        engines_file=tmp_path / "engines.local.json",
        runs_root=tmp_path / "runs",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
    )
    with TestClient(app) as test_client:
        yield test_client


def test_cascade_page_renders(client):
    r = client.get("/cascade")

    assert r.status_code == 200, r.text
    assert "Hive Cascade" in r.text
    assert 'id="cascade-form"' in r.text
    assert "quality_defaults" in r.text


def test_stacks_page_does_not_link_to_the_removed_console(client):
    r = client.get("/stacks")

    assert r.status_code == 200, r.text
    assert 'href="/runs"' in r.text
    assert 'href="/server"' not in r.text

