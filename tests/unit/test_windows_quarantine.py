"""Linux-first boundary: the VHDX/WSL2 setup routes stay Windows-only.

The code remains for the later Windows review, but the Linux surface must not
expose it. The CI matrix is Ubuntu-only unless the manual ``windows`` input is
set.
"""

from __future__ import annotations

import sys

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


@pytest.mark.skipif(sys.platform == "win32", reason="Windows is the target there")
@pytest.mark.parametrize("path", ["/v1/setup/status", "/v1/setup/drives"])
def test_windows_setup_routes_are_quarantined_on_linux(client, path):
    r = client.get(path)
    assert r.status_code == 410, r.text
    assert "Windows-only" in r.json()["detail"]
