"""The new shell's assets must stay internally consistent.

The shell is plain HTML/JS served from ``harness/webui``. There is no bundler
and no build step, so a renamed panel, a missing script tag or a page without a
section fails silently in the browser. This test catches those cases and runs
``node --check`` on every shell script (skipped when node is unavailable).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "harness" / "webui"
HTML = ROOT / "index.html"
NODE = shutil.which("node")


def _html() -> str:
    return HTML.read_text(encoding="utf-8")


def test_every_nav_page_has_a_section():
    pages = set(re.findall(r'data-page="([a-z0-9-]+)"', _html()))
    sections = set(re.findall(r'id="page-([a-z0-9-]+)"', _html()))
    assert pages, "no data-page buttons found"
    assert pages <= sections, f"nav pages without sections: {sorted(pages - sections)}"


def test_every_script_asset_exists():
    srcs = re.findall(r'<script src="/([^"?]+)', _html())
    assert srcs, "no script assets found"
    missing = [src for src in srcs if not (ROOT / src).is_file()]
    assert not missing, f"missing shell assets: {missing}"


def test_the_suggestions_inbox_is_removed():
    html = _html()
    assert 'data-page="suggestions"' not in html
    assert "suggestions_panel.js" not in html
    assert not (ROOT / "suggestions_panel.js").exists()
    assert not (ROOT / "suggestions_panel.css").exists()
    assert "/api/hive_build/suggestions" not in (
        ROOT / "dev_server.py").read_text(encoding="utf-8")


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("path", sorted(ROOT.glob("*.js")), ids=lambda p: p.name)
def test_shell_javascript_is_syntactically_valid(path: Path):
    proc = subprocess.run(
        [NODE, "--check", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{path.name}: {proc.stderr or proc.stdout}"
