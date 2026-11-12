"""Report pages must produce syntactically valid JavaScript.

The remaining server-rendered pages are the run/report views and the Cascade
launch console. They still embed script blocks inside Python f-strings, where
a literal brace must be written ``{{``/``}}``. A parse error silently kills the
whole block, so this test renders each page and runs ``node --check``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from harness.reports import (
    render_cascade_page,
    render_report_page,
    render_runs_page,
)

NODE = shutil.which("node")

#: page name -> (renderer, expected script block count).
PAGES = {
    "cascade": (lambda: render_cascade_page(["tasks-hard.json"]), 1),
    "runs": (lambda: render_runs_page([]), 0),
    "report": (lambda: render_report_page({"kind": "benchmark"}, "run-x"), 0),
}


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("page", sorted(PAGES))
def test_page_scripts_are_valid_javascript(page):
    render, expected = PAGES[page]
    html = render()
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(blocks) == expected, (
        f"{page}: expected {expected} script block(s), found {len(blocks)} "
        "(a lost block means the page silently stops working)"
    )
    for i, block in enumerate(blocks):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(block)
            path = fh.name
        try:
            proc = subprocess.run(
                [NODE, "--check", path], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60,
            )
        finally:
            os.unlink(path)
        assert proc.returncode == 0, (
            f"{page} script block {i} is not valid JavaScript:\n{proc.stderr[:2000]}"
        )


def test_nav_marks_the_active_cascade_page():
    html = render_cascade_page(["tasks-hard.json"])
    assert ">Senses</button>" in html
    assert "aria-current='page'" in html
    assert "title='Upload a sensory input" in html
    assert "title='Generate, detect, repair, and escalate answers'" in html


def test_cascade_page_has_a_senses_upload_panel():
    html = render_cascade_page(["tasks-hard.json"])
    assert 'id="senses-panel"' in html
    assert 'id="senses-file"' in html
    assert 'id="senses-button"' in html
    assert "uploadSense()" in html
    assert "/v1/cascade/senses/status" in html
    assert "/v1/cascade/senses" in html


def test_assistant_context_points_at_the_studio_api():
    """The app-assistant context must name the real endpoints."""
    from harness.app import _ASSISTANT_CONTEXT

    assert "/v1/server/status" in _ASSISTANT_CONTEXT
    assert "/v1/models/local" in _ASSISTANT_CONTEXT
    assert "/v1/engines" in _ASSISTANT_CONTEXT
    assert "8765" in _ASSISTANT_CONTEXT
    assert "never guess" in _ASSISTANT_CONTEXT
