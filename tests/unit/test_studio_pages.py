"""Studio page rendering must produce syntactically valid JavaScript.

The Studio's UI is server-rendered: the inline ``<script>`` blocks live inside
Python f-strings, where a literal brace must be written ``{{``/``}}``.  A
mismatched brace, or an ``await`` in a non-async function, is a *parse* error
that silently kills the whole script block — the console then renders but
nothing works (no tab switching, no chat, no saves).

This test renders every Studio page and runs ``node --check`` on each script
block.  It is skipped when node is unavailable.
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
    render_server_page,
    render_training_pane,
)

NODE = shutil.which("node")

#: page name -> (renderer, expected script block count).  A count of 0 means the
#: page is static HTML (no behaviour to break).
PAGES = {
    "server": (render_server_page, 2),
    "cascade": (lambda: render_cascade_page(["tasks-hard.json"]), 1),
    "runs": (lambda: render_runs_page([]), 0),
    "report": (lambda: render_report_page({"kind": "benchmark"}, "run-x"), 0),
    "training": (lambda: render_training_pane(), 1),
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
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(block)
            path = fh.name
        try:
            proc = subprocess.run(
                [NODE, "--check", path], capture_output=True, text=True, timeout=60
            )
        finally:
            os.unlink(path)
        assert proc.returncode == 0, (
            f"{page} script block {i} is not valid JavaScript:\n{proc.stderr[:2000]}"
        )


def test_assistant_context_points_at_the_studio_api():
    """The app-assistant context must name the real endpoints."""
    from harness.app import _ASSISTANT_CONTEXT

    assert "/v1/server/status" in _ASSISTANT_CONTEXT
    assert "/v1/models/local" in _ASSISTANT_CONTEXT
    assert "/v1/engines" in _ASSISTANT_CONTEXT
    assert "8765" in _ASSISTANT_CONTEXT
    assert "never guess" in _ASSISTANT_CONTEXT


def test_chat_row_above_the_bar_is_the_command_row():
    """The picked commands ARE the row above the chat bar.

    The old action row (Stop / Save / Compact / Export / Import / +New) is
    gone: the row above the composer holds the commands the user picked plus
    the picker toggle, and the session-file actions moved into the chat
    head's session menu.
    """
    html = render_server_page()
    assert "chat-actions" not in html
    assert 'id="pinned-commands"' in html
    assert 'id="cmd-picker-toggle"' in html
    # the session actions live in the head menu now
    assert 'class="chat-menu"' in html
    for label in ("Save session", "Export session", "Import session"):
        assert label in html
    # and the functions those buttons call actually exist
    assert "function exportSession(" in html
    assert "function importSession(" in html


def test_pinned_command_buttons_carry_a_removable_x():
    """Every pinned command is a container with a run button and a red x."""
    html = render_server_page()
    assert 'class="pin-cmd"' in html
    assert 'class="pin-run"' in html
    assert 'class="pin-x"' in html
    # the x removes the command, it does not run it
    assert "onclick=\"togglePinnedCommand(" in html


def test_an_attached_file_rides_into_the_next_message():
    """``/import`` hands the file's text back; the chat carries it along."""
    html = render_server_page()
    assert "pendingAttachment" in html
    assert "r.attachment" in html
    assert "body.query = pendingAttachment.text" in html


def test_the_console_names_itself_and_the_user_does_not():
    """Command output is a labelled console bubble; the user needs no label."""
    html = render_server_page()
    assert "label: 'Command Console'" in html
    # the user bubble carries no label — you know you typed it
    assert "user:    {cls: 'user',    label: ''}" in html
    # and the command output uses the console voice, not the status pill
    assert "bubble('console', (r.kind === 'error'" in html


def test_a_message_is_a_label_gutter_plus_a_box():
    """The speaker sits in its own gutter; the box holds only the message."""
    html = render_server_page()
    assert "box.className = 'bubble'" in html
    assert "row.className = 'msg ' + speaker.cls" in html
    # the console is cyan now, not red (red reads as failure)
    assert "#67E8F9" in html or True  # the colour lives in the stylesheet
    css = Path(__file__).resolve().parents[2] / "harness" / "studio.css"
    text = css.read_text(encoding="utf-8")
    assert ".msg.console .bubble" in text
    assert "#67E8F9" in text
    assert "FF9B9B" not in text          # the old soft red is gone
    assert ".msg .bubble" in text        # the box is its own element
    assert "align-self: flex-end" in text  # the user's row hugs the right
    # the speaker names the box from above, so the box keeps the full width
    assert "flex-direction: column;" in text
