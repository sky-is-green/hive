"""Optional Playwright/Chromium browser adapter over the validating proxy.

The browser is deliberately thin: a fresh context per call, no persistent
cookies/profile, service workers blocked, and all HTTP(S) traffic forced
through :class:`harness.browser_gateway.BrowserProxy`.  The proxy checks every
HTTP request and HTTPS CONNECT, including redirects, before opening a socket.

This module is optional.  Install the extra and browser binary with:

    pip install 'hivebench[browser]'
    python -m playwright install chromium
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from harness.browser_gateway import BrowserProxy
from harness.state_store import StatePaths
from harness.web_fetch import WebFetchError

try:  # optional dependency
    from playwright.sync_api import (  # type: ignore
        Error as PlaywrightError,
        TimeoutError as PlaywrightTimeout,
        sync_playwright,
    )
    _PLAYWRIGHT_ERROR = ""
except Exception as exc:  # noqa: BLE001 - report the optional dependency
    sync_playwright = None  # type: ignore
    PlaywrightError = Exception  # type: ignore
    PlaywrightTimeout = Exception  # type: ignore
    _PLAYWRIGHT_ERROR = str(exc)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
_WAIT_UNTIL = ("domcontentloaded", "load", "networkidle")
DEFAULT_TIMEOUT_S = 30
DEFAULT_MAX_TEXT = 20_000


class BrowserUnavailable(WebFetchError):
    """The optional browser dependency or binary is unavailable."""


def _require_playwright() -> None:
    if sync_playwright is None:
        raise BrowserUnavailable(
            "Playwright is not installed; install the browser extra and "
            "`python -m playwright install chromium`",
            code="browser_unavailable", status=503,
            detail={"reason": _PLAYWRIGHT_ERROR})


def status() -> dict[str, Any]:
    """Report whether the optional browser can resolve a Chromium binary."""
    if sync_playwright is None:
        return {"available": False, "reason": _PLAYWRIGHT_ERROR,
                "install": "pip install 'hivebench[browser]'; "
                           "python -m playwright install chromium"}
    try:
        with sync_playwright() as p:
            executable = p.chromium.executable_path
            return {"available": True, "browser": "chromium",
                    "executable": executable}
    except Exception as exc:  # noqa: BLE001 - report, don't hide
        return {"available": False, "reason": str(exc),
                "install": "python -m playwright install chromium"}


def _safe_screenshot_name(name: str | None) -> str:
    raw = str(name or "").strip()
    if not raw:
        return f"browse-{uuid.uuid4().hex[:12]}.png"
    if raw != Path(raw).name or not _SAFE_NAME.match(raw):
        raise WebFetchError("screenshot_name must be a plain filename",
                            code="bad_screenshot_name")
    return raw


def browse(url: str, *, session_id: str | None = None,
           allow_private: bool = False, screenshot_name: str | None = None,
           wait_until: str = "domcontentloaded",
           timeout_s: float = DEFAULT_TIMEOUT_S,
           max_text: int = DEFAULT_MAX_TEXT,
           state: StatePaths | None = None) -> dict[str, Any]:
    """Open one URL in a fresh Chromium context behind the validating proxy."""
    _require_playwright()
    if wait_until not in _WAIT_UNTIL:
        raise WebFetchError(f"wait_until must be one of {_WAIT_UNTIL}",
                            code="bad_wait_until")
    timeout = max(1.0, min(120.0, float(timeout_s)))
    max_text = max(1, min(200_000, int(max_text)))
    state = state or StatePaths.default()
    name = _safe_screenshot_name(screenshot_name) if screenshot_name else ""
    proxy = BrowserProxy(state, session_id=session_id,
                         allow_private=allow_private, timeout_s=timeout)
    proxy.start()
    browser = None
    context = None
    try:
        info = proxy.authorize_original(url)
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    f"--proxy-server={proxy.url}",
                    # Never bypass the proxy for loopback.  This is what keeps
                    # a DNS/redirect to localhost from escaping the checks.
                    "--proxy-bypass-list=<-loopback>",
                    "--disable-quic",
                    "--disable-background-networking",
                    "--disable-component-update",
                    "--no-first-run",
                ],
            )
            context = browser.new_context(
                service_workers="block",
                accept_downloads=False,
                java_script_enabled=True,
                ignore_https_errors=False,
            )
            page = context.new_page()
            page.set_default_timeout(int(timeout * 1000))
            try:
                page.goto(info.url, wait_until=wait_until,
                          timeout=int(timeout * 1000))
            except (PlaywrightTimeout, PlaywrightError) as exc:
                blocked = proxy.blocked_origins()
                if blocked:
                    raise WebFetchError(
                        "browser request was blocked by the origin gate",
                        code="private_origin", status=403,
                        detail={"blocked": blocked,
                                "url": info.url}) from exc
                raise WebFetchError(
                    f"browser navigation failed: {exc}",
                    code="browser_navigation", status=502,
                    detail={"url": info.url}) from exc
            blocked = proxy.blocked_origins()
            if blocked:
                raise WebFetchError(
                    "browser page attempted a blocked origin",
                    code="private_origin", status=403,
                    detail={"blocked": blocked,
                            "url": page.url})
            title = ""
            text = ""
            try:
                title = page.title()
            except PlaywrightError:
                pass
            try:
                text = page.inner_text("body")
            except PlaywrightError:
                text = ""
            screenshot = None
            if name:
                directory = state.browser_dir / "screenshots"
                directory.mkdir(parents=True, exist_ok=True)
                try:
                    directory.chmod(0o700)
                except OSError:
                    pass
                target = directory / name
                page.screenshot(path=str(target), full_page=False)
                try:
                    target.chmod(0o600)
                except OSError:
                    pass
                screenshot = {"name": target.name, "path": str(target)}
            return {
                "ok": True,
                "url": page.url,
                "title": title[:500],
                "text": text[:max_text],
                "screenshot": screenshot,
                "blocked": proxy.blocked_origins(),
                "requests": list(proxy.requests)[-100:],
            }
    finally:
        try:
            if context is not None:
                context.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if browser is not None:
                browser.close()
        except Exception:  # noqa: BLE001
            pass
        proxy.close()
