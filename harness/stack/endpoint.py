"""Endpoint helpers shared by the stack manager and the Strata launcher.

An **endpoint-backed** stack tier names an already-running OpenAI-compatible
server (Strata's ``serve/server.py``, or any llama-server).  This module owns
the pure URL math and the health probe; :mod:`harness.stack.manager` consumes
them for verify-and-register and :mod:`harness.stack.strata` for
adopt-or-start.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import requests

#: API-base suffix Strata serves under the server root (``…/v1``).
_API_SUFFIX = "/v1"


def endpoint_base(endpoint: str) -> str:
    """The endpoint's server root: ``http://host:8080/v1`` → ``http://host:8080``.

    Both Strata and llama-server serve ``/health`` at the root while the
    OpenAI surface lives under ``/v1``; a declared endpoint may name either.
    """
    base = str(endpoint).strip().rstrip("/")
    if base.endswith(_API_SUFFIX):
        base = base[: -len(_API_SUFFIX)].rstrip("/")
    return base


def endpoint_port(endpoint: str) -> int:
    """TCP port of an endpoint URL (80/443 when the URL omits it; 0 when unusable)."""
    try:
        parts = urlsplit(str(endpoint).strip())
        return int(parts.port or (443 if parts.scheme == "https" else 80))
    except (TypeError, ValueError):
        return 0


def probe_endpoint(endpoint: str, timeout: float = 5.0) -> dict[str, Any]:
    """Health-check an endpoint-backed tier and return the facts to register.

    ``GET <root>/health`` first (Strata reports ``status``/``loaded``/
    ``max_context``; llama-server reports ``status``), falling back to
    ``GET <root>/v1/models`` for servers without a health route.  Raises
    ``RuntimeError`` with every failure when the endpoint is unreachable,
    unhealthy, or reports itself unloaded — an apply that registers a dead
    tier is worse than a failed apply.
    """
    root = endpoint_base(endpoint)
    errors: list[str] = []
    for url, kind in ((f"{root}/health", "health"), (f"{root}/v1/models", "models")):
        try:
            resp = requests.get(url, timeout=timeout)
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue
        if resp.status_code != 200:
            errors.append(f"{url}: HTTP {resp.status_code}")
            continue
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        if not isinstance(data, Mapping):
            data = {}
        if kind == "health":
            status = data.get("status")
            if status is not None and str(status).strip().lower() != "ok":
                errors.append(f"{url}: status={status!r}")
                continue
            if data.get("loaded") is False:
                raise RuntimeError(
                    f"endpoint {endpoint} is up but its model is not loaded "
                    f"(/health says loaded=false); load it (POST {root}/load) "
                    f"and apply again")
        model = str(data.get("model") or "")
        ids = [
            str(entry.get("id") or "")
            for entry in (data.get("data") or [])
            if isinstance(entry, Mapping)
        ]
        ids = [value for value in ids if value]
        return {
            "endpoint": endpoint,
            "base_url": endpoint.rstrip("/"),
            "health_url": url,
            "model": model or (ids[0] if ids else ""),
            "loaded": bool(data.get("loaded", True)),
            "max_context": data.get("max_context"),
            "service": data.get("service") or "",
        }
    raise RuntimeError(
        f"endpoint {endpoint} is not healthy: " + ("; ".join(errors) or "no response")
    )
