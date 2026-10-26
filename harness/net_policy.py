"""Origin and address classification shared by the hook and web fetcher.

The security decision is deliberately conservative: a host that cannot be
resolved is treated as private, a URL with credentials is rejected, and every
redirect hop is classified again before a request is made.  DNS rebinding
between classification and connection remains a residual risk; the fetcher
does not reuse a browser session or cookies.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
from dataclasses import dataclass

PRIVATE_HOSTS = frozenset({
    "localhost", "localhost.localdomain", "metadata", "instance-data",
    "metadata.google.internal", "host.docker.internal",
    "gateway.docker.internal", "host.containers.internal",
    "kubernetes.default.svc",
})
PRIVATE_SUFFIXES = (
    ".localhost", ".local", ".localdomain", ".internal", ".svc",
    ".cluster.local", ".home.arpa", ".lan", ".corp",
)
DEFAULT_PORTS = {"http": 80, "https": 443}


class UrlError(ValueError):
    """The URL cannot be classified for a network request."""


@dataclass(frozen=True)
class UrlInfo:
    """One validated URL and its resolved network identity."""

    url: str
    scheme: str
    host: str
    port: int
    origin: str
    addresses: tuple[str, ...]
    private: bool
    reason: str

    @property
    def host_header(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return (host if self.port == DEFAULT_PORTS[self.scheme]
                else f"{host}:{self.port}")

    @property
    def request_path(self) -> str:
        parts = urllib.parse.urlsplit(self.url)
        path = parts.path or "/"
        return path + (f"?{parts.query}" if parts.query else "")


def _host_is_private(host: str) -> tuple[bool, str]:
    host = host.strip().strip(".").lower()
    if not host:
        return True, "empty host"
    if host in PRIVATE_HOSTS or any(
            host.endswith(suffix) for suffix in PRIVATE_SUFFIXES):
        return True, "private hostname"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            resolved = socket.getaddrinfo(host, None)
        except OSError:
            return True, "unresolvable host"
        if not resolved:
            return True, "unresolvable host"
        for info in resolved:
            if _ip_is_private(info[4][0]):
                return True, "resolves to a private address"
        return False, "public"
    private = _ip_is_private(str(address))
    return private, "private address" if private else "public"


def _ip_is_private(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return not address.is_global


def _addresses(host: str) -> tuple[str, ...]:
    try:
        ipaddress.ip_address(host)
        return (host,)
    except ValueError:
        try:
            return tuple(sorted({info[4][0] for info in
                                 socket.getaddrinfo(host, None)}))
        except OSError:
            return ()


def origin_of(url: str) -> str:
    """Return a canonical ``scheme://host[:port]`` origin."""
    parts = urllib.parse.urlsplit(url if "://" in url else f"http://{url}")
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS:
        raise UrlError("only http and https URLs are supported")
    if parts.username or parts.password:
        raise UrlError("URL credentials are not allowed")
    host = (parts.hostname or "").strip().strip(".").lower()
    if not host:
        raise UrlError("URL has no host")
    try:
        port = parts.port or DEFAULT_PORTS[scheme]
    except ValueError as exc:
        raise UrlError("URL has an invalid port") from exc
    if port == 0:
        raise UrlError("URL has an invalid port")
    shown = f"[{host}]" if ":" in host else host
    if port != DEFAULT_PORTS[scheme]:
        shown += f":{port}"
    return f"{scheme}://{shown}"


def classify_url(url: str) -> UrlInfo:
    """Validate a URL and classify its first-hop destination."""
    text = str(url or "").strip()
    if not text:
        raise UrlError("URL is required")
    parts = urllib.parse.urlsplit(text)
    if parts.username or parts.password:
        raise UrlError("URL credentials are not allowed")
    if parts.scheme.lower() not in DEFAULT_PORTS:
        raise UrlError("only http and https URLs are supported")
    host = (parts.hostname or "").strip().strip(".").lower()
    if not host:
        raise UrlError("URL has no host")
    try:
        port = parts.port or DEFAULT_PORTS[parts.scheme.lower()]
    except ValueError as exc:
        raise UrlError("URL has an invalid port") from exc
    if port == 0:
        raise UrlError("URL has an invalid port")
    private, reason = _host_is_private(host)
    return UrlInfo(
        url=text, scheme=parts.scheme.lower(), host=host, port=port,
        origin=origin_of(text), addresses=_addresses(host),
        private=private, reason=reason)


def redirect_url(current: str, location: str) -> str:
    """Resolve a Location header and reject non-http(s) destinations."""
    target = urllib.parse.urljoin(current, str(location or "").strip())
    classify_url(target)  # validate scheme/host/credentials now
    return target
