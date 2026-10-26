"""A validating local HTTP/CONNECT proxy for the optional browser adapter.

Chromium route interception does not see server-side redirects, so the browser
is pointed at this proxy instead.  Every HTTP request and every HTTPS CONNECT
is classified and authorized before a socket is opened, and the connection is
made to the resolved address so DNS is pinned for that hop.
"""

from __future__ import annotations

import http.client
import socket
import socketserver
import threading
from typing import Any
from urllib.parse import urlsplit

from harness.net_policy import UrlError, classify_url
from harness.state_store import StatePaths
from harness.web_fetch import (
    BrowserGrantStore,
    WebFetchError,
    WebFetchPolicy,
    authorize_origin,
)

MAX_HEADER_BYTES = 64 * 1024
HOP_HEADERS = frozenset({
    "connection", "proxy-connection", "proxy-authorization", "keep-alive",
    "transfer-encoding", "upgrade",
})


def _split_host_port(target: str) -> tuple[str, int]:
    text = str(target or "").strip()
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        if not rest.startswith(":"):
            raise ValueError("CONNECT target has no port")
        port = int(rest[1:])
    else:
        host, sep, port_text = text.rpartition(":")
        if not sep:
            raise ValueError("CONNECT target has no port")
        port = int(port_text)
    if not host or port < 1 or port > 65535:
        raise ValueError("CONNECT target is invalid")
    return host, port


class BrowserProxy:
    """Process-local validating proxy; one instance per browser session."""

    def __init__(self, state: StatePaths | None = None, *,
                 session_id: str | None = None,
                 allow_private: bool = False,
                 timeout_s: float = 30.0) -> None:
        self.state = state or StatePaths.default()
        self.session_id = session_id
        self.allow_private = bool(allow_private)
        self.timeout_s = max(1.0, float(timeout_s))
        self.policy = WebFetchPolicy(self.state)
        self.grants = BrowserGrantStore(self.state)
        self.blocked: list[dict[str, str]] = []
        self.requests: list[str] = []
        self._lock = threading.Lock()
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("proxy is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "BrowserProxy":
        if self._server is not None:
            return self
        proxy = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                proxy.handle(self.request)

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _record(self, origin: str, reason: str) -> None:
        with self._lock:
            self.blocked.append({"origin": origin, "reason": reason})

    def blocked_origins(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(row) for row in self.blocked]

    def authorize_original(self, url: str) -> Any:
        """Authorize and, when asked, grant the original navigation origin."""
        info = classify_url(url)
        try:
            authorize_origin(
                info, session_id=self.session_id,
                allow_private=self.allow_private, grants=self.grants,
                policy=self.policy, hop=0)
        except WebFetchError as exc:
            self._record(info.origin, exc.code)
            raise
        self.allow_private = False  # only the original navigation gets this
        return info

    def authorize_url(self, url: str) -> Any:
        info = classify_url(url)
        try:
            authorize_origin(
                info, session_id=self.session_id, allow_private=False,
                grants=self.grants, policy=self.policy, hop=1)
        except WebFetchError as exc:
            self._record(info.origin, exc.code)
            raise
        return info

    def handle(self, client: socket.socket) -> None:
        try:
            client.settimeout(self.timeout_s)
            head, rest = self._read_head(client)
            lines = head.decode("iso-8859-1").split("\r\n")
            if not lines or not lines[0]:
                return
            parts = lines[0].split(" ")
            if len(parts) != 3:
                self._send_error(client, 400, "Bad Request")
                return
            method, target, version = parts
            headers: list[tuple[str, str]] = []
            for line in lines[1:]:
                if ":" not in line:
                    continue
                name, value = line.split(":", 1)
                headers.append((name.strip(), value.strip()))
            if method.upper() == "CONNECT":
                self._connect_tunnel(client, target)
                return
            self._http_forward(client, method, target, version, headers, rest)
        except (OSError, ValueError, WebFetchError) as exc:
            try:
                self._send_error(client, 502, "Bad Gateway")
            except OSError:
                pass
        finally:
            try:
                client.close()
            except OSError:
                pass

    def _read_head(self, client: socket.socket) -> tuple[bytes, bytes]:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = client.recv(8192)
            if not chunk:
                raise ValueError("client closed before headers")
            data += chunk
            if len(data) > MAX_HEADER_BYTES:
                raise ValueError("request headers are too large")
        head, rest = data.split(b"\r\n\r\n", 1)
        return head, rest

    def _connect_tunnel(self, client: socket.socket, target: str) -> None:
        try:
            host, port = _split_host_port(target)
        except ValueError:
            self._send_error(client, 400, "Bad CONNECT")
            return
        url = f"https://{host}:{port}"
        try:
            info = self.authorize_url(url)
        except WebFetchError as exc:
            self._send_error(client, 403, f"Blocked: {exc.code}")
            return
        address = info.addresses[0] if info.addresses else None
        if not address:
            self._send_error(client, 502, "Unresolvable host")
            return
        try:
            upstream = socket.create_connection(
                (address, port), timeout=self.timeout_s)
        except OSError:
            self._send_error(client, 502, "Upstream connect failed")
            return
        with self._lock:
            self.requests.append(url)
        try:
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            if rest:
                upstream.sendall(rest)
            self._relay(client, upstream)
        finally:
            try:
                upstream.close()
            except OSError:
                pass

    def _http_forward(self, client: socket.socket, method: str, target: str,
                      version: str, headers: list[tuple[str, str]],
                      rest: bytes) -> None:
        if not target.startswith("http://"):
            self._send_error(client, 400, "HTTP proxy expects an http:// URL")
            return
        try:
            info = self.authorize_url(target)
        except WebFetchError as exc:
            self._send_error(client, 403, f"Blocked: {exc.code}")
            return
        address = info.addresses[0] if info.addresses else None
        if not address:
            self._send_error(client, 502, "Unresolvable host")
            return
        parts = urlsplit(target)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        header_map = {name.lower(): value for name, value in headers}
        length = 0
        try:
            length = int(header_map.get("content-length", "0") or 0)
        except ValueError:
            self._send_error(client, 400, "Bad Content-Length")
            return
        body = rest[:length]
        while len(body) < length:
            chunk = client.recv(min(65536, length - len(body)))
            if not chunk:
                break
            body += chunk
        conn = http.client.HTTPConnection(
            address, info.port, timeout=self.timeout_s)
        try:
            conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            conn.putheader("Host", info.host_header)
            for name, value in headers:
                if name.lower() in HOP_HEADERS:
                    continue
                if name.lower() in ("host", "content-length"):
                    continue
                conn.putheader(name, value)
            if body:
                conn.putheader("Content-Length", str(len(body)))
            conn.endheaders()
            if body:
                conn.send(body)
            response = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()
            self._send_error(client, 502, "Upstream request failed")
            return
        with self._lock:
            self.requests.append(target)
        try:
            reason = response.reason or "OK"
            client.sendall(
                f"HTTP/1.1 {response.status} {reason}\r\n".encode("iso-8859-1"))
            for name, value in response.getheaders():
                if name.lower() in HOP_HEADERS:
                    continue
                client.sendall(f"{name}: {value}\r\n".encode("iso-8859-1"))
            client.sendall(b"Connection: close\r\n\r\n")
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                client.sendall(chunk)
        except OSError:
            pass
        finally:
            conn.close()

    def _relay(self, client: socket.socket, upstream: socket.socket) -> None:
        def pump(src: socket.socket, dst: socket.socket) -> None:
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                try:
                    dst.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        thread = threading.Thread(target=pump, args=(client, upstream),
                                  daemon=True)
        thread.start()
        pump(upstream, client)
        thread.join(timeout=2)

    @staticmethod
    def _send_error(client: socket.socket, status: int, text: str) -> None:
        body = text.encode("utf-8")
        client.sendall(
            f"HTTP/1.1 {status} {text}\r\n"
            f"Content-Type: text/plain\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n".encode("iso-8859-1") + body)


def check_proxy(url: str, *, state: StatePaths | None = None,
                session_id: str | None = None,
                allow_private: bool = False) -> Any:
    """Authorize one URL without starting a proxy (used by the adapter)."""
    proxy = BrowserProxy(state, session_id=session_id,
                         allow_private=allow_private)
    return proxy.authorize_original(url)
