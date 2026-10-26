# Browser adapter

**Status:** optional, implemented 2026-10-26 · `harness/browser_adapter.py` ·
`harness/browser_gateway.py`

The browser adapter is a real Chromium page rendered through Playwright. It is
optional because it needs a browser binary; the base `hivebench` install stays
light.

```bash
pip install 'hivebench[browser]'
python -m playwright install chromium
hive browser status
```

## Safety model

- **Fresh context per call.** No persistent cookies, local storage, profile,
  sessions, extensions, or imported passwords.
- **Validating local proxy.** Chromium is launched with
  `--proxy-server=http://127.0.0.1:<port>` and
  `--proxy-bypass-list=<-loopback>`, so loopback is not exempted. Every HTTP
  request and every HTTPS `CONNECT` is classified and authorized before a
  socket opens.
- **Redirects are rechecked.** A public page that redirects to a private origin
  creates a new CONNECT; the proxy blocks it before the target server sees it.
- **Subresources are checked.** Images, scripts, iframes, and fetches to a
  different private origin are blocked too.
- **DNS is pinned.** The proxy connects to the resolved address that it
  validated; it does not hand the hostname to a second resolver for that hop.
- **Private grants.** The same exact-origin grants as secure fetch are used:
  `hive web grant <origin> --access private --scope global`. `allow_private`
  with a session id approves only the original navigation origin; a later
  redirect to another private origin fails and names it.
- **Service workers blocked.** `service_workers="block"` avoids a worker
  bypassing the page-level network checks.
- **Downloads disabled.** The adapter creates contexts with
  `accept_downloads=False`; page downloads are not written. Use secure fetch
  `--download` for an explicit, gated file download.
- **No upload automation.** The adapter does not open file choosers or set
  input files. Use secure fetch `--upload` for an explicit, gated upload.
- **Screenshots** are optional and written under `browser/screenshots/` with
  mode `0600`; the name must be a plain filename.

## CLI

```bash
hive browser status
hive browser browse https://example.com/docs
hive browser browse http://127.0.0.1:8765/ --allow-private --session s1
hive browser browse https://example.com --screenshot example.png
```

The API is `POST /v1/browser/browse` and `GET /v1/browser/status`.

## Residual limits

- HTTPS traffic inside a CONNECT tunnel is opaque to the proxy beyond the
  host/port. That is enough for origin-level SSRF, but path-level policy would
  need TLS interception, which this adapter deliberately does not do.
- WebRTC, QUIC and browser-internal network paths are outside the HTTP proxy.
  The adapter disables QUIC and background networking; it is not a hardened
  anonymity browser.
- A browser page is untrusted content. The text returned to a caller is data,
  not instructions, and the agent gateway still decides whether the page may be
  opened.
- This is not a replacement for the DSH `PreToolUse` hook. The hook gates agent
  tools; the browser adapter is a separate page-rendering surface.
