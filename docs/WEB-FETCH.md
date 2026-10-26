# Secure fetch

**Status:** implemented, 2026-10-26 · `harness/web_fetch.py` ·
`harness/net_policy.py`

Secure fetch is Hive's stateless HTTP(S) primitive. It is deliberately **not a
browser**: it does not execute JavaScript, keep cookies, reuse a profile, or
follow redirects inside a library. Every hop is classified and authorized
before a socket is opened, so a public URL cannot redirect the fetcher into a
private service without a separate per-origin decision.

## What it protects

- only `http` and `https`;
- URL credentials (`user:pass@host`) are rejected;
- loopback, RFC1918, link-local, multicast, metadata hosts, `.local`/`.internal`
  suffixes, and hosts that resolve to a private address are **private**;
- an unresolvable host is treated as private, not public;
- manual redirects, each hop checked again;
- redirects to another private origin need a grant for that exact origin;
- no cookies, `Authorization`, `Host`, proxy, forwarding, or transfer headers
  are forwarded;
- the response byte limit applies before content is returned or written;
- only text-like content types are returned inline;
- downloads need `allow_download` + `download_name` and the policy switch;
- uploads need `allow_upload` + a file inside the configured upload directory
  and the policy switch;
- path traversal in a download name is rejected;
- no overwrite: an existing download gets a numeric suffix.

The download and upload directories live under
`<state>/browser/downloads` and `<state>/browser/uploads`, created with mode
`0700`.

## Private grants

Grants are keyed by the exact canonical origin, for example
`http://127.0.0.1:8765` or `https://example.com:8443`.

- **global** grants persist in `settings/browser-grants.json`;
- **session** grants live in the server process and disappear with it;
- `access=private` covers a private origin, `access=public` covers a public
  origin when public fetching is disabled.

An explicit `allow_private=true` from an API/CLI caller needs a `session_id`
and becomes a session grant for that one original origin. It does **not** cover
a later redirect to a different private origin; that hop fails and names the
missing origin.

## CLI

```bash
hive web status
hive web policy --allow-downloads
hive web grant http://127.0.0.1:8765 --access private --scope global
hive web grants
hive web fetch https://example.com/docs
hive web fetch http://127.0.0.1:8765/server --allow-private --session s1
hive web fetch https://example.com/file --download file.bin
hive web fetch https://example.com/upload --method POST \
  --upload notes.txt --allow-upload
hive web revoke http://127.0.0.1:8765
```

The API equivalents are `/v1/web/fetch`, `/v1/web/status`,
`/v1/web/grants`, `/v1/web/grant`, `/v1/web/revoke`, and `/v1/web/policy`.

## Residual limits

- DNS rebinding between classification and the actual socket connection is a
  residual risk in secure fetch: it does not pin the resolved address. The
  optional browser adapter's proxy does pin the validated address.
- It does not render JavaScript, run a browser engine, store cookies, or
  support page-level downloads/uploads. Use the optional
  [browser adapter](BROWSER.md) when JavaScript or page rendering is required.
- It is not a substitute for the DSH `PreToolUse` approval hook. The hook gates
  the agent's tools; secure fetch is a separate network primitive and API.
- A TLS error is reported as a transport failure; the fetcher does not offer a
  per-request "ignore certificate" switch.
