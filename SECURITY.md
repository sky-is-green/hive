# Security

Hive is a local-first harness. The default bind is `127.0.0.1`, and the
console, Studio API and model servers are intended for the same machine.

## Exposure

- Do not bind the Studio to `0.0.0.0` or a LAN address without a reverse proxy
  and authentication in front of it.
- `HARNESS_TOKEN` protects `/v1/*` mutations, not every read. It is not a
  replacement for network isolation.
- `providers.local.json` can contain API keys. It is gitignored and must stay
  local; the API masks keys on read.
- The sandbox is optional. If Docker is unavailable, the host fallback requires
  an explicit approval; `hive sandbox policy --local-fallback deny` disables it.
- Agent tools can execute code. Keep the default `guarded` tier unless you have
  reviewed the tool grants.

## Reporting

Report a vulnerability with GitHub's private security advisory for
[`sky-is-green/hive`](https://github.com/sky-is-green/hive/security/advisories/new).
Do not include secrets, private prompts, or full local paths in a public issue.
