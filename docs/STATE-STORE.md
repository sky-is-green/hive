# Hive state store

**Status:** core store implemented for controller state; policy migration pending,
2026-10-25.

How Hive persists choices, grants, recommendations and evidence across restarts
without mixing them into one unmaintainable settings blob.

The first slice is live in [`harness/state_store.py`](../harness/state_store.py):
`StatePaths`, per-file `flock`, `tmp → fsync → os.replace`, schema checks,
corrupt-file quarantine, atomic audit JSONL and redaction. It owns
`settings/controller.json`, `profiles/*.json`,
`profiles/controller-traces-*.jsonl` and `audit/settings-*.jsonl`.
The controller trace stores request features, the proposal, optional
explicitly-enabled encoder vectors, and lifecycle outcomes, never the raw
request text. `settings/security.json` and `settings/tool-grants.json` are now
used by the gateway core. Studio settings now read the legacy
`studio_settings.json` and write the migrated `settings/studio.json` on the
next save; the legacy file is left in place. Sandbox and browser grant stores
are still design-only.

## Principles

1. **One scope per file.** Security policy is not conversation state; a tool
   grant is not an audit record.
2. **Deny wins.** A malformed, missing or unreadable policy file falls back to
   the safest tier, never to “allow everything”.
3. **Atomic writes.** Every JSON write is `tmp → fsync → os.replace`, with a
   lock around read-modify-write.
4. **Schema first.** Every durable file has a `schema_version`; unknown keys are
   preserved, corrupt files are quarantined.
5. **Session grants are not durable.** They live in memory and disappear when
   the session ends. Only explicit “Always allow” grants persist.
6. **Audit is evidence, not configuration.** Append-only, redacted, never
   rewritten by the UI.
7. **No secrets in settings.** Browser cookies/passwords live in a protected
   profile, not in `tool-grants.json` or the audit.
8. **The AI reads the same files.** No pickle or opaque database for policy.
   JSON/JSONL, with a schema and stable keys.

## Layout

All mutable state lives under the existing state root:

```text
$HIVE_STATE_DIR/                    # default: $HIVE_HOME/harness_state
  settings/
    studio.json                     # existing controls (migrated)
    security.json                   # tier, master switches, consent
    tool-grants.json                # global grants only
    tool-registry.json              # verified external/MCP metadata
    sandbox.json                    # Docker/local fallback policy
    browser-grants.json             # private origins, downloads
    controller.json                 # selected profile, recommendation scope
  profiles/
    hardware-<id>.json              # machine capability snapshot
    recommendations-<id>.json       # generated proposals + evidence
    controller-traces-YYYYMM.jsonl  # privacy-preserving proposal/outcome log
  controllers/
    <name>/
      manifest.json
      encoder.*
      head.*
  browser/
    downloads/                      # secure-fetch downloads; mode 0700
    uploads/                        # secure-fetch upload sources; mode 0700
    screenshots/                    # optional browser screenshots; mode 0700
    profiles/<id>/                  # cookies, cache; mode 0700
  audit/
    tool-calls-YYYYMM.jsonl         # append-only; rotated
    settings-YYYYMM.jsonl           # policy/grants changes
  serve/                            # existing serve records
  conv-*.json                       # existing conversation state
  dsh_sessions/                     # existing durable agent logs
```

`HIVE_HOME` remains the root for models, stacks and backends. `HIVE_STATE_DIR`
can point at a separate disk, or at a test sandbox. A source checkout continues
to use its gitignored `harness_state/`.

### Migration

`settings/studio.json` is the new home for the existing
`harness_state/studio_settings.json`. On first start:

1. if the new file exists, use it;
2. else read the legacy file;
3. write the new file with `schema_version: 1`;
4. leave the legacy file in place until the next successful start, then remove
   it (or keep a backup under `settings/backups/`).

The same path handles the older `settings.json` naming if found.

## State classes

| state | file | lifetime | authority |
|---|---|---|---|
| Safety tier | `settings/security.json` | until changed | user |
| Master switches | `settings/security.json` | until changed | user |
| Global tool grants | `settings/tool-grants.json` | until revoked | user |
| Session tool grants | in memory | session only | user |
| Pending approvals | in memory | ≤ 10 min | user |
| Sandbox choice | `settings/sandbox.json` | until changed | user |
| Browser site grants | `settings/browser-grants.json` | until revoked/expiry | user |
| Hardware profile | `profiles/hardware-*.json` | until rescan | machine |
| Recommendations | `profiles/recommendations-*.json` | generated; auditable | system |
| Controller traces | `profiles/controller-traces-YYYYMM.jsonl` | append-only | system |
| Controller config | `settings/controller.json` | until changed | user |
| Controller artifacts | `controllers/<name>/` | until removed | user/system |
| Tool audit | `audit/tool-calls-YYYYMM.jsonl` | append-only | system |
| Conversation | `conv-*.json` | conversation | user/system |
| DSH session | `dsh_sessions/` | session log | system |
| Serve record | `serve/*.json` | process lifetime | system |

Session IDs are explicit. A session grant is keyed by
`(session_id, tool_identity, resource)` and never written to disk. A global
grant is keyed by `(tool_identity, resource)` and stored. `resource` is empty
for ordinary tools and an exact origin such as
`https://example.com:8443` for private web tools, so approving one private URL
does not approve another.

`tool_identity = source:name@version`. A version bump invalidates the grant.

## Security policy file

```jsonc
{
  "schema_version": 1,
  "tier": "guarded",
  "master": {
    "approval_prompts": true,
    "sandbox": true,
    "private_network": true
  },
  "headless": {
    "read_only": "auto",
    "side_effect": "deny"
  },
  "updated_at": "2026-10-25T21:00:00Z",
  "updated_by": "local-user"
}
```

Tiers: `strict`, `guarded` (default), `trusted`, `open`.

Some guards are not represented as disableable booleans: workspace boundary,
no privilege escalation, audit, secret protection and browser isolation remain
hard policy. `open` only relaxes approvals and sandbox friction.

## Global tool grants

```jsonc
{
  "schema_version": 1,
  "grants": [
    {
      "tool_identity": "hive:bash@1.0.0",
      "scope": "global",
      "resource": "",
      "display_name": "bash",
      "granted_at": "2026-10-25T21:00:00Z",
      "last_used_at": "2026-10-25T21:04:00Z",
      "expires_at": null,
      "reason": "user approved command execution for this box"
    }
  ]
}
```

Rules:

- only `Always allow` reaches this file;
- `Always allow` asks **global** or **session**; a session choice stays in
  memory;
- the file is read on startup and grants are matched by exact identity and
  resource (origin for private web tools);
- revocation removes the row; future calls prompt again;
- a corrupt file means **no grants**, not “allow all”;
- the audit log records the grant/revoke event, not just the tool call.

## Tool registry

```jsonc
{
  "schema_version": 1,
  "tools": [
    {
      "tool_identity": "mcp:mcp__github__search@1.0.0",
      "tool": {
        "name": "mcp__github__search",
        "source": "mcp",
        "version": "1.0.0",
        "read_only": true,
        "network": "public"
      },
      "verified": true,
      "server": "github",
      "registered_at": "2026-10-26T18:00:00Z"
    }
  ]
}
```

Only `verified: true` entries override the hook's conservative metadata.
Unregistered `mcp__<server>__<tool>` calls are side-effecting, private-network,
high-risk, and require approval. The registry is local policy; it does not
grant execution by itself.

## Sandbox policy

```jsonc
{
  "schema_version": 1,
  "mode": "docker",
  "docker_image": "hive-sandbox:1",
  "image_digest": "<64 hex characters>",
  "runtime": "",
  "network": "none",
  "memory": "512m",
  "cpus": "1",
  "pids_limit": 128,
  "timeout_s": 60,
  "local_fallback": "approval",
  "updated_at": "2026-10-25T21:00:00Z"
}
```

`runtime` is empty for the daemon default, or names an installed Docker
runtime such as `runsc` (gVisor) or `kata-runtime` (Kata Containers). The
runtime is passed as `docker run --runtime <name>`; the host must have it
registered.

`local_fallback: approval` means `/v1/sandbox/exec` refuses a host subprocess
until the caller retries with `allow_local_fallback: true` and a `session_id`;
the approval is written to the tool audit as `sandbox.local_fallback`. Docker
availability is **probed live**, never stored as truth. An unpinned image is a
sandbox failure, not a host fallback. `hive sandbox pin` resolves and stores the
digest.

## Secure fetch grants

```jsonc
{
  "schema_version": 1,
  "grants": [
    {
      "origin": "http://127.0.0.1:8765",
      "access": "private",
      "scope": "global",
      "session_id": null,
      "granted_at": "2026-10-26T17:00:00Z",
      "expires_at": null,
      "reason": "local Studio fetch"
    }
  ]
}
```

Origins are canonical `scheme://host[:port]` values. A global grant persists
here; a session grant is process-local and never written. Access is `public`
or `private`. A private grant covers only that exact origin; a redirect to a
different private origin needs its own grant.

Secure-fetch downloads and uploads live under `browser/downloads/` and
`browser/uploads/` with mode `0700`. Cookies, browser profiles and passwords
are **not** here; a future JavaScript browser adapter must keep those isolated
and explicitly imported.

## Hardware profile and recommendations

Hardware profiles are snapshots:

```jsonc
{
  "schema_version": 1,
  "profile_id": "local-2026-10-25-a1b2",
  "scanned_at": "...",
  "consent": {"hardware": true, "models": false, "traces": false, "export": false},
  "cpu": {"model": "...", "logical": 16, "physical": 8},
  "ram": {"total_gb": 30.9, "available_gb": 18.2},
  "gpus": [{"index": 0, "vram_gb": 21.46, "display": true}],
  "backends": ["rocm"],
  "storage": {"free_gb": 23.7},
  "confidence": {"cpu": "high", "gpus": "high"}
}
```

Recommendations are a separate immutable record, linked by `profile_id`:

```jsonc
{
  "schema_version": 1,
  "profile_id": "local-2026-10-25-a1b2",
  "created_at": "...",
  "proposals": [
    {
      "scope": "runtime",
      "reason": "long-context coding on a 20 GiB display card",
      "changes": {"ctx_size": 16384, "resident_gib": 4},
      "confidence": 0.78,
      "evidence": ["weights fit with 1.8 GiB reserve", "16k KV fits"],
      "requires_confirmation": true
    }
  ]
}
```

Applying a proposal is a separate action that writes `settings/controller.json`
or the normal studio controls and records the event in `settings-*.jsonl`.
Deleting or rescanning a profile does not silently mutate current settings.

## Precedence

Resolution order for runtime settings:

```text
built-in defaults
  < machine recommendations (only after accepted)
  < global settings/security
  < conversation/session overrides
  < run-specific arguments
```

For a **tool call**:

```text
hard invariants (always)
  > explicit deny/timeout
  > strict tier
  > session grant
  > global grant
  > tier policy (guarded/trusted/open)
  > unknown-tool default: ask (or deny headless)
```

An explicit deny is a decision for that call, not a permanent ban unless the
user chooses a persistent deny. A tool version change invalidates a grant.

## Atomicity and concurrency

- One lock per state file (POSIX `flock`, Windows `msvcrt.locking`).
- Write to a sibling temp file, `flush`, `fsync`, `os.replace`.
- File mode `0600` for policy/grants; `0700` for browser profiles.
- Read-modify-write under the same lock; last-writer-wins is not acceptable for
  grants.
- If a write fails, the call that needed it is denied; never proceed with an
  unrecorded grant.
- If an audit append fails, fail closed and surface the error.
- On startup, a corrupt file is moved to `settings/corrupt-<timestamp>.json` and
  safe defaults are used. Do not silently overwrite evidence.

## Restart and session semantics

After a process or machine restart:

- global grants reload;
- safety tier/sandbox/browser policy reload;
- session grants are gone;
- pending approvals are gone and are recorded as expired;
- browser profiles persist unless the user deletes them;
- audit is intact;
- conversation state and DSH logs remain where they already live.

This is deliberate. A user who chose “Allow for session” should be asked again
after restart.

## CLI and API

Proposed CLI:

```bash
hive state status
hive state export --out hive-state.json     # no secrets
hive state reset --yes
hive tools grants
hive tools revoke bash
hive tools revoke --all
hive profile scan --consent hardware,models
hive profile show
```

Proposed API:

```text
GET  /v1/state/status
GET  /v1/state/export
GET  /v1/tools/grants
POST /v1/tools/revoke
GET  /v1/profiles/hardware
POST /v1/profiles/scan
```

The state export includes settings, grants (without secrets), hardware profile
and controller manifest references. It excludes cookies, conversation text and
raw audit payloads by default.

## Tests required

- atomic write survives a simulated crash between temp write and replace;
- corrupt JSON quarantines and falls back to safe defaults;
- a corrupt grants file does **not** grant anything;
- global grant persists across process restart;
- session grant does not;
- tool version bump invalidates a grant;
- open tier still blocks private-network without a grant;
- audit redaction removes tokens/URLs marked sensitive;
- migration from legacy `studio_settings.json`;
- concurrent grant writes do not lose a revocation;
- export excludes browser cookies and conversation text.

## Implementation order

1. `JsonStore`/`StatePaths` with locks, atomic writes, schema and quarantine.
2. Migrate `studio_settings.json`; add `security.json` and `tool-grants.json`.
3. Wire the tool gateway to this store and the audit log.
4. `sandbox.json` and Docker/local-fallback approvals. **Done.**
5. Add hardware profile/recommendation storage.
6. Secure-fetch grants. **Done.** Browser profile isolation remains.
7. Add CLI/API state commands and export/import.
