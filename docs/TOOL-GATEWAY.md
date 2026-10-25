# Hive tool gateway and permission protocol

**Status:** design locked / implementation pending, 2026-10-25.

This is the contract for every tool the agent can call: DSH-native tools, MCP
tools, Hive tools, and browser adapters. It is a **gateway**, not a UI prompt:
the agent must not be able to execute a side-effect tool before Hive approves
it.

The current implementation is the opposite: `harness/agent.py` reports
`full-access`, `tools_enabled` is one global switch, and Docker falls back to a
local subprocess. That is the gap this closes.

## Layers

```text
agent / DSH / MCP / browser
        │
        ▼
Hive tool gateway ── ToolSpec metadata
        │
        ├─ read-only? ──► allow (headless) / execute
        │
        └─ side-effect
              │
              ▼
        approval broker ── reason + action + risk + isolation
              │
        ┌─────┴─────┐
      allow        deny / timeout
        │             │
        ▼             ▼
     sandbox       blocked; end turn
        │
        ▼
      audit log
```

The gateway owns the tool registry, the policy, the approval queue and the
audit log. Tool implementations do not decide whether they are safe to run.

## Tool metadata

Every tool source must declare this, or the gateway treats the tool as
**unknown / side-effectful / private-network / workspace-write**.

```jsonc
{
  "name": "browser.open",
  "source": "hive|dsh|mcp|adapter",
  "version": "1.0.0",
  "description": "Open a URL in the isolated browser and return the page summary.",
  "read_only": false,
  "side_effect": true,
  "reason_required": true,
  "network": "public|private|none",
  "filesystem": "none|read|workspace|host",
  "risk": "low|medium|high",
  "sandbox": "required|optional|unsupported"
}
```

Rules:

- `read_only` is the only basis for headless auto-approval; unknown tools are
  not read-only.
- `reason_required` is true for every side-effect tool. A side-effect call
  without a reason is denied.
- A tool cannot upgrade its own metadata at call time.
- MCP annotations are advisory; Hive's registry wins on conflict.
- Tool name + source + version is the permission identity. A changed tool
  version invalidates prior grants.

## Approval request

The model/tool gateway produces one request per call:

```jsonc
{
  "id": "req-01J...",
  "session_id": "conv-...",
  "turn_id": "turn-...",
  "tool": "bash",
  "source": "hive",
  "action": "run: rg -n 'TODO' .",
  "args_redacted": {"cwd": "<workspace>", "argv": ["rg", "-n", "TODO", "."]},
  "reason": "Find the remaining TODO markers before editing.",
  "risk": "medium",
  "sandbox": "docker",
  "network": "none",
  "filesystem": "workspace-read",
  "timeout_s": 600
}
```

The prompt shows:

- **tool + source**
- **exact action** (redacted where needed)
- **why** (the agent-supplied reason)
- **risk / network / filesystem / sandbox**
- **scope** (once / session / always)

If the reason is missing, empty, or generic garbage, the gateway denies the
call before prompting. The UI says “reason missing; denied”.

## Decisions

One dialog, four choices:

| choice | effect |
|---|---|
| **Allow once** | this call only |
| **Allow for session** | this tool in this conversation/session |
| **Always allow** | asks *globally* or *this session only*, then persists the chosen scope |
| **Deny** | no execution; end this tool attempt |

“Always allow” is revocable in settings/CLI. It is keyed by tool + source +
version. The gateway never treats a grant as covering a different tool or
version.

A session grant resets when the conversation/session ends or when
`/tools reset` is issued. A global grant persists under `HIVE_HOME`.

The dialog is the only “disable” prompt. There is no second prompt after the
first successful call.

## Tool tiers

Tiers are ordered, but any tier can be selected explicitly. **Default: Guarded.**

| tier | approval | sandbox | local fallback | browser network |
|---|---|---|---|---|
| **Strict** | every side-effect call | Docker required | never | public + private prompt |
| **Guarded** (default) | tool per session; one dialog | Docker required | session approval | public + private prompt |
| **Trusted** | persistent per-tool grants | Docker preferred | session approval | public + per-origin private prompt |
| **Open** | no prompts; audit only | optional | allowed | public + private prompt; no silent private access |

Even **Open** keeps the non-negotiable guards below. “All precautions off” only
removes prompts and sandbox friction; it does not make the host disposable.

## Non-negotiable guards

These remain in every tier, including Open:

- **Audit.** Every request, decision, tool, reason, scope, timestamp, duration
  and result status is logged. Secrets are redacted; the log is append-only.
- **Workspace boundary.** Tools cannot read or write outside the configured
  workspace without an explicit export/approval; no host home traversal.
- **No privilege escalation.** No root, `--privileged`, Docker socket, host
  device passthrough, kernel modules, raw disks, or mount of `/`.
- **Secret protection.** Do not read SSH keys, browser profiles, cloud
  credentials, tokens or environment blocks without explicit user export.
- **Browser isolation.** The bundled browser uses a fresh profile by default;
  user cookies/passwords are not imported silently.
- **Private network gate.** Public web may be allowed; localhost, RFC1918,
  link-local and `.local` origins require an explicit grant even in Open.
- **No silent fallback.** If Docker is unavailable, the UI must show that the
  sandbox is local and ask for session approval. It must never just “work.”
- **No silent download/upload.** Browser/file download and upload paths need an
  explicit destination/source decision.

## Interactive behavior

Studio is a modal over the chat. TUI is an overlay. CLI is a line prompt. All
three resolve the same request through the same broker.

- The tool call is **blocked** while the dialog is open.
- The turn can wait up to **10 minutes** for an interactive decision.
- The countdown is visible.
- Deny, timeout or window close sends the same structured decision: `denied`.
- On deny/timeout, the agent receives a blocked result, makes **no further tool
  attempts**, and ends the turn with a clear explanation.
- The user can open the pending request later only if it is still within the
  10-minute window; otherwise it is expired.

There is no background “auto-answer” for a human prompt.

## Headless behavior

- Interactive UI absent: read-only tools with explicit metadata may run.
- Side-effect tools are **denied immediately**; the turn ends with the blocked
  action and the reason.
- The 10-minute wait does not apply without a surface that can show it.
- A batch/CLI run can supply a pre-approved manifest, but that is an explicit
  grant, not a hidden bypass.

## Sandbox

Docker is the default for bash, file writes and code execution:

- image: a pinned Hive sandbox image (the existing `python:3.12-slim` is only a
  fallback for local development);
- `--network none` for shell/code;
- workspace mounted at a disposable path, not the host home;
- memory/CPU/PID limits;
- no Docker socket, no host device, no privileged mode;
- output cap and timeout;
- the sandbox result records `backend: docker` or `backend: local`.

Docker setup belongs in `hive setup` and the Studio setup page. If Docker is
missing:

1. explain why;
2. offer install/setup;
3. allow **this session only** local fallback after a warning and approval;
4. never make local fallback the default.

The browser is outside Docker because it needs a display and a real network
stack. It uses a separate profile and the network gate.

## Browser tool

Two adapter classes behind one `browser` tool contract:

1. **Hive browser** — bundled Chromium/Playwright, headless or visible.
   - navigate, read accessibility/DOM, screenshot, click, type, scroll,
     wait, download/upload with explicit approval;
   - fresh profile by default; no passwords/cookies imported silently;
   - visible mode is a user-facing window/inspector, not a second chat UI.
2. **System browser** — Chrome/Firefox/Edge via CDP/WebDriver, if installed.
   - same actions and same approval metadata;
   - the adapter declares its isolation and profile path.

MCP browser servers and DSH tools can implement the same interface. The
gateway sees them as tools, not as a separate permission system.

Network policy:

- public origins: allowed in Guarded/Trusted/Open;
- private/local/link-local: always a separate approval;
- `file://` and `data:`: approval; never auto;
- credentials/autofill: only with explicit profile import.

## API shape

The gateway is the only tool execution path:

```text
GET  /v1/tools                      registry + metadata
POST /v1/tools/call                 create a call/approval request
GET  /v1/tools/pending              pending requests for the session
POST /v1/tools/decide               once | session | always | deny
GET  /v1/tools/audit                append-only audit (filtered)
POST /v1/tools/policy               tier, grants, revocation
```

A call returns one of:

```jsonc
{"status": "approved", "call_id": "...", "result": {...}}
{"status": "pending", "request_id": "...", "expires_at": "..."}
{"status": "denied", "reason": "missing reason|user denied|timeout|headless"}
```

The UI never needs to know how a tool is implemented, only its metadata and
request.

## Revocation and UX

- Studio Settings: tier selector, per-tool grants, revoke-all, audit viewer.
- CLI: `hive tools policy`, `hive tools grants`, `hive tools revoke`.
- TUI: pending overlay and a read-only policy summary; destructive grants stay
  in Studio/CLI.
- Every grant shows tool, source, version, scope, workspace, created/expiry.
- “Turn all precautions off” requires a typed confirmation and shows the
  non-negotiable list.

## Implementation order

1. **Gateway core** — ToolSpec, request/decision, session/global store, audit,
   headless policy, tests. No UI.
2. **Approval surfaces** — Studio modal/SSE, TUI overlay, CLI prompt; shared
   broker and countdown.
3. **Docker sandbox** — setup wizard, image, no-network workspace, local
   fallback approval.
4. **Browser adapters** — bundled Chromium visible/headless, system browser,
   private-origin gate.
5. **DSH/MCP wiring** — expose gateway tools, disable/proxy native DSH tools,
   import MCP metadata.
6. **Release gate additions** — permission denial, timeout, headless block,
   private network, sandbox fallback, audit redaction.

The current agent cannot be made safe by UI alone; step 5 is the enforcement
step. Do not ship a prompt-only gate as if it were a security boundary.

## Open design decisions

- Exact sandbox image/toolchain (git, rg, Python packages).
- Whether “Always allow” can be global for a tool or must be workspace-scoped;
  current plan follows the user’s choice: ask global or session.
- Browser download/upload destination defaults.
- Whether visible browser is a separate window or an embedded panel; separate
  window is the safer first implementation.
