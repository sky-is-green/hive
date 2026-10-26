# Hive tool gateway and permission protocol

**Status:** policy/grant/audit core, DSH PreToolUse hook, terminal/web approval
surfaces, pinned Docker sandbox policy, private-origin classification for DSH
web tools, a stateless secure fetch adapter, and an optional validating-proxy
Chromium adapter implemented; MCP metadata/subagent inheritance remains,
2026-10-26. The hook passes end-to-end against both the source-built and
published DSH runtimes.

This is the contract for every tool the agent can call: DSH-native tools, MCP
tools, Hive tools, and browser adapters. It is a **gateway**, not a UI prompt:
the agent must not be able to execute a side-effect tool before Hive approves
it.

The policy/grant/audit core is implemented in
[`harness/security.py`](../harness/security.py) and exposed through
`hive tools`. `harness/agent_hooks.py` installs the DSH
`hooks-claude-code` bridge with a `PreToolUse` command that calls
`harness/tool_hook.py`; the hook posts the concrete call to the Hive gateway
and blocks on the human decision. If the hook cannot be installed, the agent
refuses to start unless `HIVE_DSH_HOOK=off` is set for an explicitly
unguarded development run. When an external runtime is selected with
`HIVE_DSH_BIN`, Hive boots it once before the first turn and refuses the agent
if DSH reports that the hook failed to import or activate. The hook path has
been exercised end-to-end against both source-built DSH 0.1.2-alpha.1 and
published DSH 0.2.1-rc.2: approve ran the tool and deny blocked it with the
hook's exit-2 reason.

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
  link-local and `.local` origins require an explicit grant even in Open. Web
  tool grants are scoped to the exact origin, so approving one private URL does
  not approve another; a private tool with no visible origin falls back to the
  `private` resource key.
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

Docker is the default for bash, file writes and code execution. The policy
lives in `settings/sandbox.json` and is managed with `hive sandbox`:

- image: a pinned Hive sandbox image; `hive sandbox pin` resolves and stores the
  sha256 digest. A mutable tag without a digest is refused, not silently
  pulled;
- `--network none` for shell/code;
- `--memory`, `--cpus`, `--pids-limit`, `--cap-drop ALL`, and
  `--security-opt no-new-privileges`;
- no Docker socket, no host device, no privileged mode;
- output cap and timeout;
- the sandbox result records `backend: docker` or `backend: local`.

If Docker is missing or its daemon is down, `/v1/sandbox/exec` returns `409`
with `requires_local_fallback_approval: true`. The caller must retry with
`allow_local_fallback: true` **and** a `session_id`; that explicit approval is
audited as `sandbox.local_fallback`. A policy of `deny` forbids the host path
entirely; `allow` is a deliberate development-box choice. A missing or unpinned
Docker image is a sandbox failure, never a reason to use the host.

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

## Implemented API (core)

```text
GET  /v1/tools/policy     # tier, master switches, saved grants
PUT  /v1/tools/policy     # tier and master switches
POST /v1/tools/call       # concrete call -> approved/pending/denied
GET  /v1/tools/pending    # pending requests
POST /v1/tools/decide     # once/session/always/deny
GET  /v1/tools/decision   # settled decision for a waiting hook
POST /v1/tools/revoke     # identity, or all=true
GET  /v1/tools/audit      # recent append-only decisions
```

The call/approval endpoints below remain the target for the execution adapters;
they are not exposed until a real tool can be gated end to end.

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

## Implementation status

Implemented:

- `ToolSpec` metadata with unknown-tool safe defaults;
- the `strict` / `guarded` / `trusted` / `open` policy store;
- reason enforcement for side-effect calls;
- in-memory session grants and durable global grants;
- headless denial for side-effect calls;
- private-network approval gating;
- pending decisions, deny, timeout, and grant scope;
- append-only tool audit with argument hashes and secret redaction;
- `hive tools status|policy|grants|pending|decide|revoke|audit`;
- the DSH `PreToolUse` hook bridge, the Studio pending cards, the TUI `a`
  overlay, and the CLI `pending` / `decide` commands;
- live hook tests against source-built DSH 0.1.2-alpha.1 and published DSH
  0.2.1-rc.2: approve ran the tool, deny blocked it;
- a pinned-image Docker sandbox policy with explicit local-fallback approval
  (`hive sandbox status|policy|pin`);
- private-origin classification and exact-origin grants for DSH web tools:
  literal private/loopback/metadata hosts, private suffixes, and DNS answers
  that resolve private are `network=private`; a read-only private fetch still
  needs approval, including in the Open tier;
- the stateless secure fetch adapter (`hive web`): manual per-hop redirect
  checks, safe headers only, byte/content-type limits, explicit download and
  upload gates, and exact-origin grants; see [Secure fetch](WEB-FETCH.md);
- an optional Playwright Chromium adapter (`hive browser`) with a fresh
  context, no persistent profile, service workers blocked, and a validating
  local proxy that checks every HTTP request and HTTPS CONNECT, including
  redirects and subresources; see [Browser adapter](BROWSER.md);
- unit tests for the above.

Still required before this is a release-complete agent security boundary:

- a real Docker-daemon execution run (the policy and mocked tests are
  implemented, but this host has no daemon and no passwordless root);
- MCP metadata registration and subagent inheritance;
- browser page-level download/upload automation (the adapter deliberately does
  not write downloads or set file inputs; use secure fetch for those gates).

Bundled-runtime verification is implemented as a start-up stderr check, in
addition to the external-runtime preflight.

## Implementation order

1. **Gateway core** — ToolSpec, request/decision, session/global store, audit,
   headless policy, tests. No UI. **Done.**
2. **Approval surfaces** — Studio pending cards, TUI `a` overlay, CLI
   `pending` / `decide`; shared broker and countdown. **Done.**
3. **Docker sandbox** — pinned image, no-network execution, resource limits,
   explicit local-fallback approval. **Done in policy/API; real-daemon run
   pending.**
4. **Browser adapters** — optional Playwright Chromium, fresh context,
   validating proxy, redirect/subresource origin checks. **Done.** Page-level
   download/upload automation remains deliberately disabled; secure fetch owns
   those explicit gates.
5. **DSH/MCP wiring** — the DSH PreToolUse hook is mounted and verified;
   MCP metadata registration and subagent inheritance remain. **Partial.**
6. **Release gate additions** — permission denial, timeout, headless block,
   private network, sandbox fallback, audit redaction. **Partial.**

The current agent cannot be made safe by UI alone; step 5 is the enforcement
step. Do not ship a prompt-only gate as if it were a security boundary.

## Open design decisions

- Exact sandbox image/toolchain (git, rg, Python packages).
- Whether “Always allow” can be global for a tool or must be workspace-scoped;
  current plan follows the user’s choice: ask global or session.
- Browser download/upload destination defaults.
- Whether visible browser is a separate window or an embedded panel; separate
  window is the safer first implementation.

## Hazards the first draft did not cover

These are not optional polish; they are part of the gateway's threat model.

### Prompt injection and untrusted tool output

A web page, file, MCP response, or tool result can contain instructions aimed at
the model. The policy layer must treat all tool output as **untrusted data**:

- never inherit instructions from tool output, a page, or an MCP server;
- the approval reason and the user-facing action summary are claims, not
  authority; the gateway displays the concrete action alongside them;
- never let a tool result change the tier, grants, sandbox or network policy;
- do not expose a tool's raw output to a later privileged tool without a
  boundary marker and, for sensitive actions, a fresh approval;
- screenshot/OCR output is untrusted too.

### SSRF, redirects and DNS rebinding

For browser and network tools:

- resolve every hostname and check **all** resolved addresses against private,
  loopback, link-local, multicast and cloud-metadata ranges;
- re-check after every redirect and subresource request;
- block `file://`, `data:`, `javascript:` and similar local schemes by default;
- pin the approved origin; a redirect to a different origin needs a new
  decision;
- do not trust the reason or the displayed URL without the resolved origin.

### Egress and exfiltration

Public network access is allowed, but moving user data out is a separate act:

- downloads, uploads, form posts and clipboard writes show source, destination,
  size and content class;
- secrets found in a page or a file must be redacted before display/audit;
- private-network approval is independent of public browsing;
- a browser profile with user cookies is a high-risk grant, never a silent
  default.

### Subagents and nested sessions

Child sessions/workers must use the same gateway and inherit **no broader**
grants than their parent. A delegated worker cannot:

- approve its own request;
- reuse a parent's global grant for a different tool identity;
- bypass the private-network gate;
- inherit a “session” grant from a different session.

The current `agent_subagents.py` nested loop must be covered explicitly; a
gateway that only wraps the face agent is not a boundary.

### Approval flooding and turn control

- At most one pending approval per turn is actionable at a time; additional
  calls queue visibly.
- A deny/timeout cancels that tool attempt and ends the turn; no retry with a
  disguised tool name.
- 10 minutes is a hard interactive limit; the UI shows a countdown.
- A cancelled/closed UI counts as deny.
- The audit records the decision even if the model then ends the turn.

### MCP and third-party tool trust

- MCP server identity + version is part of every tool identity;
- changed schemas or binaries require re-approval;
- server processes themselves run under the sandbox policy where possible;
- a tool's declared `read_only` metadata is not trusted until the server is
  registered/verified;
- network endpoints are not automatically trusted because they are local.

### Docker image and supply chain

- Pin the sandbox image by digest, not only a mutable tag.
- Record image provenance and checksum.
- Do not mount the Docker socket, host home, SSH directory, browser profile or
  GPU devices unless an explicit, separate grant exists.
- A missing image is a sandbox failure; do not silently use the host.

### Approval DoS and long waits

Ten minutes is long enough to block a turn. The UI must allow cancel/stop at
any time, and the agent must have a deterministic blocked result. Headless runs
fail fast; they do not hold a process for ten minutes.

### Audit limitations

An append-only file is tamper-evident only if protected. At minimum:

- mode `0600`, rotate by month, never let the tool API rewrite it;
- record hashes of redacted arguments and results;
- optionally chain each record's hash to the previous record;
- if the audit sink fails, deny the call.
