# Activity, Stop, and Cease

**Status:** `hive_activity_control` branch · 2026-10-26

Hive has three different controls, and they should not be confused:

| Control | Scope | What it does |
| --- | --- | --- |
| **Observe** | live + durable | Records the user prompt, assistant text, exact tool arguments, exact tool results, model/tier spans, and errors. |
| **Stop** | current turn | Ends the turn and terminates the DSH runtime. Session and grants survive. |
| **Cease** | whole session | Blocks every further tool call, force-kills the runtime process tree, kills that session's sandbox containers, revokes its grants, and requires an explicit resume. |

The point of **Cease** is the case this repository cannot technically contain:
the user sees something happening that the OS/user boundary does not cover, and
needs the agent to lose access *now*.

## What is recorded

The activity ledger is an append-only JSONL file under the AI state directory:

```text
$HIVE_STATE_DIR/audit/activity-YYYYMM.jsonl
```

Each record carries `ts`, `seq`, `type`, `conversation_id`, `session_id`, and
`source`. The event types are:

- `user` — the exact text Hive sent to DSH, including any handoff/context
  prefix, plus the original user message.
- `assistant` — committed assistant text per agent step.
- `tool` — `phase: call` with `args`, and `phase: result` with `result`.
  Face-level and delegated tools both carry the concrete payload.
- `model` — tier dispatch start/end: tier, model, task, duration, output.
- `lifecycle` — turn/step boundaries.
- `tripwire` — a rule matched; includes the rule id, action, and the event.
- `cease` / `resume` — the emergency control changed state, with the reason
  and the actions that were attempted.
- `error` — the runtime or stream failed.

Payloads pass through the same recursive key redaction as the audit log
(`token`, `api_key`, `password`, `cookie`, …), and long strings are truncated.
Redaction is **key-based**, not a guarantee that no secret is ever in a value.
Treat the ledger as sensitive.

The ledger is not the approval audit. Approval records deliberately hash tool
arguments; the activity ledger keeps the concrete input/output because that is
what “show me exactly what it did” requires.

## Live feed

```text
GET /v1/activity?conversation_id=...&limit=200
GET /v1/activity/stream?conversation_id=...      # SSE
```

The Studio chat already consumes the agent stream. The additional tool detail
now comes from the enriched face events; the **Cease** button calls
`POST /v1/activity/control`.

A separate process can follow the same feed without touching the Studio UI:

```bash
hive activity tail --follow --json
```

## Cease semantics

`POST /v1/activity/control` with `action: cease` performs, in order:

1. Create a durable block in `settings/activity-cease.json` for the session
   and/or conversation.
2. Ask the agent service to force-stop the runtime and its child process tree.
3. Kill Docker containers labelled `hive.session=<session-id>` (the sandbox
   labels every container it starts).
4. Revoke all in-memory session grants for that session.
5. Optionally revoke every global grant with `revoke_all: true`.
6. Emit a `cease` event to the live SSE stream and record the action result.

From then on, `ApprovalGateway.authorize` denies every tool call for that
session (and inherited child sessions) with `cease requested; this session is
blocked until resume`. A new agent turn for the conversation is refused before
the runtime is rebuilt.

`action: resume` clears the durable block. It does **not** restart the runtime;
the next user message rebuilds it.

### What Cease does not cover

- A host-kernel exploit. This is still an application boundary.
- A process that is neither a descendant of the runtime nor in a labelled
  sandbox container.
- A network exfiltration already in progress through an allowed tool.
- Data already written to the AI-owned state tree.
- A compromised browser process that has escaped the runtime tree.

Cease is a fast, explicit containment action, not a proof of safety.

## Tripwire rules

Rules live in `settings/activity-rules.json` and are **disabled by default**.
An automatic cease is a policy decision, so the operator must enable it.

Schema:

```json
{
  "enabled": true,
  "rules": [
    {
      "id": "private-key-output",
      "action": "cease",
      "reason": "private key material in tool output",
      "match": {
        "type": "tool",
        "phase": "result",
        "field": "result",
        "regex": "BEGIN (RSA |OPENSSH )?PRIVATE KEY"
      }
    }
  ]
}
```

`match` fields:

| Key | Meaning |
| --- | --- |
| `type` | `user`, `assistant`, `tool`, `model`, `lifecycle`, `error`, … |
| `phase` | `call`, `result`, `start`, `end` |
| `tool` | exact tool name |
| `field` | `any`, `text`, `args`, `result`, `output`, `action` |
| `regex` | case-insensitive regular expression |
| `contains` | case-insensitive literal substring |

`action` is `warn`, `revoke`, or `cease`. `revoke` removes the session's
in-memory grants but keeps the turn running. `warn` only records a `tripwire`
event.

Examples:

```json
[
  {
    "id": "recursive-root-delete",
    "action": "cease",
    "match": {"type": "tool", "phase": "call", "tool": "bash",
              "field": "args", "regex": "rm\\s+-rf\\s+/(\\s|$)"}
  },
  {
    "id": "human-home-access",
    "action": "cease",
    "match": {"field": "any", "regex": "/home/penis|\\.guard"}
  },
  {
    "id": "curl-pipe-shell",
    "action": "warn",
    "match": {"type": "tool", "phase": "call", "tool": "bash",
              "field": "args", "regex": "curl[^|]*\\|\\s*(ba)?sh"}
  }
]
```

Rules are evaluated as events are recorded. A tool `call` event is emitted when
DSH reports the call; the approval gateway is still the pre-execution gate. A
rule is a *tripwire*, not a sandbox: it can stop the turn after a call or
result is observed, and it can revoke access, but it cannot undo an effect that
already happened.

## CLI

```bash
hive activity tail --limit 50
hive activity tail --follow --json
hive activity rules --json
hive activity rules --set rules.json --enable
hive activity rules --disable
hive activity cease --conversation c1 --reason "saw it touch GUARD" --yes
hive activity resume --conversation c1
```

`cease` needs a running Studio because it must stop the runtime; `tail` and
`rules` operate on the state directory directly.

## External sentinel

The policy/research layer lives in a separate repository, `hive-sentinel`. It
tails `activity-YYYYMM.jsonl`, applies operator rules, and calls
`POST /v1/activity/control` with `action: cease`. This keeps Hive's core small
and lets the sentinel evolve into a broader security suite without coupling
the two release cycles.

The sentinel must not run as the same Unix user as Hive if it is meant to
survive a compromised Hive process; that is part of its threat model.

## Tests

```bash
pytest tests/unit/test_activity.py tests/unit/test_agent_events_shaping.py
```

The tests cover ledger redaction/replay, rule validation, disabled-by-default
behavior, cease/resume, grant revocation, gateway denial after cease, and the
richer face-level tool events.
