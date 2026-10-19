# HiveBench Studio API

Every endpoint the Studio serves, grouped by what it is for.

**Live, always-current docs:** start the Studio and open <http://127.0.0.1:8765/docs>
(Swagger UI) or <http://127.0.0.1:8765/redoc>. The table below is the
hand-maintained map; the live docs carry the request/response schemas.

The map is checked against the running app: `tests/unit/test_api_docs.py`
fails if an endpoint is missing from it, if it documents an endpoint that
no longer exists, or if a method is wrong. **89 paths, 97 operations** (80
of the paths under `/v1/`).

```sh
python -m harness            # studio on :8765
curl http://127.0.0.1:8765/health
```

## Conventions

- **Base URL** `http://127.0.0.1:8765` (override with `--host`/`--port`).
- **Auth** — none by default (local-only). Set `STRATA_API_KEY` to require a
  bearer token on `/v1/*`; the Studio sends it itself.
- **Content type** — JSON in and out, except the multipart model import and
  the SSE streams (`/v1/agent/stream`, `/v1/splinter/stream`).
- **Optimistic concurrency** — `GET /v1/provider/config` and `GET /v1/engines`
  return a `revision` token. Send it back as `expectedRevision` on the next
  write; a stale token is a **409** (someone else changed the config first).
- **Secrets** — provider keys are stored and returned masked (`***`); the
  cascade runner's argv is redacted before it is echoed back.

## Endpoint map

### Cascade — the Hive Cascade console contract

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/cascade/frontier` | Frontier measurements per role (quality vs latency curves). |
| `GET` | `/v1/cascade/paths` | The route table: path ids, escalation paths, and expected costs. |
| `GET` | `/v1/cascade/plan` | Plan the route for a bucket + confidence + output length. |
| `GET` | `/v1/cascade/registry` | The candidate catalog: every model candidate with the roles it covers. |
| `GET` | `/v1/cascade/report/{run}` | Return the run_report.json bundle for a run directory. |
| `GET` | `/v1/cascade/roles` | The role taxonomy (A1, B1, C1, D3, E4, ...) with descriptions. |
| `POST` | `/v1/cascade/run` | Launch — body: `#/components/schemas/CascadeRunRequest` |
| `GET` | `/v1/cascade/runs` | List run bundles available for viewing. |
| `POST` | `/v1/cascade/validate` | Validate a candidate answer against a task with the validator model. |
| `GET` | `/v1/cascade/validator` | Validator service state (model, device, loaded). |

### Models — local library, Hugging Face hub

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/models` | List served models and their status (setup tooling shape). |
| `GET` | `/v1/models/hub` | Search Hugging Face for GGUF repos. |
| `POST` | `/v1/models/hub/download` | Start downloading a model file from Hugging Face. — body: `#/components/schemas/HubDownloadRequest` |
| `GET` | `/v1/models/hub/downloads` | Download job states (queued/downloading/done/error). |
| `GET` | `/v1/models/hub/files/{repo}` | List the GGUF files inside a Hugging Face repo. |
| `DELETE` | `/v1/models/linux` | Delete a GGUF from /mnt/dsh_storage/models. |
| `GET` | `/v1/models/linux` | List GGUFs in /mnt/dsh_storage/models (WSL ext4). |
| `DELETE` | `/v1/models/local` | Delete one GGUF from the local library. |
| `GET` | `/v1/models/local` | List GGUF models in the local library directory. |
| `POST` | `/v1/models/local/import` | Import GGUF files from a local folder (webkitdirectory upload). |
| `POST` | `/v1/models/local/import-path` | Link an external folder without copying (instant). |
| `POST` | `/v1/models/local/path` | Point the library at a chosen folder (no copy, instant). |
| `POST` | `/v1/models/move-to-linux` | Move a System GGUF to /mnt/dsh_storage/models via WSL. |
| `POST` | `/v1/models/move-to-windows` | Move a Linux GGUF back to System library (host models_dir). |

### Servers — launch, stop, health

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/server/log` | Last lines of the llama-server logs (startup errors live here). |
| `GET` | `/v1/server/memory` | Sidecar RSS + conversation accounting — the leak-detection probe. |
| `GET` | `/v1/server/metrics` | Proxy llama-server's prometheus /metrics (kv usage, throughput). |
| `POST` | `/v1/server/start` | Launch a local model server (llama.cpp/strata) with the given options. — body: `#/components/schemas/ServerStartRequest` |
| `GET` | `/v1/server/status` | Hardware (VRAM/RAM) plus loaded model health and instances. |
| `POST` | `/v1/server/stop` | Unload one instance (key) or the whole local fleet (no key). |
| `POST` | `/v1/server/unload` | Unload one instance and retire its provider + engine profile. — body: `#/components/schemas/ServerUnloadRequest` |

### Engines — reusable model profiles

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/ab/bench` | Alias of the A/B bench (kept for older clients). |
| `GET` | `/v1/engines` | List engine profiles and the default one. |
| `POST` | `/v1/engines` | Replace the engine profile list (name, kind, model, load options, sampling). — body: `#/components/schemas/EngineConfigRequest` |
| `POST` | `/v1/engines/ab/bench` | Benchmark two engine profiles head to head on the same port base. |
| `POST` | `/v1/engines/bench` | Alias of the A/B bench (kept for older clients). |
| `GET` | `/v1/engines/preset` | Auto preset for the Engine profiles section. |

### Providers — OpenAI-compatible endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/provider/config` | List configured providers and the default one. |
| `POST` | `/v1/provider/config` | Replace the provider list (name, base_url, api_key, model). — body: `#/components/schemas/ProviderConfigRequest` |

### Agent — dsh turns, presets

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/agent-presets/copy` | Copy an agent preset to a new id. |
| `GET` | `/v1/agent-presets/list` | List available agent presets (system and user). |
| `POST` | `/v1/agent-presets/read` | Read one agent preset's content. |
| `POST` | `/v1/agent-presets/remove` | Delete a user agent preset. |
| `POST` | `/v1/agent/cancel` | Cancel the in-flight agent turn for a conversation. |
| `GET` | `/v1/agent/status` | Agent service state: running, model, last error. |
| `POST` | `/v1/agent/stream` | Stream an agent (dsh) turn with tool use over SSE. — body: `#/components/schemas/AgentMessageRequest` |

### Commands — console command palette

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/commands` | List console commands (name, description, input hint). |
| `POST` | `/v1/commands/run` | Run a console command line, e.g. '/status'. — body: `#/components/schemas/CommandRunRequest` |

### Sandbox — isolated code execution

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/sandbox/exec` | Run a command in the sandbox (Docker when the daemon is up, else a local subprocess). — body: `#/components/schemas/SandboxExecRequest` |
| `GET` | `/v1/sandbox/status` | Sandbox availability: docker installed, daemon up, fallback mode. |

### Studio — the console's own toggles

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/studio/settings` | The console's toggles: sandbox, tool use, auto-compact, dev mode, the Strata base URL. |
| `PUT` | `/v1/studio/settings` | Change those toggles; returns the stored result. |
| `GET` | `/v1/studio/strata` | Probe the local Strata engine: alive, base URL, model, health. Never raises. |

### Compaction — checkpoints and engine surgery

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/compaction/run` | Compact a transcript into a checkpoint (manual or auto trigger). |
| `POST` | `/v1/compaction/surgery` | Offer the engine switch at that checkpoint; commits only when the canary gate passes. |
| `GET` | `/v1/compaction/status` | Whether automatic compaction is on, and the compaction settings. |

### Stacks — multi-tier model stacks

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/stacks` | List the saved stack definitions. |
| `GET` | `/v1/stacks/status` | Stack manager status: applied stack, tier, health. |
| `DELETE` | `/v1/stacks/{name}` | Delete a saved stack definition. |
| `GET` | `/v1/stacks/{name}` | One saved stack definition by name. |
| `PUT` | `/v1/stacks/{name}` | Create or replace a saved stack definition. |
| `POST` | `/v1/stacks/{name}/apply` | Apply a saved stack (load its models). |
| `POST` | `/v1/stacks/{name}/unload` | Unload a saved stack's models. |
| `POST` | `/v1/stacks/{name}/validate` | Validate a saved stack against the local hardware. |

### Training — weight training and the preset trainer

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/trainer/draft` | Draft a candidate preset from the evidence. |
| `POST` | `/v1/trainer/evaluate` | Run candidate vs baseline on bench tasks, score, compare. |
| `GET` | `/v1/trainer/evidence` | Mine session logs for tool-use patterns and outcomes. |
| `GET` | `/v1/training/engines` | List training engines and the ternary runner. |
| `POST` | `/v1/training/launch` | Launch a training run for an engine with the given fields. — body: `#/components/schemas/TrainingLaunchRequest` |
| `POST` | `/v1/training/report/{run_dir}` | Write and return the training report for a run directory. |
| `GET` | `/v1/training/status/{run_dir}` | Training run progress for a run directory. |

### Runs — bundles, reports, the protocol

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/protocol/run` | Run the mock or live HiveBench protocol into a run directory. — body: `#/components/schemas/ProtocolRunRequest` |
| `GET` | `/v1/report/{run_dir}` | Return the run_report.json bundle for a run directory. |
| `GET` | `/v1/runs` | List run bundles available for viewing. |

### System — processes, hardware, setup

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat passthrough for the mock backend. |
| `POST` | `/v1/embeddings` | OpenAI-compatible embeddings endpoint. |
| `GET` | `/v1/gpu/processes` | Per-process VRAM (from /proc/<pid>/fdinfo), on-demand. |
| `GET` | `/v1/processes` | Process manager: spawned servers (llama-servers) + child shells. |
| `POST` | `/v1/processes/kill` | Kill one process by pid or by managed key (S4 kill button). |
| `GET` | `/v1/research/queue` | Pending deep-research questions. Execution is QUEEN-only: entries are |
| `POST` | `/v1/research/queue` | Queue a deep-research question for the background worker. |
| `POST` | `/v1/setup/bootstrap` | One-click bootstrap for Docker → WebUI (mount + compose up). No Admin bare mount. |
| `POST` | `/v1/setup/create-vhdx` | Explicit VHDX creation at user-picked location (first-time user). |
| `GET` | `/v1/setup/docker-models` | Proxy to Docker 8000 /v1/models (avoids browser CORS). |
| `GET` | `/v1/setup/drives` | List drives and best VHDX path for first-time user (auto-detect). |
| `GET` | `/v1/setup/health` | Lightweight health for the wizard (VHDX/mount/shards/docker). |
| `POST` | `/v1/setup/mount-bare` | Bare-expose VHDX to WSL (requires Admin). Called by Setup UI button. |
| `GET` | `/v1/setup/status` | Setup wizard status — engine + drive + health + tier (splinter console). |

### Pages — the Studio console

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Redirect to the run index. |
| `GET` | `/bee.svg` | Bee icon used by the Studio chrome. |
| `GET` | `/cascade` | Hive Cascade launch console (every runner knob, collapsible advanced). |
| `GET` | `/favicon.svg` | Studio favicon (hive hexagon). |
| `GET` | `/health` | Liveness probe: the sidecar is up. |
| `GET` | `/runs` | HTML index of run bundles. |
| `GET` | `/server` | The Studio console page (chat, models, engines, providers, hub, settings). |
| `GET` | `/stacks` | The stack management page. |
| `GET` | `/view/{run_dir}` | Rendered HTML report page for a run bundle. |

### Ungrouped

- `/v1/compaction/run`
- `/v1/compaction/surgery`
- `/v1/compaction/status`
- `/v1/studio/settings`
- `/v1/studio/strata`

## The Studio pages

- `/` — Redirect to the run index.
- `/bee.svg` — Bee icon used by the Studio chrome.
- `/cascade` — Hive Cascade launch console (every runner knob, collapsible advanced).
- `/favicon.svg` — Studio favicon (hive hexagon).
- `/health` — Liveness probe: the sidecar is up.
- `/runs` — HTML index of run bundles.
- `/server` — The Studio console page (chat, models, engines, providers, hub, settings).
- `/stacks` — The stack management page.
- `/view/{run_dir}` — Rendered HTML report page for a run bundle.

## Related

- `docs/CASCADE-PILOT.md` — the cascade design, the console contract, and the
  measured results.
- `experiments/cascade/LEARNED-GATE-PLAN.md` — the learned gate and its
  pre-registered acceptance rule.
- `experiments/cascade/SAMPLE-CHECK-PLAN.md` — the label-minting loop that
  produces the gate's training corpus.
- `experiments/cascade/NEXT-PLAN.md` — what comes next.
