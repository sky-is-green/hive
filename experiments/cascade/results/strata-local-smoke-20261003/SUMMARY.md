# Local-first default smoke — 2026-10-03

Flow: Strata (managed via `stacks/strata-125b.json`) answers every task;
Tiny-Jev validates on **CPU**; no router, no API, no gateway.

| step | result |
|---|---|
| apply | managed start, `pid` registered, `resident=true` |
| tasks | 2 (test split: `gsm8k-0005`, `gsm8k-0006`) |
| local | 2/2 correct (305 / 173 tokens; ~26 s/task, cold first request) |
| validation | Tiny-Jev CPU accept at p=0.990 / 0.989, 0 escalations |
| API | off (`api_fallback: null`, `api_model: null`, cost **$0**) |
| report | `report.json` — `router=none`, `judge=tiny-jev`, `router_mode=gate`, `judge_device=cpu` |
| teardown | stack unload stopped the managed engine; box clean |

**Incident (two machine restarts).** The two smoke attempts before this one
loaded Tiny-Jev on CUDA while Strata was resident.  The second ROCm context
hung the host and forced a hard restart — twice.  `harness/models.py` already
warns that two independent ROCm contexts can hang this box; the validator now
defaults to CPU (`--judge-device cpu`, `TinyJevJudge(device="cpu")`) and the
runner prints a loud warning when cuda is chosen while the local engine is
alive.  A 1.7B forward pass on CPU is far cheaper than one generation.
