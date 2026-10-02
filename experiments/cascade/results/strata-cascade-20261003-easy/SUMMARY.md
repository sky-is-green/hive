# Cascade pilot (E4=Strata) — strata-cascade-20261003-easy

Strata: `/home/penis/Desktop/work/strata` @ `db4f91a1171d` (served model `qwen3.8-flash-next-q2_0`, ctx 4096)
Model sha256: `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720`
Tasks: `experiments/cascade/tasks.json` (13 tasks, measure mode)

| metric | value |
|---|---:|
| local (Strata) accuracy | 1.0 |
| API (flash) accuracy | 1.0 |
| cascade accuracy | 1.0 |
| escalation rate | 0.0 |
| false accepts | 0 |
| false rejects | 0 |
| router agreement | 1.0 |
| Strata mean ms | 3818.0923 |
| RAM available min | 22.82 GB |
| VRAM peak | [17.684452056884766, 19.07910919189453] GB |

Caveats:
- single sample per task; no held-out split
- frontier router + frontier verifier (DeepSeek via LiteLLM); local decision models not co-resident with Strata
- Strata adaptive expert cache + MTP speculation: greedy output is not byte-identical run to run
- measure mode: every task gets a local Strata answer and an API answer for comparison
