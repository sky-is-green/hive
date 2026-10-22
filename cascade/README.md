# The Hive Cascade

**A gold-free verification-and-repair loop for local language models.**

A wrong answer is rarely uncertain; it is confidently wrong. The cascade checks
every answer before it is served, repairs what can be repaired, and escalates
what cannot, with verifiers that never see the answer key.

| | |
|---|---|
| [The paper](CASCADE-PAPER.md) | the method, the gold-free contract, the measured tier results and the negative results |
| [The pilot](CASCADE-PILOT.md) | the first end-to-end run and the console contract |
| [The controller](HARNESS-CONTROLLER.md) | the four tools, the invariants and the bounded loop |
| [Prior art](CONTROLLER-PRIOR-ART.md) | how other harnesses handle verification and routing |
| [The human gate](HUMAN-GATE.md) | the escalation lane's operator surface |
| [The visual companion](https://huggingface.co/spaces/SkyIsNotGreen/hive-cascade-paper) | the seven-tab walkthrough |

The implementation is in [`../harness/cascade/`](../harness/cascade/) and
[`../experiments/cascade/`](../experiments/cascade/); the evaluation suite is
[`../hivebench/`](../hivebench/), and the measured log is
[`../docs/AB-RESULTS.md`](../docs/AB-RESULTS.md).
