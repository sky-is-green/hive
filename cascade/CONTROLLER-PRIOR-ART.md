# Prior art: how other harnesses verify and route

The role this document is about is **not generation**. It is the layer that
decides *who answers, what counts as done, and when to stop*: the
controller in `HARNESS-CONTROLLER.md`. The field builds this under
several names, and the names hide the same four-part loop.

## What the field calls the role

| the name | where it comes from | what it actually is |
|---|---|---|
| **agent harness** | Databricks, 2026 | model + tools + sandbox + memory + guardrails + feedback loops |
| **self-reflective RAG** | Self-RAG, CRAG, LangGraph | retrieve, grade, generate, grade, rewrite, retry |
| **verification loop** | Aider | edit, run the repo's lint/tests, repair |
| **self-healing tool calling** | Unsloth Studio | repair the *call*, not the answer |
| **grounding verification** | RAG production practice | claim-level support against sources |

## The landscape (what each one gets right)

**Unsloth Studio** is the closest thing to the host we are building, and it
is deliberately *not* the controller. It runs models, gives them tools,
sandboxes **Bash and Python code execution**, offers **private web search
that visits and cites pages**, RAG, and an OpenAI-compatible API. Its
`unsloth start claude|codex|opencode|hermes|...` configures an existing
agent CLI to point at the local endpoint without modifying the agent.
That is the drop-in property in both directions: the host does not care
which controller drives it.

What to take: **the host/controller split**, the sandbox, and the
tool-call healing. What it does not do: verify the *answer*. Its
"self-healing" loop repairs malformed tool calls; correctness is left to
the agent's own loop.

**OpenCode** is the closest architectural match to what we are building,
and the difference is instructive. It is a client-server coding agent
with sessions and a fixed tool catalog (`read`, `glob`, `grep`, `edit`,
`write`, `patch`, `shell`, `webfetch`, `websearch`, `question`, `skill`,
`subagent`, `execute`, plus MCP tools). It has **no answer verifier**:
verification is `shell` - the agent runs the project's tests and linter,
reads the output, and repairs - plus the human approval gate. What it
has that the others do not is a **structural policy plane**, and it is
worth copying almost verbatim.

- **Permissions** are ordered `(action, resource, effect)` rules -
  `allow`, `deny`, or `ask` - matched per agent, last match wins. A
  reviewer subagent in their own docs is defined by `edit: deny` and
  `shell: deny`: the *role* is its permission set, not its prompt.
- **Policies** are a second, non-interactive plane that can only
  *tighten*: a `deny` cannot be lifted by an agent or a project, and a
  managed/Console policy outranks both. The model cannot talk its way
  past it.
- **`steps`** caps the model steps per turn; on the last step the tools
  are removed and the model must summarize in text. A loop bound, not a
  hope.
- **Code Mode** (`execute`) lets the controller write a small JavaScript
  program that calls tools, runs independent calls in parallel, and
  returns one condensed result - nested tools still enforce their own
  permissions. That is how a controller keeps its context small while
  the policy stays auditable.
- The **Console gateway** serves other harnesses (Claude Code, Codex,
  Hermes, Pi, Raycast) with session routing - the same host/controller
  split as Unsloth Start, from the other side.

What OpenCode does *not* have is the part our campaign measured: a table
of which tier is competent at what, a cost-aware escalation, a
falsifiable check (no mutation test, no `unknown`), or claim-level
grounding. Its verification is only as good as the tests the model
chooses to run. That is precisely the gap the controller's measured
table fills.

**LangGraph's Self-RAG and CRAG** are the clearest published state
machines for the verification role. Self-RAG names four signals: whether
to retrieve (`Retrieve`), whether a passage is relevant (`ISREL`),
whether the generation is supported by the passage (`ISSUP`), and whether
the answer is useful (`ISUSE`). CRAG adds a lightweight *retrieval
evaluator* with a confidence, a web-search fallback when retrieval is
ambiguous, and knowledge refinement. Both implement the loop as a graph
with conditional edges, not as one big judge.

What to take: **stage the checks**, one cheap signal per stage, and make
the transitions explicit. What it does not do: execute the answer. A
grounding check is still a reader, with the same blind spot a judge has.

**Aider** is the cleanest example of execution as the verdict. It runs
the project's own lint and tests after each edit and feeds the failures
back for repair. The verifier is the repository, not the model.

What to take: **the check must be the domain's own tool**. What it does
not do: route between tiers; it is one model in one loop.

**Databricks' harness guide** is the best vocabulary for the whole
system: the reason-act-observe loop, sandboxed execution, context
compaction, human-in-the-loop guardrails, observability, and - named
explicitly - **weak verification** as a failure mode: agents stop early
or declare success on incomplete work.

What to take: the loop shape and the observability requirement. What it
does not do: say which tier to use; that is a policy problem.

**RAG grounding practice** (claim-level support, faithfulness,
citations - Dataiku, Google's check-grounding API, RAGAS-style metrics)
is the one place the field checks an answer against *evidence* rather
than against itself. It decomposes an answer into claims and asks
whether each is supported by the retrieved source.

What to take: **a second kind of verifier** - grounding, not execution -
for the answers code cannot settle.

## The isomorphism

The RAG loop and our cascade are the same loop with different names:

| the RAG loop | our cascade |
|---|---|
| retrieval grade (`ISREL`) | the membrane (dedup, drift, selection) |
| grounding (`ISSUP`) | `verify` - execution where possible, grounding otherwise |
| usefulness (`ISUSE`) | `score` - the judge head |
| query rewrite | repair with the next tier (Bonsai) |
| web fallback (CRAG) | escalation (Bonsai, then API) |

That is the useful part of "many only use it for RAG": the RAG harnesses
have already worked out the *shape* of the loop, the conditional edges
and the metrics. They have not worked out the execution branch, which is
where our measured value is.

## Five things to adopt

1. **Stage the checks, cheap to expensive.** Retrieval grade, then
   grounding/execution, then usefulness. One signal per stage; no single
   scalar "quality".
2. **Execution or grounding before judgement.** A judgement model is a
   reader; it cannot see a plausible wrong answer. The 176B head gave
   four of seven errors `p >= 0.97`.
3. **A check must be falsifiable.** Our addition, not borrowed: a check
   that passes a deliberately mutated answer is rejected as `unknown`.
   This is the anti-`print("PASS")` rule, and it is what a 0.8B checker
   needs in order to be safe.
4. **The host is controller-agnostic.** Unsloth Start and OpenCode's
   Console gateway are the precedents: a stable endpoint plus a
   launcher, and any agent can drive it. Our four tools are the
   equivalent contract.
5. **Enforce the invariants outside the model.** OpenCode's two-plane
   design is the model to copy: permissions say what the agent may do,
   policies say what it cannot do at all, and neither is a prompt. Our
   "a pass is authoritative" and "the controller cannot mark its own
   answer correct" should be properties of the tool surface and the
   policy engine, not sentences the controller is asked to respect.

## Two things not to copy

- **Self-grading without evidence.** The loop most chat UIs implement
  ("the model reviews its own answer") is the thing our measurements
  falsify.
- **One scalar quality score as a runtime gate.** RAGAS/faithfulness
  scores are *evaluation* metrics: they are computed after the fact and
  on a set, not per request. Using one as the gate is what produced the
  drift-threshold mistake in 2C-sexies: a number that ranks well but
  decides badly at the operating point.

## Mapping to our contract

| our tool | the prior art it should follow |
|---|---|
| `ask(model, prompt)` | Unsloth Start's model swap; the Studio registry |
| `verify(task, answer)` | Aider's lint/test loop + Unsloth's sandbox; grounding (claim-level) for the factual remainder |
| `score(task, answer)` | Self-RAG's `ISUSE`; the 176B head as the judge |
| `ask_human(...)` | Databricks' human-in-the-loop; the Studio's human gate |
| the JSONL log | Databricks' observability; LangSmith traces |
| the invariants | OpenCode's two planes: permissions (what the agent may do, `ask` allowed) and policies (hard `deny`, non-interactive, cannot be lifted by the agent) |
| the loop bound | OpenCode's `steps`: a hard cap, then the tools are removed and the model must answer |

## What this changes in our build

The prior art says the checker should be **two verifiers, not one**:

- **execution** - run the code, recompute the arithmetic (the four
  measured misses);
- **grounding** - for answers code cannot settle, check each claim
  against the retrieved context, with citations.

The judge head stays as the `unknown` fallback, not the gate. That is a
more precise version of "the checker is a 4-9B with tool calls": the
model's job is to *write the check* (execution) or *extract the claims*
(grounding); the harness runs it; the verdict is the result, not the
model's opinion.

## Sources

- Unsloth Studio (sandboxed Bash/Python, self-healing tool calls, private
  search with citations, OpenAI-compatible API, `unsloth start`):
  <https://unsloth.ai/docs/new/studio> and
  <https://unsloth.ai/docs/integrations/unsloth-start>
- OpenCode V2 (tools, agents and subagents, permissions, policies,
  `steps`, Code Mode, Console harness gateway):
  <https://opencode.ai/v2/docs/tools>,
  <https://opencode.ai/v2/docs/agents>,
  <https://opencode.ai/v2/docs/permissions>,
  <https://opencode.ai/v2/docs/policies>,
  <https://opencode.ai/v2/docs/console/harnesses>
- LangGraph, "Self-Reflective RAG" (Self-RAG `Retrieve`/`ISREL`/`ISSUP`/
  `ISUSE`, CRAG retrieval evaluator and web fallback):
  <https://www.langchain.com/blog/agentic-rag-with-langgraph>
- Self-RAG paper: <https://arxiv.org/abs/2310.11511>; CRAG:
  <https://arxiv.org/pdf/2401.15884>
- Aider lint/test loop: <https://aider.chat/docs/usage/lint-test.html>
- Databricks, "What is an AI Agent Harness?" (loop, sandbox, weak
  verification, observability): <https://www.databricks.com/blog/ai-harness>
- Claim-level grounding: Dataiku, "Grounding verification in AI";
  Google, "Check grounding with RAG"
  (<https://docs.cloud.google.com/generative-ai-app-builder/docs/check-grounding>);
  Openlayer, "Groundedness, faithfulness" (RAG evaluation).
