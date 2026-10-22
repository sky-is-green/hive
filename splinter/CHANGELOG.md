# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Detailed development history for the period before this file existed lives in
the git log; entries below summarise the capability milestones.

## [Unreleased]

### Changed

- Renamed the project from **Strata** to **Splinter**. The package is
  `splinter-memory` (import `splinter`), the facade classes are `Splinter` /
  `SplinterConfig`, the wire routes are `/v1/splinter/*`, the conversation
  header is `X-Splinter-Conversation`, and the MCP tools are `splinter_search`
  / `splinter_remember`. Docs, diagrams, and scripts are updated; the GitHub
  repository and the CI branch are renamed separately.
- Added `REPRODUCE.md` (claim-to-command map with an explicit
  deterministic / live / recorded grading), `CITATION.cff`, and this changelog.

### Fixed

- Removed hard-coded personal filesystem paths from `import_conversation.py`
  and `scripts/splinter-session.sh` in favour of environment/home resolution.

## [0.1.0] - 2026-08-22

Initial public architecture and measurement protocol.

### Added

- Separation architecture: small CPU bidirectional "drone" encoders score and
  select context while the generative model only generates (`sieve`, `cortex`,
  `retention`, `focal`, `membrane`, `backend`).
- Bounded, relevance-curated assembly with an adaptive per-route-token budget
  and a relevance floor plus per-chunk window-share cap.
- Managed decay (decay matrix, stale factor), semantic dedup, and topic-drift
  detection in the retention path.
- Remembrance / comb-surplus tier for resurrecting old, overflowed chunks.
- Store-time hygiene: secret redaction, base64 collapsing, chunk-length caps.
- Backends: LM Studio, generic OpenAI-compatible, and a mock-tested vLLM path.
- Deterministic P2 retrieval diagnostic over fixture ground truth (no
  LLM-as-judge in the evidence path).
- Falsifiable P1–P11 prediction protocol with measured PASS/FAIL verdicts, a
  quantified PES health metric, and a published threats/limitations section.
- Standalone serving boundary: an OpenAI-compatible endpoint, a `/v1/mcp`
  JSON-RPC server exposing `splinter_search` / `splinter_remember`, and a
  `splinter-serve` console script.
- Async auditor (ground-truth labelling) layer.
- CI that installs the system package clean and smoke-tests the facade import.

### Changed

- Extracted the evaluation suite and Studio sidecar into the sibling
  [hivebench](https://github.com/sky-is-green/hivebench) repository.
