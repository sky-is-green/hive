"""Model candidate registry and resident-set consolidation.

One :class:`Candidate` is one model that can fill one or more roles.  The
clearest point of the brief's consolidation rule (brief §"Role
consolidation"): *a model can serve several roles, so the scheduler should
reuse one resident model across roles rather than loading a specialist per
role*.  This module therefore holds two things:

- the catalog of candidates with the roles they cover, their device tiers and
  their resident size;
- :func:`select_resident_set`, the minimal-cover choice: a resident set that
  covers every required role within per-device capacity.  The brief's
  "consolidated vs specialist at equal cost" comparison is a later oracle
  measurement; the selection here only decides *which* candidates would need
  to be resident for the coverage.

Nothing here spawns or loads a model.  A chosen resident set is what
:mod:`harness.stack` would launch as a stack once the oracle has measured the
candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional, Sequence

from harness.cascade.roles import Device, get_role


@dataclass(frozen=True)
class Candidate:
    """One registrable model.

    ``roles`` are role ids this candidate can serve.  ``devices`` is its
    placement preference (first = preferred); ``bytes_gb`` is the resident
    weight footprint and ``precision`` the quantized form the number refers
    to.  ``bytes_gb=None`` means "unmeasured" — consolidation treats it as
    zero-size and callers should not read a fit decision as final.
    """

    id: str
    label: str
    repo: str
    file: Optional[str] = None
    params_b: Optional[float] = None
    precision: str = "bf16"
    bytes_gb: Optional[float] = None
    devices: tuple[Device, ...] = (Device.DGPU0,)
    roles: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "repo": self.repo,
            "file": self.file,
            "params_b": self.params_b,
            "precision": self.precision,
            "bytes_gb": self.bytes_gb,
            "devices": [d.value for d in self.devices],
            "roles": list(self.roles),
        }


#: The pilot's candidate catalog (brief §"Roles, not models"), sized for this
#: box where a size is known.  Local availability is a separate concern: the
#: catalog is the *declared* set; the oracle records what was measured.
DEFAULT_CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        id="minilm-l6", label="all-MiniLM-L6-v2",
        repo="sentence-transformers/all-MiniLM-L6-v2",
        params_b=0.02, precision="fp32", bytes_gb=0.09,
        devices=(Device.CPU, Device.IGPU),
        roles=("A1", "B1", "B4", "F1"),
    ),
    Candidate(
        id="bge-small-en", label="bge-small-en-v1.5",
        repo="BAAI/bge-small-en-v1.5",
        params_b=0.03, precision="fp32", bytes_gb=0.13,
        devices=(Device.CPU,),
        roles=("A1", "B1", "B4", "F1"),
    ),
    Candidate(
        id="bge-m3", label="bge-m3",
        repo="BAAI/bge-m3",
        params_b=0.57, precision="fp16", bytes_gb=2.27,
        devices=(Device.CPU, Device.IGPU),
        roles=("A1", "B1"),
    ),
    Candidate(
        id="florence-2-base", label="Florence-2-base",
        repo="microsoft/Florence-2-base",
        params_b=0.23, precision="fp16", bytes_gb=0.46,
        devices=(Device.IGPU, Device.DGPU0, Device.CPU),
        roles=("A2",),
        notes=(
            "CPU fp32 measured (A2 oracle 2026-10-17, "
            "results/a2-oracle-stage0-20261017): 0.93 GB, 0.80 s/call, 4/6 "
            "checks — OCR strong (text, signs, chart labels), OD labels weak "
            "on synthetic shapes; remote code needs transformers <4.50 "
            "(pinned env .envs/florence-legacy, 4.46.3)"
        ),
    ),
    Candidate(
        id="smolvlm2-256m", label="SmolVLM2-256M",
        repo="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        params_b=0.26, precision="fp16", bytes_gb=0.5,
        devices=(Device.IGPU, Device.DGPU0, Device.CPU),
        roles=("A2",),
        notes=(
            "CPU fp32 measured (A2 oracle 2026-10-17): 1.03 GB, 4.31 s/call, "
            "5/6 — best open-ended captioner/QA of the three; missed the "
            "chart tallest-bar question (answered A over B); native in the "
            "harness venv's transformers 5.x"
        ),
    ),
    Candidate(
        id="altclip", label="AltCLIP",
        repo="BAAI/AltCLIP",
        params_b=0.4, precision="fp16", bytes_gb=1.6,
        devices=(Device.IGPU, Device.DGPU0, Device.CPU),
        roles=("A2",),
        notes=(
            "CPU fp32 measured (A2 oracle 2026-10-17): 3.46 GB, 0.25 s/call, "
            "6/6 retrieval checks — the only true A2 encoder of the three "
            "(fp16 ~1.7 GB); prompt-sensitive on real photos: an ambiguous "
            "'contains text' distractor once outranked the scene option"
        ),
    ),
    Candidate(
        id="bge-reranker-v2-m3", label="bge-reranker-v2-m3",
        repo="BAAI/bge-reranker-v2-m3",
        params_b=0.57, precision="fp16", bytes_gb=2.27,
        devices=(Device.CPU, Device.IGPU),
        roles=("B2", "C4", "D1"),
    ),
    Candidate(
        id="jina-reranker-v3", label="jina-reranker-v3",
        repo="jinaai/jina-reranker-v3",
        params_b=0.6, precision="fp16", bytes_gb=1.2,
        devices=(Device.CPU, Device.IGPU),
        roles=("B2", "C4", "D1"),
    ),
    Candidate(
        id="mxbai-rerank", label="mxbai-rerank-base-v2",
        repo="mixedbread-ai/mxbai-rerank-base-v2",
        params_b=0.4, precision="fp16", bytes_gb=0.8,
        devices=(Device.CPU, Device.IGPU),
        roles=("B2",),
    ),
    Candidate(
        id="colbert", label="ColBERTv2",
        repo="colbert-ir/colbertv2.0",
        params_b=0.11, precision="fp16", bytes_gb=0.44,
        devices=(Device.CPU,),
        roles=("B3",),
    ),
    Candidate(
        id="qwen3.5-0.8b", label="Qwen3.5-0.8B",
        repo="Qwen/Qwen3.5-0.8B",
        params_b=0.8, precision="bf16", bytes_gb=1.6,
        devices=(Device.IGPU, Device.DGPU0),
        roles=("C1", "C2", "H1"),
    ),
    Candidate(
        id="qwen3.5-2b", label="Qwen3.5-2B",
        repo="Qwen/Qwen3.5-2B",
        params_b=2.0, precision="bf16", bytes_gb=4.0,
        devices=(Device.DGPU0,),
        roles=("C1", "D3", "E1", "E5", "F2"),
    ),
    Candidate(
        id="qwen3.5-4b", label="Qwen3.5-4B",
        repo="Qwen/Qwen3.5-4B",
        params_b=4.0, precision="bf16", bytes_gb=8.0,
        devices=(Device.DGPU0, Device.DGPU1),
        roles=("C3", "E2", "E3", "F2"),
    ),
    Candidate(
        id="qwen3.5-27b-int4", label="Qwen3.5-27B (int4)",
        repo="Qwen/Qwen3.5-27B",
        params_b=27.0, precision="int4", bytes_gb=14.0,
        devices=(Device.DGPU1, Device.DGPU0),
        roles=("E4",),
    ),
    Candidate(
        id="qwen3.5-27b-int8", label="Qwen3.5-27B (int8)",
        repo="Qwen/Qwen3.5-27B",
        params_b=27.0, precision="int8", bytes_gb=27.0,
        devices=(Device.DGPU0, Device.DGPU1),
        roles=("E4",),
        notes="spans both cards (tensor-split); excludes every other dGPU role",
    ),
    Candidate(
        id="intern-decision-0.8b", label="Intern-Decision-0.8B",
        repo="internlm/Intern-Decision-0.8B",
        params_b=0.8, precision="bf16", bytes_gb=1.71,
        devices=(Device.IGPU, Device.DGPU0, Device.CPU),
        roles=("C1", "C2", "C3", "C4", "D1", "D2", "D3", "H1"),
        notes=(
            "CPU bf16 measured (decision oracle 2026-10-17, "
            "results/decision-oracle-20261017): 1.7 GB on disk, 2.2 GiB peak "
            "RSS, 0.38 s/verdict, J-Bench test n=116: AUC 0.771, acc@0.5 "
            "74/116 (FA 35/87 wrong). Frozen-backbone trained readout "
            "(decision-readout-20261017): decision-position hidden states + "
            "ridge/MLP heads lift test AUC to 0.974/0.981 (MLP ECE 0.052, "
            "FA 3, task-disjoint split) with no model update. Stage 2 refit "
            "on J-Bench + 642 harvested 176B answers (224 organic / 418 "
            "amplified, results/decision-readout-harvest-20261017) held the "
            "bench (MLP AUC 0.980, acc 110/116, FA 2, ECE 0.045) but the "
            "held-out organic slice stays hard: refit MLP AUC 0.774 catching "
            "4/9 organic wrongs (raw 0.603 / 0 of 9). Raw 2b/4b/tiny-jev "
            "verdicts on the same 224 organic items (decision-oracle-harvest-"
            "20261017): 2b 0/9, 4b 2/9 (AUC 0.799), tiny-jev 1/9 — bigger "
            "judges do not close the gap. The refit's mid-depth L0.5 probe "
            "catches 6/9 (AUC 0.823, bench 93/116); L0.5+MLP+L1 reject-union "
            "7/9 at 20/57 corrects rejected. Two fluent micro-error wrongs "
            "evade every scorer — not shipped; next: weighted refit, then "
            "LoRA/targeted harvest"
        ),
    ),
    Candidate(
        id="intern-decision-2b", label="Intern-Decision-2B",
        repo="internlm/Intern-Decision-2B",
        params_b=2.0, precision="bf16", bytes_gb=4.43,
        devices=(Device.IGPU, Device.DGPU0, Device.CPU),
        roles=("C1", "C2", "C3", "C4", "D1", "D2", "D3", "H1"),
        notes=(
            "CPU bf16 measured (decision oracle 2026-10-17, "
            "results/decision-oracle-20261017): 4.4 GB on disk, 4.4 GiB peak "
            "RSS, 0.74 s/verdict, J-Bench test n=116: AUC 0.933, acc@0.5 "
            "95/116 (FA 19/87 wrong, FR 2/29 correct), no-leak thr 0.89 -> "
            "98/116 — the recommended resident D2/C2 verifier beside the 176B; "
            "organic harvest slice (224 items, results/decision-oracle-harvest-"
            "20261017): test-66 AUC 0.668, 0/9 organic wrongs rejected (all "
            "accepted at p>=0.57)"
        ),
    ),
    Candidate(
        id="intern-decision-4b", label="Intern-Decision-4B",
        repo="internlm/Intern-Decision-4B",
        params_b=4.0, precision="bf16", bytes_gb=9.08,
        devices=(Device.IGPU, Device.DGPU0, Device.CPU),
        roles=("C1", "C2", "C3", "C4", "D1", "D2", "D3", "H1"),
        notes=(
            "CPU bf16 measured (decision oracle 2026-10-17, "
            "results/decision-oracle-20261017): 9.1 GB on disk (language+vision "
            "shards), 8.9 GiB peak RSS, 1.94 s/verdict, J-Bench test n=116: "
            "AUC 0.979, acc@0.5 104/116 (FA 11/87, FR 1/29), no-leak thr 0.97 "
            "-> 101/116 — best quality; keep as on-demand hard verification "
            "(GPU readout reference: 184 ms/verdict); organic harvest slice "
            "(224 items): test-66 AUC 0.799, 2/9 organic wrongs rejected (the "
            "gsm8k arithmetic slips only)"
        ),
    ),
    Candidate(
        id="clef-flash-9b", label="Cloudflare Clef-Flash 9B",
        repo="Cloudflare/clef-flash",
        params_b=9.0, precision="bf16", bytes_gb=18.8,
        devices=(Device.DGPU0, Device.DGPU1),
        roles=("C1", "C2", "C3", "C4", "D1", "D2", "D3", "H1"),
        notes=(
            "Apache-2.0 decision model: Qwen3.5-9B backbone + joint schema "
            "head, SystemOne/Jev-compatible typed questions in one forward "
            "pass, no generation; median 38.8 ms (H200-class). Measured "
            "validator (CPU, bf16): 67/70 train / 26/30 test correct verdicts "
            "at 0.5, 0 false accepts train, ~3.2 s/check; ternary plan: PQ2_0 "
            "backbone + head sidecar (docs/CASCADE-PILOT.md 'Clef')"
        ),
    ),
    Candidate(
        id="clef-27b", label="Cloudflare Clef 27B",
        repo="Cloudflare/clef",
        params_b=27.0, precision="bf16", bytes_gb=54.0,
        devices=(Device.DGPU0, Device.DGPU1),
        roles=("C1", "C2", "C3", "C4", "D1", "D2", "D3", "H1"),
        notes=(
            "Apache-2.0 multimodal decision model: Qwen3.8-27B backbone + "
            "joint schema head, 64k context, SystemOne-compatible; Decision "
            "Index leader; bf16 ~54 GB needs the box alone"
        ),
    ),
    Candidate(
        id="tiny-jev-1.7b", label="Tiny-Jev-1.7B",
        repo="lostargon/Tiny-Jev-1.7B",
        params_b=1.7, precision="bf16", bytes_gb=3.4,
        devices=(Device.IGPU, Device.CPU),
        roles=("C2", "D2", "H1"),
        notes=(
            "CPU bf16 measured (decision oracle 2026-10-17, "
            "results/decision-oracle-20261017): 3.4 GiB peak RSS, 0.48 "
            "s/verdict, J-Bench test n=116: AUC 0.889, acc@0.5 87/116 "
            "(FA 25/87, FR 4/29) — the harness validator's wired default; "
            "second rung behind intern-decision-2b; organic harvest slice "
            "(224 items): test-66 AUC 0.628, 1/9 organic wrongs rejected"
        ),
    ),
    Candidate(
        id="ornith-1.5-9b", label="Ornith-1.5-9B",
        repo="ornith-ai/Ornith-1.5-9B",
        params_b=9.0, precision="bf16", bytes_gb=18.0,
        devices=(Device.DGPU1, Device.DGPU0),
        roles=("E3",),
    ),
    Candidate(
        id="ornith-1.0-9b", label="Ornith-1.0-9B",
        repo="ornith-ai/Ornith-1.0-9B",
        params_b=9.0, precision="bf16", bytes_gb=18.0,
        devices=(Device.DGPU1, Device.DGPU0),
        roles=("E3",),
    ),
    Candidate(
        id="flash-next", label="Qwen3.8-Flash-Next",
        repo="Qwen/Qwen3.8-Flash-Next",
        params_b=177.0, precision="Q2_0",
        bytes_gb=66.0,
        devices=(Device.DGPU0, Device.DGPU1),
        roles=("E4",),
        notes="GSQ-RCO Q2_0 (ISTA-DASLab); expert offload via the prism slot cache; ~40 GB active",
    ),
    Candidate(
        id="flash-next-strata", label="Qwen3.8-Flash-Next (Strata)",
        repo="Qwen/Qwen3.8-Flash-Next",
        file="Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        params_b=177.0, precision="Q2_0", bytes_gb=66.0,
        devices=(Device.DGPU0, Device.DGPU1),
        roles=("E4",),
        notes=(
            "endpoint-backed tier (stacks/strata-125b.json, "
            "http://127.0.0.1:8080/v1); measured 2026-10-02 on 2x RX 7900 XT: "
            "56.8-57.3 t/s decode, 407-523 t/s prefill, ~77% MTP acceptance; "
            "coexists with the desktop via --vram-reserve-mib 3072"
        ),
    ),
    Candidate(
        id="scion-35b-a3b", label="Scion-35B-A3B",
        repo="SkyIsNotGreen/Scion-35B-A3B",
        params_b=35.0, precision="PQ2_0", bytes_gb=11.3,
        devices=(Device.DGPU1, Device.DGPU0),
        roles=("E4",),
        notes="shipped ternary release + k=1 drafter sidecar",
    ),
)


class CandidateRegistry:
    """An id-indexed candidate set with role/device lookups."""

    def __init__(self, candidates: Iterable[Candidate] = ()) -> None:
        self._by_id: dict[str, Candidate] = {}
        for candidate in candidates:
            self.register(candidate)

    def register(self, candidate: Candidate) -> Candidate:
        if candidate.id in self._by_id:
            raise ValueError(f"duplicate candidate id {candidate.id!r}")
        self._by_id[candidate.id] = candidate
        return candidate

    def replace(self, candidate: Candidate) -> Candidate:
        """Register or overwrite (used when a measurement refines a row)."""
        self._by_id[candidate.id] = candidate
        return candidate

    def get(self, candidate_id: str) -> Candidate:
        try:
            return self._by_id[candidate_id]
        except KeyError:
            known = ", ".join(sorted(self._by_id))
            raise KeyError(
                f"unknown candidate {candidate_id!r} (known: {known})"
            ) from None

    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def for_role(self, role_id: str) -> tuple[Candidate, ...]:
        return tuple(c for c in self._by_id.values() if role_id in c.roles)

    def for_device(self, device: Device | str) -> tuple[Candidate, ...]:
        value = device.value if isinstance(device, Device) else str(device)
        return tuple(
            c for c in self._by_id.values() if any(d.value == value for d in c.devices)
        )

    def coverage(self, role_ids: Sequence[str]) -> dict[str, tuple[Candidate, ...]]:
        return {rid: self.for_role(rid) for rid in role_ids}

    def missing_roles(self, role_ids: Sequence[str]) -> tuple[str, ...]:
        """Required roles with no registered candidate, in input order."""
        return tuple(rid for rid in role_ids if not self.for_role(rid))

    def __contains__(self, candidate_id: object) -> bool:
        return candidate_id in self._by_id

    def __iter__(self):
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def to_dict(self) -> dict[str, Any]:
        return {"candidates": [c.to_dict() for c in self._by_id.values()]}


def default_registry() -> CandidateRegistry:
    """The catalog as a registry (brief candidates, no measurements)."""
    return CandidateRegistry(DEFAULT_CANDIDATES)


@dataclass(frozen=True)
class ResidentSet:
    """A coverage choice: candidates + what they leave uncovered.

    ``bytes_by_device`` is the planned resident weight footprint per device;
    ``unknown_sizes`` names candidates whose size was unmeasured, so a caller
    can tell "fits" from "we did not know".
    """

    candidates: tuple[Candidate, ...]
    uncovered: tuple[str, ...]
    bytes_by_device: Mapping[str, float]
    unknown_sizes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.uncovered

    def roles_covered(self) -> tuple[str, ...]:
        covered: list[str] = []
        for candidate in self.candidates:
            for role in candidate.roles:
                if role not in covered:
                    covered.append(role)
        return tuple(covered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.id for c in self.candidates],
            "roles_covered": list(self.roles_covered()),
            "uncovered": list(self.uncovered),
            "bytes_by_device": dict(self.bytes_by_device),
            "unknown_sizes": list(self.unknown_sizes),
            "ok": self.ok,
        }


def _capacity_map(capacity_gb: Mapping[Any, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for device, value in capacity_gb.items():
        key = device.value if isinstance(device, Device) else str(device)
        out[key] = float(value)
    return out


def select_resident_set(
    registry: CandidateRegistry,
    required_roles: Sequence[str],
    *,
    capacity_gb: Mapping[Any, float],
    prefer: Sequence[str] = (),
) -> ResidentSet:
    """Greedy minimal cover of ``required_roles`` within per-device capacity.

    The brief's consolidation objective.  Greedy is deliberate for the
    skeleton: it is deterministic, explainable, and the oracle's per-role
    measurements (not coverage count alone) are what will drive the real
    choice.  Candidates are considered by (uncovered roles served desc,
    preference order asc, declared bytes asc, id asc) and placed on the first
    of their preferred devices with room.

    Unknown sizes count as zero *and* are reported in ``unknown_sizes`` — a
    final placement decision must not read an unmeasured candidate as free.
    """
    for rid in required_roles:
        get_role(rid)  # raises KeyError on a typo'd role id

    capacities = _capacity_map(capacity_gb)
    used = {device: 0.0 for device in capacities}
    uncovered = list(dict.fromkeys(required_roles))  # de-dup, keep order
    chosen: list[Candidate] = []
    unknown: list[str] = []
    pref = {cid: index for index, cid in enumerate(prefer)}

    while uncovered:
        best: Optional[Candidate] = None
        best_key: Optional[tuple] = None
        for candidate in registry:
            gain = [rid for rid in candidate.roles if rid in uncovered]
            if not gain:
                continue
            size = 0.0 if candidate.bytes_gb is None else candidate.bytes_gb
            key = (
                -len(set(gain)),
                pref.get(candidate.id, len(pref)),
                size,
                candidate.id,
            )
            if best_key is None or key < best_key:
                best, best_key = candidate, key
        if best is None:
            break

        size = 0.0 if best.bytes_gb is None else best.bytes_gb
        placed_on: Optional[str] = None
        for device in best.devices:
            key = device.value
            if key in used and used[key] + size <= capacities[key] + 1e-9:
                placed_on = key
                break
        if placed_on is None:
            # No device has room: drop this candidate from consideration for
            # this pass and keep scanning.  Marking it via a used-set keeps
            # the loop terminating.
            registry = _without(registry, best.id)
            continue

        used[placed_on] += size
        chosen.append(best)
        if best.bytes_gb is None:
            unknown.append(best.id)
        for rid in best.roles:
            if rid in uncovered:
                uncovered.remove(rid)

    return ResidentSet(
        candidates=tuple(chosen),
        uncovered=tuple(uncovered),
        bytes_by_device=dict(used),
        unknown_sizes=tuple(unknown),
    )


def _without(registry: CandidateRegistry, candidate_id: str) -> CandidateRegistry:
    """A shallow copy of ``registry`` without one candidate (keeps the caller's
    registry untouched while the selection walks the space)."""
    clone = CandidateRegistry()
    for candidate in registry:
        if candidate.id != candidate_id:
            clone.register(candidate)
    return clone


def with_measurement(
    candidate: Candidate, *, bytes_gb: Optional[float] = None, precision: Optional[str] = None
) -> Candidate:
    """Return a copy of ``candidate`` with refined fields (oracle → registry)."""
    updates: dict[str, Any] = {}
    if bytes_gb is not None:
        updates["bytes_gb"] = float(bytes_gb)
    if precision is not None:
        updates["precision"] = precision
    return replace(candidate, **updates) if updates else candidate
