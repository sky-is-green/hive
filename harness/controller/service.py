"""Persist controller proposals, apply confirmed changes, and roll back.

The service is the only layer that writes settings.  It accepts injected
loaders/savers so a test can use a temporary settings file without starting the
Studio, and it records every proposal/apply/revert in the settings audit log.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from harness.controller.encoder import (
    EncoderError,
    EncoderLifecycle,
    TextEncoder,
    encoder_from_settings,
)
from harness.controller.features import text_features
from harness.controller.profile import HardwareProfiler
from harness.controller.recommend import (
    ModelRequest,
    ProposalError,
    model_request_from_path,
    propose,
    validate_proposal,
)
from harness.state_store import (
    AuditLog,
    JsonStore,
    SchemaMismatch,
    StatePaths,
)
from harness.studio_settings import load_settings, save_settings

SettingsLoader = Callable[[Path | None], dict[str, Any]]
SettingsSaver = Callable[[dict[str, Any], Path | None], dict[str, Any]]
SettingsRemover = Callable[[dict[str, list[str]], Path | None], dict[str, Any]]
EncoderFactory = Callable[[], TextEncoder]


E1_MIN_LABELLED = 200
E1_MIN_FEATURED = 100
_POSITIVE_OUTCOMES = frozenset({"accepted", "applied"})
_NEGATIVE_OUTCOMES = frozenset({"rejected", "reverted", "wrong", "abandoned"})


class ControllerError(RuntimeError):
    """A controller operation is not possible or is unsafe."""


class ConfirmationRequired(ControllerError):
    """A proposal may not be applied without an explicit confirmation."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _nested_get(settings: Mapping[str, Any], dotted: str) -> Any:
    value: Any = settings
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _nested_has(settings: Mapping[str, Any], dotted: str) -> bool:
    value: Any = settings
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False
        value = value[part]
    return True


def _default_remove(remove: dict[str, list[str]],
                    state_dir: Path | None) -> dict[str, Any]:
    return save_settings({}, state_dir, remove=remove)


def _group_changes(changes: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for dotted, value in changes.items():
        block, key = dotted.split(".", 1)
        grouped.setdefault(block, {})[key] = value
    return grouped


def _default_controller() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selected_profile_id": None,
        "last_apply": None,
        "last_revert": None,
    }


class ControllerService:
    def __init__(
        self,
        state: StatePaths | None = None,
        *,
        profiler: HardwareProfiler | None = None,
        settings_loader: SettingsLoader = load_settings,
        settings_saver: SettingsSaver = save_settings,
        settings_remover: SettingsRemover = _default_remove,
        encoder_factory: EncoderFactory | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.state = (state or StatePaths.default()).ensure()
        self.profiler = profiler or HardwareProfiler(self.state)
        self._load = settings_loader
        self._save = settings_saver
        self._remove = settings_remover
        self._encoder_factory = encoder_factory
        self._encoder_lifecycle: EncoderLifecycle | None = None
        self._encoder_model: str | None = None
        self._now = now or _now

    def _recommendations(self, profile_id: str) -> JsonStore:
        return JsonStore(
            self.state.recommendations_file(profile_id),
            default={"schema_version": 1, "profile_id": profile_id,
                     "proposals": []},
        )

    def _controller(self) -> JsonStore:
        return JsonStore(self.state.controller_file(), default=_default_controller())

    def _audit(self, event: str, **fields: Any) -> None:
        AuditLog(self.state.audit_file()).append({"event": event, **fields})

    def _trace(self, event: str, **fields: Any) -> None:
        """Append a proposal/outcome record; no raw request text is stored."""
        AuditLog(self.state.controller_trace_file()).append(
            {"schema_version": 1, "event": event, **fields})

    def _encoder_policy(self) -> tuple[bool, float, str]:
        block = (self._load(self.state.root).get("encoder") or {})
        block = block if isinstance(block, Mapping) else {}
        idle_unload = bool(block.get("controller_idle_unload", True))
        model = str(block.get("embedding_model") or "").strip()
        try:
            idle_seconds = float(block.get("controller_idle_seconds", 300))
        except (TypeError, ValueError):
            idle_seconds = 300.0
        return idle_unload, max(0.1, idle_seconds), model

    def _get_encoder(self) -> TextEncoder:
        idle_unload, idle_seconds, model = self._encoder_policy()
        if (self._encoder_lifecycle is not None
                and self._encoder_model != model):
            # A changed model must not keep serving vectors from the old one
            # until the idle timer happens to fire.
            self.unload_encoder()
            self._encoder_lifecycle = None
        self._encoder_model = model
        if self._encoder_lifecycle is None:
            if self._encoder_factory is not None:
                factory = self._encoder_factory
            else:
                def factory() -> TextEncoder:
                    return encoder_from_settings(
                        self._load(self.state.root), local_files_only=True)
            self._encoder_lifecycle = EncoderLifecycle(
                factory, idle_unload=idle_unload, idle_seconds=idle_seconds)
        else:
            self._encoder_lifecycle.configure(
                idle_unload=idle_unload, idle_seconds=idle_seconds)
        return self._encoder_lifecycle.get()

    def unload_encoder(self) -> bool:
        """Drop the controller's encoder now; return whether one was loaded."""
        return (self._encoder_lifecycle.unload()
                if self._encoder_lifecycle is not None else False)

    def _request_embedding(self, request: str) -> dict[str, Any]:
        try:
            encoder = self._get_encoder()
            vector = encoder.encode_one(request)
            self._encoder_lifecycle.touch()  # type: ignore[union-attr]
        except EncoderError as exc:
            raise ControllerError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - optional local model
            raise ControllerError(f"encoder failed: {exc}") from exc
        if not vector or any(not math.isfinite(float(value)) for value in vector):
            raise ControllerError("encoder returned an unusable vector")
        return {
            "model": str(getattr(encoder, "model", "") or ""),
            "dimensions": len(vector),
            "vector": [round(float(value), 6) for value in vector],
        }

    def record_outcome(self, proposal_id: str, outcome: str, *,
                       metrics: Mapping[str, Any] | None = None) -> None:
        """Attach a training/evaluation label to a proposal trace.

        Outcomes are append-only so the proposal record is never rewritten.
        ``accepted``/``rejected`` are human labels; ``applied``/``reverted``
        are lifecycle facts and are recorded automatically.
        """
        allowed = {"applied", "reverted", "accepted", "rejected", "wrong",
                   "unknown", "abandoned"}
        if outcome not in allowed:
            raise ControllerError(
                f"outcome must be one of {', '.join(sorted(allowed))}")
        self._trace("controller.outcome", proposal_id=proposal_id,
                    outcome=outcome, metrics=dict(metrics or {}))

    def trace_summary(self) -> dict[str, Any]:
        """Count the privacy-preserving traces and report E1 readiness.

        The head is not trained here.  This is the gate that stops a training
        run from treating an empty or unlabeled trace file as a dataset.
        """
        path = self.state.controller_trace_file()
        events: dict[str, int] = {}
        proposals: dict[str, dict[str, Any]] = {}
        outcomes: dict[str, int] = {}
        latest: dict[str, str] = {}
        bad_lines = 0
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except (TypeError, ValueError):
                    bad_lines += 1
                    continue
                if not isinstance(row, dict):
                    bad_lines += 1
                    continue
                event = str(row.get("event") or "")
                events[event] = events.get(event, 0) + 1
                proposal_id = str(row.get("proposal_id") or "")
                if event == "controller.proposal" and proposal_id:
                    proposals[proposal_id] = row
                elif event == "controller.outcome" and proposal_id:
                    outcome = str(row.get("outcome") or "")
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1
                    latest[proposal_id] = outcome
        labelled_ids = [pid for pid in proposals
                        if latest.get(pid) in (_POSITIVE_OUTCOMES
                                               | _NEGATIVE_OUTCOMES)]
        featured_ids = [pid for pid in labelled_ids
                        if proposals[pid].get("request_features")]
        positive = sum(1 for pid in labelled_ids
                       if latest.get(pid) in _POSITIVE_OUTCOMES)
        negative = len(labelled_ids) - positive
        missing: list[str] = []
        if len(labelled_ids) < E1_MIN_LABELLED:
            missing.append(
                f"{E1_MIN_LABELLED - len(labelled_ids)} more labelled proposals")
        if len(featured_ids) < E1_MIN_FEATURED:
            missing.append(
                f"{E1_MIN_FEATURED - len(featured_ids)} more featured labelled "
                "proposals")
        return {
            "trace_file": str(path),
            "events": events,
            "proposals": len(proposals),
            "outcomes": outcomes,
            "labelled": len(labelled_ids),
            "positive": positive,
            "negative": negative,
            "with_features": len(featured_ids),
            "with_embedding": sum(
                1 for row in proposals.values()
                if row.get("request_embedding")),
            "target_labelled": E1_MIN_LABELLED,
            "target_featured": E1_MIN_FEATURED,
            "ready": not missing,
            "missing": missing,
            "bad_lines": bad_lines,
        }

    def status(self) -> dict[str, Any]:
        profile = self.profiler.latest()
        proposals: list[dict[str, Any]] = []
        if profile:
            try:
                proposals = [
                    row for row in (
                        self._recommendations(profile["profile_id"]).read().get(
                            "proposals") or [])
                    if isinstance(row, dict)
                ]
            except SchemaMismatch:
                proposals = []
        latest = proposals[-1] if proposals else None
        try:
            controller = self._controller().read()
        except SchemaMismatch as exc:
            controller = {**_default_controller(), "error": str(exc)}
        if self._encoder_lifecycle is not None:
            encoder = self._encoder_lifecycle.status()
        else:
            idle_unload, idle_seconds, _model = self._encoder_policy()
            encoder = {"loaded": False, "idle_unload": idle_unload,
                       "idle_seconds": idle_seconds, "idle_remaining_s": None}
        return {
            "state_dir": str(self.state.root),
            "trace_file": str(self.state.controller_trace_file()),
            "profile": profile,
            "proposals": proposals,
            "latest_proposal": latest,
            "controller": controller,
            "encoder": encoder,
            "trace_summary": self.trace_summary(),
        }

    def propose(
        self,
        *,
        profile_id: str | None = None,
        model_path: str | Path | None = None,
        model_gib: float | None = None,
        native_ctx: int | None = None,
        desired_ctx: int | None = None,
        kv_type: str = "q8_0",
        request: str | None = None,
        embed_request: bool = False,
    ) -> dict[str, Any]:
        profile = self.profiler.get(profile_id) if profile_id else self.profiler.latest()
        if profile is None:
            raise ControllerError(
                "no hardware profile is saved; run `hive profile scan "
                "--consent hardware` first")
        if model_gib is not None and model_gib <= 0:
            raise ControllerError("model_gib must be > 0")
        if embed_request and not (request and request.strip()):
            raise ControllerError("embed_request requires non-empty request text")
        model: ModelRequest | None = None
        if model_path is not None:
            model = model_request_from_path(model_path, desired_ctx=desired_ctx)
            if model_gib is not None or native_ctx is not None:
                model = ModelRequest(
                    name=model.name,
                    weights_gib=float(model_gib) if model_gib is not None
                    else model.weights_gib,
                    native_ctx=int(native_ctx) if native_ctx is not None
                    else model.native_ctx,
                    desired_ctx=desired_ctx,
                    kv_type=str(kv_type or "q8_0"),
                    path=model.path,
                )
        elif model_gib is not None:
            if model_gib <= 0:
                raise ControllerError("model_gib must be > 0")
            model = ModelRequest(
                name="manual model",
                weights_gib=float(model_gib),
                native_ctx=int(native_ctx) if native_ctx is not None else None,
                desired_ctx=desired_ctx,
                kv_type=str(kv_type or "q8_0"),
            )
        features = text_features(request) if request and request.strip() else None
        try:
            proposal = propose(profile, model=model, request=features,
                               now=self._now())
        except (ProposalError, ValueError) as exc:
            raise ControllerError(str(exc)) from exc
        embedding = self._request_embedding(request) if embed_request else None
        store = self._recommendations(profile["profile_id"])

        def add(current: dict[str, Any]) -> dict[str, Any]:
            proposals = [row for row in (current.get("proposals") or [])
                         if isinstance(row, dict) and row.get("id") != proposal["id"]]
            proposals.append(proposal)
            return {
                "schema_version": 1,
                "profile_id": profile["profile_id"],
                "proposals": proposals[-50:],
            }

        store.update(add)
        self._audit(
            "controller.propose",
            proposal_id=proposal["id"],
            profile_id=profile["profile_id"],
            changes=proposal["changes"],
            confidence=proposal["confidence"],
        )
        trace_fields: dict[str, Any] = {
            "proposal_id": proposal["id"],
            "profile_id": profile["profile_id"],
            "request_features": proposal.get("request_features"),
            "changes": proposal["changes"],
            "confidence": proposal["confidence"],
            "model": proposal.get("model"),
        }
        if embedding is not None:
            trace_fields["request_embedding"] = embedding
        self._trace("controller.proposal", **trace_fields)
        return proposal

    def _find(self, proposal_id: str,
              profile_id: str | None = None) -> tuple[str, dict[str, Any]]:
        if profile_id:
            candidates = [profile_id]
        else:
            candidates = [
                path.stem[len("recommendations-"):]
                for path in sorted(self.state.profiles_dir.glob(
                    "recommendations-*.json"))
            ]
        for candidate in reversed(candidates):
            try:
                rows = self._recommendations(candidate).read().get("proposals") or []
            except SchemaMismatch:
                continue
            for row in reversed(rows):
                if isinstance(row, dict) and row.get("id") == proposal_id:
                    return candidate, row
        raise ControllerError(f"no saved recommendation {proposal_id!r}")

    def get_proposal(self, proposal_id: str,
                     profile_id: str | None = None) -> dict[str, Any]:
        return self._find(proposal_id, profile_id)[1]

    def apply(self, proposal_id: str, *,
              confirmed: bool = False,
              profile_id: str | None = None) -> dict[str, Any]:
        if not confirmed:
            raise ConfirmationRequired(
                "controller changes require explicit confirmation")
        found_profile_id, raw = self._find(proposal_id, profile_id)
        profile = self.profiler.get(found_profile_id)
        if profile is None:
            raise ControllerError(
                f"the proposal's hardware profile {found_profile_id!r} is gone")
        try:
            proposal = validate_proposal(raw, profile)
        except ProposalError as exc:
            raise ControllerError(str(exc)) from exc

        current = self._load(self.state.root)
        previous = {
            key: _nested_get(current, key)
            for key in proposal["changes"]
        }
        previous_present = {
            key: _nested_has(current, key)
            for key in proposal["changes"]
        }
        grouped = _group_changes(proposal["changes"])
        updated = self._save(grouped, self.state.root)
        stamp = self._now().astimezone(timezone.utc).isoformat(
            timespec="seconds").replace("+00:00", "Z")
        controller = self._controller().update(lambda state: {
            **state,
            "selected_profile_id": found_profile_id,
            "last_apply": {
                "proposal_id": proposal_id,
                "profile_id": found_profile_id,
                "applied_at": stamp,
                "changes": dict(proposal["changes"]),
                "previous": previous,
                "previous_present": previous_present,
            },
            "last_revert": None,
        })
        self._audit(
            "controller.apply",
            proposal_id=proposal_id,
            profile_id=found_profile_id,
            changes=proposal["changes"],
            previous=previous,
        )
        self.record_outcome(proposal_id, "applied",
                            metrics={"changes": proposal["changes"]})
        return {
            "ok": True,
            "proposal": proposal,
            "settings": updated,
            "previous": previous,
            "controller": controller,
            "current_before": current,
        }

    def revert(self, *, confirmed: bool = False) -> dict[str, Any]:
        if not confirmed:
            raise ConfirmationRequired("controller rollback requires confirmation")
        controller = self._controller().read()
        last = controller.get("last_apply")
        if not isinstance(last, Mapping) or not last.get("previous"):
            raise ControllerError("there is no applied controller change to revert")
        previous = dict(last["previous"])
        present = last.get("previous_present")
        if not isinstance(present, Mapping):
            # Older controller files predate the presence map: restore every
            # recorded value, which is the safe interpretation for a value.
            present = {key: True for key in previous}
        restore = {key: value for key, value in previous.items()
                   if present.get(key)}
        absent: dict[str, list[str]] = {}
        for key, was_present in present.items():
            if was_present:
                continue
            block, name = str(key).split(".", 1)
            absent.setdefault(block, []).append(name)
        if restore:
            updated = self._save(_group_changes(restore), self.state.root)
        else:
            updated = self._load(self.state.root)
        if absent:
            updated = self._remove(absent, self.state.root)
        stamp = self._now().astimezone(timezone.utc).isoformat(
            timespec="seconds").replace("+00:00", "Z")
        controller = self._controller().update(lambda state: {
            **state,
            "last_apply": None,
            "last_revert": {
                "from_proposal_id": last.get("proposal_id"),
                "reverted_at": stamp,
                "changes": previous,
                "removed": absent,
            },
        })
        self._audit(
            "controller.revert",
            proposal_id=last.get("proposal_id"),
            changes=previous,
        )
        if last.get("proposal_id"):
            self.record_outcome(str(last["proposal_id"]), "reverted",
                                metrics={"changes": previous})
        return {"ok": True, "settings": updated, "controller": controller}
