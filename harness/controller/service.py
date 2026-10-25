"""Persist controller proposals, apply confirmed changes, and roll back.

The service is the only layer that writes settings.  It accepts injected
loaders/savers so a test can use a temporary settings file without starting the
Studio, and it records every proposal/apply/revert in the settings audit log.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

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
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.state = (state or StatePaths.default()).ensure()
        self.profiler = profiler or HardwareProfiler(self.state)
        self._load = settings_loader
        self._save = settings_saver
        self._remove = settings_remover
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
        return {
            "state_dir": str(self.state.root),
            "profile": profile,
            "proposals": proposals,
            "latest_proposal": latest,
            "controller": controller,
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
    ) -> dict[str, Any]:
        profile = self.profiler.get(profile_id) if profile_id else self.profiler.latest()
        if profile is None:
            raise ControllerError(
                "no hardware profile is saved; run `hive profile scan "
                "--consent hardware` first")
        if model_gib is not None and model_gib <= 0:
            raise ControllerError("model_gib must be > 0")
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
        try:
            proposal = propose(profile, model=model, now=self._now())
        except (ProposalError, ValueError) as exc:
            raise ControllerError(str(exc)) from exc
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
        return {"ok": True, "settings": updated, "controller": controller}
