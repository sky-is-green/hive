"""Integration: StackManager apply / status / unload (HIVE-PLAN §B4 T37).

No GGUFs exist on this box and no llama-server binary is shipped, so the
manager is exercised against the *real* ``harness.models.LlamaServerManager``
with an injected fake spawner (a ``subprocess.Popen`` stand-in) and a fake
prober: ``load()`` runs end to end, registers a ``ServerInstance`` per tier,
and the manager is the only thing under test.  The real apply gate is T45.

Residency planning is T36's module (still a skeleton in this worktree), so
``plan_residency`` is monkeypatched here — the manager's contract with it is
the fake plan's ``ok`` / ``per_card`` / ``warnings`` shape.
"""

from __future__ import annotations

import pytest

import harness.models as models_mod
import harness.stack.endpoint as endpoint_mod
import harness.stack.manager as manager_mod
from harness.models import LlamaServerManager
from harness.stack.manager import (
    StackManager,
    TierRuntime,
    tier_env,
    tier_extra_args,
    tier_load_options,
)
from harness.stack.schema import Stack, Tier


# --------------------------------------------------------------------------
# fakes: the injected spawner + a stub residency plan
# --------------------------------------------------------------------------

class _FakeProc:
    """The minimum ``subprocess.Popen`` surface ``LlamaServerManager`` uses."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._alive = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):  # noqa: ANN001 - Popen-compatible
        self._alive = False
        return 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False


class _Spawner:
    """Records every spawn (argv + kwargs) and returns live fake processes."""

    def __init__(self) -> None:
        self.cmds: list[list[str]] = []
        self.kwargs: list[dict] = []
        self.procs: list[_FakeProc] = []

    def __call__(self, cmd, **kwargs):  # noqa: ANN001 - Popen-compatible
        self.cmds.append([str(a) for a in cmd])
        self.kwargs.append(dict(kwargs))
        proc = _FakeProc(1000 + len(self.procs))
        self.procs.append(proc)
        return proc


class _Plan:
    """Stand-in for ``residency.ResidencyPlan`` (T36)."""

    def __init__(self, *, ok: bool = True, per_card=None, warnings=None) -> None:
        self.ok = ok
        self.per_card = list(per_card or [])
        self.warnings = list(warnings or [])


_CARD = {"card": 0, "weights": 8.0, "kv": 2.0, "total": 10.0, "budget": 15.0}


def _make_manager(tmp_path, monkeypatch, port: int = 18137):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    spawner = _Spawner()
    manager = LlamaServerManager(
        binary=binary,
        models_dir=tmp_path / "models",
        port=port,
        log_dir=tmp_path / "logs",
        spawner=spawner,
        prober=lambda base_url: "fake-model",
        startup_timeout=5.0,
        memory_guard=False,
    )
    # Ports are allocated by the StackManager; nothing actually binds, so make
    # the liveness probe deterministic instead of depending on the host.
    monkeypatch.setattr(models_mod, "_port_in_use", lambda *a, **k: False)
    return manager, spawner


def _install_plan(monkeypatch, plan: _Plan) -> None:
    monkeypatch.setattr(manager_mod, "plan_residency", lambda stack, **kw: plan)


def _tier(role: str, **overrides) -> Tier:
    data = {
        "role": role,
        "repo": f"mock/{role}",
        "file": f"{role}.gguf",
        "ctx": 8192,
        "ngl": 99,
        "backend": "vulkan",
        "cache_k": "q8_0",
        "cache_v": "q8_0",
    }
    data.update(overrides)
    return Tier(**data)


def _two_tier_stack() -> Stack:
    """The §B3 ``peer-2tier`` shape: a face with MTP + split + pin, a worker."""
    return Stack(
        name="peer-2tier",
        tiers=[
            _tier("face", ctx=262144, mmproj="mmproj-F16.gguf",
                  pin="HIP_VISIBLE_DEVICES=0,1", ts="1,1",
                  spec={"type": "draft-mtp", "n_max": 3}),
            _tier("worker", ctx=131072, pin="HIP_VISIBLE_DEVICES=1"),
        ],
    )


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------

def test_apply_registers_two_instances_on_distinct_ports(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan(per_card=[_CARD]))
    stack = _two_tier_stack()

    manager_api = StackManager(manager, stacks_root=tmp_path / "stacks")
    result = manager_api.apply(stack)

    assert result["ok"] is True
    assert [t["role"] for t in result["tiers"]] == ["face", "worker"]
    ports = [t["port"] for t in result["tiers"]]
    assert ports[0] != ports[1]
    assert len({t["key"] for t in result["tiers"]}) == 2
    # two live ServerInstances registered by load()
    assert len(manager._instances) == 2
    assert sorted(i.port for i in manager._instances.values()) == sorted(ports)


def test_status_reports_the_plans_per_tier_figures(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan(per_card=[_CARD],
                                     warnings=["display budget estimated"]))
    manager_api = StackManager(manager)
    manager_api.apply(_two_tier_stack())

    status = manager_api.status()
    assert status["ok"] is True
    assert status["stack"] == "peer-2tier"
    assert status["warnings"] == ["display budget estimated"]
    rows = {row["role"]: row for row in status["tiers"]}
    assert set(rows) == {"face", "worker"}
    face, worker = rows["face"], rows["worker"]
    assert (face["ctx"], worker["ctx"]) == (262144, 131072)
    assert face["model"] == "fake-model" == worker["model"]
    assert face["backend"] == "vulkan" and face["resident"] is True
    assert face["per_card"] == [_CARD]
    assert face["port"] != worker["port"]
    assert face["vram_gb"] is None and face["tok_s"] is None
    assert face["key"].endswith("peer-2tier-face")


def test_apply_passes_the_tier_launch_config_to_load(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan(per_card=[_CARD]))
    StackManager(manager).apply(_two_tier_stack())

    face_cmd, worker_cmd = spawner.cmds
    assert face_cmd[face_cmd.index("-c") + 1] == "262144"
    assert face_cmd[face_cmd.index("-ngl") + 1] == "99"
    assert "--cache-type-k" in face_cmd and "q8_0" in face_cmd
    assert "--mmproj" in face_cmd and "mmproj-F16.gguf" in face_cmd
    assert ["--split-mode", "layer", "--tensor-split", "1,1"] == \
        face_cmd[face_cmd.index("--split-mode"):face_cmd.index("--tensor-split") + 2]
    assert ["--spec-type", "draft-mtp", "--draft-max", "3"] == \
        face_cmd[face_cmd.index("--spec-type"):face_cmd.index("--draft-max") + 2]
    # per-tier env: face pinned to both cards, worker to one
    assert spawner.kwargs[0]["env"]["HIP_VISIBLE_DEVICES"] == "0,1"
    assert spawner.kwargs[1]["env"]["HIP_VISIBLE_DEVICES"] == "1"
    # the worker keeps its own ctx and carries none of the face's MTP flags
    assert worker_cmd[worker_cmd.index("-c") + 1] == "131072"
    assert "--split-mode" not in worker_cmd and "--spec-type" not in worker_cmd


def test_apply_is_face_first_even_when_the_stack_leads_with_a_worker(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    stack = Stack(name="rev", tiers=[_tier("worker", ctx=4096),
                                    _tier("face", ctx=8192)])
    result = StackManager(manager).apply(stack)
    assert [t["role"] for t in result["tiers"]] == ["face", "worker"]
    assert spawner.cmds[0][spawner.cmds[0].index("-c") + 1] == "8192"
    assert spawner.cmds[1][spawner.cmds[1].index("-c") + 1] == "4096"


def test_apply_refuses_over_budget_before_spawning(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan(ok=False, warnings=["face: 20.00 > 16.00"]))
    with pytest.raises(RuntimeError, match="refused"):
        StackManager(manager).apply(_two_tier_stack())
    assert spawner.cmds == []
    assert manager._instances == {}


def test_apply_rolls_back_when_a_later_tier_fails(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    original = manager.load

    def flaky(*args, **kwargs):
        if str(kwargs.get("key", "")).endswith("-worker"):
            raise RuntimeError("no VRAM for the worker")
        return original(*args, **kwargs)

    monkeypatch.setattr(manager, "load", flaky)
    manager_api = StackManager(manager)
    with pytest.raises(RuntimeError, match="tier 'worker'"):
        manager_api.apply(_two_tier_stack())

    assert manager._instances == {}
    assert spawner.procs[0].terminated is True
    assert manager_api.status()["tiers"] == []


# --------------------------------------------------------------------------
# endpoint-backed tiers (Strata: verify + register, never spawn/kill)
# --------------------------------------------------------------------------

_STRATA_HEALTH = {
    "endpoint": "http://127.0.0.1:8080/v1",
    "base_url": "http://127.0.0.1:8080/v1",
    "health_url": "http://127.0.0.1:8080/health",
    "model": "qwen3.8-flash-next-q2_0",
    "loaded": True,
    "max_context": 4096,
    "service": "strata",
}

_CAPS = {"decode_tok_s": [56.8, 57.3], "prefill_tok_s": [407, 523]}


def _endpoint_tier(role: str = "face", **overrides) -> Tier:
    data = {
        "role": role, "repo": "Qwen/Qwen3.8-Flash-Next",
        "file": "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        "ctx": 4096, "backend": "strata",
        "endpoint": "http://127.0.0.1:8080/v1", "caps": dict(_CAPS),
    }
    data.update(overrides)
    return Tier(**data)


def _launch_tier(role: str = "face", **overrides) -> Tier:
    """An endpoint tier with the managed Strata lifecycle attached."""
    tier = _endpoint_tier(role=role, **overrides)
    tier.launch = {"kind": "strata", "repo": "/tmp/strata",
                   "config": "strata-q2_0-coexist.json"}
    return tier


class _FakeLauncher:
    """Records managed start/stop without spawning anything."""

    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []
        self.stopped: list[str] = []

    def start(self, *, stack_name, role, tier, endpoint):
        self.started.append((stack_name, role))
        return {
            "pid": 5150, "config": f"/state/{role}.json",
            "log": "/logs/strata_8080.log", "engine_log": "/logs/engine.log",
            "cmd": [],
            "health": dict(_STRATA_HEALTH, endpoint=endpoint, base_url=endpoint),
        }

    def stop(self, role):
        self.stopped.append(role)
        return True


class _FakeResponse:
    """The ``requests.Response`` surface ``probe_endpoint`` reads."""

    def __init__(self, status_code: int, payload) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = b"x"

    def json(self):
        return self._payload


def test_apply_endpoint_tier_registers_without_spawning(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    probed: list[str] = []

    def prober(endpoint: str) -> dict:
        probed.append(endpoint)
        return dict(_STRATA_HEALTH)

    stack_api = StackManager(manager, endpoint_prober=prober)
    result = stack_api.apply(Stack(name="strata-125b", tiers=[_endpoint_tier()]))

    assert result == {"ok": True, "tiers": [
        {"role": "face", "port": 8080, "key": "strata-125b-face"}]}
    assert probed == ["http://127.0.0.1:8080/v1"]
    assert spawner.cmds == [] and manager._instances == {}

    status = stack_api.status()
    assert status["stack"] == "strata-125b"
    row = status["tiers"][0]
    assert row["base_url"] == "http://127.0.0.1:8080/v1"
    assert row["caps"] == _CAPS
    assert (row["port"], row["ctx"], row["backend"]) == (8080, 4096, "strata")
    assert row["model"] == "qwen3.8-flash-next-q2_0"
    assert row["resident"] is True
    # no ServerInstance is invented for an endpoint we do not own
    assert stack_api.instance_for("face") is None


def test_apply_endpoint_reports_unreachable_and_registers_nothing(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())

    def prober(endpoint: str) -> dict:
        raise RuntimeError("endpoint http://127.0.0.1:8080/v1 is not healthy: refused")

    stack_api = StackManager(manager, endpoint_prober=prober)
    with pytest.raises(RuntimeError, match="failed on tier 'face'"):
        stack_api.apply(Stack(name="strata-125b", tiers=[_endpoint_tier()]))

    assert spawner.cmds == []
    assert stack_api.status() == {"ok": True, "stack": None, "tiers": [],
                                  "warnings": []}


def test_unload_endpoint_tier_deregisters_without_killing(tmp_path, monkeypatch):
    manager, _spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    stopped: list[str] = []
    monkeypatch.setattr(manager, "unload", lambda key: stopped.append(key))

    stack_api = StackManager(manager, endpoint_prober=lambda ep: dict(_STRATA_HEALTH))
    stack_api.apply(Stack(name="strata-125b", tiers=[_endpoint_tier()]))

    assert stack_api.unload() == {"ok": True, "unloaded": ["face"]}
    assert stopped == []  # the endpoint process is not ours to stop
    assert stack_api.status()["tiers"] == []
    assert stack_api.unload() == {"ok": True, "unloaded": []}


def test_mixed_stack_registers_the_endpoint_and_spawns_only_the_local_tier(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan(per_card=[_CARD]))
    # the endpoint owns the manager's base port; the worker must skip it
    endpoint = "http://127.0.0.1:18137/v1"
    stack = Stack(name="mixed", tiers=[
        _endpoint_tier(endpoint=endpoint, caps=None),
        _tier("worker", ctx=131072),
    ])
    stack_api = StackManager(
        manager,
        endpoint_prober=lambda ep: dict(_STRATA_HEALTH, endpoint=ep, base_url=ep),
    )
    result = stack_api.apply(stack)

    assert [t["role"] for t in result["tiers"]] == ["face", "worker"]
    face, worker = result["tiers"]
    assert face["port"] == 18137
    assert worker["port"] == 18138
    assert len(spawner.cmds) == 1
    assert worker["key"].endswith("mixed-worker")
    assert stack_api.status()["tiers"][0]["caps"] is None

    assert stack_api.unload() == {"ok": True, "unloaded": ["face", "worker"]}
    assert spawner.procs[0].terminated is True


def test_endpoint_registration_rolls_back_when_a_later_tier_fails(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())

    def flaky(*args, **kwargs):
        raise RuntimeError("no VRAM for the worker")

    monkeypatch.setattr(manager, "load", flaky)
    stack_api = StackManager(manager, endpoint_prober=lambda ep: dict(_STRATA_HEALTH))
    with pytest.raises(RuntimeError, match="tier 'worker'"):
        stack_api.apply(Stack(name="mixed", tiers=[
            _endpoint_tier(), _tier("worker")]))

    assert spawner.cmds == []
    assert stack_api.status()["tiers"] == []


def test_probe_endpoint_reads_strata_health(monkeypatch):
    calls: list[str] = []

    def fake_get(url: str, timeout: float):
        calls.append(url)
        return _FakeResponse(200, {
            "status": "ok", "loaded": True, "model": "strata",
            "max_context": 4096, "service": "strata"})

    monkeypatch.setattr(endpoint_mod.requests, "get", fake_get)
    info = endpoint_mod.probe_endpoint("http://127.0.0.1:8080/v1")

    assert calls == ["http://127.0.0.1:8080/health"]
    assert info["model"] == "strata"
    assert info["max_context"] == 4096
    assert info["loaded"] is True
    assert info["base_url"] == "http://127.0.0.1:8080/v1"


def test_probe_endpoint_refuses_an_unloaded_model(monkeypatch):
    monkeypatch.setattr(
        endpoint_mod.requests, "get",
        lambda url, timeout: _FakeResponse(200, {"status": "ok", "loaded": False}))
    with pytest.raises(RuntimeError, match="not loaded"):
        endpoint_mod.probe_endpoint("http://127.0.0.1:8080/v1")


def test_probe_endpoint_falls_back_to_models(monkeypatch):
    def fake_get(url: str, timeout: float):
        if url.endswith("/health"):
            return _FakeResponse(404, {})
        return _FakeResponse(200, {"data": [{"id": "llama-35b"}]})

    monkeypatch.setattr(endpoint_mod.requests, "get", fake_get)
    info = endpoint_mod.probe_endpoint("http://127.0.0.1:9000")

    assert info["model"] == "llama-35b"
    assert info["base_url"] == "http://127.0.0.1:9000"


def test_probe_endpoint_reports_every_failure(monkeypatch):
    def fake_get(url: str, timeout: float):
        raise endpoint_mod.requests.ConnectionError("connection refused")

    monkeypatch.setattr(endpoint_mod.requests, "get", fake_get)
    with pytest.raises(RuntimeError, match="is not healthy"):
        endpoint_mod.probe_endpoint("http://127.0.0.1:8080/v1")


# --------------------------------------------------------------------------
# managed lifecycle (`launch`): adopt when up, start when down, stop own
# --------------------------------------------------------------------------

def test_apply_starts_a_launch_tier_when_the_endpoint_is_down(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    launcher = _FakeLauncher()

    def prober(endpoint: str) -> dict:
        raise RuntimeError(f"endpoint {endpoint} is not healthy: refused")

    stack_api = StackManager(manager, endpoint_prober=prober,
                             strata_launcher=launcher)
    result = stack_api.apply(Stack(name="strata-125b", tiers=[_launch_tier()]))

    assert result["ok"] is True
    assert launcher.started == [("strata-125b", "face")]
    row = stack_api.status()["tiers"][0]
    assert row["managed"] is True and row["pid"] == 5150
    assert row["resident"] is True
    assert row["model"] == "qwen3.8-flash-next-q2_0"

    assert stack_api.unload() == {"ok": True, "unloaded": ["face"]}
    assert launcher.stopped == ["face"]
    assert spawner.cmds == []  # never a llama-server spawn


def test_apply_adopts_a_running_launch_tier_without_owning_it(tmp_path, monkeypatch):
    manager, _spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    launcher = _FakeLauncher()
    stack_api = StackManager(
        manager, endpoint_prober=lambda ep: dict(_STRATA_HEALTH),
        strata_launcher=launcher,
    )
    stack_api.apply(Stack(name="strata-125b", tiers=[_launch_tier()]))

    assert launcher.started == []
    row = stack_api.status()["tiers"][0]
    assert row["managed"] is False and row["pid"] is None
    assert stack_api.unload() == {"ok": True, "unloaded": ["face"]}
    assert launcher.stopped == []  # adopted: the process is not ours


def test_launch_rollback_stops_the_process_it_started(tmp_path, monkeypatch):
    manager, _spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    launcher = _FakeLauncher()

    def prober(endpoint: str) -> dict:
        raise RuntimeError("down")

    def flaky(*args, **kwargs):
        raise RuntimeError("no VRAM for the worker")

    monkeypatch.setattr(manager, "load", flaky)
    stack_api = StackManager(manager, endpoint_prober=prober,
                             strata_launcher=launcher)
    with pytest.raises(RuntimeError, match="tier 'worker'"):
        stack_api.apply(Stack(name="mixed", tiers=[_launch_tier(), _tier("worker")]))

    assert launcher.started == [("mixed", "face")]
    assert launcher.stopped == ["face"]
    assert stack_api.status()["tiers"] == []


# --------------------------------------------------------------------------
# unload
# --------------------------------------------------------------------------

def test_unload_stops_both_tiers(tmp_path, monkeypatch):
    manager, spawner = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    manager_api = StackManager(manager)
    manager_api.apply(_two_tier_stack())

    # a name that is not the applied stack is a no-op
    assert manager_api.unload("someone-else") == {"ok": True, "unloaded": []}
    assert len(manager._instances) == 2

    result = manager_api.unload()
    assert result == {"ok": True, "unloaded": ["face", "worker"]}
    assert manager._instances == {}
    assert all(proc.terminated for proc in spawner.procs)
    assert manager_api.status() == {"ok": True, "stack": None,
                                    "tiers": [], "warnings": []}
    # a second unload is a clean no-op
    assert manager_api.unload() == {"ok": True, "unloaded": []}


def test_status_before_apply_is_empty(tmp_path, monkeypatch):
    manager, _ = _make_manager(tmp_path, monkeypatch)
    status = StackManager(manager).status()
    assert status["stack"] is None
    assert status["tiers"] == []


def test_instance_for_returns_the_server_instance(tmp_path, monkeypatch):
    manager, _ = _make_manager(tmp_path, monkeypatch)
    _install_plan(monkeypatch, _Plan())
    manager_api = StackManager(manager)
    manager_api.apply(_two_tier_stack())

    instance = manager_api.instance_for("face")
    assert instance is not None
    assert instance.key.endswith("peer-2tier-face")
    assert instance.port == manager_api.status()["tiers"][0]["port"]
    assert manager_api.instance_for("missing") is None


# --------------------------------------------------------------------------
# tier config helpers (the EngineProfile.load_options seam, ADR-L4)
# --------------------------------------------------------------------------

def test_tier_load_options_maps_the_b3_fields():
    tier = _tier("face", ctx=262144, ngl=99, mmproj="mmproj-F16.gguf")
    assert tier_load_options(tier) == {
        "context": 262144,
        "gpu_layers": 99,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "mmproj": "mmproj-F16.gguf",
    }


def test_tier_load_options_accepts_the_mapping_form():
    # T43's stack_summary passes face_tier()'s plain dict, not the dataclass.
    face = {"role": "face", "ctx": 8192, "ngl": 99,
            "cache_k": "q8_0", "cache_v": "q8_0"}
    assert tier_load_options(face)["context"] == 8192
    assert tier_load_options(face)["gpu_layers"] == 99


def test_tier_env_parses_the_pin_assignment():
    assert tier_env(_tier("face", pin="HIP_VISIBLE_DEVICES=0,1")) == {
        "HIP_VISIBLE_DEVICES": "0,1"}
    assert tier_env(_tier("face", pin="1")) == {"HIP_VISIBLE_DEVICES": "1"}
    assert tier_env(_tier("face")) == {}


def test_tier_extra_args_emits_jinja_split_and_spec_flags():
    tier = _tier("face", ts="1,1", spec={"type": "draft-mtp", "n_max": 3})
    assert tier_extra_args(tier) == [
        "--jinja",
        "--split-mode", "layer", "--tensor-split", "1,1",
        "--spec-type", "draft-mtp", "--draft-max", "3",
    ]
    # a plain worker still carries --jinja: no tool template without it
    assert tier_extra_args(_tier("worker")) == ["--jinja"]


def test_tier_extra_args_appends_engine_args_verbatim():
    tier = _tier("face", engine_args=["--lazy-mode", "on", "-np", "1"])
    assert tier_extra_args(tier)[-4:] == ["--lazy-mode", "on", "-np", "1"]
    plain = tier_extra_args(_tier("face"))
    assert "--lazy-mode" not in plain and "on" not in plain


def test_tier_runtime_to_dict_is_the_status_row():
    runtime = TierRuntime(role="face", key="k", port=1234, ctx=8192,
                          model="m", per_card=[_CARD])
    assert runtime.to_dict() == {
        "role": "face", "key": "k", "port": 1234, "ctx": 8192, "model": "m",
        "backend": "vulkan", "resident": True, "vram_gb": None, "tok_s": None,
        "base_url": None, "caps": None, "managed": False, "pid": None,
        "per_card": [_CARD],
    }
