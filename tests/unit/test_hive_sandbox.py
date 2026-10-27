"""The subagent bwrap runner keeps a scratch workspace and masks the host."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "deploy" / "hive-sandbox" / "hive_bwrap.py"


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("hive_bwrap", WRAPPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hive_bwrap = _load_wrapper()


def _profile(workspace: Path) -> list[str]:
    return [
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--unshare-pid",
        "--proc", "/proc",
        "--die-with-parent",
        "--tmpfs", "/tmp",
        "--bind", str(workspace), str(workspace),
    ]


def _pairs(argv: list[str]) -> list[tuple[str, str]]:
    return list(zip(argv, argv[1:]))


def test_split_profile_uses_the_provider_separator():
    profile, command = hive_bwrap.split_profile(
        ["--ro-bind", "/", "/", "--", "bash", "-c", "true"])
    assert profile == ["--ro-bind", "/", "/"]
    assert command == ["bash", "-c", "true"]


def test_missing_separator_is_a_runner_failure():
    with pytest.raises(hive_bwrap.ProfileError, match="separator"):
        hive_bwrap.split_profile(["--ro-bind", "/", "/"])


def test_workspace_must_be_under_the_root(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(hive_bwrap.ProfileError, match="workspace bind"):
        hive_bwrap.build_bwrap_argv(
            _profile(outside), ["bash", "-c", "true"],
            workspace_root=str(root), mask_dirs=())


def test_the_root_itself_is_not_a_workspace(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    with pytest.raises(hive_bwrap.ProfileError, match="workspace bind"):
        hive_bwrap.workspace_bind(_profile(root), str(root))


def test_a_dotdot_workspace_is_refused(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    escaping = root / ".." / "outside"
    with pytest.raises(hive_bwrap.ProfileError, match="not normalized"):
        hive_bwrap.workspace_bind(_profile(escaping), str(root))


def test_namespace_flags_masks_and_rebind_are_ordered(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    workspace = root / "task-1"
    workspace.mkdir()
    secret = tmp_path / "secret"
    secret.mkdir()
    missing = tmp_path / "missing"

    argv = hive_bwrap.build_bwrap_argv(
        _profile(workspace), ["bash", "-c", "true"],
        workspace_root=str(root), mask_dirs=(str(secret), str(missing)))
    pairs = _pairs(argv)

    assert ("--unshare-user", "--disable-userns") in pairs
    assert ("--unshare-net", "--unshare-ipc") in pairs
    assert ("--unshare-uts", "--new-session") in pairs
    assert ("--hostname", "hive-subagent") in pairs
    assert ("--cap-drop", "ALL") in pairs
    assert ("--tmpfs", str(secret)) in pairs
    assert ("--tmpfs", str(missing)) not in pairs

    root_at = pairs.index(("--tmpfs", str(root)))
    bind_at = max(
        index for index, pair in enumerate(pairs)
        if pair == ("--bind", str(workspace)))
    dir_at = pairs.index(("--dir", str(workspace)))
    assert root_at < dir_at < bind_at
    assert argv[-4:] == ["--", "bash", "-c", "true"]


def test_nested_workspace_creates_each_parent(tmp_path):
    root = tmp_path / "workspaces"
    workspace = root / "conversation" / "task"
    workspace.mkdir(parents=True)
    argv = hive_bwrap.build_bwrap_argv(
        _profile(workspace), ["bash"],
        workspace_root=str(root), mask_dirs=())
    pairs = _pairs(argv)
    assert ("--dir", str(root / "conversation")) in pairs
    assert ("--dir", str(workspace)) in pairs


def test_empty_command_is_refused(tmp_path):
    root = tmp_path / "workspaces"
    workspace = root / "task"
    workspace.mkdir(parents=True)
    with pytest.raises(hive_bwrap.ProfileError, match="empty command"):
        hive_bwrap.build_bwrap_argv(
            _profile(workspace), [], workspace_root=str(root), mask_dirs=())


def test_main_reports_a_missing_profile(capsys):
    assert hive_bwrap.main([]) == 125
    assert "hive-bwrap: missing profile separator" in capsys.readouterr().err
