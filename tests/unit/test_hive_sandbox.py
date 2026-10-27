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


def _profile(policy_root: Path) -> list[str]:
    """The provider profile: the policy bind is the parent's permission root."""
    return [
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--unshare-pid",
        "--proc", "/proc",
        "--die-with-parent",
        "--tmpfs", "/tmp",
        "--bind", str(policy_root), str(policy_root),
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


def test_the_policy_bind_must_be_present(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    workspace = root / "task"
    workspace.mkdir()
    profile = _profile(root)[:-3]
    with pytest.raises(hive_bwrap.ProfileError, match="workspace bind"):
        hive_bwrap.build_bwrap_argv(
            profile, ["bash"], workspace_root=str(root), mask_dirs=(),
            cwd=str(workspace))


def test_the_command_cwd_must_be_under_the_root(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(hive_bwrap.ProfileError, match="outside"):
        hive_bwrap.build_bwrap_argv(
            _profile(root), ["bash", "-c", "true"],
            workspace_root=str(root), mask_dirs=(), cwd=str(outside))


def test_the_root_itself_is_not_a_workspace(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    with pytest.raises(hive_bwrap.ProfileError, match="subdirectory"):
        hive_bwrap.build_bwrap_argv(
            _profile(root), ["bash"], workspace_root=str(root),
            mask_dirs=(), cwd=str(root))


def test_a_dotdot_cwd_is_refused(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(hive_bwrap.ProfileError, match="outside"):
        hive_bwrap.build_bwrap_argv(
            _profile(root), ["bash"], workspace_root=str(root),
            mask_dirs=(), cwd=str(root / ".." / "outside"))


def test_namespace_flags_masks_and_rebind_are_ordered(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    workspace = root / "task-1"
    workspace.mkdir()
    secret = tmp_path / "secret"
    secret.mkdir()
    missing = tmp_path / "missing"

    argv = hive_bwrap.build_bwrap_argv(
        _profile(root), ["bash", "-c", "true"],
        workspace_root=str(root), mask_dirs=(str(secret), str(missing)),
        cwd=str(workspace))
    pairs = _pairs(argv)

    assert ("--unshare-user", "--disable-userns") in pairs
    assert ("--unshare-net", "--unshare-ipc") in pairs
    assert ("--unshare-uts", "--new-session") in pairs
    assert ("--hostname", "hive-subagent") in pairs
    assert ("--cap-drop", "ALL") in pairs
    assert ("--tmpfs", str(secret)) in pairs
    assert ("--tmpfs", str(missing)) not in pairs

    # The broad policy bind is removed; only the child's cwd is bound.
    assert ("--bind", str(root)) not in pairs
    assert ("--bind", str(workspace)) in pairs
    assert ("--chdir", str(workspace)) in pairs

    root_at = pairs.index(("--tmpfs", str(root)))
    dir_at = pairs.index(("--dir", str(workspace)))
    bind_at = pairs.index(("--bind", str(workspace)))
    assert root_at < dir_at < bind_at
    assert argv[-4:] == ["--", "bash", "-c", "true"]


def test_nested_workspace_creates_each_parent(tmp_path):
    root = tmp_path / "workspaces"
    workspace = root / "conversation" / "task"
    workspace.mkdir(parents=True)
    argv = hive_bwrap.build_bwrap_argv(
        _profile(root), ["bash"],
        workspace_root=str(root), mask_dirs=(), cwd=str(workspace))
    pairs = _pairs(argv)
    assert ("--dir", str(root / "conversation")) in pairs
    assert ("--dir", str(workspace)) in pairs


def test_run_covers_media_without_a_redundant_mask():
    assert "/run" in hive_bwrap.MASK_DIRS
    assert "/run/media" not in hive_bwrap.MASK_DIRS


def test_the_parent_tool_allows_one_delegation_level():
    import yaml

    data = yaml.safe_load((WRAPPER.parent / "cordis.patch.yml").read_text())
    rows = [entry for patch in data if "insert" in patch
            for entry in patch["insert"]]
    row = next(entry for entry in rows if entry.get("id") == "hive-subagent-tool")
    assert row["config"]["maxDepth"] == 1


def test_empty_command_is_refused(tmp_path):
    root = tmp_path / "workspaces"
    workspace = root / "task"
    workspace.mkdir(parents=True)
    with pytest.raises(hive_bwrap.ProfileError, match="empty command"):
        hive_bwrap.build_bwrap_argv(
            _profile(root), [], workspace_root=str(root), mask_dirs=(),
            cwd=str(workspace))


def test_main_reports_a_missing_profile(capsys):
    assert hive_bwrap.main([]) == 125
    assert "hive-bwrap: missing profile separator" in capsys.readouterr().err
