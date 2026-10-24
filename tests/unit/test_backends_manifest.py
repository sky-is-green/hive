"""Backend manifest: every declared build, where it is, and the pick order."""

import os

import pytest

import harness.backends as backends
from harness.models import BACKENDS


def _exe() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def _install(root, name):
    d = root / "tools" / "backends" / name
    d.mkdir(parents=True)
    (d / _exe()).write_text("", encoding="utf-8")


def test_manifest_covers_every_backend_the_manager_accepts():
    assert set(backends.names()) == set(BACKENDS)


def test_resolve_reports_the_binary_and_whether_it_is_installed(tmp_path):
    _install(tmp_path, "rocm")
    entry = backends.resolve("rocm", root=tmp_path)
    assert entry["installed"] is True
    assert entry["label"]
    assert entry["binary"].endswith(_exe())
    assert backends.resolve("vulkan", root=tmp_path)["installed"] is False


def test_unknown_backend_is_an_error():
    with pytest.raises(KeyError, match="unknown backend"):
        backends.resolve("no-such-backend")


def test_vendor_build_beats_the_portable_one(tmp_path):
    _install(tmp_path, "rocm")
    _install(tmp_path, "vulkan")
    assert backends.for_accelerator("amd", root=tmp_path) == "rocm"


def test_falls_back_to_vulkan_without_rocm(tmp_path):
    _install(tmp_path, "vulkan")
    assert backends.for_accelerator("amd", root=tmp_path) == "vulkan"


def test_nothing_installed_returns_none(tmp_path):
    assert backends.for_accelerator("amd", root=tmp_path) is None


def test_cpu_is_its_own_accelerator(tmp_path):
    _install(tmp_path, "vulkan")   # vulkan is not a cpu build
    assert backends.for_accelerator("cpu", root=tmp_path) is None
    _install(tmp_path, "cpu")
    assert backends.for_accelerator("cpu", root=tmp_path) == "cpu"


def test_rows_cover_the_manifest(tmp_path):
    rows = backends.rows(root=tmp_path)
    assert [row["name"] for row in rows] == list(backends.names())
    assert all(row["installed"] is False for row in rows)
