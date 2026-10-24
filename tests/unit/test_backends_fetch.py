"""Backend installation: asset selection and a staged, no-network install."""
import os
import shutil
import tarfile

import pytest

import harness.backends as backends
import harness.cli as cli


def _exe() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def _archive(tmp_path):
    """A release-shaped tar.gz with a server binary and a sibling library."""
    source = tmp_path / "build" / "bin"
    source.mkdir(parents=True)
    binary = source / _exe()
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    (source / "libggml-test.so").write_text("", encoding="utf-8")
    archive = tmp_path / "llama-bTEST-bin-ubuntu-vulkan-x64.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname="build/bin")
    return archive


def test_platform_asset_selection():
    assets = [
        {"name": "llama-b1-bin-ubuntu-vulkan-x64.tar.gz", "url": "u1"},
        {"name": "llama-b1-bin-win-vulkan-x64.zip", "url": "u2"},
    ]
    chosen = backends.select_asset("vulkan", assets, key="linux-x64")
    assert chosen["name"].endswith("ubuntu-vulkan-x64.tar.gz")
    with pytest.raises(RuntimeError, match="no release asset"):
        backends.select_asset("vulkan", assets, key="plan9-x64")


def test_fetch_stages_and_installs(tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "platform_key", lambda: "linux-x64")
    archive = _archive(tmp_path)
    root = tmp_path / "root"
    calls = []

    def download(url, destination, progress=None):
        calls.append(url)
        shutil.copyfile(archive, destination)
        if progress:
            progress(10, 10)

    result = backends.fetch_backend(
        "vulkan", root=root, verify=False,
        release={"tag": "bTEST", "assets": [{
            "name": archive.name, "url": "https://example.invalid/vulkan",
            "size": archive.stat().st_size}]},
        download=download)
    assert result["already"] is False
    target = root / "tools" / "backends" / "vulkan"
    assert (target / _exe()).is_file()
    assert (target / "libggml-test.so").is_file()
    assert calls == ["https://example.invalid/vulkan"]
    assert backends.resolve("vulkan", root=root)["installed"] is True

    second = backends.fetch_backend("vulkan", root=root, verify=False,
                                    release={"tag": "bTEST", "assets": []},
                                    download=download)
    assert second["already"] is True


def test_fetch_refuses_an_existing_directory_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "platform_key", lambda: "linux-x64")
    root = tmp_path / "root"
    (root / "tools" / "backends" / "vulkan").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already exists"):
        backends.fetch_backend(
            "vulkan", root=root, verify=False,
            release={"tag": "bTEST", "assets": [{
                "name": "llama-bTEST-bin-ubuntu-vulkan-x64.tar.gz",
                "url": "https://example.invalid/vulkan"}]},
            download=lambda url, dst, progress=None: shutil.copyfile(
                _archive(tmp_path), dst))


def test_cli_fetch_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.backend_manifest, "fetch_backend",
                        lambda name, **kwargs: {
                            "name": name, "path": "/tmp/vulkan",
                            "binary": "/tmp/vulkan/llama-server",
                            "installed": True, "already": False,
                            "tag": "bTEST", "asset": "a.tar.gz"})
    assert cli.main(["backends", "fetch", "vulkan", "--json"]) == 0
    import json
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "vulkan"
    assert payload["tag"] == "bTEST"
