"""Opt-in live Hub checks: search, file listing, and one small download.

Set ``HIVE_LIVE_HUB=1`` to run:

    HIVE_LIVE_HUB=1 python -m pytest tests/conformance/test_live_hub.py -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from conftest import parse_json, run_python, run_rust

pytestmark = pytest.mark.skipif(
    os.environ.get("HIVE_LIVE_HUB") != "1",
    reason="set HIVE_LIVE_HUB=1 to run the live Hugging Face checks",
)

REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
FILE = "SmolLM2-135M-Instruct-IQ3_M.gguf"


def test_search_matches_python(rust_bin, cli_env):
    env, _ = cli_env
    py = run_python(["models", "search", "smollm", "--limit", "3", "--json"], env)
    rs = run_rust(rust_bin, ["models", "search", "smollm", "--limit", "3", "--json"], env)
    assert py.returncode == rs.returncode == 0, (py.stderr, rs.stderr)
    assert parse_json(py) == parse_json(rs)


def test_files_match_python(rust_bin, cli_env):
    env, _ = cli_env
    py = run_python(["models", "files", REPO, "--json"], env)
    rs = run_rust(rust_bin, ["models", "files", REPO, "--json"], env)
    assert py.returncode == rs.returncode == 0, (py.stderr, rs.stderr)
    assert parse_json(py) == parse_json(rs)


def test_serve_model_remote_matches_python(rust_bin, cli_env, serve_fixture, tmp_path):
    env, _ = cli_env
    python_models = tmp_path / "python-serve-models"
    rust_models = tmp_path / "rust-serve-models"
    python_stacks = tmp_path / "python-stacks"
    rust_stacks = tmp_path / "rust-stacks"
    python_stacks.mkdir()
    rust_stacks.mkdir()
    common = [
        "serve",
        "--model",
        f"{REPO}:{FILE}",
        "--once",
        "--json",
        "--save",
        "remote",
        "--llama-server",
        str(serve_fixture["engine"]),
    ]
    py = run_python(
        [*common, "--models-dir", str(python_models), "--stacks-dir", str(python_stacks)],
        env,
    )
    rs = run_rust(
        rust_bin,
        [*common, "--models-dir", str(rust_models), "--stacks-dir", str(rust_stacks)],
        env,
    )
    assert py.returncode == rs.returncode == 0, (py.stderr, rs.stderr)
    python_saved = json.loads((python_stacks / "remote.json").read_text())
    rust_saved = json.loads((rust_stacks / "remote.json").read_text())
    assert python_saved["tiers"][0]["file"].endswith(FILE)
    assert rust_saved["tiers"][0]["file"].endswith(FILE)
    assert python_saved["tiers"][0]["pin"] == rust_saved["tiers"][0]["pin"]
    assert python_saved["tiers"][0].get("ts") == rust_saved["tiers"][0].get("ts")


def test_download_matches_python(rust_bin, cli_env, tmp_path):
    env, _ = cli_env
    python_models = tmp_path / "python-models"
    rust_models = tmp_path / "rust-models"
    py = run_python(
        ["models", "download", f"{REPO}:{FILE}", "--json", "--models-dir", str(python_models)],
        env,
    )
    rs = run_rust(
        rust_bin,
        ["models", "download", f"{REPO}:{FILE}", "--json", "--models-dir", str(rust_models)],
        env,
    )
    assert py.returncode == rs.returncode == 0, (py.stderr, rs.stderr)
    python_payload = parse_json(py)
    rust_payload = parse_json(rs)
    for key in ("repo", "file", "size_gb"):
        assert python_payload[key] == rust_payload[key]
    assert Path(python_payload["path"]).is_file()
    assert Path(rust_payload["path"]).is_file()
    assert Path(python_payload["path"]).stat().st_size == Path(
        rust_payload["path"]
    ).stat().st_size
