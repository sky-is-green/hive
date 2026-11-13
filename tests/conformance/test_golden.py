"""Frozen golden contracts for the Rust CLI.

Unlike ``test_cli_parity.py`` these tests never start Python: they compare the
Rust binary to snapshots committed under ``tests/conformance/golden/``. Run
``python tests/conformance/golden/generate.py`` to refresh the snapshots after
an intentional contract change.
"""

from __future__ import annotations

import json

import pytest

import golden_support
from conftest import run_rust


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("golden")
    golden_support.build_library(root)
    golden_support.build_stacks(root)
    return root, golden_support.build_env(root)


@pytest.mark.parametrize("name,args", golden_support.CASES, ids=[case[0] for case in golden_support.CASES])
def test_rust_matches_the_frozen_golden(rust_bin, corpus, name, args):
    root, env = corpus
    expected = json.loads((golden_support.GOLDEN / f"{name}.json").read_text())
    result = run_rust(rust_bin, golden_support.expand(args, root), env)
    assert result.returncode == 0, result.stderr
    actual = golden_support.normalize(
        json.loads(result.stdout), root, env["HIVE_PYTHON"]
    )
    assert actual == expected
