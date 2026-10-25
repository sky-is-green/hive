"""E0 is an explainable fit calculation, not a guessed settings vector."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.controller.recommend import (
    ModelRequest,
    ProposalError,
    model_request_from_path,
    propose,
    validate_changes,
    validate_proposal,
)


def _profile(*, display_free=2.14, include_gpu=True):
    gpus = [
        {"index": 0, "name": "AMD GPU 0", "backend": "rocm",
         "memory_gb": 21.46, "used_gb": 19.32, "free_gb": display_free,
         "display": True, "visible": True},
        {"index": 1, "name": "AMD GPU 1", "backend": "rocm",
         "memory_gb": 21.46, "used_gb": 0.55, "free_gb": 20.91,
         "display": False, "visible": True},
    ]
    return {
        "schema_version": 1,
        "profile_id": "local-test",
        "cpu": {"physical": 8, "logical": 16},
        "ram": {"total_gb": 30.9, "available_gb": 18.2},
        "gpus": gpus if include_gpu else [],
    }


def test_no_model_still_gives_a_conservative_thread_count():
    proposal = propose(
        _profile(), now=datetime(2026, 10, 25, 21, tzinfo=timezone.utc))
    assert proposal["changes"] == {"runtime_defaults.threads": 8}
    assert proposal["warnings"]
    assert proposal["requires_confirmation"] is True
    assert proposal["scope"] == "profile:local-test"


def test_a_model_that_fits_proposes_all_layers_and_context():
    proposal = propose(
        _profile(),
        model=ModelRequest(name="test", weights_gib=12.0,
                           native_ctx=32768, desired_ctx=16384),
    )
    assert proposal["changes"]["runtime_defaults.threads"] == 8
    assert proposal["changes"]["runtime_defaults.ctx_size"] == 16384
    assert proposal["changes"]["runtime_defaults.ngl"] == 999
    assert proposal["confidence"] >= 0.85
    assert any("fit in VRAM" in line for line in proposal["evidence"])


def test_a_context_smaller_than_the_steps_is_not_guessed():
    proposal = propose(
        _profile(),
        model=ModelRequest(name="tiny", weights_gib=1.0, native_ctx=2048),
    )
    assert "runtime_defaults.ctx_size" not in proposal["changes"]
    assert any("does not fit" in warning for warning in proposal["warnings"])


def test_no_gpu_does_not_propose_gpu_layers():
    proposal = propose(
        _profile(include_gpu=False),
        model=ModelRequest(name="test", weights_gib=4.0, native_ctx=32768),
    )
    assert proposal["changes"] == {"runtime_defaults.threads": 8}
    assert any("no visible GPU" in warning for warning in proposal["warnings"])


def test_a_model_too_large_for_memory_is_refused_not_guessed():
    proposal = propose(
        _profile(),
        model=ModelRequest(name="huge", weights_gib=1000.0, native_ctx=32768),
    )
    assert "runtime_defaults.ngl" not in proposal["changes"]
    assert any("does not fit" in warning for warning in proposal["warnings"])


def test_only_allowlisted_keys_are_accepted():
    assert validate_changes({"runtime_defaults.threads": 4}) == {
        "runtime_defaults.threads": 4}
    with pytest.raises(ProposalError, match="allowlist"):
        validate_changes({"quality_defaults.gate": "off"})
    with pytest.raises(ProposalError, match="outside"):
        validate_changes({"runtime_defaults.threads": 0})
    with pytest.raises(ProposalError, match="multiple"):
        validate_changes({"runtime_defaults.ctx_size": 8193})
    with pytest.raises(ProposalError, match="exceeds available RAM"):
        validate_changes({"runtime_defaults.resident_gib": 99}, _profile())


def test_a_proposal_id_is_validated():
    good = {
        "id": "rec-20261025T210000-abcdef",
        "changes": {"runtime_defaults.threads": 4},
    }
    assert validate_proposal(good)["id"] == good["id"]
    bad = {**good, "id": "../../escape"}
    with pytest.raises(ProposalError, match="invalid proposal id"):
        validate_proposal(bad)


def test_model_path_uses_size_and_metadata_only(tmp_path):
    path = tmp_path / "fake.gguf"
    path.write_bytes(b"x" * (4 * 1024 * 1024))
    request = model_request_from_path(path, desired_ctx=8192)
    assert request.path == str(path)
    assert request.weights_gib is not None and request.weights_gib > 0
    assert request.native_ctx is None
    assert request.desired_ctx == 8192
