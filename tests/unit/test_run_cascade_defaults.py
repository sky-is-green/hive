"""Unit tests for the runner's local-first defaults and the opt-in API fallback.

The parser and the fallback endpoint are pure data/seams, so they are pinned
here without a model or a network call.
"""

from __future__ import annotations

import experiments.cascade.run_cascade as rc


def test_default_flow_is_local_first_with_api_off():
    args = rc.build_parser().parse_args([])

    assert args.router == "none"          # no pre-generation gate
    assert args.judge == "tiny-jev"       # local validation
    assert args.judge_device == "cpu"     # never a second ROCm context
    assert args.router_mode == "gate"     # gate with no router = plain local
    assert args.api_fallback == ""        # API off by default
    assert args.judge_threshold == 0.5


def test_api_fallback_is_opt_in_and_configurable():
    args = rc.build_parser().parse_args([
        "--api-fallback", "http://127.0.0.1:9000/v1",
        "--api-fallback-model", "my-model",
        "--api-fallback-key", "secret",
    ])

    assert args.api_fallback.endswith("/v1")
    assert args.api_fallback_model == "my-model"
    assert args.api_fallback_key == "secret"


def test_measurement_configuration_is_still_available():
    args = rc.build_parser().parse_args([
        "--router", "api", "--judge", "api", "--router-mode", "measure",
    ])

    assert (args.router, args.judge, args.router_mode) == ("api", "api", "measure")


def test_fallback_frontier_needs_no_opencode_auth(monkeypatch):
    monkeypatch.delenv("CASCADE_API_KEY", raising=False)

    def boom():
        raise AssertionError("the fallback must not read the OpenCode auth store")

    monkeypatch.setattr(rc, "_api_key", boom)
    frontier = rc.Frontier(rc.Usage(), base_url="http://127.0.0.1:9000/v1")

    assert frontier.key == "none"
    assert frontier.base_url == "http://127.0.0.1:9000/v1"
    assert frontier.model == rc.API_MODEL
