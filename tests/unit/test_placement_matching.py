"""The Roles page must attach the right local GGUF metadata for the KV estimate."""

from __future__ import annotations

from harness.webui.dev_server import _attach_local_metadata


def test_exact_alias_wins_and_a_model_is_not_reused_for_another_role():
    local = [
        {
            "name": "LFM3-1.7B-Q4_K_M",
            "file": "LFM3-1.7B-Q4_K_M.gguf",
            "size_gb": 1.1,
            "gguf_metadata": {"model": "small"},
        },
        {
            "name": "LFM3-VL-2B-Q8_0",
            "file": "LFM3-VL-2B-Q8_0.gguf",
            "size_gb": 2.2,
            "gguf_metadata": {"model": "vision"},
        },
    ]
    models = {
        "lfm3": {"label": "LFM3 1.7B", "weights_gb": 1.02},
        "lfm3vl": {"label": "LFM3-VL-2B", "weights_gb": 1.72},
    }

    _attach_local_metadata(models, local)

    assert models["lfm3"]["gguf_metadata"] == {"model": "small"}
    assert models["lfm3vl"]["gguf_metadata"] == {"model": "vision"}
    # the actual file size wins over the UI's advisory weight table
    assert models["lfm3"]["weights_gb"] == 1.1
    assert models["lfm3vl"]["weights_gb"] == 2.2


def test_unknown_role_model_keeps_no_metadata():
    models = {"something-else": {"label": "Something"}}

    _attach_local_metadata(models, [
        {"name": "LFM3-1.7B-Q4_K_M", "file": "LFM3-1.7B-Q4_K_M.gguf",
         "gguf_metadata": {"model": "small"}},
    ])

    assert "gguf_metadata" not in models["something-else"]
