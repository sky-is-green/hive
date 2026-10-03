"""Unit tests for ``experiments.cascade.bench_report`` (MODEL-CARD renderer)."""

from __future__ import annotations

from experiments.cascade.bench_report import render


def _manifest() -> dict:
    return {
        "run": "strata-bench-test-20261003",
        "created": "2026-10-03 00:37:01",
        "generator": {
            "model_path": "/m/Qwen-Q2_0.gguf",
            "model_sha256": "a" * 64,
            "stack": "strata-125b",
            "total_slots": 1,
            "n_ctx": 4096,
            "build_info": "Strata 0.1.37",
        },
        "git": {"commit": "b" * 40, "dirty_files": 0},
        "tasks": {"name": "cascade-bench-v1", "sha256": "c" * 64, "split": "test"},
        "decode": {"samples": 3, "temperature": 0.7, "max_tokens": 3072,
                   "system": "sys"},
        "checker_notes": "notes",
        "aggregates": {
            "overall": {"tasks": 1, "samples": 3, "sample_accuracy": 1.0,
                        "task_macro": 1.0},
            "by_benchmark": {
                "gsm8k": {"tasks": 1, "samples": 3, "sample_accuracy": 1.0,
                          "task_macro": 1.0, "mean_ms": 1.0, "mean_tokens": 2.0},
            },
        },
    }


def test_render_labels_the_generator_and_prints_the_benchmark_table():
    text = render(_manifest())

    assert text.startswith("# Benchmark results — Strata 0.1.37")
    assert "| gsm8k | 1 | 3 | 1.000 | 1.000 | 1 | 2 |" in text
    assert "split `test`" in text
    assert "escalation economics" in text


def test_render_falls_back_to_the_model_file_name_without_build_info():
    manifest = _manifest()
    manifest["generator"].pop("build_info")

    assert render(manifest).startswith("# Benchmark results — Qwen-Q2_0.gguf")
