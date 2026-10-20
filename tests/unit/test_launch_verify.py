"""The launch verification probe's judgement (no model, no I/O).

The probe itself touches the disk and the box, so it is not part of the
suite; the parts that can be decided without a machine are tested here.
"""

from __future__ import annotations

from pathlib import Path

from experiments.launch_verify import bootstrap_target, find_shards, port_free


def test_a_split_model_starts_at_its_first_shard(tmp_path):
    (tmp_path / "model-00002-of-00003.gguf").write_text("b")
    (tmp_path / "model-00001-of-00003.gguf").write_text("a")
    shards = find_shards(tmp_path)
    assert len(shards) == 2
    assert bootstrap_target(shards).name == "model-00001-of-00003.gguf"


def test_a_single_file_model_is_its_own_target(tmp_path):
    (tmp_path / "solo.gguf").write_text("a")
    shards = find_shards(tmp_path)
    assert bootstrap_target(shards).name == "solo.gguf"


def test_no_weights_means_no_target(tmp_path):
    assert find_shards(tmp_path) == []
    assert bootstrap_target([]) is None


def test_the_shards_are_found_recursively(tmp_path):
    nested = tmp_path / "qwen" / "Q4_K_M"
    nested.mkdir(parents=True)
    (nested / "qwen-00001-of-00002.gguf").write_text("a")
    shards = find_shards(tmp_path)
    assert len(shards) == 1
    assert bootstrap_target(shards).name == "qwen-00001-of-00002.gguf"


def test_a_free_port_is_free():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert port_free(port) is True


def test_a_taken_port_is_not_free():
    import socket

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert port_free(port) is False
    finally:
        server.close()
