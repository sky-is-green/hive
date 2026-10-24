"""The detached-serve record: paths, round-trip, stale detection."""
import os
import sys

import pytest

from harness import serve_state


def test_record_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_STATE_DIR", str(tmp_path))
    record = {
        "name": "demo",
        "pid": os.getpid(),
        "stack": "demo-stack",
        "started_at": 1.0,
        "tiers": [{"role": "face", "port": 1234}],
    }
    path = serve_state.write_record(record)
    assert path == tmp_path / "serve" / "demo.json"
    assert path.is_file()

    loaded = serve_state.read_record("demo")
    assert loaded is not None
    assert loaded["name"] == "demo"
    assert loaded["pid"] == os.getpid()
    assert loaded["tiers"][0]["port"] == 1234
    assert loaded["updated_at"] >= 1.0

    rows = serve_state.list_records()
    assert [row["name"] for row in rows] == ["demo"]
    assert rows[0]["record_path"] == str(path)

    assert serve_state.remove_record("demo") is True
    assert serve_state.read_record("demo") is None
    assert serve_state.remove_record("demo") is False


def test_name_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_STATE_DIR", str(tmp_path))
    assert serve_state.validate_name("gpu-box_1.2") == "gpu-box_1.2"
    for bad in ("", "../escape", "a/b", "a" * 65, "-leading"):
        with pytest.raises(ValueError):
            serve_state.validate_name(bad)


def test_corrupt_record_is_skipped_by_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_STATE_DIR", str(tmp_path))
    directory = serve_state.serve_dir()
    (directory / "bad.json").write_text("{ not json", encoding="utf-8")
    serve_state.write_record({"name": "good", "pid": os.getpid()})
    assert [row["name"] for row in serve_state.list_records()] == ["good"]
    with pytest.raises(ValueError):
        serve_state.read_record("bad")


def test_pid_probe_and_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_STATE_DIR", str(tmp_path))
    assert serve_state.pid_alive(os.getpid()) is True
    assert serve_state.pid_alive(0) is False
    assert serve_state.pid_alive(-1) is False

    record = {"pid": os.getpid(),
              "cmdline": serve_state.pid_cmdline(os.getpid())}
    # /proc is Linux-only; where it is unavailable the check degrades to alive
    assert serve_state.pid_matches(record) is True
    assert serve_state.pid_matches({"pid": 0, "cmdline": "x"}) is False
    if record["cmdline"]:
        stale = dict(record, cmdline="something else entirely")
        assert serve_state.pid_matches(stale) is False


def test_terminate_on_a_dead_pid_is_a_noop():
    ok, message = serve_state.terminate(0)
    assert ok is True
    assert "stopped" in message


def test_tail_and_duration(tmp_path):
    path = tmp_path / "x.log"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    assert serve_state.tail_lines(path, 2) == ["two", "three"]
    assert serve_state.tail_lines(path, 0) == ["one", "two", "three"]
    assert serve_state.format_duration(0) == "00:00:00"
    assert serve_state.format_duration(3661) == "01:01:01"
    assert serve_state.format_duration(90061) == "1d 01:01:01"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX session semantics")
def test_spawn_starts_a_new_session(tmp_path, monkeypatch):
    """A detached child must outlive the terminal that started it."""
    monkeypatch.setenv("HIVE_STATE_DIR", str(tmp_path))
    log = tmp_path / "child.log"
    proc = serve_state.spawn([sys.executable, "-c", "import os, time; "
                              "print(os.getpid(), flush=True); time.sleep(30)"],
                             log)
    try:
        assert proc.pid > 0
        assert serve_state.pid_alive(proc.pid)
        if os.path.isdir("/proc"):
            assert os.getpgid(proc.pid) == proc.pid
    finally:
        proc.terminate()
        proc.wait(timeout=10)
