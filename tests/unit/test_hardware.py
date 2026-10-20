"""Unit tests for harness.hardware (sysfs VRAM, visibility, disk)."""

from __future__ import annotations

import sys

from harness.hardware import (
    disk_summary,
    gpu_processes,
    linux_amd_devices,
    parse_visible_indices,
)


def test_parse_visible_indices():
    assert parse_visible_indices(None) is None
    assert parse_visible_indices("") is None
    assert parse_visible_indices("1") == [1]
    assert parse_visible_indices("0,1") == [0, 1]
    assert parse_visible_indices(" 2 , 3 ") == [2, 3]
    # UUIDs / MIG ids are not index lists -> caller falls back to all
    assert parse_visible_indices("GPU-abc") is None


def test_disk_summary_reports_free(tmp_path):
    out = disk_summary({"here": str(tmp_path), "nope": "/does/not/exist"})
    assert "here" in out and out["here"]["free_gb"] > 0
    assert "nope" not in out


def test_gpu_processes_is_a_list():
    assert isinstance(gpu_processes(), list)


def test_linux_amd_devices_have_total_and_free():
    if sys.platform == "win32":
        return
    devices = linux_amd_devices()
    for device in devices:
        assert device["memory_gb"] and device["memory_gb"] > 0
        assert device["free_gb"] is not None
        assert device["free_gb"] <= device["memory_gb"]
        assert "index" in device


def test_the_non_display_card_is_the_one_a_launch_should_take():
    """The desktop's card belongs to the desktop.

    A launch that does not name a card gets the spare one, so the
    window does not stutter (and so the launch does not fail outright
    when the display's VRAM is nearly full).
    """
    from harness.hardware import non_display_index

    devices = [
        {"index": 0, "display": True, "bdf": "0000:03:00.0"},
        {"index": 1, "display": False, "bdf": "0000:07:00.0"},
    ]
    assert non_display_index(devices) == 1
    # one card, and it is the display: there is no spare to take
    assert non_display_index([{"index": 0, "display": True}]) is None
    # no display among them: the first one is as good as any
    assert non_display_index([{"index": 0, "display": False}]) == 0


def test_the_devices_are_listed_in_the_hip_order():
    """The index the launch uses is the one HIP enumerates."""
    from harness.hardware import linux_amd_devices

    devices = linux_amd_devices()
    if len(devices) < 2:
        return  # a single-card box: nothing to order
    bdfs = [d.get("bdf") or "" for d in devices]
    assert bdfs == sorted(bdfs)
    assert [d["index"] for d in devices] == list(range(len(devices)))
