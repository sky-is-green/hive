"""The resident-experts control: the launch baseline, and the engine's flag.

The Strata engine can keep the experts its VRAM cache cannot hold in
ordinary RAM (``--resident-budget-gib``), so they are never re-read from
the disk.  Measured on the 176B through the Studio's own launch: the
cold first request falls from 14.2 s to 7.5 s and the storage read on
the next request from 54-261 MB to 0.6-2.4 MB, with the warm rate
unchanged.

These tests are hermetic - no model, no GPU.  They check the contract:

  * the Global tab's runtime block carries the field, and the control
    resolves it from the settings,
  * the launch path turns the setting into the engine's flag, and leaves
    the engine's own default alone when the setting is off,
  * the flag lands in the engine's argv through the stack's merge.
"""

from __future__ import annotations

from harness import controls
from harness.stack.strata import merge_engine_args


def test_the_launch_baseline_resolves_the_resident_budget():
    assert controls.launch_of({"runtime_defaults": {"resident_gib": 8}}) == {"resident_gib": 8}
    # the alias normalises to the canonical name, so the launch path reads
    # one key whatever the control's field was called
    assert controls.launch_of({"runtime_defaults": {"resident_budget_gib": 4}}) == {"resident_gib": 4}
    # unset: the engine's own default is left alone
    assert controls.launch_of({}) == {}
    assert controls.launch_of({"runtime_defaults": {}}) == {}


def test_the_control_is_in_the_global_tabs_runtime_block():
    from harness.global_controls import RUNTIME_FIELDS

    names = {name for name, _label, _kind, _hint in RUNTIME_FIELDS}
    assert "resident_gib" in names
    hint = {name: hint for name, _l, _k, hint in RUNTIME_FIELDS}["resident_gib"]
    # the hint says what it costs, not just what it is
    assert "RAM" in hint
    assert "8" in hint          # the measured budget


def test_the_flag_lands_in_the_engines_argv():
    base = ["--native", "/m.gguf", "--vram-reserve-mib", "3072", "--mmap-experts"]
    merged = merge_engine_args(base, ["--resident-budget-gib", "8"])
    assert "--resident-budget-gib" in merged
    assert merged[merged.index("--resident-budget-gib") + 1] == "8"
    # the base's other flags are untouched
    assert merged[merged.index("--vram-reserve-mib") + 1] == "3072"


def test_the_flag_replaces_an_existing_one_rather_than_duplicating_it():
    base = ["--native", "/m.gguf", "--resident-budget-gib", "2"]
    merged = merge_engine_args(base, ["--resident-budget-gib", "8"])
    assert merged.count("--resident-budget-gib") == 1
    assert merged[merged.index("--resident-budget-gib") + 1] == "8"
