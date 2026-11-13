# Frozen CLI golden corpus

These snapshots freeze the JSON contract the Rust CLI must keep. Tests compare
the Rust binary to these files without starting Python.

Refresh them only for an intentional contract change:

```sh
python tests/conformance/golden/generate.py
git diff tests/conformance/golden/
```

The generator runs the Python CLI against deterministic fixtures and replaces
absolute fixture paths, the interpreter version, and the platform string with
placeholders. The root `help` snapshot is trimmed to the ported command set
(`RUST_COMMANDS` in `../golden_support.py`); per-topic help parity is covered by
the live Python test.

Review the diff before committing: a changed snapshot is a contract change, not
an automatic fix.
