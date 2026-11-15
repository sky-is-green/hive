# Migrating to the Rust `hive`

The Rust runtime now owns the `hive` command. The Python reference is still
installed as `hive-py`; both use the same state under `HIVE_HOME`.

## Install

```sh
scripts/install-rust.sh --prefix ~/.local
export HIVE_PYTHON=/path/to/venv/bin/python
hive version --json
```

Python research environments keep working:

```sh
.venv/bin/pip install -e ".[studio]"
.venv/bin/hive-py version --json
```

## What moved

- `hive`, `hive serve`, `status`, `stop`, `logs`, `models`, `stacks`,
  `backends`, `doctor`, `setup`, `tui`: Rust.
- Studio daemon, memory API, encoder: Rust with a Python sidecar for the
  remaining Studio/eval routes.
- `cascade`, `tools`, `sandbox`, `web`, `browser`, `ui`, `profile`,
  `controller`, `encoder`: proxied to Python where they are not native yet.

## Rollback

```sh
scripts/install-rust.sh --uninstall --prefix ~/.local
```

Then use `hive-py`. Removing the Rust binaries does not delete state, models,
stacks, or logs.

## Compatibility

JSON output is the compatibility contract; human help text may differ. State
files are forward-compatible: the Rust runtime reads the Python schemas and
preserves unknown fields.

`main` remains frozen until the final rewrite gate; this guide ships with the
integration branch and becomes the release note when that gate passes.
