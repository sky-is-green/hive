# Install

Two installs coexist for one release:

- **`hive`** — the Rust runtime (CLI, daemon, encoder, memory, sandbox).
- **`hive-py`** — the Python reference CLI, used by research and the sidecar.

## Rust runtime

```sh
scripts/install-rust.sh --prefix ~/.local
```

This builds `hive`, `hive-daemon`, `hive-encoder`, `hive-memory`, and
`hive-sandbox`, installs them into `~/.local/bin`, copies the static console to
`~/.local/share/hive/webui`, and installs
`~/.local/lib/systemd/user/hive-daemon.service`. Use `--skip-build` to install
prebuilt binaries, `--no-systemd` to skip the unit, or `--uninstall` to roll
back to the Python reference.

Then run `hive setup`. With no flags it installs a portable engine and the
native memory encoder (`hive-encoder fetch`). The daemon auto-detects the
encoder under `$HIVE_HOME/models/encoder`; `hive setup --no-fetch` skips both
downloads.

The sidecar interpreter is explicit:

```sh
export HIVE_PYTHON=/path/to/venv/bin/python
```

The daemon unit reads `~/.config/hive/daemon.env`; put `HIVE_PYTHON` there for a
user service. `hive doctor` reports missing runtime pieces.

## Python reference

```sh
python -m venv .venv
.venv/bin/pip install -e "splinter[serve]"
.venv/bin/pip install -e ".[studio]"
.venv/bin/hive-py version --json
```

`splinter[serve]` provides the sidecar's memory backend (`backend`, `cortex`,
`splinter.server`); `.[studio]` provides the console routes. The `[encoders]`
extra is only needed when using the Python encoder; the Rust runtime uses the
native `hive-encoder` and does not import torch.

The Python install does **not** provide `hive`; that name belongs to the Rust
runtime. Existing virtualenvs that still contain the old `hive` script should be
reinstalled (`pip install -e .`) so the entry point is removed.

## Release artifacts

`scripts/release_manifest.py` writes `release-manifest.json`, `SHA256SUMS`, and
a CycloneDX SBOM. Sign them with `scripts/sign_artifact.py` (or minisign); see
`docs/RELEASES.md`.

## Rollback

```sh
scripts/install-rust.sh --uninstall --prefix ~/.local
```

Then use `hive-py` from the Python environment. State under `HIVE_HOME` is
shared and forward-compatible; the uninstall does not delete it.
