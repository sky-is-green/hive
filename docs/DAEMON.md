# Hive daemon service

`hive-daemon` owns the public Studio API and a private Python sidecar. The
systemd unit in `rust/packaging/systemd/hive-daemon.service` runs it as a user
service with restart-on-failure.

## Layout assumed by the unit

```
~/.local/bin/hive-daemon
~/.local/bin/hive-encoder             # optional native encoder service
~/.local/share/hive/webui/            # harness/webui assets
~/.hive/.venv/bin/python              # Python sidecar environment
~/.hive/                              # HIVE_HOME: models, stacks, logs, state
```

Override any path in `~/.config/hive/daemon.env`:

```sh
HIVE_PYTHON=/srv/hive/.venv/bin/python
HIVE_HOME=/srv/hive/data
HARNESS_TOKEN=...          # optional; protects /v1/*
RUST_LOG=hive_daemon=debug

# Native encoder (Phase 3): the daemon starts hive-encoder serve and points
# the sidecar at it; no torch/sentence-transformers is loaded on this path.
HARNESS_EMBEDDING_BACKEND=rust
HIVE_ENCODER_DIR=/srv/hive/models/encoder   # onnx/model.onnx + tokenizer.json
# HIVE_ENCODER_PORT=8172
# HIVE_ENCODER_BIN=/srv/hive/bin/hive-encoder
```

`HIVE_PYTHON` is authoritative. If it is set but missing or not executable,
the daemon exits at startup instead of silently choosing another interpreter.

## Install

```sh
cargo build --release --manifest-path rust/Cargo.toml -p hive-cli -p hive-daemon
install -Dm755 rust/target/release/hive-daemon ~/.local/bin/hive-daemon
install -Dm755 rust/target/release/hive-encoder ~/.local/bin/hive-encoder
install -Dm755 rust/target/release/hive ~/.local/bin/hive
mkdir -p ~/.local/share/hive
cp -r harness/webui ~/.local/share/hive/webui

mkdir -p ~/.config/systemd/user
cp rust/packaging/systemd/hive-daemon.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hive-daemon
```

Check it:

```sh
systemctl --user status hive-daemon
curl -s http://127.0.0.1:8765/health
hive ui status
```

The console is served same-origin on the API port. `hive ui web --detach`
uses `~/.hive/logs/daemon.log` and `~/.hive/logs/daemon.pid` when running
outside systemd.

## Operations

```sh
systemctl --user restart hive-daemon
systemctl --user stop hive-daemon
journalctl --user -u hive-daemon -f
```

On stop, the daemon drains local engines (bounded `/slots` wait), terminates
the sidecar's process group, and leaves no model process behind. A crashed
sidecar is restarted with 1s→30s backoff.
