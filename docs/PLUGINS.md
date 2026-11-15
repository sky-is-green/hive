# WASM verifier/tool plugins

Plugins are optional WebAssembly modules under the daemon state directory:

```
$HIVE_STATE_DIR/plugins/<name>/
  plugin.json
  module.wasm
```

`plugin.json`:

```json
{
  "name": "arithmetic",
  "version": "0.1.0",
  "module": "module.wasm",
  "sha256": "<sha256 of module.wasm>",
  "fuel": 10000000,
  "memory_mb": 64,
  "max_output_bytes": 1048576
}
```

## ABI

The module has **no imports** (no WASI, no filesystem, no network). It exports:

- `memory`
- `alloc(size: i32) -> i32` — return a pointer to at least `size` bytes.
- `run(ptr: i32, len: i32) -> i32` — read the input JSON at `ptr`, return a
  pointer to a little-endian `u32` length followed by the output JSON.

The host writes the request JSON, calls `run`, checks the output length against
`max_output_bytes`, and parses the result as JSON. A trap, fuel exhaustion, or
invalid JSON becomes a 502.

## Limits

- `fuel` bounds deterministic execution; a spin loop traps.
- `memory_mb` caps linear memory growth; `memory.grow` beyond it traps.
- One instance, one table, one memory per invocation.
- Module files are content-addressed by `sha256`; a mismatch is rejected before
  compilation.
- A module that declares imports is rejected at instantiation.

## API

- `GET /v1/plugins` — manifests, validity, fuel, memory.
- `POST /v1/plugins/{name}/run` — body `{"input": <json>}`; response
  `{"plugin", "output", "fuel_used", "elapsed_ms"}`.

Plugins are installed by placing a directory under the state plugin root. The
daemon does not download or execute plugins from arbitrary paths.

## Signing

Signed manifests are required by default. The daemon trusts keys listed in
`$HIVE_STATE_DIR/plugins/trusted-keys.json` (or the file named by
`HIVE_PLUGIN_TRUSTED_KEYS`):

```json
{"keys": {"release-2026": "<base64 32-byte Ed25519 public key>"}}
```

Create a key and sign a plugin:

```sh
python scripts/sign_plugin.py generate \
  --key-file ~/.hive/keys/plugin.json --key-id release-2026 \
  --trusted-file /path/to/state/plugins/trusted-keys.json

python scripts/sign_plugin.py sign \
  --plugin-dir /path/to/state/plugins/arithmetic \
  --key-file ~/.hive/keys/plugin.json
```

The signature covers the canonical manifest payload (name, version, module,
sha256, fuel, memory limits, key_id) and is base64 Ed25519. `sign_plugin.py
verify` checks it without starting the daemon. The SHA256 module pin is still
enforced independently.

For local development only, `HIVE_PLUGIN_REQUIRE_SIGNATURE=0` allows unsigned
manifests; production should leave it at the default (`1`).
