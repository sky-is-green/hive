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

The manifest currently pins SHA256, not a publisher signature. Signed manifests
are part of the release work: sign `plugin.json` with the release key and verify
it before installation, then keep the hash pin for the module bytes.
