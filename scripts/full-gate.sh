#!/usr/bin/env bash
# Repeatable pre-approval gate. Writes a report and per-step logs.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${HIVE_TEST_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$REPO/.venv/bin/python" ]]; then
    PY="$REPO/.venv/bin/python"
  else
    PY="$(command -v python3)"
  fi
fi
OUT="${HIVE_GATE_OUT:-/tmp/opencode/hive-gate/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
if [[ -d /tmp/opencode/node-libs ]]; then
  export LD_LIBRARY_PATH="/tmp/opencode/node-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export HIVE_RUST_BIN="${HIVE_RUST_BIN:-$REPO/rust/target/debug/hive}"

FAILED=0
PASSED=0
run() {
  local name="$1"; shift
  printf '== %-32s ' "$name"
  if "$@" >"$OUT/$name.log" 2>&1; then
    printf 'ok\n'
    PASSED=$((PASSED + 1))
  else
    printf 'FAIL\n'
    FAILED=$((FAILED + 1))
  fi
}

run rust-fmt cargo fmt --manifest-path "$REPO/rust/Cargo.toml" --all -- --check
run rust-build cargo build --manifest-path "$REPO/rust/Cargo.toml" -p hive-cli -p hive-daemon -p hive-encoder -p hive-memory -p hive-sandbox
run rust-clippy cargo clippy --manifest-path "$REPO/rust/Cargo.toml" --workspace --all-targets -- -D warnings
run rust-tests cargo test --manifest-path "$REPO/rust/Cargo.toml" --workspace
run python-unit "$PY" -m pytest "$REPO/tests/unit" "$REPO/tests/integration" -q
run conformance-nonlive "$PY" -m pytest "$REPO/tests/conformance" -q
run install-layout "$PY" -m pytest "$REPO/tests/conformance/test_install_layout.py" -q

if [[ -n "${HIVE_LIVE_MODEL:-}" && -f "${HIVE_LIVE_MODEL:-}" ]]; then
  LIVE_TESTS="$(for file in "$REPO/tests/conformance/test_daemon.py" "$REPO/tests/conformance/test_live_llama.py"; do
    "$PY" -m pytest "$file" -q -k live --collect-only 2>/dev/null | grep '::' || true
  done)"
  if [[ -z "$LIVE_TESTS" ]]; then
    echo "== live-tests                    no tests collected"
  else
    while IFS= read -r test; do
      run "live-$(basename "$test" | tr -cd '[:alnum:]_-')" "$PY" -m pytest "$test" -q
    done <<< "$LIVE_TESTS"
  fi
else
  echo "== live-tests                    skipped (set HIVE_LIVE_MODEL)"
fi

if [[ -x "$REPO/rust/target/release/hive" ]]; then
  DIST="$OUT/dist"
  run release-manifest "$PY" "$REPO/scripts/release_manifest.py" --bin-dir "$REPO/rust/target/release" --out "$DIST" --allow-missing --no-cargo-metadata
  if [[ -f "$HOME/.hive/keys/dev-release.json" ]]; then
    run release-sign "$PY" "$REPO/scripts/sign_artifact.py" sign --key-file "$HOME/.hive/keys/dev-release.json" --file "$DIST/release-manifest.json" --signature "$DIST/release-manifest.json.sig"
    run release-verify "$PY" "$REPO/scripts/sign_artifact.py" verify --key-file "$HOME/.hive/keys/dev-release.json" --file "$DIST/release-manifest.json" --signature "$DIST/release-manifest.json.sig"
  fi
else
  echo "== release-manifest              skipped (no release hive)"
fi

{
  echo "# Hive pre-approval gate"
  echo
  echo "- candidate: $(git -C "$REPO" log -1 --oneline 2>/dev/null || echo unknown)"
  echo "- python: $PY"
  echo "- passed: $PASSED"
  echo "- failed: $FAILED"
  echo "- logs: $OUT"
} > "$OUT/summary.md"
cat "$OUT/summary.md"
exit $((FAILED > 0))
