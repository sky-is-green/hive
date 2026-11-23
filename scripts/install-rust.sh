#!/usr/bin/env bash
# Install the Rust runtime binaries. Python research installs keep `hive-py`.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${HIVE_PREFIX:-$HOME/.local}"
BIN_DIR="$REPO/rust/target/release"
SKIP_BUILD=0
UNINSTALL=0
WITH_SYSTEMD=1

usage() {
  cat <<'USAGE'
usage: scripts/install-rust.sh [options]

  --prefix DIR      install prefix (default: $HIVE_PREFIX or ~/.local)
  --bin-dir DIR     prebuilt binaries (default: rust/target/release)
  --skip-build      install existing binaries without cargo build
  --no-systemd      do not install the user service unit
  --uninstall       remove the installed Rust runtime files
  -h, --help        show this help

Installs: hive, hive-daemon, hive-encoder, hive-memory, hive-sandbox,
the static console, and the user hive-daemon.service unit. The Python
package provides `hive-py`, not `hive`.
USAGE
}

while (($#)); do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --bin-dir) BIN_DIR="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --no-systemd) WITH_SYSTEMD=0; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PREFIX="${PREFIX/#\~/$HOME}"
BIN_DIR="${BIN_DIR/#\~/$HOME}"
BINARIES=(hive hive-daemon hive-encoder hive-memory hive-sandbox)
SHARE="$PREFIX/share/hive"
UNIT="$PREFIX/lib/systemd/user/hive-daemon.service"

if ((UNINSTALL)); then
  for name in "${BINARIES[@]}"; do rm -f "$PREFIX/bin/$name"; done
  rm -rf "$SHARE"
  rm -f "$UNIT"
  echo "removed the Rust runtime from $PREFIX; use hive-py or reinstall Python if needed"
  exit 0
fi

if ((SKIP_BUILD == 0)); then
  if ! command -v cargo >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env" 2>/dev/null || true
  fi
  command -v cargo >/dev/null 2>&1 || {
    echo "cargo is not installed; install Rust or pass --skip-build" >&2
    exit 1
  }
  cargo build --release --manifest-path "$REPO/rust/Cargo.toml" \
    -p hive-cli -p hive-daemon -p hive-encoder -p hive-memory -p hive-sandbox
fi

for name in "${BINARIES[@]}"; do
  if [[ ! -x "$BIN_DIR/$name" ]]; then
    echo "missing executable: $BIN_DIR/$name" >&2
    exit 1
  fi
done

mkdir -p "$PREFIX/bin" "$SHARE"
for name in "${BINARIES[@]}"; do
  install -Dm755 "$BIN_DIR/$name" "$PREFIX/bin/$name"
done
# The installed binary resolves the backend manifest under HIVE_HOME.
HIVE_ROOT="${HIVE_HOME:-$HOME/.hive}"
if [[ -f "$REPO/tools/backends/manifest.json" ]]; then
  mkdir -p "$HIVE_ROOT/tools/backends" "$SHARE/tools/backends"
  cp "$REPO/tools/backends/manifest.json" "$HIVE_ROOT/tools/backends/manifest.json"
  cp "$REPO/tools/backends/manifest.json" "$SHARE/tools/backends/manifest.json"
fi
# An earlier installer copied the manifest to $HIVE_HOME/harness/data, which
# shadows the real Python `harness` package when the sidecar cwd is $HIVE_HOME.
if [[ -f "$HIVE_ROOT/harness/data/backends-manifest.json" ]]; then
  rm -f "$HIVE_ROOT/harness/data/backends-manifest.json"
  rmdir "$HIVE_ROOT/harness/data" 2>/dev/null || true
  rmdir "$HIVE_ROOT/harness" 2>/dev/null || true
fi
if [[ -d "$REPO/harness/webui" ]]; then
  rm -rf "$SHARE/webui"
  cp -a "$REPO/harness/webui" "$SHARE/webui"
fi
if ((WITH_SYSTEMD)) && [[ -f "$REPO/rust/packaging/systemd/hive-daemon.service" ]]; then
  install -Dm644 "$REPO/rust/packaging/systemd/hive-daemon.service" "$UNIT"
fi

VERSION="$("$PREFIX/bin/hive" --version 2>/dev/null | head -1 || true)"
json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
printf '{"name":"hive","version":"%s","prefix":"%s","source":"%s","installed_at":"%s"}\n' \
  "$(json_escape "${VERSION#hive }")" "$(json_escape "$PREFIX")" \
  "$(json_escape "$REPO")" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$SHARE/install.json"

echo "installed Rust runtime into $PREFIX/bin"
echo "  $("$PREFIX/bin/hive" --version 2>/dev/null || echo 'hive (version check failed)')"
echo "  Python sidecar: pip install -e 'splinter[serve]' -e '.[studio]'"
echo "  sidecar interpreter: export HIVE_PYTHON=/path/to/venv/bin/python"
