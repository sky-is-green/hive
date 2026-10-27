#!/usr/bin/env bash
# Install the durable DeepSeek V4.1 engine service for Hive.
#
# Run as root:
#   sudo ./deploy/ds41-engine/install.sh
#
# The defaults are this box's media paths. Override SRC, MODEL, ENGRAM1,
# ENGRAM14 or LIBSRC if the model or the engine checkout moves.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/strata-ds41}"
MODEL="${MODEL:-/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/models-p310/V41-GSQ}"
ENGRAM1="${ENGRAM1:-/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/v41-engram/layers1}"
ENGRAM14="${ENGRAM14:-/run/media/penis/c10e404b-4abc-422d-a370-18b5fd681981/v41-engram/layers14}"
LIBSRC="${LIBSRC:-/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/prism-qwen4exp-shrink/build-hip/bin}"
BASE="${BASE:-/srv/hive-engine}"
USER="${HIVE_ENGINE_USER:-hive-engine}"

die() { echo "x $*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root"

for p in "$SRC/serve" "$SRC/tools" "$SRC/local" "$SRC/build-hip-pre/ds41_generate" \
         "$MODEL/V41-GSQ.gguf" "$ENGRAM1/engram-L1.safetensors" \
         "$ENGRAM14/engram-L14.safetensors" "$HERE/ds41.json" \
         "$HERE/ds41-engine.service" "$HERE/hive.service.d/ds41.conf"; do
    [ -e "$p" ] || die "missing $p"
done
ls "$LIBSRC"/libggml*.so* >/dev/null 2>&1 || die "no libggml*.so* under $LIBSRC"

echo "=== python dependencies ==="
if ! /usr/bin/python3 -c 'import regex' 2>/dev/null; then
    command -v pacman >/dev/null || die "python-regex is missing and pacman is not available"
    pacman -S --needed --noconfirm python-regex
fi
/usr/bin/python3 - <<'PY'
import regex, numpy, yaml, orjson, pydantic
print("ENGINE_PY_OK", regex.__version__)
PY

echo "=== account and layout ==="
if ! getent passwd "$USER" >/dev/null; then
    useradd --system --user-group --home-dir "$BASE/home" \
        --shell /usr/sbin/nologin "$USER"
fi
if id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
    die "$USER is in the docker group; remove it before running the engine"
fi
install -d -o root -g "$USER" -m 0750 "$BASE"
install -d -o "$USER" -g "$USER" -m 0700 "$BASE/home" "$BASE/logs"
install -d -o root -g "$USER" -m 0750 "$BASE/app" "$BASE/lib"
install -d -o root -g "$USER" -m 0755 "$BASE/model" "$BASE/engram" \
    "$BASE/engram/layers1" "$BASE/engram/layers14"

echo "=== engine code, binary and libraries ==="
rm -rf "$BASE/app/serve" "$BASE/app/tools" "$BASE/app/local" \
       "$BASE/app/build-hip-pre" "$BASE/lib"/*
cp -a "$SRC/serve" "$BASE/app/"
cp -a "$SRC/tools" "$BASE/app/"
cp -a "$SRC/local" "$BASE/app/"
install -d -o root -g "$USER" -m 0755 "$BASE/app/build-hip-pre"
install -m 0755 -o root -g "$USER" "$SRC/build-hip-pre/ds41_generate" \
    "$BASE/app/build-hip-pre/ds41_generate"
cp -a "$LIBSRC"/libggml*.so* "$BASE/lib/"
if LD_LIBRARY_PATH="$BASE/lib" ldd "$BASE/app/build-hip-pre/ds41_generate" 2>/dev/null | grep -q 'not found'; then
    die "engine binary still has missing libraries"
fi
chown -R root:"$USER" "$BASE/app" "$BASE/lib"
chmod -R g+rX,o-rwx "$BASE/app" "$BASE/lib"

echo "=== config and units ==="
install -d -o root -g root -m 0755 /etc/hive-engine
install -m 0640 -o root -g "$USER" "$HERE/ds41.json" /etc/hive-engine/ds41.json
UNIT_TMP="$(mktemp)"
sed -e "s|@MODEL_SRC@|$MODEL|g" \
    -e "s|@ENGRAM1_SRC@|$ENGRAM1|g" \
    -e "s|@ENGRAM14_SRC@|$ENGRAM14|g" \
    "$HERE/ds41-engine.service" > "$UNIT_TMP"
install -m 0644 -o root -g root "$UNIT_TMP" /etc/systemd/system/ds41-engine.service
rm -f "$UNIT_TMP"
install -d -o root -g root -m 0755 /etc/systemd/system/hive.service.d
install -m 0644 -o root -g root "$HERE/hive.service.d/ds41.conf" \
    /etc/systemd/system/hive.service.d/ds41.conf
systemd-analyze verify /etc/systemd/system/ds41-engine.service
systemctl daemon-reload
systemctl enable --now ds41-engine.service

echo "=== engine health ==="
for i in $(seq 1 240); do
    if curl -fsS http://127.0.0.1:8150/health >/tmp/ds41-health.$$ 2>/dev/null; then
        if grep -q '"loaded": true' /tmp/ds41-health.$$; then
            echo "health_after=${i}s: $(cat /tmp/ds41-health.$$)"
            break
        fi
    fi
    sleep 1
done
rm -f /tmp/ds41-health.$$
curl -fsS http://127.0.0.1:8150/health >/dev/null || die "engine health endpoint is not answering"

echo "=== short inference ==="
SHORT="$(curl -fsS -m 240 http://127.0.0.1:8150/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4.1-flash-gsq","messages":[{"role":"user","content":"Reply with exactly DS41_INSTALL_OK"}],"max_tokens":32,"temperature":0}' \
    | /usr/bin/python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("choices") or [{}])[0].get("message",{}).get("content") or "")')"
echo "$SHORT" | grep -q 'DS41_INSTALL_OK' || die "engine answered but did not return DS41_INSTALL_OK: $SHORT"
echo "ENGINE_INFERENCE_OK"

echo "=== Hive ==="
if systemctl is-active --quiet hive.service; then
    systemctl restart hive.service
    for i in $(seq 1 90); do
        if curl -fsS http://127.0.0.1:8765/health >/tmp/hive-health.$$ 2>/dev/null; then
            break
        fi
        sleep 1
    done
    echo "hive_health: $(cat /tmp/hive-health.$$ 2>/dev/null || true)"
    rm -f /tmp/hive-health.$$
else
    echo "hive.service is not active; the ds41 drop-in is installed for its next start"
fi

echo "=== isolation ==="
systemctl show ds41-engine.service -p User -p Group -p MemoryMax -p ProtectHome \
    -p InaccessiblePaths -p BindReadOnlyPaths
PID="$(systemctl show -p MainPID --value ds41-engine.service)"
if [ -n "$PID" ] && [ -r "/proc/$PID/mountinfo" ]; then
    grep -E ' /home | /srv/hive-engine/(model|engram)' "/proc/$PID/mountinfo" | head -8 || true
fi

echo "done: ds41-engine.service is enabled and answering on 127.0.0.1:8150"
echo "note: Hive's DSH route 'hive-local' must point at http://127.0.0.1:8150/v1; see docs/AI-USER.md"
