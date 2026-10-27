# DS4.1 engine service (durable)

This bundle reproduces the local DeepSeek-V4.1-Flash-GSQ engine on the Hive
box. It backs Hive's `hive-local` DSH route at `http://127.0.0.1:8150/v1`.

## Files

- `ds41.json` - the exact engine config: chunk 1024, `STRATA_SERVE_STEP_MAX=0`,
  wave 16/16, `--prompt-batch 8192`, `--arena-gib 4`, `--pf-b 0`, D3 default,
  no DSpark.
- `ds41-engine.service` - the systemd unit template. `install.sh` substitutes
  `@MODEL_SRC@`, `@ENGRAM1_SRC@` and `@ENGRAM14_SRC@`.
- `hive.service.d/ds41.conf` - pins Hive to `hive-local` /
  `deepseek-v4.1-flash-gsq` and orders Hive after the engine.
- `install.sh` - root installer: account, app/binary/libs, read-only model and
  Engram bind mounts, config, unit, drop-in, restart and verification.
- `../../docs/DS41-HF-POST.md` - the public write-up with the measured numbers.

## Install

The current deployment is already installed. This is for recovery or a
rebuild:

```bash
sudo ./deploy/ds41-engine/install.sh
```

The defaults are this box's media paths. Override them if the model or the
engine checkout moves:

```bash
sudo SRC=/path/to/strata-ds41 \
     MODEL=/path/to/models-p310/V41-GSQ \
     ENGRAM1=/path/to/v41-engram/layers1 \
     ENGRAM14=/path/to/v41-engram/layers14 \
     LIBSRC=/path/to/prism-qwen4exp-shrink/build-hip/bin \
     ./deploy/ds41-engine/install.sh
```

The installer expects the DSH route to exist already. See
[`docs/AI-USER.md`](../../docs/AI-USER.md) for the `hive-local` route, the
`hive-ai` account and the rootless Docker setup.

## Measured (2026-10-27)

| workload | result |
|---|---|
| 6,110-token prefill | 51.7 s, 126.5 tok/s (cold arms 117.8-121.3) |
| 128-token decode at 6.1k | 16.2 tok/s |
| 648-token new turn | 23.3 s |
| 341-token new turn | 24.1 s |
| 73-token new turn | 18.0 s |
| 16-token follow-up on the 6.1k prefix | 7.6 s (6,115 tokens cached) |
| Hive agent, 5,734-token prompt | 67 s, 85.6 tok/s; 128 thinking at 17.6 tok/s |
| Hive tool-result follow-up | 5 s prefix reuse + 139 tokens at 15.1 tok/s |

The box: 1x RX 7900 XT 20 GB (spare GPU), Ryzen 7 7800X3D, 31.7 GiB DDR4,
152 GB V41-GSQ on a P310. The model and both Engram layers are read-only bind
mounts into `/srv/hive-engine`.

## Why these settings

- **`STRATA_SERVE_STEP_MAX=0`** is the big one. With a 4 GiB arena the exact
  pass path is only about 2.3 tok/s, so every prompt uses the exact
  layer-major batch. That took 341/648-token tool turns from 129/256 s to
  24/23 s; the 6.1k prefill stayed at about 52 s.
- **`--prefill-chunk 1024`** is the largest chunk that handles the 6.1k prompt
  on the 20 GB card. 2,048 returns 503; 4,096 fails a 17.5 GiB dense graph
  allocation. At 1,024 the 648-token turn is one dense/expert chunk instead
  of two.
- **Wave 16/16** (`STRATA_WAVE_THREADS=16`, `STRATA_WAVE_PARTS=16`) added
  3-7% over the 117.8 baseline; the final confirmation was 126.5.
- **`--pf-b 1.43` regressed** to 99.3 tok/s prefill / 11.2 decode.
  **`STEP_MAX=20`** helped the 16-token turn but dropped the 128-token decode
  to 12.4. **D3 off** was neutral. Left them off.
- **DSpark** is off: this HIP build returns
  `DSpark step: k out of range` on every request.

## Isolation

`ds41-engine.service` runs as the dedicated `hive-engine` account, with
`MemoryMax=12G`, `MemoryHigh=9G`, `ProtectSystem=strict`, `ProtectHome=yes`,
`InaccessiblePaths=/home`, no capabilities, and only `/srv/hive-engine`
writable. The model and Engram sources are read-only bind mounts. The engine
user is not in the docker group and has no docker socket. The API binds only
`127.0.0.1` and has no API key; do not expose it beyond localhost.

Backups from the tuning runs:
`/etc/hive-engine/ds41.json.pre-tune2` and `.pre-chunk1024`.
The service log is in the journal; the engine's own log is
`/srv/hive-engine/logs/engine.log`.

## Links

- [Engine notes](../../../strata-ds41/docs/ds4/V41_START_HERE.md) (local checkout)
- [Hive deployment notes](../../docs/AI-USER.md)
- [HF post](../../docs/DS41-HF-POST.md)
- [Model](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)
- [Engine](https://github.com/Indras-Mirror/Strata-DSV4.1)
