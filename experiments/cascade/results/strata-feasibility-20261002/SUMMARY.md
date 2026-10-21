# Strata feasibility - strata-feasibility-20261002

Engine: `<work>/strata` @ `db4f91a1171d`
Model: `<work>/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` sha `69820c02ec7d0b45…`
Flags: `--pack <work>/strata-data/packs/q2_0 --native <work>/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --ple-gguf <work>/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf --expert-profile <work>/strata/data/expert-profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp <work>/strata-data/mtp/rt --max-context 4096 --mmap-experts --vram-reserve-mib 3072`

| phase | wall s | prompt t | reused | prefill t/s | generated | decode t/s | finish |
|---|---:|---:|---:|---:|---:|---:|---|
| cold1k | 8.12 | 935.0 | 0.0 | 124.6 | 16.0 | 27.4 | length |
| cold4k | 7.98 | 3094.0 | 0.0 | 406.7 | 16.0 | 47.9 | length |
| warm1k | 2.06 | 935.0 | 0.0 | 523.5 | 16.0 | 63.4 | length |
| decode1 | 5.24 | 178.0 | 0.0 | 251.7 | 170.0 | 37.6 | stop |
| decode2 | 3.06 | 178.0 | 171.0 | 108.4 | 170.0 | 56.8 | stop |
| cached | 3.04 | 178.0 | 171.0 | 96.5 | 170.0 | 57.3 | stop |

RAM available min: 26.9 GB
VRAM peak: card0 18.5 GB, card1 19.2 GB

Caveats:
- single sample per prompt
- Strata's engine log timings are authoritative; API wall time includes HTTP
- no ignore_eos on the API - decode may stop at EOS before max_tokens
- system-wide 10 Hz sampling (engine runs under the serve process)
