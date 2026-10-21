# Strata feasibility - scion-strata-20261009

Engine: `<work>/strata` @ `db4f91a1171d`
Model: `<work>/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` sha `69820c02ec7d0b45…`
Flags: ``

| phase | wall s | prompt t | reused | prefill t/s | generated | decode t/s | finish |
|---|---:|---:|---:|---:|---:|---:|---|
| cold1k | 76.76 | 935.0 | 0.0 | 12.7 | 16.0 | 4.9 | length |
| cold4k | 251.58 | 3094.0 | 0.0 | 12.6 | 16.0 | 2.9 | length |
| warm1k | 76.05 | 935.0 | 0.0 | 12.7 | 16.0 | 6.7 | length |
| decode1 | 41.93 | 178.0 | 0.0 | 8.6 | 169.0 | 8.1 | stop |
| decode2 | 13.84 | 178.0 | 171.0 | 19.9 | 170.0 | 12.6 | stop |
| cached | 6.12 | 178.0 | 171.0 | 90.4 | 169.0 | 28.3 | stop |

RAM available min: 6.1 GB
VRAM peak: card0 18.6 GB, card1 20.3 GB

Caveats:
- single sample per prompt
- Strata's engine log timings are authoritative; API wall time includes HTTP
- no ignore_eos on the API - decode may stop at EOS before max_tokens
- system-wide 10 Hz sampling (engine runs under the serve process)
