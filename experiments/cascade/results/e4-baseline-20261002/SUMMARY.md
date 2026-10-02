# E4 baseline - Flash-Next Q2_0 (e4-baseline-20261002)

Engine: `prism engine/ember` @ `6bfd7640c7ae` (version: 0.5.0-dev (build 11244, commit 9cd9e3f57))
Model: `/home/penis/Desktop/work/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`
Model sha256: `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720`
Protocol: docs/EMBER.md E4 baseline protocol (2026-10-02)

| arm | slots | hit rate | cold pp 1K | cold pp 4K | warm pp 1K | decode1 | decode2 | cached | disk MB/gen | RAM min GB | VRAM pk c0/c1 GB | load s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| ncmoe0 | - | - | 214 | 312 | 361 | 25.5 | 25.9 | 28.4 | 11 | 24.8 | 20.5/19.8 | 50.0 |
| ncmoe12-mscauto | 448 | 95.2% | 142 | 285 | 434 | 23.9 | 25.3 | 26.3 | 10 | 26.0 | 19.6/19.0 | 35.0 |
| ncmoe24-msc256 | 256 | 91.3% | 115 | 236 | 350 | 22.9 | 22.7 | - | 13 | 25.6 | 12.4/19.7 | 28.5 |

Caveats:
- single sample per prompt; decode noise ~+-1 t/s on this box
- VRAM/RAM peaks are sampled at 10 Hz, not instantaneous
- disk MB via /proc/<pid>/io read_bytes (includes mmap readahead); sys_read_mb is the whole-NVMe cross-check
- -msc auto picks the slot count from free VRAM at load time
- page-cache state differs across arms; cold prefill is first-request
- desktop session (Xwayland on card1) active during the run
- ncmoe24-msc256: phases skipped: cached
