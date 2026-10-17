# A2 vision oracle — pilot stage 0

Measured on CPU beside the resident 176B serve (port 8081), 2026-10-17.
Role A2 ("vision encoding", `harness/cascade/roles.py`). Candidates are
the three registry rows (`harness/cascade/registry.py`).

Each candidate ran one fixed six-image eval set — text card, stop sign,
shapes scene, count scene, bar chart, real photo — with programmatic
checks, one call per image, three timed repetitions after a warmup call.
Quality = checks passed / 6. Latency = median of per-image medians, warm
(processor + forward + decode included), milliseconds per call.

| candidate | quality | median call | load | weights (fp32) | peak RSS |
|---|---|---|---|---|---|
| florence-2-base | 4/6 | 800.2 ms | 1.92 s | 0.926 GB | 1.73 GiB |
| smolvlm2-256m | 5/6 | 4312.6 ms | 3.48 s | 1.026 GB | 1.97 GiB |
| altclip | 6/6 | 250.7 ms | 3.16 s | 3.457 GB | 3.05 GiB |

## Per-image outputs

### florence-2-base (microsoft/Florence-2-base)

- **text_card** `<OCR>` — pass (OCR output contains '42'; [851.1, 824.9, 830.4] ms) — `</s><s>HIVE CASCADE 42stage-0 vision oracle</s>`
- **sign_stop** `<OCR>` — pass (OCR output contains 'STOP'; [711.7, 719.0, 707.0] ms) — `</s><s>STOP</s>`
- **shapes_scene** `<OD>` — FAIL (OD labels >= 2, incl. a shape word; [765.3, 774.1, 770.1] ms) — `</s><s>christmas tree<loc_372><loc_480><loc_630><loc_850></s>`
- **count_squares** `<OD>` — FAIL (OD labels >= 2 'square'; [847.3, 842.6, 842.0] ms) — `</s><s>flag<loc_411><loc_133><loc_579><loc_387><loc_736><loc_133><loc_905><loc_387><loc_86><loc_133><loc_254><loc_387></s>`
- **chart_bars** `<OCR>` — pass (OCR reads labels A, B and C; [727.7, 722.6, 722.6] ms) — `</s><s>523ABC</s>`
- **photo_planet** `<MORE_DETAILED_CAPTION>` — pass (caption has >= 4 words; [1267.0, 1236.4, 1232.6] ms) — `</s><s>The image is a digital illustration of a planet in outer space. The planet is in the center of the image, with a blue and green co...`

### smolvlm2-256m (HuggingFaceTB/SmolVLM2-256M-Video-Instruct)

- **text_card** `What text is written in this image? Answer briefly.` — pass (answer contains '42'; [4689.0, 4554.3, 4525.7] ms) — `HIVE CASCADE 42 stage-0 vision oracle.`
- **sign_stop** `What does the sign say?` — pass (answer contains 'stop'; [4301.6, 4388.5, 4361.1] ms) — `The sign says "STOP".`
- **shapes_scene** `What color is the circle in this image? Answer briefly.` — pass (answer contains 'red'; [4179.2, 4297.7, 4253.8] ms) — `Red.`
- **count_squares** `How many blue squares are there? Answer with just the number.` — pass (answer contains 3/'three'; [4269.2, 4264.2, 4211.6] ms) — `3`
- **chart_bars** `Which bar is the tallest, A, B or C? Answer with one letter.` — FAIL (answer names 'B'; [4191.2, 4209.6, 4165.8] ms) — `A.`
- **photo_planet** `Describe this image in one sentence.` — pass (answer has >= 4 words; [4489.3, 4595.3, 4574.5] ms) — `A space scene with a planet and a logo in the center.`

### altclip (BAAI/AltCLIP)

- **text_card** `top-1 of 3 text options` — pass (top-1 is option 0 ('an image containing the text HIVE CASCADE 42'); [252.8, 249.1, 250.2] ms) — `an image containing the text HIVE CASCADE 42`
- **sign_stop** `top-1 of 3 text options` — pass (top-1 is option 0 ('a red stop sign'); [235.0, 234.2, 234.0] ms) — `a red stop sign`
- **shapes_scene** `top-1 of 3 text options` — pass (top-1 is option 0 ('a red circle and a blue square and a green triangle'); [254.6, 253.3, 252.5] ms) — `a red circle and a blue square and a green triangle`
- **count_squares** `top-1 of 3 text options` — pass (top-1 is option 0 ('an image with three blue squares'); [243.6, 243.4, 245.0] ms) — `an image with three blue squares`
- **chart_bars** `top-1 of 3 text options` — pass (top-1 is option 0 ('a bar chart with three bars labeled A, B and C'); [253.6, 256.0, 255.3] ms) — `a bar chart with three bars labeled A, B and C`
- **photo_planet** `top-1 of 3 text options` — pass (top-1 is option 0 ('a planet in space surrounded by a green nebula'); [251.2, 248.9, 253.7] ms) — `a planet in space surrounded by a green nebula`

## Frontier (role=A2, unit=call)

Frontier after stage 0 — non-dominated rows for `A2`:

- **altclip** — quality 1.0, 250.7 ms/call, 3.457 GB, `fp32`

Dominated (kept for the record): florence-2-base, smolvlm2-256m.

## Reading

The three candidates do different jobs under one role id — worth splitting when
P6 gets wired:

- **altclip** is the only true A2 encoder (image → vector): 6/6 on the option
  sets, 0.25 s/call, the frontier survivor. Largest footprint (3.46 GB fp32
  ≈ 1.7 GB fp16). Prompt-sensitive on real photos (an ambiguous "contains
  text" distractor once outranked the scene option — the wallpaper carries a
  wordmark).
- **florence-2-base** is an OCR/region specialist, not a captioner first:
  perfect OCR (text card, sign, and the chart's gluesticked values+labels
  `523ABC`), a solid detailed caption of the photo, but open-vocabulary OD
  labels go sideways on synthetic scenes ("christmas tree" for a green
  triangle; three blue squares read as "flag" ×3 — the boxes were right).
  ~0.8 s/call. Its remote code pins a legacy env (transformers 4.46.3).
- **smolvlm2-256m** is a mini captioner/QA model: 5/6, the only miss was the
  tallest-bar question ("A"). Its photo caption names both content and the
  wordmark: "A space scene with a planet and a logo in the center." ~4.3 s/call
  on CPU — the natural candidate to move to an accelerator tier if caption
  latency ever matters.

Recommended placement (all in RAM/CPU — the dGPUs stay the 176B's):
`altclip` resident as the A2 encoder; `florence-2-base` for OCR-bearing P6
variants; `smolvlm2-256m` for natural-language scene descriptions. Total RAM if
all three are resident: 5.4 GB fp32 (≈ 2.7 GB at fp16), leaving the 176B's
page-cache headroom intact (MemAvailable min 19.2 GiB during the run).

Two check formulations were corrected before the final run and are recorded in
`vision_oracle.py`: Florence's chart OCR glues labels together (`523ABC`), so
the check now asks that all three letters appear; AltCLIP's original photo
distractor ("contains text HIVE CASCADE 42") was ambiguously true for the
wallpaper and was replaced with clearly-false options.

Caveats: six images, synthetic-heavy, one real photo; fp32 on CPU only (this
box exposes no iGPU); single session. `smolvlm2-256m` and `altclip` ran in the
harness venv; `florence-2-base` in the pinned side env.

## Environment

- Host: `Penis-machine`, all candidates CPU-only (torch CPU builds;
  no ROCm context beside the serve — see module docstring for why).
- Florence-2-base ran in the pinned side env `.envs/florence-legacy` (transformers 4.46.3): its remote code does not
  import under the harness venv's transformers 5.x (flash-attn rewrite
  leaves an empty `if` body → IndentationError), and the native 5.x
  Florence-2 classes expect converted weights this hub revision lacks.
- smolvlm2-256m and altclip are native in the harness venv (transformers 5.17.0).

## Coexistence with the live 176B serve

- `/health` before: 200, after: 200 (serve stayed up).
- MemAvailable across the run: min 19.24 GiB (see `memwatch.log`); total run 118.9 s.

## Artifacts

- `measurements.json` — the `CandidateMeasurement` rows (also copied to
  `experiments/cascade/a2-oracle-stage0.json` for `/v1/cascade/frontier`).
- `<candidate>.json` — full per-candidate records (raw outputs, latencies).
- `images/` — the eval set + sha256 manifest.
- `memwatch.log` — host MemAvailable samples during the run.
