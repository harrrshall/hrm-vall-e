# hrm-vall-e

> Apply the HRM (hierarchical reasoning) architecture to a VALL-E-style
> audio language model, and compare head-to-head against a vanilla
> transformer with **matched parameter count.**

## Thesis

HRM-Text's claim is that hierarchical recurrence + truncated backprop
yields more quality-per-FLOP than stacking. We test that claim on audio
tokens: same parameter budget, same task (next-token prediction over
EnCodec first-codebook tokens conditioned on a text prefix), different
compute distribution.

| variant   | unique blocks | effective block applications per forward |
| --------- | ------------- | --------------------------------------- |
| Baseline  | 8             | 8                                       |
| HRM       | 8 (4+4)       | 32  (`H=2 × L=3 × n_L=4  +  H=2 × n_H=4`) |

Same params, ~4× the effective depth. The question is whether that
buys lower val loss at convergence.

## Layout

```
hrm-vall-e/
├── README.md              ← you are here
├── PROGRESS.md            ← live status + learnings
├── JOURNAL.md             ← narrative chronology
├── modal_train.py         ← Modal GPU entrypoint
├── src/
│   ├── blocks.py          ← RMSNorm, RoPE, attention, SwiGLU, Block, BlockStack
│   ├── hrm_backbone.py    ← H/L nested loop + input injection + bp_steps
│   ├── baseline_backbone.py
│   ├── audio_lm.py        ← AR audio LM with PrefixLM mask
│   ├── data.py            ← Synthetic + LibriTTS+EnCodec datasets
│   └── train.py           ← CLI trainer (--backbone hrm|baseline)
├── tests/
│   └── test_shapes.py     ← 8 CPU smoke tests
├── scripts/
│   ├── prepare_libritts.py ← tokenize LibriTTS-R with EnCodec
│   ├── run_comparison.py   ← end-to-end multi-seed driver (shared by all)
│   ├── colab_train.py      ← self-contained Colab runner
│   └── run_and_notify.sh   ← unattended run + terminal-state ping
├── docs/
│   └── COMPUTE.md         ← GPU / credentials / account-switching guide
└── notebooks/
    ├── kaggle_train.ipynb     ← end-to-end Kaggle T4×2 run
    └── kernel-metadata.json   ← Kaggle CLI push metadata
```

## Reproduce locally (CPU smoke test)

This just verifies wiring — no audio quality involved.

```bash
cd hrm-vall-e
python3 -m venv .venv
.venv/bin/pip install torch==2.4.* --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install pytest numpy
.venv/bin/python -m pytest tests/ -v
```

All 8 tests should pass. The interesting ones:

- `test_hrm_grad_flow_truncated_bp2` — confirms that with `bp_steps=2`,
  the embedding receives **no** gradient (the surprise from
  `HRM-text/notes/math-walkthrough.md` §6.2).
- `test_hrm_grad_flow_bp5_reaches_embedding` — `bp_steps=5` unlocks it.
- `test_param_count_matches_within_1pct` — confirms HRM and baseline
  have matched parameter counts, so the comparison is fair.
- `test_effective_depth_ratio` — confirms HRM gets 4× effective depth.

You can also smoke-train on synthetic data:

```bash
.venv/bin/python -m src.train --backbone hrm --total-steps 100 --dim 128
.venv/bin/python -m src.train --backbone baseline --total-steps 100 --dim 128
```

Both should drive loss from ~7.1 down. On CPU, HRM is ~4× slower per
step (expected — 4× block applications). The whole point of the
Kaggle run is to see whether that extra compute pays off in val loss.

## Run on a cloud GPU

No local GPU is needed. All three paths call the same driver
(`scripts/run_comparison.py`), which trains both backbones over multiple
seeds and writes `results.json` + `loss_curves.png`. Full setup,
credentials, and Modal multi-account switching are in
[`docs/COMPUTE.md`](docs/COMPUTE.md).

```bash
# Modal (recommended — A100, ~$2-4 of free credit per run)
.venv/bin/modal run modal_train.py --steps 500 --seeds 0   # smoke test first
bash scripts/run_and_notify.sh modal                       # full 3-seed run

# Kaggle (free T4×2)
cd notebooks && kaggle kernels push

# Colab (free T4)
colab new -s hrm --gpu T4 && colab exec -s hrm -f scripts/colab_train.py
```

`scripts/run_and_notify.sh` streams logs to `runs/` and prints a single
`EXPERIMENT_OK` / `EXPERIMENT_FAILED` line on exit — launch it in the
background and walk away.

Outputs land in the Modal Volume `hrm-vall-e-results` (or `runs/` for
Kaggle/Colab/local): `results.json` with the per-seed and aggregate
verdict, plus `loss_curves.png`.

## What to report

The single headline number is **final val loss with matched params and
matched optimizer / data budget.** If HRM lands materially lower, the
recurrence-instead-of-stacking trade transfers from text to audio.

Secondary readouts that are worth eyeballing:

- **val loss vs wall-clock time** — HRM may *lose* on a wall-clock
  budget even though it wins on a parameter budget. Both are valid
  framings; the paper's compute-efficiency claim is per-parameter.
- **the bp_steps ramp** — log `bp_steps` per step; you should see a
  visible inflection in the HRM curve when bp_steps hits 5
  (embedding starts learning).
- **prefix vs response loss** — split val loss by token type.

## Scaling up

The default config is intentionally tiny (~10M params, 1h of audio,
5000 steps) so it fits in Kaggle's 12h session limit comfortably.
To approach research-relevant scale:

- bump `dim` to 512, `n_h=n_l=6` → ~30M params
- bump `--hours 5.0` → ~5h of audio, batch the prepare script across
  multiple Kaggle sessions
- raise `total_steps` to 20–30k

For runs above ~30M params or above 10h of audio, free Colab T4 won't
fit easily and Kaggle T4×2 won't finish in 12h. That's the threshold
where you should burn the $30 Modal H100 credit for one decisive run.

## Where the architecture lives

The HRM block is `src/hrm_backbone.py`. It is a direct port of
`HRM-text/models/baselines/hrm_nocarry_bp_warmup.py`, simplified
(no FSDP, no flash-attention dependency, no gated attention — those
are orthogonal to the architectural delta under test).

The math walkthrough that explains the recurrence and BP warmup is at
`HRM-text/notes/math-walkthrough.md` and the smoke tests in this repo
double as executable assertions of its key claims.
