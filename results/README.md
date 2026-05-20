# Results

Head-to-head: HRM hierarchical-recurrent backbone vs a matched-parameter
stacked transformer, on next-token prediction over EnCodec speech tokens.

## Headline

**At ~15M parameters, the stacked transformer baseline beats HRM by
~9% validation loss.** The result is decisive — the gap is far larger
than the seed-to-seed noise. The HRM recurrence trick, as ported here,
did **not** transfer to audio tokens at this scale.

## The trustworthy run — 20h, train-clean-100

`20h-train-clean-100/` — 3 seeds, 5000 steps, 13,155 clips
(11,839 train / 1,316 val). Enough data that neither model overfits
(final ≈ best val loss for both).

| backbone | params | best val loss (mean ± std) | per-seed |
| --- | --- | --- | --- |
| Baseline | 15,148,800 | **3.565 ± 0.003** | 3.563 / 3.569 / 3.563 |
| HRM      | 15,149,184 | 3.880 ± 0.056 | 3.818 / 3.868 / 3.953 |

→ **Baseline wins by 8.84%.** Gap 0.315 ≫ combined seed noise 0.058.

Two things stand out in `loss_curves.png`:

1. **HRM's bp-warmup costs it ~700 steps.** HRM's loss sits on a
   plateau (~5.9) until `bp_steps` ramps up, then drops sharply — the
   embedding-gradient-unlock effect. It then converges fast but never
   closes the gap to the baseline within the 5000-step budget.
2. **HRM is far less stable.** Baseline seeds land within 0.006 of
   each other; HRM seeds spread over 0.135.

## The flawed run — 1h, dev-clean (kept for the record)

`1h-dev-clean/` — only 611 clips. Both models overfit hard
(train loss → ~0, val loss climbs). Its `final_val_loss` verdict
("HRM wins 44%") is a **memorisation artefact** — HRM merely overfit
less. On `best_val_loss` the two were within ~1%. This run is the
reason the verdict metric was switched to best-val and the experiment
re-run with 20× the data. Not a valid result; kept only for honesty.

## Honest interpretation

This is a clean **negative result** for HRM-on-TTS — within scope:

- one scale (~15M params), one step budget (5000), single EnCodec
  codebook, and the gated-attention variant deliberately omitted.
- HRM's bp-warmup plateau eats ~14% of a short training budget; a much
  longer run might let the recurrence pay off. HRM's published wins are
  on puzzle/reasoning tasks and emphasise test-time compute scaling —
  neither of which this setup exercises.

So the honest claim is narrow: **at this scale and budget, stacking
beats looping on audio tokens.** Not "HRM never helps for speech."
Worth-testing follow-ups: longer schedules, larger scale, restoring
gated attention, sweeping H/L cycle counts, and inference-time cycle
scaling.
