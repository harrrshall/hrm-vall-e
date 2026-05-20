# Experiment Design V2 — a definitive test of HRM for TTS

V1 of this repo produced a negative result (a stacked transformer beat
HRM by 8.84% on first-codebook val loss at 15M params). Reading the
HRM-Text paper afterwards showed V1 answered a *narrow* question with a
*stripped-down* HRM at a *tiny* scale on a *proxy* metric. V2 is designed
so the outcome — win or lose — is **definitive, citable, and about real
text-to-speech**.

This is a pre-registered design: the hypotheses, metrics, and decision
rules below are fixed *before* running. No moving goalposts.

---

## 0. What V1 got wrong → what V2 fixes

| V1 flaw | Why it mattered | V2 fix |
| --- | --- | --- |
| 15M params (66× below the paper's 1B) | HRM's stabilisation was designed at ~1B; recurrence may need capacity | ~100M backbone |
| Stripped HRM — no MagicNorm, no gated attention | tested a strawman, not HRM-Text | faithful HRM-Text architecture |
| Proxy metric: 1st-codebook val loss | not "a TTS result" — never produced audio | real metrics: WER, speaker SIM, UTMOS |
| Single codebook | output isn't intelligible speech | full multi-codebook (delay pattern) |
| Parameter-matched only | the paper's claim is FLOPs-matched | **both** axes tested |
| 5000 steps; bp-warmup = 14-20% of budget | warmup taxed HRM disproportionately | budget long enough to amortise warmup |
| generic "is HRM better?" | no mechanism, no insight on a tie | targeted hypothesis: H-module as prosody planner |

---

## 1. Hypotheses (pre-registered)

- **H1 — primary.** At *matched training FLOPs*, an HRM backbone yields
  lower word error rate (WER) than a standard transformer on LibriTTS
  test-clean continuation.
- **H2 — parameter efficiency.** At *matched parameters*, HRM yields
  lower WER. (V1's question, redone properly.)
- **H3 — the planner hypothesis.** Any HRM advantage *grows with
  utterance length* and shows up most in *prosody* — because the slow
  H-module maps onto prosodic planning and the fast L-module onto
  local acoustic detail. This is the one place HRM's structure has a
  principled reason to help TTS.
- **H4 — test-time compute.** Raising HRM's recurrence cycles at
  inference (beyond the trained count) lowers WER — recurrence as an
  adaptive-compute knob, the property a stacked transformer cannot offer.

### Decision rules (committed in advance)

- **HRM wins** if it beats the FLOPs-matched transformer on WER by more
  than the combined 3-seed standard deviation, *and* does not regress
  speaker SIM or UTMOS.
- **Parameter-efficiency win only** if it beats param-matched but not
  FLOPs-matched.
- **Negative result** otherwise.
- H3 and H4 are exploratory — reported either way, not pass/fail gates.

---

## 2. The TTS system under test

Zero-shot TTS as an **autoregressive codec language model** (the VALL-E
family — the one architecture where HRM swaps in line-for-line).

- **Codec:** Mimi (~12.5 Hz frame rate, ~8 codebooks) — built for LMs and
  ~6× fewer tokens than EnCodec 24 kHz (75 Hz), so ~6× cheaper to train.
  EnCodec 24 kHz is the conservative fallback. *Confirm the exact frame
  rate / codebook count from the codec card before implementing.*
- **Multi-codebook:** MusicGen-style **delay pattern** → one token
  stream, one backbone. No separate NAR head — keeps the comparison to a
  single transformer-vs-HRM swap.
- **Conditioning:** phoneme (or byte) text prefix + a 3 s acoustic
  prompt; **PrefixLM** mask (bidirectional prefix, causal response) —
  same as HRM-Text and as V1.
- **The backbone is the only thing swapped.** Embeddings, codec, delay
  pattern, mask, head, loss — all identical across arms.

---

## 3. The three arms

Identical recipe, data, optimiser, schedule — only the backbone differs.

1. **HRM** — *faithful* HRM-Text: H/L dual-timescale recurrence,
   **MagicNorm**, **sigmoid-gated attention**, warmup deep credit
   assignment, Adam-atan2, parameterless RMSNorm, RoPE, SwiGLU. ~100M
   params. H2L3 cycle structure.
2. **Transformer — parameter-matched.** Standard pre-norm transformer,
   same ~100M params, sharing the same Block (RMSNorm, RoPE, SwiGLU,
   gated attention) so even attention matches. Same step budget.
3. **Transformer — FLOPs-matched.** Arm 2 trained until it has consumed
   the same *total training FLOPs* as HRM (HRM does ~4× compute per
   forward → run it ~4× longer). Checkpoint at 1× FLOPs to also serve as
   arm 2 — one long run yields both transformer points.

Faithful HRM is non-negotiable: V2 tests HRM-Text, not a simplification.

---

## 4. Held constant (the controls)

Codec & delay pattern · text frontend · PrefixLM mask · dataset & split ·
tokenisation · batch size (in tokens) · LR peak & schedule shape ·
weight decay · grad-clip policy · precision · eval protocol · seeds.
Optimiser family is matched (Adam-atan2 for both, or AdamW for both —
fixed before running, not per-arm).

---

## 5. Data

**LibriTTS-R** — the standard clean TTS benchmark.
- Train: `train-clean-360` (~360 h). Budget option: `train-clean-100`.
- Eval: held-out `test-clean`, **speaker-disjoint** from train.
- Same Mimi tokenisation for every arm.

---

## 6. Evaluation protocol

VALL-E **continuation** protocol on test-clean (3 s prompt → synthesise
the rest):

- **WER / CER — primary.** Transcribe generated audio with
  Whisper-large-v3, compare to reference text. Intelligibility.
- **Speaker SIM.** Cosine similarity of WavLM-based speaker-verification
  embeddings, generated vs prompt speaker.
- **UTMOS.** Predicted mean opinion score — perceptual quality.
- **Val NLL.** Training-curve view (the only V1 metric; kept, demoted).
- **Length-bucketed.** Every metric split by target length
  (short / medium / long) — directly tests **H3**.
- **Hard set.** Sentences with repeated words, numbers, tongue-twisters
  — codec-LM TTS is known to fail there; a sharp discriminator.

Listening samples for both arms are published so the result is audible,
not just tabular.

---

## 7. Ablations & mechanism probes

- **Test-time cycle scaling (H4):** evaluate HRM at trained cycles
  {−2, 0, +2, +4}; plot WER vs inference compute.
- **Component ablation:** HRM − MagicNorm and HRM − gated-attention, to
  attribute any gap to a specific piece of the codesign.
- **H/L prosody probe (H3):** linear probe from the H-state and L-state
  to F0 / energy / phone-duration. If H predicts prosody better than L,
  the planner story holds.
- **Cycle sweep:** H1L2 vs H2L3 vs H3L4 — does more recurrence help TTS
  at all?

---

## 8. Statistics

3 seeds per arm. Report mean ± std for every metric. Primary test:
HRM vs FLOPs-matched transformer on WER, gap judged against the combined
seed std (the "exceeds noise" rule from V1's `run_comparison.py`).
Metrics and the primary test are fixed in advance — no post-hoc switching.

---

## 9. Compute plan (budget-aware, phased)

Mimi's ~12.5 Hz frame rate is the key enabler — ~6× fewer tokens than
EnCodec means ~6× cheaper runs. Compute: Modal (≈$90 across the user's
2-3 accounts) + free Kaggle T4×2.

- **Phase 0 — Pilot / de-risk.** 1 seed, transformer only,
  `train-clean-100`, ~half steps. **Gate:** the recipe must produce
  intelligible speech (baseline WER below ~40%). If not, scale data or
  model *before* spending on the comparison. ≈$10.
- **Phase 1 — Core comparison.** HRM ×3 seeds + Transformer ×3 seeds
  (long runs → both param- and FLOPs-matched points), `train-clean-360`.
  Spread across Modal accounts + Kaggle. The bulk of the budget.
- **Phase 2 — Ablations & probes.** Smaller, cheaper runs; Kaggle T4×2.

Estimated total: **gold standard ≈ $80-150** (3 Modal accounts + free
Kaggle); **budget version ≈ $40-60** (`train-clean-100`, 2 seeds,
~60M backbone). Phase 0 gates whether to commit to Phase 1.

---

## 10. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| From-scratch TTS not intelligible on a hobby budget | Phase 0 gate; fall back to smaller scope + report on val NLL with an explicit "intelligibility ceiling" caveat |
| HRM instability at the new scale | MagicNorm + warmup credit assignment are in (the paper's stabilisation is designed exactly for this); monitor grad norms |
| Compute overrun | phased plan; budget version pre-defined; checkpoint-and-resume |
| HRM simply not suited to TTS | that *is* a real result — H3/H4 still extract *where* (if anywhere) it helps |

---

## 11. Why this design yields a "real result"

1. **Real TTS metrics** — WER, SIM, UTMOS, plus audio you can hear.
2. **Both comparison axes** — parameter- and FLOPs-matched.
3. **A faithful HRM** — HRM-Text as published, not a simplification.
4. **Pre-registered, seeded, significance-tested** — no p-hacking.
5. **A mechanism hypothesis** (H-module = prosody planner), so even a
   tie produces insight, not a shrug.

Whatever the outcome, it is citable: *"HRM does / does not help
text-to-speech — here is the controlled, faithful, audible evidence."*

---

## 12. Build order

1. `codec.py` — Mimi wrapper + delay-pattern pack/unpack.
2. `hrm_faithful.py` — MagicNorm, gated attention, Adam-atan2 (extend
   the V1 backbone).
3. `tts_lm.py` — multi-codebook codec LM with PrefixLM.
4. `eval/` — Whisper-WER, speaker-SIM, UTMOS harness.
5. `scripts/run_v2.py` — the 3-arm driver (extends `run_comparison.py`).
6. Phase 0 pilot → gate → Phase 1 → Phase 2.
