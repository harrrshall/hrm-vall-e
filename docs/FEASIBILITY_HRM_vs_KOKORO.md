# Feasibility memo — can an HRM-technique TTS model beat Kokoro-82M?

**Question.** Train a TTS model using the HRM technique to beat
Kokoro-82M on the benchmark. Is it feasible? What is the honest
probability? Should we run the experiment?

## TL;DR — honest verdict

**Probability of actually beating Kokoro: ~5% (range 3-10%). Probability
of merely being competitive: ~15-20%.**

**Recommendation: do NOT run this as "beat Kokoro with HRM."** Not
because the project is bad — because the framing fights on the wrong
architecture, against negative evidence we already have, for a goal HRM
is not the right tool for. There is a high-value experiment here; it is
not this one. The honest version is at the end.

## What Kokoro actually is

Kokoro-82M is **non-autoregressive**, based on **StyleTTS 2 + an iSTFTNet
vocoder**. It is a *modular pipeline* of small specialised networks:

- a phoneme-level BERT text encoder (PL-BERT),
- a style encoder (prosody control),
- a duration predictor (predicts per-phoneme length, builds alignment),
- a decoder + iSTFTNet vocoder (produces the waveform in one forward pass),
- a WavLM discriminator (used only for adversarial training).

It is trained in **two stages** (acoustic reconstruction, then the TTS
prediction modules with style diffusion + GAN training), on a few
hundred hours of curated permissive audio, for ~1000 A100-hours (~$1000).
It is a mature, carefully tuned system.

## What the HRM technique actually is

HRM is a **recurrent backbone replacement for the transformer inside an
autoregressive next-token language model** — a slow H-module (planner)
and fast L-module (worker), nested loop, input injection, truncated-BPTT
warmup, PrefixLM mask. Every HRM-Text benchmark win is on **reasoning /
knowledge** text (MMLU, GSM8K, MATH, ARC).

## The core problem: HRM does not fit Kokoro's architecture

**Kokoro has no autoregressive transformer backbone to swap HRM into.**
It is a NAR pipeline of small specialised nets. HRM is a technique *for
AR next-token LMs*. There is no natural slot for it in a StyleTTS-2-style
model.

So "use HRM to beat Kokoro" really means: build a **different kind of
TTS model** — an autoregressive codec language model (the VALL-E family)
with an HRM backbone — and have *that* beat Kokoro. That reframing is
where the odds collapse.

## Why beating Kokoro this way is uphill — four concrete reasons

1. **Wrong architecture family.** Kokoro (NAR / StyleTTS2 / GAN) already
   *beat* the AR codec-LM family head-to-head — XTTS v2 (467M), Fish
   Speech (~500M), MetaVoice (1.2B). AR codec LMs are slower, less stable
   (repeats / skips / mis-alignments), and rated lower for naturalness.
   An HRM-TTS model would be in the family Kokoro already won against.
2. **Direct negative evidence — our own V1.** We already tested HRM vs a
   plain transformer on audio tokens. HRM **lost by 8.84%**. It does not
   even beat a vanilla transformer within the AR family at our scale.
3. **HRM's advantage is reasoning-specific.** Its wins are all on
   deliberation / search tasks. TTS is signal generation, not reasoning.
   The mechanism that makes HRM good has no obvious purchase on
   codec-token prediction.
4. **Kokoro is mature and well-tuned.** $1000, 1000 A100-hours, two-stage
   GAN training, curated data, careful G2P. Beating it from a hobby
   budget is hard for *any* method, let alone one carrying handicaps 1-3.

## Probability breakdown (honest)

| outcome | probability |
| --- | --- |
| Build a working ~82M AR codec-LM TTS at all | ~60% |
| ...*competitive* with Kokoro (same ballpark) | ~15-20% |
| ...actually *beats* Kokoro on the benchmark | ~3-10% |
| ...where HRM (vs a plain transformer) is the decisive reason | ~2-4% |

Headline: **~5% to beat Kokoro. ~15-20% to be competitive. HRM
specifically being the edge: low single digits.**

## Should we run this experiment?

**No — not under the framing "beat Kokoro with HRM."** It is a ~5%
moonshot, on the architecture family Kokoro already beat, contradicted by
our own V1 result. Staking the budget on it is negative expected value.

## What IS worth doing instead

1. **The clean science (recommended).** Run V2 as designed —
   `docs/EXPERIMENT_DESIGN_V2.md` — but keep the *honest* success
   criterion: **"does HRM beat a matched transformer?"**, not "does it
   beat Kokoro?". The outcome either way is a real, citable result, for
   ~$80. (Current best guess from V1 + the paper analysis: HRM probably
   loses or ties — but proving it cleanly is the contribution.)
2. **If you want a real "win", drop HRM — not the ambition.**
   Tiny-beats-big in TTS is real (Kokoro itself proves it). But the
   levers are NAR architecture + frozen pretrained codec/vocoder +
   distillation from a big teacher + clean data — **not** HRM recurrence.
   That is a separate, legitimate project with far better odds.
3. **The one HRM-flavoured angle with a principled fit.** Hierarchical
   slow/fast recurrence inside the *prosody / duration predictor* of a
   NAR model — prosody genuinely is hierarchical planning (slow
   intonation contour, fast acoustic detail). That is a genuine research
   probe. But it is exploratory, it is not "the HRM technique", and it is
   not a Kokoro-beating play.

## Bottom line

It is **feasible** to *build* an HRM-based TTS model. It is **not likely**
(~5%) to **beat Kokoro**. The honest move: run V2 as science (HRM vs a
matched transformer), report the truth, and do not stake the project on
out-SOTA-ing a mature, well-tuned system with a tool built for a
different problem. Beating Kokoro is a worthy goal — it just should not
be pursued with HRM.
