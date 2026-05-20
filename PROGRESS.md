# Progress & learning report

Living status doc for the HRM-on-TTS experiment. Updated as work lands.
For the narrative chronology see `JOURNAL.md`; for the design rationale
see `README.md`; for GPU/credentials see `docs/COMPUTE.md`.

Last updated: 2026-05-20

---

## Where the experiment stands

| Phase | Status |
| --- | --- |
| Architecture (HRM + matched baseline) | done — shared `Block`, param-matched |
| CPU smoke tests | done — 8/8 pass |
| End-to-end driver (`scripts/run_comparison.py`) | done — multi-seed |
| Modal entrypoint (`modal_train.py`) | done |
| Colab runner (`scripts/colab_train.py`) | done |
| Kaggle CLI metadata (`notebooks/kernel-metadata.json`) | done |
| Unattended run watcher (`scripts/run_and_notify.sh`) | done |
| GitHub repo pushed | done — github.com/harrrshall/hrm-vall-e |
| Modal smoke test (500 steps) | done — pipeline verified green on A100 |
| Architecture PNG (`assets/architecture.png`) | done |
| 1h run (3 seeds, 5000 steps) | done — overfit, not a valid verdict |
| 20h run (3 seeds, train-clean-100) | done — VERDICT IN |

## VERDICT (2026-05-20)

The trustworthy run (20h, 13,155 clips, 3 seeds, no overfitting):

| backbone | best val loss | per-seed |
| --- | --- | --- |
| Baseline | **3.565 +/- 0.003** | 3.563 / 3.569 / 3.563 |
| HRM      | 3.880 +/- 0.056 | 3.818 / 3.868 / 3.953 |

**Baseline wins by 8.84%** — decisive, gap >> seed noise. A clean
negative result: the HRM recurrence trick did not transfer to audio
tokens at this scale. HRM's bp-warmup also costs it ~700 steps of
plateau and it never closes the gap. Full write-up: `results/README.md`.

The 1h run (611 clips) overfit and is not a valid verdict — kept in
`results/1h-dev-clean/` for the record only.

## Credentials — audited, with verification status

| Service | Verified? | How verified | Notes |
| --- | --- | --- | --- |
| GitHub (`gh`) | yes | `gh auth status` | account `harrrshall`, `repo` scope |
| Modal account #1 | yes | `modal app list` returns | profile `harshalsingh1223` |
| Kaggle | yes | `kaggle kernels list --mine` | username `harshalsinghcn` |
| Modal accounts #2 / #3 | no | not configured | only #1 set up; add later with `modal token set` |
| Colab CLI | n/a | — | interactive browser OAuth; user runs `colab new` |

Decisions taken (2026-05-20): use Modal account #1 only for now; GitHub
repo is **public** (so the in-container `git clone` works with no extra
auth); first real run uses **3 seeds per backbone** for noise control.

## Run plan

1. Smoke: `modal run modal_train.py --steps 500 --seeds 0` — DONE,
   passed green on an A100 (took 3 tries; see learnings below).
2. Full: `bash scripts/run_and_notify.sh modal` — 3 seeds x 2 backbones,
   5000 steps, ~1 h of LibriTTS-R dev.clean. A100. RUNNING.
3. Pull results: `modal volume get hrm-vall-e-results /run ./runs/modal`.
4. Read `results.json` → `verdict.winner` and `verdict.exceeds_noise`.

Note on the smoke result: at 500 steps the baseline "won" by ~21%.
That is expected and meaningless — HRM's bp-warmup ramps over the first
20% of training, so a 500-step run is almost all warmup. The 5000-step
run is the fair comparison.

## Learnings (errors hit + how they were resolved)

- **PEP 668 / externally-managed environment** — Debian 12+ blocks
  `pip install` outside a venv. Fix: everything goes in `.venv/`.
- **`\_` in a docstring** triggers a SyntaxWarning on Python 3.12
  (bad escape). Fixed earlier in `src/audio_lm.py`.
- **GitHub username != Kaggle username.** GitHub is `harrrshall`,
  Kaggle is `harshalsinghcn`. The Kaggle `kernel-metadata.json` `id`
  must use the Kaggle name; the `git clone` URLs use the GitHub name.
  Caught by verifying `kaggle kernels list --mine` instead of assuming.
- **Modal credentials are token-pair, not password.** Stored in
  `~/.modal.toml`, one `[profile]` section per account. Multi-account
  switching is `modal profile activate` — verified against Modal docs,
  not guessed.
- **Colab CLI ships file *contents*, not files**, with `colab exec -f`.
  So `scripts/colab_train.py` must be self-contained (it clones the
  repo itself). Verified against the official CLI's docs.
- **Modal `gpu=` is set on the `@app.function` decorator**, not the
  `modal run` CLI. Made it an env var (`MODAL_GPU`) so it stays
  CLI-configurable.
- **LibriTTS-R HF dataset config/split naming** — the first Modal run
  failed in tokenization: `load_dataset("mythicinfinity/libritts_r",
  "dev-clean", split="train")` is wrong. The dataset wants an HF
  *config* (`dev` / `clean` / `other` / `all`) plus a *dotted* split
  name (`dev.clean`, not `dev-clean`). Fixed with a `SPLIT_MAP` in
  `scripts/prepare_libritts.py`. Lesson: verify dataset config/split
  names against the dataset card before a paid GPU run — caught here
  only because the smoke test ran first (cost: a few cents, not $12).

Project rule in force: never guess library APIs / CLI flags / pricing —
web-search and verify. See the `feedback-verify-dont-guess` memory.

## Next actions

- [ ] `git init` + push to public `github.com/harrrshall/hrm-vall-e`.
- [ ] Smoke run on Modal, then the full 3-seed run via `run_and_notify.sh`.
- [ ] Record `results.json` + `loss_curves.png` under `runs/` and commit.
- [ ] Interpret the verdict; decide scale-up vs harder-task per README.
