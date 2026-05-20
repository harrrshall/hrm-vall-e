# Compute guide — running the experiment on a GPU

No local GPU on this machine. This is how to run the HRM-vs-baseline
comparison on the cloud, ranked by what to reach for first.

| path | GPU | cost | best for |
| --- | --- | --- | --- |
| **Modal** | A100 / H100 | ~$2-4 of free credit per run | the real comparison, fast |
| **Kaggle** | T4 ×2 | free (30 h/week) | a free fallback, slower |
| **Colab** | T4 | free (~15-30 h/week) | quick one-off checks |

All three call the same driver: `python -m scripts.run_comparison`.
That script tokenizes LibriTTS-R, trains both backbones, and writes
`results.json` + `loss_curves.png`. Nothing platform-specific lives
in the model code.

---

## 1. Modal (recommended)

Modal is CLI-first and gives the best GPU for the least friction.
The `modal` CLI is installed in this repo's venv (`.venv/bin/modal`,
v1.4.3). The current account profile `harshalsingh1223` is already
authenticated and verified working (`modal app list` succeeds).

### Credentials — what Modal needs

Modal authenticates with a **token ID + token secret** pair, stored in
`~/.modal.toml`. Each account is one `[profile]` section:

```toml
[harshalsingh1223]
token_id = "ak-XXXXXXXXXXXXXXXX"
token_secret = "as-XXXXXXXXXXXXXXXX"
active = true
```

You do **not** put a password anywhere. The token is created from an
authenticated browser session and can be revoked from the dashboard.

### First-time setup for a NEW account

```bash
.venv/bin/modal setup            # opens a browser, creates + stores the token
# OR, non-interactively if you already copied a token from the dashboard:
.venv/bin/modal token set \
    --token-id ak-XXXX --token-secret as-XXXX \
    --profile myaccount2 --activate
```

`modal setup` / `modal token new` open a browser. `modal token set` is
the headless path — paste a token created at
`modal.com → Settings → API Tokens`.

### Run the experiment

```bash
cd /home/cybernovas/Desktop/2026/experiments/hrm-vall-e
.venv/bin/modal run modal_train.py                  # A100, 5000 steps
.venv/bin/modal run --detach modal_train.py --steps 8000   # survives disconnect
MODAL_GPU=H100 .venv/bin/modal run modal_train.py   # pick a bigger GPU
```

`--detach` keeps the job alive on Modal's side even if your laptop
sleeps or the SSH session drops — essential for multi-hour runs.

Pull the results back when it finishes:

```bash
.venv/bin/modal volume get hrm-vall-e-results /run ./runs/modal
```

### Free credit and cost

A new Modal account is on the **Starter plan: $30/month of free
compute credit**, reset monthly. A100-40GB bills at roughly **$2/hr**;
the default 5000-step comparison finishes in ~1-2 h, so **~$2-4 per
run**. That means one $30 account covers ~8-15 full comparisons.

There is **no CLI command for the credit balance** — check it in the
dashboard: `modal.com → your workspace → Settings / Usage`. Set a
spend cap there too (Workspace → Budgets) so a runaway job can't drain
the whole credit.

### Switching accounts when credit runs out

The user has 2-3 Modal accounts (~$60-90 of credit total). Each lives
as its own profile in `~/.modal.toml`. To switch:

```bash
.venv/bin/modal profile list                # see all profiles + which is active
.venv/bin/modal profile current             # print the active one
.venv/bin/modal profile activate myaccount2 # switch
```

Per-run override without changing the active profile:

```bash
MODAL_PROFILE=myaccount2 .venv/bin/modal run modal_train.py
```

**Recommended workflow for 2-3 accounts:**

1. Create each account with a different email; for each, run
   `modal token set --profile acctN --token-id ... --token-secret ...`.
   They all stack in `~/.modal.toml` as separate sections.
2. Run on account 1 until the dashboard shows credit low.
3. `modal profile activate acct2` and keep going. The Modal Volume
   `hrm-vall-e-results` is **per-account**, so either pull results
   between switches, or push them to GitHub / the repo's `runs/`.
4. Token IDs/secrets never expire on their own — switching is instant,
   no re-login.

---

## 2. Kaggle (free fallback)

The `kaggle` CLI is installed (`~/.local/bin/kaggle`) and
`~/.kaggle/kaggle.json` (the API token) is present.

```bash
cd notebooks
kaggle kernels push                 # uploads kaggle_train.ipynb + kernel-metadata.json
kaggle kernels status harshalsinghcn/hrm-vall-e-comparison
kaggle kernels output harshalsinghcn/hrm-vall-e-comparison -p ../runs/kaggle
```

`kernel-metadata.json` sets `enable_gpu` and `enable_internet`. To
specifically get the **T4 ×2** accelerator, open the kernel in the
Kaggle editor once and pick it under *Settings → Accelerator* — the
notebook code already adapts via `torch.cuda.device_count()`.

Free tier: T4 ×2 (16 GB each), 12 h max session, ~30 h/week.

## 3. Colab (quick checks)

The official Colab CLI is new (`github.com/googlecolab/google-colab-cli`).

```bash
uv tool install git+https://github.com/googlecolab/google-colab-cli
colab new -s hrm --gpu T4
colab exec -s hrm -f scripts/colab_train.py
colab download -s hrm /content/hrm-vall-e/runs/colab/results.json
colab stop -s hrm          # always stop — idle compute still counts
```

`colab exec -f` ships the file's *contents* to the VM, which is why
`scripts/colab_train.py` is self-contained (clones the repo itself).
Free tier: T4 (16 GB), ~12 h session, ~90-min idle timeout.

---

## What to actually do

1. **First run → Modal.** `modal run modal_train.py --steps 500` as a
   smoke test (~10 min, a few cents), confirm the pipeline is green.
2. **Then the real run.** `modal run --detach modal_train.py` for the
   full 5000-step comparison.
3. **If Modal credit is gone** before you're done, switch profiles
   (section above) or fall back to Kaggle.
4. Commit `runs/*/results.json` + `loss_curves.png` to the repo so the
   result is recorded regardless of which account produced it.
