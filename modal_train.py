"""
Modal entrypoint — run the HRM-vs-baseline comparison on a cloud GPU.

  modal run modal_train.py                       # default: A100, 5000 steps
  modal run --detach modal_train.py --steps 8000 # long run, survives disconnect
  MODAL_GPU=H100 modal run modal_train.py        # pick a different GPU

What it does
------------
The container clones this repo fresh from GitHub at run time, tokenizes a
LibriTTS-R subset with EnCodec, trains both backbones, and writes
results.json + loss_curves.png to a persistent Modal Volume named
"hrm-vall-e-results". Pull them back afterwards with:

  modal volume get hrm-vall-e-results /run ./runs/modal

Cost (Modal Starter = $30/mo free credit)
-----------------------------------------
A100-40GB is ~$2/hr; the default 5000-step comparison runs in ~1-2h,
so roughly $2-4 of credit. See docs/COMPUTE.md for credit + multi-account
switching. Tokenized audio is cached on the Volume so re-runs skip it.
"""
import os
import sys

import modal

REPO = "https://github.com/harrrshall/hrm-vall-e.git"
BRANCH = os.environ.get("HRM_BRANCH", "main")
GPU = os.environ.get("MODAL_GPU", "A100")          # A100 | H100 | L4 | T4 ...

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch", "torchaudio", "numpy", "matplotlib",
        "transformers", "datasets", "soundfile", "librosa",
    )
)

app = modal.App("hrm-vall-e")
vol = modal.Volume.from_name("hrm-vall-e-results", create_if_missing=True)


# 3 seeds x 2 backbones = 6 trainings, so the timeout is generous.
@app.function(image=image, gpu=GPU, timeout=8 * 60 * 60, volumes={"/results": vol})
def run(steps: int = 5000, hours: float = 1.0, split: str = "dev-clean",
        seeds: str = "0,1,2"):
    """Clone the repo in-container and run the full comparison on the GPU."""
    import subprocess

    work = "/root/hrm-vall-e"
    subprocess.run(
        ["git", "clone", "--depth", "1", "-b", BRANCH, REPO, work], check=True
    )
    # --data-dir lives on the Volume so EnCodec tokenization is cached
    # across runs; results land on the Volume too.
    subprocess.run(
        [sys.executable, "-m", "scripts.run_comparison",
         "--out", "/results/run",
         "--data-dir", "/results/data",
         "--steps", str(steps),
         "--hours", str(hours),
         "--split", split,
         "--seeds", seeds,
         "--device", "cuda",
         "--save-checkpoints"],
        cwd=work, check=True,
    )
    vol.commit()
    print("\nDone. Fetch results with:")
    print("  modal volume get hrm-vall-e-results /run ./runs/modal")


@app.local_entrypoint()
def main(steps: int = 5000, hours: float = 1.0, split: str = "dev-clean",
         seeds: str = "0,1,2"):
    print(f"Launching HRM-vs-baseline on Modal — gpu={GPU}, steps={steps}, "
          f"seeds={seeds}")
    run.remote(steps=steps, hours=hours, split=split, seeds=seeds)
