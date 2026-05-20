"""
Colab runner — self-contained so it works with `colab exec`.

The official Colab CLI (github.com/googlecolab/google-colab-cli) ships a
file's *contents* to the VM with `colab exec -f`, so this script must
stand alone: it clones the repo, installs deps, and runs the comparison.

Usage from your laptop
----------------------
    uv tool install git+https://github.com/googlecolab/google-colab-cli
    colab new -s hrm --gpu T4
    colab exec -s hrm -f scripts/colab_train.py
    colab download -s hrm /content/hrm-vall-e/runs/colab/results.json
    colab stop -s hrm

A free Colab T4 (16 GB) handles the default dim=384 config. Keep the
session under ~12 h and watch the ~90-min idle timeout.
"""
import subprocess
import sys

REPO = "https://github.com/harrrshall/hrm-vall-e.git"
WORK = "/content/hrm-vall-e"
STEPS = 5000
HOURS = 1.0


def sh(cmd, **kw):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def main():
    import os
    if not os.path.isdir(WORK):
        sh(["git", "clone", "--depth", "1", REPO, WORK])
    sh([sys.executable, "-m", "pip", "install", "-q",
        "transformers", "datasets", "soundfile", "librosa", "torchaudio",
        "torchcodec", "matplotlib"])
    sh([sys.executable, "-m", "scripts.run_comparison",
        "--out", f"{WORK}/runs/colab",
        "--steps", str(STEPS),
        "--hours", str(HOURS),
        "--device", "cuda",
        "--save-checkpoints"],
       cwd=WORK)
    print("\nResults in", f"{WORK}/runs/colab — pull them with `colab download`.")


if __name__ == "__main__":
    main()
