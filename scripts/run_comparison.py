"""
End-to-end HRM-vs-baseline comparison — the single source of truth.

This is what Modal, Colab, Kaggle, and a local GPU all call. It:

  1. tokenizes a LibriTTS-R subset with EnCodec (skipped if already done),
  2. trains the HRM backbone and the matched-param baseline, each over
     several seeds so the result is mean +/- spread, not a single number,
  3. writes results.json + loss_curves.png to --out.

Run it directly:

    python -m scripts.run_comparison --steps 5000 --hours 1.0 --seeds 0,1,2

The comparison is fair by construction: both backbones share the exact
same Block atom and are sized so HRM's (n_h + n_l) unique blocks equal
the baseline's n_layers. See README.md for the design rationale.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.train import TrainConfig, build_model, lr_at, count_params
from src.data import LibriTTSEnCodecDataset, TokenLayout, collate


REPO_ROOT = Path(__file__).resolve().parents[1]


def _hr(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}", flush=True)


def ensure_data(data_dir: Path, hours: float, split: str) -> list[Path]:
    """Tokenize LibriTTS-R with EnCodec unless the .pt files already exist."""
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(data_dir.glob("sample_*.pt"))
    if existing:
        print(f"data: reusing {len(existing)} pre-tokenized clips in {data_dir}")
        return existing

    _hr(f"Tokenizing ~{hours}h of LibriTTS-R {split} with EnCodec")
    subprocess.check_call(
        [sys.executable, "-m", "scripts.prepare_libritts",
         "--out", str(data_dir), "--hours", str(hours), "--split", split],
        cwd=str(REPO_ROOT),
    )
    files = sorted(data_dir.glob("sample_*.pt"))
    if not files:
        raise RuntimeError("tokenization produced no clips — check the data step")
    print(f"data: {len(files)} clips tokenized")
    return files


def run_one(cfg: TrainConfig, train_ds, val_ds, layout: TokenLayout, name: str) -> dict:
    """Train one backbone at one seed; return history + final metrics."""
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, layout).to(cfg.device)
    n_params = count_params(model)
    print(f"{name} (seed {cfg.seed}): {n_params:,} trainable params", flush=True)

    workers = 2 if cfg.device == "cuda" else 0
    tl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=collate, drop_last=True, num_workers=workers)
    vl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                    collate_fn=collate, drop_last=True, num_workers=workers)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            betas=(cfg.beta1, cfg.beta2),
                            weight_decay=cfg.weight_decay)

    hist = {"step": [], "train_loss": [], "val_loss": [], "lr": [], "bp_steps": []}
    model.train()
    step, t0 = 0, time.time()
    while step < cfg.total_steps:
        for batch in tl:
            if step >= cfg.total_steps:
                break
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            bp_steps = None
            if cfg.backbone == "hrm":
                bp_steps = model.backbone.bp_steps_for(step, cfg.total_steps)
            lr = lr_at(step, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            out = model(batch["tokens"], prefix_mask=batch["prefix_mask"],
                        labels=batch["labels"], bp_steps=bp_steps)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % cfg.log_every == 0:
                model.eval()
                with torch.no_grad():
                    vloss = sum(
                        model(b["tokens"].to(cfg.device),
                              prefix_mask=b["prefix_mask"].to(cfg.device),
                              labels=b["labels"].to(cfg.device))["loss"].item()
                        for b in vl
                    ) / max(1, len(vl))
                model.train()
                hist["step"].append(step)
                hist["train_loss"].append(out["loss"].item())
                hist["val_loss"].append(vloss)
                hist["lr"].append(lr)
                hist["bp_steps"].append(bp_steps)
                print(f"  step {step:>5d}  train {out['loss'].item():.4f}  "
                      f"val {vloss:.4f}  lr {lr:.2e}  bp {bp_steps}  "
                      f"{time.time() - t0:.0f}s", flush=True)
            step += 1

    return {
        "name": name,
        "seed": cfg.seed,
        "n_params": n_params,
        "history": hist,
        "final_val_loss": hist["val_loss"][-1],
        "best_val_loss": min(hist["val_loss"]),
        "wall_seconds": time.time() - t0,
        "state_dict": model.state_dict(),
    }


def aggregate(runs: list[dict]) -> dict:
    """Collapse a backbone's per-seed runs into mean / spread."""
    fvl = [r["final_val_loss"] for r in runs]
    bvl = [r["best_val_loss"] for r in runs]
    std = statistics.pstdev if len(runs) > 1 else (lambda _: 0.0)
    return {
        "name": runs[0]["name"],
        "n_params": runs[0]["n_params"],
        "seeds": [r["seed"] for r in runs],
        "final_val_loss": {"mean": statistics.fmean(fvl), "std": std(fvl), "values": fvl},
        "best_val_loss": {"mean": statistics.fmean(bvl), "std": std(bvl), "values": bvl},
        "wall_seconds_total": sum(r["wall_seconds"] for r in runs),
        "per_seed": [{"seed": r["seed"], "history": r["history"],
                      "final_val_loss": r["final_val_loss"],
                      "best_val_loss": r["best_val_loss"]} for r in runs],
    }


def _mean_curve(runs: list[dict]):
    """Element-wise mean val-loss curve across seeds (steps line up)."""
    steps = runs[0]["history"]["step"]
    cols = list(zip(*[r["history"]["val_loss"] for r in runs]))
    return steps, [statistics.fmean(c) for c in cols]


def plot(hrm_runs: list[dict], base_runs: list[dict], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for runs, color, label in [(hrm_runs, "C0", "HRM"), (base_runs, "C1", "Baseline")]:
        for r in runs:  # thin per-seed lines
            ax[0].plot(r["history"]["step"], r["history"]["train_loss"],
                       color=color, alpha=0.25, linewidth=1)
            ax[1].plot(r["history"]["step"], r["history"]["val_loss"],
                       color=color, alpha=0.25, linewidth=1)
        steps, mean_val = _mean_curve(runs)
        ax[1].plot(steps, mean_val, color=color, linewidth=2.5,
                   label=f"{label} (mean of {len(runs)})")
        tcols = list(zip(*[r["history"]["train_loss"] for r in runs]))
        ax[0].plot(steps, [statistics.fmean(c) for c in tcols],
                   color=color, linewidth=2.5, label=f"{label} (mean)")
    ax[0].set(title="train loss", xlabel="step", ylabel="loss")
    ax[1].set(title="val loss", xlabel="step", ylabel="loss")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    plt.suptitle(f"HRM ({hrm_runs[0]['n_params']:,}) vs "
                 f"Baseline ({base_runs[0]['n_params']:,}) — matched params")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"saved {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "latest")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--seeds", default="0,1,2",
                    help="comma-separated seeds; each backbone is trained once per seed")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-checkpoints", action="store_true",
                    help="save seed-0 checkpoints for both backbones")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir or (out / "data")

    _hr("HRM vs Baseline — Audio LM Head-to-Head")
    print(f"device: {args.device}  "
          f"cuda devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"steps: {args.steps}  seeds: {seeds}  out: {out}")

    files = ensure_data(data_dir, args.hours, args.split)
    layout = TokenLayout()
    split = max(1, int(0.9 * len(files)))
    train_files, val_files = files[:split], files[split:] or files[-1:]
    train_ds = LibriTTSEnCodecDataset(train_files, layout, max_seq_len=args.seq_len)
    val_ds = LibriTTSEnCodecDataset(val_files, layout, max_seq_len=args.seq_len)
    print(f"train clips: {len(train_ds)}  val clips: {len(val_ds)}")

    common = dict(dim=args.dim, num_heads=6, expansion=4.0, max_seq_len=args.seq_len,
                  batch_size=args.batch_size, total_steps=args.steps,
                  warmup_steps=max(50, args.steps // 25), lr=3e-4,
                  log_every=max(25, args.steps // 100), device=args.device)

    hrm_runs, base_runs = [], []
    for seed in seeds:
        _hr(f"Seed {seed} — HRM backbone")
        hrm_cfg = TrainConfig(backbone="hrm", n_h=4, n_l=4, h_cycles=2, l_cycles=3,
                              seed=seed, **common)
        hrm_runs.append(run_one(hrm_cfg, train_ds, val_ds, layout, "HRM"))
        _hr(f"Seed {seed} — Baseline backbone")
        base_cfg = TrainConfig(backbone="baseline", n_layers=8, seed=seed, **common)
        base_runs.append(run_one(base_cfg, train_ds, val_ds, layout, "Baseline"))

        if args.save_checkpoints and seed == seeds[0]:
            torch.save({"state_dict": hrm_runs[-1].pop("state_dict"),
                        "cfg": hrm_cfg.__dict__}, out / "hrm_ckpt.pt")
            torch.save({"state_dict": base_runs[-1].pop("state_dict"),
                        "cfg": base_cfg.__dict__}, out / "baseline_ckpt.pt")
            print(f"saved seed-{seed} checkpoints to {out}")

    plot(hrm_runs, base_runs, out / "loss_curves.png")

    hrm = aggregate(hrm_runs)
    base = aggregate(base_runs)
    # Verdict uses BEST val loss, not final — with small data both backbones
    # can overfit late, which makes final-step loss a memorisation artefact.
    h_mean = hrm["best_val_loss"]["mean"]
    b_mean = base["best_val_loss"]["mean"]
    delta = b_mean - h_mean
    pct = 100.0 * delta / b_mean
    winner = "HRM" if delta > 0 else "Baseline"
    # rough significance: is the gap bigger than the combined spread?
    spread = hrm["best_val_loss"]["std"] + base["best_val_loss"]["std"]
    decisive = abs(delta) > spread

    results = {
        "config": {"steps": args.steps, "hours": args.hours, "dim": args.dim,
                   "batch_size": args.batch_size, "seq_len": args.seq_len,
                   "seeds": seeds, "device": args.device},
        "hrm": hrm,
        "baseline": base,
        "verdict": {"winner": winner, "val_loss_delta": delta, "delta_pct": pct,
                    "combined_std": spread, "exceeds_noise": decisive},
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))

    _hr("VERDICT")
    print(f"  params      HRM {hrm['n_params']:,}   Baseline {base['n_params']:,}")
    print(f"  seeds       {seeds}")
    print(f"  best  val   HRM {h_mean:.4f} +/- {hrm['best_val_loss']['std']:.4f}   "
          f"Baseline {b_mean:.4f} +/- {base['best_val_loss']['std']:.4f}")
    print(f"  final val   HRM {hrm['final_val_loss']['mean']:.4f}   "
          f"Baseline {base['final_val_loss']['mean']:.4f}   (overfit check)")
    print(f"  -> {winner} wins by {abs(pct):.2f}% mean BEST val loss")
    print(f"  -> gap {'EXCEEDS' if decisive else 'is WITHIN'} combined seed noise "
          f"({abs(delta):.4f} vs {spread:.4f})")
    if not decisive:
        print("     treat as inconclusive — consider more seeds or a longer run")
    print(f"\n  results.json + loss_curves.png written to {out}")


if __name__ == "__main__":
    main()
