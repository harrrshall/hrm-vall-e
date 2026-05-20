"""
One trainer that handles both backbones — the only difference is the
backbone module passed in. This makes the head-to-head comparison
literally a config flag (--backbone hrm | baseline).

Usable as a CLI for smoke tests and as a function from a Kaggle notebook.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .audio_lm import AudioLM, AudioLMConfig
from .baseline_backbone import BaselineBackbone, BaselineConfig
from .hrm_backbone import HRMBackbone, HRMConfig
from .data import SyntheticDataset, TokenLayout, collate


@dataclass
class TrainConfig:
    backbone: str = "hrm"             # "hrm" or "baseline"
    dim: int = 256
    num_heads: int = 4
    expansion: float = 4.0
    # HRM only
    n_h: int = 4
    n_l: int = 4
    h_cycles: int = 2
    l_cycles: int = 3
    # baseline only
    n_layers: int = 8                 # set so total unique blocks = n_h + n_l
    # optimisation
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 50
    total_steps: int = 500
    batch_size: int = 16
    log_every: int = 25
    # data
    text_len: int = 16
    audio_len: int = 48
    max_seq_len: int = 128
    n_train: int = 4096
    n_val: int = 512
    seed: int = 0
    device: str = "cpu"


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_model(cfg: TrainConfig, layout: TokenLayout) -> AudioLM:
    if cfg.backbone == "hrm":
        bcfg = HRMConfig(dim=cfg.dim, num_heads=cfg.num_heads,
                         n_h=cfg.n_h, n_l=cfg.n_l,
                         h_cycles=cfg.h_cycles, l_cycles=cfg.l_cycles,
                         expansion=cfg.expansion, max_seq_len=cfg.max_seq_len)
        backbone = HRMBackbone(bcfg)
    elif cfg.backbone == "baseline":
        bcfg = BaselineConfig(dim=cfg.dim, num_heads=cfg.num_heads,
                              n_layers=cfg.n_layers, expansion=cfg.expansion,
                              max_seq_len=cfg.max_seq_len)
        backbone = BaselineBackbone(bcfg)
    else:
        raise ValueError(cfg.backbone)
    lm_cfg = AudioLMConfig(text_vocab_size=layout.text_vocab_size,
                           audio_vocab_size=layout.audio_vocab_size,
                           num_specials=layout.num_specials,
                           dim=cfg.dim)
    return AudioLM(lm_cfg, backbone)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    # cosine to 10% of max LR
    p = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model: AudioLM, loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["tokens"], prefix_mask=batch["prefix_mask"], labels=batch["labels"])
        total_loss += out["loss"].item()
        n += 1
    model.train()
    return total_loss / max(1, n)


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    layout = TokenLayout()

    model = build_model(cfg, layout).to(cfg.device)
    n_params = count_params(model)

    train_ds = SyntheticDataset(layout, n_examples=cfg.n_train,
                                text_len=cfg.text_len, audio_len=cfg.audio_len,
                                seed=cfg.seed)
    val_ds = SyntheticDataset(layout, n_examples=cfg.n_val,
                              text_len=cfg.text_len, audio_len=cfg.audio_len,
                              seed=cfg.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            weight_decay=cfg.weight_decay)

    history = {"step": [], "train_loss": [], "lr": [], "val_loss": []}
    model.train()
    step = 0
    t0 = time.time()

    while step < cfg.total_steps:
        for batch in train_loader:
            if step >= cfg.total_steps:
                break
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            # HRM-style bp_steps schedule (no-op for baseline backbone)
            bp_steps = None
            if cfg.backbone == "hrm":
                hb: HRMBackbone = model.backbone  # type: ignore
                bp_steps = hb.bp_steps_for(step, cfg.total_steps)

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
                vl = evaluate(model, val_loader, cfg.device)
                history["step"].append(step)
                history["train_loss"].append(out["loss"].item())
                history["val_loss"].append(vl)
                history["lr"].append(lr)
                elapsed = time.time() - t0
                print(f"step {step:>5d}  train_loss={out['loss'].item():.4f}  "
                      f"val_loss={vl:.4f}  lr={lr:.2e}  bp_steps={bp_steps}  "
                      f"{elapsed:.1f}s")
            step += 1

    return {"model": model, "history": history, "n_params": n_params}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["hrm", "baseline"], default="hrm")
    ap.add_argument("--total-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = TrainConfig(
        backbone=args.backbone,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        dim=args.dim,
        device=args.device,
        seed=args.seed,
    )
    result = train(cfg)
    print(f"\nFinal params: {result['n_params']:,}")
    print(f"Final val_loss: {result['history']['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
