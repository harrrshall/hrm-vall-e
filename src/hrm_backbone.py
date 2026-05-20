"""
HRM backbone — minimal adaptation of HRM-Text's hierarchical recurrence,
specialised for an audio-token language model.

The shape contract is identical to a vanilla stacked transformer:

    backbone(x: [B, S, d], cos, sin, prefix_mask) -> z_H: [B, S, d]

Internally we run the H/L nested loop with input injection and bp_steps
truncation, exactly mirroring `models/baselines/hrm_nocarry_bp_warmup.py`
in HRM-Text. See `notes/math-walkthrough.md` in HRM-text/ for the math.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .blocks import BlockStack, _build_rope_cache


@dataclass
class HRMConfig:
    dim: int = 384
    num_heads: int = 6
    n_h: int = 4              # layers in H module
    n_l: int = 4              # layers in L module
    h_cycles: int = 2
    l_cycles: int = 3
    expansion: float = 4.0
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    bp_min: int = 2
    bp_max: int = 5
    bp_warmup_ratio: float = 0.2


class HRMBackbone(nn.Module):
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.cfg = cfg
        self.h_stack = BlockStack(cfg.n_h, cfg.dim, cfg.num_heads, cfg.expansion)
        self.l_stack = BlockStack(cfg.n_l, cfg.dim, cfg.num_heads, cfg.expansion)
        # learned init for z_L (broadcast across the sequence)
        self.zL_init = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        nn.init.trunc_normal_(self.zL_init, std=0.02)

        cos, sin = _build_rope_cache(cfg.max_seq_len, cfg.dim // cfg.num_heads,
                                     cfg.rope_base, device="cpu", dtype=torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ---- bp_steps schedule ----------------------------------------------------

    def bp_steps_for(self, step: int, total_steps: int) -> int:
        cfg = self.cfg
        warm_total = max(1, int(total_steps * cfg.bp_warmup_ratio))
        frac = min(1.0, step / warm_total)
        return cfg.bp_min + int(frac * (cfg.bp_max - cfg.bp_min))

    # ---- forward --------------------------------------------------------------

    def forward(self, x: torch.Tensor, prefix_mask: torch.Tensor | None = None,
                bp_steps: int | None = None) -> torch.Tensor:
        """
        x:           [B, S, d]  initial hidden state (e.g. embed(tokens))
        prefix_mask: [B, S] bool, True where token belongs to the prompt prefix
                     (PrefixLM); None for pure causal.
        bp_steps:    how many of the last unrolled iterations to backprop
                     through. If None, full BPTT (equivalent to torch's default).

        Returns z_H: [B, S, d]
        """
        cfg = self.cfg
        S = x.shape[1]
        cos = self.rope_cos[:S]
        sin = self.rope_sin[:S]

        z_H = x
        z_L = self.zL_init.expand(x.shape[0], S, cfg.dim)

        if bp_steps is None:
            # full BPTT
            return self._loop_full(z_H, z_L, cos, sin, prefix_mask)

        # truncated bp_steps: split the budget across H and L as HRM-Text does
        H_bp_steps = min(cfg.h_cycles, bp_steps - 1)
        L_bp_steps = bp_steps - H_bp_steps

        total_L_iters = cfg.h_cycles * cfg.l_cycles
        grad_on = torch.is_grad_enabled()

        for i in range(cfg.h_cycles):
            for k in range(i * cfg.l_cycles, (i + 1) * cfg.l_cycles):
                enable = grad_on and (k >= total_L_iters - L_bp_steps)
                with torch.set_grad_enabled(enable):
                    z_L = self.l_stack(z_L + z_H, cos, sin, prefix_mask)
            enable_h = grad_on and (i >= cfg.h_cycles - H_bp_steps)
            with torch.set_grad_enabled(enable_h):
                z_H = self.h_stack(z_H + z_L, cos, sin, prefix_mask)

        return z_H

    def _loop_full(self, z_H, z_L, cos, sin, prefix_mask):
        cfg = self.cfg
        for _ in range(cfg.h_cycles):
            for _ in range(cfg.l_cycles):
                z_L = self.l_stack(z_L + z_H, cos, sin, prefix_mask)
            z_H = self.h_stack(z_H + z_L, cos, sin, prefix_mask)
        return z_H

    # ---- helpers --------------------------------------------------------------

    @property
    def effective_block_applications(self) -> int:
        cfg = self.cfg
        return cfg.h_cycles * cfg.l_cycles * cfg.n_l + cfg.h_cycles * cfg.n_h
