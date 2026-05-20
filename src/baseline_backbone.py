"""
Vanilla transformer backbone — the matched-parameter-count baseline.

Same `Block` atom as HRMBackbone; the only delta is that we stack
`n_layers` distinct blocks once instead of running two smaller stacks
in a nested loop. Param count is matched to HRM by setting:

    n_layers = n_h + n_l

so the comparison is "same parameters, different compute distribution."
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .blocks import BlockStack, _build_rope_cache


@dataclass
class BaselineConfig:
    dim: int = 384
    num_heads: int = 6
    n_layers: int = 8
    expansion: float = 4.0
    max_seq_len: int = 1024
    rope_base: float = 10000.0


class BaselineBackbone(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.stack = BlockStack(cfg.n_layers, cfg.dim, cfg.num_heads, cfg.expansion)

        cos, sin = _build_rope_cache(cfg.max_seq_len, cfg.dim // cfg.num_heads,
                                     cfg.rope_base, device="cpu", dtype=torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, prefix_mask: torch.Tensor | None = None,
                bp_steps=None) -> torch.Tensor:
        # bp_steps is accepted-but-ignored to keep call sites identical
        S = x.shape[1]
        cos = self.rope_cos[:S]
        sin = self.rope_sin[:S]
        return self.stack(x, cos, sin, prefix_mask)

    @property
    def effective_block_applications(self) -> int:
        return self.cfg.n_layers
