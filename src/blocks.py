"""
Shared transformer-block atom used by both baseline and HRM variants.

Deliberately minimal: RMSNorm + multi-head causal/PrefixLM attention with
RoPE + SwiGLU MLP. No fancy fused kernels — uses torch SDPA so it runs on
CPU for shape/grad smoke tests and on GPU (with flash-attention backend
auto-selected) for real training.

We intentionally do NOT copy the "gated attention" piece from HRM-Text
here. We want the baseline and the HRM variant to share the EXACT same
block, so the only architectural delta under test is "stacked once vs
applied in nested H/L loop with input injection."
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learned gain — present here (HRM-Text drops it; for a small model
        # the gain helps stability and the param cost is negligible).
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, S, D]; cos/sin: [S, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Attention (causal or PrefixLM)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                prefix_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, H, S, D]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if prefix_mask is None:
            # pure causal
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # PrefixLM: a [B, S] boolean of "is in prefix". Build a mask:
            #   allow j -> i if (j is prefix and i is anywhere) or (j <= i and j is response)
            # equivalently: forbid j -> i if (j is response and j > i)
            # We build [B, 1, S, S] additive mask in float.
            ar = torch.arange(S, device=x.device)
            causal = ar.unsqueeze(0) <= ar.unsqueeze(1)        # [S, S], True where j<=i
            j_is_prefix = prefix_mask.unsqueeze(1)               # [B, 1, S]
            # [B, S, S]: allowed if (j is prefix) OR (j <= i)
            allowed = j_is_prefix | causal.unsqueeze(0)
            attn_mask = torch.zeros(B, 1, S, S, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~allowed.unsqueeze(1), float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.proj(out)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, dim: int, expansion: float = 4.0):
        super().__init__()
        # Match a vanilla 4x MLP's param count: 4 * dim * 2/3, rounded to 64.
        hidden = int(round(expansion * dim * 2 / 3))
        hidden = ((hidden + 63) // 64) * 64
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Single transformer block (pre-norm, residual)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, expansion: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, expansion)

    def forward(self, x, cos, sin, prefix_mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, prefix_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Block-stack helper used by both baseline and HRM modules
# ---------------------------------------------------------------------------

class BlockStack(nn.Module):
    def __init__(self, num_layers: int, dim: int, num_heads: int, expansion: float = 4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(dim, num_heads, expansion) for _ in range(num_layers)
        ])

    def forward(self, x, cos, sin, prefix_mask=None):
        for layer in self.layers:
            x = layer(x, cos, sin, prefix_mask)
        return x
