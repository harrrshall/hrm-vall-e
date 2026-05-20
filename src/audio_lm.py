"""
A small VALL-E-style audio LM. We model a single (flattened) stream of
audio codec tokens conditioned on a text prefix — i.e. the first codebook
of EnCodec / Mimi, plus a small text vocab prepended as a PrefixLM prompt.

Token layout in a sequence (length S):

    [BOS] text_1 text_2 ... text_T  [SEP] audio_1 audio_2 ... audio_A  [EOS]
    |_____________ prefix _______________| |________ response ________|

Loss is computed only on the response positions (target_only). Prefix is
bidirectional in attention (PrefixLM); response is causal.

Both backbones plug in here through the same `Backbone` interface, so the
experiment is a one-line swap.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_LABEL = -100


@dataclass
class AudioLMConfig:
    text_vocab_size: int = 256        # byte-level text tokens (placeholder)
    audio_vocab_size: int = 1024      # EnCodec / Mimi first codebook
    num_specials: int = 4             # BOS, SEP, EOS, PAD
    dim: int = 384


class AudioLM(nn.Module):
    """
    Args:
      backbone: a module with signature
                  backbone(x, prefix_mask=None, bp_steps=None) -> [B, S, d]
    """

    def __init__(self, cfg: AudioLMConfig, backbone: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone

        # Shared embedding table for text + audio + specials. The model
        # learns to read all three from one vocab; the loss only supervises
        # audio positions, so text positions just provide conditioning.
        vocab = cfg.text_vocab_size + cfg.audio_vocab_size + cfg.num_specials
        self.vocab_size = vocab
        self.embed = nn.Embedding(vocab, cfg.dim)
        self.norm_out = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, vocab, bias=False)

        nn.init.trunc_normal_(self.embed.weight, std=0.02)
        nn.init.trunc_normal_(self.lm_head.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, prefix_mask: torch.Tensor,
                labels: torch.Tensor | None = None, bp_steps: int | None = None):
        """
        tokens:      [B, S] int64
        prefix_mask: [B, S] bool — True where the position is part of the
                     prompt prefix (bidirectional attention region)
        labels:      [B, S] int64, with IGNORE_LABEL on positions whose
                     loss we don't want to compute (i.e. the prefix and pad)
        """
        x = self.embed(tokens)                                  # [B, S, d]
        z = self.backbone(x, prefix_mask=prefix_mask, bp_steps=bp_steps)
        z = self.norm_out(z)
        logits = self.lm_head(z)                                # [B, S, V]

        out = {"logits": logits}
        if labels is not None:
            # next-token CE: predict labels[i] from logits[i-1]. The dataset
            # is expected to already have shifted labels, so we just CE.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                ignore_index=IGNORE_LABEL,
                reduction="mean",
            )
            out["loss"] = loss
        return out
