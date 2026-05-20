"""
CPU smoke tests. These check three things:

  1. Both backbones produce [B, S, d] outputs of the right shape.
  2. Gradients flow end-to-end through the audio LM with both backbones.
  3. The HRM and matched-baseline configs have parameter counts within 1%.

If all three pass, the experiment is correctly wired. Audio quality is a
separate question answered by the Kaggle training run.
"""
from __future__ import annotations

import sys
import pathlib

# Allow running from anywhere
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.blocks import Block, BlockStack
from src.hrm_backbone import HRMBackbone, HRMConfig
from src.baseline_backbone import BaselineBackbone, BaselineConfig
from src.audio_lm import AudioLM, AudioLMConfig, IGNORE_LABEL


# -- shared small config --------------------------------------------------------

def _hrm_cfg() -> HRMConfig:
    return HRMConfig(dim=128, num_heads=4, n_h=2, n_l=2, h_cycles=2, l_cycles=3,
                     max_seq_len=64)

def _baseline_cfg_matching(h: HRMConfig) -> BaselineConfig:
    # matched param count: n_layers = n_h + n_l (each backbone has the same
    # unique-block count). HRM gets effective depth by recurrence.
    return BaselineConfig(dim=h.dim, num_heads=h.num_heads,
                          n_layers=h.n_h + h.n_l, max_seq_len=h.max_seq_len)

def _build_lm(backbone) -> AudioLM:
    lm_cfg = AudioLMConfig(text_vocab_size=32, audio_vocab_size=64, num_specials=4, dim=backbone.cfg.dim)
    return AudioLM(lm_cfg, backbone)


# -- 1. shapes ------------------------------------------------------------------

def test_hrm_forward_shape():
    cfg = _hrm_cfg()
    m = HRMBackbone(cfg)
    x = torch.randn(2, 16, cfg.dim)
    y = m(x)
    assert y.shape == (2, 16, cfg.dim), y.shape


def test_baseline_forward_shape():
    h = _hrm_cfg()
    cfg = _baseline_cfg_matching(h)
    m = BaselineBackbone(cfg)
    x = torch.randn(2, 16, cfg.dim)
    y = m(x)
    assert y.shape == (2, 16, cfg.dim), y.shape


# -- 2. gradient flow ----------------------------------------------------------

def _run_one_step(backbone, with_bp_steps: int | None):
    lm = _build_lm(backbone)
    B, S = 2, 16
    tokens = torch.randint(0, lm.vocab_size, (B, S))
    # arbitrary 8-token prefix; rest is response
    prefix_mask = torch.zeros(B, S, dtype=torch.bool)
    prefix_mask[:, :8] = True
    # supervise only response positions
    labels = tokens.clone()
    labels[prefix_mask] = IGNORE_LABEL
    out = lm(tokens, prefix_mask=prefix_mask, labels=labels, bp_steps=with_bp_steps)
    out["loss"].backward()
    return lm


def test_hrm_grad_flow_full_bptt():
    cfg = _hrm_cfg()
    lm = _run_one_step(HRMBackbone(cfg), with_bp_steps=None)
    # every trainable param should have received a gradient under full BPTT
    missing = [n for n, p in lm.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params with no grad under full BPTT: {missing[:5]}"


def test_hrm_grad_flow_truncated_bp2():
    # With bp_steps=2 the embedding should NOT receive a gradient (this is
    # the surprise from the math walkthrough §6.2). The H/L stacks still do.
    cfg = _hrm_cfg()
    lm = _run_one_step(HRMBackbone(cfg), with_bp_steps=2)
    embed_grad = lm.embed.weight.grad
    h_grad = lm.backbone.h_stack.layers[0].attn.qkv.weight.grad
    l_grad = lm.backbone.l_stack.layers[0].attn.qkv.weight.grad
    assert embed_grad is None or embed_grad.abs().sum() == 0, "embedding should get no grad with bp_steps=2"
    assert h_grad is not None and h_grad.abs().sum() > 0, "H stack should get grad"
    assert l_grad is not None and l_grad.abs().sum() > 0, "L stack should get grad"


def test_hrm_grad_flow_bp5_reaches_embedding():
    # With bp_steps=5 the embedding receives gradient (first H step enters graph)
    cfg = _hrm_cfg()
    lm = _run_one_step(HRMBackbone(cfg), with_bp_steps=5)
    embed_grad = lm.embed.weight.grad
    assert embed_grad is not None and embed_grad.abs().sum() > 0, "embedding should get grad with bp_steps=5"


def test_baseline_grad_flow():
    h = _hrm_cfg()
    cfg = _baseline_cfg_matching(h)
    lm = _run_one_step(BaselineBackbone(cfg), with_bp_steps=None)
    missing = [n for n, p in lm.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"baseline params with no grad: {missing[:5]}"


# -- 3. param-count parity -----------------------------------------------------

def _count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_param_count_matches_within_1pct():
    h = _hrm_cfg()
    b = _baseline_cfg_matching(h)
    hrm = HRMBackbone(h)
    base = BaselineBackbone(b)
    n_hrm = _count_params(hrm)
    n_base = _count_params(base)
    rel = abs(n_hrm - n_base) / max(n_hrm, n_base)
    assert rel < 0.01, f"param count mismatch: HRM={n_hrm}, baseline={n_base} ({rel*100:.2f}%)"


# -- 4. effective depth claim --------------------------------------------------

def test_effective_depth_ratio():
    h = _hrm_cfg()
    hrm = HRMBackbone(h)
    base = BaselineBackbone(_baseline_cfg_matching(h))
    # HRM effective: H*L*n_L + H*n_H = 2*3*2 + 2*2 = 16 block applications
    # baseline:     n_layers = 4
    # ratio = 4x (matches what the README claims for similar configs)
    eff = hrm.effective_block_applications
    base_eff = base.effective_block_applications
    assert eff == 16, eff
    assert base_eff == 4, base_eff
    assert eff / base_eff == 4.0
