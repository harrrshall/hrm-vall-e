"""
Two data sources, same Sample interface:

- SyntheticDataset: random tokens with a synthetic "text-prefix -> repeat-the-prefix-as-audio"
  task. Lets the smoke test confirm the loss actually goes down.

- LibriTTSEnCodecDataset: real LibriTTS-R audio, encoded to first-codebook EnCodec
  tokens, with the transcript as a byte-level text prefix.

Sample format:

    tokens:      [S]  int64
    prefix_mask: [S]  bool   True on prompt positions
    labels:      [S]  int64  IGNORE_LABEL on prompt/pad positions
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import torch
from torch.utils.data import Dataset


IGNORE_LABEL = -100


# Token-id layout
# ------------------------------------------------------------------
# Reserved specials at the front of the vocab, then text byte tokens,
# then audio codec tokens. Helper functions below give absolute IDs.

@dataclass
class TokenLayout:
    num_specials: int = 4           # 0=PAD 1=BOS 2=SEP 3=EOS
    text_vocab_size: int = 256      # byte-level
    audio_vocab_size: int = 1024    # EnCodec / Mimi codebook size

    @property
    def pad(self) -> int: return 0
    @property
    def bos(self) -> int: return 1
    @property
    def sep(self) -> int: return 2
    @property
    def eos(self) -> int: return 3

    def text_id(self, byte: int) -> int:
        assert 0 <= byte < self.text_vocab_size
        return self.num_specials + byte

    def audio_id(self, code: int) -> int:
        assert 0 <= code < self.audio_vocab_size
        return self.num_specials + self.text_vocab_size + code

    @property
    def total_vocab(self) -> int:
        return self.num_specials + self.text_vocab_size + self.audio_vocab_size


# ------------------------------------------------------------------
# Synthetic dataset for smoke tests
# ------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    Task: prefix is a random text byte sequence; audio response is the
    same sequence reinterpreted as audio codes (modulo audio_vocab_size).
    A working model will quickly learn this copy.
    """

    def __init__(self, layout: TokenLayout, n_examples: int = 1024,
                 text_len: int = 16, audio_len: int = 48, seed: int = 0):
        self.layout = layout
        self.text_len = text_len
        self.audio_len = audio_len
        self.n_examples = n_examples
        self.rng = random.Random(seed)
        # Pre-generate to keep epochs reproducible
        self.items = [self._make_one() for _ in range(n_examples)]

    def _make_one(self):
        text_bytes = [self.rng.randrange(self.layout.text_vocab_size) for _ in range(self.text_len)]
        # audio code = text_byte mod audio_vocab; repeat the text once across audio_len
        audio_codes = [(text_bytes[i % self.text_len]) % self.layout.audio_vocab_size
                       for i in range(self.audio_len)]
        return text_bytes, audio_codes

    def __len__(self): return self.n_examples

    def __getitem__(self, idx):
        text_bytes, audio_codes = self.items[idx]
        L = self.layout
        seq = [L.bos] + [L.text_id(b) for b in text_bytes] + [L.sep] \
              + [L.audio_id(c) for c in audio_codes] + [L.eos]
        prefix_len = 1 + self.text_len + 1  # bos + text + sep
        S = len(seq)
        tokens = torch.tensor(seq, dtype=torch.long)
        prefix_mask = torch.zeros(S, dtype=torch.bool)
        prefix_mask[:prefix_len] = True
        # supervise next-token prediction only on response positions
        # labels[i] = tokens[i+1] for i in response region, else IGNORE
        labels = torch.full((S,), IGNORE_LABEL, dtype=torch.long)
        labels[prefix_len - 1 : -1] = tokens[prefix_len:]  # predict sep->audio_1, audio_1->audio_2, ..., audio_last->eos
        return {"tokens": tokens, "prefix_mask": prefix_mask, "labels": labels}


def collate(batch):
    # all sequences are the same length in our synthetic case
    tokens = torch.stack([b["tokens"] for b in batch])
    prefix_mask = torch.stack([b["prefix_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"tokens": tokens, "prefix_mask": prefix_mask, "labels": labels}


# ------------------------------------------------------------------
# LibriTTS + EnCodec dataset (for Kaggle)
# ------------------------------------------------------------------

class LibriTTSEnCodecDataset(Dataset):
    """
    Expects pre-tokenized data on disk:

        {save_dir}/sample_{i}.pt  with keys: text_bytes [T], audio_codes [A] (first codebook)

    The notebook in notebooks/ produces these from LibriTTS-R + EnCodec 24kHz.

    Sequences are padded/truncated to `max_seq_len`. We use a fixed-length
    layout so all batches stack cleanly.
    """

    def __init__(self, file_paths, layout: TokenLayout, max_seq_len: int = 512,
                 max_text_len: int = 100, max_audio_len: int = 400):
        self.paths = list(file_paths)
        self.layout = layout
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        item = torch.load(self.paths[idx])
        text = item["text_bytes"][: self.max_text_len].tolist()
        audio = item["audio_codes"][: self.max_audio_len].tolist()
        L = self.layout
        seq = [L.bos] + [L.text_id(b) for b in text] + [L.sep] \
              + [L.audio_id(c) for c in audio] + [L.eos]
        # pad to max_seq_len
        S = self.max_seq_len
        seq = seq[:S]
        prefix_len = min(1 + len(text) + 1, S)
        tokens = torch.full((S,), L.pad, dtype=torch.long)
        tokens[: len(seq)] = torch.tensor(seq, dtype=torch.long)
        prefix_mask = torch.zeros(S, dtype=torch.bool)
        prefix_mask[:prefix_len] = True
        labels = torch.full((S,), IGNORE_LABEL, dtype=torch.long)
        # supervise next-token on response positions only
        if prefix_len < len(seq):
            n = len(seq) - prefix_len
            labels[prefix_len - 1 : prefix_len - 1 + n] = torch.tensor(seq[prefix_len:], dtype=torch.long)
        return {"tokens": tokens, "prefix_mask": prefix_mask, "labels": labels}
