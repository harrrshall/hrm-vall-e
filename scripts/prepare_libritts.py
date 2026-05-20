"""
Tokenize a LibriTTS-R subset with EnCodec 24kHz (first codebook only).

Usage (designed to be run from a Kaggle notebook cell):

    python -m scripts.prepare_libritts \
        --out /kaggle/working/data \
        --hours 1.0 \
        --split dev-clean

Outputs:
    {out}/sample_0000.pt, sample_0001.pt, ...
    each with keys:
        text_bytes  : LongTensor [T]
        audio_codes : LongTensor [A]   # EnCodec codebook-0 ids, 0..1023
        sample_rate : 24000
        duration_s  : float
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torchaudio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hours", type=float, default=1.0,
                    help="approximate hours of speech to tokenize")
    ap.add_argument("--split", default="dev-clean",
                    choices=["dev-clean", "test-clean", "train-clean-100"])
    ap.add_argument("--max-sec-per-clip", type=float, default=10.0)
    ap.add_argument("--min-sec-per-clip", type=float, default=1.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- EnCodec model ----
    from transformers import EncodecModel, AutoProcessor
    print("loading EnCodec 24kHz...")
    codec = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()
    proc = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    if torch.cuda.is_available():
        codec = codec.cuda()
    sr = codec.config.sampling_rate
    assert sr == 24000

    # ---- LibriTTS dataset (downloads if missing) ----
    print(f"loading LibriTTS-R {args.split}...")
    # torchaudio bundles a LIBRITTS class but for LibriTTS-R we use HF datasets
    # for simplicity in Kaggle. Fall back to LIBRITTS if HF not available.
    try:
        from datasets import load_dataset
        ds = load_dataset("mythicinfinity/libritts_r", args.split, split="train", streaming=True)
        iterator = iter(ds)
        def next_clip():
            item = next(iterator)
            wav = torch.tensor(item["audio"]["array"], dtype=torch.float32).unsqueeze(0)
            this_sr = item["audio"]["sampling_rate"]
            return wav, this_sr, item["text_normalized"]
    except Exception as e:  # noqa: BLE001
        print(f"HF datasets unavailable ({e}); falling back to torchaudio LIBRITTS")
        td_split = args.split.replace("-", "-")  # same names
        td_ds = torchaudio.datasets.LIBRITTS(
            root=str(args.out.parent / "libritts_raw"),
            url=td_split,
            download=True,
        )
        td_iter = iter(td_ds)
        def next_clip():
            wav, this_sr, _, _, _, _, txt = next(td_iter)
            return wav, this_sr, txt

    target_samples = int(args.hours * 3600 * sr)
    written_samples = 0
    idx = 0
    while written_samples < target_samples:
        try:
            wav, this_sr, text = next_clip()
        except StopIteration:
            break
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if this_sr != sr:
            wav = torchaudio.functional.resample(wav, this_sr, sr)
        n = wav.shape[-1]
        if n < args.min_sec_per_clip * sr:
            continue
        if n > args.max_sec_per_clip * sr:
            wav = wav[..., : int(args.max_sec_per_clip * sr)]
            n = wav.shape[-1]

        inputs = proc(raw_audio=wav.squeeze(0).numpy(), sampling_rate=sr, return_tensors="pt")
        input_values = inputs["input_values"]
        if torch.cuda.is_available():
            input_values = input_values.cuda()
        with torch.no_grad():
            enc = codec.encode(input_values, bandwidth=6.0)
        # enc.audio_codes: [num_chunks, B, num_codebooks, T]
        codes = enc.audio_codes[0, 0]  # [num_codebooks, T]
        first_book = codes[0].cpu()    # [T]  values in 0..1023

        text_bytes = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

        torch.save({
            "text_bytes": text_bytes,
            "audio_codes": first_book.long(),
            "sample_rate": sr,
            "duration_s": n / sr,
        }, args.out / f"sample_{idx:06d}.pt")
        idx += 1
        written_samples += n

        if idx % 50 == 0:
            print(f"  wrote {idx} clips, {written_samples / sr / 60:.1f} min")

    print(f"done: {idx} clips, ~{written_samples / sr / 60:.1f} min")


if __name__ == "__main__":
    main()
