#!/usr/bin/env python3
"""Tuolumne smoke: score strings with Meta's pretrained BLT entropy patcher."""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from bytelatent.data.patcher import to_device
from bytelatent.entropy_model import load_entropy_model
from bytelatent.hf import BltTokenizerAndPatcher
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer


DEFAULT_TEXTS = [
    "The capital of France is Paris.",
    "BLT predicts UTF-8 bytes directly.",
    "Unicode: café, π, 東京, and 🚀.",
    "def add(a, b):\n    return a + b",
    "Noise?! 001101... @@ byte_boundary_test ##",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blt-repo", default="facebook/blt-1b")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--text",
        action="append",
        help="Text to score; repeat for multiple examples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = args.text or DEFAULT_TEXTS
    os.makedirs(args.cache_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)

    load_started = time.monotonic()
    entropy_state = hf_hub_download(
        args.blt_repo,
        "entropy_model/consolidated.pth",
        cache_dir=args.cache_dir,
    )
    hf_hub_download(
        args.blt_repo,
        "entropy_model/params.json",
        cache_dir=args.cache_dir,
    )
    entropy_model, _ = load_entropy_model(
        os.path.dirname(entropy_state),
        entropy_state,
    )

    model = ByteLatentTransformer.from_pretrained(
        args.blt_repo,
        cache_dir=args.cache_dir,
    )
    tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(
        args.blt_repo,
        cache_dir=args.cache_dir,
    )
    tokenizer = tok_and_patcher.tokenizer_args.build()
    if not isinstance(tokenizer, BltTokenizer):
        raise TypeError(f"Expected BltTokenizer, got {type(tokenizer).__name__}")
    patcher = tok_and_patcher.patcher_args.build()

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[tok_and_patcher.distributed_args.model_dtype]
    model = model.to(device="cuda", dtype=dtype).eval()
    patcher.realtime_patching = True
    patcher.entropy_model, patching_device = to_device(
        entropy_model,
        tok_and_patcher.patcher_args.patching_device,
    )
    entropy_model.eval()
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - load_started

    total_nll = 0.0
    total_targets = 0
    total_patches = 0
    score_started = time.monotonic()
    with torch.inference_mode():
        for item_id, text in enumerate(texts):
            encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
            tokens = torch.tensor(
                encoded,
                dtype=torch.long,
                device="cuda",
            ).unsqueeze(0)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            patch_lengths, entropies = patcher.patch(
                inputs,
                include_next_token=True,
            )
            logits = model(inputs, patch_lengths=patch_lengths)
            nll = F.cross_entropy(
                logits.flatten(0, 1).float(),
                targets.flatten(0, 1),
                reduction="sum",
            ).item()

            nonzero_lengths = patch_lengths[0][patch_lengths[0] > 0].tolist()
            target_count = targets.numel()
            record = {
                "item": item_id,
                "text": text,
                "utf8_bytes": len(text.encode("utf-8")),
                "predicted_tokens": target_count,
                "patch_count": len(nonzero_lengths),
                "patch_lengths": nonzero_lengths,
                "mean_patch_length": (
                    sum(nonzero_lengths) / len(nonzero_lengths)
                    if nonzero_lengths
                    else 0.0
                ),
                "entropy_min": float(entropies.min().item()),
                "entropy_mean": float(entropies.mean().item()),
                "entropy_max": float(entropies.max().item()),
                "nll": nll,
                "bpb": nll / math.log(2) / target_count,
            }
            print(
                "PRETRAINED_EVAL_ITEM "
                + json.dumps(record, ensure_ascii=False),
                flush=True,
            )
            total_nll += nll
            total_targets += target_count
            total_patches += len(nonzero_lengths)

    torch.cuda.synchronize()
    score_seconds = time.monotonic() - score_started
    summary = {
        "model": args.blt_repo,
        "examples": len(texts),
        "predicted_tokens": total_targets,
        "patches": total_patches,
        "aggregate_nll": total_nll,
        "aggregate_bpb": total_nll / math.log(2) / total_targets,
        "load_seconds": load_seconds,
        "score_seconds": score_seconds,
        "patching_device": str(patching_device),
        "dtype": str(dtype),
        "peak_memory_gib": round(
            torch.cuda.max_memory_allocated(local_rank) / 2**30,
            3,
        ),
    }
    print(
        "PRETRAINED_EVAL_SUMMARY "
        + json.dumps(summary, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
