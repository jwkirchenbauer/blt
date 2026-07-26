#!/usr/bin/env python3
"""Tuolumne smoke: run entropy-patched generation with Meta's native BLT."""

import argparse
import json
import os
import time

import torch
from huggingface_hub import hf_hub_download

from bytelatent.data.patcher import to_device
from bytelatent.distributed import DistributedArgs, setup_torch_distributed
from bytelatent.entropy_model import load_entropy_model
from bytelatent.generate_blt import generate_nocache
from bytelatent.hf import BltTokenizerAndPatcher
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.transformer import LMTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blt-repo", default="facebook/blt-1b")
    parser.add_argument("--entropy-repo", default="facebook/blt-entropy")
    parser.add_argument("--entropy-from-blt-repo", action="store_true")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-gen-len", type=int, default=16)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--shard-prompts-by-rank", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = args.prompt or ["A byte-latent transformer"]
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.shard_prompts_by_rank:
        if len(prompts) != world_size:
            raise ValueError(
                f"Prompt sharding needs one prompt per rank: "
                f"{len(prompts)} prompts for WORLD_SIZE={world_size}"
            )
        prompts = [prompts[rank]]
        print(f"Rank {rank} prompt: {prompts[0]!r}", flush=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)

    started = time.monotonic()
    if args.entropy_from_blt_repo:
        print(f"Loading nested entropy model from {args.blt_repo}", flush=True)
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
        entropy_dir = os.path.dirname(entropy_state)
        entropy_model, _ = load_entropy_model(entropy_dir, entropy_state)
    else:
        print(f"Loading entropy model from {args.entropy_repo}", flush=True)
        entropy_model = LMTransformer.from_pretrained(
            args.entropy_repo,
            cache_dir=args.cache_dir,
        )
    print(
        f"Loaded entropy model in {time.monotonic() - started:.2f}s",
        flush=True,
    )

    started = time.monotonic()
    print(f"Loading BLT model from {args.blt_repo}", flush=True)
    model = ByteLatentTransformer.from_pretrained(
        args.blt_repo,
        cache_dir=args.cache_dir,
    )
    tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(
        args.blt_repo,
        cache_dir=args.cache_dir,
    )
    print(f"Loaded BLT model in {time.monotonic() - started:.2f}s", flush=True)

    tokenizer = tok_and_patcher.tokenizer_args.build()
    if not isinstance(tokenizer, BltTokenizer):
        raise TypeError(f"Expected BltTokenizer, got {type(tokenizer).__name__}")
    patcher = tok_and_patcher.patcher_args.build()

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[tok_and_patcher.distributed_args.model_dtype]

    started = time.monotonic()
    model = model.to(device="cuda", dtype=dtype).eval()
    patcher.realtime_patching = True
    patcher.entropy_model, _ = to_device(
        entropy_model,
        tok_and_patcher.patcher_args.patching_device,
    )
    print(f"Moved models to GPU in {time.monotonic() - started:.2f}s", flush=True)

    distributed_args = DistributedArgs()
    distributed_args.configure_world()
    if not torch.distributed.is_initialized():
        setup_torch_distributed(distributed_args)

    try:
        for prompt in prompts:
            encoded = tokenizer.encode(prompt, add_eos=False)
            tokens = torch.tensor([encoded], dtype=torch.long, device="cuda")
            patch_lengths, entropies = patcher.patch(
                tokens,
                include_next_token=True,
            )
            nonzero_lengths = patch_lengths[0][patch_lengths[0] > 0].tolist()
            patch_report = {
                "prompt": prompt,
                "prompt_bytes": len(encoded),
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
            }
            print("PATCH_REPORT " + json.dumps(patch_report), flush=True)

        started = time.monotonic()
        outputs = generate_nocache(
            prompts,
            model=model,
            tokenizer=tokenizer,
            patcher=patcher,
            max_gen_len=args.max_gen_len,
            use_sampling=False,
        )
        elapsed = time.monotonic() - started
        for prompt, output in zip(prompts, outputs):
            print(
                "GENERATION "
                + json.dumps(
                    {
                        "prompt": prompt,
                        "completion": tokenizer.decode(output),
                        "generated_bytes": len(output),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        print(
            "INFERENCE_REPORT "
            + json.dumps(
                {
                    "generation_seconds": elapsed,
                    "max_gen_len": args.max_gen_len,
                    "peak_memory_gib": round(
                        torch.cuda.max_memory_allocated(local_rank) / 2**30,
                        3,
                    ),
                }
            ),
            flush=True,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
