#!/usr/bin/env python3
"""Tuolumne smoke: generate with the Hugging Face Transformers BLT conversion."""

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer, BltForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="itazap/blt-1b-hf")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        args.repo,
        cache_dir=args.cache_dir,
    )
    model = BltForCausalLM.from_pretrained(
        args.repo,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    ).eval()
    load_seconds = time.monotonic() - started
    print(f"Loaded Transformers BLT in {load_seconds:.2f}s", flush=True)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(
        torch.device("cuda", local_rank)
    )
    prompt_length = inputs.input_ids.shape[1]
    torch.cuda.reset_peak_memory_stats(local_rank)

    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=False,
        )
    torch.cuda.synchronize(local_rank)
    generation_seconds = time.monotonic() - started

    generated_ids = output[0, prompt_length:]
    completion = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(
        "TRANSFORMERS_GENERATION "
        + json.dumps(
            {
                "prompt": args.prompt,
                "completion": completion,
                "generated_bytes": int(generated_ids.numel()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(
        "TRANSFORMERS_REPORT "
        + json.dumps(
            {
                "load_seconds": load_seconds,
                "generation_seconds": generation_seconds,
                "peak_memory_gib": round(
                    torch.cuda.max_memory_allocated(local_rank) / 2**30,
                    3,
                ),
                "torch_dtype": str(model.dtype),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
