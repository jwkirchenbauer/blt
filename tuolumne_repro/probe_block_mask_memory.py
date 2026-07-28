#!/usr/bin/env python3
"""Compare BLT's dense-closure and direct BlockMask constructions on one GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from timeit import default_timer as timer

import torch
from torch.nn.attention.flex_attention import create_mask, flex_attention

from bytelatent.model.blt import cross_attn_mask

flex_attention_compiled = torch.compile(flex_attention)


def make_patch_layout(
    batch_size: int,
    num_patches: int,
    patch_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    patch_lengths = torch.full(
        (batch_size, num_patches),
        patch_length,
        dtype=torch.int64,
        device=device,
    )
    patch_ids = torch.arange(
        num_patches, dtype=torch.int64, device=device
    ).repeat_interleave(patch_length)
    patch_ids = patch_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
    return patch_ids, patch_lengths, num_patches * patch_length


def build_mask(
    patch_ids: torch.Tensor,
    patch_lengths: torch.Tensor,
    token_length: int,
    *,
    patches_as_queries: bool,
    direct: bool,
):
    os.environ["BLT_DIRECT_BLOCK_MASK"] = "1" if direct else "0"
    return cross_attn_mask(
        patch_ids,
        patch_lengths,
        token_length,
        patches_as_queries=patches_as_queries,
        cross_attn_k=2,
        window=None,
        block_mask=True,
    )


def check_exact_relation(device: torch.device) -> dict[str, float]:
    patch_ids, patch_lengths, token_length = make_patch_layout(
        batch_size=2,
        num_patches=16,
        patch_length=6,
        device=device,
    )
    maximum_output_difference = 0.0
    for patches_as_queries in (False, True):
        reference = build_mask(
            patch_ids,
            patch_lengths,
            token_length,
            patches_as_queries=patches_as_queries,
            direct=False,
        )
        direct = build_mask(
            patch_ids,
            patch_lengths,
            token_length,
            patches_as_queries=patches_as_queries,
            direct=True,
        )
        q_len = patch_lengths.shape[1] * 2 if patches_as_queries else token_length
        kv_len = token_length if patches_as_queries else patch_lengths.shape[1] * 2
        reference_dense = create_mask(
            reference.mask_mod, 2, 1, q_len, kv_len, device
        )
        direct_dense = create_mask(direct.mask_mod, 2, 1, q_len, kv_len, device)
        if not torch.equal(reference_dense, direct_dense):
            raise AssertionError("Direct and reference element masks differ")
        if not torch.equal(reference.to_dense(), direct.to_dense()):
            raise AssertionError("Direct and reference BlockMask metadata differ")

        torch.manual_seed(42)
        query = torch.randn(
            2, 2, q_len, 64, dtype=torch.bfloat16, device=device
        )
        key = torch.randn(
            2, 2, kv_len, 64, dtype=torch.bfloat16, device=device
        )
        value = torch.randn(
            2, 2, kv_len, 64, dtype=torch.bfloat16, device=device
        )
        reference_output = flex_attention_compiled(
            query, key, value, block_mask=reference
        )
        direct_output = flex_attention_compiled(
            query, key, value, block_mask=direct
        )
        difference = (
            (reference_output.float() - direct_output.float()).abs().max().item()
        )
        maximum_output_difference = max(maximum_output_difference, difference)
        torch.testing.assert_close(reference_output, direct_output, rtol=0, atol=0)
    return {"maximum_output_difference": maximum_output_difference}


def measure_official_shape(
    device: torch.device,
    *,
    direct: bool,
) -> dict[str, float | int | bool]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    patch_ids, patch_lengths, token_length = make_patch_layout(
        batch_size=16,
        num_patches=4096,
        patch_length=6,
        device=device,
    )
    torch.cuda.synchronize()
    start = timer()
    encoder_mask = build_mask(
        patch_ids,
        patch_lengths,
        token_length,
        patches_as_queries=True,
        direct=direct,
    )
    decoder_mask = build_mask(
        patch_ids,
        patch_lengths,
        token_length,
        patches_as_queries=False,
        direct=direct,
    )
    torch.cuda.synchronize()
    elapsed = timer() - start
    result = {
        "direct": direct,
        "seconds": elapsed,
        "active_gib": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
        "peak_active_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "encoder_sparsity": encoder_mask.sparsity(),
        "decoder_sparsity": decoder_mask.sparsity(),
    }
    del encoder_mask, decoder_mask, patch_ids, patch_lengths
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A compute-node GPU is required")
    device = torch.device("cuda")
    report = {
        "device": torch.cuda.get_device_name(device),
        "exact_relation": check_exact_relation(device),
        "official_shape": [
            measure_official_shape(device, direct=False),
            measure_official_shape(device, direct=True),
        ],
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        workspace = Path("/p/vast1/kirchenb/hlm-root")
        if not output.is_relative_to(workspace):
            raise ValueError(f"Output must be inside {workspace}: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        temporary.write_text(report_text, encoding="utf-8")
        os.replace(temporary, output)
    print(report_text, end="", flush=True)


if __name__ == "__main__":
    main()
