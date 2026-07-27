#!/usr/bin/env python3
"""Diagnose ROCm primitives used by native BLT generation."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time

import torch
from xformers.ops import fmha

from bytelatent.model.blt import byte_group_hash_function
from bytelatent.model.utils import tokens_to_seqlen
from bytelatent.tokenizers.constants import BOS_ID, EOS_ID, OFFSET, PAD_ID


PROMPTS = [
    "The capital of France is",
    r"def fibonacci(n):\n    ",
    "Café naïve 日本語 — ",
    "Ths sentnce has typoss!!! ",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def encode(text: str) -> list[int]:
    return [BOS_ID, *(value + OFFSET for value in text.encode("utf-8"))]


def synchronize(stage: str) -> None:
    torch.cuda.synchronize()
    print(f"PRIMITIVE_STAGE_OK {stage}", flush=True)


def test_sequence_lengths(device: torch.device, iterations: int) -> dict:
    no_eos = torch.tensor(
        [encode(prompt)[:24] for prompt in PROMPTS],
        dtype=torch.long,
        device=device,
    )
    with_eos = no_eos.clone()
    with_eos[0, 6] = EOS_ID
    with_eos[1, 11] = EOS_ID
    with_eos[2, 4] = EOS_ID
    with_eos[2, 17] = EOS_ID

    expected_no_eos = tokens_to_seqlen(no_eos.cpu(), EOS_ID)
    expected_with_eos = tokens_to_seqlen(with_eos.cpu(), EOS_ID)
    for _ in range(iterations):
        observed_no_eos = tokens_to_seqlen(no_eos, EOS_ID)
        observed_with_eos = tokens_to_seqlen(with_eos, EOS_ID)
        if observed_no_eos != expected_no_eos:
            raise AssertionError((observed_no_eos, expected_no_eos))
        if observed_with_eos != expected_with_eos:
            raise AssertionError((observed_with_eos, expected_with_eos))
    synchronize("tokens_to_seqlen")
    return {
        "no_eos": expected_no_eos,
        "with_eos": expected_with_eos,
    }


def test_hashing(device: torch.device, iterations: int) -> dict:
    encoded = [encode(prompt) for prompt in PROMPTS]
    max_length = max(len(row) for row in encoded)
    tokens = torch.full(
        (len(encoded), max_length),
        PAD_ID,
        dtype=torch.long,
        device=device,
    )
    for row_id, row in enumerate(encoded):
        tokens[row_id, : len(row)] = torch.tensor(row, device=device)

    checksums = {}
    for group_size in range(3, 9):
        expected = byte_group_hash_function(
            tokens.cpu(),
            group_size=group_size,
            hash_func_nb=0,
            max_hash=500002,
        )
        for _ in range(iterations):
            observed = byte_group_hash_function(
                tokens,
                group_size=group_size,
                hash_func_nb=0,
                max_hash=500002,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(observed.cpu(), expected, rtol=0, atol=0)
        checksums[str(group_size)] = int(expected.sum().item())
        print(f"PRIMITIVE_STAGE_OK hash_group_{group_size}", flush=True)
    return checksums


def test_xformers(device: torch.device, iterations: int) -> dict:
    reports = {}
    for batch_size in (1, 4):
        for seq_len in (24, 28, 32):
            shape = (batch_size, seq_len, 16, 64)
            query = torch.randn(shape, dtype=torch.bfloat16, device=device)
            key = torch.randn(shape, dtype=torch.bfloat16, device=device)
            value = torch.randn(shape, dtype=torch.bfloat16, device=device)
            bias = fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
                [seq_len] * batch_size
            ).make_local_attention(512)
            query_flat = query.reshape(1, batch_size * seq_len, 16, 64)
            key_flat = key.reshape(1, batch_size * seq_len, 16, 64)
            value_flat = value.reshape(1, batch_size * seq_len, 16, 64)

            output = None
            for _ in range(iterations):
                output = fmha.memory_efficient_attention(
                    query_flat,
                    key_flat,
                    value_flat,
                    attn_bias=bias,
                )
                torch.cuda.synchronize()
                if not torch.isfinite(output).all().item():
                    raise RuntimeError(
                        f"Non-finite xFormers output for batch={batch_size}, "
                        f"sequence={seq_len}"
                    )
            assert output is not None
            key_name = f"batch_{batch_size}_sequence_{seq_len}"
            reports[key_name] = {
                "shape": list(output.shape),
                "mean": float(output.float().mean().item()),
            }
            print(f"PRIMITIVE_STAGE_OK xformers_{key_name}", flush=True)
    return reports


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    properties = torch.cuda.get_device_properties(device)
    header = {
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device_name": properties.name,
        "device_total_memory": properties.total_memory,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "iterations": args.iterations,
    }
    print("PRIMITIVE_DIAGNOSTIC_START " + json.dumps(header), flush=True)

    started = time.monotonic()
    report = {
        **header,
        "sequence_lengths": test_sequence_lengths(device, args.iterations),
        "hash_checksums": test_hashing(device, args.iterations),
        "xformers": test_xformers(device, args.iterations),
        "seconds": time.monotonic() - started,
    }
    print("PRIMITIVE_DIAGNOSTIC_PASS " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
