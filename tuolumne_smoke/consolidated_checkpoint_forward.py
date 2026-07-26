#!/usr/bin/env python
"""Tuolumne smoke: reload a consolidated BLT checkpoint and score one batch."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from bytelatent.args import TrainArgs
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.train import compute_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing consolidated/consolidated.pth and params.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank != 0:
        raise RuntimeError("This checkpoint-forward probe is intentionally single-rank.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    params_path = os.path.join(args.checkpoint_dir, "params.json")
    state_path = os.path.join(
        args.checkpoint_dir, "consolidated", "consolidated.pth"
    )

    with open(params_path) as handle:
        train_args = TrainArgs.model_validate_json(handle.read())
    if train_args.model is None:
        raise ValueError("Expected a BLT model config, not an entropy-model config.")

    started = time.perf_counter()
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    model = ByteLatentTransformer(train_args.model)
    incompatible = model.load_state_dict(payload["model"], strict=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    load_seconds = time.perf_counter() - started

    loader = train_args.data.build_from_rank(rank=0, world_size=1)
    batch = next(loader.create_iter())
    batch_x = torch.from_numpy(batch.x).to(device)
    batch_y = torch.from_numpy(batch.y).to(device)
    patch_lengths = torch.from_numpy(batch.patch_lengths).to(device)
    mask = None if batch.mask is None else torch.from_numpy(batch.mask).to(device)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(batch_x, patch_lengths=patch_lengths)
        loss, _ = compute_loss(logits, batch_y, mask, scale=1.0)
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started

    if not math.isfinite(loss.item()):
        raise RuntimeError(f"Non-finite reload loss: {loss.item()}")
    report = {
        "checkpoint_dir": args.checkpoint_dir,
        "model_state_tensors": len(payload["model"]),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "input_shape": list(batch_x.shape),
        "patch_lengths": patch_lengths.tolist(),
        "logits_shape": list(logits.shape),
        "loss": loss.item(),
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "peak_memory_gib": round(
            torch.cuda.max_memory_allocated(device) / (1024**3), 3
        ),
    }
    print("CONSOLIDATED_CHECKPOINT_FORWARD " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
