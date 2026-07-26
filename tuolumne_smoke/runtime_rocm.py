#!/usr/bin/env python3
"""Tuolumne smoke: validate the one-node ROCm runtime used by BLT."""

import json
import os
import socket
from datetime import timedelta
from importlib.metadata import version

import torch
import torch.distributed as dist
import xformers.ops as xops


def required_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Required launcher variable {name} is unset")
    return int(value)


def main() -> None:
    rank = required_int("RANK")
    local_rank = required_int("LOCAL_RANK")
    world_size = required_int("WORLD_SIZE")

    if world_size != 4:
        raise RuntimeError(f"Expected WORLD_SIZE=4, got {world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access ROCm GPUs")
    if torch.cuda.device_count() != 4:
        raise RuntimeError(
            f"Expected four visible GPUs, got {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",  # PyTorch's NCCL backend uses RCCL on ROCm.
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )

    try:
        torch.manual_seed(1234 + rank)
        torch.cuda.reset_peak_memory_stats(device)

        properties = torch.cuda.get_device_properties(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not reported as supported")

        left = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        right = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        product = left @ right
        if not torch.isfinite(product).all().item():
            raise RuntimeError("BF16 matrix multiplication produced non-finite values")

        reduced = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        if reduced.item() != 10.0:
            raise RuntimeError(f"All-reduce returned {reduced.item()}, expected 10")

        shape = (2, 128, 8, 64)  # batch, sequence, heads, head dimension
        query = torch.randn(
            shape, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        key = torch.randn(
            shape, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        value = torch.randn(
            shape, device=device, dtype=torch.bfloat16, requires_grad=True
        )

        attention = xops.memory_efficient_attention(
            query,
            key,
            value,
            op=xops.MemoryEfficientAttentionCkOp,
        )
        attention.float().square().mean().backward()
        torch.cuda.synchronize(device)

        if not torch.isfinite(attention).all().item():
            raise RuntimeError("CK attention forward produced non-finite values")
        if query.grad is None or not torch.isfinite(query.grad).all().item():
            raise RuntimeError(
                "CK attention backward failed or produced non-finite values"
            )

        report = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "host": socket.gethostname(),
            "logical_device": torch.cuda.current_device(),
            "visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "gpu_name": properties.name,
            "gpu_arch": getattr(properties, "gcnArchName", "unknown"),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "xformers": version("xformers"),
            "xformers_forward": xops.MemoryEfficientAttentionCkOp[0].NAME,
            "xformers_backward": xops.MemoryEfficientAttentionCkOp[1].NAME,
            "all_reduce": reduced.item(),
            "peak_memory_mib": round(
                torch.cuda.max_memory_allocated(device) / 2**20, 2
            ),
        }
        print(json.dumps(report, sort_keys=True), flush=True)

        dist.barrier()
        if rank == 0:
            print("BLT ROCm runtime smoke test: PASS", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
