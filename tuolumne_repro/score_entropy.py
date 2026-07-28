#!/usr/bin/env python3
"""Rank-parallel entropy scoring with the repository's exact file scorer.

Each rank owns a deterministic strided subset of the input chunks.  The only
systems change relative to ``preprocess_entropies.main`` is caching the loaded
entropy model when a rank processes more than one input file.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bytelatent.preprocess.preprocess_entropies as scorer


DEFAULT_CHECKPOINT = Path(
    "/p/vast1/kirchenb/hlm-root/hf-cache/hub/"
    "models--facebook--blt-1b/snapshots/"
    "8134b32f0b1d25d1248c30e8c7bdfd442d3bb380/entropy_model"
)
ALLOWED_ROOTS = (
    Path("/p/vast1/kirchenb/hlm-root"),
    Path("/p/vast1/pretrain/datasets/blt"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def checked_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not any(is_relative_to(resolved, root.resolve()) for root in ALLOWED_ROOTS):
        raise ValueError(f"Refusing path outside the authorized trees: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--entropy-checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--entropy-model-name", default="transformer_100m")
    parser.add_argument(
        "--max-files-per-rank",
        type=int,
        help="Bounded rehearsal only; omit to process every assigned chunk",
    )
    parser.add_argument("--log-step", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = checked_path(args.input_dir)
    output_root = checked_path(args.output_root)
    checkpoint_dir = checked_path(args.entropy_checkpoint_dir)
    state_path = checkpoint_dir / "consolidated.pth"
    params_path = checkpoint_dir / "params.json"
    for required in (state_path, params_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank={rank}, world_size={world_size}")
    torch.cuda.set_device(local_rank)

    input_files = sorted(input_dir.glob("*.chunk.*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No BLT chunks in {input_dir}")
    assigned = input_files[rank::world_size]
    if args.max_files_per_rank is not None:
        if args.max_files_per_rank < 1:
            raise ValueError("--max-files-per-rank must be positive")
        assigned = assigned[: args.max_files_per_rank]
    if not assigned:
        raise RuntimeError(f"No input files assigned to rank {rank}")

    dataset_name = input_files[0].name.split(".chunk.", maxsplit=1)[0]
    output_dir = output_root / dataset_name / args.entropy_model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    original_loader = scorer.load_entropy_model

    @functools.lru_cache(maxsize=1)
    def cached_loader(
        entropy_model_checkpoint_dir: str,
        state_dict_path: str,
        device: str = "cpu",
    ):
        return original_loader(
            entropy_model_checkpoint_dir,
            state_dict_path,
            device=device,
        )

    # ``main`` resolves this module global at call time.  Reusing the exact
    # model object avoids checkpoint reloads without changing tokenization,
    # scoring, FP16 conversion, Arrow schema, or record order.
    scorer.load_entropy_model = cached_loader

    run_started = time.monotonic()
    file_results: list[dict[str, Any]] = []
    for input_file in assigned:
        output_file = output_dir / f"{input_file.name}.shard_00.arrow"
        complete_file = Path(f"{output_file}.complete")
        if complete_file.exists():
            if not output_file.is_file():
                raise RuntimeError(f"Orphan completion marker: {complete_file}")
            print(f"Skipping completed output {output_file}", flush=True)
            file_results.append(
                {
                    "input": str(input_file),
                    "output": str(output_file),
                    "status": "already_complete",
                }
            )
            continue
        if output_file.exists():
            raise FileExistsError(
                f"Refusing incomplete output {output_file}; inspect it before removal"
            )

        started = time.monotonic()
        scorer.main(
            input_file=str(input_file),
            output_file=str(output_file),
            patching_device="cuda",
            log_step=args.log_step,
            entropy_model_checkpoint_dir=str(checkpoint_dir),
            entropy_model_state_dict_path=str(state_path),
            # BLT's byte tokenizer does not read this unless BPE delimiters
            # are explicitly enabled.
            bpe_tokenizer_path="/dev/null",
        )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        result = {
            "input": str(input_file),
            "input_bytes": input_file.stat().st_size,
            "output": str(output_file),
            "output_bytes": output_file.stat().st_size,
            "elapsed_seconds": elapsed,
            "status": "completed",
        }
        file_results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "host": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "cuda_device_name": torch.cuda.get_device_name(local_rank),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "entropy_checkpoint_dir": str(checkpoint_dir),
        "entropy_params_sha256": sha256_file(params_path),
        "entropy_state_sha256": sha256_file(state_path),
        "assignment": "sorted_input_files[rank::world_size]",
        "max_files_per_rank": args.max_files_per_rank,
        "elapsed_seconds": time.monotonic() - run_started,
        "files": file_results,
    }
    summary_path = (
        output_root
        / "_run_summaries"
        / f"world_{world_size:04d}.rank_{rank:04d}.json"
    )
    atomic_write_json(summary_path, summary)
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
