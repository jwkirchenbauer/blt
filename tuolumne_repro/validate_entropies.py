#!/usr/bin/env python3
"""Validate Arrow scores, calibrate a threshold, and audit rank-local capacity."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bytelatent.preprocess.preprocess_entropies import get_id_from_doc


ALLOWED_ROOTS = (
    Path("/p/vast1/kirchenb/hlm-root"),
    Path("/p/vast1/pretrain/datasets/blt"),
)
FLOAT16_VALUES = np.arange(1 << 16, dtype=np.uint16).view(np.float16)


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


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def global_patch_count(
    histogram: np.ndarray,
    *,
    threshold: float,
    mandatory_patches: int,
) -> int:
    finite = np.isfinite(FLOAT16_VALUES)
    dynamic = int(histogram[finite & (FLOAT16_VALUES > threshold)].sum())
    return mandatory_patches + dynamic


def choose_threshold(
    histogram: np.ndarray,
    *,
    total_tokens: int,
    mandatory_patches: int,
    target_patch_size: float,
) -> dict[str, Any]:
    finite_values = FLOAT16_VALUES[np.isfinite(FLOAT16_VALUES)]
    finite_counts = histogram[np.isfinite(FLOAT16_VALUES)]
    occupied = finite_counts > 0
    values = finite_values[occupied].astype(np.float32)
    counts = finite_counts[occupied].astype(np.int64)
    if len(values) == 0:
        raise ValueError("No finite thresholdable entropy values")

    order = np.argsort(values)
    values = values[order]
    counts = counts[order]
    counts_greater = counts.sum() - np.cumsum(counts)
    patch_counts = mandatory_patches + counts_greater
    realized_sizes = total_tokens / patch_counts
    best = int(np.argmin(np.abs(realized_sizes - target_patch_size)))
    threshold = float(values[best])
    patch_count = int(patch_counts[best])
    return {
        "threshold": threshold,
        "target_patch_size": target_patch_size,
        "realized_patch_size": total_tokens / patch_count,
        "total_tokens": total_tokens,
        "patches": patch_count,
        "mandatory_patches": mandatory_patches,
        "dynamic_patches": patch_count - mandatory_patches,
        "threshold_tie_count": int(counts[best]),
        "rule": (
            "first two positions of every document are patch starts; among "
            "remaining non-final positions, entropy > threshold starts a patch"
        ),
    }


def validate_file(
    raw_path: Path,
    arrow_path: Path,
    histogram: np.ndarray,
    capacity_workers_per_chunk: int,
) -> dict[str, Any]:
    expected_schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("entropies", pa.list_(pa.float16()), nullable=False),
        ]
    )
    records = 0
    text_bytes = 0
    token_positions = 0
    thresholdable_positions = 0
    entropy_min = math.inf
    entropy_max = -math.inf
    entropy_sum = 0.0
    entropy_count = 0
    file_histogram = np.zeros(1 << 16, dtype=np.int64)
    worker_histograms = np.zeros(
        (capacity_workers_per_chunk, 1 << 16), dtype=np.int64
    )
    worker_records = [0] * capacity_workers_per_chunk
    worker_token_positions = [0] * capacity_workers_per_chunk

    with raw_path.open(encoding="utf-8") as raw_handle, arrow_path.open("rb") as source:
        reader = pa.ipc.open_file(source)
        if not reader.schema.equals(expected_schema):
            raise ValueError(
                f"Unexpected schema in {arrow_path}: {reader.schema}"
            )
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index).to_pydict()
            for sample_id, text, entropy_list in zip(
                batch["sample_id"],
                batch["text"],
                batch["entropies"],
                strict=True,
            ):
                line = raw_handle.readline()
                if not line:
                    raise ValueError(f"Arrow has extra row {records} in {arrow_path}")
                raw = json.loads(line)
                if sample_id != get_id_from_doc(raw):
                    raise ValueError(f"ID mismatch at row {records} in {arrow_path}")
                if text != raw["text"]:
                    raise ValueError(f"Text mismatch at row {records} in {arrow_path}")

                expected_length = len(text.encode("utf-8")) + 2
                entropies = np.asarray(entropy_list, dtype=np.float16)
                if len(entropies) != expected_length:
                    raise ValueError(
                        f"Entropy length {len(entropies)} != token length "
                        f"{expected_length} at row {records} in {arrow_path}"
                    )
                if not np.isfinite(entropies).all():
                    raise ValueError(
                        f"Non-finite entropy at row {records} in {arrow_path}"
                    )

                # Training calls the patcher with include_next_token=False.
                # It fixes starts at token positions 0 and 1, then thresholds
                # stored entropy positions [1:-1].
                candidates = np.ascontiguousarray(entropies[1:-1])
                if len(candidates):
                    counts = np.bincount(
                        candidates.view(np.uint16), minlength=1 << 16
                    ).astype(np.int64)
                    histogram += counts
                    file_histogram += counts
                    worker_histograms[records % capacity_workers_per_chunk] += counts
                entropy32 = entropies.astype(np.float32)
                entropy_min = min(entropy_min, float(entropy32.min()))
                entropy_max = max(entropy_max, float(entropy32.max()))
                entropy_sum += float(entropy32.sum(dtype=np.float64))
                entropy_count += len(entropies)
                thresholdable_positions += len(candidates)
                text_bytes += len(text.encode("utf-8"))
                token_positions += expected_length
                worker_id = records % capacity_workers_per_chunk
                worker_records[worker_id] += 1
                worker_token_positions[worker_id] += expected_length
                records += 1

        if raw_handle.readline():
            raise ValueError(f"Raw JSONL has extra rows after {records}: {raw_path}")

    return {
        "raw": str(raw_path),
        "arrow": str(arrow_path),
        "raw_bytes": raw_path.stat().st_size,
        "arrow_bytes": arrow_path.stat().st_size,
        "records": records,
        "utf8_text_bytes": text_bytes,
        "token_positions": token_positions,
        "thresholdable_positions": thresholdable_positions,
        "entropy_min": entropy_min,
        "entropy_max": entropy_max,
        "entropy_mean": entropy_sum / entropy_count,
        # Removed before serialization after the global threshold is known.
        "_threshold_histogram": file_histogram,
        "_capacity_worker_histograms": worker_histograms,
        "_capacity_worker_records": worker_records,
        "_capacity_worker_token_positions": worker_token_positions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--entropy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-patch-size", type=float, default=4.5)
    parser.add_argument(
        "--diagnostic-threshold",
        type=float,
        default=1.335442066192627,
    )
    parser.add_argument("--capacity-batch-size", type=int, default=16)
    parser.add_argument("--capacity-seq-len", type=int, default=4096)
    parser.add_argument("--capacity-buffer-size", type=int, default=512)
    parser.add_argument("--capacity-steps", type=int, default=5299)
    parser.add_argument(
        "--capacity-workers-per-chunk",
        type=int,
        default=1,
        help=(
            "Number of BLT strided Arrow workers assigned to each input chunk; "
            "use world_size / number_of_chunks"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = checked_path(args.input_dir)
    entropy_dir = checked_path(args.entropy_dir)
    output = checked_path(args.output)
    if args.capacity_workers_per_chunk < 1:
        raise ValueError("--capacity-workers-per-chunk must be positive")
    raw_files = sorted(input_dir.glob("*.chunk.*.jsonl"))
    arrow_files = sorted(entropy_dir.glob("*.chunk.*.jsonl.shard_*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No Arrow files in {entropy_dir}")

    raw_by_name = {path.name: path for path in raw_files}
    histogram = np.zeros(1 << 16, dtype=np.int64)
    files: list[dict[str, Any]] = []
    for arrow_path in arrow_files:
        raw_name = arrow_path.name.split(".shard_", maxsplit=1)[0]
        raw_path = raw_by_name.get(raw_name)
        if raw_path is None:
            raise FileNotFoundError(f"No raw file corresponding to {arrow_path}")
        complete = Path(f"{arrow_path}.complete")
        if not complete.is_file():
            raise FileNotFoundError(complete)
        print(f"Validating {arrow_path}", flush=True)
        files.append(
            validate_file(
                raw_path,
                arrow_path,
                histogram,
                args.capacity_workers_per_chunk,
            )
        )

    totals = {
        key: sum(file_result[key] for file_result in files)
        for key in (
            "raw_bytes",
            "arrow_bytes",
            "records",
            "utf8_text_bytes",
            "token_positions",
            "thresholdable_positions",
        )
    }
    mandatory_patches = 2 * totals["records"]
    calibrated = choose_threshold(
        histogram,
        total_tokens=totals["token_positions"],
        mandatory_patches=mandatory_patches,
        target_patch_size=args.target_patch_size,
    )
    rank_stream_patch_counts = []
    for file_result in files:
        file_histogram = file_result.pop("_threshold_histogram")
        worker_histograms = file_result.pop("_capacity_worker_histograms")
        worker_records = file_result.pop("_capacity_worker_records")
        worker_token_positions = file_result.pop(
            "_capacity_worker_token_positions"
        )
        file_mandatory_patches = 2 * file_result["records"]
        file_patches = global_patch_count(
            file_histogram,
            threshold=calibrated["threshold"],
            mandatory_patches=file_mandatory_patches,
        )
        file_result["patches_at_calibrated_threshold"] = file_patches
        file_result["realized_patch_size_at_calibrated_threshold"] = (
            file_result["token_positions"] / file_patches
        )
        capacity_workers = []
        for worker_id in range(args.capacity_workers_per_chunk):
            worker_patches = global_patch_count(
                worker_histograms[worker_id],
                threshold=calibrated["threshold"],
                mandatory_patches=2 * worker_records[worker_id],
            )
            rank_stream_patch_counts.append(worker_patches)
            capacity_workers.append(
                {
                    "worker_id": worker_id,
                    "num_workers": args.capacity_workers_per_chunk,
                    "records": worker_records[worker_id],
                    "token_positions": worker_token_positions[worker_id],
                    "patches_at_calibrated_threshold": worker_patches,
                    "realized_patch_size_at_calibrated_threshold": (
                        worker_token_positions[worker_id] / worker_patches
                    ),
                }
            )
        file_result["capacity_workers"] = capacity_workers
        if sum(item["records"] for item in capacity_workers) != file_result["records"]:
            raise RuntimeError("Capacity-worker record partition is not lossless")
        if (
            sum(item["token_positions"] for item in capacity_workers)
            != file_result["token_positions"]
        ):
            raise RuntimeError("Capacity-worker token partition is not lossless")
        if (
            sum(
                item["patches_at_calibrated_threshold"]
                for item in capacity_workers
            )
            != file_patches
        ):
            raise RuntimeError("Capacity-worker patch partition is not lossless")

    if sum(rank_stream_patch_counts) != calibrated["patches"]:
        raise RuntimeError("Rank-stream patches do not reconstruct the global total")

    if args.capacity_buffer_size % args.capacity_batch_size != 0:
        raise ValueError(
            "--capacity-buffer-size must be divisible by --capacity-batch-size"
        )
    patches_per_rank_step = args.capacity_batch_size * args.capacity_seq_len
    patches_per_buffer = args.capacity_buffer_size * args.capacity_seq_len
    optimizer_steps_per_buffer = (
        args.capacity_buffer_size // args.capacity_batch_size
    )
    stream_nominal_optimizer_steps = [
        patch_count // patches_per_rank_step
        for patch_count in rank_stream_patch_counts
    ]
    stream_complete_buffers = [
        patch_count // patches_per_buffer
        for patch_count in rank_stream_patch_counts
    ]
    stream_guaranteed_optimizer_steps = [
        buffers * optimizer_steps_per_buffer
        for buffers in stream_complete_buffers
    ]
    required_complete_buffers = math.ceil(
        args.capacity_steps / optimizer_steps_per_buffer
    )
    rank_stream_patch_counts_array = np.asarray(
        rank_stream_patch_counts, dtype=np.float64
    )
    rank_stream_capacity = {
        "input_chunks": len(files),
        "workers_per_chunk": args.capacity_workers_per_chunk,
        "rank_streams": len(rank_stream_patch_counts),
        "assignment": (
            "Arrow row indices where row_index % workers_per_chunk == worker_id"
        ),
        "batch_size_per_rank": args.capacity_batch_size,
        "patches_per_sequence": args.capacity_seq_len,
        "sequences_per_shuffle_buffer": args.capacity_buffer_size,
        "patches_per_rank_step": patches_per_rank_step,
        "patches_per_shuffle_buffer": patches_per_buffer,
        "optimizer_steps_per_complete_shuffle_buffer": optimizer_steps_per_buffer,
        "requested_optimizer_steps": args.capacity_steps,
        "nominal_required_patches_per_rank": (
            args.capacity_steps * patches_per_rank_step
        ),
        "complete_shuffle_buffers_required": required_complete_buffers,
        "buffer_safe_required_patches_per_rank": (
            required_complete_buffers * patches_per_buffer
        ),
        "minimum_rank_stream_patches": int(min(rank_stream_patch_counts)),
        "maximum_rank_stream_patches": int(max(rank_stream_patch_counts)),
        "mean_rank_stream_patches": float(rank_stream_patch_counts_array.mean()),
        "rank_stream_patch_count_cv": float(
            rank_stream_patch_counts_array.std()
            / rank_stream_patch_counts_array.mean()
        ),
        "rank_stream_patch_count_max_min_ratio": (
            max(rank_stream_patch_counts) / min(rank_stream_patch_counts)
        ),
        "minimum_nominal_optimizer_steps_before_repeat": int(
            min(stream_nominal_optimizer_steps)
        ),
        "maximum_nominal_optimizer_steps_before_repeat": int(
            max(stream_nominal_optimizer_steps)
        ),
        "minimum_buffer_safe_optimizer_steps_before_repeat": int(
            min(stream_guaranteed_optimizer_steps)
        ),
        "maximum_buffer_safe_optimizer_steps_before_repeat": int(
            max(stream_guaranteed_optimizer_steps)
        ),
        "all_rank_streams_cover_requested_steps_without_buffer_repeat": (
            min(stream_guaranteed_optimizer_steps) >= args.capacity_steps
        ),
    }
    diagnostic_patches = global_patch_count(
        histogram,
        threshold=args.diagnostic_threshold,
        mandatory_patches=mandatory_patches,
    )
    diagnostic = {
        "threshold": args.diagnostic_threshold,
        "realized_patch_size": totals["token_positions"] / diagnostic_patches,
        "patches": diagnostic_patches,
    }
    result = {
        "schema_version": 3,
        "created_at": utc_now(),
        "input_dir": str(input_dir),
        "entropy_dir": str(entropy_dir),
        "files": files,
        "totals": totals,
        "arrow_to_raw_byte_ratio": totals["arrow_bytes"] / totals["raw_bytes"],
        "calibrated_global_threshold": calibrated,
        "rank_local_capacity_at_calibrated_threshold": rank_stream_capacity,
        "released_threshold_diagnostic": diagnostic,
    }
    atomic_write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
