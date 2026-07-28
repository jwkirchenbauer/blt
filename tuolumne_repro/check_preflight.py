#!/usr/bin/env python3
"""Read-only recipe and artifact gates before BLT-1B training."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bytelatent.args import TrainArgs
from bytelatent.config_parser import parse_args_to_pydantic_model


RELEASE_ROOT = (
    REPO_ROOT
    / ".."
    / "hf-cache"
    / "hub"
    / "models--facebook--blt-1b"
    / "snapshots"
    / "8134b32f0b1d25d1248c30e8c7bdfd442d3bb380"
)
MAIN_CONFIG = REPO_ROOT / "tuolumne_repro/configs/blt_1b_32n_100b.yaml"
ENTROPY_CONFIG = REPO_ROOT / "tuolumne_repro/configs/entropy_100m_2n.yaml"
DEFAULT_VALIDATION = Path(
    "/p/vast1/pretrain/datasets/blt/entropy/dclm-pilot-terashuf-v1/"
    "official/validation_all_64_world128.json"
)
ALLOWED_OUTPUT_ROOTS = (
    Path("/p/vast1/kirchenb/hlm-root"),
    Path("/p/vast1/pretrain/datasets/blt"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checked_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not any(resolved.is_relative_to(root) for root in ALLOWED_OUTPUT_ROOTS):
        raise ValueError(f"Output is outside the authorized trees: {resolved}")
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def parse_config(path: Path, overrides: dict[str, Any] | None = None) -> TrainArgs:
    config = OmegaConf.create({"config": str(path)})
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.create(overrides))
    # The upstream parser prints its fully expanded config. That is useful for
    # training logs but would obscure this concise preflight report.
    with contextlib.redirect_stdout(io.StringIO()):
        return parse_args_to_pydantic_model(TrainArgs, cli_args=config)


def nested_differences(
    expected: Any, observed: Any, prefix: str = ""
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key in sorted(set(expected) | set(observed)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in expected:
                differences.append(
                    {"path": path, "expected": "<missing>", "observed": observed[key]}
                )
            elif key not in observed:
                differences.append(
                    {"path": path, "expected": expected[key], "observed": "<missing>"}
                )
            else:
                differences.extend(
                    nested_differences(expected[key], observed[key], path)
                )
    elif expected != observed:
        differences.append(
            {"path": prefix, "expected": expected, "observed": observed}
        )
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-config", type=Path, default=MAIN_CONFIG)
    parser.add_argument("--entropy-config", type=Path, default=ENTROPY_CONFIG)
    parser.add_argument("--release-root", type=Path, default=RELEASE_ROOT)
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_VALIDATION,
        help="Entropy validation/calibration report for the intended main run",
    )
    parser.add_argument(
        "--require-production-data",
        action="store_true",
        help="Fail unless the configured 100B materialization and views are complete",
    )
    parser.add_argument(
        "--require-capacity",
        action="store_true",
        help="Fail unless every calibrated rank stream covers all 5,299 updates",
    )
    parser.add_argument(
        "--require-clean-repo",
        action="store_true",
        help="Fail if the BLT worktree has uncommitted changes",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    release_root = args.release_root.resolve()
    validation_path = args.validation.resolve()
    validation = load_json(validation_path)
    threshold = float(validation["calibrated_global_threshold"]["threshold"])

    released_main = TrainArgs.model_validate(
        load_json(release_root / "train_args.json")
    )
    released_entropy = TrainArgs.model_validate(
        load_json(release_root / "entropy_model/params.json")
    )
    candidate_main = parse_config(
        args.main_config.resolve(),
        {
            "data": {"patcher_args": {"threshold": threshold}},
            "model": {"patching_threshold": threshold},
        },
    )
    candidate_entropy = parse_config(args.entropy_config.resolve())
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    checks: list[dict[str, Any]] = []

    def check(
        name: str,
        passed: bool,
        *,
        observed: Any = None,
        expected: Any = None,
        required: bool = True,
    ) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "required": required,
                "observed": observed,
                "expected": expected,
            }
        )

    released_model = released_main.model.model_dump(mode="json")
    candidate_model = candidate_main.model.model_dump(mode="json")
    model_differences = nested_differences(released_model, candidate_model)
    check(
        "main model differences are calibrated threshold only",
        {item["path"] for item in model_differences} == {"patching_threshold"},
        observed=model_differences,
        expected=["patching_threshold"],
    )
    check(
        "main optimizer exactly matches release",
        candidate_main.optim == released_main.optim,
        observed=candidate_main.optim.model_dump(mode="json"),
        expected=released_main.optim.model_dump(mode="json"),
    )
    distributed_differences = nested_differences(
        released_main.distributed.model_dump(mode="json"),
        candidate_main.distributed.model_dump(mode="json"),
    )
    check(
        "main distributed differences are limited to data-parallel degree",
        {item["path"] for item in distributed_differences} <= {"dp_replicate"},
        observed=distributed_differences,
        expected="none or dp_replicate only",
    )

    released_global_patches = (
        released_main.distributed.dp_replicate
        * released_main.data.batch_size
        * released_main.data.seq_len
        * released_main.grad_acc_steps
    )
    candidate_global_patches = (
        candidate_main.distributed.dp_replicate
        * candidate_main.data.batch_size
        * candidate_main.data.seq_len
        * candidate_main.grad_acc_steps
    )
    check(
        "main global patch batch exactly matches release",
        candidate_global_patches == released_global_patches == 4_194_304,
        observed=candidate_global_patches,
        expected=released_global_patches,
    )
    check(
        "main topology preserves a supported exact global batch",
        (
            (
                candidate_main.distributed.dp_replicate,
                candidate_main.data.batch_size,
            )
            in ((128, 8), (256, 4))
            and candidate_main.grad_acc_steps == 1
        ),
        observed={
            "ranks": candidate_main.distributed.dp_replicate,
            "batch_size": candidate_main.data.batch_size,
            "grad_acc_steps": candidate_main.grad_acc_steps,
        },
        expected={
            "allowed_rank_batch_pairs": [[128, 8], [256, 4]],
            "grad_acc_steps": 1,
        },
    )
    main_data_fields = (
        "seed",
        "seq_len",
        "max_encoder_seq_length",
        "buffer_size",
        "arrow_batch_size",
        "load_async",
        "async_persist_type",
        "prefetch_size",
        "add_bos",
        "add_eos",
        "add_patches",
        "pad_to_max_length",
        "enable_byte_ngrams",
    )
    main_data_observed = {
        field: getattr(candidate_main.data, field) for field in main_data_fields
    }
    main_data_expected = {
        field: getattr(released_main.data, field) for field in main_data_fields
    }
    check(
        "main sequence, buffer, and asynchronous loader recipe matches release",
        main_data_observed == main_data_expected,
        observed=main_data_observed,
        expected=main_data_expected,
    )
    main_patcher_differences = nested_differences(
        released_main.data.patcher_args.model_dump(mode="json"),
        candidate_main.data.patcher_args.model_dump(mode="json"),
    )
    check(
        "main patcher differences are calibrated threshold only",
        {item["path"] for item in main_patcher_differences} == {"threshold"},
        observed=main_patcher_differences,
        expected=["threshold"],
    )
    check(
        "main threshold is paired between loader and model",
        (
            candidate_main.data.patcher_args.threshold
            == candidate_main.model.patching_threshold
            == threshold
        ),
        observed={
            "loader": candidate_main.data.patcher_args.threshold,
            "model": candidate_main.model.patching_threshold,
        },
        expected=threshold,
    )
    check(
        "100B step budget is exact",
        candidate_main.steps == 5299,
        observed=candidate_main.steps,
        expected=5299,
    )
    check(
        "main checkpoint cadence matches release",
        (
            candidate_main.checkpoint.dump == released_main.checkpoint.dump
            and candidate_main.checkpoint.eval == released_main.checkpoint.eval
        ),
        observed={
            "dump": candidate_main.checkpoint.dump.model_dump(mode="json"),
            "eval": candidate_main.checkpoint.eval.model_dump(mode="json"),
        },
        expected={
            "dump": released_main.checkpoint.dump.model_dump(mode="json"),
            "eval": released_main.checkpoint.eval.model_dump(mode="json"),
        },
    )

    check(
        "entropy architecture exactly matches released checkpoint",
        candidate_entropy.entropy_model == released_entropy.entropy_model,
        observed=candidate_entropy.entropy_model.model_dump(mode="json"),
        expected=released_entropy.entropy_model.model_dump(mode="json"),
    )
    check(
        "entropy optimizer exactly matches released checkpoint",
        candidate_entropy.optim == released_entropy.optim,
        observed=candidate_entropy.optim.model_dump(mode="json"),
        expected=released_entropy.optim.model_dump(mode="json"),
    )
    check(
        "entropy distributed recipe exactly matches released checkpoint",
        candidate_entropy.distributed == released_entropy.distributed,
        observed=candidate_entropy.distributed.model_dump(mode="json"),
        expected=released_entropy.distributed.model_dump(mode="json"),
    )
    entropy_data_fields = (
        "seed",
        "batch_size",
        "seq_len",
        "max_encoder_seq_length",
        "buffer_size",
        "arrow_batch_size",
        "load_async",
        "prefetch_size",
        "add_bos",
        "add_eos",
        "add_patches",
        "pad_to_max_length",
        "enable_byte_ngrams",
    )
    entropy_data_observed = {
        field: getattr(candidate_entropy.data, field)
        for field in entropy_data_fields
    }
    entropy_data_expected = {
        field: getattr(released_entropy.data, field)
        for field in entropy_data_fields
    }
    check(
        "entropy sequence and loader-shape recipe matches release",
        entropy_data_observed == entropy_data_expected,
        observed=entropy_data_observed,
        expected=entropy_data_expected,
    )
    entropy_global_positions = (
        candidate_entropy.distributed.dp_replicate
        * candidate_entropy.data.batch_size
        * candidate_entropy.data.seq_len
    )
    check(
        "entropy global byte-position batch and step budget match release",
        entropy_global_positions == 524_288 and candidate_entropy.steps == 100_000,
        observed={
            "positions_per_update": entropy_global_positions,
            "steps": candidate_entropy.steps,
        },
        expected={"positions_per_update": 524_288, "steps": 100_000},
    )

    calibrated = validation["calibrated_global_threshold"]
    capacity = validation["rank_local_capacity_at_calibrated_threshold"]
    expected_workers_per_chunk = candidate_main.distributed.dp_replicate // 64
    check(
        "calibrated mean patch size is within one percent of 4.5",
        abs(float(calibrated["realized_patch_size"]) - 4.5) / 4.5 <= 0.01,
        observed=calibrated["realized_patch_size"],
        expected=4.5,
    )
    check(
        "capacity report represents selected rank assignment",
        (
            capacity["workers_per_chunk"] == expected_workers_per_chunk
            and capacity["rank_streams"]
            == candidate_main.distributed.dp_replicate
        ),
        observed={
            "workers_per_chunk": capacity["workers_per_chunk"],
            "rank_streams": capacity["rank_streams"],
        },
        expected={
            "workers_per_chunk": expected_workers_per_chunk,
            "rank_streams": candidate_main.distributed.dp_replicate,
        },
    )
    check(
        "all rank streams cover 5,299 updates without buffer repeat",
        capacity["all_rank_streams_cover_requested_steps_without_buffer_repeat"],
        observed=capacity["minimum_buffer_safe_optimizer_steps_before_repeat"],
        expected=5299,
        required=args.require_capacity,
    )

    data_root = Path(candidate_main.data.root_dir)
    audit_path = data_root / "audit.json"
    world_8 = data_root / "views/world_0008/view.json"
    world_4 = data_root / "views/world_0004/view.json"
    production_paths = [data_root, audit_path, world_8, world_4]
    check(
        "production materialization and topology views exist",
        all(path.exists() for path in production_paths),
        observed={str(path): path.exists() for path in production_paths},
        expected="all true",
        required=args.require_production_data,
    )
    check(
        "repository worktree is clean",
        not git_status,
        observed="clean" if not git_status else git_status.splitlines(),
        expected="clean",
        required=args.require_clean_repo,
    )
    if audit_path.is_file():
        audit = load_json(audit_path)
        training_text_bytes = sum(
            int(item["utf8_text_bytes"])
            for item in audit["output"]["files"]
            if "/dclm_baseline_1.0/" in item["path"]
        )
        check(
            "production record-multiset audit passed",
            audit["comparison"]["passed"],
            observed=audit["comparison"]["passed"],
            expected=True,
        )
        check(
            "post-validation training text is at least 100B bytes",
            training_text_bytes >= 100_000_000_000,
            observed=training_text_bytes,
            expected=100_000_000_000,
        )

    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "repo_head": git_head,
        "repo_dirty": bool(git_status),
        "inputs": {
            "main_config": str(args.main_config.resolve()),
            "main_config_sha256": sha256_file(args.main_config.resolve()),
            "entropy_config": str(args.entropy_config.resolve()),
            "entropy_config_sha256": sha256_file(args.entropy_config.resolve()),
            "release_train_args": str(release_root / "train_args.json"),
            "release_train_args_sha256": sha256_file(
                release_root / "train_args.json"
            ),
            "release_entropy_params": str(
                release_root / "entropy_model/params.json"
            ),
            "release_entropy_params_sha256": sha256_file(
                release_root / "entropy_model/params.json"
            ),
            "validation": str(validation_path),
            "validation_sha256": sha256_file(validation_path),
        },
        "calibrated_threshold": threshold,
        "checks": checks,
        "required_checks_passed": all(
            item["passed"] for item in checks if item["required"]
        ),
    }
    if args.output:
        output = checked_output(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["required_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
