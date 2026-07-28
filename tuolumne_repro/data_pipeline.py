#!/usr/bin/env python3
"""Pinned, byte-preserving DCLM preparation for the BLT reproduction.

The commands in this file deliberately separate inventory, download,
materialization, and audit.  A failed or interrupted network transfer can be
resumed without touching a completed shuffled materialization, and an audit
can be repeated without changing the training data.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


REPO_ID = "mlfoundations/dclm-baseline-1.0"
REVISION = "a3b142c183aebe5af344955ae20836eb34dcf69b"
TERASHUF_REPO = "https://github.com/alexandres/terashuf.git"
TERASHUF_REVISION = "29a65ed74808925266a5dcf4ffccd29552dad0e0"
TERASHUF_SOURCE_SHA256 = (
    "05b08bfed31765b6f4f3430efbc06442459292a2d8f6f68612d4053026ca678c"
)
DATASET_NAME = "dclm_baseline_1.0"
FILE_ORDER_DOMAIN = "blt-dclm-file-v1"
DEFAULT_SEED = 42
DEFAULT_DATA_ROOT = Path("/p/vast1/pretrain/datasets/blt")
REPO_WORKSPACE = Path("/p/vast1/kirchenb/hlm-root")
ALLOWED_DATA_ROOT = Path("/p/vast1/pretrain/datasets/blt")
MATERIALIZATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
UINT256_MODULUS = 1 << 256


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_data_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    allowed = (ALLOWED_DATA_ROOT.resolve(), REPO_WORKSPACE.resolve())
    if not any(is_relative_to(resolved, root) for root in allowed):
        raise ValueError(
            f"Refusing data root outside the authorized trees: {resolved}"
        )
    return resolved


def validate_materialization(name: str) -> str:
    if not MATERIALIZATION_RE.fullmatch(name):
        raise ValueError(
            "Materialization names may contain only letters, digits, '.', '_', and '-'"
        )
    return name


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, path)


def inventory_paths(data_root: Path) -> tuple[Path, Path]:
    base = data_root / "manifests" / DATASET_NAME / REVISION
    return base / "inventory.jsonl", base / "summary.json"


def source_root(data_root: Path) -> Path:
    return data_root / "sources" / "dclm-baseline-1.0" / REVISION


def terashuf_root(data_root: Path) -> Path:
    return data_root / "tools" / "terashuf" / TERASHUF_REVISION


def selection_key(source_path: str, seed: int) -> str:
    material = f"{FILE_ORDER_DOMAIN}\0{seed}\0{source_path}".encode()
    return hashlib.sha256(material).hexdigest()


def command_inventory(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    data_root = validate_data_root(args.data_root)
    inventory_path, summary_path = inventory_paths(data_root)
    info = HfApi().dataset_info(
        REPO_ID, revision=REVISION, files_metadata=True
    )
    if info.sha != REVISION:
        raise RuntimeError(f"Resolved revision {info.sha}, expected {REVISION}")

    records: list[dict[str, Any]] = []
    for sibling in info.siblings:
        path = sibling.rfilename
        if not path.endswith(".jsonl.zst"):
            continue
        lfs = getattr(sibling, "lfs", None)
        lfs_sha256 = getattr(lfs, "sha256", None)
        size = sibling.size
        if size is None or lfs_sha256 is None:
            raise RuntimeError(f"Missing size or LFS digest for {path}")
        records.append(
            {
                "path": path,
                "compressed_bytes": int(size),
                "lfs_sha256": lfs_sha256,
                "selection_key": selection_key(path, args.seed),
            }
        )

    records.sort(key=lambda item: (item["selection_key"], item["path"]))
    for rank, record in enumerate(records):
        record["selection_rank"] = rank

    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "repo_id": REPO_ID,
        "revision": REVISION,
        "selection_seed": args.seed,
        "selection_algorithm": (
            "ascending sha256('blt-dclm-file-v1\\0' + seed + "
            "'\\0' + source_path)"
        ),
        "file_count": len(records),
        "compressed_bytes": sum(row["compressed_bytes"] for row in records),
    }
    atomic_write_jsonl(inventory_path, records)
    atomic_write_json(summary_path, summary)
    print(inventory_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


def read_inventory(data_root: Path) -> list[dict[str, Any]]:
    inventory_path, _ = inventory_paths(data_root)
    if not inventory_path.is_file():
        raise FileNotFoundError(
            f"Missing {inventory_path}; run the inventory command first"
        )
    records: list[dict[str, Any]] = []
    with inventory_path.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    expected_ranks = list(range(len(records)))
    actual_ranks = [record["selection_rank"] for record in records]
    if actual_ranks != expected_ranks:
        raise RuntimeError("Inventory is not in selection-rank order")
    return records


def selected_inventory(
    data_root: Path, source_files: int
) -> list[dict[str, Any]]:
    inventory = read_inventory(data_root)
    if source_files < 1 or source_files > len(inventory):
        raise ValueError(
            f"--source-files must be in [1, {len(inventory)}], got {source_files}"
        )
    return inventory[:source_files]


def local_source_path(data_root: Path, record: dict[str, Any]) -> Path:
    return source_root(data_root) / record["path"]


def command_download(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    data_root = validate_data_root(args.data_root)
    selected = selected_inventory(data_root, args.source_files)
    destination = source_root(data_root)
    destination.mkdir(parents=True, exist_ok=True)

    for number, record in enumerate(selected, start=1):
        expected = destination / record["path"]
        print(
            f"[{number}/{len(selected)}] rank={record['selection_rank']} "
            f"{record['path']}",
            flush=True,
        )
        downloaded = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=record["path"],
                repo_type="dataset",
                revision=REVISION,
                local_dir=destination,
            )
        )
        if downloaded.resolve() != expected.resolve():
            raise RuntimeError(f"Unexpected download path: {downloaded}")
        if downloaded.stat().st_size != record["compressed_bytes"]:
            raise RuntimeError(f"Size mismatch for {downloaded}")
        if args.verify_sha256:
            actual = sha256_file(downloaded)
            if actual != record["lfs_sha256"]:
                raise RuntimeError(
                    f"SHA-256 mismatch for {downloaded}: {actual}"
                )

    selection_manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "repo_id": REPO_ID,
        "revision": REVISION,
        "source_files": len(selected),
        "compressed_bytes": sum(row["compressed_bytes"] for row in selected),
        "files": selected,
    }
    manifest_path = (
        data_root
        / "manifests"
        / DATASET_NAME
        / REVISION
        / f"selected_first_{len(selected):05d}.json"
    )
    atomic_write_json(manifest_path, selection_manifest)
    print(manifest_path)


def run_checked(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def command_build_terashuf(args: argparse.Namespace) -> None:
    data_root = validate_data_root(args.data_root)
    tool_root = terashuf_root(data_root)
    source_dir = tool_root / "src"
    executable = source_dir / "terashuf"
    tool_root.mkdir(parents=True, exist_ok=True)

    if not (source_dir / ".git").is_dir():
        if source_dir.exists():
            raise RuntimeError(
                f"{source_dir} exists but is not a terashuf Git checkout"
            )
        run_checked(["git", "clone", TERASHUF_REPO, str(source_dir)])

    actual_revision = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != TERASHUF_REVISION:
        status = subprocess.check_output(
            ["git", "-C", str(source_dir), "status", "--porcelain"], text=True
        )
        if status:
            raise RuntimeError(
                f"Refusing to change dirty terashuf checkout at {source_dir}"
            )
        run_checked(
            [
                "git",
                "-C",
                str(source_dir),
                "fetch",
                "origin",
                TERASHUF_REVISION,
            ]
        )
        run_checked(
            [
                "git",
                "-C",
                str(source_dir),
                "checkout",
                "--detach",
                TERASHUF_REVISION,
            ]
        )

    source_digest = sha256_file(source_dir / "terashuf.cc")
    if source_digest != TERASHUF_SOURCE_SHA256:
        raise RuntimeError(
            f"terashuf.cc digest {source_digest}, expected {TERASHUF_SOURCE_SHA256}"
        )
    run_checked(["make", "-C", str(source_dir), "-j", str(args.jobs)])
    if not executable.is_file():
        raise RuntimeError(f"Build did not create {executable}")

    compiler = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    metadata = {
        "schema_version": 1,
        "created_at": utc_now(),
        "repository": TERASHUF_REPO,
        "revision": TERASHUF_REVISION,
        "source_sha256": source_digest,
        "executable_sha256": sha256_file(executable),
        "compiler": compiler,
        "makefile_sha256": sha256_file(source_dir / "Makefile"),
    }
    atomic_write_json(tool_root / "build.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def get_terashuf_metadata(data_root: Path) -> tuple[Path, dict[str, Any]]:
    tool_root = terashuf_root(data_root)
    executable = tool_root / "src" / "terashuf"
    metadata_path = tool_root / "build.json"
    if not executable.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Pinned terashuf build is missing; run build-terashuf first"
        )
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["revision"] != TERASHUF_REVISION:
        raise RuntimeError("Unexpected terashuf revision in build metadata")
    actual_digest = sha256_file(executable)
    if actual_digest != metadata["executable_sha256"]:
        raise RuntimeError("terashuf executable changed after it was built")
    return executable, metadata


def source_files_for_selection(
    data_root: Path, selected: list[dict[str, Any]]
) -> list[Path]:
    paths = [local_source_path(data_root, record) for record in selected]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing selected source archives:\n{preview}")
    return paths


def split_suffix_length(n_chunks: int) -> int:
    return max(2, len(str(n_chunks - 1)))


def run_shuffle_pipeline(
    *,
    sources: list[Path],
    terashuf: Path,
    seed: int,
    memory_gib: float,
    n_chunks: int,
    output_prefix: Path,
    tmp_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    suffix_length = split_suffix_length(n_chunks)
    env = os.environ.copy()
    env.update(
        {
            "SEED": str(seed),
            "MEMORY": str(memory_gib),
            "TMPDIR": str(tmp_dir),
            "LC_ALL": "C",
        }
    )
    split_command = [
        "split",
        "-n",
        f"r/{n_chunks}",
        "-d",
        f"--suffix-length={suffix_length}",
        "--additional-suffix=.jsonl",
        "-",
        str(output_prefix),
    ]
    tmp_dir.mkdir(parents=True, exist_ok=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("wb") as log:
        split_process = subprocess.Popen(
            split_command,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            env=env,
        )
        assert split_process.stdin is not None
        shuffle_process = subprocess.Popen(
            [str(terashuf)],
            stdin=subprocess.PIPE,
            stdout=split_process.stdin,
            stderr=log,
            env=env,
        )
        split_process.stdin.close()
        assert shuffle_process.stdin is not None

        source_error: tuple[Path, int] | None = None
        boundary_newlines_added = 0
        try:
            # DCLM archives do not end in newlines. Passing several paths to
            # zstdcat would concatenate the last JSON object of one archive
            # with the first object of the next. Stream each archive
            # separately so record bytes remain unchanged and insert a
            # delimiter only at a missing inter-file boundary.
            for source_index, source in enumerate(sources):
                decompressor = subprocess.Popen(
                    ["zstdcat", str(source)],
                    stdout=subprocess.PIPE,
                    stderr=log,
                    env=env,
                )
                assert decompressor.stdout is not None
                final_byte: int | None = None
                while block := decompressor.stdout.read(8 * 1024 * 1024):
                    shuffle_process.stdin.write(block)
                    final_byte = block[-1]
                decompressor.stdout.close()
                returncode = decompressor.wait()
                if returncode != 0:
                    source_error = (source, returncode)
                    break
                if final_byte is None:
                    raise RuntimeError(f"Decompressed source is empty: {source}")
                if source_index < len(sources) - 1 and final_byte != ord("\n"):
                    shuffle_process.stdin.write(b"\n")
                    boundary_newlines_added += 1
        finally:
            shuffle_process.stdin.close()

        shuffle_returncode = shuffle_process.wait()
        split_returncode = split_process.wait()

    if source_error is not None:
        source, returncode = source_error
        raise RuntimeError(
            f"zstdcat failed with {returncode}: {source}"
        )
    if shuffle_returncode != 0:
        raise RuntimeError(
            f"terashuf failed with {shuffle_returncode}; see {log_path}"
        )
    if split_returncode != 0:
        raise RuntimeError(
            f"split failed with {split_returncode}; see {log_path}"
        )
    return {
        "method": "manifest-order concatenation with LF at missing file boundaries",
        "source_files": len(sources),
        "boundary_newlines_added": boundary_newlines_added,
        "record_bytes_changed": False,
    }


def extract_validation(
    chunk_paths: list[Path], validation_path: Path, records_per_chunk: int
) -> dict[str, Any]:
    temporary_validation = validation_path.with_name(
        f".{validation_path.name}.tmp.{os.getpid()}"
    )
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    counts: list[int] = []

    with temporary_validation.open("wb") as validation:
        for chunk_path in chunk_paths:
            temporary_train = chunk_path.with_name(
                f".{chunk_path.name}.train.tmp.{os.getpid()}"
            )
            selected = 0
            with chunk_path.open("rb") as source, temporary_train.open("wb") as train:
                for _ in range(records_per_chunk):
                    line = source.readline()
                    if not line:
                        break
                    validation.write(line)
                    selected += 1
                shutil.copyfileobj(source, train, length=8 * 1024 * 1024)
            os.replace(temporary_train, chunk_path)
            counts.append(selected)
    os.replace(temporary_validation, validation_path)
    return {
        "records_per_chunk_requested": records_per_chunk,
        "records_per_chunk_observed": counts,
        "validation_records": sum(counts),
    }


def command_prepare(args: argparse.Namespace) -> None:
    data_root = validate_data_root(args.data_root)
    materialization = validate_materialization(args.materialization)
    if args.n_chunks < 1:
        raise ValueError("--n-chunks must be positive")
    if args.validation_per_chunk < 0:
        raise ValueError("--validation-per-chunk must be nonnegative")
    if args.memory_gib <= 0:
        raise ValueError("--memory-gib must be positive")

    selected = selected_inventory(data_root, args.source_files)
    sources = source_files_for_selection(data_root, selected)
    terashuf, terashuf_metadata = get_terashuf_metadata(data_root)

    prepared_root = data_root / "prepared"
    final_root = prepared_root / materialization
    if final_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed materialization {final_root}"
        )
    staging_root = prepared_root / f".{materialization}.incomplete.{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"Staging path already exists: {staging_root}")

    dataset_dir = staging_root / DATASET_NAME
    validation_dir = staging_root / "validation"
    logs_dir = staging_root / "logs"
    dataset_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    tmp_dir = (
        data_root
        / "tmp"
        / "terashuf"
        / f"{materialization}.{os.getpid()}"
    )
    output_prefix = dataset_dir / f"{DATASET_NAME}.chunk."
    log_path = logs_dir / "shuffle.log"
    started_at = utc_now()
    start_time = time.monotonic()

    input_join = run_shuffle_pipeline(
        sources=sources,
        terashuf=terashuf,
        seed=args.seed,
        memory_gib=args.memory_gib,
        n_chunks=args.n_chunks,
        output_prefix=output_prefix,
        tmp_dir=tmp_dir,
        log_path=log_path,
    )

    suffix_length = split_suffix_length(args.n_chunks)
    chunk_paths = [
        dataset_dir
        / f"{DATASET_NAME}.chunk.{index:0{suffix_length}d}.jsonl"
        for index in range(args.n_chunks)
    ]
    missing_chunks = [path for path in chunk_paths if not path.is_file()]
    if missing_chunks:
        raise RuntimeError(f"split did not create all chunks: {missing_chunks}")

    validation_path = validation_dir / f"{DATASET_NAME}.val.jsonl"
    validation = extract_validation(
        chunk_paths, validation_path, args.validation_per_chunk
    )
    elapsed_seconds = time.monotonic() - start_time
    metadata = {
        "schema_version": 2,
        "materialization": materialization,
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "dataset": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "name": DATASET_NAME,
            "source_files": len(selected),
            "source_compressed_bytes": sum(
                row["compressed_bytes"] for row in selected
            ),
            "ordered_files": selected,
            "input_join": input_join,
            "record_transform": None,
            "text_transform": None,
        },
        "shuffle": {
            "implementation": "terashuf",
            "seed": args.seed,
            "memory_gib": args.memory_gib,
            "tmp_dir": str(tmp_dir),
            **terashuf_metadata,
        },
        "split": {
            "implementation": "GNU split",
            "mode": f"r/{args.n_chunks}",
            "n_chunks": args.n_chunks,
            "validation": validation,
        },
        "paths": {
            "dataset_dir": str(final_root / DATASET_NAME),
            "validation": str(
                final_root / "validation" / f"{DATASET_NAME}.val.jsonl"
            ),
            "shuffle_log": str(final_root / "logs" / "shuffle.log"),
        },
    }
    atomic_write_json(staging_root / "materialization.json", metadata)
    prepared_root.mkdir(parents=True, exist_ok=True)
    os.replace(staging_root, final_root)
    print(final_root)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def empty_digest() -> dict[str, int]:
    return {
        "records": 0,
        "record_bytes_without_newline": 0,
        "stream_bytes": 0,
        "utf8_text_bytes": 0,
        "invalid_json_records": 0,
        "digest_sum": 0,
        "digest_xor": 0,
    }


def update_digest(accumulator: dict[str, int], line: bytes) -> None:
    accumulator["records"] += 1
    accumulator["stream_bytes"] += len(line)
    record = line[:-1] if line.endswith(b"\n") else line
    accumulator["record_bytes_without_newline"] += len(record)
    digest_int = int.from_bytes(hashlib.sha256(record).digest(), "big")
    accumulator["digest_sum"] = (
        accumulator["digest_sum"] + digest_int
    ) % UINT256_MODULUS
    accumulator["digest_xor"] ^= digest_int
    try:
        import orjson

        value = orjson.loads(record)
        text = value["text"]
        if not isinstance(text, str):
            raise TypeError("text is not a string")
        accumulator["utf8_text_bytes"] += len(text.encode("utf-8"))
    except (KeyError, TypeError, ValueError):
        accumulator["invalid_json_records"] += 1


def digest_stream(stream: BinaryIO) -> dict[str, Any]:
    result = empty_digest()
    final_line_ends_with_lf: bool | None = None
    for line in stream:
        update_digest(result, line)
        final_line_ends_with_lf = line.endswith(b"\n")
    result["final_line_ends_with_lf"] = final_line_ends_with_lf
    return result


def digest_source(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    process = subprocess.Popen(
        ["zstdcat", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    result = digest_stream(process.stdout)
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"zstdcat failed for {path}: {stderr}")
    result["path"] = str(path)
    result["compressed_bytes"] = path.stat().st_size
    result["compressed_sha256"] = sha256_file(path)
    return result


def digest_plain(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    with path.open("rb") as handle:
        result = digest_stream(handle)
    result["path"] = str(path)
    result["file_sha256"] = sha256_file(path)
    return result


def combine_digests(parts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    combined = empty_digest()
    files: list[dict[str, Any]] = []
    for part in parts:
        for key in (
            "records",
            "record_bytes_without_newline",
            "stream_bytes",
            "utf8_text_bytes",
            "invalid_json_records",
        ):
            combined[key] += int(part[key])
        combined["digest_sum"] = (
            combined["digest_sum"] + int(part["digest_sum"])
        ) % UINT256_MODULUS
        combined["digest_xor"] ^= int(part["digest_xor"])
        files.append(part)
    combined["digest_sum"] = f"{combined['digest_sum']:064x}"
    combined["digest_xor"] = f"{combined['digest_xor']:064x}"
    combined["files"] = files
    return combined


def parallel_map(
    function: Any, paths: list[Path], workers: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_path = {
            pool.submit(function, str(path)): path for path in paths
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_to_path), start=1
        ):
            path = future_to_path[future]
            result = future.result()
            print(f"[{completed}/{len(paths)}] audited {path}", flush=True)
            results.append(result)
    results.sort(key=lambda result: result["path"])
    return results


def comparable_digest(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "records",
        "record_bytes_without_newline",
        "utf8_text_bytes",
        "invalid_json_records",
        "digest_sum",
        "digest_xor",
    )
    return {key: value[key] for key in keys}


def command_audit(args: argparse.Namespace) -> None:
    data_root = validate_data_root(args.data_root)
    materialization = validate_materialization(args.materialization)
    materialization_root = data_root / "prepared" / materialization
    metadata_path = materialization_root / "materialization.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    selected = metadata["dataset"]["ordered_files"]
    sources = source_files_for_selection(data_root, selected)
    dataset_dir = materialization_root / DATASET_NAME
    outputs = sorted(dataset_dir.glob(f"{DATASET_NAME}.chunk.*.jsonl"))
    outputs.append(
        materialization_root / "validation" / f"{DATASET_NAME}.val.jsonl"
    )
    missing_outputs = [path for path in outputs if not path.is_file()]
    if missing_outputs:
        raise FileNotFoundError(f"Missing materialized outputs: {missing_outputs}")

    started = time.monotonic()
    input_parts = parallel_map(digest_source, sources, args.workers)
    output_parts = parallel_map(digest_plain, outputs, args.workers)
    input_digest = combine_digests(input_parts)
    output_digest = combine_digests(output_parts)
    input_comparable = comparable_digest(input_digest)
    output_comparable = comparable_digest(output_digest)
    passed = input_comparable == output_comparable
    audit = {
        "schema_version": 2,
        "created_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "workers": args.workers,
        "definition": (
            "SHA-256 each complete JSON record without its trailing LF; combine "
            "record hashes with uint256 modular sum and XOR"
        ),
        "input": input_digest,
        "output": output_digest,
        "comparison": {
            "passed": passed,
            "input": input_comparable,
            "output": output_comparable,
        },
    }
    atomic_write_json(materialization_root / "audit.json", audit)
    print(json.dumps(audit["comparison"], indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("Materialization multiset audit failed")


def command_derive_view(args: argparse.Namespace) -> None:
    data_root = validate_data_root(args.data_root)
    materialization = validate_materialization(args.materialization)
    if args.target_chunks < 1:
        raise ValueError("--target-chunks must be positive")
    materialization_root = data_root / "prepared" / materialization
    metadata_path = materialization_root / "materialization.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    audit_path = materialization_root / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Refusing to derive a view before the canonical audit: {audit_path}"
        )
    with audit_path.open(encoding="utf-8") as handle:
        canonical_audit = json.load(handle)
    if not canonical_audit["comparison"]["passed"]:
        raise RuntimeError(
            f"Refusing to derive a view from failed materialization {materialization}"
        )

    source_chunks = int(metadata["split"]["n_chunks"])
    if source_chunks % args.target_chunks != 0:
        raise ValueError(
            f"Source chunk count {source_chunks} is not divisible by "
            f"target count {args.target_chunks}"
        )
    source_dir = materialization_root / DATASET_NAME
    source_paths = sorted(
        source_dir.glob(f"{DATASET_NAME}.chunk.*.jsonl")
    )
    if len(source_paths) != source_chunks:
        raise RuntimeError(
            f"Expected {source_chunks} source chunks, found {len(source_paths)}"
        )

    view_name = f"world_{args.target_chunks:04d}"
    view_root = materialization_root / "views" / view_name
    if view_root.exists():
        raise FileExistsError(f"Refusing to overwrite derived view {view_root}")
    staging_root = (
        materialization_root
        / "views"
        / f".{view_name}.incomplete.{os.getpid()}"
    )
    target_dir = staging_root / DATASET_NAME
    target_dir.mkdir(parents=True)
    suffix_length = split_suffix_length(args.target_chunks)
    target_paths = [
        target_dir
        / f"{DATASET_NAME}.chunk.{index:0{suffix_length}d}.jsonl"
        for index in range(args.target_chunks)
    ]

    start = time.monotonic()
    source_handles = [path.open("rb") for path in source_paths]
    target_handles = [path.open("wb") for path in target_paths]
    records_written = [0] * args.target_chunks
    bytes_written = [0] * args.target_chunks
    rounds = 0
    try:
        while True:
            round_had_record = False
            for source_index, source_handle in enumerate(source_handles):
                line = source_handle.readline()
                if not line:
                    continue
                round_had_record = True
                target_index = source_index % args.target_chunks
                target_handles[target_index].write(line)
                records_written[target_index] += 1
                bytes_written[target_index] += len(line)
            if not round_had_record:
                break
            rounds += 1
    finally:
        for handle in source_handles:
            handle.close()
        for handle in target_handles:
            handle.close()

    # Read both sides back from storage.  This proves that the derived view is
    # a lossless repartition of the canonical training chunks, rather than
    # relying only on the copy loop's counters.
    source_parts = parallel_map(digest_plain, source_paths, args.audit_workers)
    target_parts = parallel_map(digest_plain, target_paths, args.audit_workers)
    source_digest = combine_digests(source_parts)
    target_digest = combine_digests(target_parts)
    passed = comparable_digest(source_digest) == comparable_digest(target_digest)
    if not passed:
        raise RuntimeError("Derived view multiset audit failed")

    view_metadata = {
        "schema_version": 1,
        "created_at": utc_now(),
        "materialization": materialization,
        "view": view_name,
        "source_chunks": source_chunks,
        "target_chunks": args.target_chunks,
        "algorithm": (
            "round-robin merge canonical chunks to reconstruct the post-validation "
            "global shuffled stream, then GNU-split-equivalent round-robin repartition"
        ),
        "scientific_effect": (
            "none: record bytes, global order, and canonical validation exclusion "
            "are unchanged"
        ),
        "rounds": rounds,
        "records_per_target": records_written,
        "bytes_per_target": bytes_written,
        "elapsed_seconds": time.monotonic() - start,
        "audit": {
            "passed": passed,
            "source": comparable_digest(source_digest),
            "target": comparable_digest(target_digest),
            "target_file_sha256": {
                Path(part["path"]).name: part["file_sha256"]
                for part in target_parts
            },
        },
        "dataset_dir": str(view_root / DATASET_NAME),
    }
    atomic_write_json(staging_root / "view.json", view_metadata)
    os.replace(staging_root, view_root)
    print(view_root)
    print(json.dumps(view_metadata, indent=2, sort_keys=True))


def command_scan_sources(args: argparse.Namespace) -> None:
    data_root = validate_data_root(args.data_root)
    selected = selected_inventory(data_root, args.source_files)
    sources = source_files_for_selection(data_root, selected)
    started = time.monotonic()
    unordered_parts = parallel_map(digest_source, sources, args.workers)
    parts_by_path = {part["path"]: part for part in unordered_parts}

    cumulative_text_bytes = 0
    files: list[dict[str, Any]] = []
    recommended_source_files: int | None = None
    for index, (selection, path) in enumerate(zip(selected, sources, strict=True), 1):
        part = parts_by_path[str(path)]
        if part["compressed_sha256"] != selection["lfs_sha256"]:
            raise RuntimeError(f"Downloaded source digest mismatch: {path}")
        cumulative_text_bytes += int(part["utf8_text_bytes"])
        files.append(
            {
                **selection,
                "records": part["records"],
                "record_bytes_without_newline": part[
                    "record_bytes_without_newline"
                ],
                "stream_bytes": part["stream_bytes"],
                "utf8_text_bytes": part["utf8_text_bytes"],
                "invalid_json_records": part["invalid_json_records"],
                "final_line_ends_with_lf": part["final_line_ends_with_lf"],
                "record_digest_sum": f"{part['digest_sum']:064x}",
                "record_digest_xor": f"{part['digest_xor']:064x}",
                "cumulative_utf8_text_bytes": cumulative_text_bytes,
            }
        )
        if (
            args.target_text_bytes is not None
            and recommended_source_files is None
            and cumulative_text_bytes >= args.target_text_bytes
        ):
            recommended_source_files = index

    scan = {
        "schema_version": 2,
        "created_at": utc_now(),
        "repo_id": REPO_ID,
        "revision": REVISION,
        "source_files_scanned": len(selected),
        "workers": args.workers,
        "elapsed_seconds": time.monotonic() - started,
        "target_utf8_text_bytes": args.target_text_bytes,
        "recommended_source_files": recommended_source_files,
        "scanned_utf8_text_bytes": cumulative_text_bytes,
        "files": files,
    }
    scan_path = (
        data_root
        / "manifests"
        / DATASET_NAME
        / REVISION
        / f"source_scan_first_{len(selected):05d}.json"
    )
    atomic_write_json(scan_path, scan)
    print(scan_path)
    print(
        json.dumps(
            {
                key: scan[key]
                for key in (
                    "source_files_scanned",
                    "elapsed_seconds",
                    "target_utf8_text_bytes",
                    "recommended_source_files",
                    "scanned_utf8_text_bytes",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


def command_summarize_selection(args: argparse.Namespace) -> None:
    data_root = validate_data_root(args.data_root)
    selected = selected_inventory(data_root, args.source_files)
    summary = {
        "source_files": len(selected),
        "compressed_bytes": sum(row["compressed_bytes"] for row in selected),
        "first_selection_rank": selected[0]["selection_rank"],
        "last_selection_rank": selected[-1]["selection_rank"],
        "first_path": selected[0]["path"],
        "last_path": selected[-1]["path"],
    }
    if args.pilot_audit is not None:
        with args.pilot_audit.open(encoding="utf-8") as handle:
            pilot = json.load(handle)
        compressed = sum(
            item["compressed_bytes"] for item in pilot["input"]["files"]
        )
        text_bytes = pilot["input"]["utf8_text_bytes"]
        ratio = text_bytes / compressed
        summary["pilot_utf8_text_bytes_per_compressed_byte"] = ratio
        summary["estimated_utf8_text_bytes"] = int(
            summary["compressed_bytes"] * ratio
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Artifact root (default: {DEFAULT_DATA_ROOT})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--seed", type=int, default=DEFAULT_SEED)
    inventory.set_defaults(function=command_inventory)

    download = subparsers.add_parser("download")
    download.add_argument("--source-files", type=int, required=True)
    download.add_argument(
        "--verify-sha256",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    download.set_defaults(function=command_download)

    build = subparsers.add_parser("build-terashuf")
    build.add_argument("--jobs", type=int, default=8)
    build.set_defaults(function=command_build_terashuf)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--materialization", required=True)
    prepare.add_argument("--source-files", type=int, required=True)
    prepare.add_argument("--n-chunks", type=int, default=64)
    prepare.add_argument("--validation-per-chunk", type=int, default=10_000)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--memory-gib", type=float, default=192)
    prepare.set_defaults(function=command_prepare)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--materialization", required=True)
    audit.add_argument("--workers", type=int, default=32)
    audit.set_defaults(function=command_audit)

    derive = subparsers.add_parser("derive-view")
    derive.add_argument("--materialization", required=True)
    derive.add_argument("--target-chunks", type=int, required=True)
    derive.add_argument("--audit-workers", type=int, default=32)
    derive.set_defaults(function=command_derive_view)

    scan = subparsers.add_parser("scan-sources")
    scan.add_argument("--source-files", type=int, required=True)
    scan.add_argument("--workers", type=int, default=32)
    scan.add_argument("--target-text-bytes", type=int)
    scan.set_defaults(function=command_scan_sources)

    summarize = subparsers.add_parser("summarize-selection")
    summarize.add_argument("--source-files", type=int, required=True)
    summarize.add_argument("--pilot-audit", type=Path)
    summarize.set_defaults(function=command_summarize_selection)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
