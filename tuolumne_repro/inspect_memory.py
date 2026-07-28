#!/usr/bin/env python3
"""Report host/cgroup/accelerator memory visible inside a Tuolumne job."""

from __future__ import annotations

import json
from pathlib import Path

import psutil
import torch


def cgroup_v2_path() -> Path | None:
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        if hierarchy == "0" and controllers == "":
            return Path("/sys/fs/cgroup") / path.lstrip("/")
    return None


def read_cgroup_memory(path: Path | None) -> dict[str, str | None]:
    if path is None:
        return {}
    result: dict[str, str | None] = {"path": str(path)}
    for name in (
        "memory.current",
        "memory.high",
        "memory.max",
        "memory.peak",
        "memory.swap.current",
        "memory.swap.max",
    ):
        candidate = path / name
        result[name] = (
            candidate.read_text().strip() if candidate.is_file() else None
        )
    return result


def main() -> None:
    visible_devices = []
    for device in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(device)
        visible_devices.append(
            {
                "index": device,
                "name": torch.cuda.get_device_name(device),
                "free_bytes": free,
                "total_bytes": total,
            }
        )

    host = psutil.virtual_memory()
    report = {
        "host_memory": {
            "total_bytes": host.total,
            "available_bytes": host.available,
        },
        "cgroup_v2": read_cgroup_memory(cgroup_v2_path()),
        "accelerators": visible_devices,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
