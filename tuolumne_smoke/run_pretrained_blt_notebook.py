#!/usr/bin/env python3
"""Execute the pretrained BLT notebook's code cells without Jupyter.

This is a compute-node validation driver, not an alternate implementation:
each code cell is compiled and executed in order in one shared namespace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("pretrained_blt_exploration.ipynb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blt-repo",
        help="Override the notebook's default Hugging Face BLT repository.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.blt_repo is not None:
        os.environ["BLT_REPO"] = args.blt_repo

    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {
        "__name__": "__blt_notebook__",
        "__file__": str(NOTEBOOK),
    }
    code_cell_number = 0

    for notebook_cell_number, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        code_cell_number += 1
        source = "".join(cell["source"])
        print(
            "NOTEBOOK_CELL_START "
            f"code={code_cell_number} notebook={notebook_cell_number}",
            flush=True,
        )
        exec(
            compile(
                source,
                f"{NOTEBOOK}:code-cell-{code_cell_number}",
                "exec",
            ),
            namespace,
        )
        print(
            "NOTEBOOK_CELL_PASS "
            f"code={code_cell_number} notebook={notebook_cell_number}",
            flush=True,
        )

    print(
        f"NOTEBOOK_EXECUTION_PASS code_cells={code_cell_number}",
        flush=True,
    )


if __name__ == "__main__":
    main()
