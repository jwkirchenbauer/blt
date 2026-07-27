#!/usr/bin/env python3
"""Execute the pretrained BLT notebook's code cells without Jupyter.

This is a compute-node validation driver, not an alternate implementation:
each code cell is compiled and executed in order in one shared namespace.
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("pretrained_blt_exploration.ipynb")


def main() -> None:
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
