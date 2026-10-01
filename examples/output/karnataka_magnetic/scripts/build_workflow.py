"""The full-resolution magnetic runs (with the magnetization-vector run) in one DAG viewer
workflow, into karnataka_magnetic.geoinv3d.json and its viewer.

    py examples/output/karnataka_magnetic/scripts/build_workflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Built by karnataka_inputs/build_all_workflow.py, which builds the three studies'
    viewers alike (their run lists, names and the overlays of the reference runs)."""
    sys.path.insert(0, str(ROOT.parent / "karnataka_inputs"))
    import build_all_workflow
    build_all_workflow.build("magnetic", "--with-2km" in sys.argv)


if __name__ == "__main__":
    main()
