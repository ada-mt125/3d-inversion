"""The gravity comparison with terrain in one DAG viewer workflow: the 8 full-resolution EC2
runs (--with-2km adds the 16 runs of the 2 km study), into karnataka_gravity_terrain.geoinv3d.json
and its viewer.

    py examples/output/karnataka_gravity_terrain/scripts/build_workflow.py [--with-2km]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Built by karnataka_inputs/scripts/build_all_workflow.py, which builds the three studies'
    viewers alike (their run lists, names and the overlays of the reference runs)."""
    sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "scripts"))
    import build_all_workflow
    build_all_workflow.build("gravity", "--with-2km" in sys.argv)


if __name__ == "__main__":
    main()
