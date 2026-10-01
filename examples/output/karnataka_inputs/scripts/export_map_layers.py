"""Write the Sandur localities and schematic geology as GeoJSON (UTM 43N metres).

The study viewers carry these layers already (build_all_workflow.py); the file is for the
upload page: add it in a workspace's 3D view (＋ Add map layer) to see the mines, towns and
the schist belt over new runs.

    py examples/output/karnataka_inputs/scripts/export_map_layers.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
import literature   # noqa: E402

OUT = HERE.parent / "shared" / "karnataka_map_layers.geojson"


def main():
    gj = literature.map_layers_geojson()
    OUT.write_text(json.dumps(gj, separators=(",", ":")), encoding="utf-8")
    print(f"{len(gj['features'])} features ->", OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
