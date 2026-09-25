"""Generate and optionally serve an interactive DAG viewer.

Usage:
    python -m geoinv3d.viz.serve_dag workflow.geoinv3d.json
    python -m geoinv3d.viz.serve_dag workflow.geoinv3d.json --output viewer.html
    python -m geoinv3d.viz.serve_dag workflow.geoinv3d.json --serve
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def generate_viewer(workflow_path: str, output_path: str | None = None) -> str:
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow_data = json.load(f)

    # Compact the JSON to reduce large arrays
    compact_data = _compact_workflow(workflow_data)

    template_path = Path(__file__).parent / "dag_interactive.html"
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Inject the workflow data as an embedded script element
    inject = (
        f'\n<script id="embedded-data" type="application/json">'
        f'{json.dumps(compact_data, separators=(",", ":"))}'
        f'</script>\n'
    )
    html = html.replace('</body>', inject + '</body>')

    if output_path is None:
        base = Path(workflow_path).stem
        output_path = str(Path(workflow_path).parent / f"{base}_viewer.html")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


def _compact_workflow(data: dict) -> dict:
    nodes = []
    for node in data.get("nodes", []):
        compact_node = {
            "id": node["id"],
            "type": node["type"],
            "name": node["name"],
            "params": _compact_params(node.get("params", {})),
            "inputs": node.get("inputs", []),
        }
        if "output" in node:
            compact_node["output"] = node["output"]
        nodes.append(compact_node)
    result = {"version": data.get("version", 1), "nodes": nodes}
    if "created" in data:
        result["created"] = data["created"]
    return result


def _compact_params(params: dict) -> dict:
    result = {}
    for key, val in params.items():
        if isinstance(val, list) and len(val) > 50:
            if all(isinstance(v, (int, float)) for v in val):
                result[key] = _array_summary(val)
            elif all(isinstance(v, list) for v in val) and len(val) > 20:
                result[key] = f"[{len(val)} x {len(val[0])}D array]"
            else:
                result[key] = val
        else:
            result[key] = val
    return result


def _array_summary(arr: list) -> dict:
    import statistics
    nonzero = sum(1 for v in arr if v != 0)
    return {
        "__array_summary__": True,
        "length": len(arr),
        "min": min(arr),
        "max": max(arr),
        "mean": round(statistics.mean(arr), 6),
        "nonzero": nonzero,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GeoInv3D DAG Viewer")
    parser.add_argument("workflow", help="Path to .geoinv3d.json workflow file")
    parser.add_argument("--output", "-o", help="Output HTML file path")
    parser.add_argument("--serve", action="store_true", help="Open in browser")
    args = parser.parse_args()

    out = generate_viewer(args.workflow, args.output)
    print(f"Generated viewer: {out}")

    if args.serve:
        import webbrowser
        webbrowser.open(f"file:///{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
