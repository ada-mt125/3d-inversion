"""Main entry point for GeoInv3D."""

from __future__ import annotations

import argparse
import json
import sys

from .core.serialize import load_workflow, save_workflow, graph_to_dict
from .core.node import Graph


def main():
    parser = argparse.ArgumentParser(
        description="GeoInv3D — DAG-based 3D geophysical joint inversion"
    )
    sub = parser.add_subparsers(dest="command")

    # Show workflow
    show = sub.add_parser("show", help="Print a saved workflow as JSON")
    show.add_argument("workflow", help="Path to .geoinv3d.json workflow file")

    # Visualize DAG
    viz = sub.add_parser("dag", help="Visualize a saved workflow DAG")
    viz.add_argument("workflow", help="Path to .geoinv3d.json workflow file")
    viz.add_argument("-o", "--output", help="Save figure to file")

    # API server
    serve = sub.add_parser("serve", help="Start the inversion API server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--bucket", default=None, help="S3 bucket name")
    serve.add_argument("--region", default=None, help="AWS region")
    serve.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.command == "show":
        graph, id_map = load_workflow(args.workflow)
        print(json.dumps(graph_to_dict(graph), indent=2))

    elif args.command == "dag":
        graph, id_map = load_workflow(args.workflow)
        from .viz.dag_view import plot_dag
        import matplotlib.pyplot as plt
        plot_dag(graph, save_path=args.output)
        if not args.output:
            plt.show()

    elif args.command == "serve":
        from .api.server import main as serve_main
        if args.bucket:
            import os
            os.environ["GEOINV3D_S3_BUCKET"] = args.bucket
        if args.region:
            import os
            os.environ["GEOINV3D_AWS_REGION"] = args.region
        sys.argv = ["geoinv3d", "--host", args.host,
                     "--port", str(args.port)]
        if args.reload:
            sys.argv.append("--reload")
        serve_main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
