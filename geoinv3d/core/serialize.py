"""Graph (de)serialization and export provenance.

The transformation graph is the system of record, so it must be serializable:
every node knows its own parameters (params()) and how to rebuild itself
(from_params()); this module turns a graph into a portable JSON document
and back.
"""

from __future__ import annotations

import heapq
import json
from datetime import datetime, timezone
from typing import Callable, Optional

from .node import NODE_REGISTRY, Graph, Node

SCHEMA_VERSION = 1


def node_to_dict(node: Node, include_outputs: bool = False) -> dict:
    """One node as {id, type, name, params, inputs:[ids]}."""
    d = {
        "id": node.id,
        "type": type(node).__name__,
        "name": node.name,
        "params": node.params(),
        "inputs": [i.id for i in node.inputs],
    }
    if include_outputs and node._output is not None:
        output_summary = _extract_output_summary(node)
        if output_summary:
            d["output"] = output_summary
    return d


def _extract_output_summary(node: Node) -> dict | None:
    """Extract a JSON-serializable summary of a node's computed output."""
    import numpy as np
    from ..datamodel.result import InversionResult, JointInversionResult
    from ..datamodel.model import PhysicalModel
    from ..datamodel.survey import SurveyData
    from ..datamodel.mesh import Mesh3D

    out = node._output
    if isinstance(out, InversionResult):
        result = {
            "type": "InversionResult",
            "method": out.method,
            "converged": out.converged,
            "n_iterations": out.n_iterations,
            "iterations": [],
        }
        for snap in out.iterations:
            vals = np.asarray(snap.model_values)
            result["iterations"].append({
                "iteration": snap.iteration,
                "phi_d": snap.phi_d,
                "phi_m": snap.phi_m,
                "phi_total": snap.phi_total,
                "beta": snap.beta,
                "model_min": float(vals.min()),
                "model_max": float(vals.max()),
                "model_mean": float(vals.mean()),
                "model_std": float(vals.std()),
            })
        if out.final_model is not None:
            fv = np.asarray(out.final_model.values)
            result["final_model"] = {
                "prop": out.final_model.prop.value
                if hasattr(out.final_model.prop, "value") else str(out.final_model.prop),
                "min": float(fv.min()),
                "max": float(fv.max()),
                "mean": float(fv.mean()),
                "n_cells": int(fv.size),
            }
        return result

    if isinstance(out, JointInversionResult):
        result = {
            "type": "JointInversionResult",
            "methods": out.methods,
            "weights": out.weights,
            "converged": out.converged,
            "n_iterations": out.n_iterations,
            "iterations": [],
        }
        for snap in out.iterations:
            vals = np.asarray(snap.model_values)
            result["iterations"].append({
                "iteration": snap.iteration,
                "phi_d": snap.phi_d,
                "phi_m": snap.phi_m,
                "phi_total": snap.phi_total,
                "beta": snap.beta,
                "model_min": float(vals.min()),
                "model_max": float(vals.max()),
                "model_mean": float(vals.mean()),
            })
        return result

    if isinstance(out, PhysicalModel):
        vals = np.asarray(out.values)
        return {
            "type": "PhysicalModel",
            "prop": out.prop.value if hasattr(out.prop, "value") else str(out.prop),
            "n_cells": int(vals.size),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
        }

    if isinstance(out, Mesh3D):
        return {
            "type": "Mesh3D",
            "shape": list(out.shape),
            "n_cells": out.n_cells,
        }

    if isinstance(out, SurveyData):
        obs = np.asarray(out.observed)
        return {
            "type": "SurveyData",
            "n_stations": int(obs.size),
            "method": out.method,
            "data_min": float(obs.min()),
            "data_max": float(obs.max()),
        }

    if isinstance(out, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(out.shape),
            "min": float(out.min()),
            "max": float(out.max()),
            "mean": float(out.mean()),
        }

    return None


def graph_to_dict(graph: Graph, include_outputs: bool = False) -> dict:
    """The whole graph as a JSON-ready document."""
    nodes = sorted(graph.nodes, key=lambda n: n.id)
    d = {
        "version": SCHEMA_VERSION,
        "nodes": [node_to_dict(n, include_outputs=include_outputs) for n in nodes],
    }
    if include_outputs:
        d["created"] = datetime.now(timezone.utc).isoformat()
    return d


def graph_from_dict(
    doc: dict, file_resolver: Optional[Callable[[str], str]] = None
) -> tuple[Graph, dict[int, Node]]:
    """Rebuild a live Graph from a graph_to_dict document.

    Uses Kahn's topological sort to instantiate nodes in dependency order.
    Returns (graph, id_map) where id_map maps saved ids to new nodes.
    """
    node_dicts: dict[int, dict] = {d["id"]: d for d in doc["nodes"]}
    id_map: dict[int, Node] = {}
    graph = Graph()

    for saved_id, d in node_dicts.items():
        for iid in d.get("inputs", []):
            if iid not in node_dicts:
                raise ValueError(
                    f"node {saved_id} ({d.get('type')}) references missing input {iid}"
                )

    waiting = {
        sid: len(set(d.get("inputs", []))) for sid, d in node_dicts.items()
    }
    deps: dict[int, list[int]] = {sid: [] for sid in node_dicts}
    for sid, d in node_dicts.items():
        for iid in set(d.get("inputs", [])):
            deps[iid].append(sid)

    ready = [sid for sid, count in waiting.items() if count == 0]
    heapq.heapify(ready)

    while ready:
        saved_id = heapq.heappop(ready)
        d = node_dicts[saved_id]
        inputs = [id_map[iid] for iid in d.get("inputs", [])]
        cls = NODE_REGISTRY[d["type"]]
        params = dict(d.get("params", {}))
        if file_resolver is not None and cls.reads_file:
            params["path"] = file_resolver(params["path"])
        node = cls.from_params(params, inputs)
        if d.get("name"):
            node.name = d["name"]
        id_map[saved_id] = node
        graph.add(node)
        for dep in deps[saved_id]:
            waiting[dep] -= 1
            if waiting[dep] == 0:
                heapq.heappush(ready, dep)

    if len(id_map) != len(node_dicts):
        raise ValueError("cycle detected in saved graph")

    return graph, id_map


def provenance_for(node: Node, graph: Graph) -> dict:
    """Build a provenance record for a node: its full lineage sub-DAG."""
    lineage = graph.lineage(node)
    return {
        "tool": "geoinv3d",
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "output_node": node_to_dict(node),
        "lineage": [node_to_dict(n) for n in sorted(lineage, key=lambda n: n.id)],
    }


def write_provenance(node: Node, graph: Graph, path: str) -> str:
    """Write provenance JSON sidecar and return its path."""
    prov_path = path + ".provenance.json"
    with open(prov_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(provenance_for(node, graph), fh, indent=2)
    return prov_path


def save_workflow(
    graph: Graph, path: str, *, include_outputs: bool = False
) -> None:
    """Save the entire DAG to a JSON workflow file.

    With include_outputs=True, each evaluated node also stores a summary of its
    computed output (iteration history, model statistics, etc.).
    """
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(graph_to_dict(graph, include_outputs=include_outputs), fh, indent=2)


def load_workflow(
    path: str, file_resolver: Optional[Callable[[str], str]] = None
) -> tuple[Graph, dict[int, Node]]:
    """Load a DAG from a saved JSON workflow file."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return graph_from_dict(doc, file_resolver)
