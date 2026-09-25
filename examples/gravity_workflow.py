"""Example: Build a gravity forward modeling workflow with DAG tracking.

This example demonstrates:
1. Creating a mesh and model as DAG nodes
2. Setting up a gravity survey
3. Running forward modeling
4. Changing parameters and seeing dirty propagation
5. Saving/loading the workflow
6. Visualizing the DAG

Run with: python examples/gravity_workflow.py
(Requires: numpy, matplotlib. SimPEG needed only for forward modeling.)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geoinv3d.core import Graph, graph_to_dict
from geoinv3d.core.serialize import save_workflow
from geoinv3d.nodes.input_nodes import MeshCreateNode, ModelCreateNode, SurveyCreateNode
from geoinv3d.nodes.transform_nodes import ScaleNode
from geoinv3d.nodes.regularization_nodes import RegularizationNode
from geoinv3d.viz.dag_view import dag_to_mermaid


def main():
    print("=" * 60)
    print("GeoInv3D — Gravity Workflow Example")
    print("=" * 60)

    # ── Step 1: Build the DAG ──────────────────────────────────
    graph = Graph()

    # Create a 20x20x10 mesh with 100m spacing
    mesh_node = MeshCreateNode(
        nx=20, ny=20, nz=10,
        dx=100.0, dy=100.0, dz=50.0,
        name="Survey Mesh",
    )
    graph.add(mesh_node)
    print(f"\n[OK] Created mesh node: {mesh_node}")

    # Create a constant density model (background = 0 g/cm³)
    model_node = ModelCreateNode(
        mesh_node, value=0.0, prop="density", name="Background Density"
    )
    graph.add(model_node)
    print(f"[OK] Created model node: {model_node}")

    # Scale the model (simulate an anomaly by scaling)
    scaled = ScaleNode(model_node, factor=2.5, name="Anomaly Scale")
    graph.add(scaled)
    print(f"[OK] Created scale node: {scaled}")

    # Create survey stations on a regular grid at z=0
    n_stations = 100
    np.random.seed(42)
    locs = np.column_stack([
        np.random.uniform(200, 1800, n_stations),
        np.random.uniform(200, 1800, n_stations),
        np.zeros(n_stations),
    ])
    survey_node = SurveyCreateNode(
        locations=locs,
        observed=np.zeros(n_stations),
        std=np.ones(n_stations) * 0.01,
        method="gravity",
        name="Gravity Survey",
    )
    graph.add(survey_node)
    print(f"[OK] Created survey node: {survey_node}")

    # Add regularization config
    reg_node = RegularizationNode(
        alpha_s=1e-4, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        name="Smoothness Reg",
    )
    graph.add(reg_node)
    print(f"[OK] Created regularization node: {reg_node}")

    # ── Step 2: Evaluate nodes ─────────────────────────────────
    print("\n--- Evaluating DAG ---")

    mesh = mesh_node.evaluate()
    print(f"  Mesh: {mesh.shape}, {mesh.n_cells} cells")

    model = model_node.evaluate()
    print(f"  Model: {model.prop.value}, {len(model.values)} values")

    reg = reg_node.evaluate()
    print(f"  Regularization: alpha_s={reg.alpha_s}, type={reg.reg_type}")

    # ── Step 3: Parameter change → dirty propagation ───────────
    print("\n--- Parameter change tracking ---")
    print(f"  Scale node dirty before change: {scaled._dirty}")

    model_node.set_value(1.5)
    print(f"  Changed model value to 1.5")
    print(f"  Model node dirty: {model_node._dirty}")
    print(f"  Scale node dirty (downstream): {scaled._dirty}")

    # Re-evaluate — only recomputes what changed
    new_model = scaled.evaluate()
    print(f"  Re-evaluated scaled model: mean={new_model.values.mean():.1f}")

    # ── Step 4: Inspect the DAG ────────────────────────────────
    print("\n--- DAG structure ---")
    print(f"  Total nodes: {len(graph)}")
    for node in graph.nodes:
        inp_ids = [n.id for n in node.inputs]
        print(f"  #{node.id} {type(node).__name__:30s} inputs={inp_ids}")

    # ── Step 5: Serialize ──────────────────────────────────────
    doc = graph_to_dict(graph)
    print(f"\n--- Serialized DAG ---")
    print(f"  Version: {doc['version']}")
    print(f"  Nodes: {len(doc['nodes'])}")
    for n in doc["nodes"]:
        print(f"    #{n['id']} {n['type']}: {n['name']}")

    # Save workflow
    workflow_path = "example_gravity.geoinv3d.json"
    save_workflow(graph, workflow_path)
    print(f"\n[OK] Workflow saved to: {workflow_path}")

    # ── Step 6: Mermaid diagram ────────────────────────────────
    mermaid = dag_to_mermaid(graph)
    print(f"\n--- Mermaid diagram ---")
    print(mermaid)

    # ── Step 7: Lineage / provenance ───────────────────────────
    print(f"\n--- Lineage of Scale node ---")
    lineage = graph.lineage(scaled)
    for node in lineage:
        print(f"  #{node.id} {node.name}")

    print("\n" + "=" * 60)
    print("Done! The workflow DAG is fully tracked and serialized.")
    print("=" * 60)


if __name__ == "__main__":
    main()
