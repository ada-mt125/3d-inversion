"""Viewer demo: every regularization and beta/lambda choice as DAG nodes.

Builds a workflow on the magnetic synthetic of ``l1l2_paper_synthetic.py``
(TMI, B0 = 42000 nT, I = 45, D = 30 deg, 26 x 26 stations at 2 km height,
chi = 0.03 SI prism 2-10 km deep, sigma = 2 nT) with one
``RegularizedInversionNode`` per choice:

  * L1–L2, Utsugi (2019) CDA lambda path, wS1, alpha = 0.8, lambda by L-curve
  * the same with lambda by GCV
  * MGS focusing with the a-priori bound chi <= 0.04
  * total variation
  * lp-norm sparse with beta by the L-curve (fixed-beta sweep)
  * smooth L2

and writes ``regularization_comparison.geoinv3d.json`` plus its interactive
viewer (``..._viewer.html``).  Each inversion node carries the recovered and
true models (3D tab), the data fit, and, where a sweep ran, the L-curve,
chi^2 and GCV curves with every criterion's choice (Convergence tab).

Run with:
    python examples/regularization_viewer_demo.py [--out DIR]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import l1l2_paper_synthetic as syn
from geoinv3d.core.node import Graph
from geoinv3d.core.serialize import graph_to_dict
from geoinv3d.nodes.input_nodes import MeshCreateNode, ModelFromArrayNode, SurveyCreateNode
from geoinv3d.nodes.inversion_nodes import RegularizedInversionNode
from geoinv3d.nodes.regularization_nodes import RegularizationNode
from geoinv3d.viz.serve_dag import generate_viewer

SIGMA = 2.0
CHOICES = [
    ("L1–L2 (CDA, L-curve λ)", dict(regularization_type="l1l2", l1_ratio=0.8,
                                     l1l2_weighting="S1", beta_selection="auto")),
    ("L1–L2 (CDA, GCV λ)", dict(regularization_type="l1l2", l1_ratio=0.8,
                                 l1l2_weighting="S1", beta_selection="gcv")),
    ("MGS focusing χ≤0.04", dict(regularization_type="mgs", bounds=(0.0, 0.04))),
    ("Total variation", dict(regularization_type="tv", bounds=(0.0, 1.0))),
    ("Sparse (L-curve β)", dict(regularization_type="sparse", bounds=(0.0, 1.0),
                                beta_selection="lcurve")),
    ("Smooth L2", dict(regularization_type="l2")),
]


def _round(a, digits=5):
    return [float(f"{v:.{digits}g}") for v in np.asarray(a, dtype=float)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    syn.setup_problem()
    P = syn._P
    d_obs = syn.observed(SIGMA)
    dmesh = P["dmesh"]

    graph = Graph()
    mesh_node = MeshCreateNode(nx=syn.NX, ny=syn.NY, nz=syn.NZ, dx=syn.DX, dy=syn.DY,
                               dz=syn.DZ, origin=syn.ORIGIN, name="Mesh 2×2×1 km")
    start = ModelFromArrayNode(mesh_node, np.zeros(dmesh.n_cells), prop="susceptibility",
                               name="Start model (χ = 0)")
    survey = SurveyCreateNode(P["locs"], d_obs, np.full(len(d_obs), SIGMA), method="magnetics",
                              name=f"TMI 26×26, σ = {SIGMA:g} nT")
    reg = RegularizationNode(alpha_s=1e-4, name="Regularization weights")
    graph.add(ModelFromArrayNode(mesh_node, P["m_true"], prop="susceptibility",
                                 name="True prism (χ = 0.03)"))
    inversions = []
    for name, kwargs in CHOICES:
        node = RegularizedInversionNode(
            start, survey, reg, method_type="magnetics",
            method_kwargs={"inducing_field": [syn.B0, syn.INC, syn.DEC]},
            max_iter=60, max_irls_iterations=60, name=name, **kwargs)
        graph.add(node)
        inversions.append(node)

    for node in inversions:
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            node.evaluate()
        m = node.evaluate().final_model.values
        print(f"{node.name:<24} peak χ = {m.max():.4f}  ({time.time() - t0:.0f} s)", flush=True)

    workflow = graph_to_dict(graph, include_outputs=True)
    # Add the 3D model and data fit the viewer's other tabs display
    edges = {"x_edges": _round(dmesh.nodes_x), "y_edges": _round(dmesh.nodes_y),
             "z_edges": _round(dmesh.nodes_z)}
    shape = dict(zip(("nx", "ny", "nz"), map(int, dmesh.shape_cells)))
    locs = P["locs"]
    by_id = {n.id: n for n in inversions}
    for entry in workflow["nodes"]:
        node = by_id.get(entry["id"])
        if node is None:
            continue
        m = np.asarray(node.evaluate().final_model.values)
        pred = P["G"] @ m
        resid = d_obs - pred
        entry["output"]["final_model"]["unit"] = "SI"
        entry["output"]["model_3d"] = {**shape, **edges, "values": _round(m, 4),
                                       "true_values": _round(P["m_true"], 4)}
        entry["output"]["data_fit"] = {
            "unit": "nT", "x_stations": _round(locs[:, 0], 6),
            "y_stations": _round(locs[:, 1], 6), "observed": _round(d_obs),
            "predicted": _round(pred), "residuals": _round(resid),
            "rms": float(np.sqrt(np.mean(resid**2))),
            "chi2": float(np.sum((resid / SIGMA) ** 2) / len(resid)),
        }

    path = os.path.join(args.out, "regularization_comparison.geoinv3d.json")
    with open(path, "w") as f:
        json.dump(workflow, f, separators=(",", ":"))
    html = generate_viewer(path)
    print(f"workflow: {path} ({os.path.getsize(path) / 1e6:.1f} MB)\nviewer:   {html}")


if __name__ == "__main__":
    main()
