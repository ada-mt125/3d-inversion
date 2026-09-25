"""Tests for RegularizedInversionNode and its viewer output."""

import json

import numpy as np
import pytest

from geoinv3d.core.node import Graph
from geoinv3d.core.serialize import graph_to_dict, node_to_dict
from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.methods.magnetics import MagneticsMethod
from geoinv3d.nodes.input_nodes import MeshCreateNode, ModelFromArrayNode, SurveyCreateNode
from geoinv3d.nodes.inversion_nodes import RegularizedInversionNode
from geoinv3d.nodes.regularization_nodes import RegularizationNode

FIELD = [50000.0, 60.0, 10.0]


@pytest.fixture(scope="module")
def inputs():
    mesh = Mesh3D.uniform(10, 10, 5, 50.0, 50.0, 50.0, origin=(0.0, 0.0, -250.0))
    cc = mesh.to_discretize().cell_centers
    m_true = ((abs(cc[:, 0] - 250) < 80) & (abs(cc[:, 1] - 250) < 80)
              & (cc[:, 2] > -175) & (cc[:, 2] < -75)) * 0.05
    xy = np.linspace(25, 475, 7)
    xx, yy = np.meshgrid(xy, xy)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 20.0)])
    survey = SurveyData(locations=locs, observed=np.zeros(49), std=np.ones(49))
    d = MagneticsMethod(inducing_field=tuple(FIELD)).make_simulation_full(mesh, survey).dpred(m_true)
    sigma = 0.02 * abs(d).max()
    d = d + np.random.default_rng(0).normal(scale=sigma, size=d.size)

    mesh_node = MeshCreateNode(nx=10, ny=10, nz=5, dx=50.0, dy=50.0, dz=50.0,
                               origin=(0.0, 0.0, -250.0))
    start = ModelFromArrayNode(mesh_node, np.zeros(mesh.n_cells), prop="susceptibility")
    survey_node = SurveyCreateNode(locs, d, np.full(49, sigma), method="magnetics")
    reg = RegularizationNode(alpha_s=1e-3)
    return start, survey_node, reg


def _node(inputs, **kwargs):
    start, survey, reg = inputs
    return RegularizedInversionNode(start, survey, reg, method_type="magnetics",
                                    method_kwargs={"inducing_field": FIELD},
                                    max_iter=30, max_irls_iterations=10, **kwargs)


def test_params_roundtrip(inputs):
    node = _node(inputs, regularization_type="mgs", focusing_scale=0.5, bounds=(0.0, 0.1),
                 beta_selection="lcurve")
    clone = RegularizedInversionNode.from_params(node.params(), list(inputs))
    assert clone.params() == node.params()
    json.dumps(node_to_dict(node))  # serializable
    with pytest.raises(ValueError, match="regularization_type"):
        _node(inputs, regularization_type="l0")


def test_cda_path_in_viewer_output(inputs):
    node = _node(inputs, regularization_type="l1l2", l1_ratio=0.8, lambda_decades=3.0)
    graph = Graph()
    graph.add(node)
    result = node.evaluate()
    assert result.final_model.values.max() > 0.01
    out = next(n for n in graph_to_dict(graph, include_outputs=True)["nodes"]
               if n["id"] == node.id)["output"]
    json.dumps(out)
    assert out["regularization"] == "elastic_net_CDA"
    assert out["n_iterations"] == len(out["iterations"]) == 31  # one per lambda
    sel = out["selection"]
    assert sel["parameter"] == "λ" and sel["criterion"] == "lcurve"
    assert sel["n_data"] == 49 and sel["weighting"] == "S1"
    assert len(sel["values"]) == len(sel["misfit"]) == len(sel["penalty"]) == 31
    assert sel["selected"] == pytest.approx(sel["chosen"]["L-curve"])
    assert set(sel["chosen"]) >= {"L-curve", "Discrepancy"}


def test_irls_sweep_in_viewer_output(inputs):
    node = _node(inputs, regularization_type="sparse", bounds=(0.0, 1.0),
                 beta_selection="gcv")
    graph = Graph()
    graph.add(node)
    node.evaluate()
    out = next(n for n in graph_to_dict(graph, include_outputs=True)["nodes"]
               if n["id"] == node.id)["output"]
    sel = out["selection"]
    assert out["regularization"] == "sparse_IRLS"
    assert sel["parameter"] == "β" and sel["criterion"] == "gcv"
    assert sel["gcv"] is not None and len(sel["gcv"]) == len(sel["values"]) == 13
    assert set(sel["chosen"]) == {"L-curve", "GCV", "Discrepancy"}
    assert sel["selected"] == pytest.approx(sel["chosen"]["GCV"])


def test_discrepancy_run_has_no_selection(inputs):
    node = _node(inputs, regularization_type="tv", bounds=(0.0, 1.0))
    graph = Graph()
    graph.add(node)
    node.evaluate()
    out = next(n for n in graph_to_dict(graph, include_outputs=True)["nodes"]
               if n["id"] == node.id)["output"]
    assert out["regularization"] == "total_variation"
    assert "selection" not in out
    assert all({"phi_d", "phi_m", "beta", "model_max"} <= set(it) for it in out["iterations"])
