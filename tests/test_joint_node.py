"""JointRegularizedInversionNode: both joint methods as a DAG node."""

import contextlib
import io
import json

import numpy as np
import pytest

from geoinv3d.core.node import Graph
from geoinv3d.core.serialize import graph_from_dict, graph_to_dict, node_to_dict
from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.methods.gravity import GravityMethod
from geoinv3d.methods.magnetics import MagneticsMethod
from geoinv3d.nodes.input_nodes import MeshCreateNode, ModelFromArrayNode, SurveyCreateNode
from geoinv3d.nodes.inversion_nodes import JointRegularizedInversionNode
from geoinv3d.nodes.regularization_nodes import RegularizationNode

FIELD = [50000.0, 60.0, 10.0]
MESH = Mesh3D.uniform(10, 10, 5, 50.0, 50.0, 50.0, origin=(0.0, 0.0, -250.0))
CC = MESH.to_discretize().cell_centers
BODY = ((abs(CC[:, 0] - 250) < 80) & (abs(CC[:, 1] - 250) < 80)
        & (CC[:, 2] > -175) & (CC[:, 2] < -75))
TYPES = ["gravity", "gravity", "magnetics"]
KWARGS = [{}, {"component": "gzz"}, {"inducing_field": FIELD}]


@pytest.fixture(scope="module")
def inputs():
    """gz, gzz and TMI of one block: start models, surveys and alphas."""
    rng = np.random.default_rng(0)
    xy = np.linspace(25, 475, 7)
    xx, yy = np.meshgrid(xy, xy)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 20.0)])
    mesh_node = MeshCreateNode(nx=10, ny=10, nz=5, dx=50.0, dy=50.0, dz=50.0,
                               origin=(0.0, 0.0, -250.0))
    nodes = []
    for method, prop, mtype in ((GravityMethod(), "density", "gravity"),
                                (GravityMethod("gzz"), "density", "gravity"),
                                (MagneticsMethod(inducing_field=tuple(FIELD)), "susceptibility",
                                 "magnetics")):
        s = SurveyData(locations=locs, observed=np.zeros(49), std=np.ones(49))
        d = method.make_simulation_full(MESH, s).dpred((0.3 if prop == "density" else 0.05) * BODY)
        sigma = 0.02 * abs(d).max()
        nodes += [ModelFromArrayNode(mesh_node, np.zeros(MESH.n_cells), prop=prop),
                  SurveyCreateNode(locs, d + rng.normal(scale=sigma, size=d.size),
                                   np.full(49, sigma), method=mtype),
                  RegularizationNode(alpha_s=1.0)]
    return nodes


def _output(node):
    graph = Graph()
    graph.add(node)
    with contextlib.redirect_stdout(io.StringIO()):
        node.evaluate()
    out = next(n for n in graph_to_dict(graph, include_outputs=True)["nodes"]
               if n["id"] == node.id)["output"]
    json.dumps(out, allow_nan=False)   # strict JSON for the viewer
    return out


def test_params_roundtrip_and_validation(inputs):
    node = JointRegularizedInversionNode(
        TYPES, KWARGS, models=["density", "density", None], regularization_type="mgs",
        regularizations=[None, None, {"regularization_type": "tv", "bounds_lower": 0.0}],
        group_lasso={"gl_lambda2": 0.5}, bounds=(-1.0, 1.0), inputs=inputs)
    clone = JointRegularizedInversionNode.from_params(node.params(), inputs)
    assert clone.params() == node.params()
    json.dumps(node_to_dict(node))
    # and through a saved workflow
    graph = Graph()
    graph.add(node)
    _, ids = graph_from_dict(json.loads(json.dumps(graph_to_dict(graph))))
    rebuilt = next(n for n in ids.values() if isinstance(n, JointRegularizedInversionNode))
    assert rebuilt.params() == node.params() and len(rebuilt.inputs) == 9
    with pytest.raises(ValueError, match="regularization_type"):
        JointRegularizedInversionNode(TYPES, regularization_type="l0", inputs=inputs)
    with pytest.raises(ValueError, match="3 per dataset"):
        JointRegularizedInversionNode(TYPES, inputs=inputs[:6])
    with pytest.raises(ValueError, match="models has 2"):
        JointRegularizedInversionNode(TYPES, models=["a", "b"], inputs=inputs)
    with pytest.raises(ValueError, match="bogus"):
        JointRegularizedInversionNode(TYPES, regularizations=[{"bogus": 1}, None, None],
                                      inputs=inputs)
    with pytest.raises(ValueError, match="gl_bogus"):
        JointRegularizedInversionNode(TYPES, group_lasso={"gl_bogus": 1}, inputs=inputs)


def test_joint_inversion_per_model_settings(inputs):
    """Method one: sparse density from gz + gzz, bounded MGS susceptibility."""
    node = JointRegularizedInversionNode(
        TYPES, KWARGS, models=["density", "density", None], regularization_type="sparse",
        regularizations=[None, None, {"regularization_type": "mgs", "bounds_lower": 0.0}],
        max_iter=15, max_irls_iterations=5, inputs=inputs)
    out = _output(node)
    result = node._output
    assert out["regularization"] == "joint_mixed"
    assert set(result.final_models) == {"density", "magnetics"}
    assert result.final_models["density"].prop.value == "density"
    assert result.final_models["magnetics"].prop.value == "susceptibility"
    assert result.recovered_models["magnetics"].min() >= 0.0
    assert out["models"]["density"]["kind"] == "sparse"
    assert out["models"]["magnetics"]["kind"] == "mgs"
    assert {k: v["model"] for k, v in out["datasets"].items()} == {
        "gravity": "density", "gravity_2": "density", "magnetics": "magnetics"}
    assert all(v["chi2"] < 3 * 49 for v in out["datasets"].values())
    assert out["n_iterations"] == len(out["iterations"]) > 0
    assert set(out["final_models"]) == {"density", "magnetics"}


def test_group_lasso_node(inputs):
    """Method two: the group lasso with a cross-gradient, its lambda1 sweep in the output."""
    node = JointRegularizedInversionNode(
        TYPES, KWARGS, models=["density", "density", None], regularization_type="group_lasso",
        group_lasso={"gl_n_lambda1": 5, "gl_lambda1_decades": 2.0, "gl_cross_gradient": 0.05},
        inputs=inputs)
    out = _output(node)
    assert out["regularization"] == "group_lasso_ADMM"
    sel = out["selection"]
    assert sel["parameter"] == "λ1" and len(sel["values"]) == len(sel["misfit"]) == 5
    assert sel["values"] == sorted(sel["values"], reverse=True)
    gl = out["group_lasso"]
    assert gl["models"] == {"density": ["gravity", "gravity_2"], "magnetics": ["magnetics"]}
    assert gl["cross_gradient"] == 0.05
    result = node._output
    assert result.recovered_models["density"][BODY].mean() > 0.02
    assert set(result.per_method_phi_d) == {"gravity", "gravity_2", "magnetics"}
