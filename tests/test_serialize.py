"""Tests for DAG serialization and deserialization."""

import json
import pytest
import numpy as np

from geoinv3d.core import Graph, graph_to_dict, graph_from_dict
from geoinv3d.core.serialize import save_workflow, load_workflow, provenance_for
from geoinv3d.nodes.input_nodes import (
    MeshCreateNode, ModelCreateNode, SurveyCreateNode,
)
from geoinv3d.nodes.transform_nodes import ScaleNode
from geoinv3d.nodes.regularization_nodes import RegularizationNode


class TestSerialize:
    def _build_graph(self):
        graph = Graph()
        mesh = MeshCreateNode(nx=5, ny=5, nz=3, name="Mesh")
        model = ModelCreateNode(mesh, value=1.0, prop="density", name="Model")
        scaled = ScaleNode(model, factor=2.0, name="Scaled")
        graph.add(scaled)
        return graph, mesh, model, scaled

    def test_graph_to_dict(self):
        graph, mesh, model, scaled = self._build_graph()
        doc = graph_to_dict(graph)
        assert doc["version"] == 1
        assert len(doc["nodes"]) == 3

    def test_round_trip(self):
        graph, mesh, model, scaled = self._build_graph()

        # Evaluate before serialization
        result_before = scaled.evaluate()

        # Serialize
        doc = graph_to_dict(graph)
        json_str = json.dumps(doc)

        # Deserialize
        doc2 = json.loads(json_str)
        graph2, id_map = graph_from_dict(doc2)
        assert len(graph2) == 3

        # Find the ScaleNode in rebuilt graph
        scale_nodes = [n for n in graph2.nodes if type(n).__name__ == "ScaleNode"]
        assert len(scale_nodes) == 1
        result_after = scale_nodes[0].evaluate()
        np.testing.assert_array_equal(result_before.values, result_after.values)

    def test_node_types_preserved(self):
        graph, *_ = self._build_graph()
        doc = graph_to_dict(graph)
        types = {n["type"] for n in doc["nodes"]}
        assert "MeshCreateNode" in types
        assert "ModelCreateNode" in types
        assert "ScaleNode" in types

    def test_params_preserved(self):
        graph, mesh, model, scaled = self._build_graph()
        doc = graph_to_dict(graph)
        mesh_dict = next(n for n in doc["nodes"] if n["type"] == "MeshCreateNode")
        assert mesh_dict["params"]["nx"] == 5
        assert mesh_dict["params"]["dy"] == 100.0

    def test_input_edges_preserved(self):
        graph, mesh, model, scaled = self._build_graph()
        doc = graph_to_dict(graph)
        scale_dict = next(n for n in doc["nodes"] if n["type"] == "ScaleNode")
        model_dict = next(n for n in doc["nodes"] if n["type"] == "ModelCreateNode")
        assert model_dict["id"] in scale_dict["inputs"]

    def test_provenance(self):
        graph, mesh, model, scaled = self._build_graph()
        scaled.evaluate()
        prov = provenance_for(scaled, graph)
        assert prov["tool"] == "geoinv3d"
        assert len(prov["lineage"]) == 3

    def test_save_load_workflow(self, tmp_path):
        graph, *_ = self._build_graph()
        path = str(tmp_path / "test.geoinv3d.json")
        save_workflow(graph, path)

        graph2, id_map = load_workflow(path)
        assert len(graph2) == len(graph)

    def test_cycle_detection(self):
        doc = {
            "version": 1,
            "nodes": [
                {"id": 1, "type": "MeshCreateNode", "name": "A",
                 "params": {"nx": 5, "ny": 5, "nz": 3, "dx": 100,
                            "dy": 100, "dz": 50, "origin": [0, 0, 0]},
                 "inputs": [2]},
                {"id": 2, "type": "MeshCreateNode", "name": "B",
                 "params": {"nx": 5, "ny": 5, "nz": 3, "dx": 100,
                            "dy": 100, "dz": 50, "origin": [0, 0, 0]},
                 "inputs": [1]},
            ],
        }
        with pytest.raises(ValueError, match="cycle"):
            graph_from_dict(doc)


class TestRegularizationSerialize:
    def test_round_trip(self):
        graph = Graph()
        reg = RegularizationNode(alpha_s=0.01, alpha_x=2.0, name="MyReg")
        graph.add(reg)

        doc = graph_to_dict(graph)
        graph2, id_map = graph_from_dict(doc)
        reg2 = list(graph2.nodes)[0]
        result = reg2.evaluate()
        assert result.alpha_s == 0.01
        assert result.alpha_x == 2.0


class TestSurveySerialize:
    def test_round_trip(self):
        graph = Graph()
        locs = np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0]], dtype=float)
        obs = np.array([1.0, 2.0, 3.0])
        std = np.array([0.1, 0.1, 0.1])
        survey = SurveyCreateNode(locs, obs, std, method="gravity", name="S")
        graph.add(survey)

        doc = graph_to_dict(graph)
        graph2, _ = graph_from_dict(doc)
        s2 = list(graph2.nodes)[0]
        result = s2.evaluate()
        np.testing.assert_array_equal(result.observed, obs)
        assert result.method == "gravity"
