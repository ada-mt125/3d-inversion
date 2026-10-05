"""Results -> viewer workflow: data.npz in the result, and build_workflow."""

import io
import json
import zipfile

import numpy as np
import pytest

from geoinv3d.cloud.worker import pack_result, run_data_pipeline
from geoinv3d.viz.result_workflow import _g, build_workflow, load_result, result_mesh
from tests.test_data_pipeline import _single, _station_grid, _synthetic, _write_csv


def _chain(by_id, node):
    """Names of the branch nodes above a run, nearest first, and the inputs of the top one."""
    names = []
    while by_id[node["inputs"][0]]["type"].endswith("BranchNode"):
        node = by_id[node["inputs"][0]]
        names.append(node["name"])
    return names, node["inputs"]


def _run(tmp_path, **extra):
    locs = _station_grid(0.0)
    _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
    params = _single("gravity", ["g.csv"], param_mode="manual", regularization_type="l2",
                     max_iter=5, **extra)
    result = run_data_pipeline(params, str(tmp_path))
    zpath = tmp_path / "result.zip"
    pack_result(result, str(zpath))
    return result, zpath, locs


class TestResultData:
    def test_result_keeps_the_data_next_to_the_model(self, tmp_path):
        result, zpath, locs = _run(tmp_path)
        data = result["data"]
        assert data["observed"].shape == data["predicted"].shape == (len(locs),)
        np.testing.assert_allclose(data["locations"][:, :2], locs[:, :2])
        # the predicted data are those of the final model: chi^2 = the last phi_d
        chi2 = np.sum(((data["observed"] - data["predicted"]) / data["std"]) ** 2)
        assert chi2 == pytest.approx(result["iterations"][-1]["phi_d"], rel=1e-6)
        assert result["settings"]["regularization_type"] == "l2"
        with zipfile.ZipFile(zpath) as zf:
            assert "data.npz" in zf.namelist()
            meta = json.loads(zf.read("result.json"))
        assert "data" not in meta and "predicted" not in meta


class TestBuildWorkflow:
    def test_tensor_result(self, tmp_path):
        result, zpath, locs = _run(tmp_path)
        run = load_result(zpath)
        mesh = result_mesh(run)
        wf = build_workflow([run], true_model=np.ones(mesh.n_cells))
        types = [n["type"] for n in wf["nodes"]]
        assert types == ["MeshCreateNode", "SurveyCreateNode", "ModelFromArrayNode",
                         "RegularizedInversionNode"]
        out = wf["nodes"][-1]["output"]
        m3 = out["model_3d"]
        # the core only: 6 x 6 cells of 100 m over the 600 m survey, 4 layers
        assert (m3["nx"], m3["ny"], m3["nz"]) == (6, 6, 4)
        assert len(m3["values"]) == len(m3["true_values"]) == 6 * 6 * 4
        assert m3["z_edges"][0] > m3["z_edges"][-1]   # top down
        fit = out["data_fit"]
        assert len(fit["observed"]) == len(locs)
        assert fit["chi2"] == pytest.approx(result["iterations"][-1]["phi_d"] / len(locs),
                                            rel=1e-3)
        params = wf["nodes"][-1]["params"]
        assert params["regularization_type"] == "l2" and params["max_iter"] == 5
        json.dumps(wf)   # plain JSON for the viewer

    def test_octree_result_is_sampled_on_a_grid(self, tmp_path):
        _, zpath, _ = _run(tmp_path, mesh_type="octree")
        wf = build_workflow([load_result(zpath)])
        m3 = wf["nodes"][-1]["output"]["model_3d"]
        assert m3["nx"] * m3["ny"] * m3["nz"] == len(m3["values"])
        assert m3["nx"] >= 6 and m3["nz"] >= 4

    def test_runs_on_different_meshes_get_their_own_mesh_node(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        _, za, _ = _run(tmp_path / "a")
        _, zb, _ = _run(tmp_path / "b", mesh_type="octree")
        runs = [load_result(za), load_result(za), load_result(zb)]
        runs[2]["_backend"] = "local"
        wf = build_workflow(runs)
        by_id = {n["id"]: n for n in wf["nodes"]}
        meshes = [n for n in wf["nodes"] if n["type"] == "MeshCreateNode"]
        surveys = [n for n in wf["nodes"] if n["type"] == "SurveyCreateNode"]
        assert len(meshes) == 2 and len(surveys) == 1   # same data, two meshes
        inv = [n for n in wf["nodes"] if n["type"] == "RegularizedInversionNode"]
        assert [n["id"] for n in inv] == [10, 11, 12]
        # each mesh has its own tree; the two identical runs share every branch node
        assert inv[0]["inputs"] == inv[1]["inputs"]
        names, top = _chain(by_id, inv[0])
        assert names == ["sensitivity", f"α_s = {_g(inv[0]['params']['alpha_s'])}", "smooth L2"]
        assert top == [1, 2]
        _, top = _chain(by_id, inv[2])
        assert by_id[top[0]]["params"]["mesh_type"] == "octree"
        assert inv[2]["params"]["backend"] == "local" and inv[0]["params"]["backend"] == "ec2"
        assert len({n["id"] for n in wf["nodes"]}) == len(wf["nodes"])

    def test_runs_branch_by_regularization_then_settings(self, tmp_path):
        _, zpath, _ = _run(tmp_path)
        base = load_result(zpath)

        def variant(name, **settings):
            run = dict(base)
            run["settings"] = {**base["settings"], "regularization_type": "sparse", "alpha_s": 1.0,
                               "norms": [0, 2, 2, 2], "depth_weighting": "depth", **settings}
            run["_name"] = name
            return run

        runs = [variant("b05", depth_weighting_exponent=0.5), variant("b1", depth_weighting_exponent=1.0),
                variant("orig", alpha_s=1e-4, depth_weighting="sensitivity", norms=[0, 2, 2, 1]),
                variant("l1l2", regularization_type="l1l2", l1_ratio=0.8, depth_weighting="sensitivity")]
        wf = build_workflow(runs)
        by_id = {n["id"]: n for n in wf["nodes"]}
        leaf = {n["params"]["task"]: n for n in wf["nodes"] if n["type"] == "RegularizedInversionNode"}
        # one column per setting, every run through every column that applies to it:
        # regularization, α_s (L1 share for L1–L2), depth weighting, norms (sparse only)
        assert _chain(by_id, leaf["b05"]) == (["p = [0, 2, 2, 2]", "depth, β = 0.5", "α_s = 1", "sparse (IRLS)"], [1, 2])
        assert _chain(by_id, leaf["orig"]) == (["p = [0, 2, 2, 1]", "sensitivity", "α_s = 1e-4", "sparse (IRLS)"], [1, 2])
        assert _chain(by_id, leaf["l1l2"]) == (["sensitivity", "L1 share 0.8", "L1–L2"], [1, 2])
        # runs sharing a setting share its node: both α_s = 1 runs, all three sparse runs
        up = lambda n, k: n if k == 0 else up(by_id[n["inputs"][0]], k - 1)   # noqa: E731
        a1 = up(leaf["b05"], 3)
        assert up(leaf["b1"], 3) is a1 and a1["branch"]["n_runs"] == 2
        sparse = up(leaf["orig"], 4)
        assert up(a1, 1) is sparse and sparse["type"] == "RegularizationBranchNode"
        assert sparse["branch"]["n_runs"] == 3 and a1["type"] == "ParameterBranchNode"
        # a result card names the whole combination below the regularization
        assert leaf["b05"]["name"] == "α_s=1 · β=0.5 · p=[0,2,2,2]"
        assert leaf["l1l2"]["name"] == "L1 share 0.8 · sensitivity"
        assert leaf["b05"]["branch"]["path"] == ["sparse (IRLS)", "α_s = 1", "depth, β = 0.5", "p = [0, 2, 2, 2]"]
        # each setting has a title and a column of its own in the tree: the L1–L2 run, which
        # has no norms, skips that column instead of moving another setting into it
        title = lambda n: (n["branch"]["title"], n["branch"]["column"])   # noqa: E731
        assert title(sparse) == ("Regularization", 0) and title(a1) == ("α_s", 1)
        assert title(up(leaf["b05"], 2)) == ("Depth weighting", 2) and title(up(leaf["b05"], 1)) == ("Norms p", 3)
        assert title(up(leaf["l1l2"], 2)) == ("L1 share", 1) and title(up(leaf["l1l2"], 1)) == ("Depth weighting", 2)
        # depth-first order for the viewer: sparse (α_s = 1e-4, then α_s = 1: β 0.5, β 1), L1–L2
        seq = [sparse, leaf["orig"], a1, leaf["b05"], leaf["b1"], up(leaf["l1l2"], 3), leaf["l1l2"]]
        assert [n["order"] for n in seq] == sorted(n["order"] for n in seq)
        assert len({n["id"] for n in wf["nodes"]}) == len(wf["nodes"])
        # the viewer page keeps the tree
        from geoinv3d.viz.serve_dag import _compact_workflow
        compact = {n["id"]: n for n in _compact_workflow(wf)["nodes"]}
        assert compact[a1["id"]]["branch"] == a1["branch"] and compact[a1["id"]]["order"] == a1["order"]
        json.dumps(wf)

    def test_short_setting_labels(self):
        assert [_g(v) for v in (1e-4, 2.5e-4, 0.01, 0.5, 1.0, 2)] == ["1e-4", "2.5e-4", "0.01", "0.5", "1", "2"]

    def test_old_results_without_data_are_refused_clearly(self, tmp_path):
        result, zpath, _ = _run(tmp_path)
        run = load_result(zpath)
        run.pop("_data")
        with pytest.raises(ValueError, match="data.npz"):
            build_workflow([run])


class TestWorkflowEndpoint:
    def test_view_in_dag(self, tmp_path, monkeypatch):
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient
        from geoinv3d.api import server
        _, zpath, _ = _run(tmp_path)

        class Backend:
            def fetch_result(self, record, output_dir):
                return str(zpath)

        store = server.JobStore(tmp_path / "jobs.json")
        store.put({"job_id": "i-1", "task_id": "t1", "note": "dense test", "status": "SUCCEEDED",
                   "summary": {}})
        monkeypatch.setattr(server, "_store", store)
        monkeypatch.setattr(server, "_backend", Backend())
        wf = TestClient(server.app).get("/api/inversion/i-1/workflow").json()
        inv = wf["nodes"][-1]
        assert inv["type"] == "RegularizedInversionNode" and "dense test" in inv["name"]
        assert "model_3d" in inv["output"] and "data_fit" in inv["output"]


def test_a_job_of_the_api_carries_its_id_and_how_it_converged(tmp_path):
    result, zpath, _ = _run(tmp_path)
    assert result["convergence"]["status"] in ("converged", "at_limit", "not_converged")
    pack_result(result, str(zpath))
    run = load_result(zpath)
    run.update(_job_id="local-abc", _rerun=True, _backend="local")
    node = build_workflow([run])["nodes"][-1]
    assert node["job"] == {"id": "local-abc", "rerun": True, "backend": "local"}
    assert node["params"]["backend"] == "local"
    assert node["output"]["convergence"]["target"] == result["n_data"]
    plain = build_workflow([load_result(zpath)])["nodes"][-1]   # a result file: no job
    assert "job" not in plain


def test_a_stand_alone_viewer_keeps_the_map_layers(tmp_path):
    from geoinv3d.viz.serve_dag import generate_viewer
    layers = [{"name": "Mines", "kind": "points", "items": [{"x": 1.0, "y": 2.0, "label": "K"}]}]
    src = tmp_path / "w.geoinv3d.json"
    src.write_text(json.dumps({"version": 1, "nodes": [], "map_layers": layers}))
    html = open(generate_viewer(str(src)), encoding="utf-8").read()
    data = json.loads(html.split('<script id="embedded-data" type="application/json">')[1].split("</script>")[0])
    assert data["map_layers"] == layers
    # opened without the server: the logo travels inside the page
    assert 'src="assets/' not in html and 'src="data:image/png;base64,' in html


class TestViewerGrid:
    """The 3D tab's grid: from the ground down, and small enough for the browser."""

    @staticmethod
    def _meta(mesh, ext, h, dz, depth, active=None, topography=None):
        return {"mesh_design": {"used": {"core_cell_m": h, "core_cell_z_m": dz,
                                         "depth_core_m": depth, "pad_distance_m": 1000.0}},
                "cell_centers_x": [ext[0], ext[1]], "cell_centers_y": [ext[2], ext[3]],
                "_active": active, "topography": topography}

    def test_an_octree_starts_at_the_ground_not_the_top_of_its_box(self):
        pytest.importorskip("discretize")
        from geoinv3d.cloud.worker import _build_octree_mesh
        from geoinv3d.viz.result_workflow import ViewerGrid
        ext, flat = (0.0, 2000.0, 0.0, 2000.0), lambda x, y: np.zeros_like(np.asarray(x, float))
        mesh = _build_octree_mesh(ext, flat, 100.0, 50.0, 1000.0, 1000.0, [4, 4, 4]).to_discretize()
        assert mesh.cell_centers[:, 2].max() > 500           # the box reaches far into the air
        active = mesh.cell_centers[:, 2] < 0
        g = ViewerGrid(self._meta(mesh, ext, 100.0, 50.0, 1000.0, active), mesh)
        assert g.z_edges.max() == pytest.approx(0.0) and g.z_edges.min() == pytest.approx(-1000.0)
        assert g.factors == (1, 1) and g.shape[2] == 20

    def test_a_large_grid_is_coarsened(self, monkeypatch):
        pytest.importorskip("discretize")
        from discretize import TensorMesh
        import geoinv3d.viz.result_workflow as rw
        monkeypatch.setattr(rw, "MAX_VIEWER_CELLS", 1000)
        # a tensor core of 20 x 20 x 10 cells of 50 m x 25 m, 3 padding cells each side
        n_pad = 3
        from geoinv3d.cloud.meshing import padding_cells
        monkeypatch.setattr("geoinv3d.cloud.meshing.padding_cells", lambda h, p: n_pad)
        hx = [(50.0, n_pad, -1.3), (50.0, 20), (50.0, n_pad, 1.3)]
        hz = [(25.0, n_pad, -1.3), (25.0, 10)]
        mesh = TensorMesh([hx, hx, hz], origin="CC0")
        meta = self._meta(mesh, (-500, 500, -500, 500), 50.0, 25.0, 250.0)
        g = rw.ViewerGrid(meta, mesh)
        assert np.prod(g.shape) <= 1000 and g.factors != (1, 1)
        k, kz = g.factors
        assert g.shape == (int(np.ceil(20 / k)), int(np.ceil(20 / k)), int(np.ceil(10 / kz)))
        model = np.arange(mesh.n_cells, dtype=float)
        v = g.values(model)
        assert v.size == np.prod(g.shape) and np.isfinite(v).all()
        # each value is the mean of its block of core cells
        core = model.reshape(mesh.shape_cells, order="F")[n_pad:-n_pad, n_pad:-n_pad, n_pad:]
        top = core[:k, :k, ::-1][:, :, :kz].mean()
        assert v[0] == pytest.approx(top)


def test_a_posterior_hangs_after_its_run(tmp_path):
    """A Bayesian posterior (``_posterior_of`` a run of the workflow) is a node after that
    run's, with no mesh, survey or branch nodes of its own."""
    _, zpath, _ = _run(tmp_path)
    base, other, post = (load_result(zpath) for _ in range(3))
    base["_job_id"], other["_job_id"], post["_job_id"] = "j-1", "j-2", "j-post"
    other["settings"] = {**other.get("settings", {}), "alpha_s": 0.5}
    post["_posterior_of"] = "j-1"
    post["bayes"] = {"n_samples": 30, "beta": 1e-3, "chi2_per_datum": 1.0}
    post["_prob_body"] = np.full(len(post["_model"]), 0.5)
    wf = build_workflow([base, other, post])
    kinds = [n["type"] for n in wf["nodes"]]
    assert kinds.count("MeshCreateNode") == 1 and kinds.count("SurveyCreateNode") == 1
    assert kinds.count("RegularizedInversionNode") == 2
    (p,) = [n for n in wf["nodes"] if n["type"] == "BayesianPosteriorNode"]
    parent = next(n for n in wf["nodes"] if n.get("job", {}).get("id") == "j-1")
    assert p["inputs"] == [parent["id"]] and p["params"]["posterior_of"] == "j-1"
    assert p["output"]["model_3d"]["prob_values"] and p["job"]["id"] == "j-post"
    assert len({n["id"] for n in wf["nodes"]}) == len(wf["nodes"])          # ids unique
    # without its run in the workflow it is a run of its own
    alone = build_workflow([other, post])
    assert not [n for n in alone["nodes"] if n["type"] == "BayesianPosteriorNode"]
