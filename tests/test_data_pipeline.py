"""Tests for the raw-data pipeline (cloud worker 'data' mode), run locally.

Synthetic surveys are forward-modelled with SimPEG on a small mesh and
written to the file formats the upload page accepts, then passed through
run_data_pipeline exactly as the worker would after downloading from S3.
"""

import json
import zipfile

import numpy as np
import pytest

import geoinv3d.cloud.worker as worker
from geoinv3d.cloud.worker import pack_result, run_data_pipeline
from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.io.readers import (
    read_esri_ascii, read_npz_points, read_station_table, read_ubc_obs,
)
from geoinv3d.methods.gravity import GravityMethod
from geoinv3d.methods.joint import JointInversion, MethodSetup
from geoinv3d.methods.magnetics import MagneticsMethod

# Survey: 7 x 7 stations, 100 m spacing, over a buried block
XS = np.arange(0.0, 601.0, 100.0)
YS = np.arange(0.0, 601.0, 100.0)
INDUCING = [50000.0, 60.0, 10.0]

# Small, fast mesh settings for every pipeline run
SMALL_MESH = {
    "core_cell_m": 100.0,
    "core_cell_z_m": 100.0,
    "depth_core_m": 400.0,
    "pad_distance_m": 200.0,
}


def _station_grid(z=0.0):
    xx, yy = np.meshgrid(XS, YS)
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])


def _synthetic(method, locs):
    """Forward-model a block (200-400 m in x/y, 100-300 m deep) at `locs`."""
    mesh = Mesh3D.uniform(10, 10, 6, 100.0, 100.0, 100.0, origin=(-200.0, -200.0, -600.0))
    cc = mesh.to_discretize().cell_centers
    block = ((cc[:, 0] > 200) & (cc[:, 0] < 400) & (cc[:, 1] > 200) & (cc[:, 1] < 400)
             & (cc[:, 2] > -300) & (cc[:, 2] < -100))
    survey = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
    if method == "gravity":
        return GravityMethod().make_simulation(mesh, survey).dpred(block * 0.3)
    sim = MagneticsMethod(inducing_field=tuple(INDUCING)).make_simulation(mesh, survey)
    return sim.dpred(block * 0.02)


def _write_surfer_ascii(path, values):
    """Surfer ASCII grid (DSAA) on the XS x YS nodes; rows run south to north."""
    lines = ["DSAA", f"{len(XS)} {len(YS)}", f"{XS[0]} {XS[-1]}", f"{YS[0]} {YS[-1]}",
             f"{np.min(values)} {np.max(values)}"]
    for row in values.reshape(len(YS), len(XS)):
        lines.append(" ".join(f"{v:.8g}" for v in row))
    path.write_text("\n".join(lines) + "\n")


def _write_csv(path, locs, values, with_z=True, header=True):
    cols = [locs[:, 0], locs[:, 1]] + ([locs[:, 2]] if with_z else []) + [values]
    names = ["x", "y"] + (["z"] if with_z else []) + ["value"]
    rows = np.column_stack(cols)
    text = (",".join(names) + "\n") if header else ""
    text += "\n".join(",".join(f"{v:.8g}" for v in r) for r in rows)
    path.write_text(text + "\n")


def _dem_values(x, y):
    """Gently sloping terrain: 50 m in the west rising to 110 m in the east."""
    return 50.0 + 0.1 * np.asarray(x)


def _write_dem_asc(path):
    """ESRI ASCII DEM covering well beyond the survey."""
    x = np.arange(-400.0, 1001.0, 50.0)
    y = np.arange(-400.0, 1001.0, 50.0)
    xx, yy = np.meshgrid(x, y[::-1])  # rows north to south
    z = _dem_values(xx, yy)
    header = (f"ncols {len(x)}\nnrows {len(y)}\nxllcenter {x[0]}\nyllcenter {y[0]}\n"
              f"cellsize 50\nNODATA_value -9999\n")
    path.write_text(header + "\n".join(" ".join(f"{v:.3f}" for v in r) for r in z) + "\n")


@pytest.fixture
def capture(monkeypatch):
    """Replace execute_task so a test can inspect the task/mesh it receives."""
    seen = {}

    def fake_execute(task, mesh=None):
        seen["task"], seen["mesh"] = task, mesh
        return {"task_id": task.task_id, "n_iterations": 0, "iterations": []}

    monkeypatch.setattr(worker, "execute_task", fake_execute)
    return seen


@pytest.fixture
def grav_grid_dir(tmp_path):
    values = _synthetic("gravity", _station_grid(0.0))
    _write_surfer_ascii(tmp_path / "grav.grd", values)
    return tmp_path


def _single(method, files, **extra):
    ds = {"type": method, "method": method, "files": files,
          "noise_pct": 0.05, "noise_floor": 0.01}
    ds.update(extra.pop("dataset", {}))
    params = {
        "method_type": method, "inversion_mode": "single", "datasets": [ds],
        "data_file": files[0], "noise_pct": ds["noise_pct"],
        "noise_floor": ds["noise_floor"], "method_kwargs": ds.get("method_kwargs", {}),
        "topography": {"flat_elevation": 0}, "param_mode": "auto",
        "regularization_type": "sparse", "mesh_type": "tensor", **SMALL_MESH,
    }
    params.update(extra)
    return params


# ── Readers ─────────────────────────────────────────────────────────────

class TestReaders:
    def test_station_table_header_names(self, tmp_path):
        p = tmp_path / "s.csv"
        p.write_text("Easting,Northing,Elevation,err,gzz\n1,2,3,0.1,9\n4,5,6,0.2,10\n")
        pts = read_station_table(str(p), value_name="gzz")
        np.testing.assert_allclose(pts.locations, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(pts.values, [9, 10])

    def test_station_table_three_columns_has_no_z(self, tmp_path):
        p = tmp_path / "s.txt"
        p.write_text("100 200 1.5\n110 210 1.7\n")
        pts = read_station_table(str(p))
        assert np.isnan(pts.locations[:, 2]).all()
        np.testing.assert_allclose(pts.values, [1.5, 1.7])

    def test_geosoft_xyz_comment_header(self, tmp_path):
        p = tmp_path / "s.xyz"
        p.write_text("/ X Y TMI\nLine 10\n0 0 5\n10 0 6\nLine 20\n0 10 7\n")
        pts = read_station_table(str(p))
        assert len(pts.values) == 3
        np.testing.assert_allclose(pts.values, [5, 6, 7])
        assert np.isnan(pts.locations[:, 2]).all()

    def test_ubc_obs_gravity_and_magnetics(self, tmp_path):
        g = tmp_path / "g.obs"
        g.write_text("2\n0 0 5 0.1 0.01\n10 0 5 0.2 0.01\n")
        pts = read_ubc_obs(str(g))
        np.testing.assert_allclose(pts.locations[:, 2], [5, 5])
        np.testing.assert_allclose(pts.values, [0.1, 0.2])

        m = tmp_path / "m.obs"
        m.write_text("65 12 52000\n65 12 1\n1\n0 0 30 12.5 1\n")
        pts = read_ubc_obs(str(m))
        assert pts.metadata["inducing_field"] == (52000.0, 65.0, 12.0)
        np.testing.assert_allclose(pts.values, [12.5])

    def test_esri_ascii_orientation(self, tmp_path):
        p = tmp_path / "d.asc"
        p.write_text("ncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 10\n"
                     "NODATA_value -9999\n1 2\n3 -9999\n")
        grid = read_esri_ascii(str(p))
        np.testing.assert_allclose(grid.x, [5, 15])
        np.testing.assert_allclose(grid.y, [15, 5])  # north row first
        assert grid.values[0, 1] == 2 and np.isnan(grid.values[1, 1])

    def test_npz_points(self, tmp_path):
        p = tmp_path / "d.npz"
        np.savez(p, x=[0.0, 1.0], y=[2.0, 3.0], values=[7.0, 8.0])
        pts = read_npz_points(str(p))
        assert np.isnan(pts.locations[:, 2]).all()
        np.testing.assert_allclose(pts.values, [7, 8])


# ── Pipeline plumbing (inversion replaced by a spy) ─────────────────────

class TestPipelinePlumbing:
    def test_legacy_params_keep_auto_defaults(self, grav_grid_dir, capture):
        """Old-style params (no datasets/topography) behave as before."""
        params = {"method_type": "gravity", "data_file": "grav.grd", **SMALL_MESH}
        result = run_data_pipeline(params, str(grav_grid_dir))
        task, mesh = capture["task"], capture["mesh"]

        assert task.method_type == "gravity"
        assert task.method_kwargs == {"component": "gz"}
        assert task.regularization_type == "sparse"
        assert (task.alpha_s, task.alpha_x, task.alpha_y, task.alpha_z) == (1e-4, 1, 1, 1)
        assert task.norms == (0.0, 2.0, 2.0, 1.0)
        assert (task.max_iter, task.beta0_ratio, task.cooling_factor) == (30, 1.0, 2.0)
        assert task.max_irls_iterations == 30 and task.use_preconditioner
        assert task.active_cells is None
        np.testing.assert_allclose(task.station_locations[:, 2], 0.0)
        np.testing.assert_allclose(task.data_std, 0.05 * np.abs(task.observed_data) + 0.5)
        # Mesh: 6 core cells + 3 padding each side; top of mesh at z = 0
        assert mesh.shape == (12, 12, 7)
        assert np.isclose(mesh.origin[2] + mesh.hz.sum(), 0.0)
        assert result["mesh_type"] == "tensor"

    def test_auto_mode_ignores_manual_keys(self, grav_grid_dir, capture):
        params = _single("gravity", ["grav.grd"], alpha_s=5.0, norms=[2, 2, 2, 2])
        run_data_pipeline(params, str(grav_grid_dir))
        assert capture["task"].alpha_s == 1e-4
        assert capture["task"].norms == (0.0, 2.0, 2.0, 1.0)

    def test_auto_mode_iteration_limits_are_editable(self, grav_grid_dir, capture):
        params = _single("gravity", ["grav.grd"], max_iter=12, max_irls_iterations=7)
        run_data_pipeline(params, str(grav_grid_dir))
        assert (capture["task"].max_iter, capture["task"].max_irls_iterations) == (12, 7)

    def test_manual_parameters_forwarded(self, grav_grid_dir, capture):
        params = _single(
            "gravity", ["grav.grd"], param_mode="manual", regularization_type="l2",
            norms=[2, 2, 2, 2], alpha_s=0.01, alpha_x=2.0, alpha_y=3.0, alpha_z=4.0,
            beta0_ratio=10.0, cooling_factor=4.0, max_iter=7, max_irls_iterations=3,
            use_preconditioner=False, bounds_lower=-0.5, bounds_upper=0.8,
        )
        run_data_pipeline(params, str(grav_grid_dir))
        t = capture["task"]
        assert t.regularization_type == "l2"
        assert t.norms == (2.0, 2.0, 2.0, 2.0)
        assert (t.alpha_s, t.alpha_x, t.alpha_y, t.alpha_z) == (0.01, 2.0, 3.0, 4.0)
        assert (t.beta0_ratio, t.cooling_factor, t.max_iter) == (10.0, 4.0, 7)
        assert t.max_irls_iterations == 3 and t.use_preconditioner is False
        assert (t.bounds_lower, t.bounds_upper) == (-0.5, 0.8)

    def test_point_files_concatenated_with_dataset_noise(self, tmp_path, capture):
        locs = _station_grid(0.0)
        values = _synthetic("gravity", locs)
        half = len(locs) // 2
        _write_csv(tmp_path / "a.csv", locs[:half], values[:half], with_z=False)
        (tmp_path / "b.xyz").write_text(
            "/ X Y GZ\n" + "\n".join(f"{x} {y} {v}" for (x, y, _), v in
                                      zip(locs[half:], values[half:])) + "\n")
        params = _single("gravity", ["a.csv", "b.xyz"],
                         dataset={"noise_pct": 0.1, "noise_floor": 0.02})
        result = run_data_pipeline(params, str(tmp_path))
        t = capture["task"]
        assert len(t.observed_data) == len(locs)
        np.testing.assert_allclose(t.data_std, 0.1 * np.abs(t.observed_data) + 0.02)
        assert result["datasets"][0]["files"] == ["a.csv", "b.xyz"]

    def test_dem_drapes_stations_and_masks_air_cells(self, grav_grid_dir, capture):
        _write_dem_asc(grav_grid_dir / "dem.asc")
        params = _single("gravity", ["grav.grd"], topography={"file": "dem.asc"},
                         station_height=2.0)
        result = run_data_pipeline(params, str(grav_grid_dir))
        t, mesh = capture["task"], capture["mesh"]

        locs = t.station_locations
        np.testing.assert_allclose(locs[:, 2], _dem_values(locs[:, 0], locs[:, 1]) + 2.0,
                                   atol=1e-6)
        cc = mesh.to_discretize().cell_centers
        ground = _dem_values(cc[:, 0], cc[:, 1])
        np.testing.assert_array_equal(t.active_cells, cc[:, 2] < ground)
        assert 0 < t.active_cells.sum() < mesh.n_cells
        top = mesh.origin[2] + mesh.hz.sum()
        assert top >= _dem_values(XS[-1], 0) - 1e-6  # mesh reaches the highest ground
        assert result["topography"] == {"source": "dem", "file": "dem.asc"}
        assert result["n_active_cells"] == int(t.active_cells.sum())

    def test_point_dem_and_station_elevations_kept(self, tmp_path, capture):
        locs = _station_grid(0.0)
        locs[:, 2] = 500.0  # e.g. airborne survey with its own heights
        grav = _synthetic("gravity", locs)
        obs = tmp_path / "grav.obs"
        obs.write_text(f"{len(locs)}\n" + "\n".join(
            f"{x} {y} {z} {v} 0.01" for (x, y, z), v in zip(locs, grav)) + "\n")
        dem = tmp_path / "topo.xyz"
        gx, gy = np.meshgrid(np.arange(-400, 1001, 100.0), np.arange(-400, 1001, 100.0))
        dem.write_text("\n".join(f"{x} {y} {_dem_values(x, y)}"
                                 for x, y in zip(gx.ravel(), gy.ravel())) + "\n")
        params = _single("gravity", ["grav.obs"], topography={"file": "topo.xyz"})
        run_data_pipeline(params, str(tmp_path))
        t = capture["task"]
        np.testing.assert_allclose(t.station_locations[:, 2], 500.0)
        assert t.active_cells is not None and 0 < t.active_cells.sum()

    def test_flat_elevation_sets_station_and_mesh_top(self, grav_grid_dir, capture):
        params = _single("gravity", ["grav.grd"], topography={"flat_elevation": 120.0})
        run_data_pipeline(params, str(grav_grid_dir))
        t, mesh = capture["task"], capture["mesh"]
        np.testing.assert_allclose(t.station_locations[:, 2], 120.0)
        assert np.isclose(mesh.origin[2] + mesh.hz.sum(), 120.0)
        assert t.active_cells is None

    def test_mesh_spans_union_of_survey_extents(self, tmp_path, capture):
        g_locs = _station_grid(0.0)
        m_locs = g_locs + np.array([900.0, 300.0, 0.0])  # offset magnetic survey
        _write_csv(tmp_path / "g.csv", g_locs, _synthetic("gravity", g_locs))
        _write_csv(tmp_path / "m.csv", m_locs, _synthetic("magnetics", m_locs))
        params = _joint_params(["g.csv"], ["m.csv"])
        result = run_data_pipeline(params, str(tmp_path))
        mesh = capture["mesh"]
        # 1500 m x 900 m union at 100 m cells, plus 3 padding cells each side
        assert mesh.shape[:2] == (15 + 6, 9 + 6)
        assert result["cell_centers_x"] == (0.0, 1500.0)
        assert result["cell_centers_y"] == (0.0, 900.0)

    def test_joint_routing_weights_and_components(self, tmp_path, capture):
        locs = _station_grid(0.0)
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        _write_csv(tmp_path / "m.csv", locs, _synthetic("magnetics", locs))
        params = _joint_params(["g.csv"], ["m.csv"], param_mode="manual",
                               joint_weights=[1.0, 0.25], cross_gradient_weight=3.0)
        run_data_pipeline(params, str(tmp_path))
        t = capture["task"]
        assert t.joint_methods == ["gravity", "magnetics"]
        assert t.joint_weights == [1.0, 0.25]
        assert t.cross_gradient_weight == 3.0
        assert t.joint_kwargs_list[0] == {"component": "gz"}
        assert t.joint_kwargs_list[1] == {"inducing_field": INDUCING, "component": "tmi"}
        assert [len(s["observed"]) for s in t.joint_surveys] == [len(locs)] * 2

    def test_joint_auto_defaults(self, tmp_path, capture):
        locs = _station_grid(0.0)
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        _write_csv(tmp_path / "m.csv", locs, _synthetic("magnetics", locs))
        # auto mode must ignore these even if present
        params = _joint_params(["g.csv"], ["m.csv"], joint_weights=[9, 9],
                               cross_gradient_weight=9)
        run_data_pipeline(params, str(tmp_path))
        t = capture["task"]
        assert t.joint_weights == [1.0, 1.0]
        assert t.cross_gradient_weight == 1.0

    def test_octree_mesh_type(self, grav_grid_dir, capture):
        params = _single("gravity", ["grav.grd"], mesh_type="octree")
        result = run_data_pipeline(params, str(grav_grid_dir))
        t, mesh = capture["task"], capture["mesh"]
        from discretize import TreeMesh
        assert isinstance(mesh.to_discretize(), TreeMesh)
        # Octree extends above the ground, so air cells are masked
        cc = mesh.to_discretize().cell_centers
        np.testing.assert_array_equal(t.active_cells, cc[:, 2] < 0.0)
        assert result["mesh_type"] == "octree"
        assert result["mesh"]["__class__"] == "TreeMesh"

    def test_l1l2_parameters_forwarded(self, grav_grid_dir, capture):
        params = _single("gravity", ["grav.grd"], param_mode="manual",
                         regularization_type="l1l2", l1_ratio=0.7, max_iter=9)
        result = run_data_pipeline(params, str(grav_grid_dir))
        t = capture["task"]
        assert t.regularization_type == "l1l2"
        assert t.l1_ratio == 0.7 and t.max_iter == 9
        assert result["regularization_type"] == "l1l2"
        assert any("L1–L2" in n for n in result["notes"])

    @pytest.mark.parametrize("extra", [
        dict(regularization_type="l1l2", l1_ratio=0.8, l1l2_solver="cda",
             l1l2_weighting="S1", lambda_decades=3.5, beta_selection="auto"),
        dict(regularization_type="mgs", focusing_percentile=90, focusing_scale=0.5,
             bounds_lower=0.0, bounds_upper=0.04, beta_selection="lcurve"),
        dict(regularization_type="tv", focusing_percentile=95, beta_selection="gcv"),
    ])
    def test_upload_page_choices_forwarded(self, grav_grid_dir, capture, extra):
        """Every regularization / beta choice the upload page sends reaches the task."""
        params = _single("gravity", ["grav.grd"], param_mode="manual", **extra)
        run_data_pipeline(params, str(grav_grid_dir))
        t = capture["task"]
        for key, value in extra.items():
            assert getattr(t, key) == value, key

    def test_auto_mode_ignores_l1_ratio(self, grav_grid_dir, capture):
        params = _single("gravity", ["grav.grd"], l1_ratio=0.9)
        run_data_pipeline(params, str(grav_grid_dir))
        assert capture["task"].l1_ratio == 0.5
        assert capture["task"].regularization_type == "sparse"

    def test_joint_with_l1l2_is_reported(self, tmp_path, capture):
        locs = _station_grid(0.0)
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        _write_csv(tmp_path / "m.csv", locs, _synthetic("magnetics", locs))
        params = _joint_params(["g.csv"], ["m.csv"], param_mode="manual",
                               regularization_type="l1l2", l1_ratio=0.5)
        result = run_data_pipeline(params, str(tmp_path))
        assert any("L1–L2 regularization is not applied" in n for n in result["notes"])

    def test_mesh_recommended_from_data_spacing(self, grav_grid_dir, capture):
        """Without mesh settings the mesh follows the data (100 m grid, 600 m wide)."""
        params = _single("gravity", ["grav.grd"])
        for key in SMALL_MESH:
            params.pop(key)
        result = run_data_pipeline(params, str(grav_grid_dir))
        design = result["mesh_design"]
        assert design["source"] == "auto"
        assert design["data_spacing"] == [
            {"method": "gravity", "spacing_m": 100.0, "kind": "grid"}]
        assert design["used"] == {"core_cell_m": 100.0, "core_cell_z_m": 50.0,
                                  "depth_core_m": 300.0, "pad_distance_m": 250.0}
        assert list(capture["mesh"].shape) == design["recommended"]["shape"]

    def test_mesh_user_and_mixed_settings(self, grav_grid_dir, capture):
        user = run_data_pipeline(_single("gravity", ["grav.grd"]), str(grav_grid_dir))
        assert user["mesh_design"]["source"] == "user"
        assert user["mesh_design"]["used"]["core_cell_m"] == SMALL_MESH["core_cell_m"]

        params = _single("gravity", ["grav.grd"], mesh_design={"note": "from the page"})
        for key in ("core_cell_z_m", "depth_core_m", "pad_distance_m"):
            params.pop(key)
        mixed = run_data_pipeline(params, str(grav_grid_dir))["mesh_design"]
        assert mixed["source"] == "mixed"
        assert mixed["used"]["core_cell_m"] == 100.0        # given
        assert mixed["used"]["depth_core_m"] == 300.0       # recommended
        assert mixed["client"] == {"note": "from the page"}

    def test_point_data_spacing(self, tmp_path, capture):
        locs = _station_grid(0.0)
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        params = _single("gravity", ["g.csv"])
        for key in SMALL_MESH:
            params.pop(key)
        design = run_data_pipeline(params, str(tmp_path))["mesh_design"]
        assert design["data_spacing"][0]["kind"] == "points"
        assert design["data_spacing"][0]["spacing_m"] == pytest.approx(100.0)

    def test_topography_file_not_used_as_data(self, grav_grid_dir, capture):
        """Legacy auto-detect must skip the DEM file."""
        _write_dem_asc(grav_grid_dir / "a_dem.asc")
        params = {"method_type": "gravity", "topography": {"file": "a_dem.asc"},
                  **SMALL_MESH}
        result = run_data_pipeline(params, str(grav_grid_dir))
        assert result["datasets"][0]["files"] == ["grav.grd"]


class TestPipelineErrors:
    def test_dc_is_rejected_clearly(self, grav_grid_dir):
        params = _single("dc", ["grav.grd"])
        with pytest.raises(NotImplementedError, match="gravity, magnetics"):
            run_data_pipeline(params, str(grav_grid_dir))

    def test_missing_file(self, grav_grid_dir):
        with pytest.raises(FileNotFoundError, match="nope.csv"):
            run_data_pipeline(_single("gravity", ["nope.csv"]), str(grav_grid_dir))

    def test_zero_uncertainty(self, tmp_path):
        _write_csv(tmp_path / "g.csv", _station_grid(), np.zeros(len(XS) * len(YS)))
        params = _single("gravity", ["g.csv"], dataset={"noise_pct": 0.05, "noise_floor": 0})
        with pytest.raises(ValueError, match="noise floor"):
            run_data_pipeline(params, str(tmp_path))

    def test_joint_weight_count(self, tmp_path):
        locs = _station_grid()
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        _write_csv(tmp_path / "m.csv", locs, _synthetic("magnetics", locs))
        params = _joint_params(["g.csv"], ["m.csv"], param_mode="manual",
                               joint_weights=[1.0])
        with pytest.raises(ValueError, match="joint_weights"):
            run_data_pipeline(params, str(tmp_path))


def _joint_params(grav_files, mag_files, **extra):
    datasets = [
        {"type": "gravity", "method": "gravity", "files": grav_files,
         "noise_pct": 0.05, "noise_floor": 0.01, "component": "gz"},
        {"type": "magnetic", "method": "magnetic", "files": mag_files,
         "noise_pct": 0.05, "noise_floor": 1.0, "component": "tmi",
         "method_kwargs": {"inducing_field": INDUCING}},
    ]
    params = {
        "method_type": "joint", "inversion_mode": "joint", "datasets": datasets,
        "joint_methods": ["gravity", "magnetic"],
        "joint_kwargs_list": [{}, {"inducing_field": INDUCING}],
        "topography": {"flat_elevation": 0}, "param_mode": "auto",
        "regularization_type": "sparse", "mesh_type": "tensor", **SMALL_MESH,
    }
    params.update(extra)
    return params


# ── End-to-end runs through SimPEG ──────────────────────────────────────

class TestPipelineEndToEnd:
    def test_gravity_sparse_with_dem(self, grav_grid_dir, tmp_path):
        _write_dem_asc(grav_grid_dir / "dem.asc")
        params = _single("gravity", ["grav.grd"], topography={"file": "dem.asc"},
                         param_mode="manual", norms=[0, 2, 2, 1], alpha_s=1e-4,
                         alpha_x=1, alpha_y=1, alpha_z=1, beta0_ratio=1.0,
                         cooling_factor=2.0, max_iter=6, max_irls_iterations=2,
                         use_preconditioner=True)
        result = run_data_pipeline(params, str(grav_grid_dir))

        assert result["regularization"] == "sparse_IRLS"
        assert result["n_iterations"] > 0
        n_active = result["n_active_cells"]
        assert result["recovered_model"].shape == (n_active,)
        assert np.all(np.isfinite(result["recovered_model"]))
        phi_d = [it["phi_d"] for it in result["iterations"]]
        assert phi_d[-1] < phi_d[0]

        # The packed result carries the active-cell mask next to the model
        path = pack_result(result, str(tmp_path / "result.zip"))
        with zipfile.ZipFile(path) as zf:
            assert {"result.json", "recovered_model.npy", "active_cells.npy"} <= set(
                zf.namelist())
            meta = json.loads(zf.read("result.json"))
        assert meta["topography"]["source"] == "dem"
        assert meta["mesh"]["__class__"] == "TensorMesh"

    def test_magnetic_bz_l2_on_octree(self, tmp_path):
        locs = _station_grid(0.0)
        from discretize import TensorMesh  # noqa: F401  (ensures discretize present)
        mesh = Mesh3D.uniform(10, 10, 6, 100.0, 100.0, 100.0,
                              origin=(-200.0, -200.0, -600.0))
        cc = mesh.to_discretize().cell_centers
        block = ((cc[:, 0] > 200) & (cc[:, 0] < 400) & (cc[:, 1] > 200)
                 & (cc[:, 1] < 400) & (cc[:, 2] > -300) & (cc[:, 2] < -100))
        survey = SurveyData(locations=locs, observed=np.zeros(len(locs)),
                            std=np.ones(len(locs)))
        bz = MagneticsMethod(tuple(INDUCING), component="bz") \
            .make_simulation(mesh, survey).dpred(block * 0.02)
        np.savez(tmp_path / "bz.npz", locations=locs[:, :2], values=bz)

        params = _single(
            "magnetic", ["bz.npz"], mesh_type="octree", param_mode="manual",
            regularization_type="l2", norms=[2, 2, 2, 2], alpha_s=1e-4, alpha_x=1,
            alpha_y=1, alpha_z=1, beta0_ratio=1.0, cooling_factor=2.0, max_iter=4,
            max_irls_iterations=1, use_preconditioner=False,
            dataset={"component": "bz", "noise_floor": 1.0,
                     "method_kwargs": {"inducing_field": INDUCING}},
        )
        result = run_data_pipeline(params, str(tmp_path))
        assert result["method"] == "magnetics"
        assert result["regularization"] == "smooth_L2"
        assert result["mesh_type"] == "octree"
        assert result["recovered_model"].shape == (result["n_active_cells"],)
        assert result["datasets"][0]["component"] == "bz"

    def test_joint_gravity_magnetics_with_cross_gradient(self, tmp_path, monkeypatch):
        locs = _station_grid(0.0)
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        _write_csv(tmp_path / "m.csv", locs, _synthetic("magnetics", locs))
        _write_dem_asc(tmp_path / "dem.asc")

        built = []
        original_init = JointInversion.__init__

        def spy_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            built.append(self)

        monkeypatch.setattr(JointInversion, "__init__", spy_init)
        params = _joint_params(
            ["g.csv"], ["m.csv"], topography={"file": "dem.asc"}, param_mode="manual",
            norms=[2, 2, 2, 2], regularization_type="l2", alpha_s=1e-3, alpha_x=1,
            alpha_y=1, alpha_z=1, beta0_ratio=1.0, cooling_factor=2.0, max_iter=3,
            max_irls_iterations=1, use_preconditioner=False,
            joint_weights=[1.0, 0.5], cross_gradient_weight=2.0,
        )
        result = run_data_pipeline(params, str(tmp_path))

        joint = built[-1]
        assert joint.cross_gradient_weight == 2.0
        assert [s.weight for s in joint.setups] == [1.0, 0.5]
        assert joint.reg_kwargs["alpha_s"] == 1e-3
        assert joint.setups[0].active_cells is not None
        assert joint.setups[1].method.inducing_field == INDUCING

        n_active = result["n_active_cells"]
        assert set(result["recovered_models"]) == {"gravity", "magnetics"}
        for model in result["recovered_models"].values():
            assert model.shape == (n_active,)
            assert np.all(np.isfinite(model))
        assert result["methods"] == ["gravity", "magnetics"]
        # Only the padding diagnostic may speak up (joint L2 on this tiny mesh)
        assert set(result["outside_core_share"]) == {"gravity", "magnetics"}
        assert all("outside the core" in n for n in result.get("notes", []))


class TestElasticNetEndToEnd:
    """L1–L2 (Utsugi 2019) through the full pipeline on synthetic TMI data."""

    @staticmethod
    def _run(tmp_path, l1_ratio, **extra):
        locs = _station_grid(0.0)
        tmi = _synthetic("magnetics", locs)
        tmi = tmi + np.random.default_rng(0).normal(scale=1.0, size=tmi.size)
        np.savez(tmp_path / "tmi.npz", locations=locs, values=tmi)
        params = _single(
            "magnetic", ["tmi.npz"], param_mode="manual",
            regularization_type=extra.pop("regularization_type", "l1l2"), l1_ratio=l1_ratio, beta0_ratio=1.0, cooling_factor=2.0,
            max_iter=extra.pop("max_iter", 30),
            max_irls_iterations=extra.pop("max_irls_iterations", 10), use_preconditioner=True,
            dataset={"noise_pct": 0.0, "noise_floor": 1.0,
                     "method_kwargs": {"inducing_field": INDUCING}},
            **extra,
        )
        return run_data_pipeline(params, str(tmp_path))

    def test_l1_ratio_controls_compactness(self, tmp_path):
        smooth = self._run(tmp_path, 0.0, l1l2_solver="irls")
        sparse = self._run(tmp_path, 1.0, max_irls_iterations=20, l1l2_solver="irls")
        for r in (smooth, sparse):
            assert r["regularization"] == "elastic_net_IRLS"
            assert "norms" not in r
            final = r["iterations"][-1]["phi_d"]
            assert abs(final / r["n_data"] - 1) < 0.2  # fits to about chi^2 = N
        assert (smooth["l1_ratio"], sparse["l1_ratio"]) == (0.0, 1.0)

        def n_significant(m):
            return int(np.sum(m > 0.1 * m.max()))

        m_smooth, m_sparse = smooth["recovered_model"], sparse["recovered_model"]
        # L1 concentrates the anomaly in fewer cells with a higher peak
        assert n_significant(m_sparse) < 0.5 * n_significant(m_smooth)
        assert m_sparse.max() > m_smooth.max()

    @pytest.mark.parametrize("regularization_type", ["l1l2", "sparse", "mgs", "tv"])
    def test_positivity_bound(self, tmp_path, regularization_type):
        """Bounds with m0 on the lower bound used to leave the model at zero."""
        result = self._run(tmp_path, 0.5, bounds_lower=0.0, bounds_upper=1.0,
                           regularization_type=regularization_type)
        m = result["recovered_model"]
        assert m.min() >= 0.0
        assert m.max() > 1e-3
        assert result["iterations"][-1]["phi_d"] < 0.1 * result["iterations"][0]["phi_d"]


class TestSmoothL2EndToEnd:
    """The upload page's "Smooth L2" option (regularization_type="l2")."""

    @staticmethod
    def _core_centroid_depth(result):
        """Depth of the centroid of the positive anomaly inside the core mesh."""
        import discretize
        mesh = getattr(discretize, result["mesh"]["__class__"]).deserialize(result["mesh"])
        cc, m = mesh.cell_centers, result["recovered_model"]
        core = ((cc[:, 0] >= 0) & (cc[:, 0] <= 600) & (cc[:, 1] >= 0) & (cc[:, 1] <= 600)
                & (cc[:, 2] > -400))
        w = np.clip(m, 0.0, None) * core
        return float(np.sum(w * cc[:, 2]) / np.sum(w))

    def test_depth_weighting_lifts_the_surface_bias(self, tmp_path):
        run = TestElasticNetEndToEnd._run
        l2 = run(tmp_path, 0.5, regularization_type="l2")
        legacy = run(tmp_path, 0.5, regularization_type="smooth")
        assert l2["regularization"] == "smooth_L2"
        assert l2["depth_weighting"] == "sensitivity" and "norms" not in l2
        assert abs(l2["iterations"][-1]["phi_d"] / l2["n_data"] - 1) < 0.2
        # True block: 100-300 m deep (centroid -200 m).  Without depth weighting
        # the legacy path keeps the anomaly shallow; depth weighting moves it
        # clearly deeper.  (On this shallow 4-layer core it overshoots into the
        # bottom padding, as the sparse path does: SimPEG's volume-normalized
        # sensitivity weights make large padding cells cheap.)
        assert self._core_centroid_depth(l2) < self._core_centroid_depth(legacy) - 50
        # ...and the result says so
        assert l2["outside_core_share"]["magnetics"] > 0.5
        assert any("outside the core" in n for n in l2["notes"])

    def test_bounds(self, tmp_path):
        result = TestElasticNetEndToEnd._run(
            tmp_path, 0.5, regularization_type="l2", bounds_lower=0.0, bounds_upper=1.0)
        assert result["recovered_model"].min() >= 0.0
        assert result["recovered_model"].max() > 1e-3


class TestOutsideCoreShare:
    def test_share_counts_moment_outside_the_core(self):
        from geoinv3d.cloud.worker import _outside_core_share
        dmesh = Mesh3D(hx=[200, 100, 100, 200], hy=[200, 100, 100, 200], hz=[300, 100, 100],
                       origin=(-200.0, -200.0, -500.0)).to_discretize()
        cc = dmesh.cell_centers
        core = (abs(cc[:, 0] - 100) < 100) & (abs(cc[:, 1] - 100) < 100) & (cc[:, 2] > -200)
        extent, margin, z_bottom = (0.0, 200.0, 0.0, 200.0), 0.0, -200.0
        assert _outside_core_share(dmesh, None, core * 1.0, extent, margin, z_bottom) == 0.0
        assert _outside_core_share(dmesh, None, (~core) * 1.0, extent, margin, z_bottom) == 1.0
        # weighted by cell volume: one 200x200x300 padding cell vs one 100^3 core cell
        m = np.zeros(dmesh.n_cells)
        m[np.flatnonzero(core)[0]] = 1.0
        m[0] = 1.0  # corner padding cell, 12x the volume
        share = _outside_core_share(dmesh, None, m, extent, margin, z_bottom)
        assert np.isclose(share, 12 / 13)


class TestMethodFixes:
    def test_active_cell_simulation(self):
        """make_simulation_active works with SimPEG >= 0.24 (no valInactive)."""
        mesh = Mesh3D.uniform(6, 6, 4, 100.0, 100.0, 100.0, origin=(0.0, 0.0, -400.0))
        active = mesh.to_discretize().cell_centers[:, 2] < -150
        locs = _station_grid(1.0)
        survey = SurveyData(locations=locs, observed=np.zeros(len(locs)),
                            std=np.ones(len(locs)))
        for method in (GravityMethod("gzz"), MagneticsMethod(component="bz")):
            sim = method.make_simulation_active(mesh, survey, active)
            d = sim.dpred(np.full(int(active.sum()), 0.1))
            assert d.shape == (len(locs),) and np.all(np.isfinite(d))

    def test_cross_gradient_three_methods(self):
        """Pairwise cross-gradient terms act on the full joint model."""
        mesh = Mesh3D.uniform(4, 4, 3, 100.0, 100.0, 100.0, origin=(0.0, 0.0, -300.0))
        locs = np.array([[150.0, 150.0, 1.0], [250.0, 250.0, 1.0]])
        survey = SurveyData(locations=locs, observed=np.zeros(2), std=np.ones(2))
        setups = [MethodSetup(method=GravityMethod(), survey=survey, mesh=mesh,
                              initial_model=np.zeros(mesh.n_cells)) for _ in range(3)]
        comps = JointInversion(setups, cross_gradient_weight=1.0).build()
        pairs = comps["cross_grad_list"]
        assert len(pairs) == 3
        n = mesh.n_cells
        rng = np.random.default_rng(0)
        m = rng.normal(size=3 * n)
        assert comps["combo_reg"].deriv(m).shape == m.shape
        assert comps["combo_reg"].deriv2(m, m).shape == m.shape

        # The (0, 2) term equals a plain 2-model cross-gradient on [m0 | m2]
        from simpeg import maps
        from simpeg.regularization import CrossGradient
        direct = CrossGradient(mesh.to_discretize(), maps.Wires(("a", n), ("b", n)))
        assert np.isclose(pairs[1](m), direct(np.r_[m[:n], m[2 * n:]]))

        # Gradient matches finite differences
        dm = rng.normal(size=3 * n) * 1e-6
        fd = pairs[1](m + dm) - pairs[1](m - dm)
        assert np.isclose(fd, 2 * pairs[1].deriv(m) @ dm, rtol=1e-4)
