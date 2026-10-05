"""Geology-constrained reference models (methods/geology.py) and their use in inversions."""

import json
import struct
import zipfile

import numpy as np
import pytest

from geoinv3d.io.vector import read_dbf, read_shapefile, read_vector
from geoinv3d.methods.geology import (
    build_constraints, points_in_polygon, read_samples, units_from_samples,
)

SAMPLES_CSV = (
    "Latitude,Longitude,Density (g/cc),Rock_type,Susceptibility(X 10^(-6) CGS units),Toposheets\n"
    "15.10,76.60,3.38,Banded Iron Formation,50000,\"57A/1, A/2\"\n"
    "15.11,76.61,3.40,Banded Iron Formation,52000,\n"
    "15.12,76.62,2.96,Amphibolite,800,\n"
    "15.13,76.63,3.01,Dolerite,900,\n"
    "15.14,76.64,2.61,Grey Granite,20,\n"
    "15.15,76.65,2.65,Granite,30,\n"
    "15.16,76.66,2.70,Quartzite,5,\n"
)


class TestUnits:
    def test_read_samples_ngpm_style(self, tmp_path):
        p = tmp_path / "samples.csv"
        p.write_text(SAMPLES_CSV)
        s = read_samples(p, "density", "EPSG:32643")
        assert len(s) == 7 and s[0]["rock"] == "Banded Iron Formation"
        assert 600e3 < s[0]["x"] < 700e3 and 1.6e6 < s[0]["y"] < 1.7e6   # projected to UTM 43N
        chi = read_samples(p, "susceptibility", "EPSG:32643")
        assert chi[0]["value"] == pytest.approx(50000e-6 * 4 * np.pi)       # CGS -> SI

    def test_values_from_samples(self, tmp_path):
        p = tmp_path / "samples.csv"
        p.write_text(SAMPLES_CSV)
        s = read_samples(p, "density", "EPSG:32643")
        units = {u.name: u for u in units_from_samples(
            {"BIF": {"rock_types": ["banded iron formation"], "weight": 10},
             "mafic": {"rock_types": ["amphibolite", "dolerite*"]},
             "granitoid": {"rock_types": ["*granite*"], "margin": 0.0},
             "cover": {"value": -0.1, "lower": -0.3, "upper": 0.0}}, s)}
        bif = units["BIF"]
        assert bif.value == pytest.approx(3.39 - 2.67) and bif.n_samples == 2 and bif.weight == 10
        assert bif.lower == pytest.approx(3.38 - 2.67 - 0.05) and bif.upper == pytest.approx(3.40 - 2.67 + 0.05)
        assert units["mafic"].n_samples == 2
        g = units["granitoid"]
        assert g.value == pytest.approx(2.63 - 2.67) and g.lower == pytest.approx(2.61 - 2.67)
        assert units["cover"].source == "given" and units["cover"].upper == 0.0

    def test_unit_without_samples_needs_a_value(self):
        with pytest.raises(ValueError, match="no samples"):
            units_from_samples({"BIF": {"rock_types": ["banded iron formation"]}}, [])
        with pytest.raises(ValueError, match="value"):
            units_from_samples({"X": {}}, [])


def test_points_in_polygon_with_a_hole():
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(4, 4), (6, 4), (6, 6), (4, 6)]
    px, py = np.array([1, 5, 9, 11]), np.array([1, 5, 9, 5])
    np.testing.assert_array_equal(points_in_polygon(px, py, [outer, hole]), [True, False, True, False])


# ── Placing units on a mesh ─────────────────────────────────────────────


@pytest.fixture
def grid_cells():
    """20 x 20 x 10 cells of 100 m under flat ground at 0 m."""
    xs = (np.arange(20) + 0.5) * 100
    zs = -(np.arange(10) + 0.5) * 100
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    centres = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    half = np.full_like(centres, 50.0)
    return centres, half, (lambda x, y: np.zeros(np.shape(x)))


def _units():
    return {"dense": {"value": 0.5, "lower": 0.3, "upper": 0.7, "weight": 8},
            "light": {"value": -0.1, "lower": -0.2, "upper": 0.0}}


class TestConstraints:
    def test_box_body_between_depths(self, grid_cells):
        centres, half, surface = grid_cells
        geo = build_constraints({"units": _units(), "sources": [
            {"type": "body", "unit": "dense", "box": [500, 900, 500, 900], "top_m": 200, "bottom_m": 600}],
            "unconstrained": {"lower": -0.2, "upper": 0.8}}, centres, half, surface)
        x, y, z = centres.T
        expect = (x > 500) & (x < 900) & (y > 500) & (y < 900) & (-z >= 200) & (-z <= 600)
        np.testing.assert_array_equal(geo.unit_index == 0, expect)
        assert expect.sum() == 4 * 4 * 4
        np.testing.assert_allclose(geo.reference[expect], 0.5)
        np.testing.assert_allclose(geo.lower[expect], 0.3)
        np.testing.assert_allclose(geo.weights[expect], 8)
        np.testing.assert_allclose(geo.reference[~expect], 0.0)
        np.testing.assert_allclose(geo.upper[~expect], 0.8)
        np.testing.assert_allclose(geo.weights[~expect], 1.0)
        s = geo.summary()
        assert s["n_constrained"] == 64 and s["units"][0]["n_cells"] == 64
        assert s["sources"][0]["cells_by_unit"] == {"dense": 64}

    def test_dipping_body_moves_down_dip(self, grid_cells):
        centres, half, surface = grid_cells
        # a 200 m wide dyke dipping 45 degrees to the east (azimuth 90)
        geo = build_constraints({"units": _units(), "sources": [
            {"type": "body", "unit": "dense", "box": [400, 600, 0, 2000], "top_m": 0, "bottom_m": 1000,
             "dip": 45, "dip_direction": 90}]}, centres, half, surface)
        x, z = centres[:, 0], centres[:, 2]
        on = geo.unit_index == 0
        for depth in (50, 450, 850):
            layer = on & np.isclose(-z, depth)
            assert np.allclose(x[layer].mean(), 500 + depth, atol=60)

    def test_spec_from_the_page_model_builder(self, grid_cells):
        """What the upload page's model builder sends: given values, outlines in metres, no files."""
        centres, half, surface = grid_cells
        circle = [[1000 + 300 * np.cos(a), 1000 + 300 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 32, endpoint=False)]
        spec = {"name": "2 drawn bodies", "property": "density", "background": 2.67,
                "units": {"Granite": {"value": -0.07, "lower": -0.12, "upper": -0.02, "weight": 10},
                          "BIF": {"value": 0.63, "lower": 0.625, "upper": 0.635, "weight": 100}},
                "sources": [{"type": "body", "unit": "Granite", "polygon": circle, "top_m": 0, "bottom_m": 400},
                            {"type": "body", "unit": "BIF", "polygon": [[200, 0], [400, 0], [400, 2000], [200, 2000]],
                             "top_m": 100, "bottom_m": 700, "dip": 60, "dip_direction": 90}],
                "unconstrained": {"lower": -0.3, "upper": 0.8},
                "builder": [{"name": "Granite", "shape": "cylinder", "cyl": {"x": 1000, "y": 1000, "r": 300}}]}
        geo = build_constraints(spec, centres, half, surface, "", default_bounds=(-1.0, 1.0))
        x, y, z = centres.T
        r = np.hypot(x - 1000, y - 1000)
        inner = (r < 300 - 50 * np.sqrt(2)) & (-z < 400)          # cells wholly in the cylinder
        assert np.all(geo.unit_index[inner] == 0)
        np.testing.assert_allclose(geo.reference[inner], -0.07)
        # cells the circle cuts carry their share of it; its volume is kept to a few %
        edge = (r > 300 - 50 * np.sqrt(2)) & (r < 300 + 50 * np.sqrt(2)) & (-z < 400)
        assert np.all(geo.share[edge] < 1) and np.all(geo.reference[edge] <= 0)
        np.testing.assert_allclose(geo.share[np.isclose(-z, 50)].sum() * 100 * 100, np.pi * 300 ** 2, rtol=0.03)
        on = geo.unit_index == 1
        assert not np.any(on & (-z < 100)) and not np.any(on & (-z > 700))
        # 60 degrees to the east: the slab has moved (650 - 100) / tan(60) = 318 m by 650 m depth
        deep = on & np.isclose(-z, 650)
        assert abs(x[deep].mean() - (300 + 550 / np.tan(np.radians(60)))) < 60
        np.testing.assert_allclose(geo.weights[on & (geo.share == 1)], 100)
        np.testing.assert_allclose(geo.upper[geo.share == 0], 0.8)

    def test_samples_boreholes_and_override_order(self, grid_cells, tmp_path):
        centres, half, surface = grid_cells
        (tmp_path / "s.csv").write_text("x,y,density,rock\n250,250,3.4,BIF\n1750,1750,2.6,granite\n")
        (tmp_path / "holes.csv").write_text(
            "x,y,bearing,cl_inclina,length_m,commodity_\n"
            "1050,1050,0,90,380,IRON ORE\n"          # vertical, 380 m: 4 cells
            "1450,1050,90,45,420,IRON ORE\n"         # 45 degrees to the east
            "1450,1450,0,90,200,Limestone\n")        # no unit for limestone
        spec = {"units": {"dense": {"rock_types": ["bif"], "weight": 10},
                          "light": {"rock_types": ["granite"]}},
                "samples": {"file": "s.csv"},
                "sources": [{"type": "samples", "depth_m": 300},
                            {"type": "boreholes", "file": "holes.csv",
                             "unit_by": {"field": "commodity_", "values": {"iron*": "dense"}}},
                            {"type": "body", "unit": "light", "box": [1000, 1100, 1000, 1100],
                             "top_m": 0, "bottom_m": 100}]}
        geo = build_constraints(spec, centres, half, surface, str(tmp_path))
        x, y, z = centres.T
        col = (np.abs(x - 250) < 1) & (np.abs(y - 250) < 1)
        assert np.all(geo.unit_index[col & (-z <= 300)] == 0) and np.all(geo.unit_index[col & (-z > 300)] == -1)
        # the vertical hole: 4 cells, the top one then overridden by the later body
        hole = (np.abs(x - 1050) < 1) & (np.abs(y - 1050) < 1)
        assert geo.unit_index[hole & np.isclose(-z, 50)] == 1
        assert np.all(geo.unit_index[hole & (-z > 100) & (-z < 400)] == 0)
        # the inclined hole leaves its column eastwards as it deepens
        incl = (geo.unit_index == 0) & (y > 1000) & (y < 1100) & (x > 1400)
        assert incl.sum() >= 4 and x[incl][np.argmax(-z[incl])] > 1600
        by = {s["type"]: s for s in geo.summary()["sources"]}
        assert by["boreholes"]["n_features"] == 2 and by["boreholes"]["n_holes"] == 3

    def test_borehole_radius_tapers_the_log_into_the_cells_around(self, grid_cells, tmp_path):
        """radius_m: share 1 - d / radius of the nearest logged unit, whole cells on the trace."""
        centres, half, surface = grid_cells
        (tmp_path / "holes.csv").write_text(
            "x,y,bearing,cl_inclina,length_m,unit\n"
            "1050,1050,0,90,300,dense\n"             # vertical: dense down to 300 m
            "1050,1050,0,90,150,light\n")            # the same collar: a later hole wins its cells
        spec = {"units": _units(), "unconstrained": {"lower": -1.0, "upper": 1.0},
                "sources": [{"type": "boreholes", "file": "holes.csv", "radius_m": 250,
                             "unit_by": {"field": "unit"}}]}
        geo = build_constraints(spec, centres, half, surface, str(tmp_path))
        x, y, z = centres.T
        d = np.hypot(x - 1050, y - 1050)
        at = lambda dx, depth: (np.isclose(d, dx)) & np.isclose(-z, depth)      # noqa: E731
        # on the trace: whole cells, the later hole's unit in its two cells, then the first's
        assert np.all(geo.share[at(0, 50)] == 1) and np.all(geo.unit_index[at(0, 150)] == 1)
        assert np.all(geo.unit_index[at(0, 250)] == 0)
        # 100 m beside the trace at 250 m depth: 1 - 100/250 of dense, mixed with the free part
        s = 1 - 100 / 250
        np.testing.assert_allclose(geo.share[at(100, 250)], s)
        np.testing.assert_allclose(geo.reference[at(100, 250)], s * 0.5)
        np.testing.assert_allclose(geo.upper[at(100, 250)], s * 0.7 + (1 - s) * 1.0)
        np.testing.assert_allclose(geo.weights[at(100, 250)], s * 8 + (1 - s) * 1.0)
        # beyond the radius (and below the hole's end by more than it) nothing changes
        assert np.all(geo.share[d > 250] == 0) and np.all(geo.share[-z > 300 + 250] == 0)
        assert geo.summary()["sources"][0]["radius_m"] == 250

    def test_map_polygons_from_geojson(self, grid_cells, tmp_path):
        centres, half, surface = grid_cells
        fc = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:32643"}},
              "features": [{"type": "Feature", "properties": {"LITHO": "Banded iron formation"},
                            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [300, 0], [300, 300], [0, 300], [0, 0]]]}},
                           {"type": "Feature", "properties": {"LITHO": "Alluvium"},
                            "geometry": {"type": "Polygon", "coordinates": [[[1500, 1500], [1900, 1500], [1900, 1900], [1500, 1500]]]}}]}
        (tmp_path / "map.geojson").write_text(json.dumps(fc))
        geo = build_constraints({"units": _units(), "sources": [
            {"type": "map", "file": "map.geojson", "bottom_m": 500,
             "unit_by": {"field": "LITHO", "values": {"banded*": "dense"}}}]},
            centres, half, surface, str(tmp_path))
        assert np.sum(geo.unit_index == 0) == 3 * 3 * 5 and np.sum(geo.unit_index == 1) == 0

    def test_missing_files_are_left_out_not_fatal(self, grid_cells, tmp_path):
        """Files are optional: what depends on a missing one is skipped, with notes."""
        centres, half, surface = grid_cells
        spec = {"units": {"BIF": {"rock_types": ["bif"]}, **_units()},
                "samples": {"file": "nope.csv"},
                "sources": [{"type": "samples", "depth_m": 300},
                            {"type": "boreholes", "file": "holes.shp", "unit": "BIF"},
                            {"type": "body", "unit": "BIF", "box": [0, 500, 0, 500]},
                            {"type": "body", "unit": "dense", "box": [500, 900, 500, 900],
                             "top_m": 0, "bottom_m": 300}]}
        geo = build_constraints(spec, centres, half, surface, str(tmp_path))
        s = geo.summary()
        assert [u["name"] for u in s["units"]] == ["dense", "light"]      # BIF had no samples
        assert s["n_constrained"] == 4 * 4 * 3
        assert [x.get("skipped") for x in s["sources"]] == ["no samples", "file not found",
                                                             "unit left out", None]
        text = " ".join(s["notes"])
        assert "nope.csv" in text and "holes.shp" in text and "BIF" in text

    def test_unknown_unit_and_source(self, grid_cells):
        centres, half, surface = grid_cells
        with pytest.raises(ValueError, match="not in the unit table"):
            build_constraints({"units": _units(), "sources": [
                {"type": "body", "unit": "nope", "box": [0, 1, 0, 1]}]}, centres, half, surface)
        with pytest.raises(ValueError, match="Unknown geology source"):
            build_constraints({"units": _units(), "sources": [{"type": "magic"}]},
                              centres, half, surface)


def _write_point_shapefile(base, points, records):
    """A minimal point shapefile (.shp/.dbf/.prj) for the reader's test."""
    n = len(points)
    content = b"".join(struct.pack(">2i", i + 1, 10) + struct.pack("<i2d", 1, x, y)
                       for i, (x, y) in enumerate(points))
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    header = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + len(content)) // 2) \
        + struct.pack("<2i4d4d", 1000, 1, min(xs), min(ys), max(xs), max(ys), 0, 0, 0, 0)
    base.with_suffix(".shp").write_bytes(header + content)
    fields = [("name", "C", 20), ("depth", "N", 10)]
    hdr = struct.pack("<4BIHH20x", 3, 26, 9, 29, n, 32 + 32 * len(fields) + 1,
                      1 + sum(f[2] for f in fields))
    fdesc = b"".join(struct.pack("<11sc4xBB14x", nm.encode(), t.encode(), ln, 0) for nm, t, ln in fields)
    body = b"".join(b" " + r[0].encode().ljust(20) + str(r[1]).encode().rjust(10) for r in records)
    base.with_suffix(".dbf").write_bytes(hdr + fdesc + b"\r" + body + b"\x1a")
    base.with_suffix(".prj").write_text('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984"]]')


def test_shapefile_reader(tmp_path):
    base = tmp_path / "holes"
    _write_point_shapefile(base, [(76.5, 15.0), (76.6, 15.1)], [("KCY-1", 94.5), ("KCY-2", 115)])
    feats, geographic = read_shapefile(base.with_suffix(".shp"))
    assert geographic and len(feats) == 2
    assert feats[0]["coordinates"] == (76.5, 15.0) and feats[1]["properties"] == {"name": "KCY-2", "depth": 115.0}
    assert read_dbf(base.with_suffix(".dbf"))[0]["depth"] == 94.5
    assert read_vector(base.with_suffix(".dbf"))[0][1]["properties"]["name"] == "KCY-2"


# ── Through the pipeline ────────────────────────────────────────────────


def _gravity_files(tmp_path):
    """A shallow dense block (300-700 m E/N, 0-300 m deep) under 11 x 11 stations."""
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.gravity import GravityMethod

    xy = np.arange(0.0, 1001.0, 100.0)
    xx, yy = np.meshgrid(xy, xy)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 0.0)])
    mesh = Mesh3D.uniform(20, 20, 10, 100.0, 100.0, 100.0, origin=(-500.0, -500.0, -1000.0))
    cc = mesh.to_discretize().cell_centers
    block = ((cc[:, 0] > 300) & (cc[:, 0] < 700) & (cc[:, 1] > 300) & (cc[:, 1] < 700)
             & (cc[:, 2] > -300))
    survey = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
    gz = -GravityMethod().make_simulation(mesh, survey).dpred(block * 0.5)   # positive down
    (tmp_path / "grav.csv").write_text("x,y,gz\n" + "\n".join(f"{a},{b},{c:.6f}" for a, b, c in
                                                              zip(locs[:, 0], locs[:, 1], gz)))
    return tmp_path


def _params(**extra):
    p = {"method_type": "gravity", "inversion_mode": "single",
         "datasets": [{"type": "gravity", "method": "gravity", "files": ["grav.csv"],
                       "noise_pct": 0.0, "noise_floor": 0.02, "component": "gz"}],
         "data_file": "grav.csv", "topography": {"flat_elevation": 0}, "param_mode": "manual",
         "regularization_type": "l2", "alpha_s": 1.0, "max_iter": 15, "mesh_type": "tensor",
         "core_cell_m": 100.0, "core_cell_z_m": 100.0, "depth_core_m": 1000.0,
         "pad_distance_m": 300.0, "bounds_lower": -0.2, "bounds_upper": 0.8}
    p.update(extra)
    return p


GEOLOGY = {"name": "block", "units": {"dense": {"value": 0.5, "lower": 0.3, "upper": 0.7, "weight": 10}},
           "sources": [{"type": "body", "unit": "dense", "box": [300, 700, 300, 700],
                        "top_m": 0, "bottom_m": 300}]}


class TestPipeline:
    def test_constraints_pull_the_body_to_the_surface(self, tmp_path):
        from geoinv3d.cloud.worker import pack_result, run_data_pipeline
        from geoinv3d.viz.result_workflow import build_workflow, load_result

        _gravity_files(tmp_path)
        free = run_data_pipeline(_params(), str(tmp_path))
        geo = run_data_pipeline(_params(geology=GEOLOGY), str(tmp_path))

        s = geo["geology"]
        assert s["n_constrained"] == 4 * 4 * 3 and s["units"][0]["n_cells"] == 48
        assert geo["settings"]["geology"] == "block"
        ref = geo["reference_model"]
        assert ref.shape == geo["recovered_model"].shape and np.sum(ref == 0.5) == 48

        from geoinv3d.viz.result_workflow import result_mesh
        cc = result_mesh(geo).cell_centers
        body = (cc[:, 0] > 300) & (cc[:, 0] < 700) & (cc[:, 1] > 300) & (cc[:, 1] < 700) & (cc[:, 2] > -300)
        # the constrained model keeps the block within the unit's range, and nearer the
        # true 0.5 than the unconstrained one (0.47 against 0.35 when written)
        assert np.all(geo["recovered_model"][body] >= 0.3 - 1e-6)
        assert abs(np.mean(geo["recovered_model"][body]) - 0.5) \
            < abs(np.mean(free["recovered_model"][body]) - 0.5) - 0.05
        # both fit the data
        for r in (free, geo):
            res = r["data"]["observed"] - r["data"]["predicted"]
            assert np.sqrt(np.mean(res ** 2)) < 0.3 * np.sqrt(np.mean(r["data"]["observed"] ** 2))

        path = pack_result(geo, str(tmp_path / "result.zip"))
        with zipfile.ZipFile(path) as zf:
            assert "reference_model.npy" in zf.namelist()
            assert "reference_model" not in json.loads(zf.read("result.json"))
        run = load_result(path)
        run["_name"] = "geology"
        m3 = next(n for n in build_workflow([run])["nodes"]
                  if n["type"] == "RegularizedInversionNode")["output"]["model_3d"]
        assert m3["true_label"] == "Geology reference" and max(m3["true_values"]) == 0.5

    def test_cda_uses_the_reference(self, tmp_path):
        from geoinv3d.cloud.worker import run_data_pipeline
        _gravity_files(tmp_path)
        r = run_data_pipeline(_params(regularization_type="l1l2", l1l2_solver="cda", l1_ratio=0.9,
                                      lambda_decades=2.0, geology=GEOLOGY), str(tmp_path))
        m, ref = r["recovered_model"], r["reference_model"]
        assert np.all(m >= -0.2 - 1e-9) and np.all(m <= 0.8 + 1e-9)
        # the elastic net shrinks towards the reference: the block keeps most of its 0.5
        assert np.mean(m[ref == 0.5]) > 0.3

    def test_not_for_joint_jobs(self, tmp_path):
        from tests.test_data_pipeline import _joint_params, _station_grid, _synthetic, _write_csv
        from geoinv3d.cloud.worker import run_data_pipeline
        locs = _station_grid()
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        _write_csv(tmp_path / "m.csv", locs, _synthetic("magnetics", locs))
        params = _joint_params(["g.csv"], ["m.csv"], geology=GEOLOGY, max_iter=2)
        r = run_data_pipeline(params, str(tmp_path))
        assert any("Geology constraints apply to single" in n for n in r.get("notes", []))
        assert "geology" not in r


# ── Layer stacks, volume shares, resistivity, sharp boundaries ──────────


def _stack(layers, **extra):
    """A layers source and its unit table: layers = [(name, thickness or None, value)]."""
    units = {n: {"value": v} for n, _, v in layers}
    src = {"type": "layers", "layers": [{"unit": n, "thickness_m": t} for n, t, _ in layers], **extra}
    return units, src


def _depth(grid_cells):
    return -grid_cells[0][:, 2]


class TestLayers:
    def test_thin_layers_are_mixed_by_volume(self, grid_cells):
        """Ten 30 m layers in 100 m cells: none is lost, each cell holds the volume average."""
        centres, half, surface = grid_cells
        layers = [(f"L{k}", 30.0, 0.02 * k) for k in range(10)]
        units, src = _stack(layers)
        geo = build_constraints({"units": units, "sources": [src]}, centres, half, surface)
        col = (np.abs(centres[:, 0] - 1050) < 1) & (np.abs(centres[:, 1] - 1050) < 1)
        ref = {round(d): r for d, r in zip(_depth(grid_cells)[col], geo.reference[col])}
        # 0-100 m: L0, L1, L2 (30 m each) and 10 m of L3
        assert np.isclose(ref[50], (30 * 0 + 30 * 0.02 + 30 * 0.04 + 10 * 0.06) / 100)
        # the contrast times thickness is kept down the column (gravity sees the same mass)
        assert np.isclose(np.sum(geo.reference[col] * 100), sum(30 * v for _, _, v in layers))
        s = geo.summary()
        assert all(u["n_touched"] > 0 for u in s["units"])
        # 0-300 m is filled: three whole cells per column, each a mix
        assert np.isclose(sum(u["volume_cells"] for u in s["units"]), 400 * 3)
        assert s["n_touched"] == 400 * 3 and s["n_partial"] == 0
        assert any("Thinner than the cells" in n and "L3 (30 m in 100 m cells)" in n for n in s["notes"])

    def test_half_space_and_outline(self, grid_cells):
        centres, half, surface = grid_cells
        units, src = _stack([("cover", 150.0, -0.1), ("basement", None, 0.2)], box=[0, 1000, 0, 2000])
        geo = build_constraints({"units": units, "sources": [src]}, centres, half, surface)
        x, d = centres[:, 0], _depth(grid_cells)
        west, east = x < 1000, x > 1000
        assert np.all(geo.reference[east] == 0) and np.all(geo.share[east] == 0)
        np.testing.assert_allclose(geo.reference[west & np.isclose(d, 150)], (50 * -0.1 + 50 * 0.2) / 100)
        np.testing.assert_allclose(geo.reference[west & (d > 200)], 0.2)         # down to the bottom

    def test_tilted_interfaces(self, grid_cells):
        """Interfaces dipping 10 degrees east: the boundary deepens by tan(10) per metre east."""
        centres, half, surface = grid_cells
        units, src = _stack([("upper", 300.0, 0.0), ("lower", None, 0.3)],
                            dip=10.0, dip_direction=90.0, origin=[1000.0, 1000.0])
        geo = build_constraints({"units": units, "sources": [src]}, centres, half, surface)
        x, y = centres[:, 0], centres[:, 1]
        for xc in (250.0, 1050.0, 1750.0):
            col = np.isclose(x, xc) & np.isclose(y, 1050.0)
            lower_thickness = np.sum(geo.reference[col] / 0.3 * 100)             # m of "lower" in 0-1000 m
            expect = 300.0 + (xc - 1000.0) * np.tan(np.radians(10.0))
            assert abs((1000.0 - lower_thickness) - expect) < 1.0

    def test_elevation_reference_ignores_the_ground(self, grid_cells):
        centres, half, _ = grid_cells
        step = lambda x, y: 200.0 * (np.asarray(x) > 1000)                       # noqa: E731
        units, src = _stack([("slab", 300.0, 0.1)], reference="elevation", top_m=-100.0)
        geo = build_constraints({"units": units, "sources": [src]}, centres, half, step)
        z = centres[:, 2]
        on = geo.share > 0.99          # -100 to -400 m elevation on both sides of the step
        assert on.any() and np.all((z[on] < -100) & (z[on] > -400))
        assert np.sum(on & (centres[:, 0] < 1000)) == np.sum(on & (centres[:, 0] > 1000))


class TestResistivity:
    @staticmethod
    def _two_halves(grid_cells, mixing=None):
        """10 ohm m over 1000 ohm m, 50 m each: the top 100 m cells are half and half."""
        centres, half, surface = grid_cells
        units = {"conductor": {"value": 10.0}, "resistor": {"value": 1000.0, "lower": 500, "upper": 5000}}
        src = {"type": "layers", "layers": [{"unit": "conductor", "thickness_m": 50.0},
                                            {"unit": "resistor", "thickness_m": 50.0}]}
        spec = {"property": "resistivity", "units": units, "sources": [src]}
        if mixing:
            spec["mixing"] = mixing
        return build_constraints(spec, centres, half, surface, default_reference=np.log(0.01))

    def test_values_in_ohm_m_model_in_log_conductivity(self, grid_cells):
        geo = self._two_halves(grid_cells)
        u = {x.name: x for x in geo.units}
        assert np.isclose(u["conductor"].value, -np.log(10.0))
        # default range: a factor 2; given bounds swap (the lowest resistivity is the highest log sigma)
        assert np.isclose(u["conductor"].upper, -np.log(5.0)) and np.isclose(u["conductor"].lower, -np.log(20.0))
        assert np.isclose(u["resistor"].lower, -np.log(5000)) and np.isclose(u["resistor"].upper, -np.log(500))
        d = u["resistor"].as_dict()
        assert (d["value_ohm_m"], d["lower_ohm_m"], d["upper_ohm_m"]) == (1000.0, 500.0, 5000.0)
        # below 100 m: the method's background, log(0.01)
        np.testing.assert_allclose(geo.reference[_depth(grid_cells) > 100], np.log(0.01))

    @pytest.mark.parametrize("mixing,rho", [(None, 100.0), ("conductance", 1 / ((0.1 + 0.001) / 2)),
                                            ("resistance", (10.0 + 1000.0) / 2)])
    def test_mixing_laws(self, grid_cells, mixing, rho):
        """Geometric mean, or keeping the conductance, or the transverse resistance."""
        geo = self._two_halves(grid_cells, mixing)
        top = _depth(grid_cells) < 100
        np.testing.assert_allclose(np.exp(-geo.reference[top]), rho, rtol=1e-9)
        assert np.all(geo.lower[top] <= geo.reference[top]) and np.all(geo.reference[top] <= geo.upper[top])

    def test_resistivity_samples(self, tmp_path):
        (tmp_path / "r.csv").write_text("x,y,rock,Resistivity (ohm m)\n500000,1700000,shale,10\n"
                                        "500000,1700000,shale,1000\n500000,1700000,granite,5000\n")
        samples = read_samples(str(tmp_path / "r.csv"), "resistivity")
        units = units_from_samples({"shale": {"rock_types": ["shale"]}}, samples, "resistivity")
        assert np.isclose(np.exp(-units[0].value), 100.0)                     # geometric mean
        assert np.isclose(np.exp(-units[0].upper), 5.0) and np.isclose(np.exp(-units[0].lower), 2000.0)
        (tmp_path / "c.csv").write_text("x,y,lithology,Conductivity (mS/m)\n500000,1700000,clay,100\n")
        assert np.isclose(read_samples(str(tmp_path / "c.csv"), "resistivity")[0]["value"], 10.0)


class TestSharpBoundaries:
    def test_labels_and_face_weights(self, grid_cells):
        import discretize
        from simpeg import regularization
        from geoinv3d.cloud.worker import _face_breaks

        centres, half, surface = grid_cells
        units = {"sill": {"value": 0.3, "sharp": True}, "soft": {"value": 0.1}}
        geo = build_constraints({"units": units, "sources": [
            {"type": "body", "unit": "soft", "box": [0, 2000, 0, 2000], "top_m": 0, "bottom_m": 200},
            {"type": "body", "unit": "sill", "box": [500, 1500, 500, 1500], "top_m": 300, "bottom_m": 500}]},
            centres, half, surface)
        lab = geo.smoothness_labels()
        assert set(np.unique(lab)) == {0, 1}                 # the sill's cells; "soft" is not sharp
        mesh = discretize.TensorMesh([np.full(20, 100.0), np.full(20, 100.0), np.full(10, 100.0)],
                                     origin=(0, 0, -1000))
        order = np.lexsort((centres[:, 0], centres[:, 1], centres[:, 2]))    # the mesh's cell order
        reg = regularization.WeightedLeastSquares(mesh)
        smooth_z = next(t for t in reg.objfcts if isinstance(t, regularization.SmoothnessFirstOrder)
                        and t.orientation == "z")
        w = _face_breaks(smooth_z, lab[order], 0.01)
        assert w.shape == (smooth_z.regularization_mesh.aveCC2Fz.shape[0],)
        assert np.sum(w == 0.01) == 2 * 10 * 10                # the sill's top and bottom faces
        smooth_z.set_weights(geology=w)                      # SimPEG takes face weights
        assert build_constraints({"units": {"a": {"value": 1}}, "sources": [
            {"type": "body", "unit": "a", "box": [0, 5, 0, 5], "crs": "job"}]}, centres, half,
            surface).smoothness_labels() is None


class TestResistivityPipeline:
    def test_dc_with_a_layered_prior(self, tmp_path):
        from tests.test_data_pipeline import TestEMData
        from geoinv3d.cloud.worker import run_data_pipeline
        TestEMData._dc(tmp_path)
        geology = {"property": "resistivity",
                   "units": {"cover": {"value": 30.0, "weight": 2},
                             "body": {"value": 10.0, "sharp": True, "weight": 5}},
                   "sources": [{"type": "layers", "layers": [{"unit": "cover", "thickness_m": 20.0}]},
                               {"type": "body", "unit": "body", "box": [-60, 60, -60, 60],
                                "top_m": 0, "bottom_m": 150}]}
        params = {"method_type": "dc", "inversion_mode": "single", "param_mode": "manual",
                  "datasets": [{"method": "dc", "files": ["dc.npz"],
                                "method_kwargs": {"sigma_background": 0.01}}],
                  "regularization_type": "l2", "max_iter": 3, "max_irls_iterations": 1,
                  "core_cell_m": 50.0, "core_cell_z_m": 50.0, "depth_core_m": 300.0,
                  "pad_distance_m": 400.0, "topography": {"flat_elevation": 0.0}, "geology": geology}
        r = run_data_pipeline(params, str(tmp_path))
        s = r["geology"]
        assert s["property"] == "resistivity" and s["mixing"] == "log" and s["units"][1]["sharp"]
        ref = np.asarray(r["reference_model"])
        # log sigma between the background (100 ohm m) and the body's 10 ohm m; the body's own
        # cells at log(0.1); 20 m of 30 ohm m mixed into the top 50 m cells elsewhere
        assert np.all(ref <= np.log(0.1) + 1e-9) and np.all(ref >= np.log(0.01) - 1e-9)
        assert np.sum(np.isclose(ref, np.log(0.1))) >= 3
        assert np.any(np.isclose(ref, 0.4 * np.log(1 / 30) + 0.6 * np.log(0.01)))
        assert any("Thinner than the cells" in n for n in r.get("notes", []))
        assert np.all(np.isfinite(r["recovered_model"]))
