"""EDI files and MT datasets: reading, frames, frequencies, the pipeline."""

import os
import re
import warnings

import numpy as np
import pytest
from discretize import TensorMesh

from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.io.edi import MU0, EDIStation, read_edi, rotate, write_edi
from geoinv3d.io.mt_data import common_frequencies, mt_dataset, read_edi_files
from geoinv3d.methods.mt import MTMethod
from geoinv3d.methods.solvers import pde_solver

warnings.filterwarnings("ignore", message=".*solver.*")


def _halfspace(freq, rho=100.0):
    """Z of a halfspace in MT's frame (x north, y east; e^{+i omega t}): Zxy in quadrant 1."""
    zxy = np.sqrt(1j * 2 * np.pi * np.asarray(freq) * MU0 * rho)
    z = np.zeros((len(freq), 2, 2), complex)
    z[:, 0, 1], z[:, 1, 0] = zxy, -zxy
    return z


def _station(name="S1", lat=-12.5, lon=76.25, freq=(1.0, 10.0, 100.0), tipper=True):
    f = np.asarray(freq, float)
    z = _halfspace(f)
    t = np.column_stack([np.full(len(f), 0.1 + 0.05j), np.full(len(f), -0.2 + 0.01j)]) if tipper else None
    return EDIStation(name, lat, lon, 650.0, f, z, 0.03 * np.abs(z) + 1e-12, t,
                      None if t is None else np.full((len(f), 2), 0.02))


class TestEDI:
    def test_round_trip(self, tmp_path):
        st = _station()
        st.z[1, 0, 0] = np.nan                       # a missing value (EMPTY in the file)
        write_edi(tmp_path / "s.edi", st)
        r = read_edi(tmp_path / "s.edi")
        assert (r.name, r.lat, r.lon, r.elev) == ("S1", -12.5, 76.25, 650.0)
        assert np.isnan(r.z[1, 0, 0])
        ok = np.isfinite(st.z)
        np.testing.assert_allclose(r.z[ok], st.z[ok], rtol=1e-6)
        np.testing.assert_allclose(r.t, st.t, rtol=1e-6)
        np.testing.assert_allclose(r.z_std[ok], st.z_std[ok], rtol=1e-5)
        rho, phase = r.rho_phase()
        np.testing.assert_allclose(rho[:, 0, 1], 100.0, rtol=1e-6)
        np.testing.assert_allclose(phase[:, 0, 1], 45.0, atol=1e-6)     # MT's Zxy: quadrant 1
        np.testing.assert_allclose(phase[:, 1, 0], -135.0, atol=1e-6)
        # field units in the file: mV/km per nT
        text = (tmp_path / "s.edi").read_text()
        zxyr = float(re.search(r">ZXYR[^\n]*\n\s*(\S+)", text).group(1))
        assert zxyr == pytest.approx(st.z[0, 0, 1].real / (4e-4 * np.pi), rel=1e-6)

    def test_rotated_data_come_back_to_north(self, tmp_path):
        st = _station()
        z30, t30 = rotate(st.z, st.t, 30.0)
        turned = EDIStation("R", st.lat, st.lon, st.elev, st.freq, z30, None, t30)
        write_edi(tmp_path / "r.edi", turned)
        text = (tmp_path / "r.edi").read_text()
        text = re.sub(r"(>[ZT]ROT //3\n)[^\n>]*", lambda m: m.group(1) + " 30.0 30.0 30.0", text)
        (tmp_path / "r.edi").write_text(text)
        r = read_edi(tmp_path / "r.edi")             # (the file keeps 7 digits)
        np.testing.assert_allclose(r.z, st.z, rtol=1e-5, atol=1e-9)
        np.testing.assert_allclose(r.t, st.t, rtol=1e-5, atol=1e-9)

    def test_spectra_only_files_are_refused(self, tmp_path):
        (tmp_path / "sp.edi").write_text(">HEAD\n DATAID=X\n>=SPECTRASECT\n NFREQ=1\n>END\n")
        with pytest.raises(ValueError, match="SPECTRASECT"):
            read_edi(tmp_path / "sp.edi")


class TestMTDataset:
    def test_frequencies_are_matched_across_stations(self):
        assert np.allclose(common_frequencies([[1.0, 10.0], [1.01, 10.0, 100.0]]), [1.0, 10.0, 100.0])
        assert len(common_frequencies([[1.0, 1000.0]], per_decade=2)) == 7
        a = _station("A", freq=(1.0, 10.0, 100.0))
        b = _station("B", lon=76.26, freq=(1.01, 10.0))           # 1 % off, and no 100 Hz
        ds = mt_dataset([a, b], np.array([[0.0, 0.0], [1000.0, 0.0]]))
        assert np.allclose(ds["frequencies"], [1.0, 10.0, 100.0])
        assert ds["components"] == ["xy_real", "xy_imag", "yx_real", "yx_imag",
                                    "zx_real", "zx_imag", "zy_real", "zy_imag"]
        assert np.isfinite(ds["values"][:2]).all()
        assert np.isfinite(ds["values"][2, :, 0]).all() and np.isnan(ds["values"][2, :, 1]).all()
        assert np.isnan(ds["std"][2, :, 1]).all()          # imaginary parts too: no datum at all

    def test_frames_floors_and_data_types(self):
        st = _station(tipper=True)
        ds = mt_dataset([st], np.zeros((1, 2)), impedance="full", data_type="rho_phase",
                        error_floor=0.1, tipper_floor=0.05)
        v, s = ds["values"][:, :, 0], ds["std"][:, :, 0]
        c = {name: k for k, name in enumerate(ds["components"])}
        # SimPEG (x east, y north): its yx is EDI's xy (quadrant 1), its xy is EDI's yx
        np.testing.assert_allclose(v[:, c["yx_phase"]], 45.0, atol=1e-6)
        np.testing.assert_allclose(v[:, c["xy_phase"]], -135.0, atol=1e-6)
        np.testing.assert_allclose(v[:, c["yx_rho"]], 100.0, rtol=1e-6)
        assert np.isnan(v[:, c["xx_rho"]]).all()           # zero diagonal: no rho, no datum
        # the floor (10 % of sqrt|Zxy Zyx|) is above the files' 3 %: 2 x 10 % for rho
        np.testing.assert_allclose(s[:, c["yx_rho"]], 0.2 * 100.0, rtol=1e-6)
        np.testing.assert_allclose(s[:, c["yx_phase"]], np.degrees(0.1), rtol=1e-6)
        # the tipper turns over with z: SimPEG's Tzx is -EDI's Tzy
        np.testing.assert_allclose(v[:, c["zx_real"]], 0.2)
        np.testing.assert_allclose(v[:, c["zy_imag"]], -0.05)
        np.testing.assert_allclose(s[:, c["zx_real"]], 0.05)            # the tipper floor

    @pytest.mark.parametrize("data_type", ["impedance", "rho_phase"])
    def test_edi_data_are_simpegs_prediction(self, tmp_path, data_type):
        """A halfspace's EDI data, through the frames and units, are what SimPEG computes."""
        write_edi(tmp_path / "hs.edi", EDIStation("HS", 0.0, 0.0, 0.0, np.array([1.0, 10.0]),
                                                  _halfspace([1.0, 10.0]), None))
        ds = mt_dataset(read_edi_files([tmp_path / "hs.edi"]), np.zeros((1, 2)), tipper=False,
                        data_type=data_type)
        hx = [(400.0, 5, -1.5), (400.0, 4), (400.0, 5, 1.5)]
        below = [(50.0, 10, -1.4), (50.0, 10)]
        tm = TensorMesh([hx, hx, below + [(50.0, 2), (50.0, 8, 1.5)]],
                        origin=["C", "C", -TensorMesh([below]).h[0].sum()])
        ground = tm.cell_centers[:, 2] < 0
        mesh = Mesh3D(hx=tm.h[0], hy=tm.h[1], hz=tm.h[2], origin=tuple(tm.origin))
        mask = np.isfinite(ds["values"])
        mt = MTMethod(frequencies=ds["frequencies"], components=ds["components"],
                      sigma_background=1e-2, mask=mask)
        sv = SurveyData(locations=ds["locations"], observed=ds["values"][mask], std=ds["std"][mask],
                        method="mt")
        pred = mt.make_simulation_active(mesh, sv, ground).dpred(np.full(ground.sum(), np.log(1e-2)))
        np.testing.assert_allclose(pred, ds["values"][mask], rtol=0.03, atol=0.6)  # degrees for phase

    @pytest.mark.skipif(not os.environ.get("GEOINV3D_SLOW_TESTS") and pde_solver()[1] == "SolverLU",
                        reason="slow with SuperLU: set GEOINV3D_SLOW_TESTS, or install MUMPS or PARDISO")
    def test_induction_arrows_point_at_a_conductor(self):
        """A conductor east of the station: back in EDI's frame, the real Parkinson arrow
        (-Re Tzx, -Re Tzy) = (north, east) points east."""
        hx = [(200.0, 6, -1.5), (200.0, 16), (200.0, 6, 1.5)]
        below = [(100.0, 8, -1.4), (100.0, 8)]
        tm = TensorMesh([hx, hx, below + [(100.0, 2), (100.0, 8, 1.5)]],
                        origin=["C", "C", -TensorMesh([below]).h[0].sum()])
        cc = tm.cell_centers
        ground = cc[:, 2] < 0
        m = np.full(ground.sum(), np.log(1e-2))
        g = cc[ground]
        m[(g[:, 0] > 400) & (g[:, 0] < 1000) & (np.abs(g[:, 1]) < 600) & (g[:, 2] > -500)] = 0.0
        mt = MTMethod(frequencies=[1.0], components=["zx_real", "zx_imag", "zy_real", "zy_imag"],
                      sigma_background=1e-2)
        sv = SurveyData(locations=np.zeros((1, 3)), observed=np.zeros(4), std=np.ones(4), method="mt")
        d = mt.make_simulation_active(Mesh3D(hx=tm.h[0], hy=tm.h[1], hz=tm.h[2], origin=tuple(tm.origin)),
                                      sv, ground).dpred(m)
        edi_tzx, edi_tzy = -(d[2] + 1j * d[3]), -(d[0] + 1j * d[1])
        assert -edi_tzy.real > 0.02 and abs(edi_tzx.real) < 0.2 * abs(edi_tzy.real)


def test_pipeline_reads_edi_files(tmp_path):
    """EDI files in a job: the UTM zone of their stations, the stations on the ground, the
    missing frequencies left out, and what the result reports."""
    from geoinv3d.cloud import worker
    from geoinv3d.cloud.worker import run_data_pipeline

    seen = {}

    def fake_execute(task, mesh=None):
        seen["task"], seen["mesh"] = task, mesh
        return {"task_id": task.task_id, "n_iterations": 0, "iterations": []}

    mp = pytest.MonkeyPatch()
    mp.setattr(worker, "execute_task", fake_execute)
    try:
        write_edi(tmp_path / "a.edi", _station("A", lat=-12.50, lon=76.250, freq=(1.0, 10.0, 100.0)))
        write_edi(tmp_path / "b.edi", _station("B", lat=-12.50, lon=76.255, freq=(1.0, 10.0)))
        params = {"method_type": "mt", "mesh_type": "tensor",
                  "datasets": [{"method": "mt", "files": ["a.edi", "b.edi"], "impedance": "offdiagonal",
                                "tipper": False, "error_floor": 0.05}],
                  "core_cell_m": 100.0, "core_cell_z_m": 100.0, "depth_core_m": 400.0,
                  "pad_distance_m": 300.0, "topography": {"flat_elevation": 0.0}}
        result = run_data_pipeline(params, str(tmp_path))
    finally:
        mp.undo()
    task = seen["task"]
    assert result["crs"] == "EPSG:32743"                          # UTM 43 S
    assert task.observed_data.size == 3 * 4 + 2 * 4                # B has no 100 Hz
    assert np.allclose(task.station_locations[:, 2], 0.0)          # on the (flat) ground
    info = result["datasets"][0]["mt"]
    assert info["n_stations"] == 2 and info["n_missing"] == 4 and info["names"] == ["A", "B"]
    assert info["components"] == ["xy_real", "xy_imag", "yx_real", "yx_imag"]
    assert task.method_kwargs["mask"][2][0] == [True, False]
