"""DC resistivity and MT: log-conductivity models, electrodes, joint inversion."""

import contextlib
import io
import os
import warnings

import numpy as np
import pytest
from discretize import TensorMesh

from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.methods.dc_resistivity import DCResistivityMethod, electrode_rows
from geoinv3d.methods.mt import MTMethod
from geoinv3d.methods.solvers import pde_solver

warnings.filterwarnings("ignore", message=".*default solver.*")


def _mesh(h, nz_core, pad=4, origin="CCN", pad_z=True):
    xy = [(h, pad, -1.5), (h, 16), (h, pad, 1.5)]
    z = [(h, pad, -1.5), (h, nz_core)] if origin == "CCN" else \
        [(h, pad, -1.6), (h, nz_core), (h, pad, 1.6)]
    tm = TensorMesh([xy, xy, z], origin=origin)
    return tm, Mesh3D(hx=tm.h[0], hy=tm.h[1], hz=tm.h[2], origin=tuple(tm.origin))


def _dipole_dipole(xs, y=0.0, ns=(1, 2)):
    rows = []
    for i in range(len(xs) - 1):
        for n in ns:
            j = i + 1 + n
            if j + 1 < len(xs):
                rows.append([xs[i], y, 0, xs[i + 1], y, 0, xs[j], y, 0, xs[j + 1], y, 0])
    return np.array(rows)


def _survey(locs, n=None):
    n = len(locs) if n is None else n
    return SurveyData(locations=locs, observed=np.zeros(n), std=np.ones(n))


@pytest.fixture(scope="module")
def dc_mesh():
    return _mesh(25.0, 8)


class TestDC:
    def test_half_space_apparent_resistivity(self, dc_mesh):
        tm, mesh = dc_mesh
        rows = _dipole_dipole(np.arange(-150.0, 151.0, 25.0))
        dc = DCResistivityMethod(sigma_background=1e-2, data_type="apparent_resistivity")
        sim = dc.make_simulation_full(mesh, _survey(rows))
        rho = sim.dpred(np.full(tm.nC, dc.default_model_value))   # log(sigma_background)
        # 100 ohm m, to the discretization error of this coarse mesh
        assert np.all((rho > 95) & (rho < 130))
        # consecutive rows with the same A, B are one source
        assert sim.survey.nSrc == len(np.unique(rows[:, :6], axis=0))
        # pole-dipole
        pd = electrode_rows(rows[:, 0:3], None, rows[:, 6:9], rows[:, 9:12])
        rho = dc.make_simulation_full(mesh, _survey(pd)).dpred(np.full(tm.nC, np.log(1e-2)))
        assert np.all((rho > 95) & (rho < 130))

    def test_data_keep_their_order(self, dc_mesh):
        tm, mesh = dc_mesh
        rows = _dipole_dipole(np.arange(-150.0, 151.0, 25.0))
        perm = np.random.default_rng(0).permutation(len(rows))
        m = np.full(tm.nC, np.log(1e-2))
        cc = tm.cell_centers
        m[(abs(cc[:, 0]) < 50) & (abs(cc[:, 1]) < 50) & (cc[:, 2] > -100)] = np.log(0.1)
        dc = DCResistivityMethod()
        a = dc.make_simulation_full(mesh, _survey(rows)).dpred(m)
        b = dc.make_simulation_full(mesh, _survey(rows[perm])).dpred(m)
        np.testing.assert_allclose(b, a[perm], rtol=1e-10, atol=1e-14)

    def test_active_cells_and_jacobian(self, dc_mesh):
        tm, mesh = dc_mesh
        rows = _dipole_dipole(np.arange(-100.0, 101.0, 25.0))
        active = tm.cell_centers[:, 2] < -30
        dc = DCResistivityMethod()
        sim = dc.make_simulation_active(mesh, _survey(rows), active)
        m = np.full(int(active.sum()), np.log(1e-2))
        J = sim.getJ(m)
        v = np.random.default_rng(1).normal(size=m.size)
        fd = (sim.dpred(m + 1e-3 * v) - sim.dpred(m - 1e-3 * v)) / 2e-3
        # the central difference's own truncation error is ~1e-5 here (1.2e-5 seen on Windows);
        # a wrong Jacobian is off by far more
        np.testing.assert_allclose(fd, J @ v, rtol=5e-5, atol=1e-8 * abs(fd).max())

    def test_bad_electrodes(self):
        dc = DCResistivityMethod()
        with pytest.raises(ValueError, match="12"):
            dc.make_survey(_survey(np.zeros((3, 3))))
        rows = _dipole_dipole(np.arange(-100.0, 101.0, 25.0))
        rows[0, 0] = np.nan
        with pytest.raises(ValueError, match="A and an M"):
            dc.make_survey(_survey(rows))
        with pytest.raises(ValueError, match="data_type"):
            DCResistivityMethod(data_type="ohm")


class TestMT:
    @pytest.fixture(scope="class")
    def mt_mesh(self):
        hm = [(100.0, 4, -1.6), (100.0, 6), (100.0, 4, 1.6)]
        tm = TensorMesh([hm, hm, [(100.0, 6, -1.6), (100.0, 6), (100.0, 6, 1.6)]],
                        origin="CCC")
        return tm, Mesh3D(hx=tm.h[0], hy=tm.h[1], hz=tm.h[2], origin=tuple(tm.origin))

    def test_half_space_is_log_conductivity(self, mt_mesh):
        tm, mesh = mt_mesh
        mt = MTMethod(frequencies=[10.0], components=["xy_real", "xy_imag"],
                      sigma_background=1e-2)
        locs = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]])
        Z = mt.make_simulation_full(mesh, _survey(locs, 4)).dpred(
            np.full(tm.nC, mt.default_model_value))
        z = Z[0:2] + 1j * Z[2:4]
        rho_a = np.abs(z) ** 2 / (2 * np.pi * 10.0 * 4e-7 * np.pi)
        np.testing.assert_allclose(rho_a, 100.0, rtol=0.05)

    def test_mapped_matches_full_and_active(self, mt_mesh):
        from simpeg import maps
        tm, mesh = mt_mesh
        mt = MTMethod(frequencies=[10.0], components=["xy_real", "xy_imag"])
        locs = np.array([[0.0, 0.0, 0.0]])
        m = np.full(tm.nC, np.log(1e-2))
        m[tm.nC // 2] = np.log(0.2)
        full = mt.make_simulation_full(mesh, _survey(locs, 2)).dpred(m)
        wires = maps.Wires(("other", 5), ("sigma", tm.nC))
        mapped = mt.make_simulation_mapped(mesh, _survey(locs, 2), wires.sigma)
        np.testing.assert_allclose(mapped.dpred(np.r_[np.zeros(5), m]), full, rtol=1e-10)
        active = np.ones(tm.nC, bool)
        active[:10] = False           # inactive cells are air (1e-8 S/m); their layer stays ground
        # the fields where the full simulation takes them (not on the ground and in the air)
        mt = MTMethod(frequencies=[10.0], components=["xy_real", "xy_imag"], h_in_air=False)
        act = mt.make_simulation_active(mesh, _survey(locs, 2), active)
        m_bg = m.copy()
        m_bg[~active] = np.log(1e-8)
        full_bg = mt.make_simulation_full(mesh, _survey(locs, 2)).dpred(m_bg)
        np.testing.assert_allclose(act.dpred(m_bg[active]), full_bg, rtol=1e-8)

    def test_primary_is_air_over_the_background(self):
        """The primary 1D model: background in the layers that are mostly ground, air above."""
        tm = TensorMesh([[(100.0, 4)], [(100.0, 4)], [(50.0, 8)]], origin=[0.0, 0.0, -300.0])
        cc = tm.cell_centers
        mt = MTMethod(sigma_background=1e-2)
        np.testing.assert_array_equal(mt.primary_1d(tm), 1e-2)        # no air: a whole space
        flat = mt.primary_1d(tm, cc[:, 2] < 0)
        np.testing.assert_array_equal(flat, [1e-2] * 6 + [1e-8] * 2)
        # a staircase: the ground at 50 m under three of the four columns, at 0 under one
        ground = np.where(cc[:, 0] < 100, 0.0, 50.0)
        stairs = mt.primary_1d(tm, cc[:, 2] < ground)
        np.testing.assert_array_equal(stairs, [1e-2] * 7 + [1e-8])  # the 0-50 m layer: 3/4 ground
        sim = mt.make_simulation_active(Mesh3D(hx=tm.h[0], hy=tm.h[1], hz=tm.h[2],
                                               origin=tuple(tm.origin)),
                                        _survey(np.zeros((1, 3)), 2), cc[:, 2] < 0)
        sigma = sim.sigmaMap * np.full(int((cc[:, 2] < 0).sum()), np.log(1e-2))
        np.testing.assert_allclose(sigma[cc[:, 2] > 0], 1e-8)        # the air cells

    @pytest.mark.skipif(not os.environ.get("GEOINV3D_SLOW_TESTS") and pde_solver()[1] == "SolverLU",
                        reason="an hour with SuperLU (a minute with PARDISO): set GEOINV3D_SLOW_TESTS, install MUMPS or PARDISO, or run deploy/mt_accuracy.py")
    def test_two_layers_match_the_1d_solution(self):
        """10 ohm m over 1000 ohm m (300 m) on the mesh designed from the skin depths, the
        primary a smooth layering fitted to the data (worker._mt_primaries): the impedance is
        the 1D one of the cells' own layering within 3 % at 0.3, 3 and 30 Hz."""
        from geoinv3d.cloud.meshing import MU0, _rho_a_1d, recommend_mt_mesh, smooth_1d_layers
        from geoinv3d.cloud.worker import _active_below_surface, _build_mt_octree_mesh
        flat = lambda x, y: np.zeros_like(np.asarray(x, dtype=float))   # noqa: E731
        freqs = np.array([0.3, 3.0, 30.0])
        ra = _rho_a_1d(freqs, [10.0, 1000.0], [300.0])
        stats = {"frequencies": list(freqs), "p10": list(ra), "p50": list(ra), "p90": list(ra)}
        ext = (0.0, 2400.0, 0.0, 2400.0)
        mesh = _build_mt_octree_mesh(ext, flat, False, recommend_mt_mesh(ext, 1200.0, freqs, stats))
        dm = mesh.to_discretize()
        active = _active_below_surface(dm, flat)
        m = np.where(dm.cell_centers[active, 2] > -300, np.log(0.1), np.log(1e-3))
        mt = MTMethod(frequencies=freqs, sigma_background=1 / np.exp(np.mean(np.log(ra))),
                      primary_layers=smooth_1d_layers(stats))
        d = mt.make_simulation_active(mesh, _survey(np.array([[1200.0, 1200.0, 0.0]]), 6), active,
                                      forward_only=True).dpred(m)
        rho = np.abs(d[0::2] + 1j * d[1::2]) ** 2 / (2 * np.pi * freqs * MU0)
        # the cells' own layering under the station
        nodes = dm.origin[2] + np.r_[0.0, np.cumsum(dm.h[2])]
        zc = 0.5 * (nodes[1:] + nodes[:-1])
        g = zc < 0
        layers = np.where(zc[g] > -300, 10.0, 1000.0)[::-1]
        ref = _rho_a_1d(freqs, layers, dm.h[2][g][::-1][:-1])
        np.testing.assert_allclose(rho, ref, rtol=0.03)


def test_joint_dc_and_mt_share_a_conductivity_model():
    """MT and DC data of one log-conductivity model, jointly (JointInversion)."""
    from geoinv3d.methods.joint import JointInversion, MethodSetup, ModelRegularization
    hm = [(50.0, 3, -1.6), (50.0, 8), (50.0, 3, 1.6)]
    hz = TensorMesh([[(50.0, 4, -1.6), (50.0, 6), (50.0, 3, 1.6)]]).h[0]
    # the ground surface at z = 0 on top of the 10th cell; 3 air cells above
    tm = TensorMesh([hm, hm, hz], origin=["C", "C", -hz[:10].sum()])
    mesh = Mesh3D(hx=tm.h[0], hy=tm.h[1], hz=tm.h[2], origin=tuple(tm.origin))
    cc = tm.cell_centers
    ground = cc[:, 2] < 0
    body = (abs(cc[:, 0]) < 75) & (abs(cc[:, 1]) < 75) & (cc[:, 2] < -25) & (cc[:, 2] > -175)
    m_true = np.log(1e-2) * np.ones(ground.sum())
    m_true[body[ground]] = np.log(1e-1)
    rng = np.random.default_rng(0)
    dc = DCResistivityMethod(sigma_background=1e-2)
    rows = np.vstack([_dipole_dipole(np.arange(-175.0, 176.0, 50.0), y, ns=(1, 2, 3))
                      for y in (-50.0, 50.0)])
    mt = MTMethod(frequencies=[10.0, 100.0], components=["xy_real", "xy_imag"],
                  sigma_background=1e-2, sigma_inactive=1e-8)
    locs = np.array([[x, y, 0.0] for x in (-50.0, 50.0) for y in (-50.0, 50.0)])
    data = []
    for method, loc in ((dc, rows), (mt, locs)):
        n = len(loc) if method is dc else 2 * 2 * len(loc)
        d = method.make_simulation_active(mesh, _survey(loc, n), ground).dpred(m_true)
        sd = 0.03 * abs(d) + 1e-3 * abs(d).max()
        data.append(SurveyData(locations=loc, observed=d + sd * rng.normal(size=d.size),
                               std=sd, method=method.method_name))
    m0 = np.full(ground.sum(), np.log(1e-2))
    setups = [MethodSetup(dc, data[0], mesh, m0, active_cells=ground, model="sigma"),
              MethodSetup(mt, data[1], mesh, m0, active_cells=ground, model="sigma")]
    joint = JointInversion(setups, regularization=ModelRegularization("l2", alpha_s=1e-2),
                           max_iter=6, max_irls_iterations=2)
    with contextlib.redirect_stdout(io.StringIO()):
        r = joint.run()
    assert list(r.recovered_models) == ["sigma"]
    m = r.recovered_models["sigma"]
    assert m[body[ground]].mean() > m[~body[ground]].mean() + 0.2   # the conductor shows
    chi2 = {k: v["chi2"] for k, v in r.extras["datasets"].items()}
    assert set(chi2) == {"dc_resistivity", "mt"}
    phi0 = sum(float(np.sum(((s.survey.observed - s.method.make_simulation_active(
        mesh, s.survey, ground).dpred(m0)) / s.survey.std) ** 2)) for s in setups)
    assert sum(chi2.values()) < 0.2 * phi0
