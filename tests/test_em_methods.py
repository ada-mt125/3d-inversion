"""DC resistivity and MT: log-conductivity models, electrodes, joint inversion."""

import contextlib
import io
import warnings

import numpy as np
import pytest
from discretize import TensorMesh

from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.methods.dc_resistivity import DCResistivityMethod, electrode_rows
from geoinv3d.methods.mt import MTMethod

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
        np.testing.assert_allclose(fd, J @ v, rtol=1e-5, atol=1e-8 * abs(fd).max())

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
        active[:10] = False           # inactive cells stay at the background
        act = mt.make_simulation_active(mesh, _survey(locs, 2), active)
        m_bg = m.copy()
        m_bg[~active] = np.log(1e-2)
        np.testing.assert_allclose(act.dpred(m_bg[active]), full, rtol=1e-8)


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
