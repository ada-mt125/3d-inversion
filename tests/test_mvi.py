"""Magnetization-vector inversion (MVI): a remanent body that an induced susceptibility
cannot fit (the systematic residual of the Karnataka magnetic data)."""

import numpy as np
import pytest

from geoinv3d.cloud.task import InversionTask
from geoinv3d.cloud.worker import execute_task, magnetization_directions, pack_result
from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.methods.magnetics import MagneticsMethod

FIELD = (50000.0, 60.0, 0.0)
NX, NZ, H = 12, 6, 50.0


def _direction(inc, dec):
    """Unit vector (east, north, up) of an inclination (down) and declination."""
    i, d = np.radians(inc), np.radians(dec)
    return np.array([np.cos(i) * np.sin(d), np.cos(i) * np.cos(d), -np.sin(i)])


def remanent_data():
    mesh = Mesh3D.uniform(NX, NX, NZ, H, H, H, origin=(0.0, 0.0, -NZ * H))
    xy = np.linspace(25.0, NX * H - 25.0, NX)
    xx, yy = np.meshgrid(xy, xy)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 20.0)])
    cc = mesh.to_discretize().cell_centers
    body = ((cc[:, 0] > 200) & (cc[:, 0] < 400) & (cc[:, 1] > 200) & (cc[:, 1] < 400)
            & (cc[:, 2] > -200) & (cc[:, 2] < -50))
    true_dir = _direction(-30.0, 120.0)            # remanent: upwards, to the south-east
    M = np.outer(body, 0.05 * true_dir)            # (n, 3)
    sv = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
    sim = MagneticsMethod(inducing_field=FIELD, magnetization="vector").make_simulation_full(mesh, sv)
    clean = sim.dpred(M.T.ravel())
    std = 0.02 * np.abs(clean).max() * np.ones(clean.size)
    d = clean + np.random.default_rng(3).normal(scale=std)
    return {"locs": locs, "d": d, "std": std, "true_dir": true_dir, "body": body, "cc": cc}


@pytest.fixture(scope="module")
def remanent():
    return remanent_data()


def _task(p, magnetization, **extra):
    return InversionTask(
        task_id=f"mvi-{magnetization}", hx=np.full(NX, H), hy=np.full(NX, H), hz=np.full(NZ, H),
        origin=(0.0, 0.0, -NZ * H), method_type="magnetics",
        method_kwargs={"inducing_field": list(FIELD), "magnetization": magnetization},
        observed_data=p["d"], data_std=p["std"], station_locations=p["locs"],
        initial_model=np.zeros(NX * NX * NZ), regularization_type="sparse",
        norms=[0.0, 2.0, 2.0, 2.0], alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        max_iter=40, max_irls_iterations=20, **extra)


def test_mvi_fits_a_remanent_body_that_induced_cannot(remanent, tmp_path):
    p, n = remanent, len(remanent["d"])
    induced = execute_task(_task(p, "induced", bounds_lower=0.0, bounds_upper=1.0))
    chi_induced = float(np.sum(((p["d"] - induced["predicted"]) / p["std"]) ** 2)) / n
    mvi = execute_task(_task(p, "vector", bounds_upper=1.0))
    chi_mvi = float(np.sum(((p["d"] - mvi["predicted"]) / p["std"]) ** 2)) / n
    assert chi_induced > 3.0           # a susceptibility along the present field cannot
    assert chi_mvi < 1.5
    assert mvi["regularization"] == "mvi_sparse_IRLS"
    M = mvi["magnetization_vector"]
    assert M.shape == (NX * NX * NZ, 3)
    np.testing.assert_allclose(mvi["recovered_model"], np.linalg.norm(M, axis=1))
    info = mvi["magnetization"]
    got = _direction(info["resultant_inclination"], info["resultant_declination"])
    assert np.degrees(np.arccos(np.clip(got @ p["true_dir"], -1, 1))) < 30.0
    # the amplitude is centred on the body (horizontally within a cell) and concentrated
    # in it (on this run 30 % of it in 5.6 % of the cells, about 1.5 cells too deep)
    amp, cc = mvi["recovered_model"], p["cc"]
    centre = (amp[:, None] * cc).sum(axis=0) / amp.sum()
    assert np.hypot(*(centre[:2] - cc[p["body"]].mean(axis=0)[:2])) < H
    assert amp[p["body"]].sum() / amp.sum() > 4 * p["body"].mean()
    # the components are packed with the result
    import zipfile
    path = pack_result(mvi, str(tmp_path / "mvi.zip"))
    assert "magnetization_vector.npy" in zipfile.ZipFile(path).namelist()


def test_directions_and_refusals():
    M = np.outer(np.r_[np.ones(5), 0.01 * np.ones(20)], _direction(45.0, -90.0))
    info = magnetization_directions(M)
    assert info["n_cells"] == 5
    assert info["resultant_inclination"] == pytest.approx(45.0)
    assert info["resultant_declination"] == pytest.approx(-90.0)
    assert info["resultant_coherence"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="magnetization"):
        MagneticsMethod(magnetization="remanent")
    with pytest.raises(ValueError, match="single magnetic"):
        MagneticsMethod(magnetization="vector").make_simulation_mapped(None, None, None)
