"""MT's frequencies on processes side by side (geoinv3d.methods.parallel): the same numbers as
one process — the predicted data, J v, J^T v and diag(J^T J) — and fresh fields for a new model."""

import numpy as np
import pytest
from discretize import TensorMesh

from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.methods.mt import MTMethod

FREQS, COMPS = [1.0, 10.0, 100.0], ["xy_real", "xy_imag", "yx_real", "yx_imag"]


def _setup():
    h = [(100.0, 4, -1.5), (100.0, 8), (100.0, 4, 1.5)]
    tm = TensorMesh([h, h, [(100.0, 6, -1.5), (50.0, 8), (50.0, 4), (100.0, 6, 1.5)]], origin="CCN")
    mesh = Mesh3D(hx=tm.h[0], hy=tm.h[1], hz=tm.h[2], origin=tuple(tm.origin))
    active = tm.cell_centers[:, 2] < 0
    stations = np.array([[-150.0, 0, 0], [150.0, 0, 0], [0, 150, 0]])
    mask = np.ones((3, 4, 3), bool)
    mask[1, :, 2] = False                 # a station without the middle frequency
    n = int(mask.sum())
    survey = SurveyData(locations=stations, observed=np.zeros(n), std=np.ones(n))
    return mesh, active, mask, survey


def test_parallel_frequencies_match_one_process():
    mesh, active, mask, survey = _setup()
    rng = np.random.default_rng(0)
    m = np.log(1e-2) + 0.5 * rng.standard_normal(active.sum())
    one = MTMethod(frequencies=FREQS, components=COMPS, mask=mask).make_simulation_active(mesh, survey, active)
    par = MTMethod(frequencies=FREQS, components=COMPS, mask=mask, n_workers=2).make_simulation_active(
        mesh, survey, active)
    try:
        assert par.n_workers == 2 and par.survey.nD == one.survey.nD == mask.sum()
        f1, f2 = one.fields(m), par.fields(m)
        v, w = rng.standard_normal(m.size), rng.standard_normal(one.survey.nD)
        for a, b in ((one.dpred(m, f=f1), par.dpred(m, f=f2)),
                     (one.Jvec(m, v, f=f1), par.Jvec(m, v, f=f2)),
                     (one.Jtvec(m, w, f=f1), par.Jtvec(m, w, f=f2)),
                     (one.getJtJdiag(m, f=f1), par.getJtJdiag(m, f=f2))):
            np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-12 * np.abs(a).max())
        # another model: the processes solve again
        np.testing.assert_allclose(par.dpred(m + 0.1), one.dpred(m + 0.1), rtol=1e-8)
    finally:
        par.close()


def test_worker_errors_reach_the_caller():
    from geoinv3d.methods.parallel import WorkerError
    mesh, active, mask, survey = _setup()
    par = MTMethod(frequencies=FREQS, components=COMPS, mask=mask, n_workers=2).make_simulation_active(
        mesh, survey, active)
    try:
        m = np.full(int(active.sum()), np.log(1e-2))
        with pytest.raises(WorkerError):
            par.Jtvec(m, np.zeros(2))        # data of the wrong size: an error in a process
        assert par.dpred(m).shape == (survey.observed.size,)     # and the processes still work
    finally:
        par.close()


def test_n_workers_checks_and_auto():
    with pytest.raises(ValueError):
        MTMethod(frequencies=FREQS, n_workers=0)
    m = MTMethod(frequencies=FREQS, n_workers="auto")
    assert m.workers_for(1000) == 1                     # small meshes: one process
    assert 1 <= m.workers_for(10 ** 6) <= len(FREQS)
    assert MTMethod(frequencies=FREQS, n_workers=8).workers_for(10 ** 6) == len(FREQS)
