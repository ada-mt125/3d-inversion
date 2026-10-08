"""Run under mpiexec by tests/test_mt_mpi.py::test_under_mpiexec:

    mpiexec -n 3 python tests/mpi_mt_check.py

Rank 0 solves the parallel test's MT survey on the other ranks and on itself alone, and prints
"MPI OK <ranks> ranks" when the data, J v, J^T v and diag(J^T J) agree; the other ranks serve.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def job():
    from test_mt_parallel import COMPS, FREQS, _setup
    from geoinv3d.methods.mt import MTMethod
    from geoinv3d.methods.parallel import parallel_backend
    assert parallel_backend() == "mpi", parallel_backend()
    mesh, active, mask, survey = _setup()
    rng = np.random.default_rng(0)
    m = np.log(1e-2) + 0.5 * rng.standard_normal(active.sum())
    one = MTMethod(frequencies=FREQS, components=COMPS, mask=mask).make_simulation_active(mesh, survey, active)
    par = MTMethod(frequencies=FREQS, components=COMPS, mask=mask, n_workers="auto").make_simulation_active(
        mesh, survey, active)
    v, w = rng.standard_normal(m.size), rng.standard_normal(one.survey.nD)
    for a, b in ((one.dpred(m), par.dpred(m)), (one.Jvec(m, v), par.Jvec(m, v)),
                 (one.Jtvec(m, w), par.Jtvec(m, w)), (one.getJtJdiag(m), par.getJtJdiag(m))):
        np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-12 * np.abs(a).max())
    print(f"MPI OK {par.n_workers} ranks", flush=True)
    par.close()
    return 0


if __name__ == "__main__":
    from geoinv3d.methods.parallel import mpi_run
    sys.exit(mpi_run(job))
