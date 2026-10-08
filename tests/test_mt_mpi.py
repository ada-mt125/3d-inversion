"""MT's frequencies on the ranks of an MPI job (geoinv3d.methods.parallel, the "mpi" transport):
the same numbers as one process, several simulations sharing the ranks, errors reaching rank 0,
and the job's rank 0 stopping the serving ranks at its end.

Most tests run the ranks as threads behind a stand-in communicator (every message pickled, as
MPI sends it), so they need no MPI library; ``test_under_mpiexec`` runs the real thing when
mpi4py and mpiexec are there.
"""

import os
import pickle
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from geoinv3d.methods import parallel
from geoinv3d.methods.mt import MTMethod

sys.path.insert(0, str(Path(__file__).parent))
from test_mt_parallel import COMPS, FREQS, _setup   # noqa: E402


class FakeComm:
    """Rank ``rank`` of an in-process MPI world of ``size`` ranks (mpi4py's lower-case API)."""

    def __init__(self, rank, size, boxes):
        self.rank, self.size, self.boxes = rank, size, boxes

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def send(self, obj, dest, tag):
        self.boxes[(self.rank, dest, tag)].put(pickle.dumps(obj))

    def recv(self, source, tag):
        return pickle.loads(self.boxes[(source, self.rank, tag)].get(timeout=600))

    def Abort(self, code=1):
        raise RuntimeError(f"MPI_Abort({code})")


@pytest.fixture
def mpi_world():
    """An MPI world of 3 ranks: rank 0 here, ranks 1 and 2 serving in threads."""
    size = 3

    class Boxes(dict):
        def __missing__(self, key):
            self[key] = queue.Queue()
            return self[key]

    boxes = Boxes()
    comms = [FakeComm(r, size, boxes) for r in range(size)]
    threads = [threading.Thread(target=parallel.serve_mpi, args=(comms[r],), daemon=True)
               for r in range(1, size)]
    for t in threads:
        t.start()
    parallel.set_mpi_comm(comms[0])
    try:
        yield comms[0]
    finally:
        parallel.stop_mpi_ranks(comms[0])
        for t in threads:
            t.join(timeout=30)
        parallel.set_mpi_comm(None)
        assert not any(t.is_alive() for t in threads)


def test_the_transport_follows_the_launch(mpi_world, monkeypatch):
    assert parallel.mpi_launched() and parallel.parallel_backend() == "mpi"
    assert parallel.available_workers() == 2
    monkeypatch.setenv("GEOINV3D_PARALLEL", "processes")
    assert parallel.parallel_backend() == "processes"
    monkeypatch.setenv("GEOINV3D_PARALLEL", "ranks")
    with pytest.raises(ValueError):
        parallel.parallel_backend()


def test_mpi_ranks_match_one_process(mpi_world):
    mesh, active, mask, survey = _setup()
    rng = np.random.default_rng(0)
    m = np.log(1e-2) + 0.5 * rng.standard_normal(active.sum())
    one = MTMethod(frequencies=FREQS, components=COMPS, mask=mask).make_simulation_active(mesh, survey, active)
    mt = MTMethod(frequencies=FREQS, components=COMPS, mask=mask, n_workers="auto")
    # in an MPI job "auto" takes the ranks, however small the mesh, and no more than there are
    assert mt.workers_for(100) == 2
    assert MTMethod(frequencies=FREQS, components=COMPS, mask=mask, n_workers=3).workers_for(10 ** 6) == 2
    par = mt.make_simulation_active(mesh, survey, active)
    try:
        assert par.backend == "mpi" and par.n_workers == 2 and par._workers == []
        v, w = rng.standard_normal(m.size), rng.standard_normal(one.survey.nD)
        f1, f2 = one.fields(m), par.fields(m)
        for a, b in ((one.dpred(m, f=f1), par.dpred(m, f=f2)),
                     (one.Jvec(m, v, f=f1), par.Jvec(m, v, f=f2)),
                     (one.Jtvec(m, w, f=f1), par.Jtvec(m, w, f=f2)),
                     (one.getJtJdiag(m, f=f1), par.getJtJdiag(m, f=f2))):
            np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-12 * np.abs(a).max())
        np.testing.assert_allclose(par.dpred(m + 0.1), one.dpred(m + 0.1), rtol=1e-8)
    finally:
        par.close()


def test_simulations_share_the_ranks_and_errors_reach_rank_0(mpi_world):
    """Two parallel simulations alive at once on the same ranks (by id); a rank's error is
    raised on rank 0 and the ranks keep serving; a closed simulation is dropped there."""
    mesh, active, mask, survey = _setup()
    m = np.full(int(active.sum()), np.log(1e-2))
    a = MTMethod(frequencies=FREQS, components=COMPS, mask=mask, n_workers=2).make_simulation_active(
        mesh, survey, active)
    b = MTMethod(frequencies=FREQS, components=COMPS, mask=mask, n_workers=2).make_simulation_active(
        mesh, survey, active)
    try:
        da = a.dpred(m)
        np.testing.assert_allclose(b.dpred(m + 0.2), b.dpred(m + 0.2))
        np.testing.assert_allclose(a.dpred(m), da)          # a's fields untouched by b's model
        with pytest.raises(parallel.WorkerError, match="Traceback"):
            a.Jtvec(m, np.zeros(2))                          # data of the wrong size
        np.testing.assert_allclose(a.dpred(m + 0.1), b.dpred(m + 0.1))
    finally:
        a.close()
    with pytest.raises(parallel.WorkerError, match="KeyError"):
        a._send("dpred", [None, None])                       # dropped on the ranks
    b.close()


def test_closing_opened_drops_a_jobs_simulations(mpi_world):
    mesh, active, mask, survey = _setup()
    with parallel.closing_opened() as started:
        sim = MTMethod(frequencies=FREQS, components=COMPS, mask=mask, n_workers=2).make_simulation_active(
            mesh, survey, active)
    assert list(started) == [(2, sim.threads, "mpi")]
    with pytest.raises(parallel.WorkerError, match="KeyError"):
        sim._send("dpred", [None, None])


def test_mpi_run_rank_0_and_the_others():
    """mpi_run: rank 0 runs the job and then stops the serving ranks; an error on rank 0
    aborts the job; a serving rank only serves."""
    boxes = {}

    def box(key):
        return boxes.setdefault(key, queue.Queue())

    class Comm(FakeComm):
        def send(self, obj, dest, tag):
            box((self.rank, dest, tag)).put(pickle.dumps(obj))

        def recv(self, source, tag):
            return pickle.loads(box((source, self.rank, tag)).get(timeout=30))

    c0, c1 = Comm(0, 2, None), Comm(1, 2, None)
    served = threading.Thread(target=lambda: parallel.mpi_run(lambda: pytest.fail("rank 1 ran the job"), c1))
    served.start()
    assert parallel.mpi_run(lambda: sys.exit(3), c0) == 3        # its exit code; rank 1 told to stop
    served.join(timeout=30)
    assert not served.is_alive()
    with pytest.raises(RuntimeError, match="MPI_Abort"):
        parallel.mpi_run(lambda: 1 / 0, Comm(0, 2, None))
    assert parallel.mpi_run(lambda: 5) == 5                      # no MPI: just the job


MPI_CHECK = Path(__file__).with_name("mpi_mt_check.py")


@pytest.mark.skipif(not shutil.which("mpiexec") or not __import__("importlib").util.find_spec("mpi4py"),
                    reason="needs mpi4py and mpiexec")
def test_under_mpiexec():
    """The real thing: 3 ranks of mpiexec, rank 0 comparing with one process."""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])}
    out = subprocess.run(["mpiexec", "-n", "3", sys.executable, str(MPI_CHECK)], capture_output=True,
                         text=True, timeout=900, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "MPI OK 2 ranks" in out.stdout, out.stdout + out.stderr
