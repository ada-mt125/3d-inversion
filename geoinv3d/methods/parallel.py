"""Simulations split over workers: MT's frequencies solved side by side, on processes or MPI ranks.

Each frequency of an MT survey is a PDE of its own (its own matrix, factorized once per model),
independent of the others: the survey split by frequency into k simulations runs on k workers
at once — the fields, the predicted data, J v, J^T v and diag(J^T J) each computed by every
worker for its frequencies, the main one only adding up.  The workers keep their own fields and
factorizations (as many in all as one process would keep: the memory is about the same), so a
model goes to them once and only vectors come and go.

Two transports carry the same messages to the same worker loop (``_Server``):

``processes``
    Processes started here with "spawn" (forking a process whose MKL / OpenMP threads are
    running can hang the child), as daemons, each with its share of the cores' threads
    (``threads``).  For one machine: a laptop, an EC2 instance.

``mpi``
    The ranks of an MPI job (mpi4py), for a cluster.  The job is started on N + 1 ranks, e.g.

        mpiexec -n 4 python -m geoinv3d.cloud.worker --local params.json DATA OUT
        srun -n 4 --cpus-per-task 8 python -m geoinv3d.cloud.worker --local params.json DATA OUT

    with OMP_NUM_THREADS / MKL_NUM_THREADS set to the cores of a rank.  Rank 0 runs the job as
    usual (``mpi_run``); ranks 1..N wait in ``serve_mpi`` and each takes a group of the
    frequencies of every parallel simulation the job makes (a rank may hold several, by id).
    Only rank 0 talks to them, one operation at a time, so simulations alive together share the
    ranks safely.  If rank 0 fails, it aborts the MPI job (the ranks would otherwise wait for
    ever, until the scheduler's wall time).

The transport is chosen by ``parallel_backend()``: GEOINV3D_PARALLEL ("processes", "mpi" or
"auto", the default: "mpi" when started by an MPI launcher on more than one rank).

SimPEG has the same idea (``simpeg.meta.MultiprocessingMetaSimulation``); this one starts its
processes with spawn as daemons (a job that ends or fails does not wait on them), raises a
worker's error in the main process (SimPEG's returns it as a value), limits each process's
threads, and has the MPI transport.
"""

from __future__ import annotations

import contextlib
import contextvars
import itertools
import multiprocessing as mp
import os
import pickle
import sys
import traceback
import weakref

import numpy as np
import scipy.sparse as sp
from simpeg.meta import MetaSimulation
from simpeg.props import HasModel

_CTX = mp.get_context("spawn")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
BACKENDS = ("processes", "mpi")
# the number of ranks as MPI launchers give it: Open MPI, MPICH / Intel MPI / MS-MPI (PMI), Slurm
MPI_SIZE_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_STEP_NUM_TASKS")
_TAG = 7301                 # the tag of the messages between rank 0 and the ranks serving
# the simulations started inside closing_opened() (in this thread's context: one job's)
_OPENED = contextvars.ContextVar("geoinv3d_parallel_opened", default=None)
_IDS = itertools.count(1)   # a parallel simulation's id on the workers (MPI ranks hold several)
_COMM = None                # the MPI communicator: COMM_WORLD, or one set by set_mpi_comm (tests)


# ── the transport ────────────────────────────────────────────────────────────

def set_mpi_comm(comm) -> None:
    """Use ``comm`` (anything with Get_rank, Get_size, send, recv and Abort, as mpi4py's
    communicators) instead of MPI.COMM_WORLD; None goes back to it.  For tests."""
    global _COMM
    _COMM = comm


def mpi_launched() -> bool:
    """Was this process started by an MPI launcher (mpiexec, srun) on more than one rank?
    Read from the launcher's environment, without starting MPI."""
    for var in MPI_SIZE_VARS:
        try:
            if int(os.environ.get(var, "0")) > 1:
                return True
        except ValueError:
            pass
    return _COMM is not None and _COMM.Get_size() > 1


def mpi_comm():
    """The MPI communicator (mpi4py's COMM_WORLD unless set_mpi_comm gave another)."""
    global _COMM
    if _COMM is None:
        try:
            from mpi4py import MPI
        except ImportError as e:
            raise RuntimeError("The MPI transport needs mpi4py (and an MPI library): "
                               "pip install mpi4py") from e
        _COMM = MPI.COMM_WORLD
    return _COMM


def parallel_backend(asked: str | None = None) -> str:
    """The transport of the parallel simulations: ``asked``, else GEOINV3D_PARALLEL, else
    "auto": "mpi" when launched on several MPI ranks, else "processes"."""
    choice = (asked or os.environ.get("GEOINV3D_PARALLEL") or "auto").strip().lower()
    if choice == "auto":
        return "mpi" if mpi_launched() else "processes"
    if choice not in BACKENDS:
        raise ValueError(f"GEOINV3D_PARALLEL must be 'processes', 'mpi' or 'auto', got {choice!r}")
    return choice


def mpi_workers() -> int:
    """The ranks that serve (all but rank 0); 0 when the job is not an MPI job."""
    if not mpi_launched():
        return 0
    return max(0, mpi_comm().Get_size() - 1)


def available_workers(backend: str | None = None) -> int:
    """How many workers a parallel simulation can have: the cores (processes), or the ranks
    serving (MPI)."""
    if parallel_backend(backend) == "mpi":
        return mpi_workers()
    return os.cpu_count() or 1


def _env_threads() -> int | None:
    for var in THREAD_VARS[:2]:
        try:
            return max(1, int(os.environ[var]))
        except (KeyError, ValueError):
            continue
    return None


class _Processes:
    """Workers on processes started here (one per group of simulations)."""

    kind = "processes"

    def __init__(self, n: int, threads: int):
        saved = {k: os.environ.get(k) for k in THREAD_VARS}
        self.workers = []
        try:
            os.environ.update({k: str(threads) for k in THREAD_VARS})    # read by the children
            for _ in range(n):
                w = _Worker()
                w.start()
                self.workers.append(w)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self._finalizer = weakref.finalize(self, _stop, self.workers)

    def send(self, i, msg):
        self.workers[i].tasks.put(msg)

    def recv(self, i):
        return self.workers[i].results.get()

    def close(self, sim_id):
        self._finalizer()


class _MPIRanks:
    """Workers on the ranks 1..n of the MPI job, which serve in ``serve_mpi``."""

    kind = "mpi"

    def __init__(self, n: int):
        self.comm = mpi_comm()
        if self.comm.Get_rank() != 0:
            raise RuntimeError("Parallel simulations are made on rank 0 of an MPI job")
        if n > self.comm.Get_size() - 1:
            raise ValueError(f"{n} workers asked for, but the MPI job has only "
                             f"{self.comm.Get_size() - 1} ranks to serve")
        self.ranks = list(range(1, n + 1))
        self.closed = False

    def send(self, i, msg):
        self.comm.send(msg, dest=self.ranks[i], tag=_TAG)

    def recv(self, i):
        return self.comm.recv(source=self.ranks[i], tag=_TAG)

    def close(self, sim_id):
        """Drop the simulation on its ranks (they keep serving the job)."""
        if self.closed:
            return
        self.closed = True
        try:
            for i in range(len(self.ranks)):
                self.send(i, (sim_id, "drop", None))
            for i in range(len(self.ranks)):
                self.recv(i)
        except Exception:            # noqa: BLE001 — at exit the ranks may be gone already
            pass


# ── the workers ─────────────────────────────────────────────────────────────

class WorkerError(RuntimeError):
    """An error in a simulation worker (its traceback in the message)."""


class _Failed:
    def __init__(self, text):
        self.text = text


class _Server:
    """What a worker does: holds simulations by id (each a MetaSimulation of a group of
    simulations), their model and its fields, and answers one message at a time."""

    def __init__(self):
        self.sims = {}

    def handle(self, sim_id, op, arg):
        if op == "sim":
            self.sims[sim_id] = [pickle.loads(arg), None]
            return None
        if op == "drop":
            self.sims.pop(sim_id, None)
            return None
        entry = self.sims[sim_id]
        sim, fields = entry
        if op == "model":
            sim.model, entry[1] = arg, None
            return None
        if op == "fields":
            entry[1] = sim.fields(sim.model)
            return None
        if op == "dpred":
            return sim.dpred(sim.model, f=fields)
        if op == "jvec":
            return sim.Jvec(sim.model, arg, f=fields)
        if op == "jtvec":
            return sim.Jtvec(sim.model, arg, f=fields)
        if op == "jtj":
            return sim.getJtJdiag(sim.model, W=arg, f=fields)
        raise ValueError(f"unknown task {op!r}")

    def answer(self, msg):
        try:
            return self.handle(*msg)
        except BaseException:          # noqa: BLE001 — every error goes back to the caller
            return _Failed(traceback.format_exc())


class _Worker(_CTX.Process):
    """A process serving one parallel simulation (the processes transport)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.tasks = _CTX.Queue()
        self.results = _CTX.Queue()

    def run(self):     # (in the process)
        server = _Server()
        while True:
            msg = self.tasks.get()
            if msg is None:
                break
            self.results.put(server.answer(msg))


def _stop(workers):
    for w in workers:
        try:
            if w.is_alive():
                w.tasks.put(None)
                w.join(timeout=5)
            if w.is_alive():
                w.terminate()
        except (OSError, ValueError, AssertionError):     # already gone
            pass


def serve_mpi(comm=None) -> None:
    """A serving rank's loop (ranks 1..N): answer rank 0 until it says stop (None)."""
    comm = comm or mpi_comm()
    server = _Server()
    while True:
        msg = comm.recv(source=0, tag=_TAG)
        if msg is None:
            break
        comm.send(server.answer(msg), dest=0, tag=_TAG)


def stop_mpi_ranks(comm=None) -> None:
    """Rank 0: let the serving ranks leave serve_mpi (at the end of the job)."""
    comm = comm or mpi_comm()
    for rank in range(1, comm.Get_size()):
        comm.send(None, dest=rank, tag=_TAG)


def mpi_run(main, comm=None):
    """Run ``main()`` as an MPI job's rank 0 while the other ranks serve; without MPI just
    ``main()``.  Returns main's exit code on rank 0 (0 on the others).  An error on rank 0
    aborts the job, so the serving ranks do not wait for ever."""
    if comm is None and not mpi_launched():
        return main()
    comm = comm or mpi_comm()
    if comm.Get_size() == 1:
        return main()
    if comm.Get_rank() != 0:
        serve_mpi(comm)
        return 0
    try:
        code = main()
    except SystemExit as e:
        code = e.code
    except BaseException:          # noqa: BLE001
        traceback.print_exc()
        sys.stderr.flush()
        comm.Abort(1)
        raise
    stop_mpi_ranks(comm)
    return code


# ── the simulation ──────────────────────────────────────────────────────────

@contextlib.contextmanager
def closing_opened():
    """Stop, on leaving, the workers of every ParallelMetaSimulation started inside (a job's:
    its simulations are not kept, but SimPEG's objects refer to each other, so they may wait
    for the garbage collector).  Yields the (workers, threads per worker, transport) of each
    simulation started, for the record.  Contexts do not cross threads, so a job running in
    a thread of its own closes only its own."""
    opened = []
    token = _OPENED.set(opened)
    try:
        yield _Started(opened)
    finally:
        _OPENED.reset(token)
        for ref, *_ in opened:
            sim = ref()
            if sim is not None:
                sim.close()


class _Started:
    """The (workers, threads, transport) of the simulations started so far (closing_opened)."""

    def __init__(self, opened):
        self._opened = opened

    def __iter__(self):
        return iter([tuple(rest) for _, *rest in self._opened])

    def __len__(self):
        return len(self._opened)


class ParallelMetaSimulation(MetaSimulation):
    """``simulations`` (their data in this order), each with its mapping from the model, on
    ``n_workers`` workers (contiguous groups of them; default one each) of the ``backend``
    transport (see parallel_backend); with processes each has ``threads`` threads for its
    solver (default the cores shared out), with MPI the ranks' own (OMP_NUM_THREADS)."""

    def __init__(self, simulations, mappings, n_workers=None, threads=None, backend=None):
        super().__init__(simulations, mappings)
        self.backend = parallel_backend(backend)
        n_sim = len(self.simulations)
        cap = n_sim if self.backend == "processes" else min(n_sim, max(1, mpi_workers()))
        n = max(1, min(int(n_workers or n_sim), cap))
        groups = [g for g in np.array_split(np.arange(n_sim), n) if g.size]
        # pickled here: an error is raised at once (a queue's feeder thread would only print
        # it, and the worker would wait for a simulation that never comes)
        blobs, offsets = [], [0]
        for g in groups:
            chunk = MetaSimulation([self.simulations[i] for i in g], [self.mappings[i] for i in g])
            blobs.append(pickle.dumps(chunk))
            offsets.append(offsets[-1] + chunk.survey.nD)
        self._id = next(_IDS)
        if self.backend == "processes":
            self.threads = int(threads or max(1, (os.cpu_count() or 1) // len(groups)))
            self._transport = _Processes(len(groups), self.threads)
        else:
            self.threads = _env_threads()
            self._transport = _MPIRanks(len(groups))
        self._n = len(groups)
        self._offsets = np.asarray(offsets)
        self._finalizer = weakref.finalize(self, self._transport.close, self._id)
        self._fields_id = 0         # the model whose fields the workers hold
        self._send("sim", blobs)
        opened = _OPENED.get()
        if opened is not None:      # a job's: closed when it ends (closing_opened)
            opened.append((weakref.ref(self), self.n_workers, self.threads, self.backend))

    @property
    def n_workers(self) -> int:
        return self._n

    @property
    def _workers(self):
        """The processes (processes transport; [] with MPI)."""
        return getattr(self._transport, "workers", [])

    def close(self):
        """Stop the processes, or drop the simulation on the MPI ranks (also when the
        simulation is collected, and at exit)."""
        self._finalizer()

    def _send(self, op, args):
        for i, a in enumerate(args):
            self._transport.send(i, (self._id, op, a))
        out = [self._transport.recv(i) for i in range(self._n)]
        bad = [r.text for r in out if isinstance(r, _Failed)]
        if bad:
            raise WorkerError("A simulation worker failed:\n" + bad[0])
        return out

    @MetaSimulation.model.setter
    def model(self, value):
        # the workers get the model (they map it themselves); none of the local copies is used
        if HasModel.model.fset(self, value):
            self._send("model", [self._model] * self.n_workers)
            self._fields_id += 1
            self._have_fields = None

    def fields(self, m):
        self.model = m
        if getattr(self, "_have_fields", None) != self._fields_id:
            self._send("fields", [None] * self.n_workers)
            self._have_fields = self._fields_id
        return ("fields", self._fields_id)

    def _ready(self, m, f):
        """The workers hold the fields of ``m`` (computed again if ``f`` is of another model)."""
        if m is not None:
            self.model = m
        if f is None or f != ("fields", self._fields_id) or getattr(self, "_have_fields", None) != self._fields_id:
            self.fields(self.model)

    def dpred(self, m=None, f=None):
        self._ready(m, f)
        return np.concatenate(self._send("dpred", [None] * self.n_workers))

    def Jvec(self, m, v, f=None):
        self._ready(m, f)
        return np.concatenate(self._send("jvec", [v] * self.n_workers))

    def Jtvec(self, m, v, f=None):
        self._ready(m, f)
        o = self._offsets
        return np.sum(self._send("jtvec", [v[o[i]:o[i + 1]] for i in range(self.n_workers)]), axis=0)

    def getJtJdiag(self, m, W=None, f=None):
        self._ready(m, f)
        if W is None:
            w = np.ones(self.survey.nD)
        else:      # a diagonal matrix (SimPEG's data misfit's W) or its diagonal
            w = np.asarray(W.diagonal() if sp.issparse(W) else W, dtype=float)
        o = self._offsets
        return np.sum(self._send("jtj", [w[o[i]:o[i + 1]] for i in range(self.n_workers)]), axis=0)
