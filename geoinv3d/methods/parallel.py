"""Simulations split over processes: MT's frequencies solved side by side.

Each frequency of an MT survey is a PDE of its own (its own matrix, factorized once per model),
independent of the others: the survey split by frequency into k simulations runs on k processes
at once — the fields, the predicted data, J v, J^T v and diag(J^T J) each computed by every
process for its frequencies, the main process only adding up.  The processes keep their own
fields and factorizations (as many in all as one process would keep: the memory is about the
same), so a model goes to them once and only vectors come and go.

With SciPy's LU (single-threaded) k processes are about k times faster; with PARDISO (threaded)
each process gets its share of the cores (``threads``).  Processes are started with "spawn":
forking a process whose MKL / OpenMP threads are running can hang the child.

SimPEG has the same idea (``simpeg.meta.MultiprocessingMetaSimulation``); this one starts its
processes with spawn as daemons (a job that ends or fails does not wait on them), raises a
process's error in the main process (SimPEG's returns it as a value), and limits each
process's threads.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import traceback
import weakref

import numpy as np
import scipy.sparse as sp
from simpeg.meta import MetaSimulation
from simpeg.props import HasModel

_CTX = mp.get_context("spawn")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")


class WorkerError(RuntimeError):
    """An error in a simulation process (its traceback in the message)."""


class _Failed:
    def __init__(self, text):
        self.text = text


class _Worker(_CTX.Process):
    """A process holding one simulation, its model and the fields of that model."""

    def __init__(self):
        super().__init__(daemon=True)
        self.tasks = _CTX.Queue()
        self.results = _CTX.Queue()

    def run(self):     # (in the process)
        sim = fields = None
        while True:
            task = self.tasks.get()
            if task is None:
                break
            op, arg = task
            try:
                if op == "sim":
                    sim, fields, out = pickle.loads(arg), None, None
                elif op == "model":
                    sim.model, fields, out = arg, None, None
                elif op == "fields":
                    fields, out = sim.fields(sim.model), None
                elif op == "dpred":
                    out = sim.dpred(sim.model, f=fields)
                elif op == "jvec":
                    out = sim.Jvec(sim.model, arg, f=fields)
                elif op == "jtvec":
                    out = sim.Jtvec(sim.model, arg, f=fields)
                elif op == "jtj":
                    out = sim.getJtJdiag(sim.model, W=arg, f=fields)
                else:
                    raise ValueError(f"unknown task {op!r}")
            except BaseException:          # noqa: BLE001 — every error goes back to the caller
                out = _Failed(traceback.format_exc())
            self.results.put(out)


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


class ParallelMetaSimulation(MetaSimulation):
    """``simulations`` (their data in this order), each with its mapping from the model, on
    ``n_workers`` processes (contiguous groups of them; default one each), each with
    ``threads`` threads for its solver (default the cores shared out)."""

    def __init__(self, simulations, mappings, n_workers=None, threads=None):
        super().__init__(simulations, mappings)
        n_sim = len(self.simulations)
        n = max(1, min(int(n_workers or n_sim), n_sim))
        groups = [g for g in np.array_split(np.arange(n_sim), n) if g.size]
        self.threads = int(threads or max(1, (os.cpu_count() or 1) // len(groups)))
        saved = {k: os.environ.get(k) for k in THREAD_VARS}
        workers, offsets = [], [0]
        try:
            os.environ.update({k: str(self.threads) for k in THREAD_VARS})    # read by the children
            for g in groups:
                chunk = MetaSimulation([self.simulations[i] for i in g], [self.mappings[i] for i in g])
                # pickled here: a queue's feeder thread would only print the error, and the
                # process would wait for a simulation that never comes
                blob = pickle.dumps(chunk)
                w = _Worker()
                w.start()
                workers.append(w)
                w.tasks.put(("sim", blob))
                offsets.append(offsets[-1] + chunk.survey.nD)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self._workers = workers
        self._offsets = np.asarray(offsets)
        self._finalizer = weakref.finalize(self, _stop, workers)
        self._fields_id = 0         # the model whose fields the processes hold
        self._gather()              # (the simulations arrived)

    @property
    def n_workers(self) -> int:
        return len(self._workers)

    def close(self):
        """Stop the processes (also when the simulation is collected, and at exit)."""
        self._finalizer()

    def _gather(self):
        out = []
        for w in self._workers:
            r = w.results.get()
            out.append(r)
        bad = [r.text for r in out if isinstance(r, _Failed)]
        if bad:
            raise WorkerError("A simulation process failed:\n" + bad[0])
        return out

    def _send(self, op, args):
        for w, a in zip(self._workers, args):
            w.tasks.put((op, a))
        return self._gather()

    @MetaSimulation.model.setter
    def model(self, value):
        # the processes get the model (they map it themselves); none of the local copies is used
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
        """The processes hold the fields of ``m`` (computed again if ``f`` is of another model)."""
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
