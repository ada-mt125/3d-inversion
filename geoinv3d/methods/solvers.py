"""The sparse linear solver of the methods that solve a PDE (MT, DC resistivity).

Each MT frequency (and each DC source configuration) is one large sparse system, factorized
once and solved for every right-hand side; the solver sets how large a model can be and how
long it takes.  ``pde_solver`` returns the fastest one installed, in pymatsolver's terms:

    Pardiso   Intel MKL's PARDISO (``pydiso``): x86 Linux / Windows / Intel Macs, conda-forge
    Mumps     MUMPS (``python-mumps``): every platform, Apple silicon included, conda-forge
    SolverLU  SciPy's SuperLU: always there, but many times slower and larger on 3D meshes

``GEOINV3D_SOLVER`` (Pardiso, Mumps or SolverLU) picks one by name.
"""

from __future__ import annotations

import os

SLOW = "SolverLU"
SLOW_NOTE = ("The sparse solver is SciPy's SuperLU, as neither PARDISO (pydiso) nor MUMPS "
             "(python-mumps) is installed: fine for small meshes, slow and memory-hungry for "
             "large ones")


def pde_solver():
    """(solver class, name): the fastest pymatsolver solver available, or the one asked for."""
    import pymatsolver
    from pymatsolver import AvailableSolvers

    asked = os.environ.get("GEOINV3D_SOLVER")
    if asked:
        if asked not in ("Pardiso", "Mumps", SLOW):
            raise ValueError(f"GEOINV3D_SOLVER must be Pardiso, Mumps or {SLOW}, got {asked!r}")
        if asked != SLOW and not AvailableSolvers.get(asked):
            raise ValueError(f"GEOINV3D_SOLVER={asked}, but it is not installed")
        return _picklable(getattr(pymatsolver, asked)), asked
    for name in ("Pardiso", "Mumps"):
        if AvailableSolvers.get(name):
            return getattr(pymatsolver, name), name
    return _picklable(pymatsolver.SolverLU), SLOW


def _picklable(cls):
    """pymatsolver makes SolverLU at run time (its module says "abc"): pickle looks a class up
    by its module and name, so a simulation holding it could not go to another process
    (geoinv3d.methods.parallel).  It is pymatsolver.SolverLU."""
    if getattr(cls, "__module__", None) == "abc":
        cls.__module__, cls.__qualname__ = "pymatsolver", cls.__name__
    return cls
