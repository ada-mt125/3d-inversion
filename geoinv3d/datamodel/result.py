"""Inversion result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .model import PhysicalModel


@dataclass(frozen=True)
class IterationSnapshot:
    """One iteration's state, stored as a DAG node output."""

    iteration: int
    model_values: NDArray[np.float64]
    phi_d: float
    phi_m: float
    phi_total: float
    beta: float

    def __post_init__(self) -> None:
        v = np.asarray(self.model_values, dtype=np.float64).copy()
        v.setflags(write=False)
        object.__setattr__(self, "model_values", v)


@dataclass
class InversionResult:
    """Complete result of an inversion run.

    Mutable because iterations are appended during the run, but each
    IterationSnapshot is frozen.
    """

    method: str
    iterations: list[IterationSnapshot] = field(default_factory=list)
    converged: bool = False
    final_model: Optional[PhysicalModel] = None

    def add_iteration(self, snap: IterationSnapshot) -> None:
        self.iterations.append(snap)

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    @property
    def phi_d_history(self) -> NDArray[np.float64]:
        return np.array([s.phi_d for s in self.iterations])

    @property
    def phi_m_history(self) -> NDArray[np.float64]:
        return np.array([s.phi_m for s in self.iterations])

    @property
    def beta_history(self) -> NDArray[np.float64]:
        return np.array([s.beta for s in self.iterations])


@dataclass
class JointInversionResult:
    """Result of a joint inversion with multiple methods.

    Holds the combined convergence history plus per-method
    recovered models and individual misfits.
    """

    methods: list[str]
    iterations: list[IterationSnapshot] = field(default_factory=list)
    converged: bool = False
    per_method_phi_d: dict[str, list[float]] = field(default_factory=dict)
    recovered_models: dict[str, NDArray] = field(default_factory=dict)
    weights: list[float] = field(default_factory=list)

    def add_iteration(self, snap: IterationSnapshot) -> None:
        self.iterations.append(snap)

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    @property
    def phi_d_history(self) -> NDArray[np.float64]:
        return np.array([s.phi_d for s in self.iterations])

    @property
    def phi_m_history(self) -> NDArray[np.float64]:
        return np.array([s.phi_m for s in self.iterations])

    def to_summary(self) -> dict:
        """JSON-serializable summary of the joint inversion."""
        return {
            "methods": self.methods,
            "weights": self.weights,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "final_phi_d": float(self.phi_d_history[-1]) if self.n_iterations else None,
            "final_phi_m": float(self.phi_m_history[-1]) if self.n_iterations else None,
            "per_method_phi_d": {
                m: vals for m, vals in self.per_method_phi_d.items()
            },
        }
