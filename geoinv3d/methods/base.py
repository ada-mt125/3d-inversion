"""Abstract base for geophysical methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..datamodel.mesh import Mesh3D
from ..datamodel.model import PhysicalModel
from ..datamodel.survey import SurveyData


class MethodBase(ABC):
    """Base interface for a geophysical forward/inverse method.

    Each method wraps a SimPEG simulation and provides:
      - forward(): predict data from a model
      - sensitivity(): Jacobian or Jacobian-vector product
      - make_simulation(): build the SimPEG simulation object
      - make_survey(): build the SimPEG survey from our SurveyData
    """

    method_name: str = ""

    @abstractmethod
    def make_simulation(self, mesh: Mesh3D, survey: SurveyData, **kwargs) -> Any:
        """Build and return a SimPEG Simulation object."""

    @abstractmethod
    def forward(self, model: PhysicalModel, survey: SurveyData) -> NDArray:
        """Run forward modeling: model -> predicted data."""

    @abstractmethod
    def make_dmis(self, survey: SurveyData, simulation: Any) -> Any:
        """Build a SimPEG DataMisfit object."""

    def make_reg(self, mesh: Mesh3D, **kwargs) -> Any:
        """Build a SimPEG L2 Regularization object."""
        from simpeg import regularization
        dmesh = mesh.to_discretize()
        return regularization.WeightedLeastSquares(dmesh, **kwargs)

    def make_sparse_reg(
        self,
        mesh: Mesh3D,
        norms: tuple[float, ...] = (0.0, 2.0, 2.0, 1.0),
        active_cells: Any = None,
        **kwargs,
    ) -> Any:
        """Build a SimPEG Sparse (IRLS) Regularization object."""
        from simpeg import regularization
        dmesh = mesh.to_discretize()
        reg_kwargs = dict(**kwargs)
        if active_cells is not None:
            reg_kwargs["active_cells"] = active_cells
        return regularization.Sparse(dmesh, norms=list(norms), **reg_kwargs)
