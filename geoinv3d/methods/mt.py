"""Magnetotellurics (MT) method: forward modeling and inversion via SimPEG.

Wraps SimPEG's natural_source module for conductivity -> impedance.
The model is the natural log of the conductivity (S/m) on the active cells
(like the DC method, so the two can share one model in a joint inversion).

Data order (SimPEG's): frequency by frequency, then component by component
(``components``), then station by station (``SurveyData.locations``, (n, 3)).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..datamodel.mesh import Mesh3D
from ..datamodel.model import PhysicalModel
from ..datamodel.survey import SurveyData
from .base import MethodBase


class MTMethod(MethodBase):
    """Magnetotelluric forward and inverse using SimPEG.

    The model parameter is log-conductivity (natural log).
    Uses the primary/secondary field formulation.
    """

    method_name = "mt"
    linear = False

    def __init__(
        self,
        frequencies: list[float] | NDArray | None = None,
        components: list[str] | None = None,
        sigma_background: float = 1e-2,
        sigma_inactive: float | None = None,
        store_sensitivities: bool = True,
    ) -> None:
        """
        Args:
            frequencies: MT sounding frequencies in Hz.
            components: Impedance components to compute.
                Each is "{orientation}_{part}", e.g. "xy_real", "xy_imag".
                Defaults to ["xy_real", "xy_imag"].
            sigma_background: Background conductivity (S/m) for
                primary field computation; its log is the reference and
                default starting model.
            sigma_inactive: conductivity of inactive cells (S/m); None: the
                background (the primary field is that of a uniform
                background, see Simulation3DPrimarySecondary).
            store_sensitivities: keep the sensitivity matrix once computed
                (``storeJ``: fast Jvec / Jtvec, sensitivity weights and the
                group lasso's Gauss–Newton steps; n_data x n_cells floats).
        """
        self.frequencies = (
            np.asarray(frequencies, dtype=float) if frequencies is not None
            else np.logspace(-3, 1, 5)
        )
        self.components = list(components or ["xy_real", "xy_imag"])
        self.sigma_background = float(sigma_background)
        self.sigma_inactive = None if sigma_inactive is None else float(sigma_inactive)
        self.store_sensitivities = bool(store_sensitivities)
        self._parse_components()

    @property
    def default_model_value(self) -> float:
        return float(np.log(self.sigma_background))

    def _parse_components(self) -> list[tuple[str, str]]:
        """Parse component strings into (orientation, part) tuples."""
        parsed = []
        for comp in self.components:
            parts = comp.split("_")
            if len(parts) != 2:
                raise ValueError(
                    f"Component '{comp}' must be 'orientation_part' "
                    f"(e.g. 'xy_real')"
                )
            parsed.append((parts[0], parts[1]))
        return parsed

    def _build_survey(self, locations: NDArray):
        """Build a SimPEG natural source survey."""
        from simpeg.electromagnetics.natural_source import (
            receivers, sources, survey,
        )

        parsed = self._parse_components()

        rx_list = []
        for orientation, part in parsed:
            rx_list.append(
                receivers.Impedance(
                    locations,
                    orientation=orientation,
                    component=part,
                )
            )

        src_list = []
        sigma_1d = np.full(1, self.sigma_background)
        for freq in self.frequencies:
            src_list.append(
                sources.PlanewaveXYPrimary(
                    receiver_list=rx_list,
                    frequency=freq,
                    sigma_primary=sigma_1d,
                )
            )

        return survey.Survey(src_list)

    def _sigma_map(self, dmesh, mapping=None, active_cells=None):
        """ExpMap of the (injected) log-conductivity: model -> sigma on every cell."""
        from simpeg import maps

        m = maps.ExpMap(dmesh)
        if active_cells is not None:
            inactive = self.sigma_background if self.sigma_inactive is None \
                else self.sigma_inactive
            m = m * maps.InjectActiveCells(dmesh, np.asarray(active_cells, dtype=bool),
                                           np.log(inactive))
        if mapping is not None:
            m = m * mapping
        return m

    def make_simulation(self, mesh: Mesh3D, survey: SurveyData, mapping=None,
                        active_cells=None, **kwargs) -> Any:
        from simpeg.electromagnetics.natural_source import (
            Simulation3DPrimarySecondary,
        )

        dmesh = mesh.to_discretize()
        kwargs.setdefault("storeJ", self.store_sensitivities)
        return Simulation3DPrimarySecondary(
            mesh=dmesh,
            survey=self._build_survey(survey.locations),
            sigmaMap=self._sigma_map(dmesh, mapping, active_cells),
            sigmaPrimary=np.full(dmesh.nC, self.sigma_background),
            **kwargs,
        )

    def forward(self, model: PhysicalModel, survey: SurveyData) -> NDArray:
        sim = self.make_simulation(model.mesh, survey)
        return sim.dpred(model.values)

    def make_dmis(self, survey: SurveyData, simulation: Any) -> Any:
        from simpeg import data_misfit, data

        simpeg_data = data.Data(
            simulation.survey,
            dobs=survey.observed,
            standard_deviation=survey.std,
        )
        return data_misfit.L2DataMisfit(data=simpeg_data, simulation=simulation)

    def make_simulation_full(
        self, mesh: Mesh3D, survey: SurveyData, **kwargs
    ) -> Any:
        """Model: log-conductivity on every cell."""
        return self.make_simulation(mesh, survey, **kwargs)

    def make_simulation_active(
        self, mesh: Mesh3D, survey: SurveyData, active_cells: NDArray, **kwargs
    ) -> Any:
        """Model: log-conductivity on the active cells."""
        return self.make_simulation(mesh, survey, active_cells=active_cells, **kwargs)

    def make_simulation_mapped(
        self, mesh: Mesh3D, survey: SurveyData, mapping: Any,
        active_cells: NDArray | None = None, **kwargs
    ) -> Any:
        """Simulation of a slice of a joint model (``mapping``: joint -> active cells).

        The slice is log-conductivity, as for the other simulations.
        """
        return self.make_simulation(mesh, survey, mapping=mapping, active_cells=active_cells,
                                    **kwargs)
