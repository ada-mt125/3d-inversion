"""Magnetotellurics (MT) method: forward modeling and inversion via SimPEG.

Wraps SimPEG's natural_source module for conductivity -> impedance.
Supports impedance (Zxy, Zyx, etc.) and tipper data.
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

    def __init__(
        self,
        frequencies: list[float] | NDArray | None = None,
        components: list[str] | None = None,
        sigma_background: float = 1e-2,
    ) -> None:
        """
        Args:
            frequencies: MT sounding frequencies in Hz.
            components: Impedance components to compute.
                Each is "{orientation}_{part}", e.g. "xy_real", "xy_imag".
                Defaults to ["xy_real", "xy_imag"].
            sigma_background: Background conductivity (S/m) for
                primary field computation.
        """
        self.frequencies = (
            np.asarray(frequencies) if frequencies is not None
            else np.logspace(-3, 1, 5)
        )
        self.components = components or ["xy_real", "xy_imag"]
        self.sigma_background = sigma_background

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

    def make_simulation(self, mesh: Mesh3D, survey: SurveyData, **kwargs) -> Any:
        from simpeg.electromagnetics.natural_source import (
            Simulation3DPrimarySecondary,
        )
        from simpeg import maps

        dmesh = mesh.to_discretize()
        simpeg_survey = self._build_survey(survey.locations)

        sigma_bg = np.full(dmesh.nC, self.sigma_background)

        return Simulation3DPrimarySecondary(
            mesh=dmesh,
            survey=simpeg_survey,
            sigmaMap=maps.ExpMap(dmesh),
            sigmaPrimary=sigma_bg,
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
        return self.make_simulation(mesh, survey, **kwargs)

    def make_simulation_mapped(
        self, mesh: Mesh3D, survey: SurveyData, mapping: Any, **kwargs
    ) -> Any:
        """Simulation with a custom mapping (for joint inversion)."""
        from simpeg.electromagnetics.natural_source import (
            Simulation3DPrimarySecondary,
        )

        dmesh = mesh.to_discretize()
        simpeg_survey = self._build_survey(survey.locations)
        sigma_bg = np.full(dmesh.nC, self.sigma_background)

        return Simulation3DPrimarySecondary(
            mesh=dmesh,
            survey=simpeg_survey,
            sigmaMap=mapping,
            sigmaPrimary=sigma_bg,
            **kwargs,
        )
