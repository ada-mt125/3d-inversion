"""Gravity method: forward modeling and inversion via SimPEG."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..datamodel.mesh import Mesh3D
from ..datamodel.model import PhysicalModel, PhysicalProperty
from ..datamodel.survey import SurveyData
from .base import MethodBase


class GravityMethod(MethodBase):
    """Scalar gravity (gz) forward and inverse using SimPEG.

    Wraps SimPEG.potential_fields.gravity for density -> gz modeling.
    """

    method_name = "gravity"

    def __init__(self, component: str = "gz") -> None:
        """
        Args:
            component: Receiver component, e.g. "gz" (anomaly) or "gzz" (gradient).
        """
        self.component = component

    def make_simulation(self, mesh: Mesh3D, survey: SurveyData, **kwargs) -> Any:
        from simpeg.potential_fields import gravity
        from simpeg import maps

        dmesh = mesh.to_discretize()
        rx_locs = survey.locations

        rx = gravity.receivers.Point(rx_locs, components=[self.component])
        src_field = gravity.sources.SourceField(receiver_list=[rx])
        simpeg_survey = gravity.survey.Survey(source_field=src_field)

        idenmap = maps.IdentityMap(nP=dmesh.nC)
        sim = gravity.simulation.Simulation3DIntegral(
            survey=simpeg_survey,
            mesh=dmesh,
            rhoMap=idenmap,
            store_sensitivities="forward_only",
            **kwargs,
        )
        return sim

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

    def make_simulation_full(self, mesh: Mesh3D, survey: SurveyData, **kwargs) -> Any:
        """Simulation that stores full sensitivities (for inversion)."""
        from simpeg.potential_fields import gravity
        from simpeg import maps

        dmesh = mesh.to_discretize()
        rx = gravity.receivers.Point(survey.locations, components=[self.component])
        src_field = gravity.sources.SourceField(receiver_list=[rx])
        simpeg_survey = gravity.survey.Survey(source_field=src_field)
        idenmap = maps.IdentityMap(nP=dmesh.nC)

        return gravity.simulation.Simulation3DIntegral(
            survey=simpeg_survey,
            mesh=dmesh,
            rhoMap=idenmap,
            store_sensitivities="ram",
            **kwargs,
        )

    def make_simulation_mapped(
        self, mesh: Mesh3D, survey: SurveyData, mapping: Any, **kwargs
    ) -> Any:
        """Simulation with a custom mapping (for joint inversion)."""
        from simpeg.potential_fields import gravity

        dmesh = mesh.to_discretize()
        rx = gravity.receivers.Point(survey.locations, components=[self.component])
        src_field = gravity.sources.SourceField(receiver_list=[rx])
        simpeg_survey = gravity.survey.Survey(source_field=src_field)

        return gravity.simulation.Simulation3DIntegral(
            survey=simpeg_survey,
            mesh=dmesh,
            rhoMap=mapping,
            store_sensitivities="ram",
            **kwargs,
        )

    def make_simulation_active(
        self,
        mesh: Mesh3D,
        survey: SurveyData,
        active_cells: NDArray,
        **kwargs,
    ) -> Any:
        """Simulation with active cells (for topography support)."""
        from simpeg.potential_fields import gravity
        from simpeg import maps

        dmesh = mesh.to_discretize()
        # The integral simulation works on active cells only, so the model
        # (and therefore the mapping) has one value per active cell.
        act_map = maps.IdentityMap(nP=int(active_cells.sum()))

        rx = gravity.receivers.Point(survey.locations, components=[self.component])
        src_field = gravity.sources.SourceField(receiver_list=[rx])
        simpeg_survey = gravity.survey.Survey(source_field=src_field)

        return gravity.simulation.Simulation3DIntegral(
            survey=simpeg_survey,
            mesh=dmesh,
            rhoMap=act_map,
            active_cells=active_cells,
            store_sensitivities="ram",
            **kwargs,
        )
