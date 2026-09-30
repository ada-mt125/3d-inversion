"""Magnetics method: forward modeling and inversion via SimPEG."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..datamodel.mesh import Mesh3D
from ..datamodel.model import PhysicalModel
from ..datamodel.survey import SurveyData
from .base import MethodBase


MAGNETIZATIONS = ("induced", "vector")


class MagneticsMethod(MethodBase):
    """Total magnetic intensity (TMI) forward and inverse using SimPEG.

    Wraps SimPEG.potential_fields.magnetics for susceptibility -> TMI, or, with
    ``magnetization="vector"``, for a magnetization vector per cell (SimPEG's
    Cartesian MVI: the model is [m_x | m_y | m_z], effective susceptibility along
    east, north and up), which also holds remanent magnetization.
    """

    method_name = "magnetics"

    def __init__(
        self,
        inducing_field: tuple[float, float, float] = (50000.0, 90.0, 0.0),
        component: str = "tmi",
        magnetization: str = "induced",
    ) -> None:
        """
        Args:
            inducing_field: (amplitude_nT, inclination_deg, declination_deg)
            component: Receiver component, e.g. "tmi" or "bz".
            magnetization: "induced" (a susceptibility along the present field) or
                "vector" (three components per cell; single inversions only).
        """
        if magnetization not in MAGNETIZATIONS:
            raise ValueError(f"Unknown magnetization '{magnetization}' "
                             f"(expected one of {MAGNETIZATIONS})")
        self.inducing_field = inducing_field
        self.component = component
        self.magnetization = magnetization

    @property
    def vector(self) -> bool:
        return self.magnetization == "vector"

    @property
    def n_components(self) -> int:
        return 3 if self.vector else 1

    def _model_type(self) -> dict:
        return {"model_type": "vector"} if self.vector else {}

    def make_simulation(self, mesh: Mesh3D, survey: SurveyData, **kwargs) -> Any:
        from simpeg.potential_fields import magnetics
        from simpeg import maps

        dmesh = mesh.to_discretize()

        rx = magnetics.receivers.Point(survey.locations, components=[self.component])
        src_field = magnetics.sources.UniformBackgroundField(
            receiver_list=[rx],
            amplitude=self.inducing_field[0],
            inclination=self.inducing_field[1],
            declination=self.inducing_field[2],
        )
        simpeg_survey = magnetics.survey.Survey(source_field=src_field)

        idenmap = maps.IdentityMap(nP=self.n_components * dmesh.nC)
        return magnetics.simulation.Simulation3DIntegral(
            survey=simpeg_survey,
            mesh=dmesh,
            chiMap=idenmap,
            store_sensitivities="forward_only",
            **self._model_type(),
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

    def make_simulation_full(self, mesh: Mesh3D, survey: SurveyData, **kwargs) -> Any:
        from simpeg.potential_fields import magnetics
        from simpeg import maps

        dmesh = mesh.to_discretize()
        rx = magnetics.receivers.Point(survey.locations, components=[self.component])
        src_field = magnetics.sources.UniformBackgroundField(
            receiver_list=[rx],
            amplitude=self.inducing_field[0],
            inclination=self.inducing_field[1],
            declination=self.inducing_field[2],
        )
        simpeg_survey = magnetics.survey.Survey(source_field=src_field)
        idenmap = maps.IdentityMap(nP=self.n_components * dmesh.nC)

        return magnetics.simulation.Simulation3DIntegral(
            survey=simpeg_survey,
            mesh=dmesh,
            chiMap=idenmap,
            store_sensitivities="ram",
            **self._model_type(),
            **kwargs,
        )

    def make_simulation_mapped(
        self, mesh: Mesh3D, survey: SurveyData, mapping: Any, **kwargs
    ) -> Any:
        """Simulation with a custom mapping (for joint inversion)."""
        from simpeg.potential_fields import magnetics

        if self.vector:
            raise ValueError("The magnetization-vector inversion (MVI) is for single magnetic "
                             "inversions in this version; joint inversions use the induced "
                             "susceptibility")

        dmesh = mesh.to_discretize()
        rx = magnetics.receivers.Point(survey.locations, components=[self.component])
        src_field = magnetics.sources.UniformBackgroundField(
            receiver_list=[rx],
            amplitude=self.inducing_field[0],
            inclination=self.inducing_field[1],
            declination=self.inducing_field[2],
        )
        simpeg_survey = magnetics.survey.Survey(source_field=src_field)

        return magnetics.simulation.Simulation3DIntegral(
            survey=simpeg_survey,
            mesh=dmesh,
            chiMap=mapping,
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
        from simpeg.potential_fields import magnetics
        from simpeg import maps

        dmesh = mesh.to_discretize()
        # The integral simulation works on active cells only, so the model
        # (and therefore the mapping) has one value per active cell.
        act_map = maps.IdentityMap(nP=self.n_components * int(active_cells.sum()))

        rx = magnetics.receivers.Point(survey.locations, components=[self.component])
        src_field = magnetics.sources.UniformBackgroundField(
            receiver_list=[rx],
            amplitude=self.inducing_field[0],
            inclination=self.inducing_field[1],
            declination=self.inducing_field[2],
        )
        simpeg_survey = magnetics.survey.Survey(source_field=src_field)

        return magnetics.simulation.Simulation3DIntegral(
            survey=simpeg_survey,
            mesh=dmesh,
            chiMap=act_map,
            active_cells=active_cells,
            store_sensitivities="ram",
            **self._model_type(),
            **kwargs,
        )
