"""DC Resistivity method: forward modeling and inversion via SimPEG."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..datamodel.mesh import Mesh3D
from ..datamodel.model import PhysicalModel
from ..datamodel.survey import SurveyData
from .base import MethodBase


class DCResistivityMethod(MethodBase):
    """DC resistivity forward and inverse using SimPEG.

    Wraps SimPEG.electromagnetics.static.resistivity for
    conductivity -> apparent resistivity.
    """

    method_name = "dc_resistivity"

    def make_simulation(self, mesh: Mesh3D, survey: SurveyData, **kwargs) -> Any:
        from simpeg.electromagnetics.static import resistivity as dc
        from simpeg import maps

        dmesh = mesh.to_discretize()

        # For DC, the survey setup is more complex — electrode arrays.
        # This is a simplified skeleton; real usage will build from
        # electrode positions in survey.locations.
        idenmap = maps.IdentityMap(nP=dmesh.nC)

        # Placeholder: actual electrode/survey setup requires
        # source-receiver pairs which we'll build from SurveyData
        sim = dc.simulation.Simulation3DNodal(
            mesh=dmesh,
            sigmaMap=idenmap,
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

    def make_survey_from_electrodes(
        self,
        a_locs: NDArray,
        b_locs: NDArray,
        m_locs: NDArray,
        n_locs: NDArray,
    ) -> Any:
        """Build a SimPEG DC survey from electrode positions.

        Args:
            a_locs: Current electrode A positions (n, 3)
            b_locs: Current electrode B positions (n, 3)
            m_locs: Potential electrode M positions (n, 3)
            n_locs: Potential electrode N positions (n, 3)
        """
        from simpeg.electromagnetics.static import resistivity as dc

        src_list = []
        for i in range(len(a_locs)):
            rx = dc.receivers.Dipole(
                locations_m=m_locs[i:i+1],
                locations_n=n_locs[i:i+1],
            )
            src = dc.sources.Dipole(
                receiver_list=[rx],
                location_a=a_locs[i],
                location_b=b_locs[i],
            )
            src_list.append(src)
        return dc.survey.Survey(src_list)
