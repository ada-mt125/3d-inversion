"""DC resistivity method: forward modeling and inversion via SimPEG.

The model is the natural log of the conductivity (S/m) on the active cells,
like the MT method, so the two can share one model in a joint inversion.

Electrodes
----------
``SurveyData.locations`` holds one row of electrode positions per datum,
shape (n_data, 12): A, B, M and N as (x, y, z) each.  A NaN B makes a pole
source, a NaN N a pole receiver (4 columns of NaNs are allowed for either).
Consecutive rows with the same A and B form one SimPEG source, so data
listed source by source need one field solution per source, and the data
keep their order whatever it is.  ``observed`` are potential differences
(V, ``data_type="volt"``, for a unit current) or apparent resistivities
(ohm m, ``data_type="apparent_resistivity"``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..datamodel.mesh import Mesh3D
from ..datamodel.model import PhysicalModel
from ..datamodel.survey import SurveyData
from .base import MethodBase
from .solvers import pde_solver

DC_DATA_TYPES = ("volt", "apparent_resistivity")


def electrode_rows(a, b=None, m=None, n=None) -> NDArray:
    """Electrode positions (n, 3) each -> the (n, 12) ``SurveyData.locations`` of DC.

    ``b`` or ``n`` None: pole source / pole receiver (NaNs).
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    nan = np.full_like(a, np.nan)
    parts = [a] + [nan if x is None else np.atleast_2d(np.asarray(x, dtype=float))
                   for x in (b, m, n)]
    if any(p.shape != a.shape or p.shape[1] != 3 for p in parts):
        raise ValueError("A, B, M and N need the same number of (x, y, z) rows")
    return np.hstack(parts)


class DCResistivityMethod(MethodBase):
    """3D DC resistivity (SimPEG ``Simulation3DNodal``): log-conductivity -> data.

    Args:
        sigma_background: background conductivity (S/m); the reference and
            default starting model is its log.
        data_type: "volt" or "apparent_resistivity".
        sigma_inactive: conductivity of inactive cells (air above
            topography), S/m.
        store_sensitivities: keep the sensitivity matrix (fast Jvec/Jtvec and
            sensitivity weights; n_data x n_cells floats).
    """

    method_name = "dc_resistivity"
    linear = False

    def __init__(self, sigma_background: float = 1e-2, data_type: str = "volt",
                 sigma_inactive: float = 1e-8, store_sensitivities: bool = True) -> None:
        if not sigma_background > 0 or not sigma_inactive > 0:
            raise ValueError("Conductivities must be > 0")
        if data_type not in DC_DATA_TYPES:
            raise ValueError(f"data_type must be one of {DC_DATA_TYPES}, got {data_type!r}")
        self.sigma_background = float(sigma_background)
        self.data_type = data_type
        self.sigma_inactive = float(sigma_inactive)
        self.store_sensitivities = bool(store_sensitivities)

    @property
    def default_model_value(self) -> float:
        return float(np.log(self.sigma_background))

    # -- survey ------------------------------------------------------------

    def make_survey(self, survey: SurveyData):
        """The SimPEG DC survey of ``survey.locations`` (see the module docstring)."""
        from simpeg.electromagnetics.static import resistivity as dc

        rows = np.asarray(survey.locations, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != 12:
            raise ValueError("DC resistivity locations must be (n_data, 12): A, B, M, N as "
                             f"(x, y, z) each; got shape {rows.shape}")
        if len(rows) != len(survey.observed):
            raise ValueError(f"{len(rows)} electrode rows for {len(survey.observed)} data")
        if np.isnan(rows[:, 0:3]).any() or np.isnan(rows[:, 6:9]).any():
            raise ValueError("Every datum needs an A and an M electrode")
        # consecutive rows with the same source form one SimPEG source
        src_key = np.nan_to_num(rows[:, 0:6], nan=np.inf)
        starts = np.flatnonzero(np.r_[True, np.any(src_key[1:] != src_key[:-1], axis=1)])
        ends = np.r_[starts[1:], len(rows)]
        sources = []
        for i0, i1 in zip(starts, ends):
            block = rows[i0:i1]
            pole_rx = np.isnan(block[:, 9:12]).any(axis=1)
            if pole_rx.any() and not pole_rx.all():
                raise ValueError("Pole and dipole receivers of one source must be listed as "
                                 "separate sources (split the rows)")
            if pole_rx.all():
                rx = dc.receivers.Pole(block[:, 6:9], data_type=self.data_type)
            else:
                rx = dc.receivers.Dipole(locations_m=block[:, 6:9], locations_n=block[:, 9:12],
                                         data_type=self.data_type)
            a, b = block[0, 0:3], block[0, 3:6]
            if np.isnan(b).any():
                sources.append(dc.sources.Pole([rx], location=a))
            else:
                sources.append(dc.sources.Dipole([rx], location_a=a, location_b=b))
        simpeg_survey = dc.survey.Survey(sources)
        if self.data_type == "apparent_resistivity":
            simpeg_survey.set_geometric_factor()
        return simpeg_survey

    def make_survey_from_electrodes(self, a_locs: NDArray, b_locs: NDArray,
                                    m_locs: NDArray, n_locs: NDArray) -> Any:
        """A SimPEG DC survey from electrode positions (n, 3) each."""
        rows = electrode_rows(a_locs, b_locs, m_locs, n_locs)
        return self.make_survey(SurveyData(locations=rows, observed=np.zeros(len(rows)),
                                           std=np.ones(len(rows))))

    # -- simulations -------------------------------------------------------

    def _sigma_map(self, dmesh, mapping=None, active_cells=None):
        """ExpMap of the (injected) log-conductivity: model -> sigma on every cell."""
        from simpeg import maps

        m = maps.ExpMap(dmesh)
        if active_cells is not None:
            active = np.asarray(active_cells, dtype=bool)
            m = m * maps.InjectActiveCells(dmesh, active, np.log(self.sigma_inactive))
        if mapping is not None:
            m = m * mapping
        return m

    def make_simulation(self, mesh: Mesh3D, survey: SurveyData, mapping=None,
                        active_cells=None, **kwargs) -> Any:
        from simpeg.electromagnetics.static import resistivity as dc

        dmesh = mesh.to_discretize()
        kwargs.setdefault("storeJ", self.store_sensitivities)
        kwargs.setdefault("solver", pde_solver()[0])
        return dc.simulation.Simulation3DNodal(
            mesh=dmesh, survey=self.make_survey(survey),
            sigmaMap=self._sigma_map(dmesh, mapping, active_cells), **kwargs)

    def make_simulation_full(self, mesh: Mesh3D, survey: SurveyData, **kwargs) -> Any:
        """Model: log-conductivity on every cell."""
        return self.make_simulation(mesh, survey, **kwargs)

    def make_simulation_active(self, mesh: Mesh3D, survey: SurveyData,
                               active_cells: NDArray, **kwargs) -> Any:
        """Model: log-conductivity on the active cells; the others are ``sigma_inactive``."""
        return self.make_simulation(mesh, survey, active_cells=active_cells, **kwargs)

    def make_simulation_mapped(self, mesh: Mesh3D, survey: SurveyData, mapping: Any,
                               active_cells: NDArray | None = None, **kwargs) -> Any:
        """Simulation of a slice of a joint model (``mapping``: joint -> active cells)."""
        return self.make_simulation(mesh, survey, mapping=mapping, active_cells=active_cells,
                                    **kwargs)

    def forward(self, model: PhysicalModel, survey: SurveyData) -> NDArray:
        """Data of a log-conductivity model on every cell."""
        return self.make_simulation(model.mesh, survey).dpred(model.values)

    def make_dmis(self, survey: SurveyData, simulation: Any) -> Any:
        from simpeg import data_misfit, data

        simpeg_data = data.Data(simulation.survey, dobs=survey.observed,
                                standard_deviation=survey.std)
        return data_misfit.L2DataMisfit(data=simpeg_data, simulation=simulation)
