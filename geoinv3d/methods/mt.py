"""Magnetotellurics (MT) method: forward modeling and inversion via SimPEG.

Wraps SimPEG's natural_source module for conductivity -> impedance.
The model is the natural log of the conductivity (S/m) on the active cells
(like the DC method, so the two can share one model in a joint inversion).

The inactive cells are the air (``sigma_inactive``, 1e-8 S/m): the mesh must reach well above
the ground, as the fields are solved in the air too.  The primary field (SimPEG's
primary/secondary formulation) is that of a 1D model, air over the background: each of the
mesh's horizontal layers is ground where most of its volume is active, so the interface sits
at the mean ground level and the secondary field accounts for the topography and the 3D model.
With every cell active (no air in the mesh) the primary is a uniform whole space.

Data order (SimPEG's): frequency by frequency, then component by component
(``components``), then station by station (``SurveyData.locations``, (n, 3)); with a ``mask``
(frequency, component, station) only the stations it marks, so stations may miss frequencies.

Components are SimPEG's, in the mesh's frame (x east, y north, z up), as
``"{orientation}_{part}"``: the impedance ``xx``, ``xy``, ``yx``, ``yy`` with ``real``,
``imag``, ``rho`` (apparent resistivity, ohm m) or ``phase`` (degrees), and the tipper ``zx``,
``zy`` with ``real`` or ``imag``.  EDI files use x north, y east: see geoinv3d.io.mt_data.
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
        sigma_inactive: float = 1e-8,
        store_sensitivities: bool = True,
        mask=None,
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
            sigma_inactive: conductivity of the inactive cells, the air above the
                ground (S/m); also the air of the primary model.
            store_sensitivities: keep the sensitivity matrix once computed
                (``storeJ``: fast Jvec / Jtvec, sensitivity weights and the
                group lasso's Gauss–Newton steps; n_data x n_cells floats).
            mask: (frequencies, components, stations) booleans: the data there are; None:
                every station at every frequency and component.
        """
        self.frequencies = (
            np.asarray(frequencies, dtype=float) if frequencies is not None
            else np.logspace(-3, 1, 5)
        )
        self.components = list(components or ["xy_real", "xy_imag"])
        if not sigma_background > 0 or not sigma_inactive > 0:
            raise ValueError("Conductivities must be > 0")
        self.sigma_background = float(sigma_background)
        self.sigma_inactive = float(sigma_inactive)
        self.store_sensitivities = bool(store_sensitivities)
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        if self.mask is not None and self.mask.shape[:2] != (len(self.frequencies), len(self.components)):
            raise ValueError(f"mask has shape {self.mask.shape}, expected ({len(self.frequencies)}, "
                             f"{len(self.components)}, stations)")
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
            o, part = parts
            if o in ("xx", "xy", "yx", "yy"):
                ok = part in ("real", "imag", "rho", "phase")
            elif o in ("zx", "zy"):
                ok = part in ("real", "imag")
            else:
                ok = False
            if not ok:
                raise ValueError(f"Component '{comp}': the impedance (xx, xy, yx, yy) takes real, "
                                 f"imag, rho or phase, the tipper (zx, zy) real or imag")
            parsed.append((o, part))
        return parsed

    @property
    def n_data(self) -> int | None:
        """The number of data with the mask (None without one: it depends on the stations)."""
        return None if self.mask is None else int(self.mask.sum())

    def _build_survey(self, locations: NDArray, sigma_primary: NDArray | None = None):
        """Build a SimPEG natural source survey; ``sigma_primary``: the primary 1D model on
        the mesh's vertical cells (the background, uniform, if None)."""
        from simpeg.electromagnetics.natural_source import (
            receivers, sources, survey,
        )

        parsed = self._parse_components()
        locations = np.atleast_2d(np.asarray(locations, dtype=float))
        mask = self.mask
        if mask is None:
            mask = np.ones((len(self.frequencies), len(parsed), len(locations)), dtype=bool)
        elif mask.shape[2] != len(locations):
            raise ValueError(f"mask is for {mask.shape[2]} stations, the survey has {len(locations)}")

        def receiver(orientation, part, where):
            if orientation in ("zx", "zy"):
                return receivers.Tipper(where, orientation=orientation, component=part)
            component = {"rho": "apparent_resistivity"}.get(part, part)
            return receivers.Impedance(where, orientation=orientation, component=component)

        src_list = []
        sigma_1d = np.full(1, self.sigma_background) if sigma_primary is None else sigma_primary
        for f, freq in enumerate(self.frequencies):
            rx_list = [receiver(o, part, locations[mask[f, c]])
                       for c, (o, part) in enumerate(parsed) if mask[f, c].any()]
            if not rx_list:
                continue
            src_list.append(
                sources.PlanewaveXYPrimary(
                    receiver_list=rx_list,
                    frequency=freq,
                    sigma_primary=sigma_1d,
                )
            )

        return survey.Survey(src_list)

    def primary_1d(self, dmesh, active_cells=None) -> NDArray:
        """The primary model on the mesh's vertical cells (``dmesh.h[-1]``, bottom up): the
        background where most of a layer's volume is active (ground), the air above."""
        hz = np.asarray(dmesh.h[-1], dtype=float)
        sigma = np.full(hz.size, self.sigma_background)
        if active_cells is None:
            return sigma
        active = np.asarray(active_cells, dtype=bool)
        nodes = dmesh.origin[2] + np.r_[0.0, np.cumsum(hz)]
        layer = np.clip(np.searchsorted(nodes, dmesh.cell_centers[:, 2]) - 1, 0, hz.size - 1)
        vol = dmesh.cell_volumes
        total = np.bincount(layer, weights=vol, minlength=hz.size)
        ground = np.bincount(layer, weights=vol * active, minlength=hz.size)
        share = np.divide(ground, total, out=np.zeros_like(total), where=total > 0)
        # a layer no cell centre falls in (inside a coarse cell) takes the one below it
        for k in np.flatnonzero(total == 0):
            share[k] = share[k - 1] if k > 0 else 1.0
        sigma[share < 0.5] = self.sigma_inactive
        return sigma

    def _sigma_map(self, dmesh, mapping=None, active_cells=None):
        """ExpMap of the (injected) log-conductivity: model -> sigma on every cell."""
        from simpeg import maps

        m = maps.ExpMap(dmesh)
        if active_cells is not None:
            m = m * maps.InjectActiveCells(dmesh, np.asarray(active_cells, dtype=bool),
                                           np.log(self.sigma_inactive))
        if mapping is not None:
            m = m * mapping
        return m

    def make_simulation(self, mesh: Mesh3D, survey: SurveyData, mapping=None,
                        active_cells=None, **kwargs) -> Any:
        from simpeg.electromagnetics.natural_source import (
            Simulation3DPrimarySecondary,
        )

        from .solvers import pde_solver

        dmesh = mesh.to_discretize()
        kwargs.setdefault("storeJ", self.store_sensitivities)
        kwargs.setdefault("solver", pde_solver()[0])
        sigma_1d = self.primary_1d(dmesh, active_cells)
        nodes = dmesh.origin[2] + np.r_[0.0, np.cumsum(dmesh.h[-1])]
        layer = np.clip(np.searchsorted(nodes, dmesh.cell_centers[:, 2]) - 1, 0, sigma_1d.size - 1)
        return Simulation3DPrimarySecondary(
            mesh=dmesh,
            survey=self._build_survey(survey.locations, sigma_1d),
            sigmaMap=self._sigma_map(dmesh, mapping, active_cells),
            sigmaPrimary=sigma_1d[layer],
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
