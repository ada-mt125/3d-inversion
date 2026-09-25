"""Physical property model on a 3D mesh."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .mesh import Mesh3D


class PhysicalProperty(Enum):
    """Supported physical properties."""
    DENSITY = "density"
    SUSCEPTIBILITY = "susceptibility"
    CONDUCTIVITY = "conductivity"
    RESISTIVITY = "resistivity"
    VELOCITY_P = "vp"
    VELOCITY_S = "vs"
    SLOWNESS = "slowness"


PROPERTY_UNITS: dict[PhysicalProperty, str] = {
    PhysicalProperty.DENSITY: "g/cm³",
    PhysicalProperty.SUSCEPTIBILITY: "SI",
    PhysicalProperty.CONDUCTIVITY: "S/m",
    PhysicalProperty.RESISTIVITY: "Ω·m",
    PhysicalProperty.VELOCITY_P: "m/s",
    PhysicalProperty.VELOCITY_S: "m/s",
    PhysicalProperty.SLOWNESS: "s/m",
}


@dataclass(frozen=True)
class PhysicalModel:
    """A physical property distribution on a 3D mesh.

    Attributes:
        mesh:     The underlying 3D mesh.
        values:   Flat array of property values, one per cell.
        prop:     Which physical property this model represents.
        name:     Display name.
    """

    mesh: Mesh3D
    values: NDArray[np.float64]
    prop: PhysicalProperty
    name: str = ""

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64).ravel()
        if len(v) != self.mesh.n_cells:
            raise ValueError(
                f"values length {len(v)} != mesh n_cells {self.mesh.n_cells}"
            )
        v = v.copy()
        v.setflags(write=False)
        object.__setattr__(self, "values", v)

    @property
    def units(self) -> str:
        return PROPERTY_UNITS.get(self.prop, "")

    def values_3d(self) -> NDArray[np.float64]:
        """Reshape values to (nx, ny, nz)."""
        return self.values.reshape(self.mesh.shape)

    @classmethod
    def constant(
        cls,
        mesh: Mesh3D,
        value: float,
        prop: PhysicalProperty,
        name: str = "",
    ) -> PhysicalModel:
        """Create a homogeneous model."""
        return cls(
            mesh=mesh,
            values=np.full(mesh.n_cells, value),
            prop=prop,
            name=name,
        )
