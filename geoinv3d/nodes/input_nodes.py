"""Input nodes: load models, surveys, and meshes from files or parameters."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.node import Node, register_node
from ..datamodel.mesh import Mesh3D
from ..datamodel.model import PhysicalModel, PhysicalProperty
from ..datamodel.survey import SurveyData


@register_node
class MeshCreateNode(Node[Mesh3D]):
    """Create a uniform 3D mesh from parameters."""

    evictable = False

    def __init__(
        self,
        nx: int = 20, ny: int = 20, nz: int = 10,
        dx: float = 100.0, dy: float = 100.0, dz: float = 50.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: str = "Mesh",
    ) -> None:
        super().__init__(name)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.origin = origin

    def _compute(self, inputs: list[Any]) -> Mesh3D:
        return Mesh3D.uniform(
            self.nx, self.ny, self.nz,
            self.dx, self.dy, self.dz,
            origin=self.origin,
            name=self.name,
        )

    def set_dimensions(self, nx: int, ny: int, nz: int) -> None:
        if (nx, ny, nz) != (self.nx, self.ny, self.nz):
            self.nx, self.ny, self.nz = nx, ny, nz
            self.invalidate()

    def set_spacing(self, dx: float, dy: float, dz: float) -> None:
        if (dx, dy, dz) != (self.dx, self.dy, self.dz):
            self.dx, self.dy, self.dz = dx, dy, dz
            self.invalidate()

    def params(self) -> dict:
        return {
            "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "dx": self.dx, "dy": self.dy, "dz": self.dz,
            "origin": list(self.origin), "name": self.name,
        }

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> MeshCreateNode:
        return cls(
            nx=params["nx"], ny=params["ny"], nz=params["nz"],
            dx=params["dx"], dy=params["dy"], dz=params["dz"],
            origin=tuple(params["origin"]), name=params.get("name", "Mesh"),
        )


@register_node
class ModelCreateNode(Node[PhysicalModel]):
    """Create a constant-value model on a mesh."""

    evictable = False

    def __init__(
        self,
        mesh_node: Node[Mesh3D],
        value: float = 0.0,
        prop: str = "density",
        name: str = "Model",
    ) -> None:
        super().__init__(name, inputs=[mesh_node])
        self.value = value
        self.prop_name = prop

    def _compute(self, inputs: list[Any]) -> PhysicalModel:
        mesh: Mesh3D = inputs[0]
        prop = PhysicalProperty(self.prop_name)
        return PhysicalModel.constant(mesh, self.value, prop, name=self.name)

    def set_value(self, value: float) -> None:
        if value != self.value:
            self.value = value
            self.invalidate()

    def params(self) -> dict:
        return {
            "value": self.value, "prop": self.prop_name, "name": self.name,
        }

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> ModelCreateNode:
        return cls(
            mesh_node=inputs[0],
            value=params["value"],
            prop=params["prop"],
            name=params.get("name", "Model"),
        )


@register_node
class ModelFromArrayNode(Node[PhysicalModel]):
    """Create a model from an explicit values array."""

    evictable = False

    def __init__(
        self,
        mesh_node: Node[Mesh3D],
        values: np.ndarray,
        prop: str = "density",
        name: str = "Model",
    ) -> None:
        super().__init__(name, inputs=[mesh_node])
        self.values = values.copy()
        self.prop_name = prop

    def _compute(self, inputs: list[Any]) -> PhysicalModel:
        mesh: Mesh3D = inputs[0]
        prop = PhysicalProperty(self.prop_name)
        return PhysicalModel(mesh=mesh, values=self.values, prop=prop, name=self.name)

    def params(self) -> dict:
        return {
            "values": self.values.tolist(),
            "prop": self.prop_name,
            "name": self.name,
        }

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> ModelFromArrayNode:
        return cls(
            mesh_node=inputs[0],
            values=np.array(params["values"]),
            prop=params["prop"],
            name=params.get("name", "Model"),
        )


@register_node
class SurveyCreateNode(Node[SurveyData]):
    """Create a survey from station locations and observed data."""

    evictable = False

    def __init__(
        self,
        locations: np.ndarray,
        observed: np.ndarray,
        std: np.ndarray,
        method: str = "",
        name: str = "Survey",
    ) -> None:
        super().__init__(name)
        self.locations = locations.copy()
        self.observed = observed.copy()
        self.std = std.copy()
        self.method = method

    def _compute(self, inputs: list[Any]) -> SurveyData:
        return SurveyData(
            locations=self.locations,
            observed=self.observed,
            std=self.std,
            method=self.method,
            name=self.name,
        )

    def params(self) -> dict:
        return {
            "locations": self.locations.tolist(),
            "observed": self.observed.tolist(),
            "std": self.std.tolist(),
            "method": self.method,
            "name": self.name,
        }

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> SurveyCreateNode:
        return cls(
            locations=np.array(params["locations"]),
            observed=np.array(params["observed"]),
            std=np.array(params["std"]),
            method=params.get("method", ""),
            name=params.get("name", "Survey"),
        )
