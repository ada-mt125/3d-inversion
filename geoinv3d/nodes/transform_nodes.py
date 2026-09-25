"""Transform nodes: model parameter transformations (log, exp, etc.)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.node import Node, register_node
from ..datamodel.model import PhysicalModel, PhysicalProperty


@register_node
class LogTransformNode(Node[PhysicalModel]):
    """Apply log10 transform to model values."""

    evictable = True

    def __init__(self, model_node: Node[PhysicalModel], name: str = "Log10") -> None:
        super().__init__(name, inputs=[model_node])

    def _compute(self, inputs: list[Any]) -> PhysicalModel:
        model: PhysicalModel = inputs[0]
        from dataclasses import replace
        return replace(model, values=np.log10(model.values), name=f"log10({model.name})")

    def params(self) -> dict:
        return {"name": self.name}

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> LogTransformNode:
        return cls(model_node=inputs[0], name=params.get("name", "Log10"))


@register_node
class ScaleNode(Node[PhysicalModel]):
    """Scale model values by a constant factor."""

    evictable = True

    def __init__(
        self, model_node: Node[PhysicalModel], factor: float = 1.0,
        name: str = "Scale",
    ) -> None:
        super().__init__(name, inputs=[model_node])
        self.factor = factor

    def _compute(self, inputs: list[Any]) -> PhysicalModel:
        model: PhysicalModel = inputs[0]
        from dataclasses import replace
        return replace(model, values=model.values * self.factor)

    def set_factor(self, factor: float) -> None:
        if factor != self.factor:
            self.factor = factor
            self.invalidate()

    def params(self) -> dict:
        return {"factor": self.factor, "name": self.name}

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> ScaleNode:
        return cls(
            model_node=inputs[0],
            factor=params["factor"],
            name=params.get("name", "Scale"),
        )


@register_node
class OffsetNode(Node[PhysicalModel]):
    """Add a constant offset to model values."""

    evictable = True

    def __init__(
        self, model_node: Node[PhysicalModel], offset: float = 0.0,
        name: str = "Offset",
    ) -> None:
        super().__init__(name, inputs=[model_node])
        self.offset = offset

    def _compute(self, inputs: list[Any]) -> PhysicalModel:
        model: PhysicalModel = inputs[0]
        from dataclasses import replace
        return replace(model, values=model.values + self.offset)

    def set_offset(self, offset: float) -> None:
        if offset != self.offset:
            self.offset = offset
            self.invalidate()

    def params(self) -> dict:
        return {"offset": self.offset, "name": self.name}

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> OffsetNode:
        return cls(
            model_node=inputs[0],
            offset=params["offset"],
            name=params.get("name", "Offset"),
        )


@register_node
class ConductivityToResistivityNode(Node[PhysicalModel]):
    """Convert conductivity (S/m) to resistivity (Ω·m) and vice versa."""

    evictable = True

    def __init__(self, model_node: Node[PhysicalModel], name: str = "σ↔ρ") -> None:
        super().__init__(name, inputs=[model_node])

    def _compute(self, inputs: list[Any]) -> PhysicalModel:
        model: PhysicalModel = inputs[0]
        new_prop = (
            PhysicalProperty.RESISTIVITY
            if model.prop == PhysicalProperty.CONDUCTIVITY
            else PhysicalProperty.CONDUCTIVITY
        )
        from dataclasses import replace
        return replace(model, values=1.0 / model.values, prop=new_prop)

    def params(self) -> dict:
        return {"name": self.name}

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> ConductivityToResistivityNode:
        return cls(model_node=inputs[0], name=params.get("name", "σ↔ρ"))
