"""Forward modeling nodes: compute predicted data from a model."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.node import Node, register_node
from ..datamodel.model import PhysicalModel
from ..datamodel.survey import SurveyData


@register_node
class ForwardNode(Node[NDArray]):
    """Run forward modeling for a single geophysical method.

    Inputs: [model_node, survey_node]
    Output: predicted data array

    The method_type parameter selects which SimPEG simulation to use.
    """

    evictable = True

    def __init__(
        self,
        model_node: Node[PhysicalModel],
        survey_node: Node[SurveyData],
        method_type: str = "gravity",
        method_kwargs: dict | None = None,
        name: str = "Forward",
    ) -> None:
        super().__init__(name, inputs=[model_node, survey_node])
        self.method_type = method_type
        self.method_kwargs = method_kwargs or {}

    def _get_method(self):
        from ..methods.gravity import GravityMethod
        from ..methods.magnetics import MagneticsMethod
        from ..methods.dc_resistivity import DCResistivityMethod
        from ..methods.mt import MTMethod

        methods = {
            "gravity": GravityMethod,
            "magnetics": MagneticsMethod,
            "dc_resistivity": DCResistivityMethod,
            "mt": MTMethod,
        }
        cls = methods[self.method_type]
        return cls(**self.method_kwargs)

    def _compute(self, inputs: list[Any]) -> NDArray:
        model: PhysicalModel = inputs[0]
        survey: SurveyData = inputs[1]
        method = self._get_method()
        return method.forward(model, survey)

    def params(self) -> dict:
        return {
            "method_type": self.method_type,
            "method_kwargs": self.method_kwargs,
            "name": self.name,
        }

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> ForwardNode:
        return cls(
            model_node=inputs[0],
            survey_node=inputs[1],
            method_type=params["method_type"],
            method_kwargs=params.get("method_kwargs", {}),
            name=params.get("name", "Forward"),
        )
