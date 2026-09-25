"""Output nodes: export results to files."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..core.node import Node, register_node
from ..datamodel.model import PhysicalModel
from ..datamodel.result import InversionResult


@register_node
class ModelExportNode(Node[str]):
    """Export a model to a NumPy .npy file.

    Input: [model_node]
    Output: the file path written.
    """

    evictable = False

    def __init__(
        self,
        model_node: Node[PhysicalModel],
        path: str,
        name: str = "Export",
    ) -> None:
        super().__init__(name, inputs=[model_node])
        self.path = path

    def _compute(self, inputs: list[Any]) -> str:
        model: PhysicalModel = inputs[0]
        np.save(self.path, model.values)
        return self.path

    def params(self) -> dict:
        return {"path": self.path, "name": self.name}

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> ModelExportNode:
        return cls(
            model_node=inputs[0],
            path=params["path"],
            name=params.get("name", "Export"),
        )


@register_node
class ResultExportNode(Node[str]):
    """Export inversion result (convergence history) to JSON.

    Input: [inversion_node]
    Output: the file path written.
    """

    evictable = False

    def __init__(
        self,
        inversion_node: Node[InversionResult],
        path: str,
        name: str = "ResultExport",
    ) -> None:
        super().__init__(name, inputs=[inversion_node])
        self.path = path

    def _compute(self, inputs: list[Any]) -> str:
        result: InversionResult = inputs[0]
        doc = {
            "method": result.method,
            "converged": result.converged,
            "n_iterations": result.n_iterations,
            "phi_d": result.phi_d_history.tolist(),
            "phi_m": result.phi_m_history.tolist(),
        }
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        return self.path

    def params(self) -> dict:
        return {"path": self.path, "name": self.name}

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> ResultExportNode:
        return cls(
            inversion_node=inputs[0],
            path=params["path"],
            name=params.get("name", "ResultExport"),
        )
