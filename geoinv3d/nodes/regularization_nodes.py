"""Regularization configuration nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..core.node import Node, register_node
from ..datamodel.mesh import Mesh3D


@dataclass(frozen=True)
class RegularizationConfig:
    """Regularization parameters, stored as the node's output."""

    alpha_s: float = 1e-4
    alpha_x: float = 1.0
    alpha_y: float = 1.0
    alpha_z: float = 1.0
    reg_type: str = "smoothness"
    ref_model: Optional[np.ndarray] = None


@register_node
class RegularizationNode(Node[RegularizationConfig]):
    """Configure regularization parameters for an inversion.

    Output: RegularizationConfig with all weight parameters.
    """

    evictable = True

    def __init__(
        self,
        alpha_s: float = 1e-4,
        alpha_x: float = 1.0,
        alpha_y: float = 1.0,
        alpha_z: float = 1.0,
        reg_type: str = "smoothness",
        name: str = "Regularization",
    ) -> None:
        super().__init__(name)
        self.alpha_s = alpha_s
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.alpha_z = alpha_z
        self.reg_type = reg_type

    def _compute(self, inputs: list[Any]) -> RegularizationConfig:
        return RegularizationConfig(
            alpha_s=self.alpha_s,
            alpha_x=self.alpha_x,
            alpha_y=self.alpha_y,
            alpha_z=self.alpha_z,
            reg_type=self.reg_type,
        )

    def set_alphas(
        self,
        alpha_s: float | None = None,
        alpha_x: float | None = None,
        alpha_y: float | None = None,
        alpha_z: float | None = None,
    ) -> None:
        changed = False
        if alpha_s is not None and alpha_s != self.alpha_s:
            self.alpha_s = alpha_s
            changed = True
        if alpha_x is not None and alpha_x != self.alpha_x:
            self.alpha_x = alpha_x
            changed = True
        if alpha_y is not None and alpha_y != self.alpha_y:
            self.alpha_y = alpha_y
            changed = True
        if alpha_z is not None and alpha_z != self.alpha_z:
            self.alpha_z = alpha_z
            changed = True
        if changed:
            self.invalidate()

    def params(self) -> dict:
        return {
            "alpha_s": self.alpha_s,
            "alpha_x": self.alpha_x,
            "alpha_y": self.alpha_y,
            "alpha_z": self.alpha_z,
            "reg_type": self.reg_type,
            "name": self.name,
        }

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> RegularizationNode:
        return cls(
            alpha_s=params["alpha_s"],
            alpha_x=params["alpha_x"],
            alpha_y=params["alpha_y"],
            alpha_z=params["alpha_z"],
            reg_type=params.get("reg_type", "smoothness"),
            name=params.get("name", "Regularization"),
        )


@register_node
class CrossGradientNode(Node[dict]):
    """Cross-gradient coupling constraint between two models.

    Output: dict with coupling weight and configuration.
    """

    evictable = True

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "CrossGradient",
    ) -> None:
        super().__init__(name)
        self.weight = weight

    def _compute(self, inputs: list[Any]) -> dict:
        return {"type": "cross_gradient", "weight": self.weight}

    def set_weight(self, weight: float) -> None:
        if weight != self.weight:
            self.weight = weight
            self.invalidate()

    def params(self) -> dict:
        return {"weight": self.weight, "name": self.name}

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> CrossGradientNode:
        return cls(weight=params["weight"], name=params.get("name", "CrossGradient"))
