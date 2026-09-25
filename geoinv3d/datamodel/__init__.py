"""Immutable data structures for 3D geophysical modeling."""

from .mesh import Mesh3D, DiscretizeMesh
from .model import PhysicalModel
from .survey import SurveyData
from .result import InversionResult, JointInversionResult, IterationSnapshot

__all__ = [
    "Mesh3D", "DiscretizeMesh", "PhysicalModel", "SurveyData",
    "InversionResult", "IterationSnapshot",
]
