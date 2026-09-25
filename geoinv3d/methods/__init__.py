"""Geophysical method wrappers around SimPEG."""

from .base import MethodBase
from .gravity import GravityMethod
from .magnetics import MagneticsMethod
from .dc_resistivity import DCResistivityMethod
from .mt import MTMethod
from .joint import JointInversion

__all__ = [
    "MethodBase", "GravityMethod", "MagneticsMethod",
    "DCResistivityMethod", "MTMethod", "JointInversion",
]
