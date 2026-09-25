"""Survey / observation data container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SurveyData:
    """Geophysical observation data.

    Attributes:
        locations:  Station positions, shape (n_stations, 3).
        observed:   Observed data vector.
        std:        Standard deviations (uncertainties).
        method:     Geophysical method name (e.g. "gravity", "magnetics").
        components: Component labels (e.g. ["gz"] or ["Zxx", "Zxy"]).
    """

    locations: NDArray[np.float64]
    observed: NDArray[np.float64]
    std: NDArray[np.float64]
    method: str = ""
    components: tuple[str, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        for attr in ("locations", "observed", "std"):
            arr = np.asarray(getattr(self, attr), dtype=np.float64).copy()
            arr.setflags(write=False)
            object.__setattr__(self, attr, arr)

    @property
    def n_stations(self) -> int:
        return self.locations.shape[0]

    @property
    def n_data(self) -> int:
        return len(self.observed)
