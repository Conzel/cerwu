from dataclasses import dataclass
from typing import Optional
import torch
from abc import ABC, abstractmethod


class Scaling(ABC):
    """Base class for a weight scaling class."""

    def __init__(
        self, nbins: int, x: Optional[torch.Tensor] = None, groupsize: int = -1
    ) -> None:
        """Initializes a scaler that can prepare weights for quantization to nbins. x might be passed to
        initialize the scaler with appropriate values (weight range etc.)."""
        super().__init__()
        self._nbins = nbins
        self._groupsize = groupsize
        if x is not None:
            self.find_params(x)

    @property
    def delta(self) -> torch.Tensor:
        """Returns the step size of the scaling."""
        raise NotImplementedError

    @property
    def nbins(self) -> int:
        """Returns the number of bins."""
        return self._nbins

    @abstractmethod
    def find_params(self, x: torch.Tensor) -> None:
        """Adapts the scaling to the statistics of the given weight vector."""
        raise NotImplementedError

    @abstractmethod
    def scale(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def unscale(self, y: torch.Tensor) -> torch.Tensor:
        """Undoes the scaling done by the scale method. This is not guaranteed to be an exact inverse,
        as f.e. clipping might occur during the scaling. If y is the same vector that find_params
        was called on, then this must be an exact inverse:

            s = Scale(...)
            s.find_params(y)
            assert (y == s.unscale(s.scale(y))).all()

        Swapping the operations does NOT guarantee equality of the operation.
        """
        raise NotImplementedError
