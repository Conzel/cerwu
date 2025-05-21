from typing import Literal
import torch

from data_utils.arrays import strided_index_pairs
import torch
from nn_compression.quantisation._interfaces import Scaling


class AbsMaxScaling(Scaling):
    """
    Scales tensors using absolute maximum values. Maps values to [-nbins//2, nbins//2] range
    by dividing by max(|x|) and multiplying by nbins//2. Supports global or group-wise scaling.
    """

    def find_params(self, x: torch.Tensor) -> None:
        if self._groupsize < 0:
            self.min = self.max = x.abs().max().item()
            self._delta = torch.tensor(
                self.nbins // 2 / self.max
            )  # divide by 2 for symmetry
            self.clamp_ceil = self.nbins // 2
        else:
            maxs = []
            mins = []

            for u, l in strided_index_pairs(x.shape[0], self._groupsize):
                max_group = x[u:l].abs().max().item()
                maxs.append(max_group)
                mins.append(max_group)
            self.max = self.min = torch.tensor(maxs)
            self._delta = self.nbins // 2 / self.max
            self.clamp_ceil = self.nbins // 2

    def scale(self, y: torch.Tensor) -> torch.Tensor:
        """Scales the input tensor to be
            x <- x / max(|x|) * nbins
        Afterwards, |max(x)| == nbins//2

        Handles both global scaling (groupsize < 0) and group-wise scaling (groupsize >= 1)
        """
        if self._groupsize < 0:
            # Global scaling - single delta value for all elements
            return torch.clamp_(self._delta * y, -self.clamp_ceil, self.clamp_ceil)
        else:
            result = torch.zeros_like(y)
            for i, (u, l) in enumerate(
                strided_index_pairs(y.shape[0], self._groupsize)
            ):
                group_delta = self._delta[i]
                result[u:l] = torch.clamp_(
                    group_delta * y[u:l], -self.clamp_ceil, self.clamp_ceil
                )
            return result

    def unscale(self, y: torch.Tensor) -> torch.Tensor:
        """
        (Approximately) undoes the scaling method. Outputs might differ if the input vector
        to scale had values larger than the calculated maximum of the tensor that was
        used to determine the parameters.

        Handles both global scaling (groupsize < 0) and group-wise scaling (groupsize >= 1)
        """
        if self._groupsize < 0:
            return y / self._delta
        else:
            result = torch.zeros_like(y)
            for i, (u, l) in enumerate(
                strided_index_pairs(y.shape[0], self._groupsize)
            ):
                group_delta = self._delta[i]
                result[u:l] = y[u:l] / group_delta
            return result

    @property
    def delta(self) -> torch.Tensor:
        return self._delta
