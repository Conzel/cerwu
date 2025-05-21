from pathlib import Path
from typing import Callable, Optional
from nnc.compression import compress, decompress
import numpy as np
import torch
import torch.nn as nn
import bz2

from data_utils.arrays import strided_index_pairs
from nn_compression._interfaces import quantisable
from nn_compression.networks import recursively_find_named_children, TransformWeights


class _WeightToIndex:
    def __init__(self, groupsize: int = -1):
        self.groupsize = int(groupsize)

    def _make_index_single(self, qparam: torch.Tensor):
        """Returns the index of the quantised parameter in a regular grid."""
        eps = 1e-12
        levels = qparam.unique()
        deltas = levels[1:] - levels[:-1]
        significant_deltas = deltas[deltas > eps]  # floating point errors
        # if significant_deltas.numel() <= 1:
        if significant_deltas.numel() == 0:
            return np.zeros_like(qparam.cpu().detach(), dtype=np.int32), 0.0
        step = significant_deltas.min().item()

        indices = torch.round(qparam / step)
        return indices.cpu().detach().numpy().astype(np.int32), step

    def make_index(self, layer: nn.Module):
        tweights = TransformWeights(type(layer))
        transformed = tweights.into_2d(layer.weight)
        if self.groupsize > 0:
            indices = np.zeros_like(transformed.cpu().detach().numpy(), dtype=np.int32)
            steps = []
            for u, l in strided_index_pairs(transformed.shape[0], self.groupsize):
                idx, step = self._make_index_single(transformed[u:l])
                indices[u:l] = idx
                steps.append(step)
            return indices, np.array(steps)
        else:
            return self._make_index_single(transformed)


class DeepCABAC:
    """DeepCABAC-based en/decoder. Uses a state-based entropy coder based on CABAC."""

    def __init__(
        self,
        path: Path | str = "bitstream.nnc",
        verbose: bool = False,
        groupsize: int = -1,
        filter: Optional[Callable] = None,
        transpose: bool = False,
    ) -> None:
        """Instantiates the DeepCABAC-based en/decoder.
        args:
            path: path to save the compressed stream to
            verbose: print progress
            per_row_grid: encode the weights row-wise
            filter: a function f: string -> bool, where only parameters n are encoded for which f(n) is True
            transpose: transpose the indices before encoding, going from row-major scan order (default) to column-major.
        """
        self.verbose = verbose
        self.path = path if isinstance(path, str) else path.as_posix()
        self.compressed_stream = None
        self.nquant = None
        self.unquantised = None
        self.stepsizes = None
        self.groupsize = groupsize
        self.transpose = transpose
        self.filter = filter if filter is not None else lambda _: True

    def encode(self, model: torch.nn.Module):
        """Encodes the model parameters and saves the compressed stream to the path."""
        self.compressed_stream = self.compress(
            self.get_quantised_parameters(model),
            self.path,
            verbose=self.verbose,
            return_bitstream=True,
        )

    @property
    def bytes(self):
        """Returns the number of bytes in the compressed stream."""
        if self.compressed_stream is not None:
            return len(self.compressed_stream)
        else:
            return 0

    @property
    def bpw(self):
        """Returns the number of bits per weight in the compressed stream."""
        if self.compressed_stream is not None:
            assert self.nquant is not None
            return self.bytes * 8 / self.nquant
        else:
            return 0

    def compress(self, *args, **kwargs):
        return compress(*args, **kwargs)

    def get_quantised_parameters(self, model: nn.Module):
        self.nquant = 0
        self.zeropoints = {}
        self.unquantised = {}
        self.steps = {}
        self.quantised = {}

        for n, child in recursively_find_named_children(model):
            if quantisable(child) and self.filter(n):
                self.nquant += child.weight.numel()
                idx, step = _WeightToIndex(groupsize=self.groupsize).make_index(child)
                self.quantised[f"{n}.weight"] = idx.T if self.transpose else idx
                self.steps[f"{n}.weight"] = step
        for n, p in model.state_dict().items():
            if n not in self.quantised:
                self.unquantised[n] = p
        return self.quantised

    def dummy_decode(self):
        _ = decompress(self.path, verbose=self.verbose)

    def decode(self) -> dict:
        """Decodes the compressed stream and returns the reconstructed parameters
        in a state dict. This is not implemented (yet) for row-wise grid encoding."""
        if self.groupsize > 0:
            raise NotImplementedError("Row-wise-grid decoding not implemented.")
        assert (
            self.compressed_stream is not None
        ), "No compressed stream found. Call encode first."
        assert self.unquantised is not None
        decompressed = decompress(self.path, verbose=self.verbose)
        assert isinstance(decompressed, dict)
        quant_params = {
            n: torch.tensor(a) * self.steps[n] for n, a in decompressed.items()
        }
        assert isinstance(quant_params, dict)
        return quant_params | self.unquantised


class Bz2(DeepCABAC):
    def compress(
        self, parameters: dict, path: Path, verbose: bool, return_bitstream: bool
    ):
        """Compress the given parameter dictionary.
        Args:
            parameters: dictionary of parameters to compress, keys are the parameter names,
                values are the indices of the quantised parameters. Values
                must be below 256, as we compress the weights as bytes.
            path: path to save the compressed stream
            verbose: print progress
            return_bitstream: return the compressed stream as bytes"""
        bytestream = []
        for _, idx in parameters.items():
            bytestream += list(idx.flatten().astype(np.uint8))
        compressed = bz2.compress(bytearray(bytestream), compresslevel=9)
        return compressed

    @staticmethod
    def _interleave_indices_array(indices: np.ndarray):
        """Interleaves the indices of a 2D array.
        Example: [-1, -3, 0, 4] -> [2, 6, 0, 7]"""
        if np.abs(indices).max() > 128:
            raise ValueError("Indices must be in the range [-128, 127].")
        indices = indices.flatten()
        neg = indices < 0
        pos = indices > 0
        indices[neg] = -indices[neg] * 2
        indices[pos] = indices[pos] * 2 - 1
        assert np.all(indices >= 0)
        return indices
