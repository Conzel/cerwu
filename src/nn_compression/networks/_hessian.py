from pathlib import Path
import torch.nn as nn
import numpy as np
import math
import torch
from transformers.pytorch_utils import Conv1D

from nn_compression._interfaces import quantisable
from ._utils import recursively_find_named_children
from typing import Optional
import copy


def track_hessians(net, is_large_net: bool = False):
    def record_hessian(module, input, output):
        if not quantisable(module):
            raise ValueError(
                f"This layer should be quantisable. Internal error: {module}"
            )

        if not hasattr(module, "hessian"):
            module.hessian = LayerWiseHessian(module)
        with torch.no_grad():
            module.hessian.add_batch(
                input[0], output
            )  # input is list with args in order

    handles = []
    hessians = {}
    for n, layer in recursively_find_named_children(net):
        if quantisable(layer):
            handles.append(layer.register_forward_hook(record_hessian))
            if not hasattr(layer, "hessian"):
                layer.hessian = LayerWiseHessian(layer, is_large_net=is_large_net)  # type: ignore
            hessians[n] = layer.hessian

    net._handles_hessian = handles
    return hessians


def untrack_hessians(net):
    for handle in net._handles_hessian:
        handle.remove()
    for n, layer in recursively_find_named_children(net):
        if quantisable(layer):
            if not hasattr(layer, "hessian"):
                print(
                    f"WARNING: Hessian not computed for layer {n}. This is likely because the layer was never called during the forward pass. This layer will not be quantised."
                )
                layer.quantisable = False  # type: ignore
                continue
            layer.hessian.precalculate()
    del net._handles_hessian


def estimate_hessians(
    net,
    dataloader,
    nbatches: int,
    save_to: Optional[Path] = None,
    device=torch.device("cpu"),
):
    """Estimates Hessians on a neural network using the dataloader."""
    net.to(device)
    i = 0
    with LayerWiseHessianTracker(net, save_to=save_to):
        for x, _ in dataloader:
            x = x.to(device)
            if i >= nbatches:
                break
            i += 1
            net(x)
    net.to("cpu")


class LayerWiseHessianTracker:
    """Context-Object to track the hessians of a neural network.

    Use as follows (assume that x is a calibration sample):

    ```
    with LayerWiseHessianTracker(...):
        net(x)
    print(x.hessian)
    ```
    afterwards, each layer in the network net contains a LayerWiseHessian object,
    accessible over the `hessian` attribute
    """

    def __init__(
        self,
        net: nn.Module,
        save_to: Optional[Path] = None,
        is_large_net: bool = False,
    ) -> None:
        """Initialize the tracker for the given network."""
        self.net = net
        if isinstance(save_to, str):
            save_to = Path(save_to)
        self.save_to = save_to
        self.is_large_net = is_large_net

    def __enter__(self):
        self.hessians = track_hessians(self.net, is_large_net=self.is_large_net)

    def __exit__(self, *args):
        untrack_hessians(self.net)
        if self.save_to is not None:
            layer_pointers = {}
            # We need to remove the layer pointers, else the Hessian unnecessarily
            # saves the whole weight vectors
            for k, v in self.hessians.items():
                layer_pointers[k] = v.layer
                v.layer = None
            self.save_to.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.hessians, self.save_to)
            # reattach the layer pointers
            for k, v in self.hessians.items():
                v.layer = layer_pointers[k]


class TransformWeights:
    """Transforms weights between the shape used in GPTQ (2D, with (out, in)) and the original
    shape used in PyTorch modules."""

    def __init__(self, kind: type):
        self.kind = kind

    def into_2d(self, W: torch.Tensor):
        """Transforms given weights to a form that can be used in the GPTQ quantisation method."""
        if issubclass(self.kind, nn.Linear):
            return W
        elif issubclass(self.kind, nn.Conv2d):
            return W.flatten(1)
        elif issubclass(self.kind, Conv1D):
            return W.t()
        else:
            raise ValueError("Only nn.Conv2d and nn.Linear layers are supported")

    def from_2d(self, W: torch.Tensor, orig_shape: torch.Size | list[int]):
        """Transforms the weights back to the original shape."""
        if issubclass(self.kind, nn.Conv2d):
            W = W.reshape(orig_shape)
        elif issubclass(self.kind, nn.Linear):
            W = W
        elif issubclass(self.kind, Conv1D):
            W = W.t()
        return W


class LayerWiseHessian:
    """A representation of the layer-wise Hessian. This class is used to estimate the Hessian matrix of a layer.
    This is useful to encompass different types of layers, such as linear and convolutional layers and treat them the same way.
    This class provides both the Hessian and the Cholesky Decomposition of the inverse of the Hessian, which is used in the GPTQ quantisation method.
    """

    depthwise_warning_printed = False

    def __init__(
        self,
        layer: nn.Module,
        percdamp: float = 0.01,
        is_large_net: bool = False,
    ) -> None:
        """Initialise the LayerWiseHessian for a layer of a neural network.

        Args:
            layer: The layer for which the Hessian should be estimated.
            percdamp: Dampening when calculating the inverse.
            is_large_net: Performs calculations in a memory-sensitive manner.
        """
        self.layer = layer
        if isinstance(self.layer, nn.Conv2d):
            self.type = "conv"
        elif isinstance(self.layer, nn.Linear):
            self.type = "linear"
        elif isinstance(self.layer, Conv1D):
            self.type = "conv1d-t"
        else:
            raise ValueError("Only nn.Conv2d and nn.Linear layers are supported")

        self.is_large_net = is_large_net

        self.device = layer.weight.device if not is_large_net else "cpu"
        self.chol_device = layer.weight.device

        self.depthwise_separable = is_depthwise_separable(layer)
        if self.depthwise_separable and not self.depthwise_warning_printed:

            print(
                "Warning: Depthwise separable layer detected. This might result in large Hessians."
            )
            self.depthwise_warning_printed = True
            # self.columns = self.columns * self.layer.groups
        self.columns = self.get_weight().shape[1]
        self.hsize = self.columns
        if self.depthwise_separable:
            self.hsize *= self.layer.groups
        self._H = torch.zeros((self.hsize, self.hsize), device=self.device)
        self._chol_inv = None
        self.nsamples = 0
        self.percdamp = percdamp

    def to(self, device):
        """Moves hessian to device."""
        self.device = device
        self._H = self._H.to(device)
        if self._chol_inv is not None:
            self._chol_inv = self._chol_inv.to(device)
        return self

    def deepcopy(self, layer):
        hessian = LayerWiseHessian(layer, self.percdamp)
        hessian._H = copy.deepcopy(self._H.detach())
        if self._chol_inv is None:
            self.precalculate()
        assert self._chol_inv is not None
        hessian._chol_inv = copy.deepcopy(self._chol_inv.detach())
        return hessian

    @staticmethod
    def load_into_model(
        model: nn.Module, hessians: str | Path | dict[str, "LayerWiseHessian"]
    ):
        """Loads all hessians into the given model."""
        if isinstance(hessians, str):
            hessians = Path(hessians)
        if isinstance(hessians, Path):
            hessians = torch.load(hessians, next(model.parameters()).device)
        for n, layer in recursively_find_named_children(model):
            if quantisable(layer):
                if hasattr(layer, "hessian"):
                    layer.hessian.to("cpu")  # free memory
                    del layer.hessian
                layer.hessian = hessians[n]  # type: ignore
                layer.hessian.layer = layer  # else we still point to the old layer

    @staticmethod
    def clear_hessians(net):
        """Removes hessians from the given network."""
        for _, layer in recursively_find_named_children(net):
            if hasattr(layer, "hessian"):
                del layer.hessian

    @staticmethod
    def hessians_to(net: nn.Module, device: torch.device | str):
        """Moves all hessians in network to device."""
        for _, layer in recursively_find_named_children(net):
            if hasattr(layer, "hessian"):
                layer.hessian.to(device)

    @property
    def H(self):
        """Returns the Hessian matrix as H = 2*X^T*X, where X is the input to the layer."""
        return self._H

    def precalculate(self):
        return self.cholesky_inverse

    @staticmethod
    def safe_pd_linalg(func, percdamp=0.01):
        """Returns a function that safely performs linear algebra that depends on positive semidefinitness by continually adding dampening."""

        def safe_linalg(mat, *args, **kwargs):
            dead = torch.diag(mat) == 0
            mat[dead, dead] = 1

            damp = percdamp * torch.mean(torch.diag(mat)).item()

            for factor in [1, 10, 100]:
                try:
                    mat.diagonal().add_(damp * factor)
                    return func(mat, *args, **kwargs)
                except (torch._C._LinAlgError, np.linalg.LinAlgError):  # type: ignore
                    print(
                        f"WARNING: Linear Algebra failed. Adding more dampening (factor={factor})."
                    )
            else:
                print("WARNING: All attempts failed. Using identity as fallback.")
                mat.diagonal().add_(1)
                return func(mat, *args, **kwargs)

        return safe_linalg

    @staticmethod
    def safe_cholesky_invert(H: torch.Tensor, percdamp: float):
        """Returns Chol(T^{-1}). Guaranteed to return a valid cholesky by gradually adding dampening."""
        chol_invert = lambda h: torch.linalg.cholesky(torch.inverse(h), upper=True)
        return LayerWiseHessian.safe_pd_linalg(chol_invert, percdamp)(H)

    @property
    def cholesky_inverse(self) -> torch.Tensor:
        """Returns the cholesky decomposition of the inverse of the Hessian."""
        if self._chol_inv is None:
            H = self._H.clone()
            H.to(self.chol_device)

            self._chol_inv = self.safe_cholesky_invert(H, self.percdamp)
            self._chol_inv.to(self.device)

        assert self._chol_inv is not None
        return self._chol_inv

    def reset(self):
        """Sets the Hessian matrix to zero."""
        self._H = torch.zeros((self.hsize, self.hsize), device=self.device)
        self.nsamples = 0
        self._chol_inv = None

    def get_weight(self):
        """See transform_weights."""
        W = self.layer.weight.data
        return TransformWeights(type(self.layer)).into_2d(W)

    def set_weight(self, W: torch.Tensor):
        """Sets the weights of the layer. The given weights should have the same shape as
        the ones returned from get_weights."""
        shape = list(self.layer.weight.shape)
        W = TransformWeights(type(self.layer)).from_2d(W, shape)
        self.layer.weight.data = W

    def transform_input(self, inp: torch.Tensor):
        """Returns the input in a transformed way f(X) such that the output Y is calculated by
        Y = f(X)W. For example, convolutions are unfolded."""
        if self.type == "linear" or self.type == "conv1d-t":
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if self.type == "conv":
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,  # type: ignore
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        return inp

    def add_batch(self, inp: torch.Tensor, output: Optional[torch.Tensor] = None):
        """Adds a calibration batch to the GPTQ quantiser. The batch is used to estimate the Hessian matrix."""
        inp = inp.to(self.device)
        self._chol_inv = None  # have to recalculate the inverse
        inp = inp.detach()
        tmp = inp.shape[0]
        inp = self.transform_input(inp)
        self._H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self._H += inp.matmul(inp.t())


def is_depthwise_separable(layer: torch.nn.Module) -> bool:
    """Returns true if the layer is a depthwise-separable module."""
    if isinstance(layer, torch.nn.Conv2d):
        if layer.groups == layer.in_channels and layer.groups == layer.out_channels:
            return True
    return False
