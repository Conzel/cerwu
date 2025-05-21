import copy
from typing import Callable, Literal, Optional
import torch
import torch.nn as nn
from nn_compression.evaluation import entropy_net_estimation
from nn_compression._interfaces import quantisable
from nn_compression.networks import recursively_find_named_children

from ._rd_quant_free_functions import _quant_layer, _quant_weight, QuantParams

METHOD_T = Literal["optq-rd", "cerwu", "optq", "rtn"]


class EntropyQuantisation:
    """Entropy-guided neural network quantization."""

    def __init__(
        self,
        nbins: int,
        lm: Optional[float] = None,
        method: METHOD_T = "optq",
        groupsize: int = -1,
        entropy_model: Literal["deepcabac", "shannon"] = "deepcabac",
        device: str = "cpu",
        scan_order_major: str = "row",
        filter_fn: Optional[Callable[[str], bool]] = None,
        verbose: bool = False,
    ):
        """Constructs an RD-Quantisation object for network weight quantization."""
        if lm is None:
            if method in ["optq-rd", "cerwu"]:
                raise ValueError(f"Method {method} requires specifying a λ value.")
            lm = 0

        self.filter_fn = filter_fn if filter_fn is not None else lambda _: True

        # Create quantization parameters
        self.quant_params = QuantParams(
            method=method,
            nbins=nbins,
            lm=lm,
            entropy_model=entropy_model,
            device=device,
            scan_order_major=scan_order_major,
            groupsize=groupsize,
            verbose=verbose,
        )

    def quantize_network(self, net: nn.Module):
        """Quantizes applicable layers in a neural network."""
        net = copy.deepcopy(net)

        for n, l in recursively_find_named_children(net):
            if not self.filter_fn(n) or not quantisable(l):
                continue

            if not hasattr(l, "hessian") and self.quant_params.method in [
                "optq",
                "optq-rd",
                "cerwu",
            ]:
                raise ValueError(
                    f"Layer {n} of type {l} has no Hessian attribute, but the {self.quant_params.method} "
                    f"method requires it. Please instantiate hessians before calling this function."
                )

            _quant_layer(l, self.quant_params)

        return net

    def quantize_tensor(self, w: torch.Tensor, H: torch.Tensor):
        """Quantizes a weight tensor using the Hessian matrix for guidance."""
        if len(w.shape) == 1:
            w = w.reshape(1, -1)

        assert len(w.shape) == 2, f"w must be 2D, found shape {w.shape}"

        return _quant_weight(w, H, self.quant_params)

    def estimate_entropy(self, netq: nn.Module):
        """Estimates entropy of quantized network weights."""
        transpose = self.quant_params.scan_order_major == "col"

        return entropy_net_estimation(
            netq,
            self.quant_params.entropy_model,  # type: ignore
            groupsize=self.quant_params.groupsize,
            verification_fn=self.filter_fn,
            transpose=transpose,
        )
