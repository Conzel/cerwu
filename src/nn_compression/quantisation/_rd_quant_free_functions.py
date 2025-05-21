import gc
import math
import time
from dataclasses import dataclass, replace
from typing import Literal, Tuple, Union, Dict, Any
import torch
import torch.nn as nn

from data_utils.arrays import strided_index_pairs
from nn_compression.networks import RegularizedCholeskyApproximator
from nn_compression.networks import (
    LayerWiseHessian,
    TransformWeights,
)
from nn_compression._core import GPTQ
from ._scaling import AbsMaxScaling

# Constants
HESSIAN_REGULARIZATION = 0.01
DIAGONAL_DAMPENING = 0.01
APPROXIMATOR_ITERATIONS = 15
VERBOSE = True

METHOD_T = Literal["optq-rd", "cerwu", "optq", "rtn"]


@dataclass(frozen=True)
class QuantParams:
    """Configuration parameters for neural network weight quantization."""

    method: METHOD_T
    nbins: int
    lm: float
    entropy_model: str = "deepcabac"
    device: str = "cpu"
    scan_order_major: str = "col"
    groupsize: int = -1
    verbose: bool = False

    def __post_init__(self):
        """Validate parameters after initialization."""
        object.__setattr__(
            self, "_validate_params", None
        )  # Trick to run validation in frozen dataclass

    @property
    def _validate_params(self):
        if self.nbins <= 0:
            raise ValueError(f"nbins must be positive, got {self.nbins}")
        if self.method not in ("rtn", "optq", "optq-rd", "cerwu"):
            raise ValueError(f"Unknown quantization method: {self.method}")
        return None

    @_validate_params.setter
    def _validate_params(self, _):
        pass

    @property
    def max_idx(self) -> int:
        """Returns maximum quantization index based on number of bins."""
        return self.nbins // 2

    @property
    def effective_lm(self) -> float:
        """Returns effective lambda value, accounting for optq special case."""
        return 0 if self.method == "optq" else self.lm

    def with_updates(self, **kwargs) -> "QuantParams":
        """Creates new parameter instance with specified updates."""
        return replace(self, **kwargs)


def _prepare_hessian(H: torch.Tensor, dead_value: float = 1.0) -> torch.Tensor:
    """Prepares Hessian matrix by fixing zero diagonals and adding regularization."""
    dead = torch.diag(H) == 0
    H[dead, dead] = dead_value
    H.diagonal().add_(H.diag().mean() * DIAGONAL_DAMPENING)
    return H


def _get_entropy_options(params: QuantParams, gamma: float) -> Dict[str, Any]:
    """Creates entropy model options dict for GPTQ based on quantization parameters."""
    entropy_model_options = {
        "base_model": params.entropy_model,
        "max_idx": params.max_idx,
        "gamma": gamma,
    }

    if params.method == "cerwu":
        return {
            "max_idx": params.nbins,
            "entropy_model": "regularized",
            "entropy_model_options": entropy_model_options,
            "scan_order_major": params.scan_order_major,
        }
    elif params.method == "optq-rd" or params.method == "optq":
        return {
            "max_idx": params.max_idx,
            "entropy_model": params.entropy_model,
            "scan_order_major": params.scan_order_major,
        }
    else:
        raise ValueError(f"Unknown method {params.method}")


def calc_gamma(w: torch.Tensor) -> float:
    """Calculates regularization parameter gamma from weight variance."""
    return float(1 / (math.log(2) * torch.var(w)))


def rob_transform(
    w: torch.Tensor, H: torch.Tensor, gamma: Union[float, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies robust transformation to weights and Hessian, returns transformed pair."""
    Hprime = H.clone()
    Hprime.diagonal().add_(gamma)
    Hprime_inv = torch.inverse(Hprime)
    wprime = w.unsqueeze(0) @ H @ Hprime_inv
    return wprime.reshape(w.shape), Hprime


def _quant_layer(
    layer: nn.Module,
    params: QuantParams,
    return_integers: bool = False,
) -> torch.Tensor:
    """Quantizes layer weights in-place, returns quantized weights (integers or scaled)."""
    start_time = time.time()

    orig_device = layer.weight.data.device
    w = TransformWeights(type(layer)).into_2d(layer.weight.data).to("cpu")

    if params.method == "rtn":
        H = torch.eye(w.shape[1], device="cpu")
    else:
        H = layer.hessian.H.to("cpu")

    H = _prepare_hessian(H)

    wq, wq_scaled = _quant_weight(w, H, params)

    assert wq_scaled is not None
    layer.weight.data = TransformWeights(type(layer)).from_2d(
        wq_scaled.to(orig_device), layer.weight.data.shape
    )

    if params.verbose:
        print(
            f"Quantized layer of shape {layer.weight.shape} in {time.time() - start_time:.2f}s"
        )

    return wq if return_integers else wq_scaled


def _quant_weight_rtn(
    w: torch.Tensor, params: QuantParams
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes weights with round-to-nearest, returns (quantized_ints, scaled_weights)."""
    s = AbsMaxScaling(params.nbins, w)
    wq = s.scale(w).round().clamp(-params.max_idx, params.max_idx)
    return wq, s.unscale(wq)


def _quant_full_matrix(
    w: torch.Tensor,
    H: torch.Tensor,
    params: QuantParams,
    gamma: float,
) -> torch.Tensor:
    """Quantizes weights as full matrix, returns integer quantized weights."""
    if params.method == "cerwu":
        wp, Hp = rob_transform(w, H, gamma * params.effective_lm)
    else:
        wp, Hp = w, H

    Cp = LayerWiseHessian.safe_cholesky_invert(Hp, HESSIAN_REGULARIZATION)
    return _quant_rows(wp, Cp, params, gamma)


def _quant_by_groups_with_approximator(
    ws: torch.Tensor,
    H: torch.Tensor,
    s: AbsMaxScaling,
    params: QuantParams,
) -> torch.Tensor:
    """Quantizes weights by groups using approximator, returns integer quantized weights."""
    wq = torch.zeros_like(ws, dtype=torch.int32, device=ws.device)

    approximator = RegularizedCholeskyApproximator(
        ws,
        H,
        params.effective_lm,
        s.delta,
        groupsize=params.groupsize,
        n_iter=APPROXIMATOR_ITERATIONS,
        verbose=VERBOSE,
        device=params.device,
        no_approx=True,
    )

    for (u, l), approx in approximator:
        wp = ws @ (H / approx.delta**2) @ approx.inverse_regularized_hessian
        wq[u:l] = torch.tensor(
            _quant_rows(wp[u:l], approx.cholesky_of_inverse, params, approx.gamma),
            dtype=torch.int32,
        )

    del approx
    return wq


def _quant_by_groups_standard(
    ws: torch.Tensor,
    H: torch.Tensor,
    s: AbsMaxScaling,
    params: QuantParams,
) -> torch.Tensor:
    """Quantizes weights by groups using standard approach, returns integer quantized weights. Expects ws to be scaled to integers, but H to be the
    Hessian corresponding to the unscaled weights."""
    wq = torch.zeros_like(ws, dtype=torch.int32, device=ws.device)
    Cp = LayerWiseHessian.safe_cholesky_invert(H, HESSIAN_REGULARIZATION)

    assert params.method in ["optq-rd", "optq"]

    for i, (u, l) in enumerate(strided_index_pairs(ws.shape[0], params.groupsize)):
        delta = s.delta[i].item()
        Cps = Cp * delta
        wq[u:l] = torch.tensor(
            _quant_rows(ws[u:l], Cps, params, 0),  # OPTQ-RD does not need Gamma
            dtype=torch.int32,
        )
        del Cps

    del Cp
    return wq


def _quant_rows(
    wp: torch.Tensor,
    Cp: torch.Tensor,
    params: QuantParams,
    gamma: float,
) -> torch.Tensor:
    """Quantizes weight rows using GPTQ, returns integer quantized weights."""
    wp = wp.contiguous()
    Cp = Cp.contiguous()

    if len(wp.shape) == 1:
        wp = wp.reshape(1, -1)

    options = _get_entropy_options(params, gamma)
    return GPTQ(options).run(wp, Cp, params.effective_lm)


def _quant_weight(
    w: torch.Tensor,
    H: torch.Tensor,
    params: QuantParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes weights using specified method, returns (quantized_ints, scaled_weights)."""
    start_time = time.time()

    w = w.to("cpu")
    H = H.to("cpu")

    if params.method == "rtn":
        return _quant_weight_rtn(w, params)

    w, H = w.clone(), H.clone()

    s = AbsMaxScaling(params.nbins, w, groupsize=params.groupsize)
    ws = s.scale(w)

    if params.groupsize <= 0:
        gamma = calc_gamma(ws)
        Hs = s.unscale(s.unscale(H))
        wq = _quant_full_matrix(ws, Hs, params, gamma)
        del Hs
    else:
        # need to scale H individually in the functions
        if params.method == "cerwu":
            # scaling of H and calculations of gamma is handled inside by Cholesky Approximator
            wq = _quant_by_groups_with_approximator(ws, H, s, params)
        else:
            # OPTQ-RD does not need Gamma
            wq = _quant_by_groups_standard(ws, H, s, params)

    del ws

    gc.collect()

    if params.verbose:
        print(
            f"Quantized tensor of shape {wq.shape} in {time.time() - start_time:.2f}s"
        )

    return wq, s.unscale(torch.tensor(wq).float())
