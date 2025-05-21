from pathlib import Path
import uuid

import numpy as np
import torch
from nn_compression._interfaces import quantisable
from nn_compression.coding._deepcabac import DeepCABAC
from nn_compression.networks import recursively_find_named_children
from nn_compression.networks import LayerWiseHessian
from typing import Callable, Literal, Optional
import torch.nn as nn
from nn_compression._core import EntropyModel as EntropyModelCpp
from ..coding._deepcabac import _WeightToIndex


def entropy_net_estimation(
    net: nn.Module,
    entropy_model: Literal["deepcabac", "shannon", "shannon_context"],
    groupsize: int = -1,
    verification_fn=None,
    transpose: bool = False,
) -> float:
    """Estimates the entropy of the given neural network under the entropy model."""
    if verification_fn is None:
        verification_fn = lambda _: True
    index_transformation = _WeightToIndex(groupsize)

    s = 0
    nel = 0
    for n, l in recursively_find_named_children(net):
        # w = LayerWiseHessian(l).get_weight()
        if not verification_fn(n) or not quantisable(l):
            continue
        wq = index_transformation.make_index(l)[0]
        if transpose:
            wq = np.ascontiguousarray(wq.T)

        estim = EntropyModelCpp(entropy_model, wq)
        s += estim.estimate_tensor(wq)
        nel += wq.size
    return s / nel


def bit_rate_deepcabac(
    net: nn.Module,
    groupsize: int = -1,
    verification_fn: Optional[Callable] = None,
):
    """Calculate the actual bit rate of the given network using deepcabac

    Args:
        net: The network to calculate the entropy of.
        transpose_deepcabac: Whether to transpose the network before encoding. Set this to true if you used OPTQ-RD to
        encode the network."""
    tmp_filepath = f"{uuid.uuid4()}.nnc"
    coder = DeepCABAC(
        tmp_filepath,
        transpose=False,
        groupsize=groupsize,
        filter=verification_fn,
    )
    try:
        coder.encode(net)
    finally:
        Path(tmp_filepath).unlink()
    return coder.bpw
