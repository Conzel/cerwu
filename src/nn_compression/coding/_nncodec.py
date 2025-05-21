import copy
from pathlib import Path
import uuid
import nnc
from torch import nn
import torch

from nn_compression._interfaces import quantisable
from nn_compression.networks import TransformWeights, recursively_find_named_children


def nnc_compress(
    net: nn.Module,
    qp: int,
    method: str = "urq",
    only_quantisable: bool = False,
    transpose: bool = False,
    verification_fn=None,
):
    """Compresses the given network with NNCodec."""
    if verification_fn is None:
        verification_fn = lambda x: True
    maybe_transpose = lambda x, y: (
        TransformWeights(type(y)).into_2d(x).T if transpose else x
    )
    maybe_retranspose = lambda x, y, orig: (
        TransformWeights(type(y)).from_2d(x.T, orig.shape) if transpose else x
    )

    quantisable_params = {}
    unquantisable_params = {}
    for n, layer in recursively_find_named_children(net):
        if quantisable(layer) and verification_fn(n):
            quantisable_params[f"{n}.weight"] = maybe_transpose(
                layer.weight.detach(), layer
            ).numpy()
    for n, p in net.state_dict().items():
        if n not in quantisable_params:
            unquantisable_params[n] = p.detach().numpy()
    assert len(quantisable_params) + len(unquantisable_params) == len(net.state_dict())

    to_quantise = quantisable_params
    if not only_quantisable:
        to_quantise = to_quantise | unquantisable_params
    nnc_filename = f"{uuid.uuid4()}.nnc"
    try:
        bitstream = nnc.compress(
            to_quantise,
            nnc_filename,
            qp=qp,
            use_dq=method == "dq",
            return_bitstream=True,
        )
        rec = nnc.decompress(nnc_filename)
        assert bitstream is not None
        bits_used = len(bitstream) * 8
    finally:
        Path(nnc_filename).unlink()
    assert isinstance(rec, dict)

    restored_weights = {}
    for n, layer in recursively_find_named_children(net):
        if quantisable(layer) and verification_fn(n):
            weight_name = f"{n}.weight"
            assert (
                weight_name in rec
            ), f"Could not find {weight_name} in the compressed model."
            restored_weights[weight_name] = torch.from_numpy(
                maybe_retranspose(rec[weight_name], layer, layer.weight)
            )
    for unquantisable in unquantisable_params.keys():
        if only_quantisable:
            val = unquantisable_params[unquantisable]
        else:
            val = rec[unquantisable]
        restored_weights[unquantisable] = torch.from_numpy(val)

    model_dec = copy.deepcopy(net)
    model_dec.load_state_dict(restored_weights)
    model_dec.train(False)

    model_params = 0
    for p in to_quantise.values():
        model_params += p.size

    entropy = bits_used / model_params

    return model_dec, entropy
