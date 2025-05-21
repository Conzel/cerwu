import torch.nn as nn
from transformers.pytorch_utils import Conv1D

Quantisable = nn.Linear | nn.Conv2d | Conv1D


def quantisable(m: nn.Module) -> bool:
    if not isinstance(m, nn.Module):
        raise ValueError(
            f"Tried to call quantisable on {m}, which is of type {type(m)}, not torch.nn.Module."
        )
    if hasattr(m, "quantisable"):
        return m.quantisable
    return isinstance(m, Quantisable)


__all__ = ["Quantisable", "quantisable"]
