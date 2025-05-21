import torch
import torch.ao.quantization as tq
from torch.utils.data import DataLoader
from torchvision.models.quantization import resnet18 as qresnet18
from torchvision.models.quantization import resnet50 as qresnet50

from nn_compression.experiments import NETWORK_T


class _Wrapper(torch.nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        self.model = m

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def wrap(model: torch.nn.Module, enum: str):
    if enum == "vgg16":
        return _Wrapper(model)
    if enum == "mobilenetv3_large":
        return _Wrapper(model)
    elif enum == "resnet18_cifar10":
        wrapped = qresnet18()
        wrapped.fc = torch.nn.Linear(512, 10)
        wrapped.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        wrapped.maxpool = torch.nn.Identity()  # type: ignore
        return wrapped
    elif enum == "resnet18_imagenet":
        qmodel = qresnet18()
        qmodel.load_state_dict(model.state_dict())
        return qmodel
    elif enum == "resnet50_imagenet":
        qmodel = qresnet50()
        qmodel.load_state_dict(model.state_dict())
        return qmodel
    else:
        raise ValueError(f"Model {enum} not supported.")


def get_lowbits_config(nbits: int):
    """Returns a quantisation configuration for fake-quantisation with a given number of bits."""
    lowbits_act_fq = tq.FakeQuantize.with_args(
        observer=tq.HistogramObserver,
        quant_min=int(0),
        quant_max=int(2**nbits - 1),
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )
    lowbits_weight_fq = tq.FakeQuantize.with_args(
        observer=tq.HistogramObserver,
        quant_min=int(-(2**nbits) / 2),
        quant_max=int((2**nbits) / 2 - 1),
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )
    qconfig = tq.QConfig(activation=lowbits_act_fq, weight=lowbits_weight_fq)
    return qconfig


def prepare_quantized_execution(
    quantized_model: torch.nn.Module,
    network_type: NETWORK_T,
    dataloader: DataLoader,
    nbits: int = 8,
    nbatches_cal: int = 5,
    wrap_net: bool = True,
):
    """Prepares a quantized model for execution by running it on a dataloader for a few batches.
    The returned model is executed in a quantized manner, resulting in inference speed-ups of ~3x for 8 bit quantization.

    Bit-widths other than 8 bits are not supported natively by the pytorch quantization library. Providing
    a different bit-width will result in a fake-quantization that does not actually speed up inference,
    but simulates the effect of quantization on the model.
    """
    if wrap_net:
        quantizable_model = wrap(quantized_model, network_type)
    else:
        quantizable_model = quantized_model

    # preparing static quantization
    if nbits == 8:
        print("Doing proper 8 bit quantization...")
        quantizable_model.qconfig = tq.get_default_qconfig("x86")  # type: ignore
    else:
        print(f"Doing fake quantization with {nbits} bits...")
        quantizable_model.qconfig = get_lowbits_config(nbits)  # type: ignore
    model_prepared = tq.prepare(quantizable_model)

    i = 0
    for x, _ in dataloader:
        model_prepared(x)
        i += 1
        if i >= nbatches_cal:
            break

    model_quantized = tq.convert(model_prepared)
    return model_quantized
