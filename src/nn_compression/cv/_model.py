from enum import Enum
from typing import Optional
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
    resnet152,
    ResNet152_Weights,
    vgg16,
    VGG16_Weights,
    vit_b_16,
    ViT_B_16_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    efficientnet_b7,
    EfficientNet_B7_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)
import torch
import torch.fx as fx
import torch.nn as nn

import timm
import detectors  # do not remove


from nn_compression.networks._utils import recursively_find_named_children
from ._datasets import cifar, Normalisation, imagenet
import os

IMAGENET_PATH = os.environ.get("IMAGENET_PATH")


def unfold_depthwise_convolutions(model: nn.Module, debug: bool = False):
    """Unfolds depthwise separable convolutions in a model to separate convolutions.
    Example:
        Instead of
            x (16, 32,32) -> DWSConv -> y (16, 32, 32)
        where each channel is separately convolved with a filter, we have
            x (16, 32, 32) -> 16 x (1, 32, 32) -> 16 x Conv2d -> 16 y (1, 32, 32) -> y (16, 32, 32)
    """

    class DWConvToSeparateConvs(fx.Transformer):
        def __init__(self, module):
            super().__init__(module)
            self.counter = 0

        def call_module(self, target, args, kwargs):
            module = self.fetch_attr(target)  # type: ignore
            if isinstance(module, nn.Conv2d) and module.groups == module.out_channels:
                new_convs = []
                for i in range(module.out_channels):
                    single_conv = nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=module.kernel_size,  # type: ignore
                        stride=module.stride,  # type: ignore
                        padding=module.padding,  # type: ignore
                        dilation=module.dilation,  # type: ignore
                        groups=1,
                        bias=module.bias is not None,
                    )
                    single_conv.weight = nn.Parameter(module.weight[i : i + 1])
                    if module.bias is not None:
                        single_conv.bias = nn.Parameter(module.bias[i : i + 1])

                    new_conv_name = f"single_conv_{self.counter}"
                    setattr(self.module, new_conv_name, single_conv)
                    self.counter += 1
                    new_convs.append(getattr(self.module, new_conv_name))

                x = args[0]
                # Split input channels and pass each through corresponding conv
                new_conv_out = torch.cat(
                    [conv(x[:, i : i + 1]) for i, conv in enumerate(new_convs)], dim=1  # type: ignore
                )
                return new_conv_out
            return super().call_module(target, args, kwargs)

    traced = fx.symbolic_trace(model)
    transformed_model = DWConvToSeparateConvs(traced).transform()
    if debug:
        transformed_model = NoDepthwiseCheckWrapper(transformed_model)
    return transformed_model


class NoDepthwiseCheckWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if has_depthwise_convolutions(self):
            raise RuntimeError("Depthwise separable convolution detected!")
        return self.model(x)


def has_depthwise_convolutions(model) -> bool:
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.groups == module.out_channels:
            return True
    else:
        return False


def test_transformation(model, transformed_model, input_shape=(1, 3, 224, 224)):
    x = torch.randn(input_shape)
    with torch.no_grad():
        orig_out = model(x)
        trans_out = transformed_model(x)
        assert torch.allclose(orig_out, trans_out, atol=1e-4), "Outputs do not match!"


class CvModel(str, Enum):
    RESNET18_CIFAR10 = "resnet18_cifar10"
    RESNET34_CIFAR10 = "resnet34_cifar10"
    RESNET50_CIFAR10 = "resnet50_cifar10"
    RESNET18_IMAGENET = "resnet18_imagenet"
    RESNET34_IMAGENET = "resnet34_imagenet"
    RESNET50_IMAGENET = "resnet50_imagenet"
    RESNET101_IMAGENET = "resnet101_imagenet"
    RESNET152_IMAGENET = "resnet152_imagenet"
    MOBILENETV3_LARGE = "mobilenetv3_large"
    MOBILENETV3_SMALL = "mobilenetv3_small"
    EFFICIENTNET_B7 = "efficientnet_b7"
    VGG16 = "vgg16"
    VIT_B_16 = "vit_b_16"
    CONVNEXT_TINY = "convnext_tiny"

    def filter_fn(self):
        if self.value.startswith("resnet"):
            return lambda n: "fc" not in n
        if self.value == "vgg16":
            return lambda n: True
        if self.value == "convnext_tiny":
            return lambda n: True
        if "mobile" in self.value:
            return lambda n: True
        else:
            raise NotImplementedError(
                f"Didn't implement filter_fn for {self.value} (yet)."
            )

    def transforms(self):
        if self.value.startswith("resnet"):
            raise ValueError("ResNet models do not require transforms.")
        elif self.value == "vgg16":
            return VGG16_Weights.IMAGENET1K_V1.transforms()
        elif self.value == "vit_b_16":
            return ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        elif self.value == "convnext_tiny":
            return ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        raise ValueError(f"Unknown model: {self.value}")

    def load(self, pretrained: bool = True, train: bool = False):
        """Loads the model from the timm library. Defaults to pretrained
        and in eval mode."""
        if self.value.startswith("resnet") and self.value.endswith("cifar10"):
            model = timm.create_model(self.value, pretrained=pretrained)
        elif self.value == "resnet18_imagenet":
            model = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        elif self.value == "resnet34_imagenet":
            model = resnet34(ResNet34_Weights.IMAGENET1K_V1)
        elif self.value == "resnet50_imagenet":
            model = resnet50(ResNet50_Weights.IMAGENET1K_V1)
        elif self.value == "resnet101_imagenet":
            model = resnet101(ResNet101_Weights.IMAGENET1K_V1)
        elif self.value == "resnet152_imagenet":
            model = resnet152(ResNet152_Weights.IMAGENET1K_V1)
        elif self.value == "vgg16":
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        elif self.value == "vit_b_16":
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            for n, l in recursively_find_named_children(model):
                # This makes the layers not count when compressing with DeepCABAC
                # They are not actually called in the forward pass in our experiments
                if "out_proj" in n:
                    l.quantisable = False  # type: ignore
        elif self.value == "mobilenetv3_large":
            model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        elif self.value == "mobilenetv3_small":
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        elif self.value == "efficientnet_b7":
            model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        elif self.value == "convnext_tiny":
            model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.train(train)
        return model

    def get_dataset(self, shuffle: bool = True, batch_size: Optional[int] = None):
        def _checked_imagenet(transforms, batch_size=16):
            if IMAGENET_PATH is None:
                raise ValueError(
                    "IMAGENET_PATH not set. Use export IMAGENET_PATH=/path/to/imagenet before you run the script."
                )
            return imagenet(IMAGENET_PATH, shuffle, transforms, batch_size)

        # this construction is to keep the respective defaults of the batch size if none is specified
        kwargs = {}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        if self.value.startswith("resnet") and self.value.endswith("cifar10"):
            return cifar("10", shuffle, Normalisation.CIFAR10_EDALTOCG, **kwargs)
        elif self.value == "resnet18_imagenet":
            return _checked_imagenet(
                ResNet18_Weights.IMAGENET1K_V1.transforms(), **kwargs
            )
        elif self.value == "resnet34_imagenet":
            return _checked_imagenet(
                ResNet34_Weights.IMAGENET1K_V1.transforms(), **kwargs
            )
        elif self.value == "resnet50_imagenet":
            return _checked_imagenet(
                ResNet50_Weights.IMAGENET1K_V1.transforms(), **kwargs
            )
        elif self.value == "resnet101_imagenet":
            return _checked_imagenet(
                ResNet101_Weights.IMAGENET1K_V1.transforms(), **kwargs
            )
        elif self.value == "resnet152_imagenet":
            return _checked_imagenet(
                ResNet152_Weights.IMAGENET1K_V1.transforms(), **kwargs
            )
        elif self.value == "vgg16":
            return _checked_imagenet(VGG16_Weights.IMAGENET1K_V1.transforms(), **kwargs)
        elif self.value == "vit_b_16":
            return _checked_imagenet(
                ViT_B_16_Weights.IMAGENET1K_V1.transforms(), **kwargs
            )
        elif self.value == "mobilenetv3_small":
            return _checked_imagenet(
                MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms(), **kwargs
            )
        elif self.value == "mobilenetv3_large":
            return _checked_imagenet(
                MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms(), **kwargs
            )
        elif self.value == "efficientnet_b7":
            return _checked_imagenet(
                EfficientNet_B7_Weights.IMAGENET1K_V1.transforms(), **kwargs
            )
        elif self.value == "convnext_tiny":
            return _checked_imagenet(
                ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms(), **kwargs
            )
        else:
            raise ValueError(f"Unknown model: {self.value}")

    @staticmethod
    def from_string(s: str):
        return CvModel(s)
