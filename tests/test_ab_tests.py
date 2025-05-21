import pytest
import torch
from torch import nn
from typing import Literal
import numpy as np
from nn_compression.cv import cifar10, CvModel
from nn_compression.quantisation import EntropyQuantisation
from nn_compression.networks import recursively_find_named_children, LayerWiseHessian
from data_utils.arrays import strided_index_pairs

# Define which layers to quantize (select two specific layers from ResNet18)
# These are typically the most computationally intensive layers
TARGET_LAYERS = ["layer1.1.conv2", "layer2.0.conv1"]


def filter_target_layers(name):
    """Only quantize specified target layers."""
    return name in TARGET_LAYERS


def test_ab_optq_rd_r():
    # ensures performance stays the same
    net_enum = CvModel.RESNET18_CIFAR10
    net = net_enum.load()
    ds = net_enum.get_dataset()
    nbins = 8
    lm = 1e-5
    nbatch_eval = 10
    LayerWiseHessian.load_into_model(net, "tests/resnet18_cifar10_5000.pt")

    q = EntropyQuantisation(
        nbins, lm, "cerwu", -1, "deepcabac", "cpu", "row", net_enum.filter_fn()
    )
    netq = q.quantize_network(net)
    acc = ds.evaluate(netq, nbatch_eval)
    ent = q.estimate_entropy(netq)

    assert acc > 0.89 and acc < 0.93
    assert ent > 0.33 and ent < 0.37


def test_ab_optq_rd():
    # ensures performance stays the same
    net_enum = CvModel.RESNET18_CIFAR10
    net = net_enum.load()
    ds = net_enum.get_dataset()
    nbins = 8
    lm = 1e-7
    nbatch_eval = 10
    LayerWiseHessian.load_into_model(net, "tests/resnet18_cifar10_5000.pt")

    q = EntropyQuantisation(
        nbins, lm, "optq-rd", -1, "deepcabac", "cpu", "row", net_enum.filter_fn()
    )
    netq = q.quantize_network(net)
    acc = ds.evaluate(netq, nbatch_eval)
    ent = q.estimate_entropy(netq)

    assert acc > 0.93 and acc < 0.96
    assert ent > 0.50 and ent < 0.52
