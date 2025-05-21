import torch
from nn_compression.cv import CvModel
from nn_compression.coding import DeepCABAC
import numpy as np
from nn_compression.evaluation import entropy_net_estimation
from nn_compression.quantisation import EntropyQuantisation
from nn_compression.networks import LayerWiseHessian
import math
import torch.nn as nn
import torch
import pytest
from nn_compression.evaluation import entropy_net_estimation


def test_rate_estimation_approximately_correct_deepcabac():
    model = CvModel.RESNET18_CIFAR10.load()
    qnet = EntropyQuantisation(2**4, method="rtn").quantize_network(model)
    LayerWiseHessian.load_into_model(
        model, torch.load("tests/resnet18_cifar10_5000.pt", map_location="cpu")
    )

    coder = DeepCABAC()
    coder.encode(qnet)
    bpw_real = coder.bpw

    bpw_estim = entropy_net_estimation(qnet, "deepcabac")

    assert np.isclose(bpw_real, bpw_estim, rtol=0.05)


@pytest.fixture
def net():
    weight = torch.tensor([[0.0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0, 1]])
    net = nn.Linear(2, 8)
    net.weight = nn.Parameter(weight)
    return net


def test_rate_estimation_approximately_correct_shannon(net):
    ent = entropy_net_estimation(net, entropy_model="shannon")
    #
    p1 = 12 / 19
    p2 = 6 / 19
    p3 = 1 / 19  # "-1"
    #
    ent_by_hand = (11 * -math.log2(p1) - 5 * math.log2(p2)) / 16
    assert abs(ent - ent_by_hand) < 1e-4
