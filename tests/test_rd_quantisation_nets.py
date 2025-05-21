import torch.nn as nn
import pytest
import torch
import copy
from torch.utils.data import DataLoader
import detectors  # do not delete this
from nn_compression.networks import recursively_find_named_children, LayerWiseHessian
from nn_compression._interfaces import quantisable
from nn_compression.cv import CvModel
from nn_compression.quantisation import EntropyQuantisation
from nn_compression.evaluation import entropy_net_estimation
from nn_compression.quantisation._rd_quant_free_functions import (
    _quant_weight,
    QuantParams,
)
import torch
from nn_compression._interfaces import quantisable
from nn_compression.evaluation import entropy_net_estimation
from nn_compression.quantisation import EntropyQuantisation
from nn_compression.networks import LayerWiseHessian, recursively_find_named_children
from nn_compression.cv import CvModel
import pytest


def extract_quant_weights(net_with_quantised_weights: torch.nn.Module) -> dict:
    """Extract the quantised weights from a network that has previously been quantised."""
    grids = {}
    for name, module in recursively_find_named_children(net_with_quantised_weights):
        if quantisable(module):
            grids[name] = module.weight
    return grids


@pytest.fixture
def net_nested():
    net = nn.Sequential(
        nn.Linear(2, 10), nn.ReLU(), nn.Sequential(nn.Linear(10, 3), nn.ReLU())
    )
    return net


@pytest.fixture
def quant_net_nested(net_nested):
    quant_net = copy.deepcopy(net_nested)
    quant_net[0].weight.data = torch.ones_like(quant_net[0].weight.data)
    quant_net[0].weight.data[0] = 2
    quant_net[2][0].weight.data = torch.ones_like(quant_net[2][0].weight.data)
    quant_net[2][0].weight.data[0] = 3
    return quant_net


def test_extract_grid(quant_net_nested):
    grids = extract_quant_weights(quant_net_nested)
    grid0 = torch.ones((10, 2))
    grid0[0] = 2
    grid1 = torch.ones((10, 2))
    grid1 = torch.ones((3, 10))
    grid1[0] = 3
    assert (grids["0"] == grid0).all()
    assert (grids["2.0"] == grid1).all()


def cifar_acc(net, steps, dataset):
    bs = 128
    acc = 0
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    for i, batch in enumerate(dataloader):
        xsanity, ysanity = batch
        predsanity = net(xsanity).softmax(dim=1).argmax(dim=1)
        accsanity = (predsanity == ysanity).sum().item() / bs
        acc += accsanity
        if i + 1 >= steps:
            break
    return acc / steps


def test_optq_rd_quantisation_performance():
    model = CvModel.RESNET18_CIFAR10
    net = model.load()
    ds = model.get_dataset()
    LayerWiseHessian.load_into_model(
        net, torch.load("tests/resnet18_cifar10_5000.pt", map_location="cpu")
    )
    assert ds.evaluate(net) > 0.9

    # EntropyQuantisation maintains the same interface, so we don't need to change this call
    netq = EntropyQuantisation(
        8, 1e-4, method="optq-rd", entropy_model="deepcabac"
    ).quantize_network(net)

    ent = entropy_net_estimation(netq, "deepcabac")
    assert ent < 0.2, f"{ent} > 0.5"
    acc = ds.evaluate(netq)
    assert ds.evaluate(net) > 0.9, f"{acc} < 0.9"
