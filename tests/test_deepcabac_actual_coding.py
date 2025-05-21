from nn_compression.cv import CvModel, cifar10
from nn_compression.coding import DeepCABAC
from nn_compression.quantisation import EntropyQuantisation
from pathlib import Path
import torch


def test_deep_cabac():
    net = CvModel.RESNET18_CIFAR10.load()
    net.train(False)
    cifar = cifar10()
    assert cifar.evaluate(net) > 0.9
    netq = EntropyQuantisation(8, method="rtn").quantize_network(net)

    coder = DeepCABAC("tmp.nnc")
    coder.encode(netq)

    for n, p in coder.decode().items():
        po = netq.state_dict()[n]
        print(f"Rec. shape: {p.shape}, Orig. shape: {po.shape}")
        assert torch.allclose(p.flatten(), po.flatten())
    Path("tmp.nnc").unlink()
