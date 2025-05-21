import pytest
from typing import Literal
from nn_compression.cv import cifar10, CvModel
from nn_compression.quantisation import EntropyQuantisation
from nn_compression.networks import recursively_find_named_children, LayerWiseHessian
import torch
from data_utils.arrays import strided_index_pairs


# Define which layers to quantize (select two specific layers from ResNet18)
# These are typically the most computationally intensive layers
TARGET_LAYERS = ["layer1.1.conv2", "layer2.0.conv1"]


def filter_target_layers(name):
    """Only quantize specified target layers."""
    return name in TARGET_LAYERS


def test_target_layer_quantization_verification():
    """Verify that only the target layers are quantized and contain expected values."""
    net = CvModel.RESNET18_CIFAR10.load()

    # Load pre-computed Hessians
    LayerWiseHessian.load_into_model(net, "tests/resnet18_cifar10_5000.pt")

    # Store original weights
    original_weights = {}
    for name, module in recursively_find_named_children(net):
        original_weights[name] = (
            module.weight.clone() if hasattr(module, "weight") else None
        )

    # Quantize with groupsize
    groupsize = 8
    nbins = 8
    quantizer = EntropyQuantisation(
        nbins=nbins, method="optq", groupsize=groupsize, filter_fn=filter_target_layers
    )
    quantized_net = quantizer.quantize_network(net)

    # Verify only target layers were quantized
    for name, module in recursively_find_named_children(quantized_net):
        if hasattr(module, "weight") and original_weights[name] is not None:
            if name in TARGET_LAYERS:
                # Target layers should be quantized (weights changed)
                assert not torch.allclose(module.weight, original_weights[name])

                w = module.weight.data
                for u, l in strided_index_pairs(w.shape[0], groupsize):
                    ws = w[u:l]
                    # Verify quantization effects - count unique values
                    unique_values = torch.unique(ws)
                    assert (
                        len(unique_values) <= nbins + 1
                    ), f"Layer {name} has {len(unique_values)} unique values"

                    print(f"Layer {name} quantized: {len(unique_values)} unique values")
            else:
                # Non-target layers should remain unchanged
                assert torch.allclose(module.weight, original_weights[name])


def test_groupsize_performance():
    """Test different groupsizes and compare their performance on target layers."""
    net = CvModel.RESNET18_CIFAR10.load()
    net.train(False)
    cifar = cifar10()

    # Verify baseline accuracy
    baseline_acc = cifar.evaluate(net)
    assert baseline_acc > 0.9

    # Load pre-computed Hessians instead of estimating them
    LayerWiseHessian.load_into_model(net, "tests/resnet18_cifar10_5000.pt")

    # Test different groupsizes
    groupsizes = [1, 8, 32, 64]
    results = {}

    for gs in groupsizes:
        netq = EntropyQuantisation(
            nbins=8, method="optq", groupsize=gs, filter_fn=filter_target_layers
        ).quantize_network(net)

        acc = cifar.evaluate(netq)
        results[gs] = acc

        # Quantizing only two layers should have less impact on accuracy
        assert acc > 0.85

    # Print results for analysis
    print(f"Baseline accuracy: {baseline_acc}")
    print(f"Target layers: {TARGET_LAYERS}")
    for gs, acc in results.items():
        print(f"Groupsize {gs}: accuracy {acc}")


def test_groupsize_vs_methods():
    """Test interaction between groupsize and different quantization methods on target layers."""
    net = CvModel.RESNET18_CIFAR10.load()
    net.train(False)
    cifar = cifar10()

    # Load pre-computed Hessians
    LayerWiseHessian.load_into_model(net, "tests/resnet18_cifar10_5000.pt")

    methods: list[Literal["optq", "optq-rd", "cerwu"]] = [
        "optq",
        "optq-rd",
        "cerwu",
    ]
    groupsizes = [4, 16]

    for method in methods:
        lm = 0 if method == "optq" else 1e-6
        for gs in groupsizes:
            netq = EntropyQuantisation(
                nbins=8,
                method=method,
                groupsize=gs,
                lm=lm,
                filter_fn=filter_target_layers,
            ).quantize_network(net)

            acc = cifar.evaluate(netq)
            print(f"Method: {method}, Groupsize: {gs}, Accuracy: {acc}")

            # Higher accuracy threshold since we're only quantizing two layers
            assert acc > 0.85, f"{method}"


def test_entropy_estimation_with_groupsize():
    """Test entropy estimation with different groupsizes on target layers."""
    net = CvModel.RESNET18_CIFAR10.load()

    # Load pre-computed Hessians
    LayerWiseHessian.load_into_model(net, "tests/resnet18_cifar10_5000.pt")

    groupsizes = [-1, 4, 16]  # -1 means per-tensor quantization
    entropies = {}

    for gs in groupsizes:
        quantizer = EntropyQuantisation(
            nbins=8, method="optq", groupsize=gs, filter_fn=filter_target_layers
        )
        netq = quantizer.quantize_network(net)

        # Estimate entropy
        entropy = quantizer.estimate_entropy(netq)
        entropies[gs] = entropy
        for n, l in recursively_find_named_children(netq):
            if filter_target_layers(n):
                print(f"{n}: \n    {l.weight.data.flatten(1)[0:3,0:5]}...")

        print(f"Groupsize {gs} has entropy {entropy}.")
        # Verify entropy is reasonable
        assert entropy > 0
        assert entropy < 2

    print(f"Target layers: {TARGET_LAYERS}")


@pytest.mark.parametrize("nbins", [4, 8, 16])
@pytest.mark.parametrize("groupsize", [1, 8, 32])
def test_groupsize_and_bins(nbins, groupsize):
    """Test interaction between groupsize and number of bins on target layers."""
    net = CvModel.RESNET18_CIFAR10.load()
    net.train(False)
    cifar = cifar10()

    # Load pre-computed Hessians
    LayerWiseHessian.load_into_model(net, "tests/resnet18_cifar10_5000.pt")

    netq = EntropyQuantisation(
        nbins=nbins, method="optq", groupsize=groupsize, filter_fn=filter_target_layers
    ).quantize_network(net)

    acc = cifar.evaluate(netq)
    print(f"Bins: {nbins}, Groupsize: {groupsize}, Accuracy: {acc}")

    # Higher accuracy thresholds since we're only quantizing two layers
    min_acc = 0.8 if nbins <= 4 else (0.85 if nbins <= 8 else 0.87)
    assert acc > min_acc, f"nbins: {nbins}, gs: {groupsize}"


def test_groupsize_negative_one_equivalence_full_network():
    """Test that groupsize=-1 gives equivalent results to a very large groupsize for the full network."""
    methods: list[Literal["optq", "optq-rd", "cerwu"]] = [
        "optq",
        "optq-rd",
        "cerwu",
    ]
    for method in methods:
        print(f"Testing {method}")
        # Setup model and dataset
        net = CvModel.RESNET18_CIFAR10.load()
        net.train(False)
        cifar = cifar10()

        # Load pre-computed Hessians instead of estimating them
        LayerWiseHessian.load_into_model(net, "tests/resnet18_cifar10_5000.pt")

        # Quantize the full network with groupsize=-1
        quantizer_neg1 = EntropyQuantisation(
            nbins=8, method=method, groupsize=-1, lm=1e-7
        )
        net_neg1 = quantizer_neg1.quantize_network(net)
        acc_neg1 = cifar.evaluate(net_neg1)
        ent_neg1 = quantizer_neg1.estimate_entropy(net_neg1)

        # Quantize the full network with a very large groupsize
        quantizer_large = EntropyQuantisation(
            nbins=8, method=method, groupsize=10_000_000, lm=1e-7
        )
        net_large = quantizer_large.quantize_network(net)
        acc_large = cifar.evaluate(net_large)
        ent_large = quantizer_large.estimate_entropy(net_large)

        # Print results for analysis
        print(f"Accuracy with groupsize=-1: {acc_neg1}")
        print(f"Accuracy with groupsize=10_000_000: {acc_large}")
        print(f"Entropy with groupsize=-1: {ent_neg1}")
        print(f"Entropy with groupsize=10_000_000: {ent_large}")

        # Assert the accuracies are close with a 0.1 tolerance
        assert (
            abs(acc_neg1 - acc_large) < 0.05
        ), f"Accuracy difference: {abs(acc_neg1 - acc_large)}"

        # Assert the entropies are close with a 0.1 tolerance
        assert (
            abs(ent_neg1 - ent_large) < 0.05
        ), f"Entropy difference: {abs(ent_neg1 - ent_large)}"
