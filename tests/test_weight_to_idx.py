import pytest
import torch
import numpy as np
from torch import nn
from nn_compression.coding._deepcabac import _WeightToIndex


def test_init_default():
    w2i = _WeightToIndex()
    assert w2i.groupsize == -1


def test_init_with_groupsize():
    w2i = _WeightToIndex(16)
    assert w2i.groupsize == 16


def test_make_index_single_uniform_values():
    w2i = _WeightToIndex()
    tensor = torch.ones(4, 4)
    indices, step = w2i._make_index_single(tensor)
    assert np.array_equal(indices, np.zeros((4, 4), dtype=np.int32))
    assert step == 0.0


def test_make_index_single_two_values():
    w2i = _WeightToIndex()
    tensor = torch.zeros(2, 2)
    tensor[0, 0] = 1.0
    tensor[1, 1] = 1.0
    indices, step = w2i._make_index_single(tensor)
    expected = np.array([[1, 0], [0, 1]], dtype=np.int32)
    assert np.array_equal(indices, expected)
    assert step == 1.0


def test_make_index_single_negative_values():
    w2i = _WeightToIndex()
    tensor = torch.tensor([[-1.0, -0.5, 0.0], [0.5, 1.0, 1.5]])
    indices, step = w2i._make_index_single(tensor)
    expected = np.array([[-2, -1, 0], [1, 2, 3]], dtype=np.int32)
    assert np.array_equal(indices, expected)
    assert step == 0.5


def test_make_index_linear_no_grouping():
    w2i = _WeightToIndex()
    linear = nn.Linear(4, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[0.0, 0.5, 1.0, 1.5], [0.5, 1.0, 1.5, 2.0]]))
    indices, step = w2i.make_index(linear)
    expected = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    assert np.array_equal(indices, expected)
    assert step == 0.5


def test_make_index_linear_with_grouping():
    w2i = _WeightToIndex(groupsize=1)
    linear = nn.Linear(4, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[0.0, 0.5, 1.0, 1.5], [0.0, 1.0, 2.0, 3.0]]))
    indices, steps = w2i.make_index(linear)
    # Check shape
    assert indices.shape == (2, 4)
    assert steps.shape == (2,)  # type: ignore
    # Check steps
    assert steps[0] == 0.5  # type: ignore
    assert steps[1] == 1.0  # type: ignore
    # Check actual indices
    expected_indices = np.array(
        [[0, 1, 2, 3], [0, 1, 2, 3]],  # First row: step 0.5  # Second row: step 1.0
        dtype=np.int32,
    )
    assert np.array_equal(indices, expected_indices)


def test_make_index_small_values():
    w2i = _WeightToIndex()
    tensor = torch.tensor([[0.0, 1e-13, 1e-6, 1e-5]])
    linear = nn.Linear(4, 1)
    with torch.no_grad():
        linear.weight.copy_(tensor)
    indices, step = w2i.make_index(linear)
    # 1e-13 should be considered as 0 due to eps threshold, but 1e-6 and 1e-5 should be quantized
    assert step > 0
    # The indices should be [0, 0, 1, 10] or similar (depending on exact implementation)
    # At minimum, first two values should be the same (0)
    assert indices[0, 0] == indices[0, 1]


def test_make_index_large_groupsize():
    w2i = _WeightToIndex(groupsize=8)
    linear = nn.Linear(4, 4)
    with torch.no_grad():
        linear.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.5, 1.0, 1.5],
                    [0.0, 1.0, 2.0, 3.0],
                    [0.0, 0.3, 0.6, 0.9],
                    [0.0, 0.2, 0.4, 0.6],
                ]
            )
        )
    indices, steps = w2i.make_index(linear)
    # With groupsize > number of rows, there should be just one group
    assert steps.shape == (1,)  # type: ignore
    assert indices.shape == (4, 4)

    # Check the actual indices - the step should be 0.1 (smallest delta)
    step = steps[0]  # type: ignore
    assert pytest.approx(step) == 0.1

    # Calculate expected indices based on step 0.1
    expected_indices = np.array(
        [[0, 5, 10, 15], [0, 10, 20, 30], [0, 3, 6, 9], [0, 2, 4, 6]], dtype=np.int32
    )
    assert np.array_equal(indices, expected_indices)


def test_make_index_groupsize_2():
    # Test with groupsize 2 to verify intermediate grouping
    w2i = _WeightToIndex(groupsize=2)
    linear = nn.Linear(4, 4)
    with torch.no_grad():
        linear.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.5, 1.0, 1.5],  # Group 1
                    [0.0, 1.0, 2.0, 3.0],  # Group 1
                    [0.0, 0.3, 0.6, 0.9],  # Group 2
                    [0.0, 0.2, 0.4, 0.6],  # Group 2
                ]
            )
        )
    indices, steps = w2i.make_index(linear)

    # Should have 2 groups
    assert steps.shape == (2,)  # type: ignore
    assert indices.shape == (4, 4)

    # First step should be 0.5 (for the first 2 rows)
    assert pytest.approx(steps[0]) == 0.5  # type: ignore
    # Second step should be 0.1 (for the last 2 rows)
    assert pytest.approx(steps[1]) == 0.1  # type: ignore

    # Expected indices for each group
    expected_indices = np.array(
        [
            [0, 1, 2, 3],  # Group 1, row 1
            [0, 2, 4, 6],  # Group 1, row 2
            [0, 3, 6, 9],  # Group 2, row 1
            [0, 2, 4, 6],  # Group 2, row 2
        ],
        dtype=np.int32,
    )
    assert np.array_equal(indices, expected_indices)
