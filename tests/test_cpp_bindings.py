from nn_compression._core import DeepCabacRdQuantiser
import numpy as np


def test_smoke_layer_quant_single_value():
    q = DeepCabacRdQuantiser(1.0, -5, 5)
    q.quantize(3, 3)


def test_smoke_layer_quant_multiple_values():
    q = DeepCabacRdQuantiser(1.0, -5, 5)
    q.quantize(np.array([1, 2, 3]), 3)


def test_smoke_layer_quant_multiple_dimensions():
    q = DeepCabacRdQuantiser(1.0, -5, 5)
    q.quantize(np.array([[1, 2, 3], [4, 5, 6]]), 3)


def test_smoke_layer_quant_multiple_posterior_values():
    q = DeepCabacRdQuantiser(1.0, -5, 5)
    q.quantize(np.array([1, 2, 3]), np.array([1, 2, 3]))


def test_smoke_layer_quant_multiple_posterior_values_multidimension():
    q = DeepCabacRdQuantiser(1.0, -5, 5)
    q.quantize(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]))


def test_smoke_layer_quant_permutations():
    q = DeepCabacRdQuantiser(1.0, -5, 5)

    q.quantize_permuted(
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([5, 2, 3, 4, 1, 0]),
    )


def test_permutations_change_order():
    # We have lm = 1, pv = 1
    # We optimize:
    #
    # argmin_{wq in G} (w-wq)^2 / pv + lm * entropy(wq)
    #
    # Therefore, by setting the closest grid point (0) to a very expensive position in terms of entropy,
    # we expect w to be quantized to a cheaper point (1 in this case).
    q = DeepCabacRdQuantiser(1.0, -20, 20)

    qx = q.quantize_permuted(
        np.array([0.1, 3, 4, 5]),
        np.array([1, 1, 1, 1 / 20_000]),
        # interpret permutations like this:
        # 0 costs as much as 1, 1 costs as much as 20, ...
        np.array(
            [20, 0, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        ),
    )

    assert np.allclose(qx, np.array([1, 1, 1, 5]))


def test_quantiser_correct_idx():
    # Equation for Rate-distortion:
    # RD = Rate * lm + 1/pv * distortion
    delta = 0.1
    min_idx = -5
    max_idx = 5
    lm = 2
    # Setup: There are 5 quantisation levels,
    # [-0.5, -0.4, ... 0.4, 0.5]
    #
    q = DeepCabacRdQuantiser(lm, min_idx, max_idx)
    # bit estimations are
    rate_estimations = [7, 6, 5, 4, 3, 1, 3, 4, 5, 6, 7]
    assert len(rate_estimations) == max_idx - min_idx + 1

    # calculating everything in weight space
    w = 0.23
    points_wspace = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    dists = [(w - x) ** 2 for x in points_wspace]

    pv = 0.3
    rd_objectives = [
        rate_estimations[i] * lm + dists[i] / pv for i in range(len(rate_estimations))
    ]
    w_opt = points_wspace[rd_objectives.index(min(rd_objectives))]

    # repeat in index space
    w_idxspace = w / delta
    points_idxspace = list(range(min_idx, max_idx + 1))
    dists_idxspace = [(w_idxspace - x) ** 2 * delta**2 for x in points_idxspace]
    rd_objectives_idxspace = [
        rate_estimations[i] * lm + dists_idxspace[i] / pv
        for i in range(len(rate_estimations))
    ]
    w_opt_idxspace = (
        points_idxspace[rd_objectives_idxspace.index(min(rd_objectives))] * delta
    )

    assert w_opt == w_opt_idxspace
    # Test c++ implementation against the the implementation above

    w_opt_idxspace_cpp = q.quantize(w, pv) * delta
    assert w_opt_idxspace_cpp == w_opt
