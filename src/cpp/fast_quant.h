#pragma once
#include <CommonLib/Quant.h>
#include "entropy_model.h"

namespace fast_quant {
    /// Rounds all elements in the given array to its nearest neighbour.
    /// @param w Pointer to array to quantize
    /// @param n Number of elements in w
    /// @return Vector of quantized elements
    std::vector<int32_t> rtn_quantize(const float *w, const size_t n);

    /// @brief Calculates cost in index space
    /// @param w  The weight to quantize in index space
    /// @param i  The index to quantize to
    /// @param lambda Tradeoff parameter between rate and distortion
    /// @param rate_estimation The estimated rate for the index
    /// @return r * lm + (w - i)^2
    float inline cost(const float w, const int32_t i, const float lambda, const float rate_estimation) {
        return rate_estimation * lambda + 0.5 * pow((w - static_cast<float>(i)), 2); // NOLINT
    }

    /// \brief Quantize a single weight. Works completely in indices.
    /// The cost can be calculated as
    ///    cost = rate * lambda + 1/pv * (w - w_q)^2
    /// Make sure to factor in the weight scale in the posterior variance (1/pv)
    /// \param w The weight to quantize
    /// \param max The maximum value the weight can take
    /// \param lambda Scaling factor for the rate term
    /// \param posterior_variance How much the weight can vary
    /// length max - min + 1
    /// \param rate_estimator Entropy model that provides update and estimate functions
    /// \return The quantized weight
    template<typename EstimatorT>
    int32_t quantize_single(float w, const int32_t max, const float lambda,
                            const float posterior_variance,
                            EstimatorT *rate_estimator) {
        float lambda_prime = lambda * posterior_variance;
        float best_cost = std::numeric_limits<float>::max();
        int32_t best_idx = 0;

        // TODO: In some cases, we can probably afford to assume a symmetric entropy model.
        // If yes, this should be changed back.
        for (int32_t i = -max; i <= max; i++) {
            const float rate_estimation = rate_estimator->estimate(i);
            const float c = cost(w, i, lambda_prime, rate_estimation);

            if (c < best_cost) {
                best_cost = c;
                best_idx = i;
            }
        }
        return best_idx;
    }

    std::vector<std::pair<float, int32_t> >
    quantize_single_top_k(float w, int32_t min, int32_t max, float lambda,
                          float posterior_variance, const DeepCabacRateEstimator &rate_estimator,
                          int32_t k);
} // namespace fast_quant

class DeepCabacRdQuantiser {
public:
    DeepCabacRdQuantiser(float lambda, const int32_t min, const int32_t max)
        : lambda(lambda), min(min), max(max) {
    };

    int32_t quantize_single(float w, float posterior_variance);

    int32_t quantize_single(float w, float posterior_variance, int min, int max);

    int32_t *quantize_multiple(const float *W, size_t size,
                               float posterior_variance, int32_t *out_buffer);

    int32_t *quantize_multiple(const float *W, size_t size,
                               const float *posterior_variance,
                               int32_t *out_buffer);

    int32_t *quantize_multiple_permuted(const float *W, size_t size,
                                        const float *posterior_variance,
                                        const int32_t *permutations,
                                        int32_t *out_buffer);

    int32_t *quantize_multiple(const float *W, size_t size,
                               const float *posterior_variance,
                               int32_t min_idx, int32_t max_idx,
                               int32_t *out_buffer);

    int32_t *quantize_multiple(const float *W, size_t size,
                               float posterior_variance, int32_t min_idx,
                               int32_t max_idx, int32_t *out_buffer);

    float estimate_rate(int32_t q);

    float lambda;
    int32_t min;
    int32_t max;

private:
    DeepCabacRateEstimator rate_estimator;
};
