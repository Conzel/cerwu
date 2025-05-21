#include "fast_quant.h"
#include "CommonLib/Quant.h"
#include <queue>

namespace fast_quant {
  /// Rounds all elements in the given array to its nearest neighbour.
  /// @param w Pointer to array to quantize
  /// @param n Number of elements in w
  /// @return Vector of quantized elements

  int32_t quantize_single_bisection(float w, int32_t min, int32_t max,
                                    float lambda, float posterior_variance,
                                    DeepCabacRateEstimator &rate_estimator) {
    float lambda_prime = lambda * posterior_variance;
    float best_cost = std::numeric_limits<float>::max();
    int32_t best_idx = 0;
    // We are employing a bisection strategy:
    // - Assumption: The entropy is minimized for x = 0
    // - Assumption: Entropy is symmetric
    // - Distortion is quadratic centered around th weight
    // -> Search between 0 and first grid point larger than w in abs
    // the cost function is convex, so we can stop when the cost increases

    int sign = w < 0 ? -1 : 1;
    w = w * sign;
    int32_t end = static_cast<int32_t>(std::ceil(w));

    for (int32_t i = 0; i <= end; i++) {
      float rate_estim = rate_estimator.estimate(sign * i);
      float c = cost(w, i, lambda_prime, rate_estim);

      if (c < best_cost) {
        best_cost = c;
        best_idx = i;
      } else {
        break;
      }
    }
    return best_idx * sign;
  }

  /// @brief Quantise a single parameter value according to RD cost in index
  /// space, rate*lm + distortion / posterior_variance
  /// @param w The weight to quantise in parameter space
  /// @param min Minimum index value
  /// @param max Maximum index value
  /// @param lambda Trade-off parameter between rate and distortion (rate *
  /// lambda
  /// + distortion)
  /// @param posterior_variance Parameter that determines how much the weight
  /// can vary
  /// @param rate_estimations
  /// @return
  std::vector<std::pair<float, int32_t> >
  quantize_single_top_k(float w, int32_t min, int32_t max, float lambda,
                        float posterior_variance, const DeepCabacRateEstimator &rate_estimator,
                        int32_t k) {
    float lambda_prime = lambda * posterior_variance;
    std::priority_queue<std::pair<float, int32_t>,
          std::vector<std::pair<float, int32_t> >,
          std::greater<std::pair<float, int32_t> > >
        pq;

    int sign = w < 0 ? -1 : 1;
    w = w * sign;
    int32_t end = std::min(max, static_cast<int32_t>(std::round(w)));

    for (int32_t i = 0; i <= end; i++) {
      int32_t i_signed = i * sign;
      float rate_estim = rate_estimator.estimate(i_signed);
      float c = cost(w * sign, i_signed, lambda_prime, rate_estim);
      pq.push(std::make_pair(c, i_signed));
    }

    std::vector<std::pair<float, int32_t> > result;
    for (int32_t i = 0; i < k && !pq.empty(); i++) {
      auto res = pq.top();
      result.push_back(res);
      pq.pop();
    }
    return result;
  }

  /// @brief Quantise a single parameter value according to RD cost in index
  /// space, rate*lm + distortion / posterior_variance
  /// @param w The weight to quantise in parameter space
  /// @param min Minimum index value
  /// @param max Maximum index value
  /// @param lambda Trade-off parameter between rate and distortion (rate *
  /// lambda
  /// + distortion)
  /// @param posterior_variance Parameter that determines how much the weight
  /// can vary
  /// @param permutation Array of length max that contains permutations for
  /// the rate-estimator, allowing for the actual bit-rate to be distributed
  /// along the weight space differently. We assume the permutations to be
  /// symmetric and to start at 0.
  /// @param rate_estimations
  /// @return
  int32_t quantize_single_permuted(float w, int32_t min, int32_t max,
                                   float lambda, float posterior_variance,
                                   const int32_t *permutations,
                                   DeepCabacRateEstimator &rate_estimator) {
    float lambda_prime = lambda * posterior_variance;
    float best_cost = std::numeric_limits<float>::max();
    int32_t best_idx = 0;
    for (int32_t i = min; i <= max; i++) {
      int sign = i < 0 ? -1 : 1;
      float rate_estim = rate_estimator.estimate(sign * permutations[sign * i]);
      float c = cost(w, i, lambda_prime, rate_estim);

      if (c < best_cost) {
        best_cost = c;
        best_idx = i;
      }
    }
    return best_idx;
  }
} // namespace fast_quant

int32_t DeepCabacRdQuantiser::quantize_single(float w,
                                              float posterior_variance) {
  return this->quantize_single(w, posterior_variance, this->min, this->max);
}

int32_t DeepCabacRdQuantiser::quantize_single(float w, float posterior_variance,
                                              int min_, int max_) {
  if (min_ >= max_) {
    std::cout << "Illegal min and max values: (" << min << "," << max << ")"
        << std::endl;
    std::cout << "Called with w: " << w << " pv: " << posterior_variance
        << std::endl;
    throw std::range_error("min must be strictly less than max");
  }
  if (this->lambda < 0) {
    throw std::invalid_argument("Lambda is smaller than 0");
  }
  // Short-cut: Skip RD-Quantization
  if (this->lambda == 0) {
    return std::clamp(static_cast<int32_t>(std::round(w)), min_, max_);
  }
  int32_t best_idx = fast_quant::quantize_single(
    w, max_, this->lambda, posterior_variance, &this->rate_estimator);
  rate_estimator.update(best_idx);
  return best_idx;
}

int32_t *
DeepCabacRdQuantiser::quantize_multiple(const float *W, size_t size,
                                        const float *posterior_variance,
                                        int32_t *out_buffer) {
  for (size_t i = 0; i < size; ++i) {
    out_buffer[i] = quantize_single(W[i], posterior_variance[i]);
  }
  return out_buffer;
}

int32_t *DeepCabacRdQuantiser::quantize_multiple_permuted(
  const float *W, size_t size, const float *posterior_variance,
  const int32_t *permutations, int32_t *out_buffer) {
  for (size_t i = 0; i < size; ++i) {
    auto best_idx = fast_quant::quantize_single_permuted(
      W[i], this->min, this->max, this->lambda, posterior_variance[i],
      permutations, this->rate_estimator);
    out_buffer[i] = best_idx;
    rate_estimator.update(best_idx);
  }
  return out_buffer;
}

int32_t *DeepCabacRdQuantiser::quantize_multiple(const float *W, size_t size,
                                                 float posterior_variance,
                                                 int32_t *out_buffer) {
  for (size_t i = 0; i < size; ++i) {
    out_buffer[i] = quantize_single(W[i], posterior_variance);
  }
  return out_buffer;
}

int32_t *DeepCabacRdQuantiser::quantize_multiple(
  const float *W, size_t size, const float *posterior_variance,
  const int32_t min_idx, const int32_t max_idx, int32_t *out_buffer) {
  for (size_t i = 0; i < size; ++i) {
    out_buffer[i] =
        quantize_single(W[i], posterior_variance[i], min_idx, max_idx);
  }
  return out_buffer;
}

int32_t *DeepCabacRdQuantiser::quantize_multiple(const float *W, size_t size,
                                                 float posterior_variance,
                                                 int32_t min_idx,
                                                 int32_t max_idx,
                                                 int32_t *out_buffer) {
  for (size_t i = 0; i < size; ++i) {
    out_buffer[i] = quantize_single(W[i], posterior_variance, min_idx, max_idx);
  }
  return out_buffer;
}

float DeepCabacRdQuantiser::estimate_rate(int32_t q) {
  return this->rate_estimator.estimate(q);
}
