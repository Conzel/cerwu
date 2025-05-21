#include "entropy_model.h"

#include <map>
#include <numeric>
#include <variant>

std::vector<int32_t> rtn_quantize(const float *w, const size_t n) {
  std::vector out(n, 0);
  for (int32_t i = 0; i < n; i++) {
    out[i] = static_cast<int32_t>(std::round(w[i]));
  }
  return out;
}

float variance(const float *w, size_t n) {
  // variable to store sum of the given w
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += w[i];
  }
  float mean = sum / n;

  float sum_squared_diffs = 0;
  for (int i = 0; i < n; i++) {
    sum_squared_diffs += (w[i] - mean) * (w[i] - mean);
  }
  return sum_squared_diffs / (n - 1);
}

// Calculate the cumulative distribution function (CDF) of the standard normal
// distribution
float32_t inline normalCDF(float x) {
  // Using the error function approximation
  return 0.5 * (1 + std::erf(x / std::sqrt(2.0)));
}

// Calculate the probability that a value falls within [a, b] in a Gaussian
// distribution with mean mu and standard deviation sigma
float32_t gaussianIntervalProbability(float32_t a, float32_t b, float32_t mu,
                                      float32_t sigma) {
  // Convert to standard normal by applying z = (x - mu) / sigma
  float32_t standardized_a = (a - mu) / sigma;
  float32_t standardized_b = (b - mu) / sigma;

  // P(a ≤ X ≤ b) = CDF(b) - CDF(a)
  return normalCDF(standardized_b) - normalCDF(standardized_a);
}

RegularizedModel::RegularizedModel(const float *w, int n, int max_idx,
                                   float gamma, const std::string &base_model)
    : gamma(gamma) {
  this->model = createEstimator(base_model, w, n, max_idx, {});
}

float32_t RegularizedModel::estimate(int32_t quantisation_index) const {
  const auto est = this->model->estimate(quantisation_index) -
                   gamma / 2 * quantisation_index * quantisation_index;
  return est;
}

void RegularizedModel::update(int32_t quantisation_index) {
  return this->model->update(quantisation_index);
}

QuantizedGaussian::QuantizedGaussian(const float *w, int n, int max_idx)
    : max_idx(max_idx) {
  rates.resize(2 * max_idx + 1);
  const double log2_e = 1.0 / std::log(2.0);
  auto s = std::sqrt(variance(w, n));
  for (int i = -max_idx; i <= max_idx; i++) {
    rates[i + max_idx] =
        -std::log(gaussianIntervalProbability(i - 0.5, i + 0.5, 0, s)) * log2_e;
  }
}

float32_t QuantizedGaussian::estimate(int32_t quantisation_index) const {
  return rates[quantisation_index + max_idx];
}

void QuantizedGaussian::update(int32_t quantisation_index) {}

StaticShannon::StaticShannon(int32_t max_idx) {
  this->counts = std::vector<size_t>(2 * max_idx + 1, 1);
  this->total = counts.size();
  this->max_idx = max_idx;
  this->frozen = false;
}

StaticShannon::StaticShannon(const int *q, int n, int max_idx) {
  if (max_idx < 0) {
    throw std::invalid_argument("Passed negative max_idx to StaticShannon: " +
                                std::to_string(max_idx));
  }
  this->counts = std::vector<size_t>(2 * max_idx + 1, 1);
  this->total = n + counts.size();
  this->max_idx = max_idx;
  this->frozen = false;

  for (int i = 0; i < n; i++) {
    int index = q[i] + max_idx; // Ensure this is within bounds
    if (index >= 0 && index < counts.size()) {
      counts[index] += 1;
    } else {
      throw std::out_of_range("Index " + std::to_string(index - max_idx) +
                              " out of bounds for max_idx " +
                              std::to_string(max_idx));
    }
  }
}

void StaticShannon::freeze() { this->frozen = true; }

float StaticShannon::estimate(int32_t quantisation_index) const {
  auto index = quantisation_index + max_idx;
  if (index < counts.size()) {
    auto p = counts[index];
    if (p > 0) {
      return -log2(static_cast<float>(p) / static_cast<float>(this->total));
    } else {
      throw std::domain_error("Probability is zero for quantisation index " +
                              std::to_string(quantisation_index));
    }
  } else {
    throw std::out_of_range(
        "Quantisation index " + std::to_string(quantisation_index) +
        " out of bounds for counts size " + std::to_string(counts.size()));
  }
}

void StaticShannon::update(int32_t quantisation_index) {
  if (this->frozen) {
    return;
  }
  counts[quantisation_index + this->max_idx] += 1;
  this->total += 1;
}

SwitchingShannon::SwitchingShannon(const int *q, const int n,
                                   const int max_idx) {
  this->state = 0;
  this->max_idx = max_idx;
  this->frozen = false;

  std::vector<std::vector<int>> parts(3);
  for (int i = 0; i < n; i++) {
    parts[state].push_back(q[i]);
    update_state(q[i]);
  }
  for (auto &p : parts) {
    if (!p.empty()) {
      this->entropy_models.emplace_back(p.data(), p.size(), this->max_idx);
    } else {
      this->entropy_models.emplace_back(this->max_idx);
    }
  }
}

float SwitchingShannon::estimate(int32_t quantisation_index) const {
  return this->entropy_models[this->state].estimate(quantisation_index);
}

void SwitchingShannon::update_state(const int32_t i) {
  if (i == 0) {
    state = 0;
  } else if (i >= 1) {
    state = 1;
  } else {
    state = 2;
  }
}

void SwitchingShannon::update(const int32_t quantisation_index) {
  if (!this->frozen) {
    this->entropy_models[this->state].update(quantisation_index);
  }
  update_state(quantisation_index);
}

void SwitchingShannon::freeze() { this->frozen = true; }

float32_t DeepCabacRateEstimator::estimate(int32_t quantisation_index) const {
  return this->est->estimate(quantisation_index);
}

void DeepCabacRateEstimator::update(int32_t quantisation_index) {
  return this->est->update(quantisation_index);
}

float getNumericValue(const std::variant<int, float, std::string> &var) {
  if (std::holds_alternative<int>(var)) {
    return static_cast<float>(std::get<int>(var));
  } else if (std::holds_alternative<float>(var)) {
    return std::get<float>(var);
  } else {
    // Use std::stof to convert string to float
    return std::stof(std::get<std::string>(var));
  }
}

std::unique_ptr<EstimatorInterface> createEstimator(
    const std::string &type, const float *w, size_t n, const size_t max_idx,
    const std::map<std::string, std::variant<int, float, std::string>>
        &options) {
  if (type == "deepcabac") {
    return std::make_unique<DeepCabacRateEstimator>();
  }
  if (type == "shannon") {
    auto q = rtn_quantize(w, n);
    auto e = std::make_unique<StaticShannon>(q.data(), n, max_idx);
    e->freeze();
    return e;
  }
  if (type == "shannon_context") {
    auto q = rtn_quantize(w, n);
    auto e = std::make_unique<SwitchingShannon>(q.data(), n, max_idx);
    e->freeze();
    return e;
  }
  if (type == "uniform") {
    auto e = std::make_unique<Uniform>();
    return e;
  }
  if (type == "gaussian") {
    auto e = std::make_unique<QuantizedGaussian>(w, n, max_idx);
    return e;
  }
  if (type == "regularized") {
    auto base_model = std::get<std::string>(options.at("base_model"));
    if (base_model == "regularized") {
      throw std::invalid_argument("Put recursive definition of models");
    }
    float gamma;
    if (options.find("gamma") != options.end()) {
      gamma = getNumericValue(options.at("gamma"));
    } else {
      gamma = 1 / (std::log(2) * variance(w, n));
    }

    auto e =
        std::make_unique<RegularizedModel>(w, n, max_idx, gamma, base_model);
    return e;
  }
  throw std::invalid_argument("Unknown estimator type: " + type);
}
