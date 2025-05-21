#pragma once
#include "CommonLib/Quant.h"
#include <cstdint>
#include <map>
#include <variant>
#include <vector>

class RegularizedModel : public EstimatorInterface {
public:
  RegularizedModel(const float *w, int n, int max_idx, float gamma,
                   const std::string &base_model);

  [[nodiscard]] float32_t estimate(int32_t quantisation_index) const override;

  void update(int32_t quantisation_index) override;

private:
  std::unique_ptr<EstimatorInterface> model;
  float gamma;
};

class QuantizedGaussian : public EstimatorInterface {
public:
  QuantizedGaussian(const float *w, int n, int max_idx);

  [[nodiscard]] float32_t estimate(int32_t quantisation_index) const override;

  void update(int32_t quantisation_index) override;

private:
  std::vector<float32_t> rates;
  int32_t max_idx;
};

class Uniform : public EstimatorInterface {
public:
  [[nodiscard]] float32_t estimate(int32_t quantisation_index) const override {
    return 0.0;
  };

  void update(int32_t quantisation_index) override {};
};

class DeepCabacRateEstimator : public EstimatorInterface {
public:
  DeepCabacRateEstimator() { this->est = std::make_shared<RateEstimation>(); }

  [[nodiscard]] float32_t estimate(int32_t quantisation_index) const override;

  void update(int32_t quantisation_index) override;

private:
  std::shared_ptr<RateEstimation> est;
};

class StaticShannon : public EstimatorInterface {
public:
  explicit StaticShannon(int32_t max_idx);

  /// Initializes an entropy model that is based on the Shannon-Entropy.
  ///
  /// @param q Pointer to (pre)-quantized weights
  /// @param n Number of elements in q
  /// @param max_idx Maximum index for quantization
  StaticShannon(const int *q, int n, int max_idx);

  [[nodiscard]] float estimate(int32_t quantisation_index) const override;

  /// Freezes the entropy model in place. Can be used to make it static after
  /// construction.
  void freeze();

  void update(int32_t quantisation_index) override;

private:
  std::vector<size_t> counts;
  size_t total;
  size_t max_idx;
  bool frozen;
};

class SwitchingShannon : public EstimatorInterface {
public:
  SwitchingShannon(const int *q, int n, int max_idx);

  [[nodiscard]] float estimate(int32_t quantisation_index) const override;

  void freeze();

  void update(int32_t quantisation_index) override;

  void update_state(int32_t i);

private:
  std::vector<StaticShannon> entropy_models;
  size_t state;
  size_t max_idx;
  bool frozen;
};

// Factory function to create the appropriate estimator
///
/// @param type The estimator to generate. May be "deepcabac", "shannon", or
/// "shannon_context".
/// @param w Pointer to weights that the estimator may use to build itself.
/// @param n Number of weights at the pointer
/// @return Pointer to an EstimatorInterface
std::unique_ptr<EstimatorInterface> createEstimator(
    const std::string &type, const float *w, size_t n, size_t max_idx,
    const std::map<std::string, std::variant<int, float, std::string>>
        &options);
