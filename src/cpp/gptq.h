// gptq.h
#pragma once
#include <fstream>
#include <map>

#include "entropy_model.h"
#include "fast_quant.h"
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace fast_quant {
  /**
   * @brief Configuration options for GPTQ algorithm.
   */
  struct GPTQConfig {
    // Core parameters
    size_t max_idx = 0; // Maximum index value for quantization grid

    // Algorithm selection
    bool use_beam_search = false; // Whether to use beam search
    size_t num_beams = 1; // Number of beams when using beam search

    // Rate estimation
    std::string entropy_model_type =
        "deepcabac"; // Entropy model type: "deepcabac", "shannon",
    // "shannon_context"
    std::map<std::string, std::variant<int, float, std::string> >
    entropy_model_options = {};

    // Debug options
    bool write_debug_output = false;
    std::string debug_file = "gptq_debug.txt";

    // Ordering
    std::string scan_major_order = "row";
  };

  /**
   * @brief A unified GPTQ (Gradient-based Post-Training Quantization)
   * implementation.
   *
   * This class handles all GPTQ variants through a single interface,
   * selecting the appropriate implementation based on configuration.
   */
  class GPTQ {
  public:
    /**
     * @brief Construct a new GPTQ object with the given configuration
     *
     * @param config Configuration options
     */
    explicit GPTQ(GPTQConfig config);

    /**
     * @brief Quantize weights using a single lambda value for all rows
     *
     * @param weights Pointer to weight matrix (shape: rows × cols)
     * @param hessian_chol Pointer to Cholesky of inverse Hessian (shape: cols ×
     * cols)
     * @param rows Number of rows in weight matrix
     * @param cols Number of columns in weight matrix
     * @param out_buffer Output buffer for quantized weights (shape: rows × cols)
     */
    void quantize_layerwise(float *weights, const float *hessian_chol, float lm,
                            size_t rows, size_t cols, int32_t *out_buffer);

    /**
     * @brief Quantize weights using row-specific lambda values
     *
     * @param weights Pointer to weight matrix (shape: rows × cols)
     * @param hessian_chol Pointer to Cholesky of inverse Hessian (shape: cols ×
     * cols)
     * @param lambdas Pointer to lambda values (shape: rows)
     * @param rows Number of rows in weight matrix
     * @param cols Number of columns in weight matrix
     * @param out_buffer Output buffer for quantized weights (shape: rows × cols)
     */
    void quantize_rowwise(float *weights, const float *hessian_chol,
                          const float *lambdas, size_t rows, size_t cols,
                          int32_t *out_buffer);

    /**
     * @brief Quantize a single row using GPTQ
     *
     * @param w_row Pointer to a single row of weights
     * @param hessian_chol Pointer to Cholesky of inverse Hessian
     * @param rate_estimator Rate estimator to use
     * @param lambda RD trade-off parameter
     * @param n Number of columns (elements in w_row)
     * @param out_row_ptr Output buffer for the quantized row
     */
    template<typename EstimatorT>
    void quantize_row(float *w_row, const float *hessian_chol,
                      EstimatorT *rate_estimator, float lambda, size_t n,
                      int32_t *out_row_ptr);

    template<class EstimatorT>
    void quantize_col(float *w, int32_t col_idx, const float *hessian_chol, EstimatorT *rate_estimator, float lambda,
                      size_t n, size_t m, int32_t *out_ptr);

  private:
    GPTQConfig config_;
    std::unique_ptr<EstimatorInterface> rate_estimator_;

    // Implementation methods for different algorithm variants

    /**
     * @brief Quantize a row using beam search
     */
    DeepCabacRateEstimator
    quantize_row_beam_search(const float *w_row, const float *hessian_chol,
                             const DeepCabacRateEstimator &estimator,
                             float lambda, size_t n, int32_t *out_row_ptr);

    /**
     * @brief Weight update helper for GPTQ
     */
    static void weight_update(float *w, const float *H, int32_t i, size_t n,
                              int32_t q);
  };

  // Inline implementation of weight_update (keeping this inline for performance)
  inline void GPTQ::weight_update(float *w, const float *H, const int32_t i,
                                  const size_t n, const int32_t q) {
    // i is the current column, n is the maximum number of columns
    auto scaling = (w[i] - q) / H[i + n * i];
    // iterate over all further columns
    for (int32_t j = i + 1; j < n; j++) {
      // if (H[j + n * i] > 0) {
      //     std::cout << H[j + n * i] << std::endl;
      // }
      w[j] -= scaling * H[j + n * i];
    }
  }

  // Template implementation for quantize_row
  template<typename EstimatorT>
  void GPTQ::quantize_row(float *w_row, const float *hessian_chol,
                          EstimatorT *rate_estimator, float lambda, size_t n,
                          int32_t *out_row_ptr) {
    std::ofstream debug_file;
    if (config_.write_debug_output) {
      debug_file.open(config_.debug_file, std::ofstream::trunc);
      if (!debug_file) {
        // Just log the error but continue processing
        std::cerr << "Error: could not open " << config_.debug_file
            << " for writing debug info." << std::endl;
      }
    }

    for (size_t i = 0; i < n; i++) {
      auto cqq = hessian_chol[i + n * i];
      auto q = quantize_single<EstimatorT>(w_row[i], config_.max_idx, lambda,
                                           cqq * cqq, rate_estimator);

      rate_estimator->update(q);
      out_row_ptr[i] = q;
      weight_update(w_row, hessian_chol, i, n, q);

      // Write debug info if enabled
      if (config_.write_debug_output && debug_file) {
        for (size_t j = (i + 1); j < n; j++) {
          debug_file << w_row[j];
          if (j < n - 1)
            debug_file << ",";
        }
        debug_file << "\n";
      }
    }
  }

  // Template implementation for quantize_row
  template<typename EstimatorT>
  void GPTQ::quantize_col(float *w, int32_t col_idx,
                          const float *hessian_chol, EstimatorT *rate_estimator, float lambda,
                          size_t n, size_t m, int32_t *out_ptr) {
    // n: number of cols

    auto cqq = hessian_chol[col_idx + n * col_idx];
    for (size_t i = 0; i < m; i++) {
      auto w_idx = col_idx + n * i;
      auto q = quantize_single<EstimatorT>(w[w_idx], config_.max_idx, lambda,
                                           cqq * cqq, rate_estimator);
      rate_estimator->update(q);
      out_ptr[w_idx] = q;
      weight_update(w + n * i, hessian_chol, col_idx, n, q);
    };
  }
} // namespace fast_quant
