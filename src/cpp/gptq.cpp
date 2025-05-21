#include "gptq.h"
#include <fstream>
#include <queue>
#include <utility>

namespace fast_quant {
    GPTQ::GPTQ(GPTQConfig config) : config_(std::move(config)) {
        // Validate configuration
        if (config_.max_idx == 0) {
            throw std::invalid_argument("Configuration must contain max_idx > 0");
        }

        // Note: We'll initialize the rate estimator on first use to avoid
        // creating it unnecessarily
    }

    void GPTQ::quantize_layerwise(
        float *weights,
        const float *hessian_chol,
        const float lm,
        size_t rows,
        size_t cols,
        int32_t *out_buffer
    ) {
        if (config_.use_beam_search) {
            // Use beam search variant
            DeepCabacRateEstimator quant;
            for (size_t i = 0; i < rows; i++) {
                float *w_row = weights + i * cols;
                int32_t *out_row = out_buffer + i * cols;
                quant = quantize_row_beam_search(
                    w_row, hessian_chol, quant, lm, cols, out_row);
            }
        } else {
            // Use standard GPTQ with rate estimation
            if (!rate_estimator_) {
                // Create the rate estimator on first use
                rate_estimator_ = createEstimator(
                    config_.entropy_model_type, weights, rows * cols, config_.max_idx, config_.entropy_model_options);
            }

            if (config_.scan_major_order == "row") {
                for (size_t i = 0; i < rows; i++) {
                    float *w_row = weights + i * cols;
                    int32_t *out_row = out_buffer + i * cols;
                    quantize_row(
                        w_row, hessian_chol, rate_estimator_.get(), lm, cols, out_row);
                }
            }
            if (config_.scan_major_order == "col") {
                for (size_t i = 0; i < cols; i++) {
                    quantize_col(
                        weights, i, hessian_chol, rate_estimator_.get(), lm, cols, rows, out_buffer);
                }
            }
        }
    }

    void GPTQ::quantize_rowwise(
        float *weights,
        const float *hessian_chol,
        const float *lambdas,
        size_t rows,
        size_t cols,
        int32_t *out_buffer
    ) {
        if (config_.use_beam_search) {
            // Use beam search variant with row-specific lambdas
            DeepCabacRateEstimator quant;
            for (size_t i = 0; i < rows; i++) {
                float *w_row = weights + i * cols;
                int32_t *out_row = out_buffer + i * cols;
                quant = quantize_row_beam_search(
                    w_row, hessian_chol, quant, lambdas[i], cols, out_row);
            }
        } else {
            // Use standard GPTQ with rate estimation and row-specific lambdas
            if (!rate_estimator_) {
                // Create the rate estimator on first use
                rate_estimator_ = createEstimator(
                    config_.entropy_model_type, weights, rows * cols, config_.max_idx, config_.entropy_model_options);
            }

            for (size_t i = 0; i < rows; i++) {
                float *w_row = weights + i * cols;
                int32_t *out_row = out_buffer + i * cols;
                quantize_row(
                    w_row, hessian_chol, rate_estimator_.get(), lambdas[i], cols, out_row);
            }
        }
    }

    // Implementation of beam search variant
    DeepCabacRateEstimator GPTQ::quantize_row_beam_search(
        const float *w_row,
        const float *hessian_chol,
        const DeepCabacRateEstimator &estimator,
        float lambda,
        size_t n,
        int32_t *out_row_ptr
    ) {
        struct Beam {
            float loss;
            DeepCabacRateEstimator entropy_model;
            std::vector<float> weights;

            bool operator<(const Beam &other) const { return loss < other.loss; }
            bool operator>(const Beam &other) const { return loss > other.loss; }
        };

        auto update_beam = [this, hessian_chol, n](const Beam &old_beam, float loss, int32_t q, int32_t i) {
            Beam new_beam = old_beam;
            new_beam.loss += loss;

            // Update weights
            float *weights = new_beam.weights.data();
            this->weight_update(weights, hessian_chol, i, n, q);

            // Update model and record quantized value
            new_beam.entropy_model.update(q);
            new_beam.weights[i] = static_cast<float>(q);

            return new_beam;
        };

        // Initialize beams
        std::vector<Beam> beams;
        beams.push_back({0.0f, estimator, std::vector<float>(w_row, w_row + n)});

        const size_t num_beams = config_.num_beams;

        for (int32_t i = 0; i < static_cast<int32_t>(n); i++) {
            std::priority_queue<Beam, std::vector<Beam>, std::greater<> > candidate_beams;
            auto Hqq = hessian_chol[i + n * i];

            for (const Beam &beam: beams) {
                // Get top-k quantization options for this weight
                const auto top_m_q = quantize_single_top_k(
                    beam.weights[i],
                    -static_cast<int32_t>(config_.max_idx),
                    static_cast<int32_t>(config_.max_idx),
                    lambda,
                    Hqq,
                    beam.entropy_model,
                    num_beams
                );

                // Create new beams for each quantization option
                for (const auto &[loss, q]: top_m_q) {
                    Beam new_beam = update_beam(beam, loss, q, i);
                    candidate_beams.push(std::move(new_beam));
                }
            }

            // Keep only the best beams
            beams.clear();
            beams.reserve(num_beams);
            for (size_t j = 0; j < num_beams && !candidate_beams.empty(); j++) {
                beams.push_back(candidate_beams.top());
                candidate_beams.pop();
            }
        }

        // Select the best beam and copy results
        if (!beams.empty()) {
            const Beam &best_beam = beams[0];
            for (size_t i = 0; i < n; i++) {
                out_row_ptr[i] = static_cast<int32_t>(best_beam.weights[i]);
            }
            return best_beam.entropy_model;
        }

        // Fallback (should not happen)
        return estimator;
    }
} // namespace fast_quant
