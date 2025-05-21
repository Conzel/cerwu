#include "CommonLib/Quant.h"
#include "cpp/entropy_model.h"
#include "cpp/fast_quant.h"
#include "cpp/gptq.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <variant>

#define PYBIND11_DETAILED_ERROR_MESSAGES

namespace py = pybind11;

int absmax(const int *arr, const size_t size) {
  if (size == 0)
    return 0; // Handle empty array case
  int max_val = arr[0];

  for (size_t i = 1; i < size; ++i) {
    if (std::abs(arr[i]) > std::abs(max_val)) {
      max_val = arr[i];
    }
  }
  return std::abs(max_val);
}

float *intToFloat(const int *arr, size_t size) {
  auto floatArr = new float[size]; // Allocate new float array
  for (size_t i = 0; i < size; ++i) {
    floatArr[i] = static_cast<float>(arr[i]);
  }
  return floatArr;
}

class EntropyModelPy : EstimatorInterface {
public:
  explicit EntropyModelPy(const std::string &entropy_model,
                          const py::array_t<int32_t> &q,
                          const py::dict &options_dict = py::dict()) {
    auto winfo = q.request();
    auto qptr = static_cast<int *>(winfo.ptr);
    int32_t max_idx;
    if (options_dict.contains("max_idx")) {
      max_idx = options_dict["max_idx"].cast<int32_t>();
    } else {
      max_idx = absmax(qptr, winfo.size);
    }
    auto q_as_float = intToFloat(qptr, winfo.size);

    // Convert Python dict to C++ map
    std::map<std::string, std::variant<int, float, std::string> > options;
    for (auto item: options_dict) {
      std::string key = py::str(item.first);
      auto value = item.second;

      if (py::isinstance<py::int_>(value)) {
        options[key] = value.cast<int>();
      } else if (py::isinstance<py::float_>(value)) {
        options[key] = value.cast<float>();
      } else if (py::isinstance<py::str>(value)) {
        options[key] = value.cast<std::string>();
      }
    }

    // Set max_idx in options
    options["max_idx"] = max_idx;

    estimator = createEstimator(entropy_model, q_as_float, winfo.size, max_idx,
                                options);
    free(q_as_float);
  };

  float32_t estimate_tensor(const py::array_t<int32_t> &q) {
    auto winfo = q.request();
    auto qptr = static_cast<int *>(winfo.ptr);
    float32_t sum = 0;
    for (int32_t i = 0; i < winfo.size; i++) {
      sum += this->estimator->estimate(qptr[i]);
      this->estimator->update(qptr[i]);
    }
    return sum;
  }

  float32_t estimate(int32_t quantisation_index) const override {
    return this->estimator->estimate(quantisation_index);
  }

  void update(int32_t quantisation_index) override {
    this->estimator->update(quantisation_index);
  };

private:
  std::unique_ptr<EstimatorInterface> estimator;
};

class DeepCabacRdQuantiserPy {
public:
  DeepCabacRdQuantiserPy(float lambda, int32_t min, int32_t max)
    : quantiser(lambda, min, max) {
  }

  int32_t quantize_single(float W, float posterior_variance) {
    return quantiser.quantize_single(W, posterior_variance);
  }

  py::array_t<int32_t> quantize_multiple(py::array_t<float> W,
                                         float posterior_variance,
                                         int32_t min_idx, int32_t max_idx) {
    auto winfo = W.request();
    auto w_ptr = (float *) winfo.ptr;

    auto quantized = py::array_t<int32_t, py::array::c_style>(winfo.size);
    int32_t *raw_ptr = (int32_t *) quantized.request().ptr;

    quantiser.quantize_multiple(w_ptr, winfo.size, posterior_variance, min_idx,
                                max_idx, raw_ptr);
    return quantized;
  }

  py::array_t<int32_t>
  quantize_multiple_permuted(py::array_t<float> W,
                             py::array_t<float> posterior_variance,
                             py::array_t<int32_t> permutations) {
    auto winfo = W.request();
    auto w_ptr = (float *) winfo.ptr;

    auto pv_info = posterior_variance.request();
    auto pv_ptr = (float *) pv_info.ptr;

    auto perm_info = permutations.request();
    auto perm_ptr = (int32_t *) perm_info.ptr;

    auto quantized = py::array_t<int32_t, py::array::c_style>(winfo.size);
    int32_t *raw_ptr = (int32_t *) quantized.request().ptr;

    if (winfo.size != pv_info.size) {
      std::cout << winfo.size << " " << pv_info.size << std::endl;
      throw std::range_error(
        "W and posterior_variance must have the same size");
    }

    if (quantiser.max != -quantiser.min) {
      std::cout << "Min: " << quantiser.min << " Max: " << quantiser.max
          << std::endl;
      throw std::range_error(
        "Max and min must be the same in abs value (symmetric grid).");
    }

    if (perm_info.size != (quantiser.max + 1)) {
      std::cout << "Permutations: " << perm_info.size
          << " Max: " << quantiser.max << std::endl;
      throw std::range_error(
        "Permutations must have the same size as the grid.");
    }

    quantiser.quantize_multiple_permuted(w_ptr, winfo.size, pv_ptr, perm_ptr,
                                         raw_ptr);
    return quantized;
  }

  py::array_t<int32_t> quantize_multiple(py::array_t<float> W,
                                         py::array_t<float> posterior_variance,
                                         int32_t min_idx, int32_t max_idx) {
    auto winfo = W.request();
    auto w_ptr = (float *) winfo.ptr;

    auto pv_info = posterior_variance.request();
    auto pv_ptr = (float *) pv_info.ptr;

    auto quantized = py::array_t<int32_t, py::array::c_style>(winfo.size);
    int32_t *raw_ptr = (int32_t *) quantized.request().ptr;

    if (winfo.size != pv_info.size) {
      std::cout << winfo.size << " " << pv_info.size << std::endl;
      throw std::range_error(
        "W and posterior_variance must have the same size");
    }

    quantiser.quantize_multiple(w_ptr, winfo.size, pv_ptr, min_idx, max_idx,
                                raw_ptr);
    return quantized;
  }

  py::array_t<int32_t>
  quantize_multiple(py::array_t<float> W,
                    py::array_t<float> posterior_variance) {
    auto winfo = W.request();
    auto w_ptr = (float *) winfo.ptr;

    auto pv_info = posterior_variance.request();
    auto pv_ptr = (float *) pv_info.ptr;
    auto quantized = py::array_t<int32_t, py::array::c_style>(winfo.size);
    int32_t *raw_ptr = (int32_t *) quantized.request().ptr;

    if (pv_info.size == 1) {
      quantiser.quantize_multiple(w_ptr, winfo.size, pv_ptr[0], raw_ptr);
      return quantized;
    }

    if (winfo.size != pv_info.size) {
      std::cout << winfo.size << " " << pv_info.size << std::endl;
      throw std::range_error(
        "W and posterior_variance must have the same size");
    }

    quantiser.quantize_multiple(w_ptr, winfo.size, pv_ptr, raw_ptr);
    return quantized;
  }

  py::array_t<int32_t> quantize_multiple(py::array_t<float> W,
                                         float posterior_variance) {
    auto winfo = W.request();
    auto w_ptr = (float *) winfo.ptr;

    auto quantized = py::array_t<int32_t, py::array::c_style>(winfo.size);
    int32_t *raw_ptr = (int32_t *) quantized.request().ptr;
    quantiser.quantize_multiple(w_ptr, winfo.size, posterior_variance, raw_ptr);
    return quantized;
  };
  float get_lm() const { return quantiser.lambda; }
  void set_lm(float value) { quantiser.lambda = value; }
  float get_min() const { return quantiser.min; }
  void set_min(float value) { quantiser.min = value; }
  float get_max() const { return quantiser.max; }
  void set_max(float value) { quantiser.max = value; }

private:
  DeepCabacRdQuantiser quantiser;
};

py::array_t<int32_t> create_result_array(size_t rows, size_t cols) {
  auto result = py::array_t<int32_t>({rows, cols});
  return result;
}

void validate_inputs(const py::buffer_info &bufw, const py::buffer_info &bufh,
                     const py::buffer_info *buflm = nullptr) {
  if (bufw.ndim < 2 || bufh.ndim < 2) {
    throw std::runtime_error("W and H must be at least 2D.");
  }
  if (buflm && (buflm->ndim != 1 || buflm->size != bufw.shape[0])) {
    throw std::runtime_error(
      "Lambda must be a 1D array with the same number of rows as W.");
  }
}

std::map<std::string, std::variant<int, float, std::string> >
parseEntropyModelOptions(const py::dict &options) {
  std::map<std::string, std::variant<int, float, std::string> > cpp_opts;
  for (auto item: options) {
    auto key = item.first.cast<std::string>();
    if (key == "gamma") {
      // cpp_opts
      cpp_opts.insert({key, item.second.cast<float>()});
    } else if (key == "base_model") {
      cpp_opts.insert({key, item.second.cast<std::string>()});
    }
  }
  return cpp_opts;
};

// Python wrapper for the GPTQ class
class GPTQPy {
public:
  // In the GPTQPy class constructor:
  explicit GPTQPy(const py::dict &config) {
    fast_quant::GPTQConfig cpp_config;
    // Define set of valid configuration keys
    std::unordered_set<std::string> valid_keys = {
      "max_idx", "nbeams", "entropy_model", "debug_output",
      "debug_file", "grid", "entropy_model_options", "scan_order_major"
    };

    // Check for invalid configuration keys
    for (auto item: config) {
      std::string key = py::str(item.first);
      if (valid_keys.find(key) == valid_keys.end()) {
        throw std::invalid_argument(
          "Unknown configuration key: '" + key +
          "'. Valid keys are: max_idx, nbeams, entropy_model, debug_output, "
          "debug_file, grid, entropy_model_options, scan_order_major");
      }
    }

    // Extract configuration from Python dict
    if (config.contains("max_idx"))
      cpp_config.max_idx = config["max_idx"].cast<size_t>();
    else
      throw std::invalid_argument("Configuration must contain max_idx > 0.");

    if (config.contains("scan_order_major"))
      cpp_config.scan_major_order = config["scan_order_major"].cast<std::string>();

    if (config.contains("nbeams")) {
      cpp_config.num_beams = config["nbeams"].cast<size_t>();
      cpp_config.use_beam_search = (cpp_config.num_beams > 1);
    }

    if (config.contains("entropy_model"))
      cpp_config.entropy_model_type =
          config["entropy_model"].cast<std::string>();

    if (config.contains("entropy_model_options")) {
      cpp_config.entropy_model_options = parseEntropyModelOptions(
        config["entropy_model_options"].cast<py::dict>());
    }

    if (config.contains("debug_output"))
      cpp_config.write_debug_output = config["debug_output"].cast<bool>();

    if (config.contains("debug_file"))
      cpp_config.debug_file = config["debug_file"].cast<std::string>();

    // Create the C++ GPTQ instance
    gptq_ = std::make_unique<fast_quant::GPTQ>(cpp_config);

    // Store grid type
    grid_type_ =
        config.contains("grid") ? config["grid"].cast<std::string>() : "layer";
  }

  static bool is_contiguous(const py::array_t<float> &arr) {
    return arr.flags() & py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_;
  }

  py::array_t<int32_t> run(const py::array_t<float> &w,
                           const py::array_t<float> &Hinv_chol,
                           const py::array_t<float> &lm) const {
    if (!is_contiguous(w)) {
      throw std::runtime_error("w is not contiguous. Use .copy(order='C') "
        "prior to passing the matrix.");
    }
    if (!is_contiguous(Hinv_chol)) {
      throw std::runtime_error("Hinv_chol is not contiguous. Use "
        ".copy(order='C') prior to passing the matrix.");
    }
    // this makes a deepcopy of w
    auto buffer = w.request();
    py::array_t<float> w_copied = py::array_t<float>(buffer);
    // Check if lm is a scalar or array and call appropriate method
    if (lm.request().shape.empty()) {
      return run_layer(w_copied, Hinv_chol,
                       static_cast<float *>(lm.request().ptr)[0]);
    } else {
      return run_row(w_copied, Hinv_chol, lm);
    }
  }

  py::array_t<int32_t> run_row(const py::array_t<float> &w,
                               const py::array_t<float> &Hinv_chol,
                               const py::array_t<float> &lm) const {
    if (grid_type_ != "row") {
      throw std::invalid_argument("Did not configure GPTQ to be used with "
        "row-wise grid, but passed array of lambda.");
    }

    // Get buffer info
    auto bufw = w.request();
    auto bufh = Hinv_chol.request();
    auto buflm = lm.request();

    // Validate inputs
    validate_inputs(bufw, bufh, &buflm);

    // Create output array
    auto result = py::array_t<int32_t>({bufw.shape[0], bufw.shape[1]});
    auto result_ptr = static_cast<int32_t *>(result.request().ptr);

    // Call the C++ implementation
    gptq_->quantize_rowwise(static_cast<float *>(bufw.ptr),
                            static_cast<float *>(bufh.ptr),
                            static_cast<float *>(buflm.ptr), bufw.shape[0],
                            bufw.shape[1], result_ptr);

    return result;
  }

  py::array_t<int32_t> run_layer(const py::array_t<float> &w,
                                 const py::array_t<float> &Hinv_chol,
                                 const float lm) const {
    if (grid_type_ != "layer") {
      throw std::invalid_argument(
        "Did not configure GPTQ to be used with layer-wise grid, but passed "
        "single float lambda.");
    }

    // Get buffer info
    auto bufw = w.request();
    auto bufh = Hinv_chol.request();

    // Validate inputs
    validate_inputs(bufw, bufh);

    // Create output array
    auto result = py::array_t<int32_t>({bufw.shape[0], bufw.shape[1]});
    auto result_ptr = static_cast<int32_t *>(result.request().ptr);

    // Call the C++ implementation
    gptq_->quantize_layerwise(static_cast<float *>(bufw.ptr),
                              static_cast<float *>(bufh.ptr), lm, bufw.shape[0],
                              bufw.shape[1], result_ptr);

    return result;
  }

private:
  std::unique_ptr<fast_quant::GPTQ> gptq_;
  std::string grid_type_;

  // Helper to validate input dimensions
  void validate_inputs(const py::buffer_info &bufw, const py::buffer_info &bufh,
                       const py::buffer_info *buflm = nullptr) const {
    if (bufw.ndim < 2 || bufh.ndim < 2) {
      throw std::runtime_error("W and H must be at least 2D.");
    }
    if (buflm && (buflm->ndim != 1 || buflm->size != bufw.shape[0])) {
      throw std::runtime_error(
        "Lambda must be a 1D array with the same number of rows as W.");
    }
  }
};

// Helper function for backward compatibility
py::array_t<int32_t> gptq(const py::array_t<float> &w,
                          const py::array_t<float> &Hinv_chol, const float lm,
                          const size_t max_idx,
                          const std::string &entropy_model) {
  // Create a GPTQ config and run it
  py::dict config;
  config["max_idx"] = max_idx;
  config["entropy_model"] = entropy_model;
  config["grid"] = "layer";

  GPTQPy gptq(config);
  return gptq.run_layer(w, Hinv_chol, lm);
}

// Helper function for backward compatibility
py::array_t<int32_t> gptq_beam_search(py::array_t<float> w,
                                      py::array_t<float> Hinv_chol, float lm,
                                      size_t max_idx, size_t m) {
  // Create a GPTQ config and run it
  py::dict config;
  config["max_idx"] = max_idx;
  config["nbeams"] = m;
  config["grid"] = "layer";

  GPTQPy gptq(config);
  return gptq.run_layer(w, Hinv_chol, lm);
}

// Helper function for backward compatibility
py::array_t<int32_t> gptq_rw_beam_search(py::array_t<float> w,
                                         py::array_t<float> Hinv_chol,
                                         py::array_t<float> lm, size_t max_idx,
                                         size_t m) {
  // Create a GPTQ config and run it
  py::dict config;
  config["max_idx"] = max_idx;
  config["nbeams"] = m;
  config["grid"] = "row";

  GPTQPy gptq(config);
  return gptq.run_row(w, Hinv_chol, lm);
}

// Helper function for backward compatibility
py::array_t<int32_t> gptq_rw_grid(py::array_t<float> w,
                                  py::array_t<float> Hinv_chol,
                                  py::array_t<float> lm, size_t max_idx,
                                  const std::string &entropy_model) {
  // Create a GPTQ config and run it
  py::dict config;
  config["max_idx"] = max_idx;
  config["entropy_model"] = entropy_model;
  config["grid"] = "row";

  GPTQPy gptq(config);
  return gptq.run_row(w, Hinv_chol, lm);
}

PYBIND11_MODULE(_core, m) {
  py::class_<EntropyModelPy>(m, "EntropyModel")
      .def(py::init<const std::string &, py::array_t<int32_t>, py::dict>(),
           py::arg("entropy_model"), py::arg("q"),
           py::arg("options") = py::dict())
      .def("estimate", &EntropyModelPy::estimate, py::arg("quantisation_index"))
      .def("estimate_tensor", &EntropyModelPy::estimate_tensor, py::arg("q"))
      .def("update", &EntropyModelPy::update, py::arg("quantisation_index"));
  py::class_<GPTQPy>(m, "GPTQ")
      .def(py::init<const py::dict &>())
      .def("run", &GPTQPy::run, py::arg("W"), py::arg("Hinv_chol"),
           py::arg("lm"));

  // Register the helper functions for backward compatibility
  m.def("gptq", &gptq, "GPTQ with a single lambda value");
  m.def("gptq_beam_search", &gptq_beam_search,
        "GPTQ with beam search and a single lambda");
  m.def("gptq_beam_search", &gptq_rw_beam_search,
        "GPTQ with beam search and row-wise lambdas");
  m.def("gptq", &gptq_rw_grid, "GPTQ with row-wise lambdas");

  py::class_<DeepCabacRdQuantiserPy>(m, "DeepCabacRdQuantiser")
      .def(py::init<float, int32_t, int32_t>(), py::arg("lambda"),
           py::arg("min"), py::arg("max"),
           R"pbdoc(Initialise the quantiser with lambda, delta, min and max.
Supply the rate-tradeoff parameter with lambda.
Delta is the scale of weight indices (so a gridpoint is obtained by multiplying its index
with delta). Min and max are the minimum respectively maximum indices one can quantise to,
they essentially bound the grid size.
          )pbdoc")
      .def(
        "quantize", &DeepCabacRdQuantiserPy::quantize_single, py::arg("w"),
        py::arg("posterior_variance"),
        R"pbdoc(Quantize a (set of) weights with a (set of) posterior variances.)pbdoc")
      .def("quantize",
           py::overload_cast<py::array_t<float>, float>(
             &DeepCabacRdQuantiserPy::quantize_multiple),
           py::arg("W"), py::arg("posterior_variance"))
      .def("quantize",
           py::overload_cast<py::array_t<float>, py::array_t<float>, int32_t,
             int32_t>(
             &DeepCabacRdQuantiserPy::quantize_multiple),
           py::arg("W"), py::arg("posterior_variance"), py::arg("min_idx"),
           py::arg("max_idx"))
      .def("quantize",
           py::overload_cast<py::array_t<float>, py::array_t<float> >(
             &DeepCabacRdQuantiserPy::quantize_multiple),
           py::arg("W"), py::arg("posterior_variance"))
      .def("quantize_permuted",
           &DeepCabacRdQuantiserPy::quantize_multiple_permuted, py::arg("W"),
           py::arg("posterior_variance"), py::arg("permutations"))
      .def_property("lm", &DeepCabacRdQuantiserPy::get_lm,
                    &DeepCabacRdQuantiserPy::set_lm)
      .doc() =
      R"pbdoc(Rate-Distortion quantiser that uses the entropy-model of DeepCABAC to estimate
the entropy of the resulting bit string. Therefore, solves
argmin_{min <= i <= max} rate(i) * lambda + 1/pv * (w - i)^2 * delta^2
           )pbdoc";
}
