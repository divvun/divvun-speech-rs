// Custom LayerNorm kernel for Vocos vocoder
// XNNPACK doesn't support LayerNorm, causing graph fragmentation.
// This provides a fast implementation for the "gaps" between XNNPACK subgraphs.

#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::ArrayRef;

// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Input: [*, normalized_shape] where * means any number of leading dimensions
// Normalizes over the last len(normalized_shape) dimensions
//
// For Vocos: normalized_shape is typically [512] (feature dimension)
// Input shape: [B, T, 512] -> normalizes over last dim
Tensor& layer_norm_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    ArrayRef<int64_t> normalized_shape,
    const executorch::aten::optional<Tensor>& weight,
    const executorch::aten::optional<Tensor>& bias,
    double eps,
    Tensor& out) {

    // Validate input
    ET_KERNEL_CHECK_MSG(
        ctx,
        input.scalar_type() == executorch::aten::ScalarType::Float,
        InvalidArgument,
        out,
        "input must be Float");

    // Calculate normalization size (product of normalized_shape)
    int64_t norm_size = 1;
    for (size_t i = 0; i < normalized_shape.size(); ++i) {
        norm_size *= normalized_shape[i];
    }

    // Number of independent normalizations (everything before normalized dims)
    int64_t num_instances = input.numel() / norm_size;

    // Resize output to match input
    ET_KERNEL_CHECK_MSG(
        ctx,
        executorch::runtime::resize_tensor(out, input.sizes()) == executorch::runtime::Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor");

    const float* in_data = input.const_data_ptr<float>();
    float* out_data = out.mutable_data_ptr<float>();

    const float* gamma = nullptr;
    const float* beta_ptr = nullptr;

    if (weight.has_value()) {
        gamma = weight.value().const_data_ptr<float>();
    }
    if (bias.has_value()) {
        beta_ptr = bias.value().const_data_ptr<float>();
    }

    const float eps_f = static_cast<float>(eps);

    // Process each instance (compiler will auto-vectorize inner loops with -O3 -ffast-math)
    for (int64_t i = 0; i < num_instances; ++i) {
        const float* x = in_data + i * norm_size;
        float* y = out_data + i * norm_size;

        // Pass 1: Compute mean
        float sum = 0.0f;
        for (int64_t j = 0; j < norm_size; ++j) {
            sum += x[j];
        }
        const float mean = sum / static_cast<float>(norm_size);

        // Pass 2: Compute variance
        float var_sum = 0.0f;
        for (int64_t j = 0; j < norm_size; ++j) {
            float diff = x[j] - mean;
            var_sum += diff * diff;
        }
        const float inv_std = 1.0f / std::sqrt(var_sum / static_cast<float>(norm_size) + eps_f);

        // Pass 3: Normalize + scale + shift
        if (gamma && beta_ptr) {
            // Most common case: both weight and bias
            for (int64_t j = 0; j < norm_size; ++j) {
                float normalized = (x[j] - mean) * inv_std;
                y[j] = gamma[j] * normalized + beta_ptr[j];
            }
        } else if (gamma) {
            // Weight only
            for (int64_t j = 0; j < norm_size; ++j) {
                float normalized = (x[j] - mean) * inv_std;
                y[j] = gamma[j] * normalized;
            }
        } else if (beta_ptr) {
            // Bias only (rare)
            for (int64_t j = 0; j < norm_size; ++j) {
                float normalized = (x[j] - mean) * inv_std;
                y[j] = normalized + beta_ptr[j];
            }
        } else {
            // No affine transform
            for (int64_t j = 0; j < norm_size; ++j) {
                y[j] = (x[j] - mean) * inv_std;
            }
        }
    }

    return out;
}

} // namespace native
} // namespace executor
} // namespace torch
