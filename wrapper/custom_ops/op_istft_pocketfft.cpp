// Fused ISTFT kernel for Vocos vocoder
// Takes magnitude and phase tensors, outputs audio directly
// Avoids complex tensor intermediates that cause ExecuTorch export issues

#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>
#include <vector>
#include <complex>

#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

namespace torch {
namespace executor {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

namespace {

// Compute Hann window
void make_hann_window(float* window, int64_t length) {
    const double pi = 3.14159265358979323846;
    for (int64_t i = 0; i < length; ++i) {
        window[i] = static_cast<float>(0.5 * (1.0 - std::cos(2.0 * pi * i / length)));
    }
}

} // namespace

// Fused ISTFT: (magnitude, phase) -> audio
// Input mag: [B, F, T] where F = n_fft/2 + 1
// Input phase: [B, F, T]
// Output: [B, audio_len] where audio_len = (T-1) * hop_length + n_fft
Tensor& istft_out(
    KernelRuntimeContext& ctx,
    const Tensor& mag,
    const Tensor& phase,
    int64_t n_fft,
    int64_t hop_length,
    int64_t win_length,
    Tensor& out) {

    // Validate inputs
    ET_KERNEL_CHECK_MSG(
        ctx,
        mag.dim() == 3,
        InvalidArgument,
        out,
        "mag must be 3D [B, F, T], got %zd dims",
        static_cast<size_t>(mag.dim()));

    ET_KERNEL_CHECK_MSG(
        ctx,
        phase.dim() == 3,
        InvalidArgument,
        out,
        "phase must be 3D [B, F, T], got %zd dims",
        static_cast<size_t>(phase.dim()));

    ET_KERNEL_CHECK_MSG(
        ctx,
        mag.scalar_type() == executorch::aten::ScalarType::Float,
        InvalidArgument,
        out,
        "mag must be Float");

    ET_KERNEL_CHECK_MSG(
        ctx,
        phase.scalar_type() == executorch::aten::ScalarType::Float,
        InvalidArgument,
        out,
        "phase must be Float");

    const int64_t batch = mag.size(0);
    const int64_t n_freqs = mag.size(1);  // n_fft/2 + 1
    const int64_t n_frames = mag.size(2);

    ET_KERNEL_CHECK_MSG(
        ctx,
        n_freqs == n_fft / 2 + 1,
        InvalidArgument,
        out,
        "n_freqs (%zd) must equal n_fft/2 + 1 (%zd)",
        static_cast<size_t>(n_freqs),
        static_cast<size_t>(n_fft / 2 + 1));

    // Calculate output length (same as torch.istft with center=True)
    const int64_t audio_len = (n_frames - 1) * hop_length + n_fft;

    // Resize output tensor
    executorch::aten::SizesType out_sizes[] = {
        static_cast<executorch::aten::SizesType>(batch),
        static_cast<executorch::aten::SizesType>(audio_len)
    };
    ET_KERNEL_CHECK_MSG(
        ctx,
        executorch::runtime::resize_tensor(
            out,
            executorch::runtime::ArrayRef<executorch::aten::SizesType>(out_sizes, 2)) == executorch::runtime::Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor");

    const float* mag_data = mag.const_data_ptr<float>();
    const float* phase_data = phase.const_data_ptr<float>();
    float* out_data = out.mutable_data_ptr<float>();

    // Allocate temporary buffers
    std::vector<float> window(static_cast<size_t>(n_fft));
    make_hann_window(window.data(), n_fft);

    // Process each batch
    for (int64_t b = 0; b < batch; ++b) {
        float* batch_out = out_data + b * audio_len;

        // Zero output for this batch
        std::fill(batch_out, batch_out + audio_len, 0.0f);

        // Window sum for normalization
        std::vector<float> window_sum(static_cast<size_t>(audio_len), 0.0f);

        // Temp buffers for one frame
        std::vector<std::complex<float>> spectrum(static_cast<size_t>(n_freqs));
        std::vector<float> frame(static_cast<size_t>(n_fft));

        // Process each frame
        for (int64_t t = 0; t < n_frames; ++t) {
            // Build complex spectrum from magnitude and phase
            for (int64_t f = 0; f < n_freqs; ++f) {
                // mag and phase are [B, F, T] - contiguous in T
                const int64_t idx = b * n_freqs * n_frames + f * n_frames + t;
                const float m = mag_data[idx];
                const float p = phase_data[idx];
                spectrum[static_cast<size_t>(f)] = std::complex<float>(
                    m * std::cos(p),
                    m * std::sin(p));
            }

            // IRFFT using pocketfft
            pocketfft::shape_t shape_out = {static_cast<size_t>(n_fft)};
            pocketfft::stride_t stride_in = {sizeof(std::complex<float>)};
            pocketfft::stride_t stride_out = {sizeof(float)};

            pocketfft::c2r(
                shape_out,
                stride_in,
                stride_out,
                static_cast<size_t>(0),  // axis
                false,  // backward (inverse)
                spectrum.data(),
                frame.data(),
                1.0f / static_cast<float>(n_fft),  // normalization factor
                1);  // single thread

            // Apply window and overlap-add
            const int64_t start = t * hop_length;
            for (int64_t i = 0; i < n_fft && (start + i) < audio_len; ++i) {
                batch_out[start + i] += frame[static_cast<size_t>(i)] * window[static_cast<size_t>(i)];
                window_sum[static_cast<size_t>(start + i)] += window[static_cast<size_t>(i)] * window[static_cast<size_t>(i)];
            }
        }

        // Normalize by window sum (COLA condition)
        for (int64_t i = 0; i < audio_len; ++i) {
            if (window_sum[static_cast<size_t>(i)] > 1e-8f) {
                batch_out[i] /= window_sum[static_cast<size_t>(i)];
            }
        }
    }

    return out;
}

} // namespace native
} // namespace executor
} // namespace torch
