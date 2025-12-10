// Fused ISTFT kernel for Vocos vocoder using pffft (SIMD-optimized)
// Takes real/imag tensors (pre-computed from mag*cos(phase), mag*sin(phase) in Python)
// Avoids complex tensor intermediates that cause ExecuTorch export issues

#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdio>

extern "C" {
#include "pffft.h"
}

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

// ISTFT: (real, imag) -> audio
// Input real: [B, F, T] where F = n_fft/2 + 1
// Input imag: [B, F, T]
// Output: [B, audio_len] where audio_len = (T-1) * hop_length + n_fft
Tensor& istft_out(
    KernelRuntimeContext& ctx,
    const Tensor& real,
    const Tensor& imag,
    int64_t n_fft,
    int64_t hop_length,
    int64_t win_length,
    Tensor& out) {

    // Validate inputs
    ET_KERNEL_CHECK_MSG(
        ctx,
        real.dim() == 3,
        InvalidArgument,
        out,
        "real must be 3D [B, F, T], got %zd dims",
        static_cast<size_t>(real.dim()));

    ET_KERNEL_CHECK_MSG(
        ctx,
        imag.dim() == 3,
        InvalidArgument,
        out,
        "imag must be 3D [B, F, T], got %zd dims",
        static_cast<size_t>(imag.dim()));

    ET_KERNEL_CHECK_MSG(
        ctx,
        real.scalar_type() == executorch::aten::ScalarType::Float,
        InvalidArgument,
        out,
        "real must be Float");

    ET_KERNEL_CHECK_MSG(
        ctx,
        imag.scalar_type() == executorch::aten::ScalarType::Float,
        InvalidArgument,
        out,
        "imag must be Float");

    const int64_t batch = real.size(0);
    const int64_t n_freqs = real.size(1);  // n_fft/2 + 1
    const int64_t n_frames = real.size(2);

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

    const float* real_data = real.const_data_ptr<float>();
    const float* imag_data = imag.const_data_ptr<float>();
    float* out_data = out.mutable_data_ptr<float>();

    // Create pffft setup (reused for all frames)
    PFFFT_Setup* setup = pffft_new_setup(static_cast<int>(n_fft), PFFFT_REAL);
    ET_KERNEL_CHECK_MSG(
        ctx,
        setup != nullptr,
        InvalidArgument,
        out,
        "Failed to create pffft setup for n_fft=%zd",
        static_cast<size_t>(n_fft));

    // Allocate aligned buffers for pffft
    float* spectrum_buf = static_cast<float*>(pffft_aligned_malloc(static_cast<size_t>(n_fft) * sizeof(float)));
    float* frame = static_cast<float*>(pffft_aligned_malloc(static_cast<size_t>(n_fft) * sizeof(float)));
    float* work = static_cast<float*>(pffft_aligned_malloc(static_cast<size_t>(n_fft) * sizeof(float)));

    // Hann window
    std::vector<float> window(static_cast<size_t>(n_fft));
    make_hann_window(window.data(), n_fft);

    // Normalization factor (pffft backward transform gives N*x)
    const float norm = 1.0f / static_cast<float>(n_fft);

    auto istft_start = std::chrono::high_resolution_clock::now();

    // Process each batch
    for (int64_t b = 0; b < batch; ++b) {
        float* batch_out = out_data + b * audio_len;

        // Zero output for this batch
        std::fill(batch_out, batch_out + audio_len, 0.0f);

        // Window sum for normalization
        std::vector<float> window_sum(static_cast<size_t>(audio_len), 0.0f);

        // Process each frame
        for (int64_t t = 0; t < n_frames; ++t) {
            // Pack real/imag directly into pffft format (no sin/cos needed!)
            // pffft real format (ordered): [DC, Nyquist, Re1, Im1, Re2, Im2, ...]
            const int64_t half_n = n_fft / 2;

            // DC component (index 0) - real only
            {
                const int64_t idx = b * n_freqs * n_frames + 0 * n_frames + t;
                spectrum_buf[0] = real_data[idx];
            }

            // Nyquist component (index n_fft/2) - real only
            {
                const int64_t idx = b * n_freqs * n_frames + half_n * n_frames + t;
                spectrum_buf[1] = real_data[idx];
            }

            // Complex components 1 to n_fft/2 - 1
            for (int64_t f = 1; f < half_n; ++f) {
                const int64_t idx = b * n_freqs * n_frames + f * n_frames + t;
                spectrum_buf[2 * f] = real_data[idx];
                spectrum_buf[2 * f + 1] = imag_data[idx];
            }

            // IRFFT using pffft
            pffft_transform_ordered(setup, spectrum_buf, frame, work, PFFFT_BACKWARD);

            // Apply normalization, window, and overlap-add
            const int64_t start = t * hop_length;
            for (int64_t i = 0; i < n_fft && (start + i) < audio_len; ++i) {
                const float windowed = frame[i] * norm * window[static_cast<size_t>(i)];
                batch_out[start + i] += windowed;
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

    auto istft_end = std::chrono::high_resolution_clock::now();
    printf("ISTFT kernel: %.2f ms (frames=%lld)\n",
           std::chrono::duration<double, std::milli>(istft_end - istft_start).count(),
           (long long)n_frames);

    // Cleanup
    pffft_aligned_free(work);
    pffft_aligned_free(frame);
    pffft_aligned_free(spectrum_buf);
    pffft_destroy_setup(setup);

    return out;
}

} // namespace native
} // namespace executor
} // namespace torch
