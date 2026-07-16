// TTS Wrapper - Thin C API around the working C++ runner
// Wraps ExecuTorch voice + vocoder inference

#include "wrapper.hpp"
#include "custom_ops/pffft.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <memory>
#include <string>

#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/extension/tensor/tensor_ptr.h>

using namespace executorch::runtime;
using namespace executorch::extension;

// Forward-declare backend query functions (from runtime/backend/interface.h)
// Using the same pattern as executorch's own pybindings.cpp
namespace executorch {
namespace runtime {
class BackendInterface;
size_t get_num_registered_backends();
Result<const char*> get_backend_name(size_t index);
BackendInterface* get_backend_class(const char* name);
} // namespace runtime
} // namespace executorch

// Find next valid FFT size (pffft needs N = 2^a * 3^b * 5^c, a >= 5)
static int next_fft_size(int n) {
    if (n < 32) return 32;
    int candidate = 32;
    while (candidate < n) {
        candidate *= 2;
    }
    return candidate;
}

// 1D Gaussian blur using FFT convolution
static void gaussian_blur_1d_fft(float* data, int len, float sigma,
                                  PFFFT_Setup* setup, int fft_size,
                                  float* work, float* freq_data, float* freq_kernel) {
    // Build Gaussian kernel in time domain (centered, zero-padded)
    std::vector<float> kernel(fft_size, 0.0f);
    int radius = static_cast<int>(std::ceil(3.0f * sigma));
    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        float val = std::exp(-(i * i) / (2.0f * sigma * sigma));
        kernel[(i + fft_size) % fft_size] = val;
        sum += val;
    }
    for (int i = 0; i < fft_size; i++) kernel[i] /= sum;

    // Copy data with zero-padding
    std::vector<float> padded(fft_size, 0.0f);
    for (int i = 0; i < len; i++) padded[i] = data[i];

    // FFT of padded data
    pffft_transform(setup, padded.data(), freq_data, work, PFFFT_FORWARD);

    // FFT of kernel
    pffft_transform(setup, kernel.data(), freq_kernel, work, PFFFT_FORWARD);

    // Multiply in frequency domain
    std::vector<float> freq_result(fft_size, 0.0f);
    pffft_zconvolve_accumulate(setup, freq_data, freq_kernel, freq_result.data(), 1.0f);

    // Inverse FFT
    pffft_transform(setup, freq_result.data(), padded.data(), work, PFFFT_BACKWARD);

    // Scale and copy back (pffft doesn't normalize)
    float scale = 1.0f / fft_size;
    for (int i = 0; i < len; i++) {
        data[i] = padded[i] * scale;
    }
}

// 2D Gaussian blur using separable FFT convolutions
static void gaussian_blur_2d_fft(float* data, int height, int width, float sigma) {
    // Setup FFT for rows
    int row_fft_size = next_fft_size(width);
    PFFFT_Setup* row_setup = pffft_new_setup(row_fft_size, PFFFT_REAL);

    // Allocate aligned buffers
    float* work = (float*)pffft_aligned_malloc(row_fft_size * sizeof(float));
    float* freq_data = (float*)pffft_aligned_malloc(row_fft_size * sizeof(float));
    float* freq_kernel = (float*)pffft_aligned_malloc(row_fft_size * sizeof(float));

    // Blur rows
    for (int y = 0; y < height; y++) {
        gaussian_blur_1d_fft(&data[y * width], width, sigma,
                            row_setup, row_fft_size, work, freq_data, freq_kernel);
    }

    pffft_destroy_setup(row_setup);

    // Setup FFT for columns
    int col_fft_size = next_fft_size(height);
    PFFFT_Setup* col_setup = pffft_new_setup(col_fft_size, PFFFT_REAL);

    // Resize buffers if needed
    if (col_fft_size > row_fft_size) {
        pffft_aligned_free(work);
        pffft_aligned_free(freq_data);
        pffft_aligned_free(freq_kernel);
        work = (float*)pffft_aligned_malloc(col_fft_size * sizeof(float));
        freq_data = (float*)pffft_aligned_malloc(col_fft_size * sizeof(float));
        freq_kernel = (float*)pffft_aligned_malloc(col_fft_size * sizeof(float));
    }

    // Blur columns
    std::vector<float> column(height);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) column[y] = data[y * width + x];

        gaussian_blur_1d_fft(column.data(), height, sigma,
                            col_setup, col_fft_size, work, freq_data, freq_kernel);

        for (int y = 0; y < height; y++) data[y * width + x] = column[y];
    }

    pffft_destroy_setup(col_setup);
    pffft_aligned_free(work);
    pffft_aligned_free(freq_data);
    pffft_aligned_free(freq_kernel);
}

// Apply mel spectrogram sharpening (matches Python _process_sharpening)
static void sharpen_mel(float* mel_data, int mel_channels, int mel_len) {
    size_t size = static_cast<size_t>(mel_channels) * mel_len;
    std::vector<float> blurred(size);

    // Pass 1: Unsharp mask with sigma=1.0, alpha=0.2
    std::memcpy(blurred.data(), mel_data, size * sizeof(float));
    gaussian_blur_2d_fft(blurred.data(), mel_channels, mel_len, 1.0f);
    for (size_t i = 0; i < size; i++) {
        mel_data[i] = mel_data[i] + 0.2f * (mel_data[i] - blurred[i]);
    }

    // Pass 2: Unsharp mask with sigma=3.0, alpha=0.1
    std::memcpy(blurred.data(), mel_data, size * sizeof(float));
    gaussian_blur_2d_fft(blurred.data(), mel_channels, mel_len, 3.0f);
    for (size_t i = 0; i < size; i++) {
        mel_data[i] = mel_data[i] + 0.1f * (mel_data[i] - blurred[i]);
    }

    // Per-frequency adjustment: mel[i][j] += (i - 40) * 0.01
    for (int i = 0; i < mel_channels; i++) {
        float adjustment = (i - 40) * 0.01f;
        for (int j = 0; j < mel_len; j++) {
            mel_data[i * mel_len + j] += adjustment;
        }
    }
}

struct TtsSynthesizer {
    // Must be pointers - Program::load takes a pointer and keeps it
    std::unique_ptr<MmapDataLoader> voice_loader;
    std::unique_ptr<MmapDataLoader> vocoder_loader;
    std::unique_ptr<Program> voice_program;
    std::unique_ptr<Program> vocoder_program;
};

static inline void set_error(TtsError* out_error, TtsError err,
                             const char** out_detail = nullptr,
                             const char* detail = nullptr) {
    if (out_error) *out_error = err;
    if (out_detail) *out_detail = detail;
}

extern "C" {

// Defined in custom_ops/register_ops.cpp
extern "C" void tts_register_custom_ops();

extern "C" size_t tts_get_num_backends() {
    return executorch::runtime::get_num_registered_backends();
}

extern "C" const char* tts_get_backend_name(size_t index) {
    auto result = executorch::runtime::get_backend_name(index);
    return result.ok() ? result.get() : nullptr;
}

extern "C" bool tts_has_backend(const char* name) {
    return executorch::runtime::get_backend_class(name) != nullptr;
}

TtsSynthesizer* tts_synthesizer_new(
    const char* voice_path,
    const char* vocoder_path,
    TtsError* out_error,
    const char** out_error_detail
) {
    // Ensure custom ops are registered (static initializer may not run on MSVC)
    tts_register_custom_ops();

    set_error(out_error, TTS_OK, out_error_detail);

    if (!voice_path || !vocoder_path) {
        set_error(out_error, TTS_ERROR_INVALID_ARGUMENT, out_error_detail,
                  "voice_path or vocoder_path is null");
        return nullptr;
    }

    auto synth = std::make_unique<TtsSynthesizer>();

    // Load voice model with mmap
    auto voice_loader_result = MmapDataLoader::from(voice_path, MmapDataLoader::MlockConfig::UseMlockIgnoreErrors);
    if (!voice_loader_result.ok()) {
        set_error(out_error, TTS_ERROR_VOICE_LOAD_FAILED, out_error_detail,
                  "failed to mmap voice model");
        return nullptr;
    }
    synth->voice_loader = std::make_unique<MmapDataLoader>(std::move(voice_loader_result.get()));

    auto voice_program_result = Program::load(synth->voice_loader.get());
    if (!voice_program_result.ok()) {
        set_error(out_error, TTS_ERROR_VOICE_LOAD_FAILED, out_error_detail,
                  "failed to load voice program");
        return nullptr;
    }
    synth->voice_program = std::make_unique<Program>(std::move(voice_program_result.get()));

    // Load vocoder model with mmap
    auto vocoder_loader_result = MmapDataLoader::from(vocoder_path, MmapDataLoader::MlockConfig::UseMlockIgnoreErrors);
    if (!vocoder_loader_result.ok()) {
        set_error(out_error, TTS_ERROR_VOCODER_LOAD_FAILED, out_error_detail,
                  "failed to mmap vocoder model");
        return nullptr;
    }
    synth->vocoder_loader = std::make_unique<MmapDataLoader>(std::move(vocoder_loader_result.get()));

    auto vocoder_program_result = Program::load(synth->vocoder_loader.get());
    if (!vocoder_program_result.ok()) {
        set_error(out_error, TTS_ERROR_VOCODER_LOAD_FAILED, out_error_detail,
                  "failed to load vocoder program");
        return nullptr;
    }
    synth->vocoder_program = std::make_unique<Program>(std::move(vocoder_program_result.get()));

    return synth.release();
}

void tts_synthesizer_free(TtsSynthesizer* synth) {
    delete synth;
}

// Shared implementation. When out_durations/out_duration_count are non-null,
// the function also returns the model's raw dur_pred prefix (length token_count).
static float* synthesize_impl(
    TtsSynthesizer* synth,
    const int64_t* tokens,
    size_t token_count,
    int64_t speaker_id,
    int64_t language_id,
    float pace,
    size_t* out_sample_count,
    float** out_durations,
    size_t* out_duration_count,
    TtsError* out_error,
    const char** out_error_detail
) {
    set_error(out_error, TTS_OK, out_error_detail);

    if (!synth || !tokens || token_count == 0 || !out_sample_count) {
        set_error(out_error, TTS_ERROR_INVALID_ARGUMENT, out_error_detail,
                  "synth, tokens, or out_sample_count is null");
        return nullptr;
    }

    *out_sample_count = 0;
    if (out_durations) *out_durations = nullptr;
    if (out_duration_count) *out_duration_count = 0;

    // Memory for voice model
    testing::ManagedMemoryManager voice_mmm(
        128 * 1024 * 1024,  // 128MB planned memory
        64 * 1024 * 1024    // 64MB method allocator
    );

    auto voice_method_result = synth->voice_program->load_method("forward", &voice_mmm.get());
    if (!voice_method_result.ok()) {
        std::string detail = "failed to load 'forward' method from voice (";
        detail += executorch::runtime::to_string(voice_method_result.error());
        detail += ")";
        set_error(out_error, TTS_ERROR_VOICE_METHOD_LOAD_FAILED, out_error_detail,
                  strdup(detail.c_str()));
        return nullptr;
    }
    Method voice_method = std::move(voice_method_result.get());

    // Set voice model inputs via mutable_input() — writes directly into the
    // model's pre-allocated tensors. The XNNPACK model has static input shape
    // [1, 512], so we zero-fill and copy actual tokens in.
    static const int32_t VOICE_MAX_SEQ_LEN = 512;

    EValue& voice_input_0 = voice_method.mutable_input(0);
    if (voice_input_0.isTensor()) {
        auto& input_tensor = voice_input_0.toTensor();
        if (token_count > VOICE_MAX_SEQ_LEN) {
            set_error(out_error, TTS_ERROR_VOICE_INPUT_FAILED, out_error_detail,
                      "token count exceeds max sequence length (512)");
            return nullptr;
        }
        // Zero-fill (token 0 = padding_idx, masked by the model), then copy actual tokens
        int64_t* data = input_tensor.mutable_data_ptr<int64_t>();
        memset(data, 0, VOICE_MAX_SEQ_LEN * sizeof(int64_t));
        memcpy(data, tokens, token_count * sizeof(int64_t));

        // Set other inputs via mutable_input
        voice_method.mutable_input(1).toTensor().mutable_data_ptr<int64_t>()[0] = speaker_id;
        voice_method.mutable_input(2).toTensor().mutable_data_ptr<int64_t>()[0] = language_id;
        voice_method.mutable_input(3).toTensor().mutable_data_ptr<float>()[0] = pace;
    } else {
        set_error(out_error, TTS_ERROR_VOICE_INPUT_FAILED, out_error_detail,
                  "failed to set text tensor");
        return nullptr;
    }

    // Run voice model
    auto voice_err = voice_method.execute();
    if (voice_err != Error::Ok) {
        set_error(out_error, TTS_ERROR_VOICE_EXECUTE_FAILED, out_error_detail,
                  "voice model execution failed");
        return nullptr;
    }

    // Get mel spectrogram output
    const EValue& mel_output = voice_method.get_output(0);
    if (!mel_output.isTensor()) {
        set_error(out_error, TTS_ERROR_VOICE_OUTPUT_INVALID, out_error_detail,
                  "voice output is not a tensor");
        return nullptr;
    }
    const auto& mel_tensor = mel_output.toTensor();

    // Get actual mel length for trimming (if available)
    int64_t actual_mel_len = mel_tensor.size(2);
    int64_t mel_full_len = mel_tensor.size(2);
    bool mel_len_from_model = false;
    if (voice_method.outputs_size() > 1) {
        const EValue& mel_lens_output = voice_method.get_output(1);
        if (mel_lens_output.isTensor()) {
            actual_mel_len = mel_lens_output.toTensor().const_data_ptr<int64_t>()[0];
            mel_len_from_model = true;
        }
    }
    fprintf(stderr,
        "[tts-diag] tokens_in=%zu  voice_outputs=%zu  mel_full_len=%lld  "
        "actual_mel_len=%lld  from_model=%d  mel_channels=%lld\n",
        token_count,
        (size_t)voice_method.outputs_size(),
        (long long)mel_full_len,
        (long long)actual_mel_len,
        (int)mel_len_from_model,
        (long long)mel_tensor.size(1));
    fflush(stderr);

    // Extract per-token durations if requested (output 2 of voice .pte).
    // dur_pred is shape [1, T_padded] float32 where T_padded == VOICE_MAX_SEQ_LEN.
    // We copy the first token_count entries; trailing entries correspond to
    // zero-padded input tokens and are not meaningful.
    if (out_durations) {
        if (voice_method.outputs_size() < 3) {
            set_error(out_error, TTS_ERROR_VOICE_OUTPUT_INVALID, out_error_detail,
                      "voice model has no duration output (re-export with updated BoundedVoiceModel)");
            return nullptr;
        }
        const EValue& dur_output = voice_method.get_output(2);
        if (!dur_output.isTensor()) {
            set_error(out_error, TTS_ERROR_VOICE_OUTPUT_INVALID, out_error_detail,
                      "voice dur_pred output is not a tensor");
            return nullptr;
        }
        const auto& dur_tensor = dur_output.toTensor();
        size_t dur_padded = static_cast<size_t>(dur_tensor.size(dur_tensor.dim() - 1));
        if (token_count > dur_padded) {
            set_error(out_error, TTS_ERROR_VOICE_OUTPUT_INVALID, out_error_detail,
                      "token_count exceeds dur_pred padded length");
            return nullptr;
        }
        float* durs = new float[token_count];
        memcpy(durs, dur_tensor.const_data_ptr<float>(), token_count * sizeof(float));
        *out_durations = durs;
        if (out_duration_count) *out_duration_count = token_count;
    }

    // Sharpen mel spectrogram
    int mel_channels = static_cast<int>(mel_tensor.size(1));
    int mel_len = static_cast<int>(mel_tensor.size(2));
    std::vector<float> mel_sharpened(mel_tensor.numel());
    memcpy(mel_sharpened.data(), mel_tensor.const_data_ptr<float>(),
           mel_tensor.numel() * sizeof(float));
    sharpen_mel(mel_sharpened.data(), mel_channels, mel_len);

    // Memory for vocoder
    testing::ManagedMemoryManager vocoder_mmm(
        512 * 1024 * 1024,  // 512MB planned memory
        128 * 1024 * 1024   // 128MB method allocator
    );

    auto vocoder_method_result = synth->vocoder_program->load_method("forward", &vocoder_mmm.get());
    if (!vocoder_method_result.ok()) {
        std::string detail = "failed to load 'forward' method from vocoder (";
        detail += executorch::runtime::to_string(vocoder_method_result.error());
        detail += ")";
        set_error(out_error, TTS_ERROR_VOCODER_METHOD_LOAD_FAILED, out_error_detail,
                  strdup(detail.c_str()));
        return nullptr;
    }
    Method vocoder_method = std::move(vocoder_method_result.get());

    // Set vocoder input (using sharpened mel)
    EValue& input_evalue = vocoder_method.mutable_input(0);
    if (input_evalue.isTensor()) {
        // XNNPACK: copy data into pre-allocated tensor
        auto& input_tensor = input_evalue.toTensor();
        if (input_tensor.numel() != mel_tensor.numel()) {
            set_error(out_error, TTS_ERROR_SHAPE_MISMATCH, out_error_detail,
                      "mel tensor size mismatch with vocoder input");
            return nullptr;
        }
        memcpy(input_tensor.mutable_data_ptr<float>(),
               mel_sharpened.data(),
               mel_sharpened.size() * sizeof(float));
    } else {
        // Portable backend: use set_input
        auto mel_for_vocoder = make_tensor_ptr<float>(
            std::vector<executorch::aten::SizesType>{
                (int32_t)mel_tensor.size(0),
                (int32_t)mel_tensor.size(1),
                (int32_t)mel_tensor.size(2)
            },
            std::move(mel_sharpened)
        );

        EValue mel_input(*mel_for_vocoder);
        auto err = vocoder_method.set_input(mel_input, 0);
        if (err != Error::Ok) {
            set_error(out_error, TTS_ERROR_VOCODER_INPUT_FAILED, out_error_detail,
                      "failed to set vocoder mel input");
            return nullptr;
        }
    }

    // Run vocoder
    auto vocoder_err = vocoder_method.execute();
    if (vocoder_err != Error::Ok) {
        set_error(out_error, TTS_ERROR_VOCODER_EXECUTE_FAILED, out_error_detail,
                  "vocoder execution failed");
        return nullptr;
    }

    // Get audio output
    const EValue& audio_output = vocoder_method.get_output(0);
    if (!audio_output.isTensor()) {
        set_error(out_error, TTS_ERROR_VOCODER_OUTPUT_INVALID, out_error_detail,
                  "vocoder output is not a tensor");
        return nullptr;
    }
    const auto& audio_tensor = audio_output.toTensor();

    // The vocoder's ISTFT runs with center=True, so the raw OLA buffer
    // begins with n_fft/2 samples of synthesis padding (zeroed by the
    // kernel's cold_trim) before the real signal. Skip past that to
    // align with what torch.istft(center=True) would return.
    const int HOP_LENGTH = 256;
    const int N_FFT_HALF = 512;  // n_fft = 1024 for Vocos
    size_t actual_audio_len = static_cast<size_t>(actual_mel_len) * HOP_LENGTH;
    size_t total_audio_len = audio_tensor.numel();
    size_t offset = (total_audio_len > static_cast<size_t>(N_FFT_HALF)) ? N_FFT_HALF : 0;
    size_t available = total_audio_len - offset;
    size_t num_samples = std::min(actual_audio_len, available);

    // Copy audio to output buffer
    float* audio = new float[num_samples];
    memcpy(audio,
           audio_tensor.const_data_ptr<float>() + offset,
           num_samples * sizeof(float));

    // The last ~30 samples ramp into a DC offset of ~3% of peak — an
    // OLA artifact at the boundary where the model's mel output
    // transitions from real frames to padding frames. Without a fade,
    // the WAV ending snaps that DC offset to silence and produces an
    // audible click. Raised-cosine taper over the last FADE_LEN samples.
    constexpr size_t FADE_LEN = 256;  // ~11.6 ms @ 22050 Hz
    if (num_samples > FADE_LEN) {
        const size_t fade_start = num_samples - FADE_LEN;
        const float pi = 3.14159265358979323846f;
        for (size_t i = fade_start; i < num_samples; ++i) {
            float p = static_cast<float>(i - fade_start) /
                      static_cast<float>(FADE_LEN);
            float gain = 0.5f * (1.0f + std::cos(pi * p));
            audio[i] *= gain;
        }
    }

    *out_sample_count = num_samples;
    return audio;
}

float* tts_synthesize(
    TtsSynthesizer* synth,
    const int64_t* tokens,
    size_t token_count,
    int64_t speaker_id,
    int64_t language_id,
    float pace,
    size_t* out_sample_count,
    TtsError* out_error,
    const char** out_error_detail
) {
    return synthesize_impl(
        synth, tokens, token_count, speaker_id, language_id, pace,
        out_sample_count,
        /*out_durations=*/nullptr, /*out_duration_count=*/nullptr,
        out_error, out_error_detail
    );
}

float* tts_synthesize_with_durations(
    TtsSynthesizer* synth,
    const int64_t* tokens,
    size_t token_count,
    int64_t speaker_id,
    int64_t language_id,
    float pace,
    size_t* out_sample_count,
    float** out_durations,
    size_t* out_duration_count,
    TtsError* out_error,
    const char** out_error_detail
) {
    if (!out_durations || !out_duration_count) {
        set_error(out_error, TTS_ERROR_INVALID_ARGUMENT, out_error_detail,
                  "out_durations or out_duration_count is null");
        return nullptr;
    }
    return synthesize_impl(
        synth, tokens, token_count, speaker_id, language_id, pace,
        out_sample_count,
        out_durations, out_duration_count,
        out_error, out_error_detail
    );
}

void tts_free_audio(float* audio) {
    delete[] audio;
}

void tts_free_durations(float* durations) {
    delete[] durations;
}

const char* tts_get_alphabet(
    TtsSynthesizer* synth,
    size_t* out_len,
    TtsError* out_error,
    const char** out_error_detail
) {
    set_error(out_error, TTS_OK, out_error_detail);

    if (!synth || !out_len) {
        set_error(out_error, TTS_ERROR_INVALID_ARGUMENT, out_error_detail,
                  "synth or out_len is null");
        return nullptr;
    }

    *out_len = 0;

    // Get the named data map from the voice program
    auto named_data_map_result = synth->voice_program->get_named_data_map();
    if (!named_data_map_result.ok()) {
        set_error(out_error, TTS_ERROR_NO_ALPHABET, out_error_detail,
                  "failed to get named data map from voice program");
        return nullptr;
    }

    const auto* named_data_map = named_data_map_result.get();
    if (!named_data_map) {
        set_error(out_error, TTS_ERROR_NO_ALPHABET, out_error_detail,
                  "named data map is null");
        return nullptr;
    }

    // Get the alphabet data
    auto alphabet_result = named_data_map->get_data("alphabet");
    if (!alphabet_result.ok()) {
        set_error(out_error, TTS_ERROR_NO_ALPHABET, out_error_detail,
                  "no 'alphabet' entry in named data map");
        return nullptr;
    }

    auto alphabet_buffer = std::move(alphabet_result.get());
    size_t len = alphabet_buffer.size();
    const void* data = alphabet_buffer.data();

    // Copy to heap-allocated buffer that caller can free
    char* result = new char[len];
    memcpy(result, data, len);
    *out_len = len;

    return result;
}

void tts_free_alphabet(const char* data) {
    delete[] data;
}

} // extern "C"
