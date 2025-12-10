// TTS Wrapper - Thin C API around the working C++ runner
// Wraps ExecuTorch voice + vocoder inference

#include "wrapper.hpp"

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <memory>

#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/extension/tensor/tensor_ptr.h>

using namespace executorch::runtime;
using namespace executorch::extension;

struct TtsSynthesizer {
    // Must be pointers - Program::load takes a pointer and keeps it
    std::unique_ptr<MmapDataLoader> voice_loader;
    std::unique_ptr<MmapDataLoader> vocoder_loader;
    std::unique_ptr<Program> voice_program;
    std::unique_ptr<Program> vocoder_program;
};

static inline void set_error(TtsError* out_error, TtsError err) {
    if (out_error) *out_error = err;
}

extern "C" {

TtsSynthesizer* tts_synthesizer_new(
    const char* voice_path,
    const char* vocoder_path,
    TtsError* out_error
) {
    set_error(out_error, TTS_OK);

    if (!voice_path || !vocoder_path) {
        set_error(out_error, TTS_ERROR_INVALID_ARGUMENT);
        return nullptr;
    }

    auto synth = std::make_unique<TtsSynthesizer>();

    // Load voice model with mmap
    auto voice_loader_result = MmapDataLoader::from(voice_path, MmapDataLoader::MlockConfig::UseMlockIgnoreErrors);
    if (!voice_loader_result.ok()) {
        set_error(out_error, TTS_ERROR_VOICE_LOAD_FAILED);
        return nullptr;
    }
    synth->voice_loader = std::make_unique<MmapDataLoader>(std::move(voice_loader_result.get()));

    auto voice_program_result = Program::load(synth->voice_loader.get());
    if (!voice_program_result.ok()) {
        set_error(out_error, TTS_ERROR_VOICE_LOAD_FAILED);
        return nullptr;
    }
    synth->voice_program = std::make_unique<Program>(std::move(voice_program_result.get()));

    // Load vocoder model with mmap
    auto vocoder_loader_result = MmapDataLoader::from(vocoder_path, MmapDataLoader::MlockConfig::UseMlockIgnoreErrors);
    if (!vocoder_loader_result.ok()) {
        set_error(out_error, TTS_ERROR_VOCODER_LOAD_FAILED);
        return nullptr;
    }
    synth->vocoder_loader = std::make_unique<MmapDataLoader>(std::move(vocoder_loader_result.get()));

    auto vocoder_program_result = Program::load(synth->vocoder_loader.get());
    if (!vocoder_program_result.ok()) {
        set_error(out_error, TTS_ERROR_VOCODER_LOAD_FAILED);
        return nullptr;
    }
    synth->vocoder_program = std::make_unique<Program>(std::move(vocoder_program_result.get()));

    return synth.release();
}

void tts_synthesizer_free(TtsSynthesizer* synth) {
    delete synth;
}

float* tts_synthesize(
    TtsSynthesizer* synth,
    const int64_t* tokens,
    size_t token_count,
    int64_t speaker_id,
    int64_t language_id,
    float pace,
    size_t* out_sample_count,
    TtsError* out_error
) {
    set_error(out_error, TTS_OK);

    if (!synth || !tokens || token_count == 0 || !out_sample_count) {
        set_error(out_error, TTS_ERROR_INVALID_ARGUMENT);
        return nullptr;
    }

    *out_sample_count = 0;

    // Memory for voice model
    testing::ManagedMemoryManager voice_mmm(
        128 * 1024 * 1024,  // 128MB planned memory
        64 * 1024 * 1024    // 64MB method allocator
    );

    auto voice_method_result = synth->voice_program->load_method("forward", &voice_mmm.get());
    if (!voice_method_result.ok()) {
        set_error(out_error, TTS_ERROR_VOICE_METHOD_LOAD_FAILED);
        return nullptr;
    }
    Method voice_method = std::move(voice_method_result.get());

    // Create input tensors
    std::vector<int64_t> text_tokens(tokens, tokens + token_count);
    auto text_tensor = make_tensor_ptr<int64_t>(
        std::vector<executorch::aten::SizesType>{1, (int32_t)token_count},
        std::move(text_tokens)
    );

    auto speaker_tensor = make_tensor_ptr<int64_t>(
        std::vector<executorch::aten::SizesType>{1},
        std::vector<int64_t>{speaker_id}
    );

    auto language_tensor = make_tensor_ptr<int64_t>(
        std::vector<executorch::aten::SizesType>{1},
        std::vector<int64_t>{language_id}
    );

    auto pace_tensor = make_tensor_ptr<float>(
        std::vector<executorch::aten::SizesType>{1},
        std::vector<float>{pace}
    );

    EValue inputs[] = {
        EValue(*text_tensor),
        EValue(*speaker_tensor),
        EValue(*language_tensor),
        EValue(*pace_tensor)
    };

    for (size_t i = 0; i < 4; i++) {
        auto err = voice_method.set_input(inputs[i], i);
        if (err != Error::Ok) {
            set_error(out_error, TTS_ERROR_VOICE_INPUT_FAILED);
            return nullptr;
        }
    }

    // Run voice model
    auto voice_err = voice_method.execute();
    if (voice_err != Error::Ok) {
        set_error(out_error, TTS_ERROR_VOICE_EXECUTE_FAILED);
        return nullptr;
    }

    // Get mel spectrogram output
    const EValue& mel_output = voice_method.get_output(0);
    if (!mel_output.isTensor()) {
        set_error(out_error, TTS_ERROR_VOICE_OUTPUT_INVALID);
        return nullptr;
    }
    const auto& mel_tensor = mel_output.toTensor();

    // Get actual mel length for trimming (if available)
    int64_t actual_mel_len = mel_tensor.size(2);
    if (voice_method.outputs_size() > 1) {
        const EValue& mel_lens_output = voice_method.get_output(1);
        if (mel_lens_output.isTensor()) {
            actual_mel_len = mel_lens_output.toTensor().const_data_ptr<int64_t>()[0];
        }
    }

    // Memory for vocoder
    testing::ManagedMemoryManager vocoder_mmm(
        512 * 1024 * 1024,  // 512MB planned memory
        128 * 1024 * 1024   // 128MB method allocator
    );

    auto vocoder_method_result = synth->vocoder_program->load_method("forward", &vocoder_mmm.get());
    if (!vocoder_method_result.ok()) {
        set_error(out_error, TTS_ERROR_VOCODER_METHOD_LOAD_FAILED);
        return nullptr;
    }
    Method vocoder_method = std::move(vocoder_method_result.get());

    // Set vocoder input
    EValue& input_evalue = vocoder_method.mutable_input(0);
    if (input_evalue.isTensor()) {
        // XNNPACK: copy data into pre-allocated tensor
        auto& input_tensor = input_evalue.toTensor();
        if (input_tensor.numel() != mel_tensor.numel()) {
            set_error(out_error, TTS_ERROR_SHAPE_MISMATCH);
            return nullptr;
        }
        memcpy(input_tensor.mutable_data_ptr<float>(),
               mel_tensor.const_data_ptr<float>(),
               mel_tensor.numel() * sizeof(float));
    } else {
        // Portable backend: use set_input
        size_t mel_numel = mel_tensor.numel();
        std::vector<float> mel_data(mel_numel);
        memcpy(mel_data.data(), mel_tensor.const_data_ptr<float>(), mel_numel * sizeof(float));

        auto mel_for_vocoder = make_tensor_ptr<float>(
            std::vector<executorch::aten::SizesType>{
                (int32_t)mel_tensor.size(0),
                (int32_t)mel_tensor.size(1),
                (int32_t)mel_tensor.size(2)
            },
            std::move(mel_data)
        );

        EValue mel_input(*mel_for_vocoder);
        auto err = vocoder_method.set_input(mel_input, 0);
        if (err != Error::Ok) {
            set_error(out_error, TTS_ERROR_VOCODER_INPUT_FAILED);
            return nullptr;
        }
    }

    // Run vocoder
    auto vocoder_err = vocoder_method.execute();
    if (vocoder_err != Error::Ok) {
        set_error(out_error, TTS_ERROR_VOCODER_EXECUTE_FAILED);
        return nullptr;
    }

    // Get audio output
    const EValue& audio_output = vocoder_method.get_output(0);
    if (!audio_output.isTensor()) {
        set_error(out_error, TTS_ERROR_VOCODER_OUTPUT_INVALID);
        return nullptr;
    }
    const auto& audio_tensor = audio_output.toTensor();

    // Calculate actual audio length (hop_length = 256 for Vocos)
    const int HOP_LENGTH = 256;
    size_t actual_audio_len = static_cast<size_t>(actual_mel_len) * HOP_LENGTH;
    size_t total_audio_len = audio_tensor.numel();
    size_t num_samples = std::min(actual_audio_len, total_audio_len);

    // Copy audio to output buffer
    float* audio = new float[num_samples];
    memcpy(audio, audio_tensor.const_data_ptr<float>(), num_samples * sizeof(float));

    *out_sample_count = num_samples;
    return audio;
}

void tts_free_audio(float* audio) {
    delete[] audio;
}

const char* tts_get_alphabet(
    TtsSynthesizer* synth,
    size_t* out_len,
    TtsError* out_error
) {
    set_error(out_error, TTS_OK);

    if (!synth || !out_len) {
        set_error(out_error, TTS_ERROR_INVALID_ARGUMENT);
        return nullptr;
    }

    *out_len = 0;

    // Get the named data map from the voice program
    auto named_data_map_result = synth->voice_program->get_named_data_map();
    if (!named_data_map_result.ok()) {
        set_error(out_error, TTS_ERROR_NO_ALPHABET);
        return nullptr;
    }

    const auto* named_data_map = named_data_map_result.get();
    if (!named_data_map) {
        set_error(out_error, TTS_ERROR_NO_ALPHABET);
        return nullptr;
    }

    // Get the alphabet data
    auto alphabet_result = named_data_map->get_data("alphabet");
    if (!alphabet_result.ok()) {
        set_error(out_error, TTS_ERROR_NO_ALPHABET);
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
