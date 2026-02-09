#pragma once
#include <cstdint>
#include <cstddef>

// Opaque handle to TTS synthesizer
typedef struct TtsSynthesizer TtsSynthesizer;

// Error codes (must match Rust Error enum in lib.rs)
enum TtsError : int32_t {
    TTS_OK = 0,
    TTS_ERROR_INVALID_ARGUMENT = 1,
    TTS_ERROR_VOICE_LOAD_FAILED = 2,
    TTS_ERROR_VOCODER_LOAD_FAILED = 3,
    TTS_ERROR_VOICE_METHOD_LOAD_FAILED = 4,
    TTS_ERROR_VOCODER_METHOD_LOAD_FAILED = 5,
    TTS_ERROR_VOICE_INPUT_FAILED = 6,
    TTS_ERROR_VOCODER_INPUT_FAILED = 7,
    TTS_ERROR_VOICE_EXECUTE_FAILED = 8,
    TTS_ERROR_VOCODER_EXECUTE_FAILED = 9,
    TTS_ERROR_VOICE_OUTPUT_INVALID = 10,
    TTS_ERROR_VOCODER_OUTPUT_INVALID = 11,
    TTS_ERROR_SHAPE_MISMATCH = 12,
    TTS_ERROR_NO_ALPHABET = 13,
};

#ifdef __cplusplus
extern "C" {
#endif

// Create a new TTS synthesizer with voice and vocoder models
// Returns NULL on failure, sets *out_error to error code
// out_error_detail optionally receives a static string with more context
TtsSynthesizer* tts_synthesizer_new(
    const char* voice_path,
    const char* vocoder_path,
    TtsError* out_error,
    const char** out_error_detail
);

// Free a TTS synthesizer
void tts_synthesizer_free(TtsSynthesizer* synth);

// Synthesize audio from token sequence
// Returns pointer to audio samples (f32, 22050 Hz), caller must free with tts_free_audio
// out_sample_count receives the number of samples
// Returns NULL on failure, sets *out_error to error code
// out_error_detail optionally receives a static string with more context
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
);

// Free audio buffer returned by tts_synthesize
void tts_free_audio(float* audio);

// Get embedded alphabet from voice model (JSON-encoded UTF-8 string)
// Returns NULL if not embedded, caller must free with tts_free_alphabet
// out_len receives the byte length of the returned buffer (not null-terminated)
// out_error_detail optionally receives a static string with more context
const char* tts_get_alphabet(
    TtsSynthesizer* synth,
    size_t* out_len,
    TtsError* out_error,
    const char** out_error_detail
);

// Free alphabet buffer returned by tts_get_alphabet
void tts_free_alphabet(const char* data);

#ifdef __cplusplus
}
#endif
