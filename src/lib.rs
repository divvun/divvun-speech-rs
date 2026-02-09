pub mod symbols;
mod text;

use std::ffi::CString;
use std::path::Path;

pub use symbols::*;
pub use text::{SymbolSet, TextProcessor};

/// Error with optional detail message from C layer.
#[derive(Debug, Clone)]
pub struct SynthesisError {
    pub code: Error,
    pub detail: Option<String>,
}

impl std::fmt::Display for SynthesisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.detail {
            Some(detail) => write!(f, "{}: {}", self.code, detail),
            None => write!(f, "{}", self.code),
        }
    }
}

impl std::error::Error for SynthesisError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.code)
    }
}

impl From<Error> for SynthesisError {
    fn from(code: Error) -> Self {
        Self { code, detail: None }
    }
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    // Rust-only errors (negative)
    #[error("Invalid JSON in alphabet")]
    InvalidAlphabetJson = -3,
    #[error("Invalid path")]
    InvalidPath = -2,
    #[error("Empty input")]
    EmptyInput = -1,

    // C API errors (match wrapper.hpp TtsError enum)
    #[error("Success")]
    Ok = 0,
    #[error("Invalid argument")]
    InvalidArgument = 1,
    #[error("Failed to load voice model")]
    VoiceLoadFailed = 2,
    #[error("Failed to load vocoder model")]
    VocoderLoadFailed = 3,
    #[error("Failed to load voice method")]
    VoiceMethodLoadFailed = 4,
    #[error("Failed to load vocoder method")]
    VocoderMethodLoadFailed = 5,
    #[error("Failed to set voice input")]
    VoiceInputFailed = 6,
    #[error("Failed to set vocoder input")]
    VocoderInputFailed = 7,
    #[error("Voice model execution failed")]
    VoiceExecuteFailed = 8,
    #[error("Vocoder execution failed")]
    VocoderExecuteFailed = 9,
    #[error("Voice model output is invalid")]
    VoiceOutputInvalid = 10,
    #[error("Vocoder output is invalid")]
    VocoderOutputInvalid = 11,
    #[error("Shape mismatch between voice and vocoder")]
    ShapeMismatch = 12,
    #[error("No alphabet embedded in voice model")]
    NoAlphabet = 13,
}

unsafe extern "C" {
    fn tts_synthesizer_new(
        voice: *const i8,
        vocoder: *const i8,
        out_error: *mut Error,
        out_error_detail: *mut *const i8,
    ) -> *mut std::ffi::c_void;
    fn tts_synthesizer_free(ptr: *mut std::ffi::c_void);
    fn tts_synthesize(
        synth: *mut std::ffi::c_void,
        tokens: *const i64,
        token_count: usize,
        speaker_id: i64,
        language_id: i64,
        pace: f32,
        out_sample_count: *mut usize,
        out_error: *mut Error,
        out_error_detail: *mut *const i8,
    ) -> *mut f32;
    fn tts_free_audio(audio: *mut f32);
    fn tts_get_alphabet(
        synth: *mut std::ffi::c_void,
        out_len: *mut usize,
        out_error: *mut Error,
        out_error_detail: *mut *const i8,
    ) -> *const i8;
    fn tts_free_alphabet(data: *const i8);
}

/// Helper to capture error detail from C layer
fn capture_error_detail(detail_ptr: *const i8) -> Option<String> {
    if detail_ptr.is_null() {
        None
    } else {
        // SAFETY: detail_ptr points to a static string in C
        let cstr = unsafe { std::ffi::CStr::from_ptr(detail_ptr) };
        Some(cstr.to_string_lossy().into_owned())
    }
}

fn make_error(code: Error, detail_ptr: *const i8) -> SynthesisError {
    SynthesisError {
        code,
        detail: capture_error_detail(detail_ptr),
    }
}

pub struct Synthesizer {
    ptr: *mut std::ffi::c_void,
    text_processor: TextProcessor,
}

unsafe impl Send for Synthesizer {}
unsafe impl Sync for Synthesizer {}

impl Drop for Synthesizer {
    fn drop(&mut self) {
        unsafe { tts_synthesizer_free(self.ptr) };
    }
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub speaker_id: i64,
    pub language_id: i64,
    pub pace: f32,
}

impl Options {
    pub fn new() -> Self {
        Self {
            speaker_id: 1,
            language_id: 1,
            pace: 1.0,
        }
    }

    pub fn with_speaker(mut self, speaker_id: i64) -> Self {
        self.speaker_id = speaker_id;
        self
    }

    pub fn with_language(mut self, language_id: i64) -> Self {
        self.language_id = language_id;
        self
    }

    pub fn with_pace(mut self, pace: f32) -> Self {
        self.pace = pace;
        self
    }
}

impl Synthesizer {
    /// Create a new TTS synthesizer with voice and vocoder models.
    ///
    /// The alphabet is automatically extracted from the voice model.
    /// If the model doesn't have an embedded alphabet, returns `Error::NoAlphabet`.
    ///
    /// # Arguments
    /// * `voice_path` - Path to the voice .pte model
    /// * `vocoder_path` - Path to the vocoder .pte model
    pub fn new<P: AsRef<Path>>(voice_path: P, vocoder_path: P) -> Result<Self, SynthesisError> {
        let voice = CString::new(voice_path.as_ref().to_str().ok_or(Error::InvalidPath)?)
            .map_err(|_| Error::InvalidPath)?;

        let vocoder = CString::new(vocoder_path.as_ref().to_str().ok_or(Error::InvalidPath)?)
            .map_err(|_| Error::InvalidPath)?;

        let mut error = Error::Ok;
        let mut error_detail: *const i8 = std::ptr::null();
        let ptr = unsafe {
            tts_synthesizer_new(voice.as_ptr(), vocoder.as_ptr(), &mut error, &mut error_detail)
        };

        if ptr.is_null() {
            return Err(make_error(error, error_detail));
        }

        // Extract alphabet from the model
        let mut len = 0usize;
        let mut alphabet_error = Error::Ok;
        let mut alphabet_error_detail: *const i8 = std::ptr::null();
        let alphabet_ptr = unsafe {
            tts_get_alphabet(ptr, &mut len, &mut alphabet_error, &mut alphabet_error_detail)
        };

        if alphabet_ptr.is_null() {
            unsafe { tts_synthesizer_free(ptr) };
            return Err(make_error(alphabet_error, alphabet_error_detail));
        }

        // Parse JSON alphabet
        let alphabet_bytes = unsafe { std::slice::from_raw_parts(alphabet_ptr as *const u8, len) };
        let symbols: Vec<String> = serde_json::from_slice(alphabet_bytes).map_err(|_| {
            unsafe {
                tts_free_alphabet(alphabet_ptr);
                tts_synthesizer_free(ptr);
            }
            SynthesisError::from(Error::InvalidAlphabetJson)
        })?;
        unsafe { tts_free_alphabet(alphabet_ptr) };

        let symbol_set = SymbolSet::from_vec(symbols);

        Ok(Self {
            ptr,
            text_processor: TextProcessor::new(symbol_set),
        })
    }

    /// Synthesize text to audio samples.
    ///
    /// Returns f32 audio samples at 22050 Hz sample rate.
    pub fn synthesize(&self, text: &str, options: &Options) -> Result<Vec<f32>, SynthesisError> {
        let tokens = self.text_processor.encode_text(text);
        self.synthesize_tokens(&tokens, options)
    }

    /// Synthesize pre-tokenized input to audio samples.
    ///
    /// Use this if you want to handle text encoding yourself.
    pub fn synthesize_tokens(
        &self,
        tokens: &[i64],
        options: &Options,
    ) -> Result<Vec<f32>, SynthesisError> {
        if tokens.is_empty() {
            return Err(Error::EmptyInput.into());
        }

        let mut sample_count = 0usize;
        let mut error = Error::Ok;
        let mut error_detail: *const i8 = std::ptr::null();
        let audio_ptr = unsafe {
            tts_synthesize(
                self.ptr,
                tokens.as_ptr(),
                tokens.len(),
                options.speaker_id,
                options.language_id,
                options.pace,
                &mut sample_count,
                &mut error,
                &mut error_detail,
            )
        };

        if audio_ptr.is_null() {
            return Err(make_error(error, error_detail));
        }

        let audio = unsafe { std::slice::from_raw_parts(audio_ptr, sample_count).to_vec() };
        unsafe { tts_free_audio(audio_ptr) };

        Ok(audio)
    }

    /// Get access to the text processor for manual encoding.
    pub fn text_processor(&self) -> &TextProcessor {
        &self.text_processor
    }
}

/// Audio sample rate for synthesized audio (22050 Hz)
pub const SAMPLE_RATE: u32 = 22050;
