mod engine;
pub mod symbols;
mod text;

use std::path::Path;

pub use symbols::*;
pub use text::{SymbolSet, TextProcessor, WordSpan};

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

impl From<engine::EngineError> for SynthesisError {
    fn from(e: engine::EngineError) -> Self {
        use engine::EngineError as E;
        let (code, detail) = match e {
            E::VoiceLoad(et) => (Error::VoiceLoadFailed, Some(format!("{et:?}"))),
            E::VocoderLoad(et) => (Error::VocoderLoadFailed, Some(format!("{et:?}"))),
            E::VoiceMethod(et) => (Error::VoiceMethodLoadFailed, Some(format!("{et:?}"))),
            E::VocoderMethod(et) => (Error::VocoderMethodLoadFailed, Some(format!("{et:?}"))),
            E::VoiceExecute(et) => (Error::VoiceExecuteFailed, Some(format!("{et:?}"))),
            E::VocoderExecute(et) => (Error::VocoderExecuteFailed, Some(format!("{et:?}"))),
            E::VoiceOutput(s) => (Error::VoiceOutputInvalid, Some(s.to_string())),
            E::VocoderOutput(s) => (Error::VocoderOutputInvalid, Some(s.to_string())),
            E::ShapeMismatch(s) => (Error::ShapeMismatch, Some(s)),
            E::NoAlphabet => (Error::NoAlphabet, None),
        };
        SynthesisError { code, detail }
    }
}

pub struct Synthesizer {
    engine: engine::Engine,
    text_processor: TextProcessor,
}

unsafe impl Send for Synthesizer {}
unsafe impl Sync for Synthesizer {}

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
        let voice = voice_path.as_ref().to_str().ok_or(Error::InvalidPath)?;
        let vocoder = vocoder_path.as_ref().to_str().ok_or(Error::InvalidPath)?;

        let engine = engine::Engine::new(voice, vocoder)?;

        // Extract and parse the embedded JSON alphabet.
        let alphabet_bytes = engine.alphabet_json()?;
        let symbols: Vec<String> = serde_json::from_slice(&alphabet_bytes)
            .map_err(|_| SynthesisError::from(Error::InvalidAlphabetJson))?;
        let symbol_set = SymbolSet::from_vec(symbols);

        Ok(Self {
            engine,
            text_processor: TextProcessor::new(symbol_set),
        })
    }

    /// Synthesize text to audio samples.
    ///
    /// Returns f32 audio samples at 22050 Hz sample rate.
    pub fn synthesize(&mut self, text: &str, options: &Options) -> Result<Vec<f32>, SynthesisError> {
        let tokens = self.text_processor.encode_text(text);
        self.synthesize_tokens(&tokens, options)
    }

    /// Synthesize pre-tokenized input to audio samples.
    ///
    /// Use this if you want to handle text encoding yourself.
    pub fn synthesize_tokens(
        &mut self,
        tokens: &[i64],
        options: &Options,
    ) -> Result<Vec<f32>, SynthesisError> {
        if tokens.is_empty() {
            return Err(Error::EmptyInput.into());
        }

        let (audio, _) = self.engine.synthesize(
            tokens,
            options.speaker_id,
            options.language_id,
            options.pace,
            false,
        )?;

        tracing::info!(
            "TTS synthesize: tokens={}, samples={} ({:.3}s @ {} Hz)",
            tokens.len(),
            audio.len(),
            audio.len() as f32 / SAMPLE_RATE as f32,
            SAMPLE_RATE
        );

        Ok(audio)
    }

    /// Synthesize text and return audio plus per-word sample-offset timings.
    ///
    /// The full utterance is synthesized in a single pass (preserving prosody);
    /// per-word boundaries are computed from FastPitch's per-token duration
    /// prediction. Words are detected by `TextProcessor::encode_text_with_spans`
    /// — a "word" is a maximal run of alphanumeric chars or apostrophe.
    ///
    /// If the loaded voice model was exported without the `dur_pred` output (see
    /// [`Self::supports_word_timings`]), audio is still returned but the timings
    /// vector is empty — the call never fails for lack of duration support.
    pub fn synthesize_with_word_timings(
        &mut self,
        text: &str,
        options: &Options,
    ) -> Result<(Vec<f32>, Vec<WordTiming>), SynthesisError> {
        let (tokens, spans) = self.text_processor.encode_text_with_spans(text);
        if tokens.is_empty() {
            return Err(Error::EmptyInput.into());
        }

        let (audio, durations) = self.synthesize_tokens_with_durations(&tokens, options)?;
        let timings = word_timings_from_durations(&spans, &durations, options.pace, audio.len());
        Ok((audio, timings))
    }

    /// Same as `synthesize_with_word_timings` but pre-tokenized. The caller is
    /// responsible for producing the parallel `spans` (one per source word).
    pub fn synthesize_tokens_with_word_timings(
        &mut self,
        tokens: &[i64],
        spans: &[WordSpan],
        options: &Options,
    ) -> Result<(Vec<f32>, Vec<WordTiming>), SynthesisError> {
        if tokens.is_empty() {
            return Err(Error::EmptyInput.into());
        }
        let (audio, durations) = self.synthesize_tokens_with_durations(tokens, options)?;
        let timings = word_timings_from_durations(spans, &durations, options.pace, audio.len());
        Ok((audio, timings))
    }

    /// Low-level: synthesize and return audio + raw per-token durations.
    ///
    /// `durations[i]` is the model's predicted mel-frame count for input token
    /// `i`, BEFORE applying `pace` or rounding (mirrors FastPitch's `dur_pred`).
    /// To convert to integer mel-frame `reps`, do
    /// `((d / pace) + 0.5).floor().max(0.0) as u32` — same rounding as
    /// `regulate_len` in the model.
    pub fn synthesize_tokens_with_durations(
        &mut self,
        tokens: &[i64],
        options: &Options,
    ) -> Result<(Vec<f32>, Vec<f32>), SynthesisError> {
        if tokens.is_empty() {
            return Err(Error::EmptyInput.into());
        }

        let (audio, durations) = self.engine.synthesize(
            tokens,
            options.speaker_id,
            options.language_id,
            options.pace,
            true,
        )?;

        tracing::info!(
            "TTS synthesize+dur: tokens={}, samples={} ({:.3}s @ {} Hz), durs={}",
            tokens.len(),
            audio.len(),
            audio.len() as f32 / SAMPLE_RATE as f32,
            SAMPLE_RATE,
            durations.len(),
        );

        Ok((audio, durations))
    }

    /// Get access to the text processor for manual encoding.
    pub fn text_processor(&self) -> &TextProcessor {
        &self.text_processor
    }

    /// Whether the loaded voice model can produce per-word timings (i.e. it was
    /// exported with the `dur_pred` output). When false,
    /// [`Self::synthesize_with_word_timings`] still returns audio but with an
    /// empty timings vector.
    pub fn supports_word_timings(&self) -> bool {
        self.engine.supports_durations()
    }
}

/// Per-word audio offsets within the synthesized waveform.
///
/// `start_sample..end_sample` is a half-open range of sample indices into the
/// audio buffer returned alongside this timing. Consecutive `WordTiming`s tile
/// the full audio buffer with no gaps; trailing punctuation/whitespace is
/// folded into the preceding word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordTiming {
    pub word: String,
    pub start_sample: usize,
    pub end_sample: usize,
}

/// Audio sample rate for synthesized audio (22050 Hz)
pub const SAMPLE_RATE: u32 = 22050;

/// Vocoder hop length — samples per mel frame at SAMPLE_RATE.
pub const HOP_LENGTH: usize = 256;

fn word_timings_from_durations(
    spans: &[WordSpan],
    durations: &[f32],
    pace: f32,
    total_samples: usize,
) -> Vec<WordTiming> {
    if spans.is_empty() || durations.is_empty() {
        return Vec::new();
    }
    // Mirror regulate_len: reps = floor((dur / pace) + 0.5), clamp to 0.
    let pace = if pace > 0.0 { pace } else { 1.0 };
    let mut frame_cumsum: Vec<usize> = Vec::with_capacity(durations.len());
    let mut acc: usize = 0;
    for &d in durations {
        let r = ((d / pace) + 0.5).floor().max(0.0) as usize;
        acc = acc.saturating_add(r);
        frame_cumsum.push(acc);
    }

    let last_idx = spans.len() - 1;
    spans
        .iter()
        .enumerate()
        .map(|(i, span)| {
            let frame_start = if span.tok_start == 0 {
                0
            } else {
                frame_cumsum
                    .get(span.tok_start - 1)
                    .copied()
                    .unwrap_or(0)
            };
            let tok_end_idx = span.tok_end.saturating_sub(1).min(frame_cumsum.len() - 1);
            let frame_end = frame_cumsum.get(tok_end_idx).copied().unwrap_or(acc);
            let mut start_sample = frame_start.saturating_mul(HOP_LENGTH).min(total_samples);
            let mut end_sample = frame_end.saturating_mul(HOP_LENGTH).min(total_samples);
            if i == last_idx {
                end_sample = total_samples;
            }
            if end_sample < start_sample {
                end_sample = start_sample;
            }
            // Clamp start in case of rounding artifacts.
            if start_sample > total_samples {
                start_sample = total_samples;
            }
            WordTiming {
                word: span.word.clone(),
                start_sample,
                end_sample,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timings_from_durations_basic() {
        // Two words, 2 tokens each: ["hi", "yo"]
        let spans = vec![
            WordSpan { word: "hi".into(), tok_start: 0, tok_end: 2 },
            WordSpan { word: "yo".into(), tok_start: 2, tok_end: 4 },
        ];
        // dur = 10, 5, 8, 7  →  pace 1.0  → reps 10, 5, 8, 7  → cumsum 10, 15, 23, 30
        let durs = vec![10.0, 5.0, 8.0, 7.0];
        let total = 30 * HOP_LENGTH;
        let t = word_timings_from_durations(&spans, &durs, 1.0, total);
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].word, "hi");
        assert_eq!(t[0].start_sample, 0);
        assert_eq!(t[0].end_sample, 15 * HOP_LENGTH);
        assert_eq!(t[1].word, "yo");
        assert_eq!(t[1].start_sample, 15 * HOP_LENGTH);
        assert_eq!(t[1].end_sample, total); // last span clamped to total
    }

    #[test]
    fn timings_from_durations_pace() {
        // pace 2.0 halves durations (then rounds).
        let spans = vec![WordSpan { word: "hi".into(), tok_start: 0, tok_end: 2 }];
        let durs = vec![10.0, 4.0]; // reps at pace 2.0: 5, 2  → cumsum 5, 7
        let total = 7 * HOP_LENGTH;
        let t = word_timings_from_durations(&spans, &durs, 2.0, total);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].start_sample, 0);
        assert_eq!(t[0].end_sample, total);
    }

    #[test]
    fn timings_from_durations_empty() {
        let t = word_timings_from_durations(&[], &[1.0], 1.0, 100);
        assert!(t.is_empty());
    }
}
