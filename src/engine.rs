//! Native synthesis engine backed by the Rust ExecuTorch port
//! (github.com/divvun/executorch-rs).
//!
//! Loads the voice + vocoder `.pte` models via the port's `Module` API,
//! registers our custom ops (`tts::istft.out`, `tts::layer_norm.out`) plus the
//! XNNPACK backend and optimized CPU kernels, runs the voice model, sharpens
//! the mel spectrogram, runs the vocoder, and trims + fades the audio.

use std::{sync::Once, time::Instant};

use executorch::extension::module::module::{LoadMode, Module};
use executorch::extension::tensor::tensor_ptr::{make_tensor_ptr_from_vec, TensorPtr};
use executorch::runtime::core::error::Error as EtError;
use executorch::runtime::core::evalue::EValue;
use executorch::runtime::core::portable_type::scalar_type::ScalarType;
use executorch::runtime::core::span::Span;
use executorch::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
use executorch::runtime::executor::program::Verification;

/// Vocoder hop length — audio samples per mel frame.
pub const HOP_LENGTH: usize = 256;
/// n_fft/2 for the Vocos vocoder (n_fft = 1024): the synthesis padding that
/// `torch.istft(center=True)` prepends and that we skip when trimming.
const N_FFT_HALF: usize = 512;
/// Raised-cosine fade length at the end of the utterance (~11.6 ms @ 22050 Hz),
/// killing the OLA boundary click.
const FADE_LEN: usize = 256;

/// Errors from the native engine, carrying the underlying ExecuTorch error
/// where relevant. Mapped to the crate's public `Error` in `lib.rs`.
#[derive(Debug)]
pub enum EngineError {
    VoiceLoad(EtError),
    VocoderLoad(EtError),
    VoiceMethod(EtError),
    VocoderMethod(EtError),
    VoiceExecute(EtError),
    VocoderExecute(EtError),
    VoiceOutput(&'static str),
    VocoderOutput(&'static str),
    ShapeMismatch(String),
    NoAlphabet,
}

/// Runtime registration (custom ops + XNNPACK backend) runs exactly once per
/// process — re-registering the same op name aborts.
static REGISTER: Once = Once::new();

fn ensure_runtime_registered() {
    REGISTER.call_once(|| {
        let err = executorch::custom_ops::register_custom_ops();
        if err != EtError::Ok {
            tracing::error!("failed to register custom ops: {err:?}");
        }
        // The voice/vocoder models are XNNPACK-delegated; the backend must be
        // registered before their methods can execute.
        let err = executorch::backends::xnnpack::register();
        if err != EtError::Ok {
            tracing::error!("failed to register XNNPACK backend: {err:?}");
        }
        // Merged CPU kernels for the non-delegated graph "gaps": optimized
        // kernels win per op, the `#[et_kernel]` codegen table (registry) fills
        // the rest. Single idempotent entry point.
        let err = executorch::kernels::optimized::register();
        if err != EtError::Ok {
            tracing::error!("failed to register CPU kernels: {err:?}");
        }
    });
}

fn span_to_vec_i32(span: Span<i32>) -> Vec<i32> {
    let n = span.size();
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(unsafe { *span.index(i) });
    }
    v
}

/// Build a reused input tensor with `'static` storage. It is created once per
/// engine and its data is overwritten in place before each run; leaking it lets
/// the resulting `EValue`s share the (`'static`) module's lifetime, which the
/// port's `Module::execute` requires. Bounded: a handful per engine.
fn leak_i64_tensor(sizes: Vec<i32>) -> &'static TensorPtr {
    let numel: usize = sizes.iter().map(|&s| s as usize).product();
    let tp = make_tensor_ptr_from_vec(
        sizes,
        vec![0i64; numel],
        Vec::new(),
        Vec::new(),
        ScalarType::Long,
        TensorShapeDynamism::STATIC,
    );
    Box::leak(Box::new(tp))
}

fn leak_f32_tensor(sizes: Vec<i32>) -> &'static TensorPtr {
    let numel: usize = sizes.iter().map(|&s| s as usize).product();
    let tp = make_tensor_ptr_from_vec(
        sizes,
        vec![0f32; numel],
        Vec::new(),
        Vec::new(),
        ScalarType::Float,
        TensorShapeDynamism::STATIC,
    );
    Box::leak(Box::new(tp))
}

pub struct Engine {
    voice: Module<'static>,
    vocoder: Module<'static>,

    // Reused, `'static` input tensors (see `leak_i64_tensor`).
    voice_tokens: &'static TensorPtr,
    voice_speaker: &'static TensorPtr,
    voice_language: &'static TensorPtr,
    voice_pace: &'static TensorPtr,
    vocoder_mel: &'static TensorPtr,

    max_seq_len: usize,
    /// Number of elements the vocoder's mel input expects (must match the voice
    /// model's mel output element count).
    vocoder_mel_numel: usize,
    /// Whether the voice model exposes a 3rd output (`dur_pred`) — required for
    /// per-word timings. Models exported without it still synthesize audio; they
    /// just can't produce durations, so timing requests degrade to empty rather
    /// than failing.
    has_durations: bool,
}

impl Engine {
    pub fn new(voice_path: &str, vocoder_path: &str) -> Result<Self, EngineError> {
        ensure_runtime_registered();

        let mut voice = Module::from_file_path(voice_path, LoadMode::Mmap, None, None, None, false);
        let err = voice.load(Verification::Minimal);
        if err != EtError::Ok {
            return Err(EngineError::VoiceLoad(err));
        }

        let mut vocoder =
            Module::from_file_path(vocoder_path, LoadMode::Mmap, None, None, None, false);
        let err = vocoder.load(Verification::Minimal);
        if err != EtError::Ok {
            return Err(EngineError::VocoderLoad(err));
        }

        // Introspect the voice model's input shapes: input 0 = tokens
        // [1, max_seq_len] (i64), 1 = speaker id, 2 = language id (i64), 3 = pace
        // (f32).
        let vmeta = voice.method_meta("forward").map_err(EngineError::VoiceMethod)?;
        if vmeta.num_inputs() < 4 {
            return Err(EngineError::VoiceOutput("voice 'forward' needs >= 4 inputs"));
        }
        let tokens_sizes = span_to_vec_i32(
            vmeta.input_tensor_meta(0).map_err(EngineError::VoiceMethod)?.sizes(),
        );
        let speaker_sizes = span_to_vec_i32(
            vmeta.input_tensor_meta(1).map_err(EngineError::VoiceMethod)?.sizes(),
        );
        let language_sizes = span_to_vec_i32(
            vmeta.input_tensor_meta(2).map_err(EngineError::VoiceMethod)?.sizes(),
        );
        let pace_sizes = span_to_vec_i32(
            vmeta.input_tensor_meta(3).map_err(EngineError::VoiceMethod)?.sizes(),
        );

        let max_seq_len = *tokens_sizes.last().unwrap_or(&0) as usize;
        if max_seq_len == 0 {
            return Err(EngineError::VoiceOutput("voice token input has no length"));
        }

        // Per-word timings need the voice model's 3rd output (`dur_pred`). Older
        // exports emit only (mel, mel_lens); detect that here so timing requests
        // can degrade gracefully instead of failing.
        let has_durations = vmeta.num_outputs() >= 3;

        // Introspect the vocoder's mel input shape [1, C, T].
        let vocmeta = vocoder.method_meta("forward").map_err(EngineError::VocoderMethod)?;
        let mel_sizes = span_to_vec_i32(
            vocmeta.input_tensor_meta(0).map_err(EngineError::VocoderMethod)?.sizes(),
        );
        let vocoder_mel_numel: usize = mel_sizes.iter().map(|&s| s as usize).product();

        let voice_tokens = leak_i64_tensor(tokens_sizes);
        let voice_speaker = leak_i64_tensor(speaker_sizes);
        let voice_language = leak_i64_tensor(language_sizes);
        let voice_pace = leak_f32_tensor(pace_sizes);
        let vocoder_mel = leak_f32_tensor(mel_sizes);

        Ok(Self {
            voice,
            vocoder,
            voice_tokens,
            voice_speaker,
            voice_language,
            voice_pace,
            vocoder_mel,
            max_seq_len,
            vocoder_mel_numel,
            has_durations,
        })
    }

    /// Whether the loaded voice model can produce per-token durations (and thus
    /// per-word timings). False for models exported without `dur_pred`.
    pub fn supports_durations(&self) -> bool {
        self.has_durations
    }

    /// The JSON-encoded alphabet embedded in the voice model's named data map.
    pub fn alphabet_json(&self) -> Result<Vec<u8>, EngineError> {
        let program = self.voice.program().ok_or(EngineError::NoAlphabet)?;
        let ndm = program.get_named_data_map().map_err(|_| EngineError::NoAlphabet)?;
        if ndm.is_null() {
            return Err(EngineError::NoAlphabet);
        }
        let buffer = unsafe { (*ndm).get_data("alphabet") }.map_err(|_| EngineError::NoAlphabet)?;
        let len = buffer.size();
        let data = buffer.data() as *const u8;
        if data.is_null() {
            return Err(EngineError::NoAlphabet);
        }
        Ok(unsafe { std::slice::from_raw_parts(data, len) }.to_vec())
    }

    /// Run voice + vocoder end to end. Returns the audio (f32 @ 22050 Hz) and,
    /// when `want_durations`, the model's raw per-token mel-frame durations
    /// (length `tokens.len()`, empty otherwise).
    pub fn synthesize(
        &mut self,
        tokens: &[i64],
        speaker_id: i64,
        language_id: i64,
        pace: f32,
        want_durations: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
        let total_start = Instant::now();
        if tokens.is_empty() {
            return Err(EngineError::ShapeMismatch("empty token sequence".into()));
        }
        if tokens.len() > self.max_seq_len {
            return Err(EngineError::ShapeMismatch(format!(
                "token count {} exceeds max sequence length {}",
                tokens.len(),
                self.max_seq_len
            )));
        }

        // Fill the reused voice inputs in place. Tokens are zero-filled (token 0
        // is padding, masked by the model) then overwritten with the sequence.
        unsafe {
            let tok_ptr = self.voice_tokens.tensor().mutable_data_ptr::<i64>();
            std::ptr::write_bytes(tok_ptr, 0, self.max_seq_len);
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), tok_ptr, tokens.len());
            *self.voice_speaker.tensor().mutable_data_ptr::<i64>() = speaker_id;
            *self.voice_language.tensor().mutable_data_ptr::<i64>() = language_id;
            *self.voice_pace.tensor().mutable_data_ptr::<f32>() = pace;
        }

        let voice_inputs = vec![
            EValue::from_tensor(self.voice_tokens.tensor()),
            EValue::from_tensor(self.voice_speaker.tensor()),
            EValue::from_tensor(self.voice_language.tensor()),
            EValue::from_tensor(self.voice_pace.tensor()),
        ];
        let voice_start = Instant::now();
        let voice_outputs = self
            .voice
            .execute("forward", &voice_inputs)
            .map_err(EngineError::VoiceExecute)?;
        let voice_elapsed = voice_start.elapsed();
        let mel_start = Instant::now();

        if voice_outputs.is_empty() || !voice_outputs[0].is_tensor() {
            return Err(EngineError::VoiceOutput("voice output 0 is not a tensor"));
        }
        let mel = voice_outputs[0].to_tensor();
        let mel_channels = mel.size(1) as usize;
        let mel_len = mel.size(2) as usize;
        let mel_numel = mel.numel() as usize;

        // Actual (untrimmed) mel length from output 1, if the model provides it.
        let mut actual_mel_len = mel_len;
        if voice_outputs.len() > 1 && voice_outputs[1].is_tensor() {
            let mel_lens = voice_outputs[1].to_tensor();
            actual_mel_len = unsafe { *mel_lens.const_data_ptr::<i64>() } as usize;
        }

        // Per-token durations from output 2 (dur_pred), if requested AND the
        // model provides them. A model without `dur_pred` (only mel + mel_lens)
        // yields empty durations: synthesis still succeeds, callers just get no
        // word timings rather than an error/crash. A too-short dur output is
        // likewise treated as "unavailable" instead of a hard failure.
        let mut durations = Vec::new();
        if want_durations && self.has_durations && voice_outputs.len() >= 3 && voice_outputs[2].is_tensor() {
            let dur = voice_outputs[2].to_tensor();
            let dur_padded = dur.size(dur.dim() - 1) as usize;
            if tokens.len() <= dur_padded {
                let dur_ptr = dur.const_data_ptr::<f32>();
                durations = unsafe { std::slice::from_raw_parts(dur_ptr, tokens.len()) }.to_vec();
            }
        }

        // Copy the mel out (we still need the voice output buffer intact until
        // here) and sharpen it.
        let mut mel_data =
            unsafe { std::slice::from_raw_parts(mel.const_data_ptr::<f32>(), mel_numel) }.to_vec();
        sharpen_mel(&mut mel_data, mel_channels, mel_len);

        if mel_data.len() != self.vocoder_mel_numel {
            return Err(EngineError::ShapeMismatch(format!(
                "mel element count {} does not match vocoder input {}",
                mel_data.len(),
                self.vocoder_mel_numel
            )));
        }

        // Run the vocoder on the sharpened mel.
        unsafe {
            let dst = self.vocoder_mel.tensor().mutable_data_ptr::<f32>();
            std::ptr::copy_nonoverlapping(mel_data.as_ptr(), dst, mel_data.len());
        }
        let vocoder_inputs = vec![EValue::from_tensor(self.vocoder_mel.tensor())];
        let mel_elapsed = mel_start.elapsed();
        let vocoder_start = Instant::now();
        let vocoder_outputs = self
            .vocoder
            .execute("forward", &vocoder_inputs)
            .map_err(EngineError::VocoderExecute)?;
        let vocoder_elapsed = vocoder_start.elapsed();
        let audio_start = Instant::now();

        if vocoder_outputs.is_empty() || !vocoder_outputs[0].is_tensor() {
            return Err(EngineError::VocoderOutput("vocoder output 0 is not a tensor"));
        }
        let audio_tensor = vocoder_outputs[0].to_tensor();
        let total_audio_len = audio_tensor.numel() as usize;

        // Skip the ISTFT center-padding prefix, then trim to the real length.
        let actual_audio_len = actual_mel_len * HOP_LENGTH;
        let offset = if total_audio_len > N_FFT_HALF { N_FFT_HALF } else { 0 };
        let available = total_audio_len - offset;
        let num_samples = actual_audio_len.min(available);

        let mut audio = unsafe {
            std::slice::from_raw_parts(audio_tensor.const_data_ptr::<f32>().add(offset), num_samples)
        }
        .to_vec();

        apply_fade(&mut audio);

        let audio_elapsed = audio_start.elapsed();
        tracing::info!(
            target: "divvun_speech::timing",
            tokens = tokens.len(),
            padded_mel_frames = mel_len,
            actual_mel_frames = actual_mel_len,
            output_samples = audio.len(),
            voice_ms = voice_elapsed.as_secs_f64() * 1000.0,
            mel_postprocess_ms = mel_elapsed.as_secs_f64() * 1000.0,
            vocoder_ms = vocoder_elapsed.as_secs_f64() * 1000.0,
            audio_postprocess_ms = audio_elapsed.as_secs_f64() * 1000.0,
            total_ms = total_start.elapsed().as_secs_f64() * 1000.0,
            "TTS phase timings"
        );

        Ok((audio, durations))
    }
}

/// Raised-cosine taper over the final `FADE_LEN` samples.
fn apply_fade(audio: &mut [f32]) {
    if audio.len() <= FADE_LEN {
        return;
    }
    let fade_start = audio.len() - FADE_LEN;
    for i in fade_start..audio.len() {
        let p = (i - fade_start) as f32 / FADE_LEN as f32;
        let gain = 0.5 * (1.0 + (std::f32::consts::PI * p).cos());
        audio[i] *= gain;
    }
}

/// Mel spectrogram sharpening (matches the C++ `sharpen_mel` / Python
/// `_process_sharpening`): two unsharp-mask passes plus a per-frequency tilt.
/// `mel` is row-major `[mel_channels, mel_len]`.
fn sharpen_mel(mel: &mut [f32], mel_channels: usize, mel_len: usize) {
    let mut blurred = mel.to_vec();

    // Pass 1: unsharp mask, sigma = 1.0, alpha = 0.2.
    gaussian_blur_2d(&mut blurred, mel_channels, mel_len, 1.0);
    for i in 0..mel.len() {
        mel[i] += 0.2 * (mel[i] - blurred[i]);
    }

    // Pass 2: unsharp mask, sigma = 3.0, alpha = 0.1.
    blurred.copy_from_slice(mel);
    gaussian_blur_2d(&mut blurred, mel_channels, mel_len, 3.0);
    for i in 0..mel.len() {
        mel[i] += 0.1 * (mel[i] - blurred[i]);
    }

    // Per-frequency tilt: mel[i][j] += (i - 40) * 0.01.
    for i in 0..mel_channels {
        let adjustment = (i as f32 - 40.0) * 0.01;
        let row = &mut mel[i * mel_len..(i + 1) * mel_len];
        for v in row.iter_mut() {
            *v += adjustment;
        }
    }
}

/// Separable 2D Gaussian blur (rows then columns) with zero boundaries, matching
/// the C++ FFT convolution: a normalized kernel of radius `ceil(3*sigma)` divided
/// by its full sum, so out-of-bounds (zero) samples attenuate the edges.
fn gaussian_blur_2d(data: &mut [f32], height: usize, width: usize, sigma: f32) {
    let radius = (3.0 * sigma).ceil() as isize;
    let ksize = (2 * radius + 1) as usize;
    let mut kernel = vec![0.0f32; ksize];
    let mut sum = 0.0f32;
    for k in -radius..=radius {
        let val = (-(k * k) as f32 / (2.0 * sigma * sigma)).exp();
        kernel[(k + radius) as usize] = val;
        sum += val;
    }
    for v in kernel.iter_mut() {
        *v /= sum;
    }

    // Blur rows into a scratch buffer.
    let mut tmp = vec![0.0f32; data.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for k in -radius..=radius {
                let xx = x as isize + k;
                if xx >= 0 && (xx as usize) < width {
                    acc += data[y * width + xx as usize] * kernel[(k + radius) as usize];
                }
            }
            tmp[y * width + x] = acc;
        }
    }

    // Blur columns back into `data`.
    for x in 0..width {
        for y in 0..height {
            let mut acc = 0.0f32;
            for k in -radius..=radius {
                let yy = y as isize + k;
                if yy >= 0 && (yy as usize) < height {
                    acc += tmp[yy as usize * width + x] * kernel[(k + radius) as usize];
                }
            }
            data[y * width + x] = acc;
        }
    }
}
