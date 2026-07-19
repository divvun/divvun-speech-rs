//! Native synthesis engine backed by the Rust ExecuTorch port
//! (github.com/divvun/executorch-rs).
//!
//! Loads the voice + vocoder `.pte` models via the port's `Module` API,
//! registers our custom ops (`tts::istft.out`, `tts::layer_norm.out`) plus the
//! XNNPACK backend and optimized CPU kernels, runs the voice model, sharpens
//! the mel spectrogram, runs the vocoder, and trims + fades the audio.

use std::mem::ManuallyDrop;
use std::sync::{Arc, Once};
use std::time::Instant;

use executorch::extension::data_loader::shared_ptr_data_loader::SharedPtrDataLoader;
use executorch::extension::module::module::{LoadMode, Module};
use executorch::extension::tensor::tensor_ptr::{make_tensor_ptr_from_vec, TensorPtr};
use executorch::runtime::core::array_ref::ArrayRef;
use executorch::runtime::core::error::Error as EtError;
use executorch::runtime::core::evalue::EValue;
use executorch::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use executorch::runtime::core::portable_type::scalar_type::ScalarType;
use executorch::runtime::core::span::Span;
use executorch::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
use executorch::runtime::executor::program::Verification;
use mmap_io::segment::Segment;

macro_rules! xnn_profile_stage {
    ($stage:literal, $event:literal) => {
        #[cfg(feature = "xnnpack-profiling")]
        tracing::info!(
            xnn_profile_stage = $stage,
            xnn_profile_event = $event,
            "XNN profile stage"
        );
    };
}

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
fn leak_i64_tensor(sizes: Vec<i32>, dynamism: TensorShapeDynamism) -> &'static TensorPtr {
    let numel: usize = sizes.iter().map(|&s| s as usize).product();
    let tp = make_tensor_ptr_from_vec(
        sizes,
        vec![0i64; numel],
        Vec::new(),
        Vec::new(),
        ScalarType::Long,
        dynamism,
    );
    Box::leak(Box::new(tp))
}

fn leak_f32_tensor(sizes: Vec<i32>, dynamism: TensorShapeDynamism) -> &'static TensorPtr {
    let numel: usize = sizes.iter().map(|&s| s as usize).product();
    let tp = make_tensor_ptr_from_vec(
        sizes,
        vec![0f32; numel],
        Vec::new(),
        Vec::new(),
        ScalarType::Float,
        dynamism,
    );
    Box::leak(Box::new(tp))
}

/// State for the split voice model (`encode` + `decode` methods, FastPitch cut
/// at regulate_len). The host performs length regulation between the methods,
/// so the decoder and (dynamic) vocoder run at the ACTUAL mel length M instead
/// of the padded max — per-call cost scales with content length.
struct SplitState {
    /// Reusable `[1, max_mel_frames, d_model]` enc_rep input for `decode`,
    /// resized to the actual M per call. Leaked like the other inputs;
    /// reclaimed in `Drop`.
    enc_rep: &'static TensorPtr,
    max_mel_frames: usize,
    d_model: usize,
    /// Whether the vocoder accepts dynamic `[1, 80, M]` input. Probed on first
    /// use: `None` = unknown, `Some(false)` = fixed-shape vocoder (pad to max).
    vocoder_dynamic: Option<bool>,
}

pub struct Engine {
    // ManuallyDrop so `Drop for Engine` can drop the modules BEFORE reclaiming
    // the leaked input tensors below: a loaded Method may alias input tensor
    // data (ExecuTorch's `share_tensor_data` path for non-memory-planned
    // inputs), so the tensors must outlive the modules.
    voice: ManuallyDrop<Module<'static>>,
    vocoder: ManuallyDrop<Module<'static>>,

    // Reused input tensors, leaked to `'static` to satisfy `Module::execute`'s
    // input lifetime bound, and reclaimed in `Drop` (see below) so engine churn
    // doesn't accumulate leaks.
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
    /// Present when the voice model is the split export (encode + decode
    /// methods); `None` for the single-pass `forward` export.
    split: Option<SplitState>,
}

impl Engine {
    pub fn new(voice_path: &str, vocoder_path: &str) -> Result<Self, EngineError> {
        ensure_runtime_registered();

        let voice = Module::from_file_path(voice_path, LoadMode::Mmap, None, None, None, false);
        let vocoder = Module::from_file_path(vocoder_path, LoadMode::Mmap, None, None, None, false);
        Self::from_modules(voice, vocoder)
    }

    /// Load voice and vocoder programs directly from owned memory-mapped
    /// segments. The module's data loader owns each segment, so ExecuTorch's
    /// borrowed program data remains valid for the lifetime of the engine.
    pub fn new_mapped(voice: Segment, vocoder: Segment) -> Result<Self, EngineError> {
        ensure_runtime_registered();

        let voice = Self::module_from_segment(voice).map_err(EngineError::VoiceLoad)?;
        let vocoder = Self::module_from_segment(vocoder).map_err(EngineError::VocoderLoad)?;
        Self::from_modules(voice, vocoder)
    }

    fn module_from_segment(segment: Segment) -> Result<Module<'static>, EtError> {
        let segment = Arc::new(segment);
        let bytes = segment.as_slice().map_err(|_| EtError::InvalidArgument)?;
        if (bytes.as_ptr() as usize) % 16 != 0 {
            return Err(EtError::InvalidArgument);
        }

        let data_ptr = bytes.as_ptr() as *const core::ffi::c_void;
        let size = bytes.len();
        let owner: Arc<dyn core::any::Any + Send + Sync> = segment.clone();
        let loader = SharedPtrDataLoader::new(owner, data_ptr, size);
        Ok(Module::from_data_loader(
            Box::new(loader),
            None,
            None,
            None,
            None,
            false,
        ))
    }

    fn from_modules(
        mut voice: Module<'static>,
        mut vocoder: Module<'static>,
    ) -> Result<Self, EngineError> {
        let start = Instant::now();
        let err = voice.load(Verification::Minimal);
        if err != EtError::Ok {
            return Err(EngineError::VoiceLoad(err));
        }
        tracing::info!(
            elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
            "voice program loaded"
        );

        let start = Instant::now();
        let err = vocoder.load(Verification::Minimal);
        if err != EtError::Ok {
            return Err(EngineError::VocoderLoad(err));
        }
        tracing::info!(
            elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
            "vocoder program loaded"
        );

        // Split export (encode + decode methods, FastPitch cut at regulate_len)
        // vs single-pass `forward` export.
        let method_names = voice.method_names().map_err(EngineError::VoiceMethod)?;
        let is_split = method_names.contains("encode") && method_names.contains("decode");
        let main_method = if is_split { "encode" } else { "forward" };

        // Loading a Module only parses the program. Eagerly load its methods so
        // callers that construct a voice for prewarming do not pay backend and
        // delegate initialization during the first synthesis.
        for method_name in if is_split {
            &["encode", "decode"][..]
        } else {
            &["forward"][..]
        } {
            let start = Instant::now();
            let err = voice.load_method_with_defaults(method_name);
            if err != EtError::Ok {
                return Err(EngineError::VoiceMethod(err));
            }
            tracing::info!(
                method = method_name,
                elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
                "voice method loaded"
            );
        }
        let start = Instant::now();
        let err = vocoder.load_method_with_defaults("forward");
        if err != EtError::Ok {
            return Err(EngineError::VocoderMethod(err));
        }
        tracing::info!(
            method = "forward",
            elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
            "vocoder method loaded"
        );

        // Introspect the voice model's input shapes: input 0 = tokens
        // [1, max_seq_len] (i64), 1 = speaker id, 2 = language id (i64), and —
        // single-pass only — 3 = pace (f32). The split export applies pace on
        // the host in regulate_len, so `encode` has no pace input.
        let vmeta = voice.method_meta(main_method).map_err(EngineError::VoiceMethod)?;
        let min_inputs = if is_split { 3 } else { 4 };
        if vmeta.num_inputs() < min_inputs {
            return Err(EngineError::VoiceOutput("voice method is missing inputs"));
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
        let pace_sizes = if is_split {
            vec![1]
        } else {
            span_to_vec_i32(vmeta.input_tensor_meta(3).map_err(EngineError::VoiceMethod)?.sizes())
        };

        let max_seq_len = *tokens_sizes.last().unwrap_or(&0) as usize;
        if max_seq_len == 0 {
            return Err(EngineError::VoiceOutput("voice token input has no length"));
        }

        // Per-word timings need per-token durations: `encode`'s 2nd output in
        // the split export, `forward`'s 3rd output (dur_pred) otherwise. Older
        // single-pass exports emit only (mel, mel_lens); detect that here so
        // timing requests degrade gracefully instead of failing.
        let has_durations = vmeta.num_outputs() >= if is_split { 2 } else { 3 };

        // Split export: introspect `decode`'s enc_rep input [1, max_mel, d_model].
        let split = if is_split {
            let dmeta = voice.method_meta("decode").map_err(EngineError::VoiceMethod)?;
            let rep_sizes = span_to_vec_i32(
                dmeta.input_tensor_meta(0).map_err(EngineError::VoiceMethod)?.sizes(),
            );
            if rep_sizes.len() != 3 {
                return Err(EngineError::VoiceOutput("decode input must be [1, M, d_model]"));
            }
            let max_mel_frames = rep_sizes[1] as usize;
            let d_model = rep_sizes[2] as usize;
            let enc_rep = leak_f32_tensor(rep_sizes, TensorShapeDynamism::DYNAMIC_BOUND);
            Some(SplitState { enc_rep, max_mel_frames, d_model, vocoder_dynamic: None })
        } else {
            None
        };

        // Introspect the vocoder's mel input shape [1, C, T].
        let vocmeta = vocoder.method_meta("forward").map_err(EngineError::VocoderMethod)?;
        let mel_sizes = span_to_vec_i32(
            vocmeta.input_tensor_meta(0).map_err(EngineError::VocoderMethod)?.sizes(),
        );
        let vocoder_mel_numel: usize = mel_sizes.iter().map(|&s| s as usize).product();

        // The token input is resized to the actual sequence length per call
        // (the voice model is exported with a bounded-dynamic seq dim), so the
        // reusable tensor is DYNAMIC_BOUND with capacity max_seq_len. Running
        // at the real length skips the encoder work for padding positions.
        let voice_tokens = leak_i64_tensor(tokens_sizes, TensorShapeDynamism::DYNAMIC_BOUND);
        let voice_speaker = leak_i64_tensor(speaker_sizes, TensorShapeDynamism::STATIC);
        let voice_language = leak_i64_tensor(language_sizes, TensorShapeDynamism::STATIC);
        let voice_pace = leak_f32_tensor(pace_sizes, TensorShapeDynamism::STATIC);
        // DYNAMIC_BOUND so the split path can feed the actual [1, 80, M]; the
        // forward path never resizes it (stays at the full capacity).
        let vocoder_mel = leak_f32_tensor(mel_sizes, TensorShapeDynamism::DYNAMIC_BOUND);

        Ok(Self {
            voice: ManuallyDrop::new(voice),
            vocoder: ManuallyDrop::new(vocoder),
            voice_tokens,
            voice_speaker,
            voice_language,
            voice_pace,
            vocoder_mel,
            max_seq_len,
            vocoder_mel_numel,
            has_durations,
            split,
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
        if self.split.is_some() {
            return self.synthesize_split(tokens, speaker_id, language_id, pace, want_durations);
        }

        // Resize the reusable token tensor to the actual sequence length (the
        // voice model has a bounded-dynamic seq dim, verified to work at
        // runtime: shorter inputs skip the encoder work for padding positions
        // and avoid pad-boundary artifacts) and fill the inputs in place.
        let tok_sizes: [i32; 2] = [1, tokens.len() as i32];
        let err = resize_tensor(
            &self.voice_tokens.tensor(),
            ArrayRef::from_raw_parts(tok_sizes.as_ptr(), tok_sizes.len()),
        );
        if err != EtError::Ok {
            return Err(EngineError::ShapeMismatch(format!(
                "failed to resize token input to [1, {}]: {err:?}",
                tokens.len()
            )));
        }
        unsafe {
            let tok_ptr = self.voice_tokens.tensor().mutable_data_ptr::<i64>();
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
        xnn_profile_stage!("voice", "begin");
        let voice_outputs = self
            .voice
            .execute("forward", &voice_inputs)
            .map_err(EngineError::VoiceExecute)?;
        xnn_profile_stage!("voice", "end");
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
        xnn_profile_stage!("vocoder", "begin");
        let vocoder_outputs = self
            .vocoder
            .execute("forward", &vocoder_inputs)
            .map_err(EngineError::VocoderExecute)?;
        xnn_profile_stage!("vocoder", "end");
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

    /// Split-model path: `encode` → host-side regulate_len → `decode` →
    /// vocoder, everything at the ACTUAL mel length M so per-call cost scales
    /// with content instead of paying the padded max.
    fn synthesize_split(
        &mut self,
        tokens: &[i64],
        speaker_id: i64,
        language_id: i64,
        pace: f32,
        want_durations: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
        let total_start = Instant::now();
        let (enc_rep, max_mel_frames, d_model) = {
            let s = self.split.as_ref().expect("split state");
            (s.enc_rep, s.max_mel_frames, s.d_model)
        };

        // ---- encode: tokens [1, N] -> enc_out [1, N, d_model] + dur [1, N] ----
        let tok_sizes: [i32; 2] = [1, tokens.len() as i32];
        let err = resize_tensor(
            &self.voice_tokens.tensor(),
            ArrayRef::from_raw_parts(tok_sizes.as_ptr(), tok_sizes.len()),
        );
        if err != EtError::Ok {
            return Err(EngineError::ShapeMismatch(format!(
                "failed to resize token input to [1, {}]: {err:?}",
                tokens.len()
            )));
        }
        unsafe {
            let tok_ptr = self.voice_tokens.tensor().mutable_data_ptr::<i64>();
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), tok_ptr, tokens.len());
            *self.voice_speaker.tensor().mutable_data_ptr::<i64>() = speaker_id;
            *self.voice_language.tensor().mutable_data_ptr::<i64>() = language_id;
        }
        let encode_inputs = vec![
            EValue::from_tensor(self.voice_tokens.tensor()),
            EValue::from_tensor(self.voice_speaker.tensor()),
            EValue::from_tensor(self.voice_language.tensor()),
        ];
        let encode_start = Instant::now();
        xnn_profile_stage!("encode", "begin");
        let encode_outputs = self
            .voice
            .execute("encode", &encode_inputs)
            .map_err(EngineError::VoiceExecute)?;
        xnn_profile_stage!("encode", "end");
        let encode_elapsed = encode_start.elapsed();

        if encode_outputs.len() < 2 || !encode_outputs[0].is_tensor() || !encode_outputs[1].is_tensor()
        {
            return Err(EngineError::VoiceOutput("encode must output (enc_out, dur_pred)"));
        }
        let enc_out = encode_outputs[0].to_tensor();
        if enc_out.dim() != 3
            || enc_out.size(2) as usize != d_model
            || (enc_out.size(1) as usize) < tokens.len()
        {
            return Err(EngineError::ShapeMismatch(format!(
                "unexpected enc_out shape (last dim {}, expected d_model {})",
                enc_out.size(2),
                d_model
            )));
        }
        let dur = encode_outputs[1].to_tensor();
        if (dur.numel() as usize) < tokens.len() {
            return Err(EngineError::VoiceOutput("dur_pred shorter than token count"));
        }
        let durations: Vec<f32> =
            unsafe { std::slice::from_raw_parts(dur.const_data_ptr::<f32>(), tokens.len()) }
                .to_vec();

        // ---- regulate_len on the host (mirrors the model: reps = floor(dur /
        // pace + 0.5), frames capped at max_mel_frames) ----
        let pace = if pace > 0.0 { pace } else { 1.0 };
        let reps: Vec<usize> = durations
            .iter()
            .map(|&d| {
                let r = (d / pace + 0.5).floor();
                if r > 0.0 { r as usize } else { 0 }
            })
            .collect();
        let m_real: usize = reps.iter().sum::<usize>().min(max_mel_frames);
        if m_real == 0 {
            return Err(EngineError::VoiceOutput("predicted zero total duration"));
        }
        // decode is traced with mel_frames >= 4; pad tiny inputs by repeating
        // the last frame (audio is trimmed back to m_real below).
        let m = m_real.max(4);

        let rep_sizes: [i32; 3] = [1, m as i32, d_model as i32];
        let err = resize_tensor(
            &enc_rep.tensor(),
            ArrayRef::from_raw_parts(rep_sizes.as_ptr(), rep_sizes.len()),
        );
        if err != EtError::Ok {
            return Err(EngineError::ShapeMismatch(format!(
                "failed to resize enc_rep to [1, {m}, {d_model}]: {err:?}"
            )));
        }
        unsafe {
            let src = enc_out.const_data_ptr::<f32>();
            let dst = enc_rep.tensor().mutable_data_ptr::<f32>();
            let mut frame = 0usize;
            'fill: for (i, &r) in reps.iter().enumerate() {
                for _ in 0..r {
                    if frame >= m {
                        break 'fill;
                    }
                    std::ptr::copy_nonoverlapping(
                        src.add(i * d_model),
                        dst.add(frame * d_model),
                        d_model,
                    );
                    frame += 1;
                }
            }
            // Pad up to the m >= 4 floor by repeating the last written frame.
            while frame > 0 && frame < m {
                std::ptr::copy_nonoverlapping(
                    dst.add((frame - 1) * d_model) as *const f32,
                    dst.add(frame * d_model),
                    d_model,
                );
                frame += 1;
            }
        }

        // ---- decode: enc_rep [1, M, d_model] -> mel [1, 80, M] ----
        let decode_inputs = vec![EValue::from_tensor(enc_rep.tensor())];
        let decode_start = Instant::now();
        xnn_profile_stage!("decode", "begin");
        let decode_outputs = self
            .voice
            .execute("decode", &decode_inputs)
            .map_err(EngineError::VoiceExecute)?;
        xnn_profile_stage!("decode", "end");
        let decode_elapsed = decode_start.elapsed();
        if decode_outputs.is_empty() || !decode_outputs[0].is_tensor() {
            return Err(EngineError::VoiceOutput("decode output 0 is not a tensor"));
        }
        let mel = decode_outputs[0].to_tensor();
        let mel_channels = mel.size(1) as usize;
        let mel_len = mel.size(2) as usize;
        let mel_numel = mel.numel() as usize;

        let mel_start = Instant::now();
        let mut mel_data =
            unsafe { std::slice::from_raw_parts(mel.const_data_ptr::<f32>(), mel_numel) }.to_vec();
        sharpen_mel(&mut mel_data, mel_channels, mel_len);
        let mel_elapsed = mel_start.elapsed();

        // ---- vocoder: dynamic [1, 80, M] if the export supports it, else pad
        // into the fixed [1, 80, T_max] (probed once, remembered) ----
        let vocoder_start = Instant::now();
        xnn_profile_stage!("vocoder", "begin");
        let prior = self.split.as_ref().expect("split state").vocoder_dynamic;
        let mut used_dynamic = false;
        let mut vocoder_outputs = None;
        if prior != Some(false) {
            let vsizes: [i32; 3] = [1, mel_channels as i32, mel_len as i32];
            let err = resize_tensor(
                &self.vocoder_mel.tensor(),
                ArrayRef::from_raw_parts(vsizes.as_ptr(), vsizes.len()),
            );
            if err == EtError::Ok {
                unsafe {
                    let dst = self.vocoder_mel.tensor().mutable_data_ptr::<f32>();
                    std::ptr::copy_nonoverlapping(mel_data.as_ptr(), dst, mel_data.len());
                }
                let vocoder_inputs = vec![EValue::from_tensor(self.vocoder_mel.tensor())];
                match self.vocoder.execute("forward", &vocoder_inputs) {
                    Ok(o) => {
                        used_dynamic = true;
                        vocoder_outputs = Some(o);
                    }
                    Err(e) => {
                        if prior == Some(true) {
                            return Err(EngineError::VocoderExecute(e));
                        }
                        tracing::info!(
                            "vocoder rejected dynamic mel input; falling back to fixed-shape padding"
                        );
                    }
                }
            }
        }
        let vocoder_outputs = match vocoder_outputs {
            Some(o) => o,
            None => {
                // Fixed-shape vocoder: zero-pad each mel row out to T_max.
                let t_max = self.vocoder_mel_numel / mel_channels;
                let vsizes: [i32; 3] = [1, mel_channels as i32, t_max as i32];
                let err = resize_tensor(
                    &self.vocoder_mel.tensor(),
                    ArrayRef::from_raw_parts(vsizes.as_ptr(), vsizes.len()),
                );
                if err != EtError::Ok {
                    return Err(EngineError::ShapeMismatch(format!(
                        "failed to restore vocoder mel input to [1, {mel_channels}, {t_max}]: {err:?}"
                    )));
                }
                unsafe {
                    let dst = self.vocoder_mel.tensor().mutable_data_ptr::<f32>();
                    std::ptr::write_bytes(dst, 0, self.vocoder_mel_numel);
                    for c in 0..mel_channels {
                        std::ptr::copy_nonoverlapping(
                            mel_data.as_ptr().add(c * mel_len),
                            dst.add(c * t_max),
                            mel_len,
                        );
                    }
                }
                let vocoder_inputs = vec![EValue::from_tensor(self.vocoder_mel.tensor())];
                self.vocoder
                    .execute("forward", &vocoder_inputs)
                    .map_err(EngineError::VocoderExecute)?
            }
        };
        self.split.as_mut().expect("split state").vocoder_dynamic = Some(used_dynamic);
        xnn_profile_stage!("vocoder", "end");
        let vocoder_elapsed = vocoder_start.elapsed();

        let audio_start = Instant::now();
        if vocoder_outputs.is_empty() || !vocoder_outputs[0].is_tensor() {
            return Err(EngineError::VocoderOutput("vocoder output 0 is not a tensor"));
        }
        let audio_tensor = vocoder_outputs[0].to_tensor();
        let total_audio_len = audio_tensor.numel() as usize;

        // Skip the ISTFT center-padding prefix, then trim to the real length.
        let actual_audio_len = m_real * HOP_LENGTH;
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
            mel_frames = m_real,
            output_samples = audio.len(),
            vocoder_dynamic = used_dynamic,
            encode_ms = encode_elapsed.as_secs_f64() * 1000.0,
            decode_ms = decode_elapsed.as_secs_f64() * 1000.0,
            mel_postprocess_ms = mel_elapsed.as_secs_f64() * 1000.0,
            vocoder_ms = vocoder_elapsed.as_secs_f64() * 1000.0,
            audio_postprocess_ms = audio_elapsed.as_secs_f64() * 1000.0,
            total_ms = total_start.elapsed().as_secs_f64() * 1000.0,
            "TTS phase timings (split)"
        );

        let durations = if want_durations { durations } else { Vec::new() };
        Ok((audio, durations))
    }
}

// Reclaim the input tensors that were leaked to `'static` for
// `Module::execute`'s input lifetime bound, so repeated Engine
// creation/destruction (e.g. voice switching) doesn't accumulate leaks.
//
// SAFETY: the modules are dropped first (they may alias input tensor data via
// ExecuTorch's `share_tensor_data` path); after that nothing references the
// tensors — borrows created in `synthesize` never outlive the call — so
// re-boxing and dropping the `Box::leak`ed allocations is sound. The
// `ManuallyDrop` fields are not touched again after this runs.
impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.voice);
            ManuallyDrop::drop(&mut self.vocoder);
            let mut leaked = vec![
                self.voice_tokens,
                self.voice_speaker,
                self.voice_language,
                self.voice_pace,
                self.vocoder_mel,
            ];
            if let Some(s) = &self.split {
                leaked.push(s.enc_rep);
            }
            for tp in leaked {
                drop(Box::from_raw(tp as *const TensorPtr as *mut TensorPtr));
            }
        }
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
