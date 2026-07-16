# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

No environment variables or system toolchains are required — the ExecuTorch
dependency is a pure Cargo git dependency that vendors and builds its own native
XNNPACK.

```bash
# Build (first build clones executorch-rs + its submodules and compiles native
# XNNPACK, so it is slow; subsequent builds are cached).
cargo build --release

# Run tests
cargo test

# Synthesize text to audio
cargo run --release --example synthesize /path/to/voice.pte /path/to/vocoder.pte "Text to synthesize" [output.wav]

# Synthesize with per-word timings
cargo run --release --example word_timings /path/to/voice.pte /path/to/vocoder.pte "Text to synthesize"
```

## Architecture

**divvun-speech** is a TTS (Text-to-Speech) library built on the native Rust
ExecuTorch port (`executorch`, from github.com/divvun/executorch-rs), targeting
Sámi languages. There is no C++ / FFI layer.

### Data Flow
```
Text → TextProcessor (symbol encoding) → Synthesizer → Engine (ExecuTorch Module API)
    → Voice model (mel-spectrogram) → mel sharpening → Vocoder + custom ISTFT → Audio (f32 @ 22050 Hz)
```

### Key Components

- **`src/lib.rs`**: `Synthesizer` (wraps `Engine`), `Options`, `WordTiming`, and the `Error`/`SynthesisError` types.
- **`src/engine.rs`**: native `Engine` on the port's `Module` API — loads voice+vocoder, registers the custom ops + XNNPACK backend + optimized CPU kernels, runs the models, sharpens the mel, trims + fades the audio. Synth methods take `&mut self`.
- **`src/text.rs`**: `TextProcessor` for character-to-token encoding, word-boundary spans (`encode_text_with_spans`), and the `SymbolSet` enum.
- **`src/symbols.rs`**: predefined symbol sets for Sámi variants (SMJ, SME, SMA, …).

### Custom ops

The vocoder graph uses two custom ops implemented in the Rust port:
`tts::istft.out` (inverse STFT via `realfft`) and `tts::layer_norm.out`. They are
registered at engine startup and must be present before the vocoder method runs.

### Alphabet & word timings

The symbol alphabet is extracted from the voice model's embedded named data
(`alphabet` JSON); a model without it yields `Error::NoAlphabet`. Per-word
timings require the voice model to export a third output (`dur_pred`); models
without it still synthesize audio but return empty timings — check
`Synthesizer::supports_word_timings()`.

## Dependency

`executorch` is a git dependency (github.com/divvun/executorch-rs) built with the
`xnnpack` feature; the exact commit is pinned in `Cargo.lock`. The delegated
voice/vocoder models run on XNNPACK, with non-delegated "gap" ops on the port's
CPU kernels.
