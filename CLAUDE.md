# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Required environment variables
export EXECUTORCH_SRC=/path/to/executorch-parent   # Parent dir of executorch/ source (for CMake includes)
export EXECUTORCH_SYSROOT=/path/to/sysroot/<arch>  # Arch-specific install prefix (for linking .a files)

# Build
cargo build --release

# Run tests
cargo test

# Run example (synthesize text to audio)
cargo run --example synthesize /path/to/voice.pte /path/to/vocoder.pte "Text to synthesize" [output.wav]
```

## Architecture

**divvun-speech** is a TTS (Text-to-Speech) library wrapping ExecuTorch for neural voice synthesis, targeting Sámi languages.

### Data Flow
```
Text → TextProcessor (symbol encoding) → Synthesizer → C++ wrapper → ExecuTorch
    → Voice model (mel-spectrogram) → Vocoder + ISTFT → Audio (f32 @ 22050 Hz)
```

### Key Components

- **`src/lib.rs`**: Main `Synthesizer` struct and FFI bindings to C++ wrapper
- **`src/text.rs`**: `TextProcessor` for character-to-token encoding, `SymbolSet` enum
- **`src/symbols.rs`**: Predefined symbol sets for Sámi variants (SMJ, SME, SMA)
- **`wrapper/`**: C++ ExecuTorch integration with custom ops (ISTFT, LayerNorm)
- **`build.rs`**: CMake-based build requiring both `EXECUTORCH_SRC` and `EXECUTORCH_SYSROOT` env vars

### FFI Pattern
Rust `Error` enum (repr i32) maps directly to C `TtsError`. Memory managed via explicit `_free` functions. Alphabet/symbol sets can be extracted from voice model JSON metadata or fall back to static definitions.

## Platform Notes

- macOS: Links CoreML, Accelerate, Foundation frameworks + clang runtime
- Linux: Links libstdc++
- Uses whole-archive linking to force ExecuTorch kernel symbol registration
