use divvun_speech::{Options, Synthesizer, SAMPLE_RATE};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: {} <voice.pte> <vocoder.pte> <text> [--pace <f32>] [--speaker <i64>] [--language <i64>]",
            args[0]
        );
        std::process::exit(1);
    }

    let voice = PathBuf::from(&args[1]);
    let vocoder = PathBuf::from(&args[2]);
    let text = &args[3];

    let mut opts = Options::new();
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--pace" => { i += 1; opts = opts.with_pace(args[i].parse()?); }
            "--speaker" => { i += 1; opts = opts.with_speaker(args[i].parse()?); }
            "--language" => { i += 1; opts = opts.with_language(args[i].parse()?); }
            _ => {}
        }
        i += 1;
    }

    let mut synth = Synthesizer::new(&voice, &vocoder)?;
    if !synth.supports_word_timings() {
        eprintln!("[note] this voice model was exported without dur_pred — no per-word timings available (audio only)");
    }
    let (audio, timings) = match synth.synthesize_with_word_timings(text, &opts) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("synthesis error: {e:?}");
            std::process::exit(1);
        }
    };

    let sr = SAMPLE_RATE as f32;
    println!(
        "audio: {} samples ({:.2}s @ {} Hz), {} words\n",
        audio.len(),
        audio.len() as f32 / sr,
        SAMPLE_RATE,
        timings.len()
    );
    if timings.is_empty() {
        println!("(no per-word timings for this model)");
    } else {
        println!("{:>8}  {:>8}  {:>7}  word", "start(s)", "end(s)", "dur(s)");
        for t in &timings {
            let start = t.start_sample as f32 / sr;
            let end = t.end_sample as f32 / sr;
            println!("{:>8.3}  {:>8.3}  {:>7.3}  {}", start, end, end - start, t.word);
        }
    }
    Ok(())
}
