use std::{env, process::ExitCode, time::Instant};

use divvun_speech::{Options, SAMPLE_RATE, Synthesizer};
use tracing_subscriber::EnvFilter;

fn usage(program: &str) -> ! {
    eprintln!(
        "usage: {program} <voice.pte> <vocoder.pte> <speaker> <language> <runs> <text> [text ...]"
    );
    std::process::exit(2);
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();

    let args = env::args().collect::<Vec<_>>();
    if args.len() < 7 {
        usage(&args[0]);
    }

    let speaker_id = args[3].parse::<i64>().unwrap_or_else(|_| usage(&args[0]));
    let language_id = args[4].parse::<i64>().unwrap_or_else(|_| usage(&args[0]));
    let runs = args[5].parse::<usize>().unwrap_or_else(|_| usage(&args[0]));
    let texts = &args[6..];
    let options = Options::new()
        .with_speaker(speaker_id)
        .with_language(language_id);

    println!(
        "xnnpack_threads={}",
        env::var("EXECUTORCH_XNNPACK_THREADS").unwrap_or_else(|_| "auto".into())
    );
    let load_start = Instant::now();
    let mut synthesizer = match Synthesizer::new(&args[1], &args[2]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("model load failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("load_ms={:.3}", load_start.elapsed().as_secs_f64() * 1000.0);

    for run in 0..=runs {
        let start = Instant::now();
        let mut samples = 0usize;
        for text in texts {
            match synthesizer.synthesize(text, &options) {
                Ok(audio) => samples += audio.len(),
                Err(error) => {
                    eprintln!("synthesis failed in run {run}: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        let label = if run == 0 {
            "cold".to_string()
        } else {
            format!("warm{run}")
        };
        println!(
            "{label}_ms={:.3} samples={samples} audio_s={:.3}",
            start.elapsed().as_secs_f64() * 1000.0,
            samples as f64 / SAMPLE_RATE as f64
        );
    }

    ExitCode::SUCCESS
}
