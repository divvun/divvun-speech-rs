use divvun_speech::{Options, SAMPLE_RATE, Synthesizer, WordTiming};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let mut pace: f32 = 1.0;
    let mut speaker: i64 = 1;
    let mut language: i64 = 1;
    let mut positional: Vec<&str> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pace" => {
                i += 1;
                pace = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            }
            "--speaker" => {
                i += 1;
                speaker = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            "--language" => {
                i += 1;
                language = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            arg => positional.push(arg),
        }
        i += 1;
    }

    if positional.len() < 4 {
        eprintln!(
            "Usage: {} <voice.pte> <vocoder.pte> <text> <out_dir> [--pace <f32>] [--speaker <i64>] [--language <i64>]",
            args[0]
        );
        std::process::exit(1);
    }

    let voice_path = PathBuf::from(positional[0]);
    let vocoder_path = PathBuf::from(positional[1]);
    let text = positional[2];
    let out_dir = PathBuf::from(positional[3]);
    fs::create_dir_all(&out_dir)?;

    println!("Loading models...");
    let mut synth = Synthesizer::new(&voice_path, &vocoder_path)?;

    let options = Options::new()
        .with_pace(pace)
        .with_speaker(speaker)
        .with_language(language);

    println!("Synthesizing: {:?}", text);
    let (audio, timings) = synth.synthesize_with_word_timings(text, &options)?;
    println!(
        "Generated {} samples ({:.2}s @ {} Hz), {} word timings",
        audio.len(),
        audio.len() as f32 / SAMPLE_RATE as f32,
        SAMPLE_RATE,
        timings.len(),
    );

    let combined_path = out_dir.join("combined.wav");
    write_wav(&combined_path, &audio)?;
    println!("Wrote {}", combined_path.display());

    println!();
    println!("Per-word timings:");
    println!("  {:>3} {:30}  start..end (s)         dur (s)", "#", "word");
    for (i, t) in timings.iter().enumerate() {
        let start_s = t.start_sample as f32 / SAMPLE_RATE as f32;
        let end_s = t.end_sample as f32 / SAMPLE_RATE as f32;
        println!(
            "  {:>3} {:30}  {:>7.3}..{:>7.3}        {:>6.3}",
            i,
            format!("{:?}", t.word),
            start_s,
            end_s,
            end_s - start_s,
        );
        let chunk = &audio[t.start_sample..t.end_sample];
        let path = out_dir.join(per_word_filename(i, t));
        write_wav(&path, chunk)?;
    }
    println!();
    println!("Wrote {} per-word files to {}", timings.len(), out_dir.display());

    Ok(())
}

fn per_word_filename(i: usize, t: &WordTiming) -> String {
    let safe: String = t
        .word
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    format!("word_{:02}_{}.wav", i, safe)
}

fn write_wav(path: &Path, samples: &[f32]) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);

    let sample_rate: u32 = SAMPLE_RATE;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 32;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = (samples.len() * 4) as u32;
    let file_size = 36 + data_size;

    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&3u16.to_le_bytes())?; // IEEE float
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;

    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for s in samples {
        file.write_all(&s.to_le_bytes())?;
    }

    Ok(())
}
