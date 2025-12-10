use divvun_speech::{Options, SAMPLE_RATE, Synthesizer};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!(
            "Usage: {} <voice.pte> <vocoder.pte> <text> [output.wav]",
            args[0]
        );
        std::process::exit(1);
    }

    let voice_path = PathBuf::from(&args[1]);
    let vocoder_path = PathBuf::from(&args[2]);
    let text = &args[3];
    let output_path = args
        .get(4)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("output.wav"));

    println!("Loading models...");
    let synth = Synthesizer::new(&voice_path, &vocoder_path)?;
    println!(
        "Alphabet: {} symbols",
        synth.text_processor().symbols().len()
    );

    println!("Synthesizing: \"{}\"", text);
    let options = Options::new().with_pace(1.05);
    let audio = synth.synthesize(text, &options)?;

    println!(
        "Generated {} samples ({:.2}s)",
        audio.len(),
        audio.len() as f32 / SAMPLE_RATE as f32
    );

    // Write WAV file
    write_wav(&output_path, &audio)?;
    println!("Wrote {}", output_path.display());

    Ok(())
}

fn write_wav(path: &std::path::Path, samples: &[f32]) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);

    let sample_rate: u32 = SAMPLE_RATE;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 32;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = (samples.len() * 4) as u32;
    let file_size = 36 + data_size;

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // chunk size
    file.write_all(&3u16.to_le_bytes())?; // format = IEEE float
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for sample in samples {
        file.write_all(&sample.to_le_bytes())?;
    }

    Ok(())
}
