//! Churn test: create + synthesize + drop the Synthesizer repeatedly in one
//! process, proving Engine's Drop reclaim (modules then leaked tensors) is
//! sound — no crash, no corruption, stable output across generations.
//!
//! Usage: cargo run --release --example engine_churn <voice.pte> <vocoder.pte>

use divvun_speech::{Options, Synthesizer};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <voice.pte> <vocoder.pte>", args[0]);
        std::process::exit(1);
    }

    let opts = Options::new();
    let mut lens = Vec::new();
    for round in 0..3 {
        let mut synth = Synthesizer::new(&args[1], &args[2])?;
        let audio = synth.synthesize("Buorre beaivi", &opts)?;
        let nans = audio.iter().filter(|x| x.is_nan()).count();
        println!("round {round}: {} samples, {} NaNs", audio.len(), nans);
        assert_eq!(nans, 0, "NaNs in round {round}");
        lens.push(audio.len());
        drop(synth); // explicit: exercise Engine::drop before the next create
    }
    assert!(lens.windows(2).all(|w| w[0] == w[1]), "output length varied across rounds: {lens:?}");
    println!("CHURN_OK: 3 create/synthesize/drop cycles, stable output");
    Ok(())
}
