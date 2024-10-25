use std::sync::Arc;

use divvun_speech::{Device, DivvunSpeech, SME_EXPANDED};
use tch::TchError;

fn run() -> Result<(), TchError> {
    // let voice_data = ds.forward(" davvisámegiella gullá sámegielaid oarjesámegielaid davvejovkui ovttas julev- ja bihtánsámegielain. ", Default::default())?;
    let s = singlethread()?;
    // let m = multithread()?;

    // println!("{s:?} {m:?}");
    println!("{s:?}");
    Ok(())
}

fn singlethread() -> Result<std::time::Duration, TchError> {
    let ds = DivvunSpeech::new(
        "/Users/brendan/git/divvun/divvun-speech-py/voice-jit.ptl",
        // "/Users/brendan/Downloads/torchscript_sme_f.pt",
        // "/Users/brendan/git/divvun/divvun-speech-py/denoiser-jit.ptl",
        "/Users/brendan/git/divvun/divvun-speech-py/vocoder-jit.ptl",
        SME_EXPANDED,
        Device::Cpu,
    )
    .unwrap();

    let start = std::time::Instant::now();
    let text = "This is an example string";

    for _ in 0..100 {
        let voice_data = match ds.forward(text, Default::default()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error: {}", e);
                return Err(e);
            }
        };
        let _wav_data = DivvunSpeech::generate_wav(voice_data.copy()).unwrap();
    }

    let end = std::time::Instant::now();

    let diff = end - start;
    println!("{:?}", diff);
    Ok(diff)
}

fn multithread() -> Result<std::time::Duration, TchError> {
    let mut threads = vec![];

    let units = (0..8)
        .map(|_| {
            Arc::new(
                DivvunSpeech::new(
                    "/Users/brendan/git/divvun/divvun-speech-rs/voice-jit.ptl",
                    // "/Users/brendan/Downloads/torchscript_sme_f.pt",
                    // "/Users/brendan/git/divvun/divvun-speech-rs/denoiser-jit.ptl",
                    "/Users/brendan/git/divvun/divvun-speech-rs/vocoder-jit.ptl",
                    SME_EXPANDED,
                    Device::Cpu,
                )
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();

    let start = std::time::Instant::now();

    for (n, unit) in units.into_iter().enumerate() {
        let handle = std::thread::spawn(move || {
            let ds = unit;
            for _ in 0..(if n % 2 == 1 { 12 } else { 13 }) {
                let voice_data = ds.forward(" davvisámegiella gullá sámegielaid oarjesámegielaid davvejovkui ovttas julev- ja bihtánsámegielain. ", Default::default()).unwrap();
                let _wav_data = DivvunSpeech::generate_wav(voice_data.copy()).unwrap();
            }
            println!("{:?}", std::thread::current().id());
        });
        threads.push(handle);
    }

    for h in threads {
        h.join().unwrap();
    }

    let end = std::time::Instant::now();

    let diff = end - start;
    println!("{:?}", diff);
    // std::fs::write("/tmp/output.wav", wav_data).unwrap();

    Ok(diff)
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(e) => eprintln!("Error: {}", e),
    }
}
