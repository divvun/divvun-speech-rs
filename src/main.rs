use std::{fmt::Display, sync::Arc};

use divvun_speech::{Device, DivvunSpeech, Options, ALL_SAMI, SME_EXPANDED};
use tch::TchError;

fn run() -> Result<(), TchError> {
    // let voice_data = ds.forward(" davvisámegiella gullá sámegielaid oarjesámegielaid davvejovkui ovttas julev- ja bihtánsámegielain. ", Default::default())?;
    let s = singlethread()?;
    // let m = multithread()?;

    // println!("{s:?} {m:?}");
    // println!("{s:?}");
    Ok(())
}

fn singlethread() -> Result<std::time::Duration, TchError> {
    let ds = DivvunSpeech::new(
        "/Users/brendan/git/divvun/divvun-speech-py/voice-jit.ptl",
        // "/Users/brendan/Downloads/torchscript_sme_f.pt",
        // "/Users/brendan/git/divvun/divvun-speech-py/denoiser-jit.ptl",
        "/Users/brendan/git/divvun/divvun-speech-py/vocoder-jit.ptl",
        ALL_SAMI,
        Device::Cpu,
    )
    .unwrap();

    let start = std::time::Instant::now();
    let text = "Sami sami sami sami sami";

    for (lang, speakers) in SPEAKERS {
        for speaker in *speakers {
            println!("{} {}", lang, speaker);
            let voice_data = ds.forward(text, Options { speaker: *speaker as i32, language: *lang as i32, pace: 1.0 })?;
            let wav_data = DivvunSpeech::generate_wav(voice_data.copy())?;
            std::fs::write(format!("./output/{lang}-{speaker}.wav"), wav_data).unwrap();
        }
    }
// ;
//     let mut wav_data = vec![];
//     for _ in 0..100 {
//         let voice_data = match ds.forward(text, Options { speaker, language, pace: 1.05 }) {
//             Ok(v) => v,
//             Err(e) => {
//                 eprintln!("Error: {}", e);
//                 return Err(e);
//             }
//         };
//         wav_data = DivvunSpeech::generate_wav(voice_data.copy()).unwrap();
//     }

    let end = std::time::Instant::now();

    let diff = end - start;
    // println!("Total: {:?}", diff);
    // println!("Per: {:?} ms", (diff.as_millis() as f64 / 1000.0));
    // std::fs::write("./output.wav", wav_data).unwrap();
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


// languages = {"South Sámi":0,
//           "North Sámi":1,
//           "Lule Sámi":2}

// speakers={#"aj0": 0,
//           "Aanna - sma": 1,
//           "acapela-male - sme": 2,
//           "Siggá - smj": 3,
//           "Biret - sme": 5,
//           #"lo": 6,
//           "ms - sme": 7,
//           "Abmut - smj": 8,
//           "Nihkol - smj": 9
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
enum Lang {
    SMA = 0,
    SME,
    SMJ,
}

impl Display for Lang {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
enum Speaker {
    Aanna = 1,
    AcapelaMale = 2,
    Siggá = 3,
    Biret = 5,
    Ms = 7,
    Abmut = 8,
    Nihkol = 9,
}

impl Display for Speaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

const SPEAKERS: &[(Lang, &[Speaker])] = &[
    (Lang::SMA, &[Speaker::Aanna]),
    (Lang::SME, &[Speaker::AcapelaMale, Speaker::Biret, Speaker::Ms]),
    (Lang::SMJ, &[Speaker::Abmut, Speaker::Nihkol, Speaker::Siggá]),
];