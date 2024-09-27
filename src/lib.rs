use std::{
    borrow::{BorrowMut, Cow},
    error::Error,
    ops::Deref,
    path::Path,
};

use memmap2::{Mmap, MmapOptions};
use ndarray::{s, Array, Array2, ArrayD};
use ndarray_ndimage::{gaussian_filter, BorderMode};
use tch::{nn::Module, IValue, Kind, Scalar, TchError, Tensor};

pub struct DivvunSpeech<'a> {
    voice: tch::CModule,
    denoiser: tch::CModule,
    vocoder: tch::CModule,
    device: tch::Device,
    text_processor: TextProcessor<'a>,
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub speaker: i32,
    pub pace: f32,
}

pub enum Device {
    Cpu,
}

impl From<Device> for tch::Device {
    fn from(value: Device) -> Self {
        match value {
            Device::Cpu => tch::Device::Cpu,
        }
    }
}

impl<'a> DivvunSpeech<'a> {
    pub unsafe fn from_memory_map(
        voice: &Mmap,
        denoiser: &Mmap,
        vocoder: &Mmap,
        symbol_set: SymbolSet<'a>,
        device: Device,
    ) -> Result<Self, TchError> {
        let device: tch::Device = device.into();

        let mut voice =
            unsafe { tch::CModule::load_ptr_on_device(voice.as_ptr() as _, voice.len(), device) }?;
        let mut denoiser =
            unsafe { tch::CModule::load_ptr_on_device(denoiser.as_ptr() as _, denoiser.len(), device) }?;
        let mut vocoder = unsafe { tch::CModule::load_ptr_on_device(vocoder.as_ptr() as _, vocoder.len(), device) }?;

        Ok(Self {
            voice,
            denoiser,
            vocoder,
            device,
            text_processor: TextProcessor::new(symbol_set),
        })
    }

    pub fn new(
        voice_path: impl AsRef<Path>,
        denoiser_path: impl AsRef<Path>,
        vocoder_path: impl AsRef<Path>,
        symbol_set: SymbolSet<'a>,
        device: Device,
    ) -> Result<Self, TchError> {
        let device: tch::Device = device.into();

        println!("Loading voice");
        let file = std::fs::File::open(voice_path).unwrap();
        let file = unsafe { MmapOptions::new().map(&file).unwrap() };
        let mut voice =
            unsafe { tch::CModule::load_ptr_on_device(file.as_ptr() as _, file.len(), tch::Device::Cpu) }?;
        voice.to(device, Kind::Float, false);
        // let mut voice = tch::CModule::load_on_device(voice_path, device)?;
        // voice.set_eval();

        println!("Loading denoiser");
        let file = std::fs::File::open(denoiser_path).unwrap();
        let file = unsafe { MmapOptions::new().map(&file).unwrap() };
        let mut denoiser =
            unsafe { tch::CModule::load_ptr_on_device(file.as_ptr() as _, file.len(), tch::Device::Cpu) }?;
        // denoiser.set_eval();
        // denoiser.to(device, Kind::Float, false);

        println!("Loading vocoder");
        let file = std::fs::File::open(vocoder_path).unwrap();
        let file = unsafe { MmapOptions::new().map(&file).unwrap() };
        let mut vocoder =
            unsafe { tch::CModule::load_ptr_on_device(file.as_ptr() as _, file.len(), tch::Device::Cpu) }?;
        // vocoder.set_eval();

        Ok(Self {
            voice,
            denoiser,
            vocoder,
            device,
            text_processor: TextProcessor::new(symbol_set),
        })
    }

    fn process_voice(&self, input: Tensor, options: &Options) -> Result<Tensor, TchError> {
        let speaker = Tensor::from_i32(options.speaker);
        let pace = Tensor::from_f32(options.pace);
        let result = self.voice.forward_is(&[
            IValue::Tensor(input),
            IValue::Tensor(speaker),
            IValue::Tensor(pace),
        ])?;
        let voice_data: Tensor = match result {
            IValue::Tuple(mut x) => x.remove(0).try_into().unwrap(),
            _ => unreachable!(),
        };
        Ok(voice_data)
    }

    fn process_vocoder(&self, mel: Tensor) -> Result<Tensor, TchError> {
        let y_g_hat = match self.vocoder.forward_ts(&[mel]) {
            Ok(v) => v.squeeze_dim(1),
            Err(e) => {
                return Err(e);
            }
        };

        Ok(y_g_hat)
    }

    fn process_denoiser(&self, y_g_hat: Tensor) -> Result<Tensor, TchError> {
        let y_g_hat = self.denoiser.forward_ts(&[y_g_hat.to_kind(Kind::Float)])?;

        Ok(y_g_hat)
    }

    fn sharpen(&self, mel: tch::Tensor) -> Result<Tensor, TchError> {
        let mel_np: ArrayD<f32> = (&mel.to_kind(Kind::Float)).try_into().unwrap();
        let mut mel_np: Array2<f32> = mel_np.slice(s![0, .., ..]).to_owned(); // Assuming mel has a batch dimension at the start

        let blurred_f: Array2<f32> = gaussian_filter(&mel_np, 1.0, 0, BorderMode::Reflect, 4);

        let alpha = 0.2;
        mel_np = &mel_np + alpha * (&mel_np - &blurred_f);

        let blurred_f: Array2<f32> = gaussian_filter(&mel_np, 3.0, 0, BorderMode::Reflect, 4);
        let alpha = 0.1;

        let mut sharpened: Array2<f32> = &mel_np + alpha * (&mel_np - &blurred_f);

        // Prepare a separate array for the adjustments
        let mut adjustments = Array2::<f32>::zeros(sharpened.dim());

        for i in 0..80 {
            adjustments.slice_mut(s![i, ..]).assign(
                &((&sharpened.slice(s![i, ..]) + (i as f32 - 40.0) * 0.01)
                    - &sharpened.slice(s![i, ..])),
            );
        }

        // Apply the adjustments to the sharpened array
        sharpened = &sharpened + &adjustments;

        let mut result = Tensor::try_from(sharpened).unwrap();
        result = result.to_kind(Kind::Float).unsqueeze(0);
        // println!("{:?}", &result);
        Ok(result)
    }

    pub fn generate_wav(y_g_hat: Tensor) -> Result<Vec<u8>, TchError> {
        let audio = y_g_hat.squeeze() * 32768.0;
        let audio = audio.to_kind(Kind::Int16);
        let audio = Vec::<i16>::try_from(audio).unwrap();

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 22050,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let out = Vec::with_capacity(audio.len() / 2 + 1);
        let mut out = std::io::Cursor::new(out);

        let mut writer = hound::WavWriter::new(&mut out, spec).unwrap();
        for sample in audio {
            writer.write_sample(sample).unwrap();
        }

        drop(writer);
        Ok(out.into_inner())
    }

    pub fn forward(&self, input: &str, options: Options) -> Result<Tensor, TchError> {
        println!("Forwarding");
        // let result = std::panic::catch_unwind(|| {
        let _guard = tch::no_grad_guard();

        println!("Text proc");
        let input = self.text_processor.encode_text(input);
        println!("Input to device");
        let input = tch::Tensor::from_slice(&input).to(self.device);
        let input = tch::Tensor::stack(&[input], 0).to(self.device);

        println!("Proc voice");
        let mel = self.process_voice(input.to(self.device), &options)?;
        println!("Sharpen");
        let mel = self.sharpen(mel.to_device(tch::Device::Cpu))?;
        println!("vocode");
        let y_g_hat = self.process_vocoder(mel)?;
        // let y_g_hat = mel.squeeze_dim(1);
        // let y_g_hat = self.process_denoiser(y_g_hat)?; //y_g_hat)?;

        Ok(y_g_hat)
        // });

        // match result {
        //     Ok(value) => value,
        //     Err(e) => {
        //         println!("Error: {:?}", e);
        //         Err(TchError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))))
        //     }
        // }
    }
}

use std::collections::HashMap;

// Define your structs and implementations here
struct TextProcessor<'a> {
    symbols: SymbolSet<'a>,
    symbol_to_id: HashMap<String, i64>,
    id_to_symbol: HashMap<i64, String>,
}

impl<'a> TextProcessor<'a> {
    fn new(symbol_set: SymbolSet<'a>) -> Self {
        let symbol_to_id = symbol_set
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), i.try_into().unwrap()))
            .collect::<HashMap<_, _>>();

        let id_to_symbol = symbol_set
            .iter()
            .enumerate()
            .map(|(i, s)| (i.try_into().unwrap(), s.to_string()))
            .collect::<HashMap<_, _>>();

        TextProcessor {
            symbols: symbol_set,
            symbol_to_id,
            id_to_symbol,
        }
    }

    fn text_to_sequence(&self, text: &str) -> Vec<i64> {
        let mut sequence = self.symbols_to_sequence(&text);
        sequence
    }

    fn sequence_to_text(&self, sequence: Vec<i64>) -> String {
        sequence
            .iter()
            .filter_map(|&symbol_id| self.id_to_symbol.get(&symbol_id).map(|x| &**x))
            .collect::<Vec<&str>>()
            .join("")
    }

    fn symbols_to_sequence(&self, symbols: &str) -> Vec<i64> {
        symbols
            .chars()
            .filter_map(|c| self.symbol_to_id.get(&c.to_string()).cloned())
            .collect()
    }

    fn encode_text(&self, text: &str) -> Vec<i64> {
        let mut text = text.to_string();

        let text_encoded = self.text_to_sequence(&text);

        text_encoded
    }
}

#[derive(Debug, Clone)]
pub struct SymbolSet<'a>(Cow<'a, [&'a str]>);

impl<'a> SymbolSet<'a> {
    pub const fn new(symbols: &'a [&'a str]) -> Self {
        Self(Cow::Borrowed(symbols))
    }
}

impl<'a> Deref for SymbolSet<'a> {
    type Target = [&'a str];

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

pub const SMJ_EXPANDED: SymbolSet<'static> = SymbolSet::new(&[
    "!", "'", "\"", ",", ".", ":", ";", "?", "-", " ", "A", "Á", "Æ", "Å", "Ä", "B", "C", "D", "E",
    "F", "G", "H", "I", "J", "K", "L", "M", "N", "Ŋ", "Ń", "Ñ", "O", "Ø", "Ö", "P", "Q", "R", "S",
    "T", "Ŧ", "U", "V", "W", "X", "Y", "Z", "a", "á", "æ", "å", "ä", "b", "c", "d", "e", "f", "g",
    "h", "i", "j", "k", "l", "m", "n", "ŋ", "ń", "ñ", "o", "ø", "ö", "p", "q", "r", "s", "t", "u",
    "v", "w", "x", "y", "z",
]);

pub const SME_EXPANDED: SymbolSet<'static> = SymbolSet::new(&[
    "!", "'", "\"", ",", ".", ":", ";", "?", "-", " ", "A", "Á", "Æ", "Å", "Ä", "B", "C", "Č", "D",
    "Đ", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "Ŋ", "O", "Ø", "Ö", "P", "Q", "R", "S",
    "Š", "T", "Ŧ", "U", "V", "W", "X", "Y", "Z", "Ž", "a", "á", "æ", "å", "ä", "b", "c", "č", "d",
    "đ", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "ŋ", "o", "ø", "ö", "p", "q", "r", "s",
    "š", "t", "ŧ", "u", "v", "w", "x", "y", "z", "ž",
]);
