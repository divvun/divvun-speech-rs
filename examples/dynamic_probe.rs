//! Probe whether the voice .pte accepts dynamic (smaller-than-max) token
//! inputs, and whether that's faster than padding to the max — i.e. whether
//! the bounded-dynamic export (`Dim("seq", 2..512)`) actually works at runtime.
//!
//! Usage: cargo run --release --example dynamic_probe <voice.pte>

use std::env;
use std::time::Instant;

use executorch::extension::module::module::{LoadMode, Module};
use executorch::extension::tensor::tensor_ptr::make_tensor_ptr_from_vec;
use executorch::runtime::core::error::Error as EtError;
use executorch::runtime::core::evalue::EValue;
use executorch::runtime::core::portable_type::scalar_type::ScalarType;
use executorch::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
use executorch::runtime::executor::program::Verification;

fn register() {
    let err = executorch::custom_ops::register_custom_ops();
    assert_eq!(err, EtError::Ok, "custom ops");
    let err = executorch::backends::xnnpack::register();
    assert_eq!(err, EtError::Ok, "xnnpack");
    let err = executorch::kernels::optimized::register();
    assert_eq!(err, EtError::Ok, "kernels");
}

fn run_case(
    module: &mut Module<'static>,
    label: &str,
    tokens: &[i64],
    seq_len: usize,
    iters: usize,
) -> Option<(f64, Vec<f32>, i64)> {
    // Build a [1, seq_len] token tensor with the sequence at the front.
    let mut data = vec![0i64; seq_len];
    data[..tokens.len()].copy_from_slice(tokens);
    let tok = make_tensor_ptr_from_vec(
        vec![1, seq_len as i32],
        data,
        Vec::new(),
        Vec::new(),
        ScalarType::Long,
        TensorShapeDynamism::STATIC,
    );
    let speaker = make_tensor_ptr_from_vec(
        vec![1],
        vec![1i64],
        Vec::new(),
        Vec::new(),
        ScalarType::Long,
        TensorShapeDynamism::STATIC,
    );
    let language = make_tensor_ptr_from_vec(
        vec![1],
        vec![1i64],
        Vec::new(),
        Vec::new(),
        ScalarType::Long,
        TensorShapeDynamism::STATIC,
    );
    let pace = make_tensor_ptr_from_vec(
        vec![1],
        vec![1.0f32],
        Vec::new(),
        Vec::new(),
        ScalarType::Float,
        TensorShapeDynamism::STATIC,
    );

    // Leak so the EValues satisfy the module's 'static input lifetime.
    let tok = Box::leak(Box::new(tok));
    let speaker = Box::leak(Box::new(speaker));
    let language = Box::leak(Box::new(language));
    let pace = Box::leak(Box::new(pace));

    let inputs = vec![
        EValue::from_tensor(tok.tensor()),
        EValue::from_tensor(speaker.tensor()),
        EValue::from_tensor(language.tensor()),
        EValue::from_tensor(pace.tensor()),
    ];

    // Warmup + check it runs at all.
    let outputs = match module.execute("forward", &inputs) {
        Ok(o) => o,
        Err(e) => {
            println!("{label}: EXECUTE FAILED: {e:?}");
            return None;
        }
    };

    let mel = outputs[0].to_tensor();
    let mel_shape: Vec<i64> = (0..mel.dim()).map(|d| mel.size(d) as i64).collect();
    let mel_lens = if outputs.len() > 1 && outputs[1].is_tensor() {
        unsafe { *outputs[1].to_tensor().const_data_ptr::<i64>() }
    } else {
        -1
    };
    // Fingerprint the first actual frames of the mel (channel-major [1, 80, T]).
    let t = mel_shape[2] as usize;
    let take = (mel_lens.max(0) as usize).min(t);
    let mel_ptr = mel.const_data_ptr::<f32>();
    let mut fp = Vec::with_capacity(80);
    for c in 0..80usize {
        let mut acc = 0f32;
        for i in 0..take {
            acc += unsafe { *mel_ptr.add(c * t + i) };
        }
        fp.push(acc);
    }
    let durs: Vec<f32> = if outputs.len() > 2 && outputs[2].is_tensor() {
        let d = outputs[2].to_tensor();
        let n = tokens.len().min(d.numel() as usize);
        let p = d.const_data_ptr::<f32>();
        (0..n).map(|i| unsafe { *p.add(i) }).collect()
    } else {
        Vec::new()
    };

    // Timed runs.
    let start = Instant::now();
    for _ in 0..iters {
        module.execute("forward", &inputs).expect("timed run");
    }
    let per_run = start.elapsed().as_secs_f64() / iters as f64;

    println!(
        "{label}: OK  mel_shape={mel_shape:?}  mel_lens={mel_lens}  {:.1} ms/run  dur[0..4]={:?}",
        per_run * 1e3,
        &durs[..durs.len().min(4)],
    );
    Some((per_run, fp, mel_lens))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <voice.pte>", args[0]);
        std::process::exit(1);
    }
    register();

    let mut module = Module::from_file_path(&args[1], LoadMode::Mmap, None, None, None, false);
    let err = module.load(Verification::Minimal);
    assert_eq!(err, EtError::Ok, "load");

    // Arbitrary valid token ids (< alphabet size 95), wrapped in [9] like the
    // export does. Exact ids don't matter for the probe.
    let tokens: Vec<i64> = {
        let mut t = vec![9i64];
        t.extend([20, 30, 40, 50, 45, 12, 9, 21, 31, 41, 33, 22].iter());
        t.push(9);
        t
    };
    let n = tokens.len();
    println!("actual token count: {n}");

    let iters = 5;
    let full = run_case(&mut module, "padded [1,512]", &tokens, 512, iters);
    let small = run_case(&mut module, &format!("dynamic [1,{n}] "), &tokens, n, iters);
    let mid = run_case(&mut module, "dynamic [1,128] ", &tokens, 128, iters);

    if let (Some((tf, fpf, lf)), Some((ts, fps, ls))) = (&full, &small) {
        let speedup = tf / ts;
        let agree = lf == ls
            && fpf
                .iter()
                .zip(fps)
                .all(|(a, b)| (a - b).abs() <= 1e-2 * a.abs().max(1.0));
        println!("\npadded {:.1} ms vs dynamic {:.1} ms  → {speedup:.2}x", tf * 1e3, ts * 1e3);
        println!("outputs agree (mel_lens + mel fingerprint): {agree}");
    }
    if let Some((tm, _, lm)) = &mid {
        println!("[1,128] per-run {:.1} ms, mel_lens={lm}", tm * 1e3);
    }
    std::process::exit(0);
}
