use std::path::PathBuf;

fn main() {
    let executorch_sysroot = std::env::var("EXECUTORCH_SYSROOT")
        .expect("EXECUTORCH_SYSROOT environment variable must be set");
    let executorch_sysroot = PathBuf::from(executorch_sysroot);

    let executorch_src =
        std::env::var("EXECUTORCH_SRC").expect("EXECUTORCH_SRC environment variable must be set");
    let executorch_src = PathBuf::from(executorch_src);

    // Build wrapper with cmake
    let dst = cmake::Config::new("wrapper")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("CMAKE_OSX_DEPLOYMENT_TARGET", "15.0")
        .env("EXECUTORCH_SYSROOT", &executorch_sysroot)
        .env("EXECUTORCH_SRC", &executorch_src)
        .build();

    // Link paths
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        executorch_sysroot.join("lib").display()
    );

    // Force-load our wrapper and custom_ops to ensure registration symbols aren't stripped
    println!("cargo:rustc-link-lib=static:+whole-archive=tts_wrapper");
    println!("cargo:rustc-link-lib=static:+whole-archive=custom_ops");

    // Force-load executorch libraries to ensure kernel registration
    let executorch_lib = executorch_sysroot.join("lib");
    if let Ok(entries) = std::fs::read_dir(&executorch_lib) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "a" {
                    if let Some(stem) = path.file_stem() {
                        let name = stem.to_string_lossy();
                        // Skip duplicates and problematic libs
                        if name.contains("_static")
                            || name.contains("kernels_util_all_deps")
                            || name.contains("portable_ops_lib")
                            || name.contains("optimized_portable")
                            || name == "liboptimized_ops_lib"
                            || name.contains("quantized_ops_lib")
                        {
                            continue;
                        }
                        let lib_name = name.strip_prefix("lib").unwrap_or(&name);
                        println!("cargo:rustc-link-lib=static:+whole-archive={}", lib_name);
                    }
                }
            }
        }
    }

    // Platform-specific libraries (use CARGO_CFG_TARGET_OS for cross-compilation)
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_os == "macos" || target_os == "ios" {
        // Find clang runtime dir
        let clang_rt_dir = std::process::Command::new("clang")
            .arg("--print-runtime-dir")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .expect("failed to find clang runtime directory");

        let rt_lib = if target_os == "ios" {
            "clang_rt.ios"
        } else {
            "clang_rt.osx"
        };

        // iOS clang_rt is a fat binary; extract thin slice for Rust linker
        if target_os == "ios" {
            let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
            // Map Rust arch names to Apple/lipo arch names
            let apple_arch = match target_arch.as_str() {
                "aarch64" => "arm64",
                "x86_64" => "x86_64",
                other => other,
            };
            let out_dir = std::env::var("OUT_DIR").unwrap();
            let fat_lib = PathBuf::from(&clang_rt_dir).join(format!("lib{}.a", rt_lib));
            let thin_lib = PathBuf::from(&out_dir).join(format!("lib{}.a", rt_lib));
            let status = std::process::Command::new("lipo")
                .args([
                    "-thin",
                    apple_arch,
                    fat_lib.to_str().unwrap(),
                    "-output",
                    thin_lib.to_str().unwrap(),
                ])
                .status()
                .expect("failed to run lipo");
            assert!(
                status.success(),
                "lipo failed to extract {} slice",
                apple_arch
            );
            println!("cargo:rustc-link-search=native={}", out_dir);
        } else {
            println!("cargo:rustc-link-search=native={}", clang_rt_dir);
        }
        println!("cargo:rustc-link-lib=static={}", rt_lib);

        if target_os == "macos" {
            println!("cargo:rustc-link-arg=-mmacosx-version-min=15.0");
        }

        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=sqlite3");
        println!("cargo:rustc-link-lib=c++");
    } else if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Rerun if wrapper source changes
    println!("cargo:rerun-if-changed=wrapper/");
    println!("cargo:rerun-if-env-changed=EXECUTORCH_SYSROOT");
    println!("cargo:rerun-if-env-changed=EXECUTORCH_SRC");
}
