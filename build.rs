use std::path::PathBuf;

fn main() {
    let executorch_sysroot = std::env::var("EXECUTORCH_SYSROOT")
        .expect("EXECUTORCH_SYSROOT environment variable must be set");
    let executorch_sysroot = PathBuf::from(executorch_sysroot);

    // Build wrapper with cmake
    let dst = cmake::Config::new("wrapper")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("CMAKE_OSX_DEPLOYMENT_TARGET", "15.0")
        .env("EXECUTORCH_SYSROOT", &executorch_sysroot)
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

    // Platform-specific libraries
    #[cfg(target_os = "macos")]
    {
        // Find clang runtime for ___isPlatformVersionAtLeast
        let clang_rt_dir = std::process::Command::new("clang")
            .arg("--print-runtime-dir")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string());
        if let Some(dir) = clang_rt_dir {
            println!("cargo:rustc-link-search=native={}", dir);
            println!("cargo:rustc-link-lib=static=clang_rt.osx");
        }

        println!("cargo:rustc-link-arg=-mmacosx-version-min=15.0");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=sqlite3");
        println!("cargo:rustc-link-lib=c++");
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Rerun if wrapper source changes
    println!("cargo:rerun-if-changed=wrapper/");
    println!("cargo:rerun-if-env-changed=EXECUTORCH_SYSROOT");
}
