fn main() {
    if let Ok(path) = std::env::var("LIBTORCH") {
        println!("cargo:rustc-link-search=native={}/lib", path);
    }

    println!("cargo:rustc-link-lib=pthreadpool");
    println!("cargo:rustc-link-lib=cpuinfo");
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=omp");
}
