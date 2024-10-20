fn main() {
    let target = std::env::var("TARGET").expect("TARGET variable not set");
    if let Ok(path) = std::env::var("LIBTORCH") {
        println!("cargo:rustc-link-search=native={}/lib", path);
    }

    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
    if target.contains("ios") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
