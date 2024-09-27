fn main() {
    let target = std::env::var("TARGET").expect("TARGET variable not set");
    if target.contains("ios") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
