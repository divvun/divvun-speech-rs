fn main() {
    let target = std::env::var("TARGET").expect("TARGET variable not set");
    if let Ok(path) = std::env::var("LIBTORCH") {
        println!("cargo:rustc-link-search=native={}/lib", path);
    }
}
