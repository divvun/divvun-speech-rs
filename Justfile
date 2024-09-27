build:
    LIBTORCH=/opt/libtorch LIBTORCH_BYPASS_VERSION_CHECK=1 cargo build --release
    install_name_tool -add_rpath /opt/libtorch/lib target/release/divvun-speech
    # cp /opt/libtorch/lib/libtorch_cpu.dylib target/release

build-ios:
    LIBTORCH_LIB=/Users/brendan/git/divvun/divvun-speech-rs/contrib/build/ios/pytorch/LibTorchLite.xcframework/ios-arm64 \
        LIBTORCH_INCLUDE=/Users/brendan/git/divvun/divvun-speech-rs/contrib/build/ios/pytorch/LibTorchLite.xcframework/ios-arm64/Headers \
        LIBTORCH=/Users/brendan/git/divvun/divvun-speech-rs/contrib/build/ios/pytorch/LibTorchLite.xcframework/ios-arm64 \
        LIBTORCH_STATIC=1 LIBTORCH_STATIC=1 cargo build --bins --release --target=aarch64-apple-ios
build-ios-sim:
    LIBTORCH_LIB=/Users/brendan/git/divvun/divvun-speech-rs/contrib/build/ios/pytorch/LibTorchLite.xcframework/ios-arm64_x86_64-simulator \
        LIBTORCH_INCLUDE=/Users/brendan/git/divvun/divvun-speech-rs/contrib/build/ios/pytorch/LibTorchLite.xcframework/ios-arm64_x86_64-simulator/Headers \
        LIBTORCH=/Users/brendan/git/divvun/divvun-speech-rs/contrib/build/ios/pytorch/LibTorchLite.xcframework/ios-arm64_x86_64-simulator \
        LIBTORCH_LITE=1 LIBTORCH_STATIC=1 cargo build --bins --release --target=aarch64-apple-ios-sim

build-lib-ios:
    LIBTORCH_LIB=/Users/brendan/git/divvun/divvun-speech-rs/contrib/build/ios/pytorch/LibTorchLite.xcframework/ios-arm64 \
        LIBTORCH_INCLUDE=/Users/brendan/git/divvun/divvun-speech-rs/contrib/build/ios/pytorch/LibTorchLite.xcframework/ios-arm64/Headers \
        LIBTORCH=/Users/brendan/git/divvun/divvun-speech-rs/contrib/build/ios/pytorch/LibTorchLite.xcframework/ios-arm64 \
        LIBTORCH_LITE=1 LIBTORCH_STATIC=1 cargo build --release --lib --target=aarch64-apple-ios