use std::process::Command;

fn main() {
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let third_party = manifest_dir.join("third_party");

    let s3q_dir = third_party.join("s3q");
    let ips4o_dir = third_party.join("ips4o");
    let range_v3_dir = third_party.join("range-v3");
    let xoshiro_dir = third_party.join("xoshiro");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .extra_warnings(false)
        .file("csrc/wrapper.cpp")
        .include("csrc")
        .include(s3q_dir.join("include"))
        .include(ips4o_dir.join("include"))
        .include(range_v3_dir.join("include"))
        .include(&xoshiro_dir);

    // GCC 15 has a conformance change that breaks lookup of `static constexpr`
    // members in nested non-template structs used as template arguments
    // (see raphinesse/s3q#<issue>). Use clang++ when available to work around this.
    // Users can override by setting the CXX environment variable.
    if std::env::var("CXX").is_err() {
        let clang_available = Command::new("clang++")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if clang_available {
            build.compiler("clang++");
        }
    }

    build.compile("s3q_wrapper");

    println!("cargo:rerun-if-changed=csrc/wrapper.cpp");
    println!("cargo:rerun-if-changed=csrc/wrapper.hpp");
    println!("cargo:rerun-if-changed=third_party/s3q/include");
    println!("cargo:rerun-if-changed=third_party/ips4o/include");
    println!("cargo:rerun-if-changed=third_party/range-v3/include");
    println!("cargo:rerun-if-changed=third_party/xoshiro");
}
