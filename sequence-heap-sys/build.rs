use std::process::Command;

fn main() {
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let third_party = manifest_dir.join("third_party");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .warnings(false)
        .extra_warnings(false)
        .file("csrc/wrapper.cpp")
        .include("csrc")
        .include(third_party.join("SequenceHeap"));

    // GCC 15 has a conformance change that breaks nested non-template structs used as
    // template arguments. Use clang++ when available to work around this.
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

    build.compile("sequence_heap_wrapper");

    println!("cargo:rerun-if-changed=csrc/wrapper.cpp");
    println!("cargo:rerun-if-changed=csrc/wrapper.hpp");
    println!("cargo:rerun-if-changed=third_party/SequenceHeap/spq");
}
