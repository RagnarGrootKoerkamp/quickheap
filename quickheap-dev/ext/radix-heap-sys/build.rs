use std::process::Command;

fn main() {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .flag("-march=native")
        .flag("-flto")
        .warnings(false)
        .extra_warnings(false)
        .file("csrc/wrapper.cpp")
        .include("csrc")
        .include("third_party/radix-heap");

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

    build.compile("radix_heap_wrapper");

    println!("cargo:rerun-if-changed=csrc/wrapper.cpp");
    println!("cargo:rerun-if-changed=csrc/wrapper.hpp");
    println!("cargo:rerun-if-changed=third_party/radix-heap/radix_heap.h");
}
