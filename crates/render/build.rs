//! Build-time WGSL for the llvm2gpu cull kernels.
//!
//! Compiles the `ornis-gpu-shaders` sources for
//! `wasm32-unknown-unknown` with the pinned rustc, translates their LLVM
//! IR through `ornis-llvm2gpu`, and writes the WGSL into `OUT_DIR`. The
//! culler includes that file, so the shipped kernel and the Rust source
//! cannot drift: a source change re-runs this script and rebuilds the
//! renderer.

use std::path::{Path, PathBuf};
use std::process::Command;

/// `ornis_gpu.types` tags of `CullParams`: six `vec4` planes + `u32` count.
const CULL_PARAMS_TAGS: [u8; 7] = [6, 6, 6, 6, 6, 6, 2];
/// One invocation per source record.
const WORKGROUP: [u32; 3] = [64, 1, 1];

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../gpu-shaders");
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-changed=build.rs");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    build(
        &root.join("src/lib.rs"),
        &out.join("cull.ll"),
        &out.join("cull.wgsl"),
    );
}

fn build(source: &Path, ll: &Path, wgsl: &Path) {
    compile(source, ll);
    let ir = std::fs::read_to_string(ll).expect("read cull llvm ir");
    let translated =
        ornis_llvm2gpu::translate_with_types_and_workgroup(&ir, &[&CULL_PARAMS_TAGS], WORKGROUP)
            .unwrap_or_else(|error| panic!("translate {}: {error}", source.display()));
    std::fs::write(wgsl, translated.wgsl).expect("write cull wgsl");
}

fn compile(source: &Path, ll: &Path) {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let status = Command::new(rustc)
        .args([
            "--edition=2024",
            "--crate-type=lib",
            "--target",
            "wasm32-unknown-unknown",
            "-C",
            "opt-level=1",
            "-C",
            "panic=abort",
            "-C",
            "target-cpu=mvp",
            "--emit=llvm-ir",
        ])
        .arg(source)
        .arg("-o")
        .arg(ll)
        .status()
        .expect("spawn rustc for a cull shader");
    assert!(status.success(), "rustc failed for {}", source.display());
}
