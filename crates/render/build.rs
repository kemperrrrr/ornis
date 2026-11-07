//! Build-time WGSL for the llvm2gpu cull kernel.
//!
//! Compiles the `ornis-gpu-shaders` source for `wasm32-unknown-unknown`
//! with the pinned rustc, translates the LLVM IR through
//! `ornis-llvm2gpu`, and writes `cull.wgsl` into `OUT_DIR`. The culler
//! includes that file, so the shipped kernel and the Rust source cannot
//! drift: a source change re-runs this script and rebuilds the renderer.

use std::path::{Path, PathBuf};
use std::process::Command;

/// `ornis_gpu.types` tags of `CullParams`: six `vec4` planes + `u32` count.
const CULL_PARAMS_TAGS: [u8; 7] = [6, 6, 6, 6, 6, 6, 2];
/// One invocation per source record.
const WORKGROUP: [u32; 3] = [64, 1, 1];

fn main() {
    println!("cargo:rerun-if-changed=../gpu-shaders/src/lib.rs");
    println!("cargo:rerun-if-changed=build.rs");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../gpu-shaders/src/lib.rs");
    let ll = out.join("cull.ll");
    compile(&source, &ll);
    let ir = std::fs::read_to_string(&ll).expect("read cull llvm ir");
    let translated =
        ornis_llvm2gpu::translate_with_types_and_workgroup(&ir, &[&CULL_PARAMS_TAGS], WORKGROUP)
            .unwrap_or_else(|error| panic!("translate cull kernel: {error}"));
    std::fs::write(out.join("cull.wgsl"), translated.wgsl).expect("write cull wgsl");
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
        .expect("spawn rustc for the cull shader");
    assert!(status.success(), "rustc failed for {}", source.display());
}
