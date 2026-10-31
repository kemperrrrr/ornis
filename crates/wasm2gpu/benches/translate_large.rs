//! Wall-time probes for translation of large guest modules.
//!
//! This benchmark prints the input size, output size, median, and quartiles.
//! It intentionally has no checked-in results: compare the same command on
//! the base and candidate revisions on one machine before drawing conclusions.
//!
//! ```text
//! cargo bench -p ornis-wasm2gpu --bench translate_large
//! ```

use std::hint::black_box;
use std::time::Instant;

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;

/// Untimed runs to warm up parsing, lowering, validation, and SPIR-V output.
const WARMUPS: usize = 3;
/// Number of wall-time samples per workload.
const SAMPLES: usize = 21;
/// Straight-line operations in the single-entry large-module workload.
const LINEAR_OPS: usize = 12_000;
/// Repeated statements in the specialized callee.
const SPECIALIZED_BODY_OPS: usize = 1_024;
/// Distinct constant-argument specializations (the translator's current cap).
const SPECIALIZATIONS: usize = 64;

fn main() {
    let workloads = [
        ("large_straight_line", large_straight_line(LINEAR_OPS)),
        (
            "many_specializations",
            many_specializations(SPECIALIZATIONS, SPECIALIZED_BODY_OPS),
        ),
    ];
    for (name, wasm) in workloads {
        report(name, &wasm);
    }
}

fn report(name: &str, wasm: &[u8]) {
    for _ in 0..WARMUPS {
        let translated = translate(wasm).expect("translate warm-up workload");
        black_box((
            translated.wgsl.len(),
            translated.vertex_spirv.len(),
            translated.fragment_spirv.len(),
        ));
    }

    let output_bytes = translate(wasm)
        .expect("translate output-size probe")
        .wgsl
        .len();
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let started = Instant::now();
        let translated = translate(black_box(wasm)).expect("translate benchmark workload");
        samples.push(started.elapsed());
        black_box((
            translated.wgsl.len(),
            translated.vertex_spirv.len(),
            translated.fragment_spirv.len(),
        ));
    }
    samples.sort_unstable();
    let median = samples[SAMPLES / 2];
    let q1 = samples[SAMPLES / 4];
    let q3 = samples[SAMPLES * 3 / 4];
    println!(
        "{name}: input={} B, WGSL={output_bytes} B, median={median:?}, \
         Q1={q1:?}, Q3={q3:?}, n={SAMPLES}",
        wasm.len(),
    );
}

fn large_straight_line(operations: usize) -> Vec<u8> {
    let mut body = String::new();
    for _ in 0..operations {
        body.push_str("f32.const 0 drop ");
    }
    body.push_str("f32.const 0 f32.const 0 f32.const 0 f32.const 1");
    module_with("", &body)
}

fn many_specializations(specializations: usize, body_operations: usize) -> Vec<u8> {
    let mut helper = String::from("(func $large (param i32) (result i32) ");
    for _ in 0..body_operations {
        helper.push_str("local.get 0 drop ");
    }
    helper.push_str("local.get 0)");

    let mut body = String::new();
    for index in 0..specializations {
        body.push_str(&format!("i32.const {index} call $large drop "));
    }
    body.push_str("f32.const 0 f32.const 0 f32.const 0 f32.const 1");
    module_with(&helper, &body)
}

fn module_with(helper: &str, fragment_body: &str) -> Vec<u8> {
    let text = format!(
        r#"(module
            {helper}
            (func (export "{vs}")
                (param f32 f32 f32 f32)
                (result f32 f32 f32 f32 f32 f32)
                local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)
            (func (export "{fs}")
                (param f32 f32)
                (result f32 f32 f32 f32)
                {fragment_body}))"#,
        vs = symbols::VS_MAIN,
        fs = symbols::FS_MAIN,
    );
    wat::parse_str(text).expect("benchmark WAT")
}
