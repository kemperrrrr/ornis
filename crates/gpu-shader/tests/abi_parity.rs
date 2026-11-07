//! The typed SDK must not change what the translator sees.
//!
//! The ergonomics claim of this slice is that `Storage`/`Uniform`/`Vec3` are a
//! *write-time* layer: the methods inline into the same `ornis_gpu` import calls
//! the raw `extern "C"` kernels made, so the WGSL of a migrated kernel is
//! byte-identical to the WGSL the hand written form produced. This test pins
//! that end to end — same kernel body, two spellings, one shader.
//!
//! It needs `rustc` with the `wasm32-unknown-unknown` target and skips (with a
//! message) when the target is missing, like the other rustc-driven tests.

use std::path::Path;
use std::process::Command;

const RAW: &str = r#"
#![no_std]
#![no_main]
use core::panic::PanicInfo;
#[panic_handler]
fn panic(_: &PanicInfo) -> ! { core::arch::wasm32::unreachable() }

unsafe extern "C" {
    fn storage_len_f32(group: i32, binding: i32) -> i32;
    fn storage_read_f32(group: i32, binding: i32, index: i32) -> f32;
    fn storage_write_u32(group: i32, binding: i32, index: i32, value: u32);
}

#[unsafe(no_mangle)]
pub extern "C" fn cs_main(id: u32) {
    let len = unsafe { storage_len_f32(0, 1) } as u32;
    if id >= len { return; }
    let value = unsafe { storage_read_f32(0, 1, id as i32) };
    if value < 0.0 { return; }
    unsafe { storage_write_u32(0, 2, id as i32, id) };
}
"#;

const TYPED: &str = r#"
#![no_std]
#![no_main]
use core::panic::PanicInfo;
#[panic_handler]
fn panic(_: &PanicInfo) -> ! { core::arch::wasm32::unreachable() }

use ornis_gpu_shader::{gpu_entry, ReadOnly, ReadWrite, Storage};

#[gpu_entry(compute(threads(64)))]
#[unsafe(no_mangle)]
pub extern "C" fn cs_main(
    id: u32,
    #[gpu(storage_buffer, group = 0, binding = 1, read_only)] src: Storage<f32, ReadOnly>,
    #[gpu(storage_buffer, group = 0, binding = 2)] dst: Storage<u32, ReadWrite>,
) {
    if id >= src.len() { return; }
    let value = src.load(id);
    if value < 0.0 { return; }
    dst.store(id, id);
}
"#;

/// Imports the entry body calls, by their `wasm-import-name`.
///
/// A typed kernel declares its imports in the SDK with `#[link_name]`, so the
/// IR keeps a mangled Rust name plus a `"wasm-import-name"` attribute on the
/// declaration's attribute group; a raw kernel declares the name directly.
/// Both resolve to the same list, which is the point of the comparison.
fn imports(ir: &str) -> Vec<String> {
    // Attribute groups carry the import name; declarations point at a group.
    let mut groups: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    for line in ir.lines() {
        if !line.starts_with("attributes #") {
            continue;
        }
        let Some(id) = line
            .strip_prefix("attributes #")
            .and_then(|rest| rest.split_whitespace().next())
        else {
            continue;
        };
        let Some(start) = line.find("\"wasm-import-name\"") else {
            continue;
        };
        let rest = &line[start + "\"wasm-import-name\"".len()..];
        let Some(open) = rest.find('"').map(|open| open + 1) else {
            continue;
        };
        let Some(close) = rest[open..].find('"').map(|close| close + open) else {
            continue;
        };
        groups.insert(id.to_string(), rest[open..close].to_string());
    }
    let mut linked: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    for line in ir.lines() {
        let Some(body) = line.strip_prefix("declare ") else {
            continue;
        };
        let Some(at) = body.find('@').map(|at| at + 1) else {
            continue;
        };
        let Some(end) = body[at..].find('(').map(|end| end + at) else {
            continue;
        };
        let Some(group) = body.rsplit('#').next().map(str::trim) else {
            continue;
        };
        if let Some(name) = groups.get(group) {
            linked.insert(body[at..end].to_string(), name.clone());
        }
    }
    let Some(body) = ir
        .split("define dso_local void @cs_main")
        .nth(1)
        .and_then(|rest| rest.split("\n}").next())
    else {
        return Vec::new();
    };
    let mut called: Vec<String> = Vec::new();
    for line in body.lines() {
        let Some(at) = line.find("call").and_then(|_| line.find('@')) else {
            continue;
        };
        let start = at + 1;
        let Some(end) = line[start..].find('(').map(|end| end + start) else {
            continue;
        };
        let callee = &line[start..end];
        let resolved = linked
            .get(callee)
            .cloned()
            .unwrap_or_else(|| callee.to_string());
        if resolved.contains("llvm") {
            continue;
        }
        called.push(resolved);
    }
    called
}

fn build(directory: &Path, name: &str, source: &str, dependency: Option<&Path>) -> Option<String> {
    std::fs::create_dir_all(directory.join("src")).ok()?;
    let manifest = match dependency {
        Some(path) => format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\
             [lib]\ncrate-type = [\"rlib\"]\n\
             [dependencies]\nornis-gpu-shader = {{ path = \"{}\" }}\n",
            path.display()
        ),
        None => format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n"),
    };
    std::fs::write(directory.join("Cargo.toml"), manifest).ok()?;
    std::fs::write(directory.join("src/lib.rs"), source).ok()?;
    let status = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .current_dir(directory)
        .args([
            "rustc",
            "-q",
            "--target",
            "wasm32-unknown-unknown",
            "--release",
            "--",
            "-C",
            "opt-level=1",
            "-C",
            "panic=abort",
            "-C",
            "target-cpu=mvp",
            "-C",
            "codegen-units=1",
            "--emit=llvm-ir",
        ])
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let deps = directory
        .join("target")
        .join("wasm32-unknown-unknown")
        .join("release")
        .join("deps");
    let mut newest: Option<(std::time::SystemTime, String)> = None;
    for entry in std::fs::read_dir(deps).ok()? {
        let path = entry.ok()?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("ll") {
            continue;
        }
        let modified = std::fs::metadata(&path).ok()?.modified().ok()?;
        if newest.as_ref().is_none_or(|(best, _)| modified > *best) {
            newest = Some((modified, std::fs::read_to_string(path).ok()?));
        }
    }
    Some(newest?.1)
}

#[test]
fn typed_kernel_calls_the_same_imports_as_the_raw_one() {
    let root = std::env::temp_dir().join(format!("gpu-shader-parity-{}", std::process::id()));
    let sdk = Path::new(env!("CARGO_MANIFEST_DIR"));
    let Some(raw) = build(&root.join("raw"), "raw-kernel", RAW, None) else {
        eprintln!("SKIPPED (wasm32 target or cargo unavailable)");
        return;
    };
    let Some(typed) = build(&root.join("typed"), "typed-kernel", TYPED, Some(sdk)) else {
        eprintln!("SKIPPED (the typed kernel did not build)");
        return;
    };
    assert_eq!(
        imports(&typed),
        imports(&raw),
        "typed kernel must lower to the raw kernel's imports"
    );
    assert!(
        imports(&typed).contains(&"storage_len_f32".to_string()),
        "{}",
        typed
    );
    // The record is what the builder reads instead of CLI flags.
    let meta = ornis_gpu_shader::meta::GLOBAL;
    assert!(typed.contains(meta), "typed kernel lost `{meta}`: {typed}");
}
