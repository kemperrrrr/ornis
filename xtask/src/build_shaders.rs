//! `xtask build-shaders`: stable Rust → LLVM IR → WGSL/SPIR-V.
//!
//! Compiles a `no_std` shader crate with `cargo rustc` for
//! `wasm32-unknown-unknown` (so path and proc-macro dependencies resolve),
//! translates the emitted LLVM IR through `ornis-llvm2gpu`, and writes
//! WGSL/SPIR-V. `--artifact <file>` writes the WGSL the renderer includes;
//! `--check` compares it instead and fails on drift. `--multisampled`
//! translates the same kernel with multisampled texture bindings (sample-0
//! loads), which is the 4x MSAA variant of an AO kernel.

use std::path::{Path, PathBuf};
use std::process::{exit, Command};

/// Shader crate compiled when `--package` is omitted.
const DEFAULT_PACKAGE: &str = "ornis-gpu-shaders";
/// Entry extracted by the translator.
const DEFAULT_ENTRY: &str = "cs_main";
/// LLVM optimisation level used for the shader crate.
const DEFAULT_OPT: &str = "1";
/// Cargo target of the shader crate.
const DEFAULT_TARGET: &str = "wasm32-unknown-unknown";
/// Artifact directory below the workspace root.
const DEFAULT_OUT: &str = "target/shaders";
/// Fallback workgroup size for shaders without entry metadata.
const DEFAULT_WORKGROUP: [u32; 3] = [64, 1, 1];
/// Fallback `ornis_gpu.types` tags for shaders without entry metadata.
const DEFAULT_TYPES: [u8; 7] = [6, 6, 6, 6, 6, 6, 2];

struct Args {
    package: String,
    entry: String,
    opt: String,
    out: PathBuf,
    artifact: Option<PathBuf>,
    check: bool,
    workgroup: [u32; 3],
    types: Vec<u8>,
    multisampled: bool,
}

pub fn build_shaders(raw: &[String]) {
    let args = match parse(raw) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("xtask build-shaders: {message}");
            exit(2);
        }
    };
    if args.entry != DEFAULT_ENTRY && args.entry != "fs_main" {
        eprintln!("xtask build-shaders: entry must be `{DEFAULT_ENTRY}` or `fs_main`");
        exit(2);
    }
    let root = super::workspace_root();
    if let Err(message) = build(&root, &args) {
        eprintln!("xtask build-shaders: {message}");
        exit(1);
    }
}

fn parse(raw: &[String]) -> Result<Args, String> {
    let mut args = Args {
        package: DEFAULT_PACKAGE.to_string(),
        entry: DEFAULT_ENTRY.to_string(),
        opt: DEFAULT_OPT.to_string(),
        out: PathBuf::from(DEFAULT_OUT),
        artifact: None,
        check: false,
        workgroup: DEFAULT_WORKGROUP,
        types: DEFAULT_TYPES.to_vec(),
        multisampled: false,
    };
    let mut iter = raw.iter();
    while let Some(flag) = iter.next() {
        if flag == "--check" {
            args.check = true;
            continue;
        }
        if flag == "--multisampled" {
            args.multisampled = true;
            continue;
        }
        let value = iter.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--package" => args.package = value.clone(),
            "--entry" => args.entry = value.clone(),
            "--opt" => args.opt = value.clone(),
            "--out" => args.out = PathBuf::from(value),
            "--artifact" => args.artifact = Some(PathBuf::from(value)),
            "--workgroup" => args.workgroup = parse_workgroup(value)?,
            "--types" => args.types = parse_types(value)?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.check && args.artifact.is_none() {
        return Err("--check needs --artifact <file>".to_string());
    }
    Ok(args)
}

fn parse_workgroup(value: &str) -> Result<[u32; 3], String> {
    let mut parts = value.split(',');
    let mut next = |default: u32| -> Result<u32, String> {
        match parts.next() {
            Some(text) => text
                .parse::<u32>()
                .map_err(|error| format!("invalid workgroup size `{text}`: {error}")),
            None => Ok(default),
        }
    };
    let x = next(0)?;
    let y = next(1)?;
    let z = next(1)?;
    if parts.next().is_some() {
        return Err("workgroup size has more than three components".to_string());
    }
    if x == 0 || y == 0 || z == 0 {
        return Err("workgroup size components must be positive".to_string());
    }
    Ok([x, y, z])
}

fn build(root: &Path, args: &Args) -> Result<(), String> {
    let ll = compile(root, args)?;
    let ir =
        std::fs::read_to_string(&ll).map_err(|error| format!("read {}: {error}", ll.display()))?;
    let translated = match ornis_llvm2gpu::extract_meta(&ir) {
        Ok(None) => {
            if args.multisampled {
                return Err("--multisampled needs a kernel with entry metadata".to_string());
            }
            let structs: Vec<&[u8]> = if args.types.is_empty() {
                Vec::new()
            } else {
                vec![args.types.as_slice()]
            };
            ornis_llvm2gpu::translate_with_types_and_workgroup(&ir, &structs, args.workgroup)
        }
        Ok(Some(_)) => {
            let sampling = if args.multisampled {
                ornis_llvm2gpu::TextureSampling::Multisampled
            } else {
                ornis_llvm2gpu::TextureSampling::SingleSample
            };
            ornis_llvm2gpu::translate_with_meta_and_sampling(&ir, sampling)
        }
        Err(error) => Err(error),
    }
    .map_err(|error| format!("translate {}: {error}", args.package))?;

    let out = resolve(root, &args.out);
    std::fs::create_dir_all(&out).map_err(|error| format!("create {}: {error}", out.display()))?;
    let wgsl = out.join(format!("{}.wgsl", args.entry));
    std::fs::write(&wgsl, &translated.wgsl)
        .map_err(|error| format!("write {}: {error}", wgsl.display()))?;
    let bytes: Vec<u8> = translated
        .compute_spirv
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let spv = out.join(format!("{}.spv", args.entry));
    std::fs::write(&spv, bytes).map_err(|error| format!("write {}: {error}", spv.display()))?;

    if let Some(artifact) = &args.artifact {
        let artifact = resolve(root, artifact);
        if args.check {
            let current = std::fs::read_to_string(&artifact)
                .map_err(|error| format!("read {}: {error}", artifact.display()))?;
            if current != translated.wgsl {
                return Err(format!(
                    "artifact {} is stale; run `cargo xtask build-shaders --artifact {}`",
                    artifact.display(),
                    artifact.display()
                ));
            }
            eprintln!("xtask build-shaders: {} is current", artifact.display());
        } else {
            std::fs::write(&artifact, &translated.wgsl)
                .map_err(|error| format!("write {}: {error}", artifact.display()))?;
            eprintln!("xtask build-shaders: wrote {}", artifact.display());
        }
    }
    eprintln!("xtask build-shaders: {} -> {}", args.entry, wgsl.display());
    Ok(())
}

fn compile(root: &Path, args: &Args) -> Result<PathBuf, String> {
    let ir = resolve(root, &args.out).join("ir");
    std::fs::create_dir_all(&ir).map_err(|error| format!("create {}: {error}", ir.display()))?;
    // The emit path is part of cargo's fingerprint, so a unique name forces
    // rustc to run and write the IR. A fixed name would let a cached build
    // answer without producing anything, and `cargo clean -p` does not remove
    // per-target artifacts here.
    let ll = ir.join(format!("{}-{}.ll", args.package, fingerprint()));
    run_cargo_rustc(root, args, &ll)?;
    if !ll.exists() {
        return Err(format!(
            "no LLVM IR for `{}` at {} — is the wasm32 target installed?",
            args.package,
            ll.display()
        ));
    }
    // Keep only this run's IR so the directory does not grow without bound.
    remove_stale_ir(&ir, &args.package, &ll);
    Ok(ll)
}

/// Unique suffix that changes on every run, including within the same process.
fn fingerprint() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default()
}

fn remove_stale_ir(ir: &Path, package: &str, keep: &Path) {
    let prefix = format!("{package}-");
    let Ok(entries) = std::fs::read_dir(ir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == keep {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if name.starts_with(&prefix) && path.extension().and_then(|ext| ext.to_str()) == Some("ll")
        {
            let _ = std::fs::remove_file(&path);
        }
    }
}

fn run_cargo_rustc(root: &Path, args: &Args, ll: &Path) -> Result<(), String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .current_dir(root)
        .args(["rustc", "-p", &args.package, "--target", DEFAULT_TARGET])
        .arg("--release")
        .args(["--", "-C"])
        .arg(format!("opt-level={}", args.opt))
        .args([
            "-C",
            "panic=abort",
            "-C",
            "target-cpu=mvp",
            "-C",
            "codegen-units=1",
        ])
        .arg(format!("--emit=llvm-ir={}", ll.display()))
        .status()
        .map_err(|error| format!("spawn cargo: {error}"))?;
    if !status.success() {
        return Err(format!("cargo rustc failed with {status}"));
    }
    Ok(())
}

fn parse_types(value: &str) -> Result<Vec<u8>, String> {
    if value.is_empty() {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .map(|tag| {
            tag.trim()
                .parse::<u8>()
                .map_err(|error| format!("invalid field tag `{tag}`: {error}"))
        })
        .collect()
}

fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults() {
        let args = parse(&[]).expect("defaults");
        assert_eq!(args.package, DEFAULT_PACKAGE);
        assert_eq!(args.entry, DEFAULT_ENTRY);
        assert_eq!(args.opt, DEFAULT_OPT);
        assert!(!args.check);
        assert!(args.artifact.is_none());
    }

    #[test]
    fn parse_overrides() {
        let raw = vec![
            "--package".to_string(),
            "my-shaders".to_string(),
            "--entry".to_string(),
            "cs_main".to_string(),
            "--opt".to_string(),
            "3".to_string(),
            "--out".to_string(),
            "artifacts".to_string(),
            "--artifact".to_string(),
            "crates/render/src/gpu_culling/cull_generated.wgsl".to_string(),
            "--workgroup".to_string(),
            "32,2".to_string(),
            "--types".to_string(),
            "5,2".to_string(),
            "--check".to_string(),
        ];
        let args = parse(&raw).expect("overrides");
        assert_eq!(args.package, "my-shaders");
        assert_eq!(args.opt, "3");
        assert_eq!(args.out, PathBuf::from("artifacts"));
        assert_eq!(args.workgroup, [32, 2, 1]);
        assert_eq!(args.types, [5, 2]);
        assert!(args.check);
        assert_eq!(
            args.artifact,
            Some(PathBuf::from(
                "crates/render/src/gpu_culling/cull_generated.wgsl"
            ))
        );
    }

    #[test]
    fn check_requires_artifact() {
        assert!(parse(&["--check".to_string()]).is_err());
    }

    #[test]
    fn multisampled_is_a_flag_not_a_value() {
        let raw = vec!["--multisampled".to_string(), "--package".to_string()];
        // `--package` still needs its value, so parsing must fail after the
        // flag is consumed without swallowing the next argument.
        assert!(parse(&raw).is_err());
        let args = parse(&["--multisampled".to_string()]).expect("flag");
        assert!(args.multisampled);
    }
}
