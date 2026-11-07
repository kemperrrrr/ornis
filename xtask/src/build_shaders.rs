//! `xtask build-shaders`: stable Rust → LLVM IR → WGSL/SPIR-V.
//!
//! The first cut compiles one `no_std` source file with
//! `rustc --emit=llvm-ir`, translates it through `ornis-llvm2gpu`, and
//! writes a content-addressed WGSL/SPIR-V pair plus stable `cs_main`
//! aliases. A matching content hash skips translation; the IR is still
//! produced so the cache key covers the actual compiler output.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::{exit, Command};

/// Source compiled when `--source` is omitted.
const DEFAULT_SOURCE: &str = "crates/gpu-shaders/src/lib.rs";
/// Entry extracted by the translator.
const DEFAULT_ENTRY: &str = "cs_main";
/// LLVM optimisation level used for the shader crate.
const DEFAULT_OPT: &str = "s";
/// Artifact directory below the workspace root.
const DEFAULT_OUT: &str = "target/shaders";
/// Workgroup size of the cull kernel (one invocation per source record).
const DEFAULT_WORKGROUP: [u32; 3] = [64, 1, 1];
/// `ornis_gpu.types` tags of the cull `CullParams` struct: six `vec4` planes
/// and the `u32` count.
const DEFAULT_TYPES: [u8; 7] = [6, 6, 6, 6, 6, 6, 2];
/// Bumped when the translator's output changes without the IR changing.
const CACHE_TAG: &str = "llvm2gpu-v1";

struct Args {
    source: PathBuf,
    entry: String,
    opt: String,
    out: PathBuf,
    workgroup: [u32; 3],
    types: Vec<u8>,
}

pub fn build_shaders(raw: &[String]) {
    let args = match parse(raw) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("xtask build-shaders: {message}");
            exit(2);
        }
    };
    if args.entry != DEFAULT_ENTRY {
        eprintln!(
            "xtask build-shaders: only `{DEFAULT_ENTRY}` is supported until entry metadata lands"
        );
        exit(2);
    }
    let root = super::workspace_root();
    let source = resolve(&root, &args.source);
    let out = resolve(&root, &args.out);
    if let Err(message) = build(&source, &out, &args) {
        eprintln!("xtask build-shaders: {message}");
        exit(1);
    }
}

fn parse(raw: &[String]) -> Result<Args, String> {
    let mut args = Args {
        source: PathBuf::from(DEFAULT_SOURCE),
        entry: DEFAULT_ENTRY.to_string(),
        opt: DEFAULT_OPT.to_string(),
        out: PathBuf::from(DEFAULT_OUT),
        workgroup: DEFAULT_WORKGROUP,
        types: DEFAULT_TYPES.to_vec(),
    };
    let mut iter = raw.iter();
    while let Some(flag) = iter.next() {
        let value = iter.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--source" => args.source = PathBuf::from(value),
            "--entry" => args.entry = value.clone(),
            "--opt" => args.opt = value.clone(),
            "--out" => args.out = PathBuf::from(value),
            "--workgroup" => args.workgroup = parse_workgroup(value)?,
            "--types" => args.types = parse_types(value)?,
            other => return Err(format!("unknown argument {other}")),
        }
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

fn build(source: &Path, out: &Path, args: &Args) -> Result<(), String> {
    if !source.is_file() {
        return Err(format!("source {} does not exist", source.display()));
    }
    std::fs::create_dir_all(out).map_err(|error| format!("create {}: {error}", out.display()))?;
    let ll = out.join(format!("{}.ll", args.entry));
    compile(source, &args.opt, &ll)?;
    let ir =
        std::fs::read_to_string(&ll).map_err(|error| format!("read {}: {error}", ll.display()))?;
    let hash = cache_key(&ir, &args.entry, &args.opt, args.workgroup, &args.types);
    let stem = format!("{}-{hash:016x}", args.entry);
    let cached =
        out.join(format!("{stem}.wgsl")).is_file() && out.join(format!("{stem}.spv")).is_file();
    let translated = if cached {
        None
    } else {
        Some({
            let structs: Vec<&[u8]> = if args.types.is_empty() {
                Vec::new()
            } else {
                vec![args.types.as_slice()]
            };
            ornis_llvm2gpu::translate_with_types_and_workgroup(&ir, &structs, args.workgroup)
                .map_err(|error| format!("translate {}: {error}", source.display()))?
        })
    };
    if let Some(translated) = translated {
        write_artifacts(out, &stem, &translated)?;
    }
    copy_latest(out, &stem, &args.entry)?;
    let state = if cached { "cached" } else { "built" };
    eprintln!(
        "xtask build-shaders: {state} {} -> {}",
        args.entry,
        out.join(format!("{}.wgsl", args.entry)).display()
    );
    Ok(())
}

fn compile(source: &Path, opt: &str, ll: &Path) -> Result<(), String> {
    let status = Command::new("rustc")
        .args([
            "--edition=2024",
            "--crate-type=lib",
            "--target",
            "wasm32-unknown-unknown",
            "-C",
            &format!("opt-level={opt}"),
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
        .map_err(|error| format!("spawn rustc: {error}"))?;
    if !status.success() {
        return Err(format!("rustc failed with {status}"));
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

fn cache_key(ir: &str, entry: &str, opt: &str, workgroup: [u32; 3], types: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    CACHE_TAG.hash(&mut hasher);
    ir.hash(&mut hasher);
    entry.hash(&mut hasher);
    opt.hash(&mut hasher);
    workgroup.hash(&mut hasher);
    types.hash(&mut hasher);
    hasher.finish()
}

fn write_artifacts(
    out: &Path,
    stem: &str,
    translated: &ornis_llvm2gpu::Translated,
) -> Result<(), String> {
    let wgsl = out.join(format!("{stem}.wgsl"));
    std::fs::write(&wgsl, &translated.wgsl)
        .map_err(|error| format!("write {}: {error}", wgsl.display()))?;
    let bytes: Vec<u8> = translated
        .compute_spirv
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let spv = out.join(format!("{stem}.spv"));
    std::fs::write(&spv, bytes).map_err(|error| format!("write {}: {error}", spv.display()))?;
    Ok(())
}

fn copy_latest(out: &Path, stem: &str, entry: &str) -> Result<(), String> {
    for extension in ["wgsl", "spv"] {
        let from = out.join(format!("{stem}.{extension}"));
        let to = out.join(format!("{entry}.{extension}"));
        std::fs::copy(&from, &to)
            .map_err(|error| format!("copy {} -> {}: {error}", from.display(), to.display()))?;
    }
    Ok(())
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
        assert_eq!(args.entry, DEFAULT_ENTRY);
        assert_eq!(args.opt, DEFAULT_OPT);
        assert_eq!(args.workgroup, DEFAULT_WORKGROUP);
        assert_eq!(args.types, DEFAULT_TYPES);
    }

    #[test]
    fn parse_overrides() {
        let raw = vec![
            "--source".to_string(),
            "kernel.rs".to_string(),
            "--entry".to_string(),
            "cs_main".to_string(),
            "--opt".to_string(),
            "3".to_string(),
            "--out".to_string(),
            "artifacts".to_string(),
            "--workgroup".to_string(),
            "32,2".to_string(),
            "--types".to_string(),
            "5,2".to_string(),
        ];
        let args = parse(&raw).expect("overrides");
        assert_eq!(args.source, PathBuf::from("kernel.rs"));
        assert_eq!(args.opt, "3");
        assert_eq!(args.out, PathBuf::from("artifacts"));
        assert_eq!(args.workgroup, [32, 2, 1]);
        assert_eq!(args.types, [5, 2]);
    }

    #[test]
    fn cache_key_tracks_ir_and_flags() {
        let base = cache_key("ir", "cs_main", "s", [64, 1, 1], &[5, 2]);
        assert_eq!(base, cache_key("ir", "cs_main", "s", [64, 1, 1], &[5, 2]));
        assert_ne!(base, cache_key("ir2", "cs_main", "s", [64, 1, 1], &[5, 2]));
        assert_ne!(base, cache_key("ir", "cs_main", "3", [64, 1, 1], &[5, 2]));
        assert_ne!(base, cache_key("ir", "cs_main", "s", [32, 1, 1], &[5, 2]));
        assert_ne!(base, cache_key("ir", "cs_main", "s", [64, 1, 1], &[5]));
    }
}
