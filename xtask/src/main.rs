//! Ornis build/task automation (cargo-xtask pattern, no shell required).
//!
//! Usage:
//!   cargo xtask editor  [--skip-wasm] [--editor-dir <path>]
//!   cargo xtask quality [--ci] [--full] [--bench] [--everything]
//!   cargo xtask bca [--install] [--write-baseline] [--report] [--full] [--init]
//!   cargo xtask fuzz <target> [-- <libfuzzer args>]
//!   cargo xtask mutants [-- <cargo-mutants args>]
//!   cargo editor   [--skip-wasm] [--editor-dir <path>]   (alias)
//!
//! `editor` builds the WASM viewport (wasm-pack) and runs the engine with the
//! remote editor server. Cross-platform: everything goes through
//! std::process::Command without a shell.

mod bca;
mod quality;

use std::path::PathBuf;
use std::process::{exit, Command};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        usage(2);
    };
    match cmd.as_str() {
        "editor" => editor(&args[1..]),
        "quality" => quality::quality(&args[1..]),
        "bca" => bca::bca(&args[1..]),
        "fuzz" => quality::fuzz(&args[1..]),
        "mutants" => quality::mutants(&args[1..]),
        "-h" | "--help" | "help" => usage(0),
        other => {
            eprintln!("xtask: unknown task '{other}'");
            usage(2);
        }
    }
}

fn usage(code: i32) -> ! {
    eprintln!(
        "xtask — Ornis task runner\n\
         \n\
         TASKS:\n  \
         editor [--skip-wasm] [--editor-dir <path>]\n      \
         Build the WASM viewport (wasm-pack) and launch the engine\n      \
         with the remote editor at http://127.0.0.1:3420\n      \
         --skip-wasm    reuse the existing editor/pkg build\n      \
         --editor-dir   editor frontend directory (default: <workspace>/editor)\n  \
         quality [--ci] [--full] [--bench] [--everything]\n      \
                  Quality gate: fmt, clippy, bca, test, audit, deny, outdated (level 1);\n      \
                  --ci adds rustdoc + wasm32 check (the exact set CI runs);\n      \
                  --full adds llvm-cov coverage + bench compile-check (level 2);\n      \
                  --bench runs the full criterion suite (long);\n      \
                  --everything = --ci + --full + --bench + mutants + fuzz smoke\n  \
         bca [--install] [--write-baseline] [--report] [--full] [--init]\n      \
                  big-code-analysis gate: complexity metrics\n      \
                  --install        cargo install big-code-analysis-cli --locked\n      \
                  --write-baseline bca check --write-baseline (updates .bca-baseline.toml)\n      \
                  --report         bca report HTML + Markdown to target/bca/\n      \
                  --init           install (if needed) + baseline + report\n      \
                  --full           --init + cargo xtask quality\n      \
                  (MPL-2.0 external binary, does NOT affect MIT OR Apache-2.0)\n  \
         fuzz <target> [-- <args>]\n      \
         Run a cargo-fuzz target (scene_ron, materialx_parse, editor_command) via +nightly\n  \
         mutants [-- <args>]\n      \
         Run cargo-mutants against ornis-core"
    );
    exit(code);
}

fn workspace_root() -> PathBuf {
    // xtask/Cargo.toml lives one level below the workspace root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent directory")
        .to_path_buf()
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("xtask: failed to spawn {what}: {e}"));
    if !status.success() {
        eprintln!("xtask: {what} failed with {status}");
        exit(status.code().unwrap_or(1));
    }
}

fn which(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn editor(args: &[String]) {
    let mut skip_wasm = false;
    let mut editor_dir: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--skip-wasm" => skip_wasm = true,
            "--editor-dir" => {
                i += 1;
                editor_dir = Some(
                    args.get(i)
                        .unwrap_or_else(|| {
                            eprintln!("xtask: --editor-dir requires a path");
                            exit(2);
                        })
                        .clone(),
                );
            }
            "-h" | "--help" => usage(0),
            other => {
                eprintln!("xtask: unknown flag '{other}'");
                usage(2);
            }
        }
        i += 1;
    }

    let root = workspace_root();
    let editor_dir = editor_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("editor"));
    let editor_dir = std::fs::canonicalize(&editor_dir).unwrap_or_else(|e| {
        eprintln!(
            "xtask: editor dir '{}' not found: {e}",
            editor_dir.display()
        );
        exit(1);
    });

    // ── 1. WASM build ────────────────────────────────────────────────
    if skip_wasm {
        eprintln!("xtask: skipping wasm build (--skip-wasm)");
    } else {
        if !which("wasm-pack") {
            eprintln!(
                "xtask: wasm-pack not found in PATH.\n\
                 Install it with:  cargo install wasm-pack\n\
                 (or pass --skip-wasm to reuse the existing editor/pkg build)"
            );
            exit(1);
        }
        let out_dir = editor_dir.join("pkg");
        eprintln!(
            "xtask: wasm-pack build crates/wasm --target web --out-dir {}",
            out_dir.display()
        );
        run(
            Command::new("wasm-pack")
                .arg("build")
                .arg(root.join("crates/wasm"))
                .arg("--target")
                .arg("web")
                .arg("--out-dir")
                .arg(&out_dir)
                .current_dir(&root),
            "wasm-pack build",
        );
    }

    // ── 2. Run the engine with the remote editor ─────────────────────
    eprintln!(
        "xtask: cargo run --features editor-only (editor dir: {})",
        editor_dir.display()
    );
    run(
        Command::new("cargo")
            .arg("run")
            .arg("--features")
            .arg("editor-only")
            .arg("--")
            .arg("--editor-dir")
            .arg(&editor_dir)
            .current_dir(&root),
        "cargo run --features editor-only",
    );
}
