//! Quality gate: fmt, clippy, tests, supply-chain (audit/deny/outdated),
//! plus level 2 (--full: coverage + bench compile-check; --bench: criterion).
//!
//! Stage architecture: each stage prints a header and PASS/FAIL/SKIP/INFO;
//! the run continues after a failed stage and prints a summary table at
//! the end; the exit code is 1 if any stage FAILs.

use std::path::Path;
use std::process::{exit, Command};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Pass,
    Fail,
    /// The tool is missing — the stage is skipped (not counted as a failure).
    Skip,
    /// Informational stage (outdated): the result does not affect the exit code.
    Info,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Fail => "FAIL",
            Status::Skip => "SKIP",
            Status::Info => "INFO",
        }
    }
}

struct StageResult {
    name: String,
    status: Status,
    note: String,
}

pub fn quality(args: &[String]) {
    let mut full = false;
    let mut bench = false;
    let mut ci = false;
    let mut everything = false;
    for a in args {
        match a.as_str() {
            "--full" => full = true,
            "--bench" => bench = true,
            "--ci" => ci = true,
            "--everything" => everything = true,
            "-h" | "--help" => quality_usage(0),
            other => {
                eprintln!("xtask quality: unknown flag '{other}'");
                quality_usage(2);
            }
        }
    }
    // --everything implies all levels: level 2 (coverage + bench
    // compile-check), criterion, the CI set (doc + wasm check) and
    // the deep static-analysis stages (mutants, fuzz smoke).
    if everything {
        full = true;
        bench = true;
        ci = true;
    }

    let root = crate::workspace_root();
    let mut results: Vec<StageResult> = Vec::new();
    // The total is computed up-front so the stage numbering stays
    // honest even when a deep stage is skipped (tool not installed).
    let total = 6
        + usize::from(ci) * 2
        + usize::from(full) * 2
        + usize::from(bench)
        + usize::from(everything) * 2;
    let mut n = 0usize;

    // ── Level 1 (mandatory set) ───────────────────────────────

    n += 1;
    results.push(run_stage(
        n,
        total,
        "fmt",
        "cargo fmt --all -- --check",
        cmd(&root, "cargo", &["fmt", "--all", "--", "--check"]),
        false,
    ));

    n += 1;
    results.push(run_stage(
        n,
        total,
        "clippy",
        "cargo clippy --workspace --all-targets -- -D warnings",
        cmd(
            &root,
            "cargo",
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        ),
        false,
    ));

    n += 1;
    results.push(run_stage(
        n,
        total,
        "test",
        "cargo test --workspace",
        cmd(&root, "cargo", &["test", "--workspace"]),
        false,
    ));

    n += 1;
    results.push(run_stage(
        n,
        total,
        "audit",
        "cargo audit",
        cmd(&root, "cargo", &["audit"]),
        false,
    ));

    n += 1;
    results.push(run_stage(
        n,
        total,
        "deny",
        "cargo deny check",
        cmd(&root, "cargo", &["deny", "check"]),
        false,
    ));

    // Informational stage: cargo-outdated returns a non-zero code when
    // dependencies are outdated — that is not a reason to fail the gate.
    n += 1;
    results.push(run_stage(
        n,
        total,
        "outdated (info)",
        "cargo outdated --workspace",
        cmd(&root, "cargo", &["outdated", "--workspace"]),
        true,
    ));

    // ── Level 2 (--full) ───────────────────────────────────────────

    if full {
        n += 1;
        results.push(run_stage(
            n,
            total,
            "coverage (llvm-cov)",
            "cargo llvm-cov --workspace --html --output-dir target/llvm-cov",
            cmd(
                &root,
                "cargo",
                &[
                    "llvm-cov",
                    "--workspace",
                    "--html",
                    "--output-dir",
                    "target/llvm-cov",
                ],
            ),
            false,
        ));

        n += 1;
        results.push(run_stage(
            n,
            total,
            "bench compile-check",
            "cargo bench --workspace --no-run",
            cmd(&root, "cargo", &["bench", "--workspace", "--no-run"]),
            false,
        ));
    }

    if bench {
        n += 1;
        results.push(run_stage(
            n,
            total,
            "criterion benches",
            "cargo bench --workspace",
            cmd(&root, "cargo", &["bench", "--workspace"]),
            false,
        ));
    }

    // ── CI set (--ci): rustdoc + wasm target check ─────────────
    // These two stages mirror what the GitHub Actions quality job
    // runs; --ci makes the local gate identical to CI by construction.
    if ci {
        n += 1;
        results.push(run_stage(
            n,
            total,
            "doc",
            "cargo doc --workspace --no-deps",
            cmd(&root, "cargo", &["doc", "--workspace", "--no-deps"]),
            false,
        ));

        n += 1;
        if wasm_target_installed() {
            results.push(run_stage(
                n,
                total,
                "wasm-check",
                "cargo check -p ornis-wasm --target wasm32-unknown-unknown",
                cmd(
                    &root,
                    "cargo",
                    &[
                        "check",
                        "-p",
                        "ornis-wasm",
                        "--target",
                        "wasm32-unknown-unknown",
                    ],
                ),
                false,
            ));
        } else {
            results.push(skip_stage(
                n,
                total,
                "wasm-check",
                "wasm32-unknown-unknown target not installed:  rustup target add wasm32-unknown-unknown",
            ));
        }
    }

    // ── Deep static analysis (--everything) ─────────────────────
    // Long-running stages, only under the explicit deep flag.
    if everything {
        n += 1;
        if cargo_subcommand_exists("mutants") {
            results.push(run_stage(
                n,
                total,
                "mutants (ornis-core)",
                "cargo mutants -p ornis-core --timeout 300",
                cmd(
                    &root,
                    "cargo",
                    &["mutants", "-p", "ornis-core", "--timeout", "300"],
                ),
                false,
            ));
        } else {
            results.push(skip_stage(
                n,
                total,
                "mutants (ornis-core)",
                "cargo-mutants not installed",
            ));
        }

        n += 1;
        if cargo_subcommand_exists("fuzz") && nightly_available() {
            results.push(run_stage(
                n,
                total,
                "fuzz smoke (scene_ron)",
                "cargo +nightly fuzz run scene_ron -- -runs=200",
                cmd(
                    &root,
                    "cargo",
                    &["+nightly", "fuzz", "run", "scene_ron", "--", "-runs=200"],
                ),
                false,
            ));
        } else {
            results.push(skip_stage(
                n,
                total,
                "fuzz smoke (scene_ron)",
                "cargo-fuzz or nightly toolchain missing",
            ));
        }
    }

    print_summary(&results);
    if results.iter().any(|r| r.status == Status::Fail) {
        exit(1);
    }
}

fn quality_usage(code: i32) -> ! {
    eprintln!(
        "xtask quality — the Ornis quality gate\n\
         \n\
         USAGE:\n  \
         cargo xtask quality           quick set (level 1): fmt, clippy, test, audit, deny, outdated\n  \
         cargo xtask quality --ci      + rustdoc and wasm32 check (same set GitHub Actions runs)\n  \
         cargo xtask quality --full    + coverage (llvm-cov → target/llvm-cov/html) and bench compile-check\n  \
         cargo xtask quality --bench   + full criterion benchmark run (slow)\n  \
         cargo xtask quality --everything\n      \
         everything: --ci + --full + --bench + mutants (ornis-core) + fuzz smoke (slow, minutes to hours)"
    );
    exit(code);
}

/// Builds a Command with the workspace root as the working directory.
fn cmd(root: &Path, program: &str, args: &[&str]) -> Command {
    let mut c = Command::new(program);
    c.args(args).current_dir(root);
    c
}

/// Runs one stage: header, tool availability check, status.
/// `informational` — a non-zero exit is not counted as FAIL (cargo-outdated).
fn run_stage(
    index: usize,
    total: usize,
    name: &str,
    display_cmd: &str,
    mut command: Command,
    informational: bool,
) -> StageResult {
    eprintln!();
    eprintln!("═══ [{index}/{total}] {name}: {display_cmd} ═══");

    // Check for an external cargo tool, with an install hint.
    // Only third-party subcommands are checked: built-in ones (test, bench, …)
    // have no cargo-<sub> binary that could be found in PATH.
    const EXTERNAL: &[&str] = &["audit", "deny", "outdated", "llvm-cov"];
    let program = command.get_program().to_string_lossy().into_owned();
    let first_arg = command
        .get_args()
        .next()
        .map(|a| a.to_string_lossy().into_owned());
    if program == "cargo" {
        if let Some(sub) = first_arg.filter(|s| EXTERNAL.contains(&s.as_str())) {
            if !cargo_subcommand_exists(&sub) {
                let hint = install_hint(&sub);
                eprintln!("xtask quality: SKIP — tool 'cargo {sub}' not found.\n{hint}");
                return StageResult {
                    name: name.to_string(),
                    status: Status::Skip,
                    note: format!("cargo-{sub} not installed"),
                };
            }
        }
    }

    match command.status() {
        Ok(status) if status.success() => {
            eprintln!(
                "── {name}: {} ──",
                if informational { "INFO" } else { "PASS" }
            );
            StageResult {
                name: name.to_string(),
                status: if informational {
                    Status::Info
                } else {
                    Status::Pass
                },
                note: String::new(),
            }
        }
        Ok(status) => {
            if informational {
                eprintln!("── {name}: INFO (exit {status} — outdated dependencies) ──");
                StageResult {
                    name: name.to_string(),
                    status: Status::Info,
                    note: format!("{status}"),
                }
            } else {
                eprintln!("── {name}: FAIL (exit {status}) ──");
                StageResult {
                    name: name.to_string(),
                    status: Status::Fail,
                    note: format!("{status}"),
                }
            }
        }
        Err(e) => {
            eprintln!("── {name}: FAIL (spawn error: {e}) ──");
            StageResult {
                name: name.to_string(),
                status: Status::Fail,
                note: format!("spawn: {e}"),
            }
        }
    }
}

fn print_summary(results: &[StageResult]) {
    eprintln!();
    eprintln!("╔════════════ QUALITY SUMMARY ════════════╗");
    for r in results {
        let note = if r.note.is_empty() {
            String::new()
        } else {
            format!(" ({})", r.note)
        };
        eprintln!("  {:<22} {:<4}{}", r.name, r.status.label(), note);
    }
    eprintln!("╚═════════════════════════════════════════╝");
}

/// Whether a cargo subcommand exists in PATH (`cargo-<sub>`).
/// Checks for the binary, not `--version`: some tools
/// (e.g. cargo-outdated) do not understand `--version` directly.
fn cargo_subcommand_exists(sub: &str) -> bool {
    let binary = if cfg!(windows) {
        format!("cargo-{sub}.exe")
    } else {
        format!("cargo-{sub}")
    };
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(&binary).is_file()))
        .unwrap_or(false)
}

/// Records a SKIP without spawning a command (no progress output).
fn skip_stage(index: usize, total: usize, name: &str, note: &str) -> StageResult {
    eprintln!();
    eprintln!("═══ [{index}/{total}] {name} ═══");
    eprintln!("── {name}: SKIP ({note}) ──");
    StageResult {
        name: name.to_string(),
        status: Status::Skip,
        note: note.to_string(),
    }
}

/// Whether the wasm32-unknown-unknown target is installed for the
/// active toolchain (`rustup target list --installed`).
fn wasm_target_installed() -> bool {
    Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .map(|out| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .any(|l| l.trim().starts_with("wasm32-unknown-unknown"))
        })
        .unwrap_or(false)
}

fn install_hint(sub: &str) -> String {
    match sub {
        "audit" => "Install:  cargo install cargo-audit --locked".to_string(),
        "deny" => "Install:  cargo install cargo-deny --locked".to_string(),
        "outdated" => "Install:  cargo install cargo-outdated --locked".to_string(),
        "llvm-cov" => "Install:  cargo install cargo-llvm-cov --locked\n\
             and the component:  rustup component add llvm-tools-preview"
            .to_string(),
        "fuzz" => "Install:  cargo install cargo-fuzz --locked".to_string(),
        "mutants" => "Install:  cargo install cargo-mutants --locked".to_string(),
        other => format!("Install:  cargo install cargo-{other} --locked"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// fuzz / mutants — separate subcommands (not part of quality default)
// ═══════════════════════════════════════════════════════════════════════

pub fn fuzz(args: &[String]) {
    let Some(target) = args.first() else {
        eprintln!(
            "xtask fuzz — runs cargo-fuzz targets (external-input parsers)\n\
             \n\
             USAGE:\n  \
             cargo xtask fuzz <target> [-- <libfuzzer args>]\n  \
             available targets: scene_ron, materialx_parse\n\
             \n\
             Example:  cargo xtask fuzz scene_ron -- -runs=1000"
        );
        exit(2);
    };
    let extra: Vec<&str> = args[1..].iter().map(String::as_str).collect();

    if !cargo_subcommand_exists("fuzz") {
        eprintln!(
            "xtask fuzz: 'cargo-fuzz' not found.\n{}",
            install_hint("fuzz")
        );
        exit(1);
    }
    if !nightly_available() {
        eprintln!(
            "xtask fuzz: nightly toolchain not found (cargo-fuzz requires nightly).\n\
             Install:  rustup toolchain install nightly\n\
             (the workspace is pinned to stable via rust-toolchain.toml — fuzz is \
             always run explicitly through +nightly)"
        );
        exit(1);
    }

    let root = crate::workspace_root();
    let mut c = Command::new("cargo");
    c.arg("+nightly")
        .arg("fuzz")
        .arg("run")
        .arg(target)
        .args(&extra)
        .current_dir(&root);
    eprintln!(
        "xtask fuzz: cargo +nightly fuzz run {target} {}",
        extra.join(" ")
    );
    let status = c
        .status()
        .unwrap_or_else(|e| panic!("xtask fuzz: failed to spawn cargo-fuzz: {e}"));
    exit(status.code().unwrap_or(1));
}

pub fn mutants(args: &[String]) {
    if !cargo_subcommand_exists("mutants") {
        eprintln!(
            "xtask mutants: 'cargo-mutants' not found.\n{}",
            install_hint("mutants")
        );
        exit(1);
    }

    let root = crate::workspace_root();
    let extra: Vec<&str> = args.iter().map(String::as_str).collect();
    let mut c = Command::new("cargo");
    c.arg("mutants")
        .arg("-p")
        .arg("ornis-core")
        .arg("--timeout")
        .arg("300")
        .args(&extra)
        .current_dir(&root);
    eprintln!(
        "xtask mutants: cargo mutants -p ornis-core --timeout 300 {}",
        extra.join(" ")
    );
    let status = c
        .status()
        .unwrap_or_else(|e| panic!("xtask mutants: failed to spawn cargo-mutants: {e}"));
    exit(status.code().unwrap_or(1));
}

fn nightly_available() -> bool {
    Command::new("cargo")
        .arg("+nightly")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
