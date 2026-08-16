//! `cargo xtask bca` — automation for big-code-analysis (bca) gate.
//! The CLI `bca` is MPL-2.0 as external binary, does NOT affect Ornis MIT OR Apache-2.0.
//! See docs/quality/bca.md and bca.toml
//!
//! User's manual sequence:
//!   cargo install big-code-analysis-cli --locked
//!   bca check --write-baseline
//!   git add .bca-baseline.toml
//!   bca report -O html -o target/bca/index.html
//!   cargo xtask quality
//!
//! This xtask wraps it:
//!   cargo xtask bca                        -> bca check
//!   cargo xtask bca --install              -> cargo install big-code-analysis-cli --locked
//!   cargo xtask bca --write-baseline       -> bca check --write-baseline (+ git add hint)
//!   cargo xtask bca --report               -> bca report HTML + Markdown
//!   cargo xtask bca --full                 -> install (if needed) + baseline + report + quality
//!   cargo xtask bca --init                 -> alias for --full without final quality run

use std::path::PathBuf;
use std::process::{exit, Command};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has parent")
        .to_path_buf()
}

fn binary_exists(bin: &str) -> bool {
    let binary = if cfg!(windows) {
        format!("{bin}.exe")
    } else {
        bin.to_string()
    };
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(&binary).is_file()))
        .unwrap_or(false)
}

fn run_or_fail(cmd: &mut Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("xtask bca: failed to spawn {what}: {e}"));
    if !status.success() {
        eprintln!("xtask bca: {what} failed with {status}");
        exit(status.code().unwrap_or(1));
    }
}

pub fn bca(args: &[String]) {
    let mut install = false;
    let mut write_baseline = false;
    let mut report = false;
    let mut full = false;
    let mut init = false;
    let mut check = false;

    if args.is_empty() {
        check = true;
    }

    for a in args {
        match a.as_str() {
            "--install" => install = true,
            "--write-baseline" | "--baseline" => write_baseline = true,
            "--report" => report = true,
            "--full" => full = true,
            "--init" => init = true,
            "--check" | "check" => check = true,
            "-h" | "--help" | "help" => bca_usage(0),
            other => {
                eprintln!("xtask bca: unknown flag '{other}'");
                bca_usage(2);
            }
        }
    }

    if full || init {
        install = true;
        write_baseline = true;
        report = true;
        if full {
            // full will also run quality at the end
        }
    }

    let root = workspace_root();

    if install {
        if binary_exists("bca") {
            eprintln!("xtask bca: `bca` already in PATH, skipping install");
        } else {
            eprintln!("xtask bca: installing big-code-analysis-cli --locked");
            let mut c = Command::new("cargo");
            c.args(["install", "big-code-analysis-cli", "--locked"])
                .current_dir(&root);
            run_or_fail(&mut c, "cargo install big-code-analysis-cli");
        }
    }

    if write_baseline || check {
        if !binary_exists("bca") {
            eprintln!(
                "xtask bca: `bca` not found in PATH.\n\
                 Install: cargo install big-code-analysis-cli --locked\n\
                 or: cargo xtask bca --install"
            );
            exit(1);
        }
    }

    if write_baseline {
        eprintln!("xtask bca: bca check --write-baseline");
        let mut c = Command::new("bca");
        c.args(["check", "--write-baseline"]).current_dir(&root);
        run_or_fail(&mut c, "bca check --write-baseline");

        eprintln!("\nxtask bca: baseline written to .bca-baseline.toml");
        eprintln!("  → review diff: bca diff-baseline .bca-baseline.old.toml .bca-baseline.toml (if you saved old)");
        eprintln!("  → then: git add .bca-baseline.toml && git commit -m \"chore(bca): update baseline\"");
    }

    if check && !write_baseline {
        eprintln!("xtask bca: bca check");
        let mut c = Command::new("bca");
        c.arg("check").current_dir(&root);
        run_or_fail(&mut c, "bca check");
    }

    if report {
        if !binary_exists("bca") {
            eprintln!("xtask bca: `bca` not found, cannot report");
            exit(1);
        }
        let out_dir = root.join("target").join("bca");
        std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| {
            eprintln!("xtask bca: failed to create {}: {e}", out_dir.display());
            exit(1);
        });

        let html = out_dir.join("index.html");
        eprintln!("xtask bca: bca report -O html -o {}", html.display());
        let mut c = Command::new("bca");
        c.args([
            "report",
            "-O",
            "html",
            "-o",
            html.to_str().unwrap(),
        ])
        .current_dir(&root);
        run_or_fail(&mut c, "bca report html");

        let md = out_dir.join("report.md");
        eprintln!("xtask bca: bca report -O markdown -o {}", md.display());
        let mut c = Command::new("bca");
        c.args([
            "report",
            "-O",
            "markdown",
            "-o",
            md.to_str().unwrap(),
        ])
        .current_dir(&root);
        // markdown report is informational, don't fail if not supported
        let status = c.status().unwrap_or_else(|e| {
            eprintln!("xtask bca: failed to spawn report markdown: {e}");
            exit(1);
        });
        if !status.success() {
            eprintln!("xtask bca: markdown report failed (non-fatal): {status}");
        } else {
            eprintln!("xtask bca: reports written to {}", out_dir.display());
        }
    }

    if full {
        eprintln!("\nxtask bca: --full → running cargo xtask quality");
        let mut c = Command::new("cargo");
        c.args(["xtask", "quality"]).current_dir(&root);
        run_or_fail(&mut c, "cargo xtask quality");
    }
}

fn bca_usage(code: i32) -> ! {
    eprintln!(
        "xtask bca — big-code-analysis helper\n\
         \n\
         USAGE:\n  \
         cargo xtask bca                 # bca check (requires bca in PATH)\n  \
         cargo xtask bca --check         # same as above\n  \
         cargo xtask bca --install       # cargo install big-code-analysis-cli --locked\n  \
         cargo xtask bca --write-baseline# bca check --write-baseline\n  \
         cargo xtask bca --report        # bca report HTML + Markdown to target/bca/\n  \
         cargo xtask bca --init          # install (if needed) + baseline + report\n  \
         cargo xtask bca --full          # --init + cargo xtask quality\n\
         \n\
         Underlying manual sequence (for reference):\n  \
         cargo install big-code-analysis-cli --locked\n  \
         bca check --write-baseline\n  \
         git add .bca-baseline.toml\n  \
         bca report -O html -o target/bca/index.html\n  \
         cargo xtask quality\n\
         \n\
         Note: bca is MPL-2.0 as external binary, does NOT affect Ornis license.\n  \
         Do NOT add as library dependency."
    );
    exit(code);
}
