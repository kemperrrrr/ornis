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

/// Flags for `cargo xtask bca` after parsing.
#[derive(Debug, Default, PartialEq, Eq)]
struct BcaArgs {
    install: bool,
    write_baseline: bool,
    report: bool,
    full: bool,
    init: bool,
    check: bool,
}

impl BcaArgs {
    /// `--full` and `--init` imply the install → baseline → report sequence.
    /// `--full` additionally runs the whole quality gate at the end.
    fn apply_implied(&mut self) {
        if self.full || self.init {
            self.install = true;
            self.write_baseline = true;
            self.report = true;
        }
    }
}

/// Outcome of parsing `cargo xtask bca` flags.
#[derive(Debug, PartialEq, Eq)]
enum ParseOutcome {
    /// Run the bca sequence with these options.
    Run(BcaArgs),
    /// `-h` / `--help` / `help` was passed — print usage, exit 0.
    Help,
    /// An unknown flag was passed — report it, exit 2.
    Unknown(String),
}

/// Parse `cargo xtask bca` flags. A bare invocation (no args) defaults to
/// `bca check`. Help and unknown flags short-circuit in argument order, so
/// the first offending flag is reported — the same contract the CLI had
/// before parsing was factored out.
fn parse_args(args: &[String]) -> ParseOutcome {
    let mut parsed = BcaArgs::default();
    if args.is_empty() {
        parsed.check = true;
    }
    for a in args {
        match a.as_str() {
            "--install" => parsed.install = true,
            "--write-baseline" | "--baseline" => parsed.write_baseline = true,
            "--report" => parsed.report = true,
            "--full" => parsed.full = true,
            "--init" => parsed.init = true,
            "--check" | "check" => parsed.check = true,
            "-h" | "--help" | "help" => return ParseOutcome::Help,
            other => return ParseOutcome::Unknown(other.to_string()),
        }
    }
    ParseOutcome::Run(parsed)
}

pub fn bca(args: &[String]) {
    let mut parsed = match parse_args(args) {
        ParseOutcome::Run(parsed) => parsed,
        ParseOutcome::Help => bca_usage(0),
        ParseOutcome::Unknown(flag) => {
            eprintln!("xtask bca: unknown flag '{flag}'");
            bca_usage(2);
        }
    };
    parsed.apply_implied();

    // `init` is consumed by `apply_implied`; it needs no handling below.
    let BcaArgs {
        install,
        write_baseline,
        report,
        full,
        check,
        ..
    } = parsed;
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

    if (write_baseline || check) && !binary_exists("bca") {
        eprintln!(
            "xtask bca: `bca` not found in PATH.\n\
             Install: cargo install big-code-analysis-cli --locked\n\
             or: cargo xtask bca --install"
        );
        exit(1);
    }

    if write_baseline {
        eprintln!("xtask bca: bca check --write-baseline");
        let mut c = Command::new("bca");
        c.args(["check", "--write-baseline"]).current_dir(&root);
        run_or_fail(&mut c, "bca check --write-baseline");

        eprintln!("\nxtask bca: baseline written to .bca-baseline.toml");
        eprintln!(
            "  → review diff: bca diff-baseline .bca-baseline.old.toml \
             .bca-baseline.toml (if you saved old)"
        );
        eprintln!(
            "  → then: git add .bca-baseline.toml && git commit -m \
             \"chore(bca): update baseline\""
        );
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
        c.args(["report", "-O", "html", "-o", html.to_str().unwrap()])
            .current_dir(&root);
        run_or_fail(&mut c, "bca report html");

        let md = out_dir.join("report.md");
        eprintln!("xtask bca: bca report -O markdown -o {}", md.display());
        let mut c = Command::new("bca");
        c.args(["report", "-O", "markdown", "-o", md.to_str().unwrap()])
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

#[cfg(test)]
mod tests {
    use super::{parse_args, BcaArgs, ParseOutcome};

    fn argv(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn parsed(list: &[&str]) -> BcaArgs {
        match parse_args(&argv(list)) {
            ParseOutcome::Run(parsed) => parsed,
            other => panic!("expected Run, got {other:?}"),
        }
    }

    fn parsed_resolved(list: &[&str]) -> BcaArgs {
        let mut parsed = parsed(list);
        parsed.apply_implied();
        parsed
    }

    #[test]
    fn bare_invocation_means_check() {
        assert_eq!(
            parsed(&[]),
            BcaArgs {
                check: true,
                ..BcaArgs::default()
            }
        );
    }

    #[test]
    fn check_flag_spellings() {
        for spelling in ["--check", "check"] {
            assert_eq!(
                parsed(&[spelling]),
                BcaArgs {
                    check: true,
                    ..BcaArgs::default()
                }
            );
        }
    }

    #[test]
    fn install_flag() {
        assert_eq!(
            parsed(&["--install"]),
            BcaArgs {
                install: true,
                ..BcaArgs::default()
            }
        );
    }

    #[test]
    fn write_baseline_flag_spellings() {
        for spelling in ["--write-baseline", "--baseline"] {
            assert_eq!(
                parsed(&[spelling]),
                BcaArgs {
                    write_baseline: true,
                    ..BcaArgs::default()
                }
            );
        }
    }

    #[test]
    fn report_flag() {
        assert_eq!(
            parsed(&["--report"]),
            BcaArgs {
                report: true,
                ..BcaArgs::default()
            }
        );
    }

    #[test]
    fn flags_combine() {
        assert_eq!(
            parsed(&["--check", "--install", "--report"]),
            BcaArgs {
                install: true,
                report: true,
                check: true,
                ..BcaArgs::default()
            }
        );
    }

    #[test]
    fn full_implies_install_baseline_report() {
        assert_eq!(
            parsed_resolved(&["--full"]),
            BcaArgs {
                install: true,
                write_baseline: true,
                report: true,
                full: true,
                ..BcaArgs::default()
            }
        );
    }

    #[test]
    fn init_implies_install_baseline_report() {
        assert_eq!(
            parsed_resolved(&["--init"]),
            BcaArgs {
                install: true,
                write_baseline: true,
                report: true,
                init: true,
                ..BcaArgs::default()
            }
        );
    }

    #[test]
    fn init_does_not_imply_full() {
        let parsed = parsed_resolved(&["--init"]);
        assert!(!parsed.full);
        assert!(parsed.init);
    }

    #[test]
    fn plain_check_is_not_escalated() {
        // Without --full/--init the bare check stays a check.
        assert_eq!(
            parsed_resolved(&["--check"]),
            BcaArgs {
                check: true,
                ..BcaArgs::default()
            }
        );
    }

    #[test]
    fn help_flag_spellings_exit_early() {
        for spelling in ["-h", "--help", "help"] {
            assert!(matches!(parse_args(&argv(&[spelling])), ParseOutcome::Help));
        }
    }

    #[test]
    fn unknown_flag_is_reported() {
        assert_eq!(
            parse_args(&argv(&["--bogus"])),
            ParseOutcome::Unknown("--bogus".to_string())
        );
    }

    #[test]
    fn first_offending_flag_wins() {
        assert_eq!(
            parse_args(&argv(&["--bogus", "--help"])),
            ParseOutcome::Unknown("--bogus".to_string())
        );
        assert!(matches!(
            parse_args(&argv(&["--help", "--bogus"])),
            ParseOutcome::Help
        ));
    }
}
