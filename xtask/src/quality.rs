//! Quality gate: fmt, clippy, tests, supply-chain (audit/deny/outdated/upgrade),
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
    /// Informational stages do not affect the exit code (kept for optional tools).
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

/// Depth flags for the quality gate.
#[derive(Default)]
struct QualityFlags {
    full: bool,
    bench: bool,
    ci: bool,
    everything: bool,
}

impl QualityFlags {
    fn parse(args: &[String]) -> Self {
        let mut f = Self::default();
        for a in args {
            match a.as_str() {
                "--full" => f.full = true,
                "--bench" => f.bench = true,
                "--ci" => f.ci = true,
                "--everything" => f.everything = true,
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
        if f.everything {
            f.full = true;
            f.bench = true;
            f.ci = true;
        }
        f
    }

    /// The total is computed up-front so the stage numbering stays
    /// honest even when a deep stage is skipped (tool not installed).
    /// Level 1 now: fmt, clippy, rustqual, smoke (editor-only), test, test (physics gpu),
    /// clippy (physics gpu), audit, deny, outdated, upgrade-check = 11
    fn total_stages(&self) -> usize {
        11 + usize::from(self.ci) * 2
            + usize::from(self.full) * 2
            + usize::from(self.bench)
            + usize::from(self.everything) * 2
    }
}

/// Running stage counter shared by the per-level runners.
struct StageList<'a> {
    root: &'a std::path::Path,
    total: usize,
    n: usize,
    results: Vec<StageResult>,
}

impl<'a> StageList<'a> {
    fn new(root: &'a std::path::Path, total: usize) -> Self {
        Self {
            root,
            total,
            n: 0,
            results: Vec::new(),
        }
    }

    fn run(&mut self, name: &str, desc: &str, command: Command, informational: bool) {
        self.n += 1;
        let result = run_stage(self.n, self.total, name, desc, command, informational);
        self.results.push(result);
    }

    fn skip(&mut self, name: &str, note: &str) {
        self.n += 1;
        let result = skip_stage(self.n, self.total, name, note);
        self.results.push(result);
    }

    fn cargo(&self, args: &[&str]) -> Command {
        cmd(self.root, "cargo", args)
    }

    fn rustqual(&self) -> Command {
        let mut c = Command::new("rustqual");
        // ponytail: rustqual.toml — единственный source of truth, не дублировать пороги здесь.
        // Ratchet: --compare baseline.json --fail-on-regression --no-fail — падает только при регрессе,
        // иначе baseline содержит 14% Score / 1633 findings и обычный check всегда красный.
        let baseline = self.root.join("baseline.json");
        if baseline.exists() {
            c.args([
                "--compare",
                "baseline.json",
                "--fail-on-regression",
                "--no-fail",
            ]);
        }
        c.current_dir(self.root);
        c
    }
}

/// ── Level 1 (mandatory set) ───────────────────────────────
fn level1(stages: &mut StageList<'_>) {
    stages.run(
        "fmt",
        "cargo fmt --all -- --check",
        stages.cargo(&["fmt", "--all", "--", "--check"]),
        false,
    );

    stages.run(
        "clippy",
        "cargo clippy --workspace --all-targets -- -D warnings",
        stages.cargo(&[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]),
        false,
    );

    // Structural quality gate — MIT (rustqual).
    // Single source of truth: rustqual.toml (no thresholds duplicated here).
    // Ratchet mode: --compare baseline.json --fail-on-regression --no-fail — fails only on regression,
    // new violations are ratcheted via baseline update, not immediate break.
    if binary_exists("rustqual") {
        let desc = if stages.root.join("baseline.json").exists() {
            "rustqual --compare baseline.json --fail-on-regression --no-fail"
        } else {
            "rustqual (no baseline.json — run: rustqual --save-baseline baseline.json)"
        };
        stages.run("rustqual", desc, stages.rustqual(), false);
    } else {
        stages.skip(
            "rustqual",
            "rustqual not installed — structural gate skipped (cargo install rustqual --locked)",
        );
    }

    // Smoke: `cargo run --features editor-only` должен стартовать HTTP на 3420 и не падать.
    // ponytail: без окна/GPU, 15s таймаут (сборка + старт), poll TcpStream — без curl/timeout deps.
    smoke_stage(stages);

    stages.run(
        "test",
        "cargo test --workspace",
        stages.cargo(&["test", "--workspace"]),
        false,
    );

    // Physics GPU solver (feature `gpu`): the shader is generated from Rust
    // via ornis-macros. This stage validates the generated WGSL with naga and
    // runs the solver against the CPU reference on a software adapter
    // (mesa/lavapipe on CI). Device tests skip gracefully without an adapter,
    // so the gate stays green on machines without GPU drivers.
    stages.run(
        "test (physics gpu)",
        "cargo test -p ornis-physics --features gpu",
        stages.cargo(&["test", "-p", "ornis-physics", "--features", "gpu"]),
        false,
    );

    stages.run(
        "clippy (physics gpu)",
        "cargo clippy -p ornis-physics --features gpu --all-targets -- -D warnings",
        stages.cargo(&[
            "clippy",
            "-p",
            "ornis-physics",
            "--features",
            "gpu",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]),
        false,
    );

    stages.run("audit", "cargo audit", stages.cargo(&["audit"]), false);

    stages.run(
        "deny",
        "cargo deny check",
        stages.cargo(&["deny", "check"]),
        false,
    );

    stages.run(
        "outdated",
        "cargo outdated --workspace --exit-code 1 (hard gate: must be latest)",
        stages.cargo(&["outdated", "--workspace", "--exit-code", "1"]),
        false,
    );

    // Hard gate: majors must be latest — cargo upgrade (cargo-edit) dry-run.
    // `cargo upgrade` is the same tool the project uses for `cargo upgrade --incompatible allow`;
    // dry-run prints `old req / latest` table if any crate lags behind latest.
    dependencies_upgrade_stage(stages);
}

fn dependencies_upgrade_stage(stages: &mut StageList<'_>) {
    // cargo-edit's `cargo upgrade --dry-run --incompatible allow` prints a table
    // with `old req` rows when a dependency lags behind latest. Exit code is 0
    // even when outdated, so we inspect stdout/stderr instead of relying on exit.
    stages.n += 1;
    let (idx, total) = (stages.n, stages.total);
    let name = "upgrade-check";
    let desc = "cargo upgrade --dry-run --incompatible allow (must be clean)";
    eprintln!();
    eprintln!("═══ [{idx}/{total}] {name}: {desc} ═══");
    if !cargo_subcommand_exists("upgrade") {
        // cargo-edit not installed — skip with hint instead of failing the gate.
        let hint = "Install:  cargo install cargo-edit --locked";
        eprintln!("── {name}: SKIP (cargo-upgrade not installed — {hint}) ──");
        stages.results.push(StageResult {
            name: name.into(),
            status: Status::Skip,
            note: "cargo-upgrade not installed".into(),
        });
        return;
    }
    let output = Command::new("cargo")
        .args(["upgrade", "--dry-run", "--incompatible", "allow"])
        .current_dir(stages.root)
        .output();
    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
            let combined = format!("{stdout}\n{stderr}");
            // `cargo upgrade` prints a table header `old req` when outdated.
            // When up-to-date it prints only `note: Re-run...` or nothing.
            let has_outdated = combined
                .lines()
                .any(|l| l.trim_start().starts_with("old req") || l.contains("old req"));
            // Fallback: also treat any `Updating` line with `->` as outdated (covers alternative output).
            let has_update_table = combined.contains("Updating") && combined.contains("->");
            if has_outdated || has_update_table || combined.contains("outdated") {
                // Check if the table actually contains data rows (not just header).
                // Heuristic: if output contains a version number like `0.` after header, it's real.
                let outdated = combined
                    .lines()
                    .any(|l| l.contains("0.") && l.contains("->"))
                    || has_outdated;
                if outdated {
                    eprintln!("{combined}");
                    eprintln!("── {name}: FAIL (dependencies not latest — run `cargo upgrade --incompatible allow`) ──");
                    if ci_annotations() {
                        annotate(
                            format!("quality-{name}"),
                            "dependencies not latest — run `cargo upgrade --incompatible allow`",
                        );
                    }
                    stages.results.push(StageResult {
                        name: name.into(),
                        status: Status::Fail,
                        note: "outdated".into(),
                    });
                    return;
                }
            }
            // No table rows → clean.
            eprintln!("── {name}: PASS ──");
            stages.results.push(StageResult {
                name: name.into(),
                status: Status::Pass,
                note: String::new(),
            });
        }
        Err(e) => {
            eprintln!("── {name}: FAIL (spawn error: {e}) ──");
            stages.results.push(StageResult {
                name: name.into(),
                status: Status::Fail,
                note: format!("spawn: {e}"),
            });
        }
    }
}

/// ── Level 2 (--full): coverage + bench compile check ──────
fn full_stages(stages: &mut StageList<'_>) {
    stages.run(
        "coverage (llvm-cov)",
        "cargo llvm-cov --workspace --html --output-dir target/llvm-cov",
        stages.cargo(&[
            "llvm-cov",
            "--workspace",
            "--html",
            "--output-dir",
            "target/llvm-cov",
        ]),
        false,
    );

    stages.run(
        "bench compile-check",
        "cargo bench --workspace --no-run",
        stages.cargo(&["bench", "--workspace", "--no-run"]),
        false,
    );
}

fn bench_stage(stages: &mut StageList<'_>) {
    stages.run(
        "criterion benches",
        "cargo bench --workspace",
        stages.cargo(&["bench", "--workspace"]),
        false,
    );
}

/// ── CI set (--ci): rustdoc + wasm target check ────────────
/// These two stages mirror what the GitHub Actions quality job
/// runs; --ci makes the local gate identical to CI by construction.
fn ci_stages(stages: &mut StageList<'_>) {
    stages.run(
        "doc",
        "cargo doc --workspace --no-deps",
        stages.cargo(&["doc", "--workspace", "--no-deps"]),
        false,
    );

    if wasm_target_installed() {
        stages.run(
            "wasm-check",
            "cargo check -p ornis-wasm --target wasm32-unknown-unknown",
            stages.cargo(&[
                "check",
                "-p",
                "ornis-wasm",
                "--target",
                "wasm32-unknown-unknown",
            ]),
            false,
        );
    } else {
        stages.skip(
            "wasm-check",
            "wasm32-unknown-unknown target not installed:  rustup target add wasm32-unknown-unknown",
        );
    }
}

/// ── Deep static analysis (--everything) ────────────────────
/// Long-running stages, only under the explicit deep flag.
fn deep_stages(stages: &mut StageList<'_>) {
    if cargo_subcommand_exists("mutants") {
        stages.run(
            "mutants (ornis-core)",
            "cargo mutants -p ornis-core --features lock-free --timeout 300",
            stages.cargo(&[
                "mutants",
                "-p",
                "ornis-core",
                "--features",
                "lock-free",
                "--timeout",
                "300",
            ]),
            false,
        );
    } else {
        stages.skip("mutants (ornis-core)", "cargo-mutants not installed");
    }

    if cargo_subcommand_exists("fuzz") && nightly_available() {
        stages.run(
            "fuzz smoke (scene_ron)",
            "cargo +nightly fuzz run scene_ron -- -runs=200",
            stages.cargo(&["+nightly", "fuzz", "run", "scene_ron", "--", "-runs=200"]),
            false,
        );
        stages.run(
            "fuzz smoke (editor_command)",
            "cargo +nightly fuzz run editor_command -- -runs=200",
            stages.cargo(&[
                "+nightly",
                "fuzz",
                "run",
                "editor_command",
                "--",
                "-runs=200",
            ]),
            false,
        );
    } else {
        stages.skip(
            "fuzz smoke (scene_ron)",
            "cargo-fuzz or nightly toolchain missing",
        );
        stages.skip(
            "fuzz smoke (editor_command)",
            "cargo-fuzz or nightly toolchain missing",
        );
    }
}

pub fn quality(args: &[String]) {
    let flags = QualityFlags::parse(args);

    let root = crate::workspace_root();
    let mut stages = StageList::new(&root, flags.total_stages());

    // ── Level 1 (mandatory set) ───────────────────────────────
    level1(&mut stages);

    // ── Level 2 (--full) ──────────────────────────────────────
    if flags.full {
        full_stages(&mut stages);
    }

    if flags.bench {
        bench_stage(&mut stages);
    }

    // ── CI set (--ci) ─────────────────────────────────────────
    if flags.ci {
        ci_stages(&mut stages);
    }

    // ── Deep static analysis (--everything) ───────────────────
    if flags.everything {
        deep_stages(&mut stages);
    }

    print_summary(&stages.results);
    if stages.results.iter().any(|r| r.status == Status::Fail) {
        exit(1);
    }
}

fn quality_usage(code: i32) -> ! {
    eprintln!(
        "xtask quality — the Ornis quality gate\n\
         \n\
         USAGE:\n  \
         cargo xtask quality           quick set (level 1): fmt, clippy, rustqual, smoke, test, audit, deny, outdated, upgrade-check\n  \
         cargo xtask quality --ci      + rustdoc and wasm32 check (same set GitHub Actions runs)\n  \
         cargo xtask quality --full    + coverage (llvm-cov → target/llvm-cov/html) and bench compile-check\n  \
         cargo xtask quality --bench   + full criterion benchmark run (slow)\n  \
         cargo xtask quality --everything\n      \
         everything: --ci + --full + --bench + mutants (ornis-core) + fuzz smoke (slow, minutes to hours)\n\
         \n\
         External tools (audit, deny, outdated, upgrade, llvm-cov, rustqual) are optional:\n  \
         missing → SKIP with install hint. rustqual is MIT.\n  \
         rustqual.toml is the single source of truth (no thresholds duplicated here).\n  \
         Baseline: rustqual --save-baseline baseline.json; CI: rustqual --compare baseline.json --fail-on-regression --no-fail\n  \
         Smoke: cargo run --features editor-only must bind 127.0.0.1:3420 within 90s and stay alive"
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

    // In CI the stage output is captured and re-printed so that a failure
    // can also be surfaced as `::error::` workflow-command annotations
    // (raw run logs live on an endpoint some sandboxes cannot reach; the
    // annotations API is the transport that always works). Locally the
    // stages keep streaming.
    let ran = if ci_annotations() {
        command.output().map(|out| {
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
            // Re-printing child output verbatim re-emits the child's
            // workflow commands (::error …): rustqual floods annotations and
            // crowds out this gate's curated diagnostics. Break the command
            // prefix in the re-print — the gate emits its own annotations.
            print!("{}", stdout.replace("::error", "::·error"));
            eprint!("{}", stderr.replace("::error", "::·error"));
            (out.status, format!("{stdout}{stderr}"))
        })
    } else {
        command.status().map(|status| (status, String::new()))
    };

    match ran {
        Ok((status, _log)) if status.success() => {
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
        Ok((status, log)) => {
            if informational {
                eprintln!("── {name}: INFO (exit {status} — outdated dependencies) ──");
                StageResult {
                    name: name.to_string(),
                    status: Status::Info,
                    note: format!("{status}"),
                }
            } else {
                eprintln!("── {name}: FAIL (exit {status}) ──");
                annotate_stage_failure(name, &log);
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

/// Whether the gate runs inside a GitHub Actions step (workflow commands
/// `::error::…` become check-run annotations there).
fn ci_annotations() -> bool {
    std::env::var_os("GITHUB_ACTIONS").is_some()
}

/// Emits the most relevant error lines of a failed stage as annotations
/// (max 8: cargo/rustc errors, failing tests, fmt diffs, clippy warnings).
fn annotate_stage_failure(name: &str, log: &str) {
    if !ci_annotations() {
        return;
    }
    // CI sets CARGO_TERM_COLOR=always: strip ANSI codes before matching,
    // otherwise colored diagnostics break the prefix checks below.
    let clean = strip_ansi(log);
    let is_match = |l: &str| {
        let t = l.trim_start();
        let lower = t.to_ascii_lowercase();
        t.starts_with("error")
            // only failing test summaries — "ok" results flood the cap
            || (t.starts_with("test result:") && t.contains("FAILED"))
            || t.starts_with("Diff in")
            || t.contains("panicked")
            || t.contains("FAILED")
            // clippy/rustc lowercase vs cargo-audit capitalized "Warning:"
            || t.starts_with("warning:")
            || t.starts_with("Warning:")
            || lower.contains("vulnerab")
            || t.contains("(limit ")
            || t.starts_with('+')
            || t.starts_with('-')
            || t.starts_with("-->")
            || (t.contains("expected") && t.contains("found"))
            || t.starts_with("note:")
    };
    // rustfmt diffs: one annotation per hunk — bodies never fit the cap.
    if name == "fmt" {
        for hunk in fmt_hunks(&clean) {
            annotate(format!("quality-{name}"), &hunk);
        }
        return;
    }
    let mut interesting: Vec<&str> = clean.lines().filter(|l| is_match(l)).collect();
    // GitHub surfaces only ~10 annotations per step: when the diff is large,
    // drop `+`/`-` bodies and keep headers/diagnostics so nothing is hidden.
    if interesting.len() > 15 {
        interesting.retain(|l| {
            let t = l.trim_start();
            !t.starts_with('+') && !t.starts_with('-')
        });
    }
    let start = interesting.len().saturating_sub(40);
    let picked = &interesting[start..];
    if picked.is_empty() {
        annotate(
            format!("quality-{}", name.replace(' ', "-")),
            "stage failed with no recognized error lines — see the raw log",
        );
    }
    for l in picked {
        annotate(format!("quality-{}", name.replace(' ', "-")), l.trim());
    }
}

/// Folds rustfmt output into one message per hunk: "header ⏎ -old ⏎ +new".
fn fmt_hunks(clean: &str) -> Vec<String> {
    let mut hunks: Vec<String> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let flush = |current: &mut Vec<&str>, hunks: &mut Vec<String>| {
        if !current.is_empty() {
            hunks.push(current.join(" ⏎ "));
            current.clear();
        }
    };
    for l in clean.lines() {
        let t = l.trim_start();
        if t.starts_with("Diff in") {
            flush(&mut current, &mut hunks);
            current.push(t);
        } else if (t.starts_with('+') || t.starts_with('-')) && !current.is_empty() {
            current.push(t.trim_end());
        }
    }
    flush(&mut current, &mut hunks);
    hunks
}

/// Removes ANSI escape sequences (SGR and friends) from colored output.
fn strip_ansi(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            match chars.next() {
                Some('[') => {
                    for c2 in chars.by_ref() {
                        // CSI sequences end at the first letter
                        if c2.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
                // two-byte escapes: drop the next char too
                Some(_) => {}
                None => break,
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// One `::error::` workflow command (GitHub Actions annotations).
fn annotate(title: String, message: &str) {
    let esc = |s: &str| -> String {
        s.replace('%', "%25")
            .replace('\r', "%0D")
            .replace('\n', "%0A")
    };
    let mut line = message.trim().to_string();
    if line.len() > 220 {
        // Truncate at a char boundary: `str::truncate` panics mid-UTF-8.
        let mut end = 220;
        while !line.is_char_boundary(end) {
            end -= 1;
        }
        line.truncate(end);
    }
    eprintln!("::error title={}::{}", esc(&title), esc(&line));
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
    if ci_annotations() {
        for r in results.iter().filter(|r| r.status == Status::Fail) {
            let note = if r.note.is_empty() {
                "-".to_string()
            } else {
                r.note.clone()
            };
            annotate(
                format!("quality-summary-{}", r.name.replace(' ', "-")),
                &format!("stage FAIL ({note})"),
            );
        }
    }
}

/// Whether a binary exists in PATH.
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

/// Whether a cargo subcommand exists in PATH (`cargo-<sub>`).
/// Checks for the binary, not `--version`: some tools
/// (e.g. cargo-outdated) do not understand `--version` directly.
fn cargo_subcommand_exists(sub: &str) -> bool {
    binary_exists(&format!("cargo-{sub}"))
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

/// Smoke: `cargo run --features editor-only` must compile, bind 127.0.0.1:3420 and stay alive.
/// ponytail: 90s ceiling — cold CI-cache-miss cargo build dominates (cargo run includes compile).
/// Poll TcpStream every 300ms, no curl/timeout dep; collect child stderr on timeout for diagnostics.
fn smoke_stage(stages: &mut StageList<'_>) {
    use std::net::TcpStream;
    use std::time::{Duration, Instant};
    stages.n += 1;
    let (idx, total) = (stages.n, stages.total);
    let name = "smoke (editor-only)";
    let desc = "cargo run --features editor-only (bind 127.0.0.1:3420, 90s)";
    eprintln!();
    eprintln!("═══ [{idx}/{total}] {name}: {desc} ═══");
    let mut child = match Command::new("cargo")
        .args(["run", "--features", "editor-only"])
        .current_dir(stages.root)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let r = StageResult {
                name: name.into(),
                status: Status::Fail,
                note: format!("spawn: {e}"),
            };
            eprintln!("── {name}: FAIL (spawn: {e}) ──");
            stages.results.push(r);
            return;
        }
    };
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut ok = false;
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().ok().flatten() {
            // Child exited before binding — drain stderr for the real error.
            let stderr = child
                .stderr
                .take()
                .map(|mut s| {
                    use std::io::Read;
                    let mut buf = vec![0u8; 4096];
                    let n = s.read(&mut buf).unwrap_or(0);
                    String::from_utf8_lossy(&buf[..n]).into_owned()
                })
                .unwrap_or_default();
            let note = if stderr.trim().is_empty() {
                format!("exited early: {status}")
            } else {
                let tail = stderr.lines().last().unwrap_or("").trim();
                format!("exited early: {status} — {tail}")
            };
            eprintln!("── {name}: FAIL ({note}) ──");
            stages.results.push(StageResult {
                name: name.into(),
                status: Status::Fail,
                note,
            });
            return;
        }
        if TcpStream::connect("127.0.0.1:3420").is_ok() {
            ok = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    let _ = child.kill();
    let _ = child.wait();
    // Give OS time to release port for next run.
    std::thread::sleep(Duration::from_millis(200));
    if ok {
        eprintln!("── {name}: PASS ──");
        stages.results.push(StageResult {
            name: name.into(),
            status: Status::Pass,
            note: String::new(),
        });
    } else {
        let note = "timeout 90s: 127.0.0.1:3420 not reachable".to_string();
        eprintln!("── {name}: FAIL ({note}) ──");
        if ci_annotations() {
            annotate(format!("quality-{}", name.replace(' ', "-")), &note);
        }
        stages.results.push(StageResult {
            name: name.into(),
            status: Status::Fail,
            note,
        });
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
        "upgrade" => "Install:  cargo install cargo-edit --locked".to_string(),
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
             available targets: scene_ron, materialx_parse, editor_command\n\
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
        .arg("--features")
        .arg("lock-free")
        .arg("--timeout")
        .arg("300")
        .args(&extra)
        .current_dir(&root);
    eprintln!(
        "xtask mutants: cargo mutants -p ornis-core --features lock-free --timeout 300 {}",
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
