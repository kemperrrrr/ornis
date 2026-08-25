//! End-to-end tests for the bca (big-code-analysis) gate integration.
//!
//! These tests drive the real `bca` binary (external tool, MPL-2.0, used
//! as a subprocess — it never becomes a library dependency). They skip —
//! not fail — when `bca` is not in PATH, mirroring the quality gate's
//! contract for optional external tools.
//!
//! Every test runs in the workspace root so `bca` auto-discovers
//! `bca.toml`, `.bcaignore` and `.bca-baseline.toml` exactly as
//! `cargo xtask quality` does. Nothing here mutates the repository: the
//! only files written go to the system temp directory.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The workspace root: the parent of the xtask crate directory.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent directory")
        .to_path_buf()
}

/// A unique scratch directory under the system temp dir, created on demand.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ornis-bca-test-{name}"));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Whether the `bca` binary is on PATH.
fn bca_available() -> bool {
    let binary = if cfg!(windows) { "bca.exe" } else { "bca" };
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(binary).is_file()))
        .unwrap_or(false)
}

/// Run `bca` in the repo root and return its exit status.
fn run_bca(args: &[&str]) -> std::process::ExitStatus {
    Command::new("bca")
        .args(args)
        .current_dir(repo_root())
        .status()
        .expect("spawn bca")
}

fn assert_file_exists_nonempty(path: &Path) {
    let meta = std::fs::metadata(path).expect("report file exists");
    assert!(
        meta.len() > 0,
        "report file is not empty: {}",
        path.display()
    );
}

#[test]
fn check_passes_against_committed_baseline() {
    if !bca_available() {
        eprintln!("skipped: `bca` not in PATH");
        return;
    }
    // `bca check` auto-discovers bca.toml / .bca-baseline.toml. With the
    // committed baseline every known offender must be filtered and the
    // gate must exit 0 (a config typo or a complexity regression turns
    // this red).
    let status = run_bca(&["check"]);
    assert!(status.success(), "bca check failed: {status}");
}

#[test]
fn empty_baseline_is_green_when_codebase_is_clean() {
    if !bca_available() {
        eprintln!("skipped: `bca` not in PATH");
        return;
    }
    // Since the complexity-debt elimination round the codebase has zero
    // threshold violations, so even an empty baseline must keep the gate
    // green (exit 0). This doubles as a canary: if it starts failing, new
    // code reintroduced a violation that the committed baseline no longer
    // hides — exactly what the ratchet is for.
    let empty = scratch("empty-baseline").join("baseline.toml");
    std::fs::write(&empty, "version = 6\n\n[provenance]\ntier = \"hard\"\n")
        .expect("write empty baseline");
    let status = run_bca(&["check", "--baseline", empty.to_str().expect("utf8 path")]);
    assert_eq!(
        status.code(),
        Some(0),
        "clean codebase must pass with an empty baseline"
    );
}

#[test]
fn write_baseline_produces_version_6_file() {
    if !bca_available() {
        eprintln!("skipped: `bca` not in PATH");
        return;
    }
    // `--write-baseline <path>` must produce a version-6 baseline in the
    // format `bca check --baseline` consumes on the next run. With the
    // codebase clean the entry list is legitimately empty; what matters is
    // the file format and that re-checking against it still exits 0.
    let out = scratch("write-baseline").join("baseline.toml");
    let status = run_bca(&[
        "check",
        "--write-baseline",
        out.to_str().expect("utf8 path"),
    ]);
    assert!(
        status.success(),
        "bca check --write-baseline failed: {status}"
    );
    let text = std::fs::read_to_string(&out).expect("baseline written");
    assert!(text.contains("version = 6"), "baseline must be version 6");
    let recheck = run_bca(&["check", "--baseline", out.to_str().expect("utf8 path")]);
    assert_eq!(
        recheck.code(),
        Some(0),
        "regenerated baseline must keep the gate green"
    );
}

#[test]
fn report_html_and_markdown_generate() {
    if !bca_available() {
        eprintln!("skipped: `bca` not in PATH");
        return;
    }
    // The exact commands `cargo xtask bca --report` runs.
    let dir = scratch("report");
    let html = dir.join("index.html");
    let md = dir.join("report.md");
    let status = run_bca(&["report", "-O", "html", "-o", html.to_str().expect("utf8")]);
    assert!(status.success(), "bca report html failed: {status}");
    let status = run_bca(&["report", "-O", "markdown", "-o", md.to_str().expect("utf8")]);
    assert!(status.success(), "bca report markdown failed: {status}");
    assert_file_exists_nonempty(&html);
    assert_file_exists_nonempty(&md);
}

#[test]
fn config_survives_print_effective_config() {
    if !bca_available() {
        eprintln!("skipped: `bca` not in PATH");
        return;
    }
    // `--print-effective-config` round-trips the manifest: if bca.toml
    // gains a key the installed bca does not understand, the merge either
    // warns or fails — either way this test catches a config/CLI drift.
    let output = Command::new("bca")
        .args(["check", "--print-effective-config"])
        .current_dir(repo_root())
        .output()
        .expect("spawn bca");
    assert!(output.status.success(), "print-effective-config failed");
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        text.contains("[thresholds]"),
        "effective config has thresholds"
    );
    assert!(
        text.contains("cognitive"),
        "effective config carries threshold keys"
    );
}
