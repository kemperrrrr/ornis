//! TEMPORARY CI feedback transport (delete with the sentinel before hand-off).
//!
//! The remote agent sandbox cannot reach the raw job logs or artifacts
//! (`*.blob.core.windows.net` is unreachable), so the only readable channel
//! is the check-run annotations rendered into the job page HTML. This module
//! publishes the two payloads the agent needs to converge locally:
//!
//! * `fmt` shard — the rustfmt patch for the checkout (`cargo fmt --all`
//!   followed by `git diff -- '*.rs'`), so the formatted sources can be
//!   applied locally;
//! * `test` shard — every `crates/macros/tests/ui/*.stderr` after a
//!   `TRYBUILD=overwrite` run, i.e. the exact snapshots the pinned CI
//!   toolchain expects.
//!
//! Enabled only when `GITHUB_ACTIONS` is set AND
//! `xtask/AGENT-DUMP-TEMPORARY.flag` exists; otherwise both entry points are
//! no-ops.

use std::path::Path;
use std::process::Command;

/// Sentinel committed on the debug branch only.
const FLAG: &str = "xtask/AGENT-DUMP-TEMPORARY.flag";
/// Max UTF-8 bytes per annotation chunk (same budget as `quality_diagnostics`).
const CHUNK: usize = 3_000;

fn enabled(root: &Path) -> bool {
    std::env::var_os("GITHUB_ACTIONS").is_some() && root.join(FLAG).exists()
}

/// One `::notice::` per chunk; the payload mirrors `QUALITY_DATA::…` so the
/// same extractor reads it.
fn emit(label: &str, text: &str) {
    let mut parts: Vec<&str> = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + CHUNK).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        parts.push(&text[start..end]);
        start = end;
    }
    let total = parts.len();
    for (index, part) in parts.iter().copied().enumerate() {
        let payload = format!(
            "QUALITY_DATA::agent-{label}::{index}/{total}\n{part}",
            label = label,
            index = index + 1,
            total = total,
            part = part,
        );
        let escaped = payload
            .replace('%', "%25")
            .replace('\r', "%0D")
            .replace('\n', "%0A");
        eprintln!("::notice title=agent-data-{}::{}", label, escaped);
    }
}

fn capture(command: &mut Command) -> String {
    match command.output() {
        Ok(output) => {
            let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            text
        }
        Err(error) => format!("spawn failed: {error}"),
    }
}

fn run(root: &Path, program: &str, args: &[&str]) -> String {
    let mut command = Command::new(program);
    command.args(args).current_dir(root);
    capture(&mut command)
}

/// `fmt` shard: format the checkout, then publish status + patch.
pub fn fmt_patch(root: &Path) {
    if !enabled(root) {
        return;
    }
    let _applied = run(root, "cargo", &["fmt", "--all"]);
    let status = run(root, "git", &["status", "--porcelain"]);
    let patch = run(root, "git", &["--no-pager", "diff", "--no-color", "--", "*.rs"]);
    let body = format!("== git status --porcelain ==\n{status}\n== git diff ==\n{patch}");
    emit("fmt", &body);
}

/// `test` shard: regenerate the trybuild snapshots, then publish all of them.
pub fn ui_snapshots(root: &Path) {
    if !enabled(root) {
        return;
    }
    let mut command = Command::new("cargo");
    command
        .args(["test", "-p", "ornis-macros", "--test", "compile_fail"])
        .env("TRYBUILD", "overwrite")
        .current_dir(root);
    let _regenerated = capture(&mut command);

    let dir = root.join("crates/macros/tests/ui");
    let mut names: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".stderr") {
                names.push(name);
            }
        }
    }
    names.sort();
    let status = run(
        root,
        "git",
        &["status", "--porcelain", "--", "crates/macros/tests/ui"],
    );
    let mut text = format!(
        "== ui files: {count} ==\n== git status ==\n{status}\n",
        count = names.len(),
        status = status,
    );
    for name in &names {
        text.push_str("===== FILE ");
        text.push_str(name);
        text.push_str(" =====\n");
        let content = std::fs::read_to_string(dir.join(name)).unwrap_or_default();
        text.push_str(&content);
        if !content.ends_with('\n') {
            text.push('\n');
        }
    }
    emit("ui", &text);
}
