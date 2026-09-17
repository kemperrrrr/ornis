//! Read-only CI diagnostics that survive failed/cancelled follow-up steps.
//! Attachments are notices, never substitutes for the strict quality result.

use std::path::Path;
use std::process::Command;

/// Emit complete bounded UTF-8 chunks through the existing annotations API.
/// No tokens or network access are required by the quality process.
pub(super) fn attachment(label: &str, text: &str) {
    let mut parts = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + 20_000).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        parts.push(&text[start..end]);
        start = end;
    }
    for (i, part) in parts.iter().enumerate() {
        let payload = format!("QUALITY_DATA::{label}::{}/{}\n{part}", i + 1, parts.len());
        let escaped = payload
            .replace('%', "%25")
            .replace('\r', "%0D")
            .replace('\n', "%0A");
        eprintln!("::notice title=quality-data-{label}::{}", escaped);
    }
}

/// Independently measure the PR base with the same pinned rustqual binary.
/// This is evidence for a stale-baseline investigation, NOT a new threshold:
/// the original baseline comparison still fails until deliberately resolved.
pub(super) fn reference_rustqual(root: &Path) {
    if std::env::var_os("GITHUB_BASE_REF").is_none() {
        return;
    }
    match measure_reference(root) {
        Ok((revision, json)) => attachment(&format!("rustqual-base-{revision}"), &json),
        Err(error) => eprintln!("quality reference diagnostics unavailable: {error}"),
    }
}

fn measure_reference(root: &Path) -> Result<(String, String), String> {
    let base = format!(
        "origin/{}",
        std::env::var("GITHUB_BASE_REF").map_err(|e| e.to_string())?
    );
    let output = Command::new("git")
        .args(["rev-parse", &base])
        .current_dir(root)
        .output()
        .map_err(|e| e.to_string())?;
    let revision = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !output.status.success()
        || revision.len() != 40
        || !revision.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("base commit is unavailable".into());
    }
    let archive = root.join("target/quality-reference.tar");
    let directory = root.join("target/quality-reference");
    if directory.exists() {
        std::fs::remove_dir_all(&directory).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
    let status = Command::new("git")
        .arg("archive")
        .arg(&revision)
        .arg("--output")
        .arg(&archive)
        .current_dir(root)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("git archive failed".into());
    }
    let status = Command::new("tar")
        .arg("-xf")
        .arg(&archive)
        .arg("-C")
        .arg(&directory)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("base extraction failed".into());
    }
    let result = root.join("target/rustqual_reference.json");
    let output = Command::new("rustqual")
        .arg("--save-baseline")
        .arg(&result)
        .current_dir(&directory)
        .output()
        .map_err(|e| e.to_string())?;
    if !result.exists() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let json = std::fs::read_to_string(&result).map_err(|e| e.to_string())?;
    Ok((revision, json))
}
