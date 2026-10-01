//! Read-only CI diagnostics that survive failed/cancelled follow-up steps.
//! Attachments are notices, never substitutes for the strict quality result.

use std::path::Path;
use std::process::Command;

/// Emit complete bounded UTF-8 chunks through the existing annotations API.
/// No tokens or network access are required by the quality process.
pub(super) fn attachment(label: &str, text: &str) {
    emit("notice", label, text);
}

/// Baseline exports use the separate warning budget; keep notice slots for
/// actionable compiler/test failures (GitHub caps each kind per step).
pub(super) fn baseline(text: &str) {
    emit("warning", "rustqual-current", text);
}

/// Max UTF-8 bytes per GitHub Actions annotation chunk.
const ANNOTATION_CHUNK_BYTES: usize = 3_000;
/// Expected length of a git SHA-1 hex revision.
const GIT_SHA1_HEX_LEN: usize = 40;

fn emit(level: &str, label: &str, text: &str) {
    let mut parts = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + ANNOTATION_CHUNK_BYTES).min(text.len());
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
        eprintln!("::{level} title=quality-data-{label}::{}", escaped);
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
        || revision.len() != GIT_SHA1_HEX_LEN
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
    let mut value: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
    if let Some(object) = value.as_object_mut() {
        object.remove("violation_details");
    }
    Ok((revision, value.to_string()))
}

/// Stream both pipes while retaining complete diagnostics. Waiting for an
/// entire cold build before printing hides the last command on runner shutdown.
pub(super) fn run_streamed(
    command: &mut Command,
) -> std::io::Result<(std::process::ExitStatus, String)> {
    use std::process::Stdio;
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("missing stdout pipe"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("missing stderr pipe"))?;
    let out = std::thread::spawn(move || pump(stdout, false));
    let err = std::thread::spawn(move || pump(stderr, true));
    let status = child.wait()?;
    let stdout = out
        .join()
        .map_err(|_| std::io::Error::other("stdout reader panicked"))??;
    let stderr = err
        .join()
        .map_err(|_| std::io::Error::other("stderr reader panicked"))??;
    Ok((status, format!("{stdout}{stderr}")))
}

fn pump(reader: impl std::io::Read, stderr: bool) -> std::io::Result<String> {
    use std::io::BufRead;
    let mut reader = std::io::BufReader::new(reader);
    let mut buffer = Vec::new();
    let mut log = String::new();
    loop {
        buffer.clear();
        if reader.read_until(b'\n', &mut buffer)? == 0 {
            break;
        }
        let line = String::from_utf8_lossy(&buffer);
        let printable = line.replace("::error", "::·error");
        if stderr {
            eprint!("{printable}");
        } else {
            print!("{printable}");
        }
        log.push_str(&line);
    }
    Ok(log)
}
