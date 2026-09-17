//! Build the editor smoke binary first, then bound its runtime readiness.
//! File-backed stderr avoids deadlocking cargo/app output on an unread pipe.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub(super) fn check(root: &Path) -> Result<(), String> {
    let build = Command::new("cargo")
        .args(["build", "--features", "editor-only"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("smoke build: {e}"))?;
    if !build.status.success() {
        return Err(format!(
            "smoke build {}:\n{}",
            build.status,
            String::from_utf8_lossy(&build.stderr)
        ));
    }
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .map(|p| if p.is_absolute() { p } else { root.join(p) })
        .unwrap_or_else(|| root.join("target"));
    let binary = target
        .join("debug")
        .join(if cfg!(windows) { "ornis.exe" } else { "ornis" });
    let log_path = target.join("smoke-stderr.log");
    let log = std::fs::File::create(&log_path).map_err(|e| format!("smoke log: {e}"))?;
    let mut child = Command::new(binary)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log))
        .spawn()
        .map_err(|e| format!("smoke spawn: {e}"))?;
    let result = wait_until_ready(&mut child);
    let _ = child.kill();
    let _ = child.wait();
    result.map_err(|reason| {
        let log = std::fs::read_to_string(log_path).unwrap_or_default();
        format!("{reason}\n{log}")
    })
}

fn wait_until_ready(child: &mut std::process::Child) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(90);
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            return Err(format!("smoke exited before readiness: {status}"));
        }
        if std::net::TcpStream::connect("127.0.0.1:3420").is_ok() {
            std::thread::sleep(Duration::from_millis(300));
            return match child.try_wait().map_err(|e| e.to_string())? {
                None => Ok(()),
                Some(status) => Err(format!("smoke exited after binding: {status}")),
            };
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err("timeout 90s: editor did not bind 127.0.0.1:3420".into())
}
