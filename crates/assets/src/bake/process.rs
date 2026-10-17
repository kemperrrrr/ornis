//! Bounded runs of the reference encoder.
//!
//! `basisu` is an external process. A hung `-version` or encode would stall
//! import, so every spawn has a deadline and is killed when it elapses.
//! On Unix the child starts in its own process group and the deadline kills
//! that group: a wrapper that forks a grandchild (instead of `exec`) still
//! drops the pipes. Elsewhere only the direct child is killed, so a
//! grandchild that inherits a pipe can hold `run_timed` until it exits.
//! Linux can return `ETXTBSY` when a just-written executable is exec'd under
//! parallel fork load; that one error is retried.

use std::ffi::OsString;
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use super::BakeError;

/// How many times to retry an exec that fails because the file is busy.
const SPAWN_ATTEMPTS: u32 = 8;

/// Pause between those retries, and between polls while the child runs.
const POLL_PAUSE: Duration = Duration::from_millis(20);

/// Runs `program` with `args`, killing it when `timeout` elapses.
///
/// Stdout and stderr are drained on another thread so a chatty tool cannot
/// fill the pipe and stall until the deadline.
///
/// # Errors
///
/// [`BakeError::Spawn`] when the process cannot be started.
/// [`BakeError::TimedOut`] when it is still running at the deadline; the
/// child is killed and reaped.
pub(super) fn run_timed(
    program: &Path,
    args: &[OsString],
    timeout: Duration,
) -> Result<Output, BakeError> {
    let mut child = spawn_child(program, args)?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_task = thread::spawn(move || read_pipe(stdout));
    let stderr_task = thread::spawn(move || read_pipe(stderr));
    let status = wait_deadline(program, &mut child, timeout);
    let stdout = joined(stdout_task);
    let stderr = joined(stderr_task);
    status.map(|status| Output {
        status,
        stdout,
        stderr,
    })
}

/// Spawns `program`, retrying a transient text-file-busy error.
fn spawn_child(program: &Path, args: &[OsString]) -> Result<Child, BakeError> {
    let mut last = None;
    for attempt in 1..=SPAWN_ATTEMPTS {
        match try_spawn(program, args) {
            Ok(child) => return Ok(child),
            Err(error) if text_busy(&error) && attempt < SPAWN_ATTEMPTS => {
                last = Some(error);
                thread::sleep(POLL_PAUSE);
            }
            Err(error) => return Err(spawn_error(program, error)),
        }
    }
    Err(spawn_error(program, last.unwrap_or_else(failed_spawn)))
}

/// One `Command` spawn with both pipes captured.
fn try_spawn(program: &Path, args: &[OsString]) -> std::io::Result<Child> {
    let mut command = Command::new(program);
    // Own process group, so a timeout can kill grandchildren with the leader.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}

/// `ETXTBSY`: the kernel refused to exec a file that is busy.
fn text_busy(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::ExecutableFileBusy
}

/// Fallback when the retry loop exits without a recorded error.
fn failed_spawn() -> std::io::Error {
    std::io::Error::other("encoder spawn failed")
}

/// [`BakeError::Spawn`] naming `program`.
fn spawn_error(program: &Path, source: std::io::Error) -> BakeError {
    BakeError::Spawn {
        program: program.display().to_string(),
        source,
    }
}

/// Polls `child` until it exits or `timeout` elapses.
fn wait_deadline(
    program: &Path,
    child: &mut Child,
    timeout: Duration,
) -> Result<ExitStatus, BakeError> {
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) if started.elapsed() >= timeout => {
                return Err(kill_timed_out(program, child, timeout));
            }
            Ok(None) => thread::sleep(POLL_PAUSE),
            Err(source) => return Err(spawn_error(program, source)),
        }
    }
}

/// Kills and reaps `child`, then reports the deadline.
fn kill_timed_out(program: &Path, child: &mut Child, timeout: Duration) -> BakeError {
    #[cfg(unix)]
    kill_group(child);
    let _ = child.kill();
    let _ = child.wait();
    BakeError::TimedOut {
        program: program.display().to_string(),
        millis: timeout.as_millis(),
    }
}

/// Kills the process group led by `child`. A missing `kill` still leaves
/// [`Child::kill`] for the direct child.
#[cfg(unix)]
fn kill_group(child: &Child) {
    let group = format!("-{}", child.id());
    let _ = Command::new("kill")
        .args(["-KILL", "--", &group])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// Reads a pipe to EOF. A missing pipe is empty output.
fn read_pipe(pipe: Option<impl Read>) -> Vec<u8> {
    let Some(mut pipe) = pipe else {
        return Vec::new();
    };
    let mut bytes = Vec::new();
    let _ = pipe.read_to_end(&mut bytes);
    bytes
}

/// Joins a pipe reader. A panicked reader contributes no bytes.
fn joined(handle: thread::JoinHandle<Vec<u8>>) -> Vec<u8> {
    handle.join().unwrap_or_default()
}
