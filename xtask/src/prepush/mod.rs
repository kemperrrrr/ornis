//! Pre-push compile check for the commits being pushed.
//!
//! Git feeds one line per ref (`local_ref local_sha remote_ref remote_sha`).
//! The hook diffs each update against the remote sha, maps changed Rust
//! inputs onto workspace packages, and runs `cargo check -p … --all-targets`
//! for that set. `Cargo.lock`, the root manifest, `.cargo/`, and
//! `rust-toolchain.toml` widen the check to the whole workspace — those
//! files can change every crate. Docs-only pushes skip the check.
//! `ORNIS_PREPUSH_FULL=1` always checks the workspace. Clippy and the
//! wasm32 target stay opt-in via `ORNIS_HOOK_FULL=1`; CI runs those itself.

mod diff;
mod packages;

use std::path::Path;
use std::process::{exit, Command};

use crate::{run, workspace_root};
use packages::CheckPlan;

/// Remote name and URL come from the git hook; the ref list is on stdin.
pub fn pre_push(args: &[String]) {
    run_with_stdin(args, &read_stdin());
}

fn run_with_stdin(args: &[String], stdin: &str) {
    let root = workspace_root();
    if env_is_one("ORNIS_PREPUSH_FULL") {
        eprintln!("pre-push: ORNIS_PREPUSH_FULL=1; cargo check --workspace --all-targets");
        run_cargo(&root, &["check", "--workspace", "--all-targets"]);
        run_hook_full(&root);
        finish();
        return;
    }

    let remote = args.first().map(String::as_str).unwrap_or("origin");
    let updates = diff::parse_ref_updates(stdin);
    if updates.is_empty() {
        eprintln!("pre-push: no refs to push; skipping cargo check");
        run_hook_full(&root);
        finish();
        return;
    }

    let index = packages::load_package_index(&root);
    let mut paths = Vec::new();
    for update in &updates {
        paths.extend(diff::changed_paths(&root, remote, update));
    }
    run_plan(&root, &packages::plan_for_paths(&index, &paths));
    run_hook_full(&root);
    finish();
}

fn finish() {
    eprintln!("pre-push: OK");
}

fn env_is_one(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

fn read_stdin() -> String {
    let mut buf = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf).unwrap_or_else(|err| {
        eprintln!("pre-push: failed to read stdin: {err}");
        exit(1);
    });
    buf
}

fn run_cargo(root: &Path, args: &[&str]) {
    let what = format!("cargo {}", args.join(" "));
    let mut cmd = Command::new("cargo");
    cmd.args(args).current_dir(root);
    run(&mut cmd, &what);
}

fn run_hook_full(root: &Path) {
    if !env_is_one("ORNIS_HOOK_FULL") {
        return;
    }
    eprintln!("pre-push [full]: clippy --workspace --all-targets");
    run_cargo(
        root,
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    );
    eprintln!("pre-push [full]: wasm32 target check");
    run_cargo(
        root,
        &[
            "check",
            "-p",
            "ornis-wasm",
            "--target",
            "wasm32-unknown-unknown",
        ],
    );
}

fn run_plan(root: &Path, plan: &CheckPlan) {
    match plan {
        CheckPlan::Skip { reason } => {
            eprintln!("pre-push: {reason}; skipping cargo check");
        }
        CheckPlan::Workspace { reason } => {
            eprintln!("pre-push: {reason}; cargo check --workspace --all-targets");
            run_cargo(root, &["check", "--workspace", "--all-targets"]);
        }
        CheckPlan::Packages(pkgs) => {
            let listed = pkgs
                .iter()
                .map(|pkg| format!("-p {pkg}"))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("pre-push: cargo check {listed} --all-targets");
            let mut args = Vec::with_capacity(pkgs.len() * 2 + 2);
            args.push("check");
            for pkg in pkgs {
                args.push("-p");
                args.push(pkg);
            }
            args.push("--all-targets");
            run_cargo(root, &args);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_flags_are_off_and_hook_full_stays_idle() {
        std::env::remove_var("ORNIS_HOOK_FULL");
        std::env::remove_var("ORNIS_PREPUSH_FULL");
        assert!(!env_is_one("ORNIS_HOOK_FULL"));
        assert!(!env_is_one("ORNIS_PREPUSH_FULL"));
        run_hook_full(&workspace_root());
        run_plan(
            &workspace_root(),
            &CheckPlan::Skip {
                reason: "no Rust changes",
            },
        );
    }

    #[test]
    fn empty_push_skips_without_compiling() {
        std::env::remove_var("ORNIS_PREPUSH_FULL");
        std::env::remove_var("ORNIS_HOOK_FULL");
        run_with_stdin(&["origin".to_string()], "");
        // `pre_push` reads process stdin. Call it only when that stdin is
        // already closed, so an interactive `cargo test` does not block.
        if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
            pre_push(&["origin".to_string()]);
        }
        assert!(!env_is_one("ORNIS_PREPUSH_FULL"));
    }
}
