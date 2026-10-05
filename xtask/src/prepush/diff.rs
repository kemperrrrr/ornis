//! Changed paths for one `git push`, from the pre-push ref list.
//!
//! Each line is `local_ref local_sha remote_ref remote_sha`. A delete
//! (local sha all zeros) contributes nothing. A new branch (remote sha
//! all zeros) is diffed against the merge-base with the remote's
//! `master`, `main`, or `HEAD`; if none of those exist, against commits
//! not yet on any remote.

use std::path::{Path, PathBuf};
use std::process::{exit, Command};

pub(super) struct RefUpdate {
    local_sha: String,
    remote_sha: String,
}

pub(super) fn parse_ref_updates(stdin: &str) -> Vec<RefUpdate> {
    let mut updates = Vec::new();
    for line in stdin.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_ref_line(line) {
            Some(update) => updates.push(update),
            None => {
                eprintln!("pre-push: malformed ref line: {line}");
                exit(1);
            }
        }
    }
    updates
}

fn parse_ref_line(line: &str) -> Option<RefUpdate> {
    let mut parts = line.split_whitespace();
    let _local_ref = parts.next()?;
    let local_sha = parts.next()?.to_string();
    let _remote_ref = parts.next()?;
    let remote_sha = parts.next()?.to_string();
    Some(RefUpdate {
        local_sha,
        remote_sha,
    })
}

fn is_zero_oid(sha: &str) -> bool {
    let sha = sha.trim();
    !sha.is_empty() && sha.bytes().all(|byte| byte == b'0')
}

pub(super) fn changed_paths(root: &Path, remote: &str, update: &RefUpdate) -> Vec<PathBuf> {
    if is_zero_oid(&update.local_sha) {
        return Vec::new();
    }
    if is_zero_oid(&update.remote_sha) {
        return match new_branch_base(root, remote, &update.local_sha) {
            Some(base) => git_diff_paths(root, &base, &update.local_sha),
            None => git_unpushed_paths(root, &update.local_sha),
        };
    }
    git_diff_paths(root, &update.remote_sha, &update.local_sha)
}

fn new_branch_base(root: &Path, remote: &str, local: &str) -> Option<String> {
    let candidates = [
        format!("{remote}/master"),
        format!("{remote}/main"),
        format!("{remote}/HEAD"),
    ];
    for candidate in &candidates {
        if let Some(sha) = merge_base(root, local, candidate) {
            return Some(sha);
        }
    }
    None
}

fn merge_base(root: &Path, a: &str, b: &str) -> Option<String> {
    let output = Command::new("git")
        .args(["merge-base", a, b])
        .current_dir(root)
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if sha.is_empty() {
        None
    } else {
        Some(sha)
    }
}

fn git_diff_paths(root: &Path, from: &str, to: &str) -> Vec<PathBuf> {
    let stdout = git_stdout(
        root,
        &["diff", "-z", "--name-status", "--find-renames", from, to],
        "git diff",
    );
    parse_name_status_z(&stdout)
}

fn git_unpushed_paths(root: &Path, sha: &str) -> Vec<PathBuf> {
    let stdout = git_stdout(
        root,
        &[
            "log",
            "--name-only",
            "--pretty=format:",
            "--find-renames",
            sha,
            "--not",
            "--remotes",
        ],
        "git log",
    );
    String::from_utf8_lossy(&stdout)
        .lines()
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn git_stdout(root: &Path, args: &[&str], what: &str) -> Vec<u8> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .unwrap_or_else(|err| {
            eprintln!("pre-push: failed to spawn {what}: {err}");
            exit(1);
        });
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        eprintln!("pre-push: {what} failed: {err}");
        exit(output.status.code().unwrap_or(1));
    }
    output.stdout
}

/// NUL-separated `--name-status` records. Renames and copies carry two paths.
fn parse_name_status_z(data: &[u8]) -> Vec<PathBuf> {
    let fields: Vec<&[u8]> = data
        .split(|byte| *byte == 0)
        .filter(|field| !field.is_empty())
        .collect();
    let mut paths = Vec::new();
    let mut index = 0;
    while index < fields.len() {
        let kind = fields[index].first().copied().unwrap_or(b'?');
        index += 1;
        let count = if kind == b'R' || kind == b'C' { 2 } else { 1 };
        for _ in 0..count {
            if index >= fields.len() {
                break;
            }
            paths.push(PathBuf::from(
                String::from_utf8_lossy(fields[index]).into_owned(),
            ));
            index += 1;
        }
    }
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Throwaway repo so the diff tests do not need this checkout's history
    /// (CI fetches a shallow clone).
    struct TempRepo {
        path: PathBuf,
    }

    impl TempRepo {
        fn new(label: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!(
                "ornis-prepush-{label}-{}-{nanos}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path)
                .unwrap_or_else(|err| panic!("mkdir {}: {err}", path.display()));
            git(&path, &["init", "-b", "master"]);
            Self { path }
        }

        fn write(&self, rel: &str, body: &str) {
            let path = self.path.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .unwrap_or_else(|err| panic!("mkdir {}: {err}", parent.display()));
            }
            std::fs::write(&path, body)
                .unwrap_or_else(|err| panic!("write {}: {err}", path.display()));
        }

        fn commit(&self, message: &str) -> String {
            git(&self.path, &["add", "."]);
            git(
                &self.path,
                &["-c", "commit.gpgsign=false", "commit", "-m", message],
            );
            git(&self.path, &["rev-parse", "HEAD"])
        }
    }

    impl Drop for TempRepo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn git(dir: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(dir)
            .env("GIT_AUTHOR_NAME", "prepush")
            .env("GIT_AUTHOR_EMAIL", "prepush@example.com")
            .env("GIT_COMMITTER_NAME", "prepush")
            .env("GIT_COMMITTER_EMAIL", "prepush@example.com")
            .output()
            .unwrap_or_else(|err| panic!("git {args:?}: {err}"));
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    #[test]
    fn ref_lines_keep_updates_and_reject_garbage() {
        let updates = parse_ref_updates(
            "\nrefs/heads/master abc refs/heads/master def\n\nrefs/heads/new 111 refs/heads/new 000\n",
        );
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[0].local_sha, "abc");
        assert_eq!(updates[0].remote_sha, "def");
        assert!(is_zero_oid("0000"));
        assert!(is_zero_oid(&"0".repeat(40)));
        assert!(!is_zero_oid("abc"));
        assert!(parse_ref_line("only-three fields here").is_none());
    }

    #[test]
    fn name_status_includes_both_sides_of_a_rename() {
        let data =
            b"M\0crates/physics/src/lib.rs\0R100\0crates/old/src/lib.rs\0crates/physics/src/engine.rs\0";
        let paths = parse_name_status_z(data);
        assert_eq!(
            paths,
            vec![
                PathBuf::from("crates/physics/src/lib.rs"),
                PathBuf::from("crates/old/src/lib.rs"),
                PathBuf::from("crates/physics/src/engine.rs"),
            ]
        );
    }

    #[test]
    fn deleting_a_ref_changes_nothing() {
        let update = RefUpdate {
            local_sha: "0".repeat(40),
            remote_sha: "abc".to_string(),
        };
        assert!(changed_paths(Path::new("."), "origin", &update).is_empty());
    }

    #[test]
    fn one_crate_commit_lists_only_that_file() {
        let repo = TempRepo::new("one-crate");
        repo.write("crates/foo/src/lib.rs", "fn before() {}\n");
        let parent = repo.commit("init");
        repo.write("crates/foo/src/lib.rs", "fn after() {}\n");
        let local = repo.commit("edit foo");
        let paths = changed_paths(
            &repo.path,
            "origin",
            &RefUpdate {
                local_sha: local,
                remote_sha: parent,
            },
        );
        assert_eq!(paths, vec![PathBuf::from("crates/foo/src/lib.rs")]);
    }

    #[test]
    fn new_branch_already_on_origin_master_has_no_diff() {
        let repo = TempRepo::new("on-master");
        repo.write("crates/foo/src/lib.rs", "fn same() {}\n");
        let local = repo.commit("init");
        git(
            &repo.path,
            &["update-ref", "refs/remotes/origin/master", "HEAD"],
        );
        let paths = changed_paths(
            &repo.path,
            "origin",
            &RefUpdate {
                local_sha: local,
                remote_sha: "0".repeat(40),
            },
        );
        assert!(paths.is_empty());
    }
}
