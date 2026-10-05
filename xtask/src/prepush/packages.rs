//! Map changed paths onto workspace packages.
//!
//! Member directories win over the root package, longest path first, so
//! `crates/editor-backend` is not `crates/editor`. The root package owns
//! `src/`, `tests/`, `examples/`, `benches/`, and a top-level `build.rs`.
//! A root manifest, lockfile, `.cargo/` entry, or toolchain file cannot
//! be pinned to one package and selects a workspace check instead.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::exit;

pub(super) enum CheckPlan {
    Skip { reason: &'static str },
    Packages(Vec<String>),
    Workspace { reason: &'static str },
}

pub(super) fn load_package_index(root: &Path) -> PackageIndex {
    let manifest_path = root.join("Cargo.toml");
    let text = read_to_string(&manifest_path);
    let mut members = Vec::new();
    for rel in member_dirs(&text) {
        let cargo_toml = root.join(&rel).join("Cargo.toml");
        let name = package_name(&read_to_string(&cargo_toml)).unwrap_or_else(|| {
            eprintln!("pre-push: no [package] name in {}", cargo_toml.display());
            exit(1);
        });
        members.push(MemberPkg {
            name,
            dir: PathBuf::from(rel),
        });
    }
    PackageIndex {
        root_name: package_name(&text),
        members,
    }
}

pub(super) fn plan_for_paths(index: &PackageIndex, paths: &[PathBuf]) -> CheckPlan {
    let mut packages = BTreeSet::new();
    let mut saw_input_outside = false;
    for path in paths {
        match classify(index, path) {
            Class::Workspace => {
                return CheckPlan::Workspace {
                    reason: "workspace manifest or toolchain changed",
                };
            }
            Class::Package(name) => {
                packages.insert(name.to_string());
            }
            Class::Outside => saw_input_outside = true,
            Class::Ignore => {}
        }
    }
    if !packages.is_empty() {
        return CheckPlan::Packages(packages.into_iter().collect());
    }
    if saw_input_outside {
        CheckPlan::Skip {
            reason: "no workspace packages touched",
        }
    } else {
        CheckPlan::Skip {
            reason: "no Rust changes",
        }
    }
}

pub(super) struct PackageIndex {
    root_name: Option<String>,
    members: Vec<MemberPkg>,
}

struct MemberPkg {
    name: String,
    dir: PathBuf,
}

enum Class<'a> {
    Ignore,
    Outside,
    Package(&'a str),
    Workspace,
}

fn read_to_string(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|err| {
        eprintln!("pre-push: cannot read {}: {err}", path.display());
        exit(1);
    })
}

fn member_dirs(manifest: &str) -> Vec<String> {
    let Some(members_at) = manifest.find("members") else {
        return Vec::new();
    };
    let after_key = &manifest[members_at..];
    let Some(open) = after_key.find('[') else {
        return Vec::new();
    };
    let body = &after_key[open + 1..];
    let Some(close) = body.find(']') else {
        return Vec::new();
    };
    quoted_strings(&body[..close])
}

fn quoted_strings(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find('"') {
        rest = &rest[start + 1..];
        let Some(end) = rest.find('"') else {
            break;
        };
        out.push(rest[..end].to_string());
        rest = &rest[end + 1..];
    }
    out
}

fn package_name(manifest: &str) -> Option<String> {
    let pkg_at = manifest.find("[package]")?;
    let after = &manifest[pkg_at + "[package]".len()..];
    let section = match after.find("\n[") {
        Some(end) => &after[..end],
        None => after,
    };
    for line in section.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("name") else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        let name = rest.trim().trim_matches('"');
        if !name.is_empty() {
            return Some(name.to_string());
        }
    }
    None
}

fn classify<'a>(index: &'a PackageIndex, path: &Path) -> Class<'a> {
    if is_workspace_trigger(path) {
        return Class::Workspace;
    }
    if !is_crate_input(path) {
        return Class::Ignore;
    }
    match index.package_name_for(path) {
        Some(name) => Class::Package(name),
        None => Class::Outside,
    }
}

fn is_workspace_trigger(path: &Path) -> bool {
    if path.starts_with(".cargo") {
        return true;
    }
    match path.file_name().and_then(|name| name.to_str()) {
        Some("Cargo.lock" | "rust-toolchain" | "rust-toolchain.toml") => true,
        Some("Cargo.toml") => path.components().count() == 1,
        _ => false,
    }
}

fn is_crate_input(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "rs")
        || path.file_name().is_some_and(|name| name == "Cargo.toml")
}

impl PackageIndex {
    fn package_name_for<'a>(&'a self, path: &Path) -> Option<&'a str> {
        let mut best: Option<&'a MemberPkg> = None;
        for member in &self.members {
            if !path.starts_with(&member.dir) {
                continue;
            }
            let longer = best.is_none_or(|current| {
                member.dir.components().count() > current.dir.components().count()
            });
            if longer {
                best = Some(member);
            }
        }
        if let Some(member) = best {
            return Some(member.name.as_str());
        }
        if is_root_package_path(path) {
            self.root_name.as_deref()
        } else {
            None
        }
    }
}

fn is_root_package_path(path: &Path) -> bool {
    match path.components().next() {
        Some(component)
            if path.components().count() == 1 && component.as_os_str() == "build.rs" =>
        {
            true
        }
        Some(component) => matches!(
            component.as_os_str().to_str(),
            Some("src" | "tests" | "examples" | "benches")
        ),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> PackageIndex {
        PackageIndex {
            root_name: Some("ornis".to_string()),
            members: vec![
                MemberPkg {
                    name: "ornis-editor".to_string(),
                    dir: PathBuf::from("crates/editor"),
                },
                MemberPkg {
                    name: "editor-backend".to_string(),
                    dir: PathBuf::from("crates/editor-backend"),
                },
                MemberPkg {
                    name: "ornis-physics".to_string(),
                    dir: PathBuf::from("crates/physics"),
                },
            ],
        }
    }

    fn plan(paths: &[&str]) -> CheckPlan {
        let owned: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
        plan_for_paths(&index(), &owned)
    }

    #[test]
    fn docs_only_skip_the_check() {
        match plan(&["README.md", "editor/index.html"]) {
            CheckPlan::Skip { reason } => assert_eq!(reason, "no Rust changes"),
            other => panic!("expected skip, got {}", plan_label(&other)),
        }
    }

    #[test]
    fn one_crate_is_checked_alone() {
        match plan(&["crates/physics/src/lib.rs"]) {
            CheckPlan::Packages(pkgs) => assert_eq!(pkgs, vec!["ornis-physics".to_string()]),
            other => panic!("expected packages, got {}", plan_label(&other)),
        }
    }

    #[test]
    fn editor_backend_is_not_the_editor_crate() {
        match plan(&[
            "crates/editor/src/lib.rs",
            "crates/editor-backend/src/lib.rs",
        ]) {
            CheckPlan::Packages(pkgs) => {
                assert_eq!(
                    pkgs,
                    vec!["editor-backend".to_string(), "ornis-editor".to_string()]
                );
            }
            other => panic!("expected packages, got {}", plan_label(&other)),
        }
    }

    #[test]
    fn root_sources_map_to_the_root_package() {
        match plan(&["src/main.rs", "examples/anim.rs"]) {
            CheckPlan::Packages(pkgs) => assert_eq!(pkgs, vec!["ornis".to_string()]),
            other => panic!("expected packages, got {}", plan_label(&other)),
        }
    }

    #[test]
    fn member_manifest_maps_to_that_package() {
        match plan(&["crates/physics/Cargo.toml"]) {
            CheckPlan::Packages(pkgs) => assert_eq!(pkgs, vec!["ornis-physics".to_string()]),
            other => panic!("expected packages, got {}", plan_label(&other)),
        }
    }

    #[test]
    fn workspace_manifest_and_lock_check_everything() {
        for path in [
            "Cargo.toml",
            "Cargo.lock",
            ".cargo/config.toml",
            "rust-toolchain.toml",
        ] {
            match plan(&["crates/physics/src/lib.rs", path]) {
                CheckPlan::Workspace { .. } => {}
                other => panic!("{path}: expected workspace, got {}", plan_label(&other)),
            }
        }
    }

    #[test]
    fn rust_outside_the_workspace_is_skipped() {
        match plan(&[
            "fuzz/fuzz_targets/scene_ron.rs",
            "third_party/rustqual/src/main.rs",
        ]) {
            CheckPlan::Skip { reason } => assert_eq!(reason, "no workspace packages touched"),
            other => panic!("expected skip, got {}", plan_label(&other)),
        }
    }

    #[test]
    fn outside_rust_does_not_hide_a_touched_crate() {
        match plan(&[
            "fuzz/fuzz_targets/scene_ron.rs",
            "crates/physics/src/lib.rs",
        ]) {
            CheckPlan::Packages(pkgs) => assert_eq!(pkgs, vec!["ornis-physics".to_string()]),
            other => panic!("expected packages, got {}", plan_label(&other)),
        }
    }

    #[test]
    fn manifest_parser_reads_members_and_package_names() {
        let manifest = "\
[workspace]\n\
members = [\n    \"crates/core\",\n    \"xtask\",\n]\n\
[package]\nname = \"ornis\"\n";
        assert_eq!(
            member_dirs(manifest),
            vec!["crates/core".to_string(), "xtask".to_string()]
        );
        assert_eq!(package_name(manifest).as_deref(), Some("ornis"));
        assert_eq!(
            package_name(
                "[workspace.package]\nversion = \"0.1.0\"\n\n[package]\nname = \"ornis-core\"\n"
            )
            .as_deref(),
            Some("ornis-core")
        );
    }

    #[test]
    fn real_workspace_maps_known_paths() {
        let index = load_package_index(&crate::workspace_root());
        assert_eq!(
            index.package_name_for(Path::new("crates/physics/src/lib.rs")),
            Some("ornis-physics")
        );
        assert_eq!(
            index.package_name_for(Path::new("crates/editor-backend/src/lib.rs")),
            Some("editor-backend")
        );
        assert_eq!(
            index.package_name_for(Path::new("crates/editor/src/lib.rs")),
            Some("ornis-editor")
        );
        assert_eq!(
            index.package_name_for(Path::new("xtask/src/main.rs")),
            Some("xtask")
        );
        assert_eq!(
            index.package_name_for(Path::new("src/main.rs")),
            Some("ornis")
        );
        assert_eq!(index.package_name_for(Path::new("fuzz/src/lib.rs")), None);
    }

    fn plan_label(plan: &CheckPlan) -> &'static str {
        match plan {
            CheckPlan::Skip { .. } => "skip",
            CheckPlan::Packages(_) => "packages",
            CheckPlan::Workspace { .. } => "workspace",
        }
    }
}
