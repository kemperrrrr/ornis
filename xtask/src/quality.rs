//! Quality gate: fmt, clippy, tests, supply-chain (audit/deny/outdated),
//! плюс уровень 2 (--full: coverage + bench compile-check; --bench: criterion).
//!
//! Архитектура стадий: каждая стадия печатает заголовок и PASS/FAIL/SKIP/INFO,
//! команда продолжает выполнение после падения стадии и в конце печатает
//! сводную таблицу; exit code — 1, если хотя бы одна стадия FAIL.

use std::path::Path;
use std::process::{exit, Command};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Pass,
    Fail,
    /// Инструмент отсутствует — стадия пропущена (не считается падением).
    Skip,
    /// Информационная стадия (outdated): результат не влияет на exit code.
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

pub fn quality(args: &[String]) {
    let mut full = false;
    let mut bench = false;
    for a in args {
        match a.as_str() {
            "--full" => full = true,
            "--bench" => bench = true,
            "-h" | "--help" => quality_usage(0),
            other => {
                eprintln!("xtask quality: unknown flag '{other}'");
                quality_usage(2);
            }
        }
    }

    let root = crate::workspace_root();
    let mut results: Vec<StageResult> = Vec::new();
    let total = 6 + usize::from(full) * 2 + usize::from(bench);
    let mut n = 0usize;

    // ── Уровень 1 (обязательный набор) ───────────────────────────────

    n += 1;
    results.push(run_stage(
        n,
        total,
        "fmt",
        "cargo fmt --all -- --check",
        cmd(&root, "cargo", &["fmt", "--all", "--", "--check"]),
        false,
    ));

    n += 1;
    results.push(run_stage(
        n,
        total,
        "clippy",
        "cargo clippy --workspace --all-targets -- -D warnings",
        cmd(
            &root,
            "cargo",
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        ),
        false,
    ));

    n += 1;
    results.push(run_stage(
        n,
        total,
        "test",
        "cargo test --workspace",
        cmd(&root, "cargo", &["test", "--workspace"]),
        false,
    ));

    n += 1;
    results.push(run_stage(
        n,
        total,
        "audit",
        "cargo audit",
        cmd(&root, "cargo", &["audit"]),
        false,
    ));

    n += 1;
    results.push(run_stage(
        n,
        total,
        "deny",
        "cargo deny check",
        cmd(&root, "cargo", &["deny", "check"]),
        false,
    ));

    // Информационная стадия: cargo-outdated возвращает ненулевой код,
    // когда есть устаревшие зависимости — это не повод валить гейт.
    n += 1;
    results.push(run_stage(
        n,
        total,
        "outdated (info)",
        "cargo outdated --workspace",
        cmd(&root, "cargo", &["outdated", "--workspace"]),
        true,
    ));

    // ── Уровень 2 (--full) ───────────────────────────────────────────

    if full {
        n += 1;
        results.push(run_stage(
            n,
            total,
            "coverage (llvm-cov)",
            "cargo llvm-cov --workspace --html --output-dir target/llvm-cov",
            cmd(
                &root,
                "cargo",
                &[
                    "llvm-cov",
                    "--workspace",
                    "--html",
                    "--output-dir",
                    "target/llvm-cov",
                ],
            ),
            false,
        ));

        n += 1;
        results.push(run_stage(
            n,
            total,
            "bench compile-check",
            "cargo bench --workspace --no-run",
            cmd(&root, "cargo", &["bench", "--workspace", "--no-run"]),
            false,
        ));
    }

    if bench {
        n += 1;
        results.push(run_stage(
            n,
            total,
            "criterion benches",
            "cargo bench --workspace",
            cmd(&root, "cargo", &["bench", "--workspace"]),
            false,
        ));
    }

    print_summary(&results);
    if results.iter().any(|r| r.status == Status::Fail) {
        exit(1);
    }
}

fn quality_usage(code: i32) -> ! {
    eprintln!(
        "xtask quality — качественный гейт Ornis\n\
         \n\
         USAGE:\n  \
         cargo xtask quality           быстрый набор (уровень 1): fmt, clippy, test, audit, deny, outdated\n  \
         cargo xtask quality --full    + coverage (llvm-cov → target/llvm-cov/html) и bench compile-check\n  \
         cargo xtask quality --bench   + полный прогон criterion-бенчмарков (долго)"
    );
    exit(code);
}

/// Собрать Command с рабочей директорией workspace root.
fn cmd(root: &Path, program: &str, args: &[&str]) -> Command {
    let mut c = Command::new(program);
    c.args(args).current_dir(root);
    c
}

/// Запустить одну стадию: заголовок, проверка инструмента, статус.
/// `informational` — ненулевой exit не считается FAIL (cargo-outdated).
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

    // Проверка наличия внешнего cargo-инструмента с подсказкой по установке.
    // Проверяем только сторонние subcommands: у встроенных (test, bench, …)
    // нет бинаря cargo-<sub>, который можно было бы найти в PATH.
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
                eprintln!("xtask quality: SKIP — инструмент 'cargo {sub}' не найден.\n{hint}");
                return StageResult {
                    name: name.to_string(),
                    status: Status::Skip,
                    note: format!("cargo-{sub} не установлен"),
                };
            }
        }
    }

    match command.status() {
        Ok(status) if status.success() => {
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
        Ok(status) => {
            if informational {
                eprintln!("── {name}: INFO (exit {status} — есть устаревшие зависимости) ──");
                StageResult {
                    name: name.to_string(),
                    status: Status::Info,
                    note: format!("{status}"),
                }
            } else {
                eprintln!("── {name}: FAIL (exit {status}) ──");
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
}

/// Есть ли cargo-subcommand в PATH (`cargo-<sub>`)?
/// Проверяем наличие бинаря, а не `--version`: часть инструментов
/// (например cargo-outdated) не понимает `--version` напрямую.
fn cargo_subcommand_exists(sub: &str) -> bool {
    let binary = if cfg!(windows) {
        format!("cargo-{sub}.exe")
    } else {
        format!("cargo-{sub}")
    };
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(&binary).is_file()))
        .unwrap_or(false)
}

fn install_hint(sub: &str) -> String {
    match sub {
        "audit" => "Установи:  cargo install cargo-audit --locked".to_string(),
        "deny" => "Установи:  cargo install cargo-deny --locked".to_string(),
        "outdated" => "Установи:  cargo install cargo-outdated --locked".to_string(),
        "llvm-cov" => "Установи:  cargo install cargo-llvm-cov --locked\n\
             и компонент:  rustup component add llvm-tools-preview"
            .to_string(),
        "fuzz" => "Установи:  cargo install cargo-fuzz --locked".to_string(),
        "mutants" => "Установи:  cargo install cargo-mutants --locked".to_string(),
        other => format!("Установи:  cargo install cargo-{other} --locked"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// fuzz / mutants — отдельные сабкоманды (не входят в quality default)
// ═══════════════════════════════════════════════════════════════════════

pub fn fuzz(args: &[String]) {
    let Some(target) = args.first() else {
        eprintln!(
            "xtask fuzz — прогон cargo-fuzz таргетов (парсеры внешнего ввода)\n\
             \n\
             USAGE:\n  \
             cargo xtask fuzz <target> [-- <libfuzzer args>]\n  \
             доступные таргеты: scene_ron, materialx_parse\n\
             \n\
             Пример:  cargo xtask fuzz scene_ron -- -runs=1000"
        );
        exit(2);
    };
    let extra: Vec<&str> = args[1..].iter().map(String::as_str).collect();

    if !cargo_subcommand_exists("fuzz") {
        eprintln!(
            "xtask fuzz: 'cargo-fuzz' не найден.\n{}",
            install_hint("fuzz")
        );
        exit(1);
    }
    if !nightly_available() {
        eprintln!(
            "xtask fuzz: nightly toolchain не найден (cargo-fuzz требует nightly).\n\
             Установи:  rustup toolchain install nightly\n\
             (workspace pinned на stable через rust-toolchain.toml — fuzz всегда \
             запускается явно через +nightly)"
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
            "xtask mutants: 'cargo-mutants' не найден.\n{}",
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
        .arg("--timeout")
        .arg("300")
        .args(&extra)
        .current_dir(&root);
    eprintln!(
        "xtask mutants: cargo mutants -p ornis-core --timeout 300 {}",
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
