//! Quality gate: fmt, clippy, tests, supply-chain (audit/deny/outdated/upgrade),
//! plus level 2 (--full: coverage + bench compile-check; --bench: criterion).
//!
//! Stage architecture: each stage prints a header and PASS/FAIL/SKIP/INFO;
//! the run continues after a failed stage and prints a summary table at
//! the end; the exit code is 1 if any stage FAILs, or — in strict mode
//! (`--ci` or `GITHUB_ACTIONS`) — if any stage is SKIPped without an
//! explicit `--allow-skip` exception.

#[path = "quality_diagnostics.rs"]
mod diagnostics;
#[path = "quality_smoke.rs"]
mod smoke;

use std::path::Path;
use std::process::{exit, Command};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Pass,
    Fail,
    /// The tool is missing — the stage is skipped. Locally this is a
    /// warning; in strict mode (`--ci` / `GITHUB_ACTIONS`) an unexcused
    /// SKIP fails the gate (see `--allow-skip`).
    Skip,
    /// Informational stages do not affect the exit code (kept for optional tools).
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
    /// Canonical stage id (`--list-stages`, `--only`, `--allow-skip`).
    id: String,
    name: String,
    status: Status,
    note: String,
    /// Wall-clock time spent in the stage (zero for instant skips).
    elapsed: Duration,
}

/// Canonical stage ids in gate order. `--only` selects a subset for CI
/// sharding; the default (no `--only`) runs everything implied by the
/// level flags, exactly as before.
const LEVEL1_IDS: [&str; 14] = [
    "fmt",
    "clippy-physics",
    "test-physics",
    "clippy",
    "rustqual",
    "smoke",
    "test",
    "test-lockfree",
    "audit",
    "deny",
    "outdated",
    "upgrade-check",
    "machete",
    "typos",
];
const CI_IDS: [&str; 2] = ["doc", "wasm-check"];
const FULL_IDS: [&str; 3] = ["coverage", "bench-compile", "hack-check"];
const BENCH_IDS: [&str; 1] = ["criterion"];
const DEEP_IDS: [&str; 3] = ["mutants", "fuzz-scene", "fuzz-editor"];

/// Depth flags for the quality gate.
#[derive(Default)]
struct QualityFlags {
    full: bool,
    bench: bool,
    ci: bool,
    everything: bool,
    /// `--only id[,id...]` — run only these stage ids (CI sharding).
    only: Option<Vec<String>>,
    /// `--allow-skip id[,id...]` — stage ids whose SKIP does not fail
    /// strict mode (`--ci` / `GITHUB_ACTIONS`). Every other SKIP fails.
    allow_skip: Vec<String>,
}

impl QualityFlags {
    fn parse(args: &[String]) -> Self {
        let mut f = Self::default();
        let mut i = 0;
        while i < args.len() {
            i = parse_arg(&mut f, args, i);
        }
        // --everything implies all levels: level 2 (coverage + bench
        // compile-check), criterion, the CI set (doc + wasm check) and
        // the deep static-analysis stages (mutants, fuzz smoke).
        if f.everything {
            f.full = true;
            f.bench = true;
            f.ci = true;
        }
        validate_only(&f);
        validate_allow_skip(&f);
        f
    }

    fn all_known_ids() -> Vec<&'static str> {
        let mut v = Vec::new();
        v.extend(LEVEL1_IDS);
        v.extend(CI_IDS);
        v.extend(FULL_IDS);
        v.extend(BENCH_IDS);
        v.extend(DEEP_IDS);
        v
    }

    /// Stages implied by the level flags, in gate order.
    fn active_ids(&self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = LEVEL1_IDS.to_vec();
        if self.ci {
            v.extend(CI_IDS);
        }
        if self.full {
            v.extend(FULL_IDS);
        }
        if self.bench {
            v.extend(BENCH_IDS);
        }
        if self.everything {
            v.extend(DEEP_IDS);
        }
        v
    }

    fn enabled(&self, id: &str) -> bool {
        match &self.only {
            None => true,
            Some(only) => only.iter().any(|s| s == id),
        }
    }

    /// Strict mode: an unexcused SKIP fails the gate. Enabled by `--ci`
    /// or by running under GitHub Actions, so CI shards cannot silently
    /// go green on a skipped stage.
    fn strict(&self) -> bool {
        self.ci || std::env::var_os("GITHUB_ACTIONS").is_some()
    }

    /// Whether a SKIP of this stage id is explicitly excused.
    fn skip_allowed(&self, id: &str) -> bool {
        self.allow_skip.iter().any(|s| s == id)
    }

    /// The total is computed up-front so the stage numbering stays
    /// honest even when a deep stage is skipped (tool not installed).
    /// With `--only` the total is the filtered count.
    fn total_stages(&self) -> usize {
        self.active_ids()
            .iter()
            .filter(|id| self.enabled(id))
            .count()
    }
}

/// Rejects unknown `--only` ids: a typo must fail loudly, never silently
/// run an empty (green) shard.
fn validate_only(f: &QualityFlags) {
    let Some(only) = &f.only else { return };
    let all = QualityFlags::all_known_ids();
    for id in only {
        if !all.contains(&id.as_str()) {
            eprintln!("xtask quality: unknown stage id '{id}' (see --list-stages)");
            quality_usage(2);
        }
    }
}

fn push_only(f: &mut QualityFlags, list: &str) {
    let v = f.only.get_or_insert_with(Vec::new);
    for part in list.split([',', ' ']) {
        let id = part.trim();
        if !id.is_empty() && !v.iter().any(|s| s == id) {
            v.push(id.to_string());
        }
    }
}

/// Parses one flag at `args[i]`; returns the next unconsumed index.
fn parse_arg(f: &mut QualityFlags, args: &[String], i: usize) -> usize {
    let a = args[i].as_str();
    if let Some(list) = a.strip_prefix("--only=") {
        push_only(f, list);
        return i + 1;
    }
    if let Some(list) = a.strip_prefix("--allow-skip=") {
        push_allow_skip(f, list);
        return i + 1;
    }
    if a == "--only" || a == "--allow-skip" {
        let value = require_value(args, i, a);
        if a == "--only" {
            push_only(f, &value);
        } else {
            push_allow_skip(f, &value);
        }
        return i + 2;
    }
    match a {
        "--full" => f.full = true,
        "--bench" => f.bench = true,
        "--ci" => f.ci = true,
        "--everything" => f.everything = true,
        "--list-stages" => print_stages(0),
        "-h" | "--help" => quality_usage(0),
        _ => {
            eprintln!("xtask quality: unknown flag '{a}'");
            quality_usage(2);
        }
    }
    i + 1
}

/// Value of a space-separated `--flag value` pair; exits on a missing value.
fn require_value(args: &[String], i: usize, flag: &str) -> String {
    match args.get(i + 1) {
        Some(value) => value.clone(),
        None => {
            eprintln!("xtask quality: {flag} requires a comma-separated stage list");
            quality_usage(2);
        }
    }
}

/// A typo in `--allow-skip` must fail loudly, never silently excuse
/// nothing (or the wrong stage) and stay green.
fn validate_allow_skip(f: &QualityFlags) {
    let all = QualityFlags::all_known_ids();
    for id in &f.allow_skip {
        if !all.contains(&id.as_str()) {
            eprintln!("xtask quality: unknown stage id '{id}' in --allow-skip (see --list-stages)");
            quality_usage(2);
        }
    }
}

fn push_allow_skip(f: &mut QualityFlags, list: &str) {
    for part in list.split([',', ' ']) {
        let id = part.trim();
        if !id.is_empty() && !f.allow_skip.iter().any(|s| s == id) {
            f.allow_skip.push(id.to_string());
        }
    }
}

fn print_stages(code: i32) -> ! {
    eprintln!("xtask quality stages (gate order):");
    for id in QualityFlags::all_known_ids() {
        eprintln!("  {id}");
    }
    exit(code);
}

/// Running stage counter shared by the per-level runners.
struct StageList<'a> {
    root: &'a std::path::Path,
    total: usize,
    n: usize,
    results: Vec<StageResult>,
    only: Option<Vec<String>>,
    /// Strict mode (`--ci` or `GITHUB_ACTIONS`): guards fail closed
    /// instead of warning when their reference data is unavailable.
    strict: bool,
}

impl<'a> StageList<'a> {
    fn new(
        root: &'a std::path::Path,
        total: usize,
        only: Option<Vec<String>>,
        strict: bool,
    ) -> Self {
        Self {
            root,
            total,
            n: 0,
            results: Vec::new(),
            only,
            strict,
        }
    }

    fn enabled(&self, id: &str) -> bool {
        match &self.only {
            None => true,
            Some(only) => only.iter().any(|s| s == id),
        }
    }

    fn run(&mut self, id: &str, name: &str, desc: &str, command: Command, informational: bool) {
        if !self.enabled(id) {
            return;
        }
        self.n += 1;
        let result = run_stage(self.n, self.total, id, name, desc, command, informational);
        self.results.push(result);
    }

    fn skip(&mut self, id: &str, name: &str, note: &str) {
        if !self.enabled(id) {
            return;
        }
        self.n += 1;
        let result = skip_stage(self.n, self.total, id, name, note);
        self.results.push(result);
    }

    /// Records a custom-stage outcome with its wall-clock time.
    fn push(
        &mut self,
        id: &'static str,
        name: &str,
        status: Status,
        note: String,
        started: Instant,
    ) {
        self.results.push(StageResult {
            id: id.into(),
            name: name.into(),
            status,
            note,
            elapsed: started.elapsed(),
        });
    }

    fn cargo(&self, args: &[&str]) -> Command {
        cmd(self.root, "cargo", args)
    }
}

/// Strict rustqual ratchet comparator (RAT-5).
///
/// The gate compares `baseline.json` against the fresh
/// `rustqual --save-baseline` output and FAILs on ANY regression:
/// lower `quality_score`/`iosp_score`, a higher count in ANY category
/// from [`COUNT_CATEGORIES`], or newly added findings by identity
/// (`file` + `name` from `violation_details`, line ignored as noise).
/// Totals alone are not enough: fixing 5 findings while adding 5 new
/// ones keeps `total_findings` flat but still FAILs via the identity set.
///
/// A missing key or an unparsable report is a hard FAIL (never 0-vs-0
/// PASS): [`compare_rustqual_reports`] returns `Err` with a message
/// naming the offending key/side.
///
/// Improvement (strictly better with no regression) is PASS, but the
/// outcome sets `baseline_stale = true`: the baseline must then be
/// refreshed (`rustqual --save-baseline baseline.json`) and may only
/// move down.
///
/// A committed baseline bump cannot sneak through the same-PR
/// comparison above: [`check_baseline_bump`] additionally compares the
/// committed `baseline.json` against the base branch version
/// (`origin/<base>` on pull_request CI, `origin/master` locally) with
/// this same comparator. Any growth FAILs unless the PR carries the
/// owner-set `baseline-bump-approved` label (plumbed as
/// `ORNIS_BASELINE_BUMP_ALLOWED=1` by the workflow) or the growth is
/// auto-accepted by [`baseline_auto_accept`] (RAT-44: scores up, no new
/// findings, soft-category density non-increasing).
///
/// `RATCHET_EPS` applies to committed `baseline.json` files, which are
/// deterministic: the same tree always serialises the same floats. It
/// does NOT cover fresh-runner float noise (see RAT-37/#93, 4th-decimal
/// drift between machines): the fresh-vs-committed comparison keeps the
/// same epsilon only because both sides are produced by the same
/// vendored binary on similar runners; do not widen it here — widening
/// would silently ratify real regressions in B-vs-C.
const RATCHET_EPS: f64 = 1e-9;

/// Cap on stored identity-diff samples per side (the reason line shows fewer).
const IDENTITY_DIFF_CAP: usize = 11;
/// Findings shown inline in a regression reason.
const REASON_SAMPLE_LEN: usize = 3;
/// New findings printed in the stage log on FAIL.
const LOGGED_NEW_FINDINGS: usize = 5;

/// Count categories under ratchet. Every key must exist in BOTH reports;
/// absence is `Err`, not zero. `total`/`version` are excluded (inventory,
/// not findings); scores are handled as floats separately.
const COUNT_CATEGORIES: &[&str] = &[
    "violations",
    "total_findings",
    "complexity_warnings",
    "magic_number_warnings",
    "nesting_depth_warnings",
    "function_length_warnings",
    "unsafe_warnings",
    "error_handling_warnings",
    "duplicate_groups",
    "dead_code_warnings",
    "dead_type_warnings",
    "fragment_groups",
    "boilerplate_warnings",
    "srp_struct_warnings",
    "srp_module_warnings",
    "wildcard_import_warnings",
    "sdp_violations",
    "coupling_warnings",
    "coupling_cycles",
    "tq_no_assertion_warnings",
    "tq_no_sut_warnings",
    "tq_untested_warnings",
    "tq_uncovered_warnings",
    "tq_untested_logic_warnings",
    "structural_srp_warnings",
    "structural_coupling_warnings",
];

/// Outcome of [`compare_rustqual_reports`] on successfully parsed inputs.
#[derive(Debug)]
struct RatchetOutcome {
    /// Non-empty when the gate must FAIL.
    regressed_reasons: Vec<String>,
    /// True when at least one metric strictly improved and nothing regressed.
    baseline_stale: bool,
    /// Human-readable improvement lines (for the "update baseline" hint).
    improvement_notes: Vec<String>,
    /// Findings identity diff (file, name); empty when details absent.
    added_findings: Vec<FindingId>,
    removed_findings: Vec<FindingId>,
    base_quality: f64,
    cur_quality: f64,
    base_iosp: f64,
    cur_iosp: f64,
}

/// Parse one rustqual report; invalid JSON is an error naming the side.
fn parse_rustqual_report(side: &str, text: &str) -> Result<serde_json::Value, String> {
    serde_json::from_str(text).map_err(|e| format!("{side} rustqual report is not valid JSON: {e}"))
}

/// Require a float key; missing/wrong-typed keys are errors, never zero.
fn require_f64(report: &serde_json::Value, side: &str, key: &str) -> Result<f64, String> {
    report
        .get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("{side} rustqual report: missing or non-numeric key `{key}`"))
}

/// Require a count key; missing/wrong-typed keys are errors, never zero.
fn require_u64(report: &serde_json::Value, side: &str, key: &str) -> Result<u64, String> {
    report
        .get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("{side} rustqual report: missing or non-integer key `{key}`"))
}

/// Identity set of findings as (file, name) pairs.
///
/// Only `Classification::Violation` functions (IOSP violations) carry
/// per-finding identity in the rustqual report — every other category
/// exists as an aggregate count only (see `create_baseline` in the
/// vendored fork: `violation_details` is built from violations alone).
/// The identity check therefore covers IOSP findings; all other
/// categories are ratcheted through their counts.
///
/// # Errors
/// A missing/non-array `violation_details`, or an entry without string
/// `file`/`name`, is a hard FAIL — silently skipping would blind the
/// "fix 5, add 5" swap detector.
fn finding_identity_set(
    report: &serde_json::Value,
    side: &str,
) -> Result<std::collections::BTreeSet<(String, String)>, String> {
    let details = report.get("violation_details").ok_or_else(|| {
        format!("{side} rustqual report: missing `violation_details` (identity check needs it)")
    })?;
    let arr = details
        .as_array()
        .ok_or_else(|| format!("{side} rustqual report: `violation_details` is not an array"))?;
    let mut set = std::collections::BTreeSet::new();
    for (i, item) in arr.iter().enumerate() {
        let file = item.get("file").and_then(|v| v.as_str()).ok_or_else(|| {
            format!("{side} rustqual report: violation_details[{i}] has no string `file`")
        })?;
        let name = item.get("name").and_then(|v| v.as_str()).ok_or_else(|| {
            format!("{side} rustqual report: violation_details[{i}] has no string `name`")
        })?;
        set.insert((file.to_string(), name.to_string()));
    }
    Ok(set)
}

/// Compare one score (higher is better): regression or improvement notes.
fn compare_score(
    base: f64,
    cur: f64,
    name: &str,
    regressed: &mut Vec<String>,
    improved: &mut Vec<String>,
) {
    if cur + RATCHET_EPS < base {
        regressed.push(format!("{name} {base:.4}→{cur:.4}"));
    } else if cur > base + RATCHET_EPS {
        improved.push(format!("{name} {base:.4}→{cur:.4}"));
    }
}

/// Compare every count category; any increase is a regression.
fn compare_counts(
    base: &serde_json::Value,
    cur: &serde_json::Value,
    regressed: &mut Vec<String>,
    improved: &mut Vec<String>,
) -> Result<(), String> {
    for key in COUNT_CATEGORIES {
        let b = require_u64(base, "baseline", key)?;
        let c = require_u64(cur, "current", key)?;
        if c > b {
            regressed.push(format!("{key} {b}→{c}"));
        } else if c < b {
            improved.push(format!("{key} {b}→{c}"));
        }
    }
    Ok(())
}

/// One finding identity: (file, rule name). Line numbers are ignored.
type FindingId = (String, String);

/// Diff findings by identity; any addition is a regression even when
/// totals are flat. Returns (added, removed) samples.
///
/// # Errors
/// Propagates the hard FAIL from [`finding_identity_set`] when either
/// side lacks usable per-finding identity.
fn diff_findings(
    base: &serde_json::Value,
    cur: &serde_json::Value,
    regressed: &mut Vec<String>,
) -> Result<(Vec<FindingId>, Vec<FindingId>), String> {
    let b = finding_identity_set(base, "baseline")?;
    let c = finding_identity_set(cur, "current")?;
    let (mut added, mut removed) = (Vec::new(), Vec::new());
    for f in c.difference(&b).take(IDENTITY_DIFF_CAP) {
        added.push(f.clone());
    }
    for f in b.difference(&c).take(IDENTITY_DIFF_CAP) {
        removed.push(f.clone());
    }
    let added_total = c.difference(&b).count();
    if added_total > 0 {
        let mut sample = added
            .iter()
            .take(REASON_SAMPLE_LEN)
            .map(|(f, n)| format!("{f}::{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        if added_total > added.len() {
            sample.push_str(", …");
        }
        regressed.push(format!("new findings +{added_total} ({sample})"));
    }
    Ok((added, removed))
}

/// Compare two raw rustqual JSON reports.
///
/// # Errors
/// Returns `Err` when either side is not valid JSON or lacks a required
/// key — the caller must treat that as gate FAIL with the message.
fn compare_rustqual_reports(base_s: &str, cur_s: &str) -> Result<RatchetOutcome, String> {
    let base = parse_rustqual_report("baseline", base_s)?;
    let cur = parse_rustqual_report("current", cur_s)?;

    let base_q = require_f64(&base, "baseline", "quality_score")?;
    let cur_q = require_f64(&cur, "current", "quality_score")?;
    let base_iosp = require_f64(&base, "baseline", "iosp_score")?;
    let cur_iosp = require_f64(&cur, "current", "iosp_score")?;

    let mut regressed_reasons = Vec::new();
    let mut improvement_notes = Vec::new();

    compare_score(
        base_q,
        cur_q,
        "quality_score",
        &mut regressed_reasons,
        &mut improvement_notes,
    );
    compare_score(
        base_iosp,
        cur_iosp,
        "iosp_score",
        &mut regressed_reasons,
        &mut improvement_notes,
    );
    compare_counts(&base, &cur, &mut regressed_reasons, &mut improvement_notes)?;

    // Findings identity: catches the "fix 5, add 5" swap that keeps
    // totals flat. Line numbers are ignored (shifts are noise).
    let (added_findings, removed_findings) = diff_findings(&base, &cur, &mut regressed_reasons)?;

    let baseline_stale = regressed_reasons.is_empty() && !improvement_notes.is_empty();
    Ok(RatchetOutcome {
        regressed_reasons,
        baseline_stale,
        improvement_notes,
        added_findings,
        removed_findings,
        base_quality: base_q,
        cur_quality: cur_q,
        base_iosp,
        cur_iosp,
    })
}

/// Hard categories (RAT-44): absolute `C.k <= B.k`, no density allowance.
/// Everything else in [`COUNT_CATEGORIES`] is soft: `C.k <= ceil(B.k *
/// C.total / B.total)` in u128 integer arithmetic, i.e. finding density
/// per unit of inventoried code must not grow.
const HARD_CATEGORIES: &[&str] = &[
    "violations",
    "unsafe_warnings",
    "error_handling_warnings",
    "sdp_violations",
    "coupling_warnings",
    "coupling_cycles",
    "structural_coupling_warnings",
    "tq_no_assertion_warnings",
    "tq_no_sut_warnings",
    "tq_uncovered_warnings",
    "tq_untested_logic_warnings",
    "complexity_warnings",
    "nesting_depth_warnings",
    "wildcard_import_warnings",
    "magic_number_warnings",
];

/// Outcome of [`baseline_auto_accept`]: the committed baseline may move
/// without the owner label.
#[derive(Debug, PartialEq, Eq)]
struct AutoAccept {
    /// True when at least one score strictly improved (beyond [`RATCHET_EPS`]).
    scores_grew: bool,
    /// True when every count category is absolutely non-increasing.
    all_counts_down: bool,
}

/// Integer ceil(B.k * C.total / B.total) without float (u128, no overflow
/// for realistic report sizes). Callers guarantee `base_total > 0`.
fn density_limit(base_k: u64, base_total: u64, cur_total: u64) -> u64 {
    let num = base_k as u128 * cur_total as u128 + base_total as u128 - 1;
    (num / base_total as u128).min(u64::MAX as u128) as u64
}

/// Bound for rule 5 as numerator/denominator: `C.total <= B.total * 3/2`.
/// Named so the ratio is not a magic number at the use site.
const TOTAL_CAP_NUM: u128 = 3;
const TOTAL_CAP_DEN: u128 = 2;

/// Require a float key on both reports at once; missing/wrong-typed
/// keys are errors, never zero (same contract as [`require_f64`]).
fn require_f64_pair(
    base: &serde_json::Value,
    cur: &serde_json::Value,
    key: &str,
) -> Result<(f64, f64), String> {
    Ok((
        require_f64(base, "base", key)?,
        require_f64(cur, "committed", key)?,
    ))
}

/// Require an integer key on both reports at once (same contract as
/// [`require_u64`]).
fn require_u64_pair(
    base: &serde_json::Value,
    cur: &serde_json::Value,
    key: &str,
) -> Result<(u64, u64), String> {
    Ok((
        require_u64(base, "base", key)?,
        require_u64(cur, "committed", key)?,
    ))
}

/// Rule 2: the committed identity set must add no (file,name) pair, and
/// `violations` must not grow. Names the finding in the reason.
fn check_identity_subset(base: &serde_json::Value, cur: &serde_json::Value) -> Result<(), String> {
    let b_ids = finding_identity_set(base, "base")?;
    let c_ids = finding_identity_set(cur, "committed")?;
    if let Some((f, n)) = c_ids.difference(&b_ids).next() {
        let added = c_ids.difference(&b_ids).count();
        return Err(format!(
            "auto-accept rule 2: +{added} new finding(s), e.g. {f}::{n} (new IOSP findings need the label)"
        ));
    }
    let b_v = require_u64(base, "base", "violations")?;
    let c_v = require_u64(cur, "committed", "violations")?;
    if c_v > b_v {
        return Err(format!(
            "auto-accept rule 2: violations {b_v}→{c_v} grew (needs the label)"
        ));
    }
    Ok(())
}

/// Rule 3: every hard category is absolutely non-increasing.
fn check_hard_categories(base: &serde_json::Value, cur: &serde_json::Value) -> Result<(), String> {
    for key in HARD_CATEGORIES {
        let b = require_u64(base, "base", key)?;
        let c = require_u64(cur, "committed", key)?;
        if c > b {
            return Err(format!(
                "auto-accept rule 3: hard category `{key}` {b}→{c} grew (limit {b}, needs the label)"
            ));
        }
    }
    Ok(())
}

/// Rule 4: every soft category stays within its density limit.
/// `base_counts` carries the already-required `(key, B.k)` pairs so the
/// missing-key check happens once in the caller.
fn check_soft_density(
    cur: &serde_json::Value,
    base_counts: &[(&&str, u64)],
    base_total: u64,
    cur_total: u64,
) -> Result<(), String> {
    for (key, b) in base_counts.iter() {
        if HARD_CATEGORIES.contains(key) {
            continue;
        }
        let c = require_u64(cur, "committed", key)?;
        let limit = density_limit(*b, base_total, cur_total);
        if c > limit {
            return Err(format!(
                "auto-accept rule 4: soft category `{key}` {b}→{c} exceeds density limit {limit} (base {b} @ total {base_total}→{cur_total}, needs the label)"
            ));
        }
    }
    Ok(())
}

/// RAT-44 auto-accept: pure function of the two committed reports
/// (base branch version `B`, PR version `C`). No env, no git.
///
/// PASS (`Ok`) iff ALL hold:
/// 1. scores: `C >= B - EPS` on both, and at least one score strictly
///    grew (`> B + EPS`) OR every count category is absolutely `<=`;
/// 2. identity: `violation_details` (file,name) set of C ⊆ B, and
///    `C.violations <= B.violations`;
/// 3. hard categories absolutely `<=`;
/// 4. soft categories within density `ceil(B.k*C.total/B.total)`;
/// 5. structure: both parse, all keys present, `B.total > 0`,
///    `C.total <= B.total * 1.5`.
///
/// # Errors
/// `Err` names the failing rule with `B→C` values and the limit, so the
/// caller can print it as the guard FAIL reason. `Err` means "needs the
/// `baseline-bump-approved` label", never silent PASS.
fn baseline_auto_accept(base_s: &str, cur_s: &str) -> Result<AutoAccept, String> {
    let base = parse_rustqual_report("base", base_s)?;
    let cur = parse_rustqual_report("committed", cur_s)?;
    let (b_q, c_q) = require_f64_pair(&base, &cur, "quality_score")?;
    let (b_iosp, c_iosp) = require_f64_pair(&base, &cur, "iosp_score")?;
    let (b_total, c_total) = require_u64_pair(&base, &cur, "total")?;
    if b_total == 0 {
        return Err("auto-accept rule 5: base `total` is 0".to_string());
    }
    // Anti-inflation cap: C.total <= B.total * NUM/DEN.
    if (c_total as u128) * TOTAL_CAP_DEN > (b_total as u128) * TOTAL_CAP_NUM {
        return Err(format!(
            "auto-accept rule 5: committed `total` {c_total} exceeds 1.5x base {b_total} (inflate guard)"
        ));
    }
    // Rule 1: scores must not drop beyond EPS.
    if c_q + RATCHET_EPS < b_q {
        return Err(format!(
            "auto-accept rule 1: quality_score {b_q:.6}→{c_q:.6} drops beyond EPS"
        ));
    }
    if c_iosp + RATCHET_EPS < b_iosp {
        return Err(format!(
            "auto-accept rule 1: iosp_score {b_iosp:.6}→{c_iosp:.6} drops beyond EPS"
        ));
    }
    let scores_grew = c_q > b_q + RATCHET_EPS || c_iosp > b_iosp + RATCHET_EPS;
    // Eagerly require all count keys present (same `require_*` semantics:
    // absence is Err, never zero) before the rule-1 second half.
    let mut base_counts = Vec::with_capacity(COUNT_CATEGORIES.len());
    let mut cur_counts = Vec::with_capacity(COUNT_CATEGORIES.len());
    for key in COUNT_CATEGORIES {
        base_counts.push((key, require_u64(&base, "base", key)?));
        cur_counts.push((key, require_u64(&cur, "committed", key)?));
    }
    let all_counts_down = cur_counts
        .iter()
        .zip(base_counts.iter())
        .all(|((_, c), (_, b))| c <= b);
    // Rules 2–4 run before the rule-1 growth half so the reason names
    // the finding/category, not just "scores did not grow".
    check_identity_subset(&base, &cur)?;
    check_hard_categories(&base, &cur)?;
    check_soft_density(&cur, &base_counts, b_total, c_total)?;
    // Rule 1, second half: without a strict score gain, every count must
    // be absolutely non-increasing — "equal scores with more findings"
    // needs the label even when density holds.
    if !scores_grew && !all_counts_down {
        return Err(
            "auto-accept rule 1: neither score strictly grew and some count increased (equal scores with more findings need the label)"
                .to_string(),
        );
    }
    Ok(AutoAccept {
        scores_grew,
        all_counts_down,
    })
}

/// RAT-44 rule 6: the fresh run must inventory exactly the committed
/// `total`, otherwise the committed baseline is forged or stale.
/// Returns `Some(FAIL reason)` on mismatch; `None` when equal or when
/// either side is unreadable (the ratchet stage already FAILs those).
fn check_fresh_total_matches_committed(fresh_s: &str, committed_s: &str) -> Option<String> {
    let fresh: serde_json::Value = serde_json::from_str(fresh_s).ok()?;
    let committed: serde_json::Value = serde_json::from_str(committed_s).ok()?;
    let f = fresh.get("total")?.as_u64()?;
    let c = committed.get("total")?.as_u64()?;
    if f != c {
        return Some(format!(
            "fresh rustqual `total` {f} != committed baseline `total` {c} — baseline is stale or edited by hand (обнови baseline: rustqual --save-baseline baseline.json)"
        ));
    }
    None
}

/// One stage-table row.
fn stage_result(name: &str, status: Status, note: String, elapsed: Duration) -> StageResult {
    StageResult {
        id: name.into(),
        name: name.into(),
        status,
        note,
        elapsed,
    }
}

/// Print the comparison table for a parsed ratchet outcome.
fn print_ratchet_comparison(o: &RatchetOutcome) {
    eprintln!();
    eprintln!("═══ Baseline Comparison (xtask ratchet) ═══");
    let q_mark = if o.cur_quality + RATCHET_EPS < o.base_quality {
        format!("(↓ {:.1}%)", (o.base_quality - o.cur_quality) * 100.0)
    } else if o.cur_quality > o.base_quality + RATCHET_EPS {
        format!("(↑ {:.1}%)", (o.cur_quality - o.base_quality) * 100.0)
    } else {
        "(unchanged)".to_string()
    };
    eprintln!(
        "  Quality: {:.1}% → {:.1}% {}",
        o.base_quality * 100.0,
        o.cur_quality * 100.0,
        q_mark
    );
    let iosp_mark = if o
        .regressed_reasons
        .iter()
        .any(|r| r.starts_with("iosp_score"))
    {
        "↓"
    } else {
        ""
    };
    eprintln!(
        "  IOSP: {:.1}% → {:.1}% {}",
        o.base_iosp * 100.0,
        o.cur_iosp * 100.0,
        iosp_mark
    );
    for r in &o.regressed_reasons {
        eprintln!("  regressed: {r}");
    }
    for (f, n) in o.added_findings.iter().take(LOGGED_NEW_FINDINGS) {
        eprintln!("  new finding: {f}::{n}");
    }
}

/// Record PASS/FAIL for a parsed ratchet outcome.
fn record_ratchet_verdict(
    stages: &mut StageList<'_>,
    name: &str,
    cur_s: &str,
    o: RatchetOutcome,
    elapsed: Duration,
) {
    if !o.regressed_reasons.is_empty() {
        if ci_annotations() {
            diagnostics::baseline(cur_s);
            diagnostics::reference_rustqual(stages.root);
        }
        let note = o.regressed_reasons.join(", ");
        eprintln!("── {name}: FAIL (ratchet regression: {note}) ──");
        if ci_annotations() {
            annotate(
                format!("quality-{name}"),
                &format!("ratchet regression: {note}"),
            );
        }
        stages
            .results
            .push(stage_result(name, Status::Fail, note, elapsed));
        return;
    }
    if o.baseline_stale {
        eprintln!(
            "  improved: {} — refresh baseline (`rustqual --save-baseline baseline.json`; values may only decrease)",
            o.improvement_notes.join(", ")
        );
    }
    if !o.removed_findings.is_empty() {
        eprintln!("  resolved findings: −{}", o.removed_findings.len());
    }
    eprintln!("── {name}: PASS (ratchet: equal or improved) ──");
    stages
        .results
        .push(stage_result(name, Status::Pass, String::new(), elapsed));
}

/// Record the ratchet verdict for the rustqual stage: FAIL with reasons
/// on any regression (or on an unreadable/corrupt report), PASS with a
/// baseline-refresh hint when strictly improved.
fn finish_rustqual_stage(
    stages: &mut StageList<'_>,
    name: &str,
    base_s: &str,
    cur_s: &str,
    elapsed: Duration,
) {
    match compare_rustqual_reports(base_s, cur_s) {
        Ok(o) => {
            print_ratchet_comparison(&o);
            record_ratchet_verdict(stages, name, cur_s, o, elapsed);
        }
        Err(msg) => {
            // Corrupt report or missing key: hard FAIL, never 0-vs-0 PASS.
            eprintln!("── {name}: FAIL ({msg}) ──");
            if ci_annotations() {
                annotate(format!("quality-{name}"), &msg);
            }
            stages
                .results
                .push(stage_result(name, Status::Fail, msg, elapsed));
        }
    }
}

/// Deleted-baseline dodge guard (RAT-24): when the base branch has a
/// baseline, its absence in the tree fails closed in strict mode and
/// warns locally. Returns true when the stage result was recorded and
/// the caller must return.
///
/// Takes `guard_ref` explicitly so tests stay hermetic: the only env
/// read (`baseline_guard_ref`) happens once at stage entry, never
/// inside the checked functions (RAT-24 review: env reads inside shared
/// helpers flake under parallel `cargo test`).
fn reject_deleted_baseline(stages: &mut StageList<'_>, guard_ref: &str) -> bool {
    if read_base_baseline(stages.root, guard_ref).is_none() {
        return false;
    }
    let note = "baseline.json deleted while the base branch has one";
    if !stages.strict {
        eprintln!("xtask quality: warning: {note} — plain run, ratchet skipped");
        return false;
    }
    stages.n += 1;
    let (idx, total) = (stages.n, stages.total);
    eprintln!();
    eprintln!("═══ [{idx}/{total}] rustqual: baseline guard ═══");
    eprintln!("── rustqual: FAIL ({note}) ──");
    stages.results.push(stage_result(
        "rustqual",
        Status::Fail,
        note.into(),
        Duration::ZERO,
    ));
    true
}

/// Bump guard verdict (RAT-24): flip the rustqual stage row to FAIL
/// when the committed baseline grew vs the base branch version.
/// `bump_allowed`/`guard_ref` come from the stage entry point, which is
/// the only place that reads the env (see [`baseline_bump_allowed`]).
fn apply_bump_guard(
    stages: &mut StageList<'_>,
    name: &str,
    committed_s: &str,
    bump_allowed: bool,
    guard_ref: &str,
) {
    if let Some(reason) = check_baseline_bump(
        stages.root,
        committed_s,
        stages.strict,
        bump_allowed,
        guard_ref,
    ) {
        eprintln!("── {name}: FAIL ({reason}) ──");
        if ci_annotations() {
            annotate(stage_title(name), &reason);
        }
        if let Some(last) = stages.results.last_mut() {
            last.status = Status::Fail;
            if last.note.is_empty() {
                last.note = reason;
            } else {
                last.note = format!("{}; {reason}", last.note);
            }
        }
    }
}

/// Owner-approved baseline bump plumbing: CI sets
/// `ORNIS_BASELINE_BUMP_ALLOWED=1` only when the PR carries the
/// `baseline-bump-approved` label (see quality.yml — the `fast` job reads
/// the label live from the PR API, so labeling needs no workflow
/// trigger). Local runs never set it: the baseline may only move down.
///
/// The parse is a pure function of the already-read value so tests cover
/// it without touching process env (RAT-24 review of PR #75); the single
/// `std::env::var` read lives in the wrapper below, called once at the
/// `rustqual_stage` entry.
/// RAT-44 rule 6 guard: when the committed baseline was auto-accepted
/// vs the base branch, the fresh run must inventory exactly the
/// committed `total`. Otherwise the committed file is forged or stale.
/// No-op when the base baseline is unreadable, when auto-accept did not
/// apply, or when the stage already FAILED (the ratchet/bump reason
/// stands on its own).
fn apply_fresh_total_guard(
    stages: &mut StageList<'_>,
    name: &str,
    fresh_s: &str,
    committed_s: &str,
    guard_ref: &str,
) {
    if !matches!(stages.results.last().map(|r| r.status), Some(Status::Pass)) {
        return;
    }
    let Some(base_s) = read_base_baseline(stages.root, guard_ref) else {
        return;
    };
    if baseline_auto_accept(&base_s, committed_s).is_err() {
        return;
    }
    if let Some(reason) = check_fresh_total_matches_committed(fresh_s, committed_s) {
        eprintln!("── {name}: FAIL ({reason}) ──");
        if ci_annotations() {
            annotate(stage_title(name), &reason);
        }
        if let Some(last) = stages.results.last_mut() {
            last.status = Status::Fail;
            if last.note.is_empty() {
                last.note = reason;
            } else {
                last.note = format!("{}; {reason}", last.note);
            }
        }
    }
}

fn parse_bump_allowed(raw: Option<&str>) -> bool {
    matches!(raw, Some("1") | Some("true"))
}

fn baseline_bump_allowed() -> bool {
    parse_bump_allowed(std::env::var("ORNIS_BASELINE_BUMP_ALLOWED").ok().as_deref())
}

/// Base ref whose `baseline.json` the committed one must not exceed:
/// `origin/<base>` on pull_request CI runs, `origin/master` otherwise
/// (local runs and push events).
///
/// Pure mapping over the already-read env values so tests can cover
/// every branch without touching process env (RAT-24 review of PR #75:
/// env-reading helpers flake under parallel `cargo test`). The two
/// `std::env::var` reads live inline at the `rustqual_stage` entry.
fn baseline_guard_ref_for(event_name: Option<&str>, base_ref: Option<&str>) -> String {
    if event_name == Some("pull_request") {
        if let Some(base) = base_ref {
            if !base.is_empty() {
                return format!("origin/{base}");
            }
        }
    }
    "origin/master".to_string()
}

/// Read the base branch `baseline.json` at `guard_ref`; `None` when
/// the ref or the file is unavailable (shallow checkout, never fetched,
/// first baseline).
fn read_base_baseline(root: &Path, guard_ref: &str) -> Option<String> {
    let out = Command::new("git")
        .args([
            "-C",
            &root.to_string_lossy(),
            "show",
            &format!("{guard_ref}:baseline.json"),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

/// RAT-24 bump guard (+ RAT-44 auto-accept): compare the committed
/// `baseline.json` against the base branch version. A non-regression
/// (RAT-44 [`baseline_auto_accept`]: scores up, no new findings, soft
/// density non-increasing) holds without the owner label. Anything else
/// is a FAIL unless the bump is owner-approved (`true` only when the
/// workflow saw the `baseline-bump-approved` label — see
/// [`baseline_bump_allowed`]). A missing base baseline fails closed in
/// strict mode and warns locally.
///
/// Both `bump_allowed` and `guard_ref` are parameters, not env reads, so
/// parallel tests cannot race each other through process env (RAT-24
/// review of PR #75).
fn check_baseline_bump(
    root: &Path,
    committed_s: &str,
    strict: bool,
    bump_allowed: bool,
    guard_ref: &str,
) -> Option<String> {
    let Some(base_s) = read_base_baseline(root, guard_ref) else {
        let note = format!("baseline guard: cannot read {guard_ref}:baseline.json");
        if strict {
            return Some(note);
        }
        eprintln!("xtask quality: {note} — bump guard skipped (fetch origin/master to enable)");
        return None;
    };
    match baseline_auto_accept(&base_s, committed_s) {
        Ok(_) => {
            eprintln!(
                "baseline guard: auto-accepted (scores ↑, no new findings, density non-increasing)"
            );
            None
        }
        Err(auto_reason) => {
            if bump_allowed {
                eprintln!("baseline guard: bump allowed by owner label ({auto_reason})");
                return None;
            }
            Some(format!(
                "baseline.json grew vs {guard_ref} ({auto_reason}) — needs `baseline-bump-approved` label"
            ))
        }
    }
}

fn rustqual_stage(stages: &mut StageList<'_>) {
    if !stages.enabled("rustqual") {
        return;
    }
    // The gate runs the vendored Ornis fork (third_party/rustqual): its
    // config uses workspace-aware layer resolution and combinable
    // allowed_in/forbidden_in, which upstream 1.8.2 misreads (452 false
    // positives). Prefer a locally built fork binary, fall back to PATH.
    let rq = rustqual_binary(stages.root);
    if !binary_exists("rustqual") && rq == "rustqual" {
        stages.skip(
            "rustqual",
            "rustqual",
            "rustqual not installed — structural gate skipped (cargo build --manifest-path third_party/rustqual/Cargo.toml)",
        );
        return;
    }
    let baseline_path = stages.root.join("baseline.json");
    // The only env reads for the bump guard live here, at stage entry:
    // helpers below take both values as parameters so parallel tests
    // cannot race through process env (RAT-24 review of PR #75).
    let gh_event = std::env::var("GITHUB_EVENT_NAME").ok();
    let gh_base = std::env::var("GITHUB_BASE_REF").ok();
    let guard_ref = baseline_guard_ref_for(gh_event.as_deref(), gh_base.as_deref());
    let bump_allowed = baseline_bump_allowed();
    if !baseline_path.exists() {
        if reject_deleted_baseline(stages, &guard_ref) {
            return;
        }
        // No baseline — run plain rustqual (findings are informational until baseline is created).
        let mut c = Command::new(&rq);
        c.current_dir(stages.root);
        stages.run(
            "rustqual",
            "rustqual",
            "rustqual (no baseline.json — run: rustqual --save-baseline baseline.json)",
            c,
            false,
        );
        return;
    }
    // Ratchet mode with own comparator — fixes rustqual 1.8.2 equal→FAIL bug (↓0.0% treated as regression).
    // We run `rustqual --save-baseline <tmp>` once, print its text output, and compare the JSON ourselves
    // with epsilon 1e-9. Equal is PASS, only true regression FAILs.
    stages.n += 1;
    let (idx, total) = (stages.n, stages.total);
    let name = "rustqual";
    let desc = "rustqual (ratchet: baseline.json — own comparator, equal=PASS)";
    eprintln!();
    eprintln!("═══ [{idx}/{total}] {name}: {desc} ═══");
    let started = Instant::now();
    let tmp_path = stages.root.join("target/rustqual_cur.json");
    if let Some(parent) = tmp_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let output = Command::new(&rq)
        .args(["--save-baseline", &tmp_path.to_string_lossy()])
        .current_dir(stages.root)
        .output();
    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            // rustqual prints findings to stdout; "Baseline saved" to stderr.
            if ci_annotations() {
                print!("{}", stdout.replace("::error", "::·error"));
                eprint!("{}", stderr.replace("::error", "::·error"));
            } else {
                print!("{}", stdout);
                eprint!("{}", stderr);
            }
            let cur_str = std::fs::read_to_string(&tmp_path);
            let base_str = std::fs::read_to_string(&baseline_path);
            match (cur_str, base_str) {
                (Ok(cur_s), Ok(base_s)) => {
                    finish_rustqual_stage(stages, name, &base_s, &cur_s, started.elapsed());
                    apply_bump_guard(stages, name, &base_s, bump_allowed, &guard_ref);
                    apply_fresh_total_guard(stages, name, &cur_s, &base_s, &guard_ref);
                }
                (Err(e), _) | (_, Err(e)) => {
                    let note = format!("read baseline/cur json: {e}");
                    eprintln!("── {name}: FAIL ({note}) ──");
                    stages.results.push(StageResult {
                        id: "rustqual".into(),
                        name: name.into(),
                        status: Status::Fail,
                        note,
                        elapsed: started.elapsed(),
                    });
                }
            }
        }
        Err(e) => {
            let note = format!("spawn: {e}");
            eprintln!("── {name}: FAIL ({note}) ──");
            stages.results.push(StageResult {
                id: "rustqual".into(),
                name: name.into(),
                status: Status::Fail,
                note,
                elapsed: started.elapsed(),
            });
        }
    }
}

/// ── Level 1 (mandatory set) ───────────────────────────────
fn level1(stages: &mut StageList<'_>) {
    stages.run(
        "fmt",
        "fmt",
        "cargo fmt --all -- --check",
        stages.cargo(&["fmt", "--all", "--", "--check"]),
        false,
    );

    // Physics is the active hardening target: fail diagnostically here
    // before building the rest of the workspace. All original
    // stages still run and retain their strict failure status.
    stages.run(
        "clippy-physics",
        "clippy (physics gpu)",
        "cargo clippy -p ornis-physics --features gpu --all-targets -- -D warnings",
        stages.cargo(&[
            "clippy",
            "-p",
            "ornis-physics",
            "--features",
            "gpu",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]),
        false,
    );

    // Physics GPU solver (feature `gpu`): the shader is generated from Rust
    // via ornis-macros. This stage validates the generated WGSL with naga and
    // runs the solver against the CPU reference on a software adapter
    // (mesa/lavapipe on CI). Device tests skip gracefully without an adapter,
    // so the gate stays green on machines without GPU drivers.
    // Serial only at runtime (`--test-threads=1` for the shared lavapipe
    // device); the build itself stays parallel via CARGO_BUILD_JOBS.
    stages.run(
        "test-physics",
        "test (physics gpu)",
        "cargo test -p ornis-physics --features gpu --no-fail-fast -- --test-threads=1",
        stages.cargo(&[
            "test",
            "-p",
            "ornis-physics",
            "--features",
            "gpu",
            "--no-fail-fast",
            "--",
            "--test-threads=1",
        ]),
        false,
    );

    // No `determinism-fast` stage (removed, RAT-26): it re-ran the
    // default-suite determinism filters (`solver_is_deterministic`,
    // `determinism_snapshot`, `avbd_determinism`) that `test-physics`
    // above already runs with identical flags — pure duplication, second
    // binary run for zero new coverage. Making it "real" via `--ignored`
    // does not fit a PR shard either: the ignored nightly gates need
    // ~7 min each in a debug build (see determinism-nightly.yml), so they
    // stay in the Determinism Nightly workflow.
    stages.run(
        "clippy",
        "clippy",
        "cargo clippy --workspace --all-targets -- -D warnings",
        stages.cargo(&[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]),
        false,
    );

    // Structural quality gate — MIT (rustqual).
    // Single source of truth: rustqual.toml (no thresholds duplicated here).
    // Ratchet: own comparator (equal=PASS, regression=FAIL) — fixes upstream
    // --compare --fail-on-regression --no-fail equal→FAIL bug (↓0.0%).
    rustqual_stage(stages);

    // Smoke: `cargo run --features editor-only` must start HTTP on 3420 and stay up.
    // ponytail: no window/GPU, 15s timeout (build + start), poll TcpStream — no curl/timeout deps.
    smoke_stage(stages);

    stages.run(
        "test",
        "test",
        "cargo test --workspace --no-fail-fast",
        stages.cargo(&["test", "--workspace", "--no-fail-fast"]),
        false,
    );

    // Lock-free lanes (`lock_free_store`, behind the `lock-free` feature)
    // are not compiled by the plain workspace test run above: without this
    // stage a CAS/double-free regression would stay green in CI (RAT-37).
    // Scoped to ornis-core so the extra cost is one more lib build.
    stages.run(
        "test-lockfree",
        "test (lock-free)",
        "cargo test -p ornis-core --features lock-free --no-fail-fast",
        stages.cargo(&[
            "test",
            "-p",
            "ornis-core",
            "--features",
            "lock-free",
            "--no-fail-fast",
        ]),
        false,
    );

    stages.run(
        "audit",
        "audit",
        "cargo audit",
        stages.cargo(&["audit"]),
        false,
    );

    stages.run(
        "deny",
        "deny",
        "cargo deny check",
        stages.cargo(&["deny", "check"]),
        false,
    );

    stages.run(
        "outdated",
        "outdated",
        "cargo outdated --workspace --exit-code 1 (hard gate: must be latest)",
        stages.cargo(&["outdated", "--workspace", "--exit-code", "1"]),
        false,
    );

    // Hard gate: majors must be latest — parsed from
    // `cargo outdated --workspace --format json` (see
    // dependencies_upgrade_stage); lagging majors are fixed with
    // `cargo upgrade --incompatible allow`.
    dependencies_upgrade_stage(stages);

    // Unused-dependency gate (cargo-machete, pinned 0.9.2): FAILs on any
    // dependency no Rust target uses. False positives are allow-listed in
    // the owning crate via `[package.metadata.cargo-machete] ignored`,
    // never by deleting the stage.
    if stages.enabled("machete") {
        if binary_exists("cargo-machete") {
            // Direct binary, never `cargo machete`: cargo's
            // external-subcommand forwarding behaves differently for a
            // `cargo` nested under `cargo run` (the gate's own
            // `cargo xtask` invocation), and the bare binary is immune.
            stages.run(
                "machete",
                "machete",
                "cargo-machete",
                cmd(stages.root, "cargo-machete", &[]),
                false,
            );
        } else {
            stages.skip(
                "machete",
                "machete",
                &format!("cargo-machete not installed — Install:  cargo install cargo-machete --version {CARGO_MACHETE_VERSION} --locked"),
            );
        }
    }

    // Spell-check gate (typos, pinned 1.51.1, config `_typos.toml`):
    // domain terms live in `[default.extend-words]`, vendored code and
    // canonical snapshots in `[files] extend-exclude`.
    if stages.enabled("typos") {
        if binary_exists("typos") {
            stages.run(
                "typos",
                "typos",
                "typos",
                cmd(stages.root, "typos", &[]),
                false,
            );
        } else {
            stages.skip(
                "typos",
                "typos",
                &format!("typos not installed — Install:  cargo install typos-cli --version {TYPOS_VERSION} --locked"),
            );
        }
    }
}

fn dependencies_upgrade_stage(stages: &mut StageList<'_>) {
    // Hard gate: majors must be latest. Implemented on top of
    // `cargo outdated --workspace --format json` (same tool the `outdated`
    // stage already requires — no cargo-edit needed): each JSON line is one
    // workspace member (`{crate_name, dependencies: [{name, project,
    // compat, latest, ...}]}`), listing only deps that lag behind. A dep
    // counts as major-lag when `latest` is newer than the semver-compatible
    // version — i.e. `cargo update` alone cannot reach it and `cargo
    // upgrade` would have to bump the requirement. Minor-only lag
    // (`latest == compat`) is the `outdated` stage's job and passes here.
    if !stages.enabled("upgrade-check") {
        return;
    }
    stages.n += 1;
    let (idx, total) = (stages.n, stages.total);
    let name = "upgrade-check";
    let desc = "cargo outdated --workspace --format json (majors must be latest)";
    eprintln!();
    eprintln!("═══ [{idx}/{total}] {name}: {desc} ═══");
    let started = Instant::now();
    if !cargo_subcommand_exists("outdated") {
        let hint = install_hint("outdated");
        eprintln!("── {name}: SKIP (cargo-outdated not installed — {hint}) ──");
        stages.push(
            "upgrade-check",
            name,
            Status::Skip,
            "cargo-outdated not installed".into(),
            started,
        );
        return;
    }
    let output = Command::new("cargo")
        .args(["outdated", "--workspace", "--format", "json"])
        .current_dir(stages.root)
        .output();
    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
            let lagging = parse_outdated_major_lag(&stdout);
            if !lagging.is_empty() {
                // Cap the printed list; the count stays exact.
                const SHOWN: usize = 10;
                for entry in lagging.iter().take(SHOWN) {
                    eprintln!("  {entry}");
                }
                if lagging.len() > SHOWN {
                    eprintln!("  … +{} more", lagging.len() - SHOWN);
                }
                let note = format!("{} deps not latest", lagging.len());
                eprintln!("── {name}: FAIL ({note} — run `cargo upgrade --incompatible allow`) ──");
                if ci_annotations() {
                    annotate(
                        stage_title(name),
                        &format!("{note} — run `cargo upgrade --incompatible allow`"),
                    );
                }
                stages.push("upgrade-check", name, Status::Fail, note, started);
                return;
            }
            if !out.status.success() {
                // No parseable major lag but cargo-outdated itself errored
                // (network, lockfile): fail loudly, never silently green.
                let tail = stderr
                    .lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("cargo outdated failed");
                let note = format!("cargo outdated failed: {tail}");
                eprintln!("── {name}: FAIL ({note}) ──");
                if ci_annotations() {
                    annotate(stage_title(name), &note);
                }
                stages.push("upgrade-check", name, Status::Fail, note, started);
                return;
            }
            // No lagging deps → clean.
            eprintln!("── {name}: PASS ──");
            stages.push("upgrade-check", name, Status::Pass, String::new(), started);
        }
        Err(e) => {
            eprintln!("── {name}: FAIL (spawn error: {e}) ──");
            stages.push(
                "upgrade-check",
                name,
                Status::Fail,
                format!("spawn: {e}"),
                started,
            );
        }
    }
}

/// Parses `cargo outdated --format json` output (one JSON object per line,
/// one per workspace member) into `name project->latest` entries for every
/// dependency with major lag: `latest` names a version newer than both the
/// locked `project` one and the semver-compatible one, i.e. `cargo update`
/// alone cannot reach it and `cargo upgrade` would have to bump the
/// requirement. Minor-only lag (`latest == compat`) belongs to the
/// `outdated` stage and passes here. Unchanged deps print as `"---"` and
/// removed ones as `"Removed"`; a `"Removed"` marker is reported as lag so
/// a vanished upstream version fails loudly instead of silently passing.
fn parse_outdated_major_lag(stdout: &str) -> Vec<String> {
    let mut lagging = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let empty = Vec::new();
        let deps = v
            .get("dependencies")
            .and_then(|d| d.as_array())
            .unwrap_or(&empty);
        for dep in deps {
            let name = dep.get("name").and_then(|n| n.as_str()).unwrap_or("?");
            let project = dep.get("project").and_then(|p| p.as_str()).unwrap_or("");
            let compat = dep.get("compat").and_then(|c| c.as_str()).unwrap_or("");
            let latest = dep.get("latest").and_then(|l| l.as_str()).unwrap_or("");
            if latest == "Removed" || compat == "Removed" {
                lagging.push(format!("{name} {project}->Removed"));
            } else if is_version_string(latest) && latest != compat {
                lagging.push(format!("{name} {project}->{latest}"));
            }
        }
    }
    lagging.sort();
    lagging.dedup();
    lagging
}

/// Structural version check (`1.2.3`, `0.31`, `2.0.0-alpha.1`): at least
/// one dot-separated numeric component. Rejects the `"---"` (unchanged)
/// and `"Removed"` markers cargo-outdated prints for clean deps.
fn is_version_string(s: &str) -> bool {
    let core = s.split('-').next().unwrap_or("");
    let mut parts = core.split('.');
    match (parts.next(), parts.next()) {
        (Some(major), Some(_minor)) => {
            !major.is_empty()
                && major.chars().all(|c| c.is_ascii_digit())
                && core.split('.').all(|p| {
                    let p = p.trim();
                    !p.is_empty() && p.chars().all(|c| c.is_ascii_digit())
                })
        }
        _ => false,
    }
}
/// Pinned third-party gate tools (single source of truth for the version
/// strings; CI mirrors them via `taiki-e/install-action` `@version` pins).
const CARGO_HACK_VERSION: &str = "0.6.45";
const CARGO_MACHETE_VERSION: &str = "0.9.2";
const TYPOS_VERSION: &str = "1.51.1";

/// ── Level 2 (--full): coverage + bench compile check ──────
fn full_stages(stages: &mut StageList<'_>) {
    coverage_stage(stages);

    stages.run(
        "bench-compile",
        "bench compile-check",
        "cargo bench --workspace --no-run",
        stages.cargo(&["bench", "--workspace", "--no-run"]),
        false,
    );

    // Feature-matrix gate (cargo-hack, pinned 0.6.45): every feature of
    // every workspace crate must check without dev-deps (catches
    // `#[cfg(feature)]` code that only compiles under default features).
    // Slow (one `cargo check` per feature) — `--full` / nightly only.
    // Direct `cargo-hack` binary (same nested-dispatch reason as above;
    // the binary takes the `hack` subcommand explicitly).
    if stages.enabled("hack-check") {
        if binary_exists("cargo-hack") {
            stages.run(
                "hack-check",
                "hack (feature matrix)",
                "cargo hack check --workspace --each-feature --no-dev-deps",
                cmd(
                    stages.root,
                    "cargo-hack",
                    &[
                        "hack",
                        "check",
                        "--workspace",
                        "--each-feature",
                        "--no-dev-deps",
                    ],
                ),
                false,
            );
        } else {
            stages.skip(
                "hack-check",
                "hack (feature matrix)",
                &format!("cargo-hack not installed — Install:  cargo install cargo-hack --version {CARGO_HACK_VERSION} --locked"),
            );
        }
    }
}

/// Coverage ratchet gate (same idea as rustqual): the workspace line
/// rate in `coverage-baseline.json` is the floor — within ±0.1 pp is PASS,
/// only a deviation beyond the band matters (below FAILs, above rewrites
/// the file — commit it). First run without a baseline saves the current
/// rate and PASSes. Without cargo-llvm-cov the stage SKIPs (same as before).
/// The JSON summary stays at `target/llvm-cov/coverage-summary.json`, the
/// HTML report at `target/llvm-cov/html` — per-crate detail lives there and
/// in the nightly artifacts, not in the gate output.
fn coverage_stage(stages: &mut StageList<'_>) {
    if !stages.enabled("coverage") {
        return;
    }
    if !cargo_subcommand_exists("llvm-cov") {
        stages.skip(
            "coverage",
            "coverage (llvm-cov)",
            "cargo-llvm-cov not installed (cargo install cargo-llvm-cov --locked; rustup component add llvm-tools-preview)",
        );
        return;
    }
    stages.n += 1;
    let started = Instant::now();
    let (idx, total) = (stages.n, stages.total);
    let name = "coverage (llvm-cov)";
    eprintln!();
    eprintln!("═══ [{idx}/{total}] {name}: cargo llvm-cov --workspace (ratchet vs {COVERAGE_BASELINE}) ═══");
    let summary_path = stages.root.join(COVERAGE_SUMMARY);
    if let Some(parent) = summary_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // One test run, machine-readable summary. The HTML report renders
    // afterwards from the same data (no test re-run).
    let tested = Command::new("cargo")
        .args([
            "llvm-cov",
            "--workspace",
            "--summary-only",
            "--json",
            "--output-path",
            &summary_path.to_string_lossy(),
        ])
        .current_dir(stages.root)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    let (status, note) = coverage_outcome(stages.root, &summary_path, tested);
    eprintln!("── {name}: {} ({note}) ──", status.label());
    stages.push("coverage", name, status, note, started);
}

/// Verdict for a finished coverage run: reads the llvm-cov summary,
/// renders the HTML report from the same data, and rates the workspace
/// line coverage against the stored floor.
fn coverage_outcome(root: &Path, summary_path: &Path, tested: bool) -> (Status, String) {
    if !tested {
        return (
            Status::Fail,
            "cargo llvm-cov --workspace did not finish (tests or collection broke)".to_string(),
        );
    }
    let Ok(text) = std::fs::read_to_string(summary_path) else {
        return (
            Status::Fail,
            "coverage summary unreadable (report step broke)".to_string(),
        );
    };
    let (rate, covered, count) = coverage_rate(&text);
    if count == 0 {
        return (
            Status::Fail,
            "coverage summary reports zero total lines (collection broke)".to_string(),
        );
    }
    let _ = Command::new("cargo")
        .args([
            "llvm-cov",
            "report",
            "--html",
            "--output-dir",
            COVERAGE_HTML,
        ])
        .current_dir(root)
        .status();
    eprintln!(
        "workspace lines: {covered}/{count} = {:.2}%",
        rate * COVERAGE_PERCENT
    );
    coverage_verdict(&root.join(COVERAGE_BASELINE), rate, covered, count)
}

/// Workspace line-coverage floor file (ratchet baseline, same role as
/// rustqual's `baseline.json`): FAIL only on true regression.
const COVERAGE_BASELINE: &str = "coverage-baseline.json";
/// Machine-readable llvm-cov summary produced by the gate run.
const COVERAGE_SUMMARY: &str = "target/llvm-cov/coverage-summary.json";
/// Human-readable report rendered from the same data (no test re-run).
const COVERAGE_HTML: &str = "target/llvm-cov/html";
/// Tolerance band for the ratchet comparison, in fractions (0.001 = 0.1 pp).
/// Coverage is not fully deterministic across machines: GPU tests depend on
/// the adapter and `cfg(target_os)` changes the instrumented line count, so
/// a sub-0.1pp wobble around the floor is noise, not a regression. Only a
/// deviation beyond the band FAILs (below) or rewrites the file (above).
const COVERAGE_EPS: f64 = 1e-3;
/// Factor for human-readable percent printing.
const COVERAGE_PERCENT: f64 = 100.0;

/// Workspace `(line_rate, covered, total)` from an llvm-cov
/// `--summary-only --json` report (`data[0].totals.lines`).
fn coverage_rate(summary: &str) -> (f64, u64, u64) {
    let report: serde_json::Value =
        serde_json::from_str(summary).unwrap_or(serde_json::Value::Null);
    let lines = report
        .get("data")
        .and_then(|d| d.get(0))
        .and_then(|d| d.get("totals"))
        .and_then(|t| t.get("lines"));
    let covered = lines
        .and_then(|l| l.get("covered"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let count = lines
        .and_then(|l| l.get("count"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let rate = if count > 0 {
        covered as f64 / count as f64
    } else {
        0.0
    };
    (rate, covered, count)
}

/// Floor stored in the baseline file, if present and parseable.
fn load_coverage_floor(path: &Path) -> Option<f64> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("line_rate").and_then(|x| x.as_f64()))
}

/// Overwrites the baseline file with the current measurement.
fn save_coverage_floor(path: &Path, rate: f64, covered: u64, count: u64) -> std::io::Result<()> {
    let doc = serde_json::json!({
        "version": 1,
        "line_rate": rate,
        "lines_covered": covered,
        "lines_total": count,
        "generated_by": "cargo xtask quality --full (cargo-llvm-cov --workspace)",
    });
    std::fs::write(
        path,
        serde_json::to_string_pretty(&doc).unwrap_or_default() + "\n",
    )
}

/// Ratchet verdict against the stored floor: a regression beyond the
/// tolerance band FAILs, first run and growth beyond the band rewrite the
/// file (commit it), anything inside the band PASSes silently. A zero-line
/// summary is a collection failure, never a floor (it must not save 0.0).
fn coverage_verdict(baseline: &Path, rate: f64, covered: u64, count: u64) -> (Status, String) {
    if count == 0 {
        return (
            Status::Fail,
            "coverage summary reports zero total lines (collection broke)".to_string(),
        );
    }
    let prior = load_coverage_floor(baseline);
    if prior.is_some_and(|b| rate + COVERAGE_EPS < b) {
        (
            Status::Fail,
            format!(
                "workspace lines {:.2}% below baseline {:.2}%",
                rate * COVERAGE_PERCENT,
                prior.unwrap_or(0.0) * COVERAGE_PERCENT
            ),
        )
    } else if prior.is_none_or(|b| rate > b + COVERAGE_EPS) {
        match save_coverage_floor(baseline, rate, covered, count) {
            Ok(()) if prior.is_none() => (
                Status::Pass,
                format!("baseline set at {:.2}%", rate * COVERAGE_PERCENT),
            ),
            Ok(()) => (
                Status::Pass,
                format!(
                    "baseline lifted to {:.2}% — commit {COVERAGE_BASELINE}",
                    rate * COVERAGE_PERCENT
                ),
            ),
            Err(e) => (
                Status::Fail,
                format!("cannot write {COVERAGE_BASELINE}: {e}"),
            ),
        }
    } else {
        (Status::Pass, String::new())
    }
}

fn bench_stage(stages: &mut StageList<'_>) {
    stages.run(
        "criterion",
        "criterion benches",
        "cargo bench --workspace",
        stages.cargo(&["bench", "--workspace"]),
        false,
    );
}

/// ── CI set (--ci): rustdoc + wasm target check ────────────
/// These two stages mirror what the GitHub Actions quality shards
/// run; --ci makes the local gate identical to CI by construction.
fn ci_stages(stages: &mut StageList<'_>) {
    // Strict rustdoc: `-D warnings` is set inside xtask (not the
    // workflow) so local and CI runs deny warnings identically; any
    // pre-existing RUSTDOCFLAGS (e.g. mold link args on CI) is preserved
    // and `-D warnings` is appended once. Inlined (not a helper fn) so
    // the rustqual baseline stays flat (no new untested function).
    let mut doc = stages.cargo(&["doc", "--workspace", "--no-deps"]);
    let rustdocflags = std::env::var("RUSTDOCFLAGS").unwrap_or_default();
    if !(rustdocflags.contains("-D") && rustdocflags.contains("warnings")) {
        let merged = if rustdocflags.trim().is_empty() {
            "-D warnings".to_string()
        } else {
            format!("{rustdocflags} -D warnings")
        };
        doc.env("RUSTDOCFLAGS", merged);
    }
    stages.run(
        "doc",
        "doc",
        "cargo doc --workspace --no-deps (RUSTDOCFLAGS includes -D warnings)",
        doc,
        false,
    );

    if wasm_target_installed() {
        stages.run(
            "wasm-check",
            "wasm-check",
            "cargo check -p ornis-wasm --target wasm32-unknown-unknown",
            stages.cargo(&[
                "check",
                "-p",
                "ornis-wasm",
                "--target",
                "wasm32-unknown-unknown",
            ]),
            false,
        );
    } else {
        stages.skip(
            "wasm-check",
            "wasm-check",
            "wasm32-unknown-unknown target not installed:  rustup target add wasm32-unknown-unknown",
        );
    }
}

/// ── Deep static analysis (--everything) ────────────────────
/// Long-running stages, only under the explicit deep flag.
fn deep_stages(stages: &mut StageList<'_>) {
    if cargo_subcommand_exists("mutants") {
        stages.run(
            "mutants",
            "mutants (ornis-core)",
            "cargo mutants -p ornis-core --features lock-free --timeout 300",
            stages.cargo(&[
                "mutants",
                "-p",
                "ornis-core",
                "--features",
                "lock-free",
                "--timeout",
                "300",
            ]),
            false,
        );
    } else {
        stages.skip(
            "mutants",
            "mutants (ornis-core)",
            "cargo-mutants not installed",
        );
    }

    if cargo_subcommand_exists("fuzz") && nightly_available() {
        stages.run(
            "fuzz-scene",
            "fuzz smoke (scene_ron)",
            "cargo +nightly fuzz run scene_ron -- -runs=200",
            stages.cargo(&["+nightly", "fuzz", "run", "scene_ron", "--", "-runs=200"]),
            false,
        );
        stages.run(
            "fuzz-editor",
            "fuzz smoke (editor_command)",
            "cargo +nightly fuzz run editor_command -- -runs=200",
            stages.cargo(&[
                "+nightly",
                "fuzz",
                "run",
                "editor_command",
                "--",
                "-runs=200",
            ]),
            false,
        );
    } else {
        stages.skip(
            "fuzz-scene",
            "fuzz smoke (scene_ron)",
            "cargo-fuzz or nightly toolchain missing",
        );
        stages.skip(
            "fuzz-editor",
            "fuzz smoke (editor_command)",
            "cargo-fuzz or nightly toolchain missing",
        );
    }
}

pub fn quality(args: &[String]) {
    let flags = QualityFlags::parse(args);

    let root = crate::workspace_root();
    let total = flags.total_stages();
    if total == 0 {
        eprintln!(
            "xtask quality: --only selected no active stages \
             (level-gated ids like doc/coverage need --ci/--full; see --list-stages)"
        );
    }
    let mut stages = StageList::new(&root, total, flags.only.clone(), flags.strict());

    // ── Level 1 (mandatory set) ───────────────────────────────
    level1(&mut stages);

    // ── Level 2 (--full) ──────────────────────────────────────
    if flags.full {
        full_stages(&mut stages);
    }

    if flags.bench {
        bench_stage(&mut stages);
    }

    // ── CI set (--ci) ─────────────────────────────────────────
    if flags.ci {
        ci_stages(&mut stages);
    }

    // ── Deep static analysis (--everything) ───────────────────
    if flags.everything {
        deep_stages(&mut stages);
    }

    print_summary(&stages.results);
    let failed = stages.results.iter().any(|r| r.status == Status::Fail);
    if skip_exit_code(&stages.results, &flags) != 0 || failed {
        exit(1);
    }
}

/// Enforces the skip policy; returns 1 when the gate must fail on skips.
/// A SKIP is a silent green locally (trailing warning only); in strict
/// mode (`--ci` or `GITHUB_ACTIONS`) an unexcused SKIP fails the gate —
/// only `--allow-skip id,...` excuses it.
fn skip_exit_code(results: &[StageResult], flags: &QualityFlags) -> i32 {
    if flags.strict() {
        let unexcused: Vec<&StageResult> = results
            .iter()
            .filter(|r| r.status == Status::Skip && !flags.skip_allowed(&r.id))
            .collect();
        if unexcused.is_empty() {
            return 0;
        }
        let names: Vec<&str> = unexcused.iter().map(|r| r.name.as_str()).collect();
        eprintln!(
            "xtask quality: {} stage(s) skipped without --allow-skip: {}",
            names.len(),
            names.join(", ")
        );
        if ci_annotations() {
            for r in &unexcused {
                annotate(
                    stage_title(&r.name),
                    &format!("stage SKIPped in CI without --allow-skip ({})", r.note),
                );
            }
        }
        return 1;
    }
    let skipped: Vec<&str> = results
        .iter()
        .filter(|r| r.status == Status::Skip)
        .map(|r| r.name.as_str())
        .collect();
    if !skipped.is_empty() {
        eprintln!(
            "xtask quality: warning: {} stage(s) skipped: {}",
            skipped.len(),
            skipped.join(", ")
        );
    }
    0
}

fn quality_usage(code: i32) -> ! {
    eprintln!(
        "xtask quality — the Ornis quality gate\n\
         \n\
         USAGE:\n  \
          cargo xtask quality           quick set (level 1): fmt, clippy-physics, test-physics, clippy, rustqual, smoke, test, test-lockfree, audit, deny, outdated, upgrade-check, machete, typos\n  \
         cargo xtask quality --ci      + rustdoc and wasm32 check (same set GitHub Actions runs)\n  \
         cargo xtask quality --full    + coverage ratchet (llvm-cov --workspace vs coverage-baseline.json), bench compile-check and the cargo-hack feature matrix\n  \
         cargo xtask quality --bench   + full criterion benchmark run (slow)\n  \
         cargo xtask quality --everything\n      \
         everything: --ci + --full + --bench + mutants (ornis-core) + fuzz smoke (slow, minutes to hours)\n  \
         cargo xtask quality --ci --only fmt,audit  (CI sharding: run a subset)\n  \
         cargo xtask quality --ci --allow-skip wasm-check  (excuse a SKIP in strict CI mode)\n  \
         cargo xtask quality --list-stages  (print canonical stage ids)\n\
          \n\
          In strict mode (--ci or GITHUB_ACTIONS) a SKIP fails the gate unless\n  \
          excused via --allow-skip. Locally skips only print a trailing warning.\n\
          \n\
          External tools (audit, deny, outdated, upgrade-check, llvm-cov, rustqual, hack, machete, typos) are optional:\n  \
          missing → SKIP with install hint. rustqual is MIT.\n  \
          rustqual.toml is the single source of truth (no thresholds duplicated here).\n  \
          Baseline: rustqual --save-baseline baseline.json; the gate re-runs rustqual\n  \
          into target/rustqual_cur.json and compares with its own ratchet\n  \
          (equal = PASS, only a true regression FAILs). The committed baseline\n  \
           must not exceed the base branch version (origin/master locally) —\n  \
           growth needs the `baseline-bump-approved` PR label unless auto-accepted\n  \
           (scores up, no new findings, soft-category density non-increasing).\n  \
           Coverage ratchet: coverage-baseline.json holds the workspace line-rate floor\n  \
           (within ±0.1 pp = PASS; deeper regression FAILs, growth rewrites the file — commit it).\n  \
           Coverage runs nightly (.github/workflows/coverage.yml), not in PR shards:\n  \
           a full instrumented workspace run takes tens of minutes.\n  \
         Smoke: cargo run --features editor-only must bind 127.0.0.1:3420 within 90s and stay alive"
    );
    exit(code);
}

/// Builds a Command with the workspace root as the working directory.
fn cmd(root: &Path, program: &str, args: &[&str]) -> Command {
    let mut c = Command::new(program);
    c.args(args).current_dir(root);
    c
}

/// Runs one stage: header, tool availability check, status.
/// `informational` — a non-zero exit is not counted as FAIL (cargo-outdated).
fn run_stage(
    index: usize,
    total: usize,
    id: &str,
    name: &str,
    display_cmd: &str,
    mut command: Command,
    informational: bool,
) -> StageResult {
    eprintln!();
    eprintln!("═══ [{index}/{total}] {name}: {display_cmd} ═══");
    let started = Instant::now();

    // Check for an external cargo tool, with an install hint.
    // Only third-party subcommands are checked: built-in ones (test, bench, …)
    // have no cargo-<sub> binary that could be found in PATH.
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
                eprintln!("xtask quality: SKIP — tool 'cargo {sub}' not found.\n{hint}");
                return StageResult {
                    id: id.to_string(),
                    name: name.to_string(),
                    status: Status::Skip,
                    note: format!("cargo-{sub} not installed"),
                    elapsed: started.elapsed(),
                };
            }
        }
    }

    // In CI the stage output is captured and re-printed so that a failure
    // can also be surfaced as `::error::` workflow-command annotations
    // (raw run logs live on an endpoint some sandboxes cannot reach; the
    // annotations API is the transport that always works). Locally the
    // stages keep streaming.
    let ran = if ci_annotations() {
        diagnostics::run_streamed(&mut command)
    } else {
        command.status().map(|status| (status, String::new()))
    };

    match ran {
        Ok((status, _log)) if status.success() => {
            eprintln!(
                "── {name}: {} ──",
                if informational { "INFO" } else { "PASS" }
            );
            StageResult {
                id: id.to_string(),
                name: name.to_string(),
                status: if informational {
                    Status::Info
                } else {
                    Status::Pass
                },
                note: String::new(),
                elapsed: started.elapsed(),
            }
        }
        Ok((status, log)) => {
            if informational {
                eprintln!("── {name}: INFO (exit {status} — outdated dependencies) ──");
                StageResult {
                    id: id.to_string(),
                    name: name.to_string(),
                    status: Status::Info,
                    note: format!("{status}"),
                    elapsed: started.elapsed(),
                }
            } else {
                eprintln!("── {name}: FAIL (exit {status}) ──");
                annotate_stage_failure(name, &log);
                StageResult {
                    id: id.to_string(),
                    name: name.to_string(),
                    status: Status::Fail,
                    note: format!("{status}"),
                    elapsed: started.elapsed(),
                }
            }
        }
        Err(e) => {
            eprintln!("── {name}: FAIL (spawn error: {e}) ──");
            StageResult {
                id: id.to_string(),
                name: name.to_string(),
                status: Status::Fail,
                note: format!("spawn: {e}"),
                elapsed: started.elapsed(),
            }
        }
    }
}

/// Whether the gate runs inside a GitHub Actions step (workflow commands
/// `::error::…` become check-run annotations there).
fn ci_annotations() -> bool {
    std::env::var_os("GITHUB_ACTIONS").is_some()
}

/// Tail bytes kept when a stage log has no `failures:` marker.
const STAGE_LOG_TAIL_BYTES: usize = 18_000;
/// Cap on interesting lines before dropping `+`/`-` bodies.
const INTERESTING_LINE_SOFT_CAP: usize = 15;
/// Max interesting lines annotated per stage (GitHub ~10/step budget).
const INTERESTING_LINE_HARD_CAP: usize = 40;
/// Max characters per GitHub Actions error annotation line.
const ANNOTATION_LINE_CHARS: usize = 220;

/// Emits the most relevant error lines of a failed stage as annotations
/// (max 8: cargo/rustc errors, failing tests, fmt diffs, clippy warnings).
fn annotate_stage_failure(name: &str, log: &str) {
    if !ci_annotations() {
        return;
    }
    // CI sets CARGO_TERM_COLOR=always: strip ANSI codes before matching,
    // otherwise colored diagnostics break the prefix checks below.
    let clean = strip_ansi(log);
    // Preserve full failure context independently of short UI annotations
    // and post-failure workflow steps (which a stopped runner may not run).
    let detail = if let Some(start) = clean.find("failures:\n") {
        let rest = &clean[start..];
        let end = rest.find("test result:").unwrap_or(rest.len());
        &rest[..end]
    } else {
        let mut tail = clean.len().saturating_sub(STAGE_LOG_TAIL_BYTES);
        while !clean.is_char_boundary(tail) {
            tail += 1;
        }
        &clean[tail..]
    };
    diagnostics::attachment(&format!("stage-{}", name.replace(' ', "-")), detail);
    let is_match = |l: &str| {
        let t = l.trim_start();
        let lower = t.to_ascii_lowercase();
        t.starts_with("error")
            // only failing test summaries — "ok" results flood the cap
            || (t.starts_with("test result:") && t.contains("FAILED"))
            || t.starts_with("Diff in")
            || t.contains("panicked")
            || t.contains("FAILED")
            // clippy/rustc lowercase vs cargo-audit capitalized "Warning:"
            || t.starts_with("warning:")
            || t.starts_with("Warning:")
            || lower.contains("vulnerab")
            || t.contains("(limit ")
            || t.starts_with('+')
            || t.starts_with('-')
            || t.starts_with("-->")
            || (t.contains("expected") && t.contains("found"))
            || t.starts_with("note:")
    };
    // rustfmt diffs: one annotation per hunk — bodies never fit the cap.
    if name == "fmt" {
        for hunk in fmt_hunks(&clean) {
            annotate(stage_title(name), &hunk);
        }
        return;
    }
    let mut interesting: Vec<&str> = clean.lines().filter(|l| is_match(l)).collect();
    // GitHub surfaces only ~10 annotations per step: when the diff is large,
    // drop `+`/`-` bodies and keep headers/diagnostics so nothing is hidden.
    if interesting.len() > INTERESTING_LINE_SOFT_CAP {
        interesting.retain(|l| {
            let t = l.trim_start();
            !t.starts_with('+') && !t.starts_with('-')
        });
    }
    let start = interesting.len().saturating_sub(INTERESTING_LINE_HARD_CAP);
    let picked = &interesting[start..];
    if picked.is_empty() {
        annotate(
            stage_title(name),
            "stage failed with no recognized error lines — see the raw log",
        );
    }
    for l in picked {
        annotate(stage_title(name), l.trim());
    }
}

/// Folds rustfmt output into one message per hunk: "header ⏎ -old ⏎ +new".
fn fmt_hunks(clean: &str) -> Vec<String> {
    let mut hunks: Vec<String> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let flush = |current: &mut Vec<&str>, hunks: &mut Vec<String>| {
        if !current.is_empty() {
            hunks.push(current.join(" ⏎ "));
            current.clear();
        }
    };
    for l in clean.lines() {
        let t = l.trim_start();
        if t.starts_with("Diff in") {
            flush(&mut current, &mut hunks);
            current.push(t);
        } else if (t.starts_with('+') || t.starts_with('-')) && !current.is_empty() {
            current.push(t.trim_end());
        }
    }
    flush(&mut current, &mut hunks);
    hunks
}

/// Removes ANSI escape sequences (SGR and friends) from colored output.
fn strip_ansi(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            match chars.next() {
                Some('[') => {
                    for c2 in chars.by_ref() {
                        // CSI sequences end at the first letter
                        if c2.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
                // two-byte escapes: drop the next char too
                Some(_) => {}
                None => break,
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// `::error` annotation title for a stage (`quality-<id>`); display
/// names are dash-normalized so titles never contain spaces.
fn stage_title(name: &str) -> String {
    format!("quality-{}", name.replace(' ', "-"))
}

/// One `::error::` workflow command (GitHub Actions annotations).
fn annotate(title: String, message: &str) {
    let esc = |s: &str| -> String {
        s.replace('%', "%25")
            .replace('\r', "%0D")
            .replace('\n', "%0A")
    };
    let mut line = message.trim().to_string();
    if line.len() > ANNOTATION_LINE_CHARS {
        // Truncate at a char boundary: `str::truncate` panics mid-UTF-8.
        let mut end = ANNOTATION_LINE_CHARS;
        while !line.is_char_boundary(end) {
            end -= 1;
        }
        line.truncate(end);
    }
    eprintln!("::error title={}::{}", esc(&title), esc(&line));
}

fn print_summary(results: &[StageResult]) {
    eprintln!();
    eprintln!("╔════════════ QUALITY SUMMARY ════════════╗");
    let total: Duration = results.iter().map(|r| r.elapsed).sum();
    for r in results {
        let note = if r.note.is_empty() {
            String::new()
        } else {
            format!(" ({})", r.note)
        };
        eprintln!(
            "  {:<22} {:<4} {:>8}{}",
            r.name,
            r.status.label(),
            format_duration(r.elapsed),
            note
        );
    }
    eprintln!("  {:<22}      {:>8}", "total", format_duration(total));
    eprintln!("╚═════════════════════════════════════════╝");
    if ci_annotations() {
        for r in results.iter().filter(|r| r.status == Status::Fail) {
            let note = if r.note.is_empty() {
                "-".to_string()
            } else {
                r.note.clone()
            };
            annotate(
                format!("quality-summary-{}", r.name.replace(' ', "-")),
                &format!("stage FAIL ({note})"),
            );
        }
    }
}

/// Short wall-clock rendering for the SUMMARY table: `0.4s`, `12s`, `3m04s`.
fn format_duration(d: Duration) -> String {
    const SECS_PER_MINUTE: u64 = 60;
    const DOUBLE_DIGIT_SECS: u64 = 10;
    let secs = d.as_secs();
    if secs >= SECS_PER_MINUTE {
        format!("{}m{:02}s", secs / SECS_PER_MINUTE, secs % SECS_PER_MINUTE)
    } else if secs >= DOUBLE_DIGIT_SECS {
        format!("{secs}s")
    } else {
        format!("{:.1}s", d.as_secs_f64())
    }
}

/// Whether a binary exists in PATH.
fn binary_exists(bin: &str) -> bool {
    let binary = if cfg!(windows) {
        format!("{bin}.exe")
    } else {
        bin.to_string()
    };
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(&binary).is_file()))
        .unwrap_or(false)
}

/// Whether a cargo subcommand exists in PATH (`cargo-<sub>`).
/// Checks for the binary, not `--version`: some tools
/// (e.g. cargo-outdated) do not understand `--version` directly.
fn cargo_subcommand_exists(sub: &str) -> bool {
    binary_exists(&format!("cargo-{sub}"))
}

/// rustqual binary to run: the vendored Ornis fork when built
/// (release preferred, debug accepted), otherwise whatever `rustqual`
/// resolves to on PATH (CI puts the release fork build there).
fn rustqual_binary(root: &std::path::Path) -> String {
    let base = root.join("third_party/rustqual/target");
    for profile in ["release", "debug"] {
        let candidate = base.join(profile).join("rustqual");
        if candidate.is_file() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "rustqual".to_string()
}

/// Records a SKIP without spawning a command (no progress output).
fn skip_stage(index: usize, total: usize, id: &str, name: &str, note: &str) -> StageResult {
    eprintln!();
    eprintln!("═══ [{index}/{total}] {name} ═══");
    eprintln!("── {name}: SKIP ({note}) ──");
    StageResult {
        id: id.to_string(),
        name: name.to_string(),
        status: Status::Skip,
        note: note.to_string(),
        elapsed: Duration::ZERO,
    }
}

/// Compile separately so cold-build time is not confused with app readiness.
/// Startup still has the same 90-second deadline and must leave a live server.
fn smoke_stage(stages: &mut StageList<'_>) {
    if !stages.enabled("smoke") {
        return;
    }
    stages.n += 1;
    eprintln!(
        "═══ [{}/{}] smoke (editor-only): build + 90s readiness ═══",
        stages.n, stages.total
    );
    let started = Instant::now();
    let (status, note) = match smoke::check(stages.root) {
        Ok(()) => (Status::Pass, String::new()),
        Err(log) => {
            annotate_stage_failure("smoke", &log);
            eprintln!("{log}");
            (
                Status::Fail,
                log.lines().next().unwrap_or("smoke failed").to_string(),
            )
        }
    };
    eprintln!("── smoke (editor-only): {} ──", status.label());
    stages.results.push(StageResult {
        id: "smoke".into(),
        name: "smoke (editor-only)".into(),
        status,
        note,
        elapsed: started.elapsed(),
    });
}

/// Whether the wasm32-unknown-unknown target is installed for the
/// active toolchain (`rustup target list --installed`).
fn wasm_target_installed() -> bool {
    Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .map(|out| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .any(|l| l.trim().starts_with("wasm32-unknown-unknown"))
        })
        .unwrap_or(false)
}

fn install_hint(sub: &str) -> String {
    match sub {
        "audit" => "Install:  cargo install cargo-audit --locked".to_string(),
        "deny" => "Install:  cargo install cargo-deny --locked".to_string(),
        "outdated" => "Install:  cargo install cargo-outdated --locked".to_string(),
        "upgrade" => "Install:  cargo install cargo-edit --locked".to_string(),
        "hack" => {
            format!("Install:  cargo install cargo-hack --version {CARGO_HACK_VERSION} --locked")
        }
        "machete" => format!(
            "Install:  cargo install cargo-machete --version {CARGO_MACHETE_VERSION} --locked"
        ),
        "llvm-cov" => "Install:  cargo install cargo-llvm-cov --locked\n\
             and the component:  rustup component add llvm-tools-preview"
            .to_string(),
        "fuzz" => "Install:  cargo install cargo-fuzz --locked".to_string(),
        "mutants" => "Install:  cargo install cargo-mutants --locked".to_string(),
        other => format!("Install:  cargo install cargo-{other} --locked"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// fuzz / mutants — separate subcommands (not part of quality default)
// ═══════════════════════════════════════════════════════════════════════

pub fn fuzz(args: &[String]) {
    let Some(target) = args.first() else {
        eprintln!(
            "xtask fuzz — runs cargo-fuzz targets (external-input parsers)\n\
             \n\
             USAGE:\n  \
             cargo xtask fuzz <target> [-- <libfuzzer args>]\n  \
             available targets: scene_ron, materialx_parse, editor_command\n\
             \n\
             Example:  cargo xtask fuzz scene_ron -- -runs=1000"
        );
        exit(2);
    };
    let extra: Vec<&str> = args[1..].iter().map(String::as_str).collect();

    if !cargo_subcommand_exists("fuzz") {
        eprintln!(
            "xtask fuzz: 'cargo-fuzz' not found.\n{}",
            install_hint("fuzz")
        );
        exit(1);
    }
    if !nightly_available() {
        eprintln!(
            "xtask fuzz: nightly toolchain not found (cargo-fuzz requires nightly).\n\
             Install:  rustup toolchain install nightly\n\
             (the workspace is pinned to stable via rust-toolchain.toml — fuzz is \
             always run explicitly through +nightly)"
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
    let status = match c.status() {
        Ok(status) => status,
        Err(e) => {
            eprintln!("xtask fuzz: failed to spawn cargo-fuzz: {e}");
            exit(1);
        }
    };
    exit(status.code().unwrap_or(1));
}

pub fn mutants(args: &[String]) {
    if !cargo_subcommand_exists("mutants") {
        eprintln!(
            "xtask mutants: 'cargo-mutants' not found.\n{}",
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
        .arg("--features")
        .arg("lock-free")
        .arg("--timeout")
        .arg("300")
        .args(&extra)
        .current_dir(&root);
    eprintln!(
        "xtask mutants: cargo mutants -p ornis-core --features lock-free --timeout 300 {}",
        extra.join(" ")
    );
    let status = match c.status() {
        Ok(status) => status,
        Err(e) => {
            eprintln!("xtask mutants: failed to spawn cargo-mutants: {e}");
            exit(1);
        }
    };
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

#[cfg(test)]
mod ratchet_tests {
    use super::*;
    use serde_json::json;

    fn report_with(overrides: serde_json::Value) -> String {
        let mut base = json!({
            "quality_score": 0.8,
            "iosp_score": 0.9,
            "total": 1000,
            "violations": 10,
            "total_findings": 100,
            "complexity_warnings": 5,
            "magic_number_warnings": 5,
            "nesting_depth_warnings": 1,
            "function_length_warnings": 5,
            "unsafe_warnings": 2,
            "error_handling_warnings": 2,
            "duplicate_groups": 3,
            "dead_code_warnings": 10,
            "dead_type_warnings": 1,
            "fragment_groups": 4,
            "boilerplate_warnings": 6,
            "srp_struct_warnings": 2,
            "srp_module_warnings": 3,
            "wildcard_import_warnings": 1,
            "sdp_violations": 0,
            "coupling_warnings": 0,
            "coupling_cycles": 0,
            "tq_no_assertion_warnings": 0,
            "tq_no_sut_warnings": 1,
            "tq_untested_warnings": 5,
            "tq_uncovered_warnings": 0,
            "tq_untested_logic_warnings": 0,
            "structural_srp_warnings": 2,
            "structural_coupling_warnings": 1,
            "violation_details": [
                {"name": "foo", "file": "a.rs", "line": 1},
                {"name": "bar", "file": "b.rs", "line": 2}
            ]
        });
        if let (Some(map), Some(ov)) = (base.as_object_mut(), overrides.as_object()) {
            for (k, v) in ov {
                map.insert(k.clone(), v.clone());
            }
        }
        base.to_string()
    }

    fn base_report() -> String {
        report_with(json!({}))
    }

    #[test]
    fn equal_is_pass_and_not_stale() {
        let b = base_report();
        let o = compare_rustqual_reports(&b, &b).expect("equal must parse");
        assert!(o.regressed_reasons.is_empty());
        assert!(!o.baseline_stale);
    }

    #[test]
    fn iosp_drop_is_regression() {
        let b = base_report();
        let c = report_with(json!({"iosp_score": 0.5}));
        let o = compare_rustqual_reports(&b, &c).expect("must parse");
        assert!(o.regressed_reasons.iter().any(|r| r.contains("iosp_score")));
    }

    #[test]
    fn quality_drop_is_regression() {
        let b = base_report();
        let c = report_with(json!({"quality_score": 0.1}));
        let o = compare_rustqual_reports(&b, &c).expect("must parse");
        assert!(o
            .regressed_reasons
            .iter()
            .any(|r| r.contains("quality_score")));
    }

    #[test]
    fn category_bump_is_regression_even_when_total_flat() {
        // unsafe +1, dead_code -1: total_findings overridden flat, but the
        // per-category ratchet must still catch the unsafe increase.
        let b = base_report();
        let c = report_with(json!({
            "unsafe_warnings": 3,
            "dead_code_warnings": 9,
            "total_findings": 100,
            "violation_details": [
                {"name": "foo", "file": "a.rs", "line": 1},
                {"name": "bar", "file": "b.rs", "line": 2}
            ]
        }));
        let o = compare_rustqual_reports(&b, &c).expect("must parse");
        assert!(o
            .regressed_reasons
            .iter()
            .any(|r| r.contains("unsafe_warnings")));
    }

    #[test]
    fn swapped_findings_with_flat_totals_is_regression() {
        // Same totals, one finding replaced: identity set catches the swap.
        let b = base_report();
        let c = report_with(json!({
            "violation_details": [
                {"name": "foo", "file": "a.rs", "line": 1},
                {"name": "NEW", "file": "c.rs", "line": 9}
            ]
        }));
        let o = compare_rustqual_reports(&b, &c).expect("must parse");
        assert!(o
            .regressed_reasons
            .iter()
            .any(|r| r.contains("new findings")));
        assert_eq!(
            o.added_findings,
            vec![("c.rs".to_string(), "NEW".to_string())]
        );
    }

    #[test]
    fn missing_key_is_fail_not_zero() {
        let mut v: serde_json::Value = serde_json::from_str(&base_report()).unwrap();
        v.as_object_mut().unwrap().remove("unsafe_warnings");
        let err = compare_rustqual_reports(&base_report(), &v.to_string()).unwrap_err();
        assert!(err.contains("unsafe_warnings"), "unexpected: {err}");
    }

    #[test]
    fn baseline_bump_fails_while_drop_passes() {
        // The bump guard reuses this same comparator (base baseline vs
        // committed baseline): a bumped count fails, a lowered one
        // passes as stale.
        let base = base_report();
        let bumped = report_with(json!({"dead_code_warnings": 11}));
        let o = compare_rustqual_reports(&base, &bumped).expect("must parse");
        assert!(
            o.regressed_reasons
                .iter()
                .any(|r| r.contains("dead_code_warnings")),
            "unexpected: {o:?}"
        );
        let lowered = report_with(json!({
            "dead_code_warnings": 9,
            "total_findings": 99,
            "violation_details": [{"name": "foo", "file": "a.rs", "line": 1}]
        }));
        let o2 = compare_rustqual_reports(&base, &lowered).expect("must parse");
        assert!(o2.regressed_reasons.is_empty());
        assert!(o2.baseline_stale);
    }

    #[test]
    fn missing_details_is_fail() {
        let mut v: serde_json::Value = serde_json::from_str(&base_report()).unwrap();
        v.as_object_mut().unwrap().remove("violation_details");
        let err = compare_rustqual_reports(&base_report(), &v.to_string()).unwrap_err();
        assert!(err.contains("violation_details"), "unexpected: {err}");
    }

    #[test]
    fn detail_entry_without_identity_is_fail() {
        let mut v: serde_json::Value = serde_json::from_str(&base_report()).unwrap();
        v["violation_details"][0]
            .as_object_mut()
            .unwrap()
            .remove("file");
        let err = compare_rustqual_reports(&base_report(), &v.to_string()).unwrap_err();
        assert!(err.contains("`file`"), "unexpected: {err}");
        let mut w: serde_json::Value = serde_json::from_str(&base_report()).unwrap();
        w["violation_details"][1] = json!({"name": "x", "file": 42, "line": 1});
        let err2 = compare_rustqual_reports(&base_report(), &w.to_string()).unwrap_err();
        assert!(err2.contains("`file`"), "unexpected: {err2}");
    }

    /// Scratch git repo with `baseline.json` committed and visible at
    /// `remote_ref` (what CI's full-history checkout provides). The ref
    /// is a parameter — deliberately not `origin/master` — so the guard
    /// tests prove they follow the passed ref instead of the ambient
    /// `GITHUB_*` env (RAT-24 review of PR #75).
    fn init_guard_repo(name: &str, base_json: &str, remote_ref: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("xtask-guard-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("baseline.json"), base_json).unwrap();
        let git = |args: &[&str]| {
            let st = Command::new("git")
                .args(args)
                .current_dir(&dir)
                .status()
                .unwrap();
            assert!(st.success(), "git {args:?} failed");
        };
        git(&["init", "-q"]);
        git(&["add", "baseline.json"]);
        git(&[
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "base",
        ]);
        git(&["update-ref", &format!("refs/remotes/{remote_ref}"), "HEAD"]);
        dir
    }

    #[test]
    fn bump_guard_fails_growth_and_passes_equal() {
        // Non-master ref: the guard must follow the parameter, not the
        // ambient base ref env.
        let guard_ref = "origin/pr-base";
        let dir = init_guard_repo("bump", &base_report(), guard_ref);
        let base = read_base_baseline(&dir, guard_ref).expect("base baseline readable");
        assert!(base.contains("quality_score"));
        // Committed == base: the ratchet holds.
        assert!(check_baseline_bump(&dir, &base_report(), true, false, guard_ref).is_none());
        // Committed above base: FAIL naming the count and the label.
        let bumped = report_with(json!({"dead_code_warnings": 11}));
        let reason =
            check_baseline_bump(&dir, &bumped, true, false, guard_ref).expect("bump must fail");
        assert!(
            reason.contains("dead_code_warnings"),
            "unexpected: {reason}"
        );
        assert!(
            reason.contains("baseline-bump-approved"),
            "unexpected: {reason}"
        );
        // Owner-approved bump: the same growth holds.
        assert!(check_baseline_bump(&dir, &bumped, true, true, guard_ref).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deleted_baseline_guard_fails_closed_in_strict() {
        let guard_ref = "origin/pr-base";
        let dir = init_guard_repo("deleted", &base_report(), guard_ref);
        let mut stages = StageList::new(&dir, 1, None, true);
        assert!(reject_deleted_baseline(&mut stages, guard_ref));
        assert_eq!(stages.results.len(), 1);
        assert_eq!(stages.results[0].status, Status::Fail);
        let mut lax = StageList::new(&dir, 1, None, false);
        assert!(!reject_deleted_baseline(&mut lax, guard_ref));
        assert!(lax.results.is_empty());
        // apply_bump_guard flips a Pass row to Fail on growth.
        let mut stages2 = StageList::new(&dir, 1, None, true);
        stages2.results.push(stage_result(
            "rustqual",
            Status::Pass,
            String::new(),
            Duration::ZERO,
        ));
        let bumped = report_with(json!({"dead_code_warnings": 11}));
        apply_bump_guard(&mut stages2, "rustqual", &bumped, false, guard_ref);
        assert_eq!(stages2.results[0].status, Status::Fail);
        assert!(stages2.results[0].note.contains("dead_code_warnings"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_json_is_fail() {
        let err = compare_rustqual_reports("{not json", &base_report()).unwrap_err();
        assert!(err.contains("not valid JSON"), "unexpected: {err}");
        let err2 = compare_rustqual_reports(&base_report(), "null").unwrap_err();
        assert!(err2.contains("missing"), "unexpected: {err2}");
    }

    #[test]
    fn improvement_marks_baseline_stale() {
        let b = base_report();
        let c = report_with(json!({
            "dead_code_warnings": 9,
            "total_findings": 99,
            "violation_details": [{"name": "foo", "file": "a.rs", "line": 1}]
        }));
        let o = compare_rustqual_reports(&b, &c).expect("must parse");
        assert!(o.regressed_reasons.is_empty());
        assert!(o.baseline_stale);
        assert!(!o.improvement_notes.is_empty());
    }

    #[test]
    fn epsilon_equal_scores_pass() {
        let b = base_report();
        let c = report_with(json!({"quality_score": 0.8 + 5e-10, "iosp_score": 0.9 - 5e-10}));
        let o = compare_rustqual_reports(&b, &c).expect("must parse");
        assert!(o.regressed_reasons.is_empty());
    }

    // ── RAT-44 auto-accept ──

    /// Minimised #97 fixture: scores up, violations flat, total 7436→7516,
    /// soft growth exactly within ceil-density.
    fn rat97_base() -> String {
        report_with(json!({
            "quality_score": 0.546799,
            "iosp_score": 0.917423,
            "violations": 391,
            "total": 7436,
            "total_findings": 2829,
            "dead_code_warnings": 421,
            "srp_module_warnings": 100,
            "tq_untested_warnings": 151
        }))
    }
    fn rat97_cur() -> String {
        report_with(json!({
            "quality_score": 0.550172,
            "iosp_score": 0.918491,
            "violations": 391,
            "total": 7516,
            "total_findings": 2845,
            "dead_code_warnings": 422,
            "srp_module_warnings": 101,
            "tq_untested_warnings": 153
        }))
    }

    #[test]
    fn auto_accept_97_fixture() {
        let o = baseline_auto_accept(&rat97_base(), &rat97_cur()).expect("97 must auto-accept");
        assert!(o.scores_grew);
        // density spot-check: tq_untested 151→153 is exactly ceil(151*7516/7436).
        assert_eq!(density_limit(151, 7436, 7516), 153);
        assert_eq!(density_limit(2829, 7436, 7516), 2860);
    }

    #[test]
    fn auto_accept_rejects_violations_bump_and_new_identity() {
        // violations 10→11 needs the label…
        let c = report_with(json!({
            "quality_score": 0.81, "iosp_score": 0.91,
            "violations": 11, "total": 1000, "total_findings": 100
        }));
        let err = baseline_auto_accept(&base_report(), &c).unwrap_err();
        assert!(err.contains("violations"), "unexpected: {err}");
        // …and so does a brand-new (file,name) pair with flat counts.
        let d = report_with(json!({
            "quality_score": 0.81, "iosp_score": 0.91,
            "violation_details": [
                {"name": "foo", "file": "a.rs", "line": 1},
                {"name": "FRESH", "file": "z.rs", "line": 3}
            ]
        }));
        let err2 = baseline_auto_accept(&base_report(), &d).unwrap_err();
        assert!(err2.contains("rule 2"), "unexpected: {err2}");
        // Bump guard mirrors it: FAIL without the label, hold with it.
        let guard_ref = "origin/pr-base";
        let dir = init_guard_repo("auto-viol", &base_report(), guard_ref);
        let reason = check_baseline_bump(&dir, &c, true, false, guard_ref).expect("must fail");
        assert!(
            reason.contains("baseline-bump-approved"),
            "unexpected: {reason}"
        );
        assert!(check_baseline_bump(&dir, &c, true, true, guard_ref).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn auto_accept_rejects_score_drop() {
        let c = report_with(json!({"quality_score": 0.8 - 1e-6}));
        let err = baseline_auto_accept(&base_report(), &c).unwrap_err();
        assert!(
            err.contains("rule 1") && err.contains("quality_score"),
            "unexpected: {err}"
        );
        // Equal scores with more findings also need the label.
        let d = report_with(json!({"dead_code_warnings": 11}));
        let err2 = baseline_auto_accept(&base_report(), &d).unwrap_err();
        assert!(
            err2.contains("rule 1") || err2.contains("rule 4"),
            "unexpected: {err2}"
        );
    }

    #[test]
    fn auto_accept_rejects_hard_growth_despite_total_growth() {
        // unsafe +1 is hard: no density allowance even when total grows.
        let b = report_with(json!({"total": 7436}));
        let c = report_with(json!({"total": 7516, "unsafe_warnings": 3}));
        let err = baseline_auto_accept(&b, &c).unwrap_err();
        assert!(
            err.contains("rule 3") && err.contains("unsafe_warnings"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn auto_accept_soft_density_boundary() {
        // tq_untested 151→153 @7436→7516 is exactly the ceil limit → PASS;
        // 154 exceeds it → label.
        let b = report_with(json!({
            "quality_score": 0.80, "iosp_score": 0.90,
            "total": 7436, "tq_untested_warnings": 151
        }));
        let at_limit = report_with(json!({
            "quality_score": 0.81, "iosp_score": 0.91,
            "total": 7516, "tq_untested_warnings": 153
        }));
        assert!(baseline_auto_accept(&b, &at_limit).is_ok());
        let over = report_with(json!({
            "quality_score": 0.81, "iosp_score": 0.91,
            "total": 7516, "tq_untested_warnings": 154
        }));
        let err = baseline_auto_accept(&b, &over).unwrap_err();
        assert!(
            err.contains("rule 4") && err.contains("tq_untested_warnings"),
            "unexpected: {err}"
        );
        // Shrinking total degenerates to absolute <=.
        let shrink =
            report_with(json!({"total": 900, "total_findings": 90, "dead_code_warnings": 11}));
        let err2 = baseline_auto_accept(&base_report(), &shrink).unwrap_err();
        assert!(err2.contains("dead_code_warnings"), "unexpected: {err2}");
    }

    #[test]
    fn auto_accept_rejects_inflated_total_and_missing_key() {
        let doubled = report_with(json!({
            "quality_score": 0.81, "iosp_score": 0.91, "total": 2000
        }));
        let err = baseline_auto_accept(&base_report(), &doubled).unwrap_err();
        assert!(err.contains("rule 5"), "unexpected: {err}");
        let mut v: serde_json::Value = serde_json::from_str(&base_report()).unwrap();
        v.as_object_mut().unwrap().remove("dead_code_warnings");
        let err2 = baseline_auto_accept(&base_report(), &v.to_string()).unwrap_err();
        assert!(err2.contains("dead_code_warnings"), "unexpected: {err2}");
    }

    #[test]
    fn fresh_total_mismatch_is_fail() {
        let b = report_with(json!({"total": 1000}));
        let f = report_with(json!({"total": 1001}));
        let reason = check_fresh_total_matches_committed(&f, &b).expect("must fail");
        assert!(reason.contains("обнови baseline"), "unexpected: {reason}");
        assert!(check_fresh_total_matches_committed(&b, &b).is_none());
    }

    #[test]
    fn fresh_total_guard_flips_pass_to_fail_on_mismatch() {
        // Auto-accepted bump (97 fixture) + fresh total drift → FAIL;
        // matching totals hold.
        let guard_ref = "origin/pr-base";
        let dir = init_guard_repo("fresh-total", &rat97_base(), guard_ref);
        let committed = rat97_cur();
        let fresh_same = rat97_cur();
        let mut ok_stages = StageList::new(&dir, 1, None, true);
        ok_stages.results.push(stage_result(
            "rustqual",
            Status::Pass,
            String::new(),
            Duration::ZERO,
        ));
        apply_fresh_total_guard(
            &mut ok_stages,
            "rustqual",
            &fresh_same,
            &committed,
            guard_ref,
        );
        assert_eq!(ok_stages.results[0].status, Status::Pass);
        let mut bad_stages = StageList::new(&dir, 1, None, true);
        bad_stages.results.push(stage_result(
            "rustqual",
            Status::Pass,
            String::new(),
            Duration::ZERO,
        ));
        let drifted = report_with(json!({
            "quality_score": 0.550172, "iosp_score": 0.918491,
            "violations": 391, "total": 7517,
            "total_findings": 2845, "dead_code_warnings": 422,
            "srp_module_warnings": 101, "tq_untested_warnings": 153
        }));
        apply_fresh_total_guard(&mut bad_stages, "rustqual", &drifted, &committed, guard_ref);
        assert_eq!(bad_stages.results[0].status, Status::Fail);
        assert!(bad_stages.results[0].note.contains("обнови baseline"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bump_guard_auto_accepts_97_without_label() {
        let guard_ref = "origin/pr-base";
        let dir = init_guard_repo("auto97", &rat97_base(), guard_ref);
        assert!(check_baseline_bump(&dir, &rat97_cur(), true, false, guard_ref).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn test_stages() -> StageList<'static> {
        StageList::new(std::path::Path::new("."), 1, None, false)
    }

    #[test]
    fn stage_records_pass_on_equal() {
        let b = base_report();
        let mut stages = test_stages();
        finish_rustqual_stage(&mut stages, "rustqual", &b, &b, Duration::ZERO);
        assert_eq!(stages.results.len(), 1);
        assert_eq!(stages.results[0].status, Status::Pass);
    }

    #[test]
    fn stage_records_fail_on_regression() {
        let b = base_report();
        let c = report_with(json!({"iosp_score": 0.1}));
        let mut stages = test_stages();
        finish_rustqual_stage(&mut stages, "rustqual", &b, &c, Duration::ZERO);
        assert_eq!(stages.results.len(), 1);
        assert_eq!(stages.results[0].status, Status::Fail);
        assert!(stages.results[0].note.contains("iosp_score"));
    }

    #[test]
    fn stage_records_fail_on_corrupt_report() {
        let mut stages = test_stages();
        finish_rustqual_stage(
            &mut stages,
            "rustqual",
            "{broken",
            &base_report(),
            Duration::ZERO,
        );
        assert_eq!(stages.results[0].status, Status::Fail);
        assert!(stages.results[0].note.contains("not valid JSON"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_strings_recognize_semver_and_reject_markers() {
        assert!(is_version_string("1.2.3"));
        assert!(is_version_string("0.31"));
        assert!(is_version_string("2.0.0-alpha.1"));
        assert!(!is_version_string("---"));
        assert!(!is_version_string("Removed"));
        assert!(!is_version_string(""));
        assert!(!is_version_string("1"));
    }

    #[test]
    fn major_lag_parser_lists_only_newer_latest() {
        let stdout = concat!(
            "{\"crate_name\":\"ornis\",\"dependencies\":[",
            "{\"name\":\"serde\",\"project\":\"1.0.200\",\"compat\":\"1.0.228\",\"latest\":\"1.0.228\",\"kind\":\"Normal\",\"platform\":null},",
            "{\"name\":\"tokio\",\"project\":\"1.40.0\",\"compat\":\"1.40.0\",\"latest\":\"1.48.0\",\"kind\":\"Normal\",\"platform\":null},",
            "{\"name\":\"gone\",\"project\":\"0.5.0\",\"compat\":\"Removed\",\"latest\":\"Removed\",\"kind\":\"Normal\",\"platform\":null},",
            "{\"name\":\"stable\",\"project\":\"2.0.0\",\"compat\":\"---\",\"latest\":\"---\",\"kind\":\"Normal\",\"platform\":null}",
            "]}\n",
            "{\"crate_name\":\"xtask\",\"dependencies\":[]}\n",
        );
        // serde is minor-only lag (latest == compat) → the outdated stage's
        // job, not upgrade-check's. tokio is major lag; gone vanished.
        assert_eq!(
            parse_outdated_major_lag(stdout),
            ["gone 0.5.0->Removed", "tokio 1.40.0->1.48.0"]
        );
        assert!(parse_outdated_major_lag("").is_empty());
        assert!(parse_outdated_major_lag("not json\n").is_empty());
    }

    #[test]
    fn durations_render_compactly() {
        assert_eq!(format_duration(Duration::from_millis(400)), "0.4s");
        assert_eq!(format_duration(Duration::from_secs(12)), "12s");
        assert_eq!(format_duration(Duration::from_secs(184)), "3m04s");
    }

    #[test]
    fn strict_mode_follows_ci_flag() {
        let strict = QualityFlags {
            ci: true,
            ..QualityFlags::default()
        };
        assert!(strict.strict());
    }

    #[test]
    fn allow_skip_matches_exact_ids() {
        let mut f = QualityFlags::default();
        assert!(!f.skip_allowed("deny"));
        push_allow_skip(&mut f, "deny, wasm-check");
        push_allow_skip(&mut f, "deny");
        assert!(f.skip_allowed("deny"));
        assert!(f.skip_allowed("wasm-check"));
        assert!(!f.skip_allowed("fmt"));
        // Valid ids validate silently (typos exit the process).
        validate_allow_skip(&f);
    }

    #[test]
    fn parse_accepts_both_list_forms() {
        let args = [
            "--ci",
            "--allow-skip",
            "deny",
            "--allow-skip=wasm-check",
            "--only=fmt",
        ];
        let owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let f = QualityFlags::parse(&owned);
        assert!(f.ci);
        assert!(f.skip_allowed("deny"));
        assert!(f.skip_allowed("wasm-check"));
        assert!(f.enabled("fmt"));
        assert!(!f.enabled("deny"));
    }

    fn skipped_result(id: &str) -> StageResult {
        StageResult {
            id: id.into(),
            name: id.into(),
            status: Status::Skip,
            note: "missing tool".into(),
            elapsed: Duration::ZERO,
        }
    }

    #[test]
    fn skip_policy_fails_only_unexcused_strict_skips() {
        let strict = QualityFlags {
            ci: true,
            ..QualityFlags::default()
        };
        let results = vec![skipped_result("deny")];
        assert_eq!(skip_exit_code(&results, &strict), 1);
        let excused = QualityFlags {
            ci: true,
            allow_skip: vec!["deny".to_string()],
            ..QualityFlags::default()
        };
        assert_eq!(skip_exit_code(&results, &excused), 0);
        // No skips at all → 0 in any mode (strict here in case the
        // tests themselves run under GITHUB_ACTIONS).
        assert_eq!(skip_exit_code(&[], &strict), 0);
    }

    #[test]
    fn guard_ref_mapping_is_env_free() {
        assert_eq!(baseline_guard_ref_for(None, None), "origin/master");
        assert_eq!(
            baseline_guard_ref_for(Some("push"), Some("master")),
            "origin/master"
        );
        assert_eq!(
            baseline_guard_ref_for(Some("pull_request"), Some("master")),
            "origin/master"
        );
        assert_eq!(
            baseline_guard_ref_for(Some("pull_request"), Some("release")),
            "origin/release"
        );
        assert_eq!(
            baseline_guard_ref_for(Some("pull_request"), None),
            "origin/master"
        );
        assert_eq!(
            baseline_guard_ref_for(Some("pull_request"), Some("")),
            "origin/master"
        );
    }

    #[test]
    fn bump_allowlist_parses_without_env() {
        assert!(parse_bump_allowed(Some("1")));
        assert!(parse_bump_allowed(Some("true")));
        assert!(!parse_bump_allowed(None));
        assert!(!parse_bump_allowed(Some("")));
        assert!(!parse_bump_allowed(Some("0")));
        assert!(!parse_bump_allowed(Some("yes")));
        // The thin env wrapper must agree with the pure parse on the
        // ambient value. Read-only: no other test mutates this var, so
        // parallel runs cannot observe each other (RAT-24 review).
        let raw = std::env::var("ORNIS_BASELINE_BUMP_ALLOWED").ok();
        assert_eq!(baseline_bump_allowed(), parse_bump_allowed(raw.as_deref()));
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::{
        coverage_outcome, coverage_rate, coverage_stage, coverage_verdict, load_coverage_floor,
        StageList, Status, COVERAGE_EPS,
    };

    /// Minimal llvm-cov summary shape: totals resolve to rate/covered/total.
    #[test]
    fn parses_summary_totals() {
        let (rate, covered, count) =
            coverage_rate(r#"{"data": [{"totals": {"lines": {"covered": 3, "count": 4}}}]}"#);
        assert_eq!((covered, count), (3, 4));
        assert!((rate - covered as f64 / count as f64).abs() < COVERAGE_EPS);
    }

    /// Garbage in — zeroed measurement out, never a panic.
    #[test]
    fn rejects_broken_summary() {
        assert_eq!(coverage_rate("not json"), (0.0, 0, 0));
        assert_eq!(coverage_rate("{}"), (0.0, 0, 0));
    }

    /// Scratch baseline: first save, silent in-band equality, lift, regression FAIL.
    #[test]
    fn ratchets_against_scratch_floor() {
        let path = std::env::temp_dir().join(format!(
            "ornis-coverage-verdict-probe-{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        assert!(load_coverage_floor(&path).is_none());
        let (status, _) = coverage_verdict(&path, 0.5, 1, 2);
        assert_eq!(status, Status::Pass);
        let (status, _) = coverage_verdict(&path, 1.0, 2, 2);
        assert_eq!(status, Status::Pass);
        assert!((load_coverage_floor(&path).unwrap_or(0.0) - 1.0).abs() < COVERAGE_EPS);
        let (status, _) = coverage_verdict(&path, 0.0, 0, 2);
        assert_eq!(status, Status::Fail);
        let _ = std::fs::remove_file(&path);
    }

    /// Sub-0.1pp wobble around the floor is noise (adapter/OS-dependent
    /// line counts), not a regression: in-band rates PASS without touching
    /// the file, only out-of-band deviations move the verdict.
    #[test]
    fn tolerance_band_ignores_noise() {
        let path = std::env::temp_dir().join(format!(
            "ornis-coverage-tolerance-probe-{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let (status, _) = coverage_verdict(&path, 0.8, 800, 1000);
        assert_eq!(status, Status::Pass);
        // 0.05 pp below the floor — PASS, file untouched.
        let (status, note) = coverage_verdict(&path, 0.7995, 7995, 10000);
        assert_eq!(status, Status::Pass, "unexpected: {note}");
        assert!((load_coverage_floor(&path).unwrap_or(0.0) - 0.8).abs() < COVERAGE_EPS);
        // 0.3 pp below the floor — FAIL.
        let (status, _) = coverage_verdict(&path, 0.797, 797, 1000);
        assert_eq!(status, Status::Fail);
        let _ = std::fs::remove_file(&path);
    }

    /// A zero-line summary is a collection failure: it FAILs and must never
    /// be saved as a 0.0 floor (which would turn every later run into PASS).
    #[test]
    fn zero_total_lines_fail_closed() {
        let path = std::env::temp_dir().join(format!(
            "ornis-coverage-zero-probe-{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let (status, note) = coverage_verdict(&path, 0.0, 0, 0);
        assert_eq!(status, Status::Fail);
        assert!(note.contains("zero total"), "unexpected: {note}");
        assert!(load_coverage_floor(&path).is_none());
    }

    /// Failure paths never panic: broken collection, unreadable summary,
    /// zero-line summary.
    #[test]
    fn outcome_fails_closed() {
        let dir = std::env::temp_dir().join(format!(
            "ornis-coverage-outcome-probe-{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let missing = dir.join("no-summary.json");
        let _ = std::fs::remove_file(&missing);
        let (status, note) = coverage_outcome(&dir, &missing, false);
        assert_eq!(status, Status::Fail);
        assert!(note.contains("did not finish"), "unexpected: {note}");
        let (status, note) = coverage_outcome(&dir, &missing, true);
        assert_eq!(status, Status::Fail);
        assert!(note.contains("unreadable"), "unexpected: {note}");
        let empty = dir.join("zero-summary.json");
        let _ = std::fs::write(
            &empty,
            r#"{"data": [{"totals": {"lines": {"covered": 0, "count": 0}}}]}"#,
        );
        let (status, note) = coverage_outcome(&dir, &empty, true);
        assert_eq!(status, Status::Fail);
        assert!(note.contains("zero total"), "unexpected: {note}");
        let _ = std::fs::remove_file(&empty);
    }

    /// A deselected stage records nothing (and stays test-covered).
    #[test]
    fn stage_stays_quiet_when_not_selected() {
        let mut stages = StageList::new(
            std::path::Path::new("."),
            1,
            Some(vec!["fmt".to_string()]),
            false,
        );
        coverage_stage(&mut stages);
        assert!(stages.results.is_empty());
    }
}
