# bca — complexity gate via big-code-analysis

> License: `bca` CLI is MPL-2.0, used as external binary it does NOT affect Ornis MIT OR Apache-2.0.
> Do NOT add `big-code-analysis` as Cargo library dependency — only `cargo install` or prebuilt binary.

## What it measures

`bca` (https://github.com/dekobon/big-code-analysis) — hard fork of Mozilla rust-code-analysis —
parses via tree-sitter and computes per-function metrics:

* `cognitive`, `cyclomatic` — paths / readability
* `halstead.effort` — vocabulary-based difficulty
* `abc`, `nargs`, `nexits`, `loc.ploc/sloc`, `nom`, `wmc`

## Integration in Ornis

* Config: `bca.toml` at repo root (auto-discovered), excludes in `.bcaignore`, baseline in `.bca-baseline.toml`.
* `paths` and `exclude_from` are **walk-scope** — they control which files are analysed at all, by `check`,
  `report` and `metrics` alike. `exclude_from` must stay at the top level, not under `[check]`: the latter
  only exempts violations from the gate and leaves the walk picking up binaries and non-source files.
* Gate: `cargo xtask quality` runs `bca check` as stage `[3/7]` if binary exists, else SKIP (same as audit/deny/outdated).
* Wrapper: `cargo xtask bca` automates your manual sequence — install + baseline + report + quality.
* Tests: `xtask/tests/bca_cli.rs` (end-to-end, skip when `bca` is not in PATH) + unit tests for the flag
  parsing in `xtask/src/bca.rs`.
* Total count: 7 base stages (was 6) — fmt, clippy, bca, test, audit, deny, outdated.

### Via xtask (recommended)

```bash
cargo xtask bca --full        # install (if missing) + baseline + report + quality = your 5-step seq
cargo xtask bca --install     # cargo install big-code-analysis-cli --locked
cargo xtask bca --write-baseline
cargo xtask bca --report      # HTML to target/bca/index.html + Markdown
cargo xtask bca               # bca check
```

### Manual (underlying commands)

```bash
cargo install big-code-analysis-cli --locked
bca check                       # uses bca.toml
bca check --write-baseline      # after intentional complexity growth
bca report -O html -o target/bca/index.html
bca diff-baseline old new --format markdown
```

## Thresholds

Chosen from upstream defaults (`bca init`) tuned for physics engine (3251 LOC):

* `cognitive=25`, `cyclomatic=25`, `halstead.effort=150k` global
* Rust override: 20 / 20 / 120k
* JS/TS: 30 / 25

All violations are ratcheted via `.bca-baseline.toml` — the committed baseline
absorbs the current offender set, so only regressions and new offenders fail.
Refresh it after an intentional complexity change:

```bash
bca check --write-baseline
git add .bca-baseline.toml && git commit -m "chore(bca): update baseline"
```

## CI

External tool, installed on demand:

```yaml
- run: cargo install big-code-analysis-cli --locked
- run: bca check
```

Exit codes (the gate treats any non-zero as FAIL): 0 clean, 1 tool error,
2 metric gate exceeded. With `[check] exit_codes = "tiered"` the metric gate
is split for CI branching: 2 = new offenders only, 3 = baseline regressions
only, 4 = mixed, 5 = hard-tier breach under `--tier=soft`.

## Agent feedback loop

For Claude/opencode (and Arena agent):

* Post-edit hook runs `bca check <file>` and feeds `[new]` / `[regr]` rows back into model context.
* Recipe: https://dekobon.github.io/big-code-analysis/recipes/agent-feedback.html
