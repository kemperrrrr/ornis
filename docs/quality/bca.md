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
* Gate: `cargo xtask quality` runs `bca check` as stage `[3/7]` if binary exists, else SKIP (same as audit/deny/outdated).
* Wrapper: `cargo xtask bca` automates your manual sequence — install + baseline + report + quality.
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

All violations are ratcheted via baseline — initial baseline is empty, populate with:

```bash
bca check --write-baseline
git add .bca-baseline.toml && git commit -m "chore(bca): initial baseline"
```

## CI

External tool, installed on demand:

```yaml
- run: cargo install big-code-analysis-cli --locked
- run: bca check
```

Exit codes: 0 clean, 2 new/regression, 1 tool error — same contract as other quality stages.

## Agent feedback loop

For Claude/opencode (and Arena agent):

* Post-edit hook runs `bca check <file>` and feeds `[new]` / `[regr]` rows back into model context.
* Recipe: https://dekobon.github.io/big-code-analysis/recipes/agent-feedback.html
