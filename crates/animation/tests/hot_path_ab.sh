#!/usr/bin/env bash
# Frame-level A/B for the animation hot path (PLAN.md, фаза E, п. 5):
# runs crates/animation/tests/hot_path_bench.rs on the merge base and on
# the working tree in --release and prints the comparison table.
#
#   ./crates/animation/tests/hot_path_ab.sh [base-ref]
#
# `base-ref` defaults to the merge base with master. Requires a checkout
# with a clean-enough git state (the script never touches your tree: the
# base side runs from a throwaway `git worktree`).
set -euo pipefail

cd "$(dirname "$0")/../../.."
base_ref="${1:-$(git merge-base HEAD origin/master 2>/dev/null || git merge-base HEAD master)}"
harness="crates/animation/tests/hot_path_bench.rs"

worktree="$(mktemp -d)/base"
git worktree add --detach "${worktree}" "${base_ref}"
trap 'git worktree remove --force "${worktree}"; rm -rf "${worktree%/base}"' EXIT
cp "${harness}" "${worktree}/${harness}"

# Every build gets its own target dir: a caller's global CARGO_TARGET_DIR
# would otherwise merge both sides into one cache and compare base with base.
tmp="${worktree%/base}"
echo "== base (${base_ref})"
(cd "${worktree}" && CARGO_TARGET_DIR="${tmp}/target-base" cargo test --release -p ornis-animation --test hot_path_bench -- --ignored --nocapture) | tee "${tmp}/base.txt"
echo "== head (working tree)"
CARGO_TARGET_DIR="${tmp}/target-head" cargo test --release -p ornis-animation --test hot_path_bench -- --ignored --nocapture | tee "${tmp}/head.txt"

# Raw logs live under ${tmp} too: trap removes them, and parallel runs of
# the script never share files.
BASE_LOG="${tmp}/base.txt" HEAD_LOG="${tmp}/head.txt" python3 - <<'PY'
import os
import re

def parse(path):
    scenes = {}
    pattern = re.compile(
        r"anim-hot-path scene=(\S+) frames=\d+ "
        r"min_us=([\d.]+) median_us=([\d.]+) mean_us=([\d.]+) p90_us=([\d.]+)"
    )
    with open(path) as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                scenes[match.group(1)] = tuple(float(v) for v in match.groups()[1:])
    return scenes

base = parse(os.environ["BASE_LOG"])
head = parse(os.environ["HEAD_LOG"])
print("| scene | base µs (med) | head µs (med) | speedup | base p90 | head p90 |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
for name in [n for n in base if n in head]:
    _, b_med, _, b_p90 = base[name]
    _, h_med, _, h_p90 = head[name]
    speedup = b_med / h_med if h_med else float("inf")
    print(
        f"| {name} | {b_med:.2f} | {h_med:.2f} | **{speedup:.2f}×** | "
        f"{b_p90:.2f} | {h_p90:.2f} |"
    )
PY
