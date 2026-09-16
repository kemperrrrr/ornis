#!/usr/bin/env python3
"""Replay scalar audit cases for master@203319d (2026-09-16).

Reads pinned Git blobs without checking out master. These are source checks
and Python/double algebra models, NOT execution of Ornis Rust code or its
integration tests. The original audit module supplies unchanged scalar cases.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = "f328bbd0add199c18c61996801c136a0f0f8fa5f"
PREVIOUS = "931d085337d201a4183589f0ee6b78db3ad5ef53"
TARGET = "203319d4b1cc1c7e3ad9f083099f11099570677b"
SOURCE_PATH = "crates/physics/src/avbd.rs"
SOURCE_SHA256 = "7163e1f3e5b2dc6631b9a110909a3c5b590de045390cfe549d0499545652ab32"


def source(revision):
    return subprocess.check_output(
        ["git", "show", f"{revision}:{SOURCE_PATH}"], cwd=ROOT, text=True
    )


def compact(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", text))


def simple_function(text, name):
    """Extract the audited small functions (no braces in their comments/strings)."""
    start = text.index(f"fn {name}(")
    opening = text.index("{", start)
    depth = 0
    for end in range(opening, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return compact(text[start : end + 1])
    raise AssertionError(f"Unclosed function: {name}")


def contact_force(cn, pens, lam, ct, mu):
    """Mirror of the new contact_force, including segment/point projections."""
    normal = min(pens[0] * cn + lam[0], 0.0)
    raw = [pens[1] * ct[0] + lam[1], pens[2] * ct[1] + lam[2]]
    b1, b2 = abs(normal) * mu[0], abs(normal) * mu[1]
    clamp = lambda v, bound: min(max(v, -bound), bound)
    if b1 <= 0.0 or b2 <= 0.0:
        t1 = 0.0 if b1 <= 0.0 else clamp(raw[0], b1)
        t2 = 0.0 if b2 <= 0.0 else clamp(raw[1], b2)
        return [normal, t1, t2]
    s = (raw[0] / b1) ** 2 + (raw[1] / b2) ** 2
    k = 1.0 / math.sqrt(s) if s > 1.0 else 1.0
    return [normal, raw[0] * k, raw[1] * k]


def row_live(c, force, penalty, mass_dt2):
    return abs(c) >= 1e-7 or abs(force) > penalty * 1e-7 or abs(force) > mass_dt2 * 1e-7


def wrap_pi(x):
    return (x + math.pi) % math.tau - math.pi


def main():
    original, previous, target = source(BASE), source(PREVIOUS), source(TARGET)
    digest = hashlib.sha256(target.encode()).hexdigest()
    assert digest == SOURCE_SHA256, "Unexpected target blob"
    unchanged = {
        name: simple_function(original, name) == simple_function(target, name)
        for name in ["row_c", "stamp_row", "quat_diff_vec", "quat_integrate", "eff_inv_mass"]
    }
    assert all(unchanged.values()), unchanged
    assert "let m_red = if ka + kb > 1e-9 { 1.0 / (ka + kb) } else { 0.0 };" in target
    assert "let pen_d = (2.0 * m_red / (DT_STEP * DT_STEP) / npts).min(PENALTY_MAX);" in target
    assert "let f_raw = j.pen_l[2] * c;" in target
    assert "let f_raw = j.pen_l[2] * c;" not in previous
    assert "let f_raw = j.pen_l[2] * c + j.lim_dual;" in previous
    assert simple_function(target, "row_live") == compact("""
        fn row_live(c: f32, f: f32, pen: f32, mass: f32) -> bool {
            c.abs() >= C_EPS || f.abs() > pen * C_EPS || f.abs() > mass * C_EPS
        }
    """)
    # Check the exact locations of the angular wrap in BOTH gear consumers.
    assert "prev_cont[0] + wrap_pi(raw[0] - prev_raw[0])" in target
    assert "Some((prev_raw, prev_cont)) => prev_cont[k] + wrap_pi(raw - prev_raw[k])," in target
    assert "if touching {\n                    pt.lam[1] = f[1];\n                    pt.lam[2] = f[2];" in target

    spec = importlib.util.spec_from_file_location("original_probes", Path(__file__).with_name("algebra_probes.py"))
    old_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_models)

    before = old_models.separated_pair(10)
    after = old_models.separated_pair(10, shared_penalty=True)
    assert abs(after["final_momentum"] + 1.0) < 1e-10
    assert after["final_energy"] < after["initial_energy"]

    cases = []
    for mu, ct, expected in [
        ([0.0, 0.5], [0.0, 0.1], [-10.0, 0.0, 5.0]),
        ([0.5, 0.0], [0.1, 0.0], [-10.0, 5.0, 0.0]),
        ([0.0, 0.0], [0.1, 0.1], [-10.0, 0.0, 0.0]),
        ([0.0, 0.5], [0.0, -0.1], [-10.0, 0.0, -5.0]),
    ]:
        force = contact_force(-0.01, [100.0] * 3, [-9.0, 0.0, 0.0], ct, mu)
        assert force == expected, (force, expected)
        cases.append(dict(mu=mu, force=force))
    slip_force = contact_force(-0.01, [100.0] * 3, [-9.0, 4.0, 0.0], [0.1, 0.0], [0.5, 0.5])
    assert slip_force == [-10.0, 5.0, 0.0]

    assert row_live(0.0, -10.0, 1e9, 3600.0)
    prismatic_al = min(1000.0 * -0.01 + -50.0, 0.0)
    prismatic_now = min(1000.0 * -0.01, 0.0)
    assert prismatic_al == -60.0 and prismatic_now == -10.0

    # Valid angular branch-cut correction versus invalid wrapping of meters.
    angular = 3.1 + wrap_pi(-3.08 - 3.1)
    assert math.isclose(angular, 3.203185307179586, abs_tol=1e-12)
    linear = 0.0 + wrap_pi(4.0 - 0.0)
    assert math.isclose(linear, -2.2831853071795862, abs_tol=1e-12)
    linear_next = linear + wrap_pi(4.01 - 4.0)
    assert math.isclose(linear_next - 4.01, -math.tau, abs_tol=1e-12)

    result = dict(
        scope="Scalar/source audit checks, NOT Rust integration test execution",
        target=TARGET,
        source_sha256=digest,
        unchanged_functions=unchanged,
        A1_rotation=old_models.rotation_probe(),
        A2_mixed_mass={"old": before, "new": after},
        A3_sliding_lambda=slip_force,
        A4_high_penalty_contact={"C": 0.0, "F": -10.0, "K": 1e9, "mass_dt2": 3600.0, "row_live": True},
        A5_prismatic={"augmented_update": prismatic_al, "current_update": prismatic_now},
        A6_degenerate_projection=cases,
        new_gear_issue={"angular_3_10_to_minus_3_08": angular, "linear_0_to_4": linear,
                        "expected_linear": 4.0, "next_linear_4_01": linear_next,
                        "persistent_linear_error": linear_next - 4.01},
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
