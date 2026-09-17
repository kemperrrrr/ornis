#!/usr/bin/env python3
"""AVBD source/scalar recheck for c97089a (2026-09-17).

Reads pinned Git blobs, does not check out master. Python/double quaternion
and scalar models are NOT execution of Ornis Rust code or its test suite.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
TARGET = "c97089a9d22ac58ce895d89c75415b3b5abcc01c"
PREVIOUS = "203319d4b1cc1c7e3ad9f083099f11099570677b"
SHA256 = "a1455b1ba30afcfbcc6fabbf9a5a463752afd1ee64d018c9deff5e3aaedb306b"


def blob(revision, path="crates/physics/src/avbd.rs"):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)


def compact(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", text))


def function(text, name):
    """Extract these audited functions after removing line comments."""
    text = re.sub(r"//[^\n]*", "", text)
    start = text.index(f"fn {name}(")
    opening = text.index("{", start)
    depth = 0
    for end in range(opening, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return compact(text[start : end + 1])
    raise AssertionError(name)


def quat_mul(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return (w*X + x*W + y*Z - z*Y, w*Y - x*Z + y*W + z*X,
            w*Z + x*Y - y*X + z*W, w*W - x*X - y*Y - z*Z)


def conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def rotation(axis, angle):
    s = math.sin(angle / 2)
    return tuple(v*s for v in axis) + (math.cos(angle / 2),)


def rotate(q, v):
    return quat_mul(quat_mul(q, tuple(v)+(0.0,)), conjugate(q))[:3]


def rotation_vector(q):
    """Unit-quaternion special case of quat_diff_vec(q, identity)."""
    v = q[:3] if q[3] >= 0 else tuple(-x for x in q[:3])
    s = min(math.sqrt(sum(x*x for x in v)), 1.0)
    return (0.0, 0.0, 0.0) if s < 1e-9 else tuple(x*2*math.asin(s)/s for x in v)


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def row_live(c, force, penalty, body_scale):
    return abs(c) >= 1e-7 or abs(force) > penalty*1e-7 or abs(force) > body_scale*1e-7


def frame_case(pitch):
    # Both bodies start with the same orientation, so q_ref=identity.
    # Rotating both poses together must not change a relative local-Z lock.
    qa = rotation((0, 1, 0), pitch)
    qb = quat_mul(qa, rotation((0, 0, 1), 0.3))
    relative = quat_mul(conjugate(qa), qb)
    local_error = rotation_vector(relative)
    world_axis = rotate(qa, (0, 0, 1))
    current = dot(local_error, world_axis)  # Current SixDof Locked C.
    expected = local_error[2]              # Error about A's local Z.
    assert math.isclose(expected, 0.3, abs_tol=1e-12)
    return dict(pitch=pitch, local_error=local_error, world_axis=world_axis,
                current_C=current, expected_C=expected,
                current_row_active_at_K_1_lambda_0=row_live(current, current, 1, 3600))


def main():
    src, prev = blob(TARGET), blob(PREVIOUS)
    assert hashlib.sha256(src.encode()).hexdigest() == SHA256
    code = compact(src)
    assert "letd_a=a.position-self.pos0[pair.a]" in function(src, "row_c_levers")
    assert "letd_b=b.position-self.pos0[pair.b]" in function(src, "row_c_levers")
    assert "Some((prev_raw,prev_cont))ifangular=>" in function(src, "gear_sides")
    assert "ifang[0]" in function(src, "update_gear_mem") and "ifang[1]" in function(src, "update_gear_mem")
    dual = function(src, "dual_update")
    assert "letf_raw=j.pen_l[2]*c+j.lim_dual;" in dual
    assert "letf_raw=j.pen_l[2]*c;" not in dual
    assert "iftouching{pt.lam[1]=f[1];pt.lam[2]=f[2];" in dual
    assert function(src, "contact_force") == function(prev, "contact_force")
    assert "letpen_d=(2.0*m_red/(DT_STEP*DT_STEP)/npts).min(PENALTY_MAX);" in code
    assert function(src, "row_live") == compact("""
        fn row_live(c: f32, f: f32, pen: f32, mass: f32) -> bool {
            c.abs() >= C_EPS || f.abs() > pen * C_EPS || f.abs() > mass * C_EPS
        }
    """)
    primal = function(src, "solve_body")
    assert "letf=j.pen_a[k]*c+j.lam_a[k];if!row_live(c,f,j.pen_a[k],m_dt2)" in primal
    mixed_frame = "letqrel=a.orientation.conjugate()*b.orientation;letdiff=quat_diff_vec(qrel,j.q_ref);"
    assert mixed_frame in primal and mixed_frame in dual
    assert "letc=diff.dot(dir);" in primal and "letc=diff.dot(dir);" in dual
    joint_tests = blob(TARGET, "crates/physics/src/avbd_joint_tests.rs")
    assert "0.3 * 0.7f32.cos()" in joint_tests

    spec = importlib.util.spec_from_file_location("audit_models", Path(__file__).with_name("algebra_probes.py"))
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    damper = models.separated_pair(10, shared_penalty=True)
    assert abs(damper["final_momentum"] + 1) < 1e-10
    theta = 0.001
    rotation_old = math.sin(theta) + theta*math.cos(theta)
    rotation_new = theta*math.cos(theta)
    assert math.isclose(rotation_old / rotation_new, 2.0, abs_tol=1e-6)
    gear_linear = [4.0, 4.01, -4.0, 100.0]  # _ => raw on linear sides.
    angular_wrap = 3.1 + ((-3.08 - 3.1 + math.pi) % math.tau - math.pi)
    assert math.isclose(angular_wrap, 3.203185307179586, abs_tol=1e-12)

    frame = [frame_case(pitch) for pitch in (0.0, 0.7, math.pi/2)]
    assert abs(frame[-1]["current_C"]) < 1e-7
    assert not frame[-1]["current_row_active_at_K_1_lambda_0"]

    # Sphere inertia matches shape.rs: I = 0.4*m*r*r. A pure angular row
    # must judge body response on I/h^2, not on the linear m/h^2 scale.
    mass, radius, dt = 1.0, 0.01, 1/60
    inertia = 0.4*mass*radius*radius
    mass_dt2, inertia_dt2 = mass/(dt*dt), inertia/(dt*dt)
    torque, penalty = 1e-4, 1e4
    current_gate = row_live(0, -torque, penalty, mass_dt2)
    angular_scale_gate = row_live(0, -torque, penalty, inertia_dt2)
    assert not current_gate and angular_scale_gate
    # At balanced equilibrium: inertia gradient +tau, warm reaction -tau.
    # Dropping the entire row removes both its force and its Hessian.
    balanced_step = -(torque-torque)/(inertia_dt2+penalty)
    dropped_row_step = -torque/inertia_dt2
    assert abs(dropped_row_step) > 6000*1e-7

    print(json.dumps(dict(
        scope="Pinned source checks and Python/double models, NOT Rust execution",
        target=TARGET, source_sha256=SHA256,
        A1_rotation={"old": rotation_old, "new": rotation_new},
        A2_momentum=damper,
        A4_original_example={"linear_delta": 10/(3600+100),
                             "high_K_contact_kept": row_live(0, -10, 1e9, 3600)},
        A5_prismatic={"old_bare_update": min(1000*-0.01, 0),
                      "new_augmented_update": min(1000*-0.01-50, 0)},
        gear={"linear_outputs": gear_linear, "angular_continuous": angular_wrap},
        sixdof_frame_cases=frame,
        angular_gate={"mass": mass, "radius": radius, "inertia": inertia,
                      "mass_dt2": mass_dt2, "inertia_dt2": inertia_dt2,
                      "torque": torque, "penalty": penalty,
                      "current_gate": current_gate, "inertia_scale_gate": angular_scale_gate,
                      "balanced_step_rad": balanced_step, "dropped_row_step_rad": dropped_row_step},
    ), indent=2))


if __name__ == "__main__":
    main()
