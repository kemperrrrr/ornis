#!/usr/bin/env python3
"""Scalar reproductions for the Ornis AVBD audit (2026-09-15).

These checks model the audited formulas, NOT the Rust executable. They use
Python doubles, not glam/f32, and are not engine regression tests. Source
fingerprinting prevents silently attributing old formulas to a newer revision.
No third-party dependencies are required.
"""

import hashlib
import json
import math
from pathlib import Path
import sys

SOURCE_SHA256 = "791210c99a0f76698942c7dd85e39d0c4f9b905718c1319525b7e4cfc31be164"
C_EPS = 1e-7


def contact_force(cn, pens, lam, ct, mu):
    """Mirror of avbd.rs:1657-1675, including the degenerate-ellipse rule."""
    fn = min(pens[0] * cn + lam[0], 0.0)
    raw = [pens[1] * ct[0] + lam[1], pens[2] * ct[1] + lam[2]]
    b1, b2 = abs(fn) * mu[0], abs(fn) * mu[1]
    s = (raw[0] / b1) ** 2 + (raw[1] / b2) ** 2 if b1 > 0 and b2 > 0 else math.inf
    k = 1 / math.sqrt(s) if s > 1 else 1.0
    return [fn, raw[0] * k, raw[1] * k]


def rotation_probe():
    """One body, local anchor X, row Y, rotation theta about Z, no translation."""
    theta = 0.001
    # Reference manifold.cpp:91 uses center displacement and J_ang*dq_ang.
    # maths.h:269 uses 2*vector(q*q0^-1), not the exact rotation log.
    reference_chord = math.cos(theta) * (2 * math.sin(theta / 2))
    reference_with_log = math.cos(theta) * theta
    # Ornis row_c includes rotated anchor displacement AND J_ang*dq_ang.
    ornis = math.sin(theta) + math.cos(theta) * theta
    assert 1.999 < ornis / reference_chord < 2.001
    # At theta=0 the finite-difference derivative is 2, but stamped J_ang=1.
    e = 1e-6
    residual = lambda t: math.sin(t) + t * math.cos(t)
    derivative = (residual(e) - residual(-e)) / (2 * e)
    assert abs(derivative - 2.0) < 1e-10
    return dict(theta=theta, reference_chord=reference_chord,
                reference_with_log=reference_with_log, ornis=ornis,
                residual_derivative_at_zero=derivative, stamped_jacobian=1.0)


def sliding_dual_probe():
    """Outside cone: reference commits clipped F, Ornis skips tangent writes."""
    pen, lam, cn, ct, mu = [100.0] * 3, [-9.0, 4.0, 0.0], -0.01, [0.1, 0.0], [0.5, 0.5]
    f = contact_force(cn, pen, lam, ct, mu)
    raw = [pen[1] * ct[0] + lam[1], pen[2] * ct[1] + lam[2]]
    b1, b2 = abs(f[0]) * mu[0], abs(f[0]) * mu[1]
    s = (raw[0] / b1) ** 2 + (raw[1] / b2) ** 2
    updated = lam.copy()
    if abs(cn) >= C_EPS:
        updated[0] = f[0]
    if s <= 1:
        for i in range(2):
            if abs(ct[i]) >= C_EPS:
                updated[i + 1] = f[i + 1]
    assert f == [-10.0, 5.0, 0.0]
    assert updated == [-10.0, 4.0, 0.0]
    return dict(raw_tangent=raw, cone_radius=b1, reference_lambda=f, ornis_lambda=updated)


def zero_residual_probe():
    """C=0 does not imply zero force in an augmented-Lagrangian method."""
    c, lam, pen, jacobian = 0.0, -10.0, 100.0, 1.0
    reference_gradient = jacobian * (pen * c + lam)
    ornis_gradient = 0.0 if abs(c) < C_EPS else reference_gradient
    assert reference_gradient == -10.0 and ornis_gradient == 0.0
    return dict(C=c, lambda_old=lam, reference_gradient=reference_gradient,
                ornis_gradient=ornis_gradient, reference_hessian=100.0, ornis_hessian=0.0)


def prismatic_dual_probe():
    """Lower limit, still violated, state unchanged between primal and dual."""
    c, penalty, lam = -0.01, 1000.0, -50.0
    primal_force = min(penalty * c + lam, 0.0)
    reference_next = min(penalty * c + lam, 0.0)
    ornis_next = min(penalty * c, 0.0)
    assert primal_force == reference_next == -60.0 and ornis_next == -10.0
    return dict(C=c, penalty=penalty, lambda_old=lam, primal_force=primal_force,
                reference_next_lambda=reference_next, ornis_next_lambda=ornis_next)


def separated_pair(iterations, shared_penalty=False):
    """1D reduction of solve_body:1712,1767-1799,2352-2356 and BDF1.

    Two collinear spheres: A (handle 0) at x=1.01, B (handle 1) at x=0,
    radii 0.5; masses 1 and 100; velocities -1 and 0; gravity/rotation=0.
    The gap is 0.01 (within GEN_MARGIN), normal +X, one witness point.
    Approach*h > gap, velocity*h < TOI gate. Thus the separated-damper
    branch is active for all iterations and other rows/prepasses are inert.
    A shared penalty is an algebraic control, not a proposed full solver.
    """
    h = 1 / 60
    mass, velocity = [1.0, 100.0], [-1.0, 0.0]
    inertial = [v * h for v in velocity]
    displacement = inertial.copy()
    common_penalty = 2 / (1 / mass[0] + 1 / mass[1]) / (h * h)
    for _ in range(iterations):
        for body in [1, 0]:  # Reverse handle order, as in step_inner.
            m_dt2 = mass[body] / (h * h)
            c = displacement[0] - displacement[1]
            if abs(c) < C_EPS:
                displacement[body] = inertial[body]
                continue
            penalty = common_penalty if shared_penalty else min(2 * m_dt2, 1e10)
            force = min(penalty * c, 0.0)
            sign = 1 if body == 0 else -1
            gradient = m_dt2 * (displacement[body] - inertial[body]) + sign * force
            displacement[body] -= gradient / (m_dt2 + penalty)
    final_velocity = [d / h for d in displacement]
    return dict(iterations=iterations, velocity=final_velocity,
                initial_momentum=sum(m * v for m, v in zip(mass, velocity)),
                final_momentum=sum(m * v for m, v in zip(mass, final_velocity)),
                initial_energy=sum(m * v * v / 2 for m, v in zip(mass, velocity)),
                final_energy=sum(m * v * v / 2 for m, v in zip(mass, final_velocity)))


def degenerate_friction_probe():
    f = contact_force(-0.01, [100.0] * 3, [-9.0, 0.0, 0.0], [0.0, 0.1], [0.0, 0.5])
    assert f == [-10.0, 0.0, 0.0]
    return dict(mu=[0.0, 0.5], ornis_force=f, expected_force=[-10.0, 0.0, 5.0])


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 algebra_probes.py /path/to/ornis")
    source = Path(sys.argv[1]) / "crates/physics/src/avbd.rs"
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise SystemExit("Source differs from audited snapshot; do not attribute these scalar models to current Rust code.")
    shell10 = separated_pair(10)
    shell100 = separated_pair(100)
    control = separated_pair(100, shared_penalty=True)
    assert shell10["final_energy"] > 8.0
    assert abs(shell100["final_momentum"] + 40.6) < 1e-10
    assert abs(control["final_momentum"] + 1.0) < 1e-10
    print(json.dumps(dict(
        scope="Scalar algebra checks, NOT an execution of Ornis Rust code",
        source_sha256=actual_hash,
        rotation=rotation_probe(),
        sliding_dual=sliding_dual_probe(),
        zero_residual=zero_residual_probe(),
        prismatic_dual=prismatic_dual_probe(),
        separated_pair_10_iterations=shell10,
        separated_pair_100_iterations=shell100,
        shared_penalty_control=control,
        degenerate_friction=degenerate_friction_probe(),
    ), indent=2))


if __name__ == "__main__":
    main()
