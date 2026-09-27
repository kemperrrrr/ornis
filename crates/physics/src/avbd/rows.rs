//! Contact discovery and row assembly for the AVBD engine.
//!
//! Owns contact-pair discovery/generation, the Taylor contact-row helpers and
//! the shared scalar kernels (assembly-exclusive 3x3 helpers plus quaternion,
//! limit, inertia and liveness helpers also used by [`assembly`](super::assembly),
//! [`joints_ext`](super::joints_ext) and the step pipeline). Joint rows and
//! dual updates live in [`joints_ext`](super::joints_ext). Moved verbatim
//! from `avbd.rs` (phase 3).

use super::*;
use crate::distance::box_box_signed_gap;
use glam::Mat3;
use std::f32::consts::TAU;

/// Anisotropic contact frame (ODE `fdir1`/`mu`/`mu2` parity, local mirror of
/// the builtin rule): body A wins `t1` (its local dir to world, projected
/// onto the plane ⊥ `n`); degenerate projections fall back to the default
/// basis. Per-axis coefficients take the `max` across the pair; a body
/// without a direction contributes `(friction, friction)`.
///
/// Directions validate through the shared typed view
/// ([`RigidBody::friction_frame`](crate::body::RigidBody::friction_frame)),
/// like the sequential-impulse path — an invalid axis is isotropic here,
/// and a checked error at [`RigidBody::set_friction_axis`](crate::body::RigidBody::set_friction_axis)
/// time for new input.
///
/// With all defaults the frame is exactly `tangent_basis(n)` with
/// `mu == mu2`, routing through the legacy circular cone bit-identically.
pub(super) fn friction_frame(a: &RigidBody, b: &RigidBody, n: Vec3) -> (Vec3, f32, f32) {
    let pick_dir = |body: &RigidBody| -> Option<Vec3> {
        let axis = body.friction_frame().ok()?.axis()?.get();
        let world = body.orientation * axis;
        let proj = world - n * world.dot(n);
        crate::invariants::UnitVec3::normalize_checked(proj).map(|u| u.get())
    };
    let t1 = pick_dir(a)
        .or_else(|| pick_dir(b))
        .unwrap_or_else(|| tangent_basis(n).0);
    let axis = |body: &RigidBody| -> (f32, f32) {
        if body.friction_dir.is_some() {
            (body.friction, body.friction_transverse)
        } else {
            (body.friction, body.friction)
        }
    };
    let (a1, a2) = axis(a);
    let (b1, b2) = axis(b);
    (t1, a1.max(b1), a2.max(b2))
}
/// Outer product `a (x) b` as a row-major 3x3.
pub(super) fn outer(a: Vec3, b: Vec3) -> [[f32; 3]; 3] {
    let av = [a.x, a.y, a.z];
    let bv = [b.x, b.y, b.z];
    [
        [av[0] * bv[0], av[0] * bv[1], av[0] * bv[2]],
        [av[1] * bv[0], av[1] * bv[1], av[1] * bv[2]],
        [av[2] * bv[0], av[2] * bv[1], av[2] * bv[2]],
    ]
}

/// Geometric stiffness of a ball-socket anchor (official
/// `geometricStiffnessBallSocket`): the Newton correction for how the lever
/// arm itself rotates. Skipping it (as a "second-order term") leaves the
/// truncated Hessian pointing the wrong way for follower forces through
/// long levers — the joint spins up exponentially. Row-major like theirs.
pub(super) fn geometric_stiffness_ball_socket(k: usize, v: Vec3) -> [[f32; 3]; 3] {
    let arr = v.to_array();
    let mut m = [[0.0f32; 3]; 3];
    m[0][0] = -arr[k];
    m[1][1] = -arr[k];
    m[2][2] = -arr[k];
    m[0][k] += arr[0];
    m[1][k] += arr[1];
    m[2][k] += arr[2];
    m
}

/// Column-length diagonal (official `diagonalize`).
pub(super) fn diagonalize(m: [[f32; 3]; 3]) -> Vec3 {
    Vec3::new(
        Vec3::new(m[0][0], m[1][0], m[2][0]).length(),
        Vec3::new(m[0][1], m[1][1], m[2][1]).length(),
        Vec3::new(m[0][2], m[1][2], m[2][2]).length(),
    )
}
/// Hamilton product on raw components (row-major convention, matching the
/// official demo's `maths.h` so the rotation helpers below are exact ports).
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// Rotation composition `exp(v/2) * q` for a rotation vector `v`
/// (angle = `|v|` about `v/|v|`).
///
/// Exact-exponential composition (NOT first-order Euler): the old chord
/// `q + 0.5*w*q` loses ~|w*dt|^2/2 of angle per step (measured 0.25%/
/// step spin decay at 10 rad/s — free spin 10 -> 7.39 in 120 steps), and the
/// loss is in the addition, not the renormalization (scaling preserves
/// angle). The closed-form `exp(v/2)*q` has only fp error (~1e-7, unbiased),
/// so no systematic decay; the trailing `normalize` is a pure drift guard.
/// One sincos per body per step is negligible next to the 6x6 solves.
pub(super) fn quat_integrate(q: Quat, v: Vec3) -> Quat {
    let theta = v.length();
    if theta < 1e-9 {
        return q;
    }
    let (s, c) = (0.5 * theta).sin_cos();
    let k = s / theta;
    let dq = quat_mul([v.x * k, v.y * k, v.z * k, c], q.to_array());
    Quat::from_xyzw(dq[0], dq[1], dq[2], dq[3]).normalize()
}

/// Relative-rotation vector (official `quat - quat`): exact angle-axis
/// recovery `2*asin(|xyz|)`, not the chord `2*xyz` (which under-reads by
/// ~θ²/24 — 0.1%/step of BDF1 spin decay at 10 rad/s through warmstart
/// feedback). Short-path convention via `w < 0` negation; identical to the
/// chord for small angles up to fp error. Linear component differences
/// under-report spin by ~2x and mishandle large angles; this is what their
/// Jacobians expect.
pub(super) fn quat_diff_vec(a: Quat, b: Quat) -> Vec3 {
    let qa = a.to_array();
    let qb = b.to_array();
    let r = quat_mul(qa, [-qb[0], -qb[1], -qb[2], qb[3]]);
    // Exact angle-axis recovery `2*asin(|xyz|)`, not the chord `2*xyz`
    // (which under-reads by ~θ²/24 — 0.1%/step of BDF1 spin decay at
    // 10 rad/s through warmstart feedback). Short-path convention via
    // `w < 0` negation; identical to the chord for small angles up to fp
    // error, so Jacobian terms are unaffected.
    let mut xyz = Vec3::new(r[0], r[1], r[2]);
    if r[3] < 0.0 {
        xyz = -xyz;
    }
    let s = xyz.length().min(1.0);
    if s < 1e-9 {
        return Vec3::ZERO;
    }
    xyz * (2.0 * s.asin() / s)
}

/// Wrap an angle to [-PI, PI] (official limit bookkeeping).
pub(super) fn wrap_pi(x: f32) -> f32 {
    (x + PI).rem_euclid(TAU) - PI
}

/// Which travel bound (if any) a joint violates: `Some(true)` = lower,
/// `Some(false)` = upper, `None` = freely inside the window. Mirrors the
/// official `hinge_limit_state` (one-sided velocity blocks + slop).
pub(super) fn limit_state(value: f32, lo: f32, hi: f32, slop: f32) -> Option<bool> {
    if value <= lo + slop {
        Some(true)
    } else if value >= hi - slop {
        Some(false)
    } else {
        None
    }
}

/// A projected inequality remains active while its multiplier unwinds,
/// even after the pose re-enters the window. Dropping it immediately is
/// an artificial bounce/limit cycle, not the augmented update.
pub(super) fn warm_limit_state(
    value: f32,
    lo: f32,
    hi: f32,
    slop: f32,
    force: f32,
) -> Option<bool> {
    limit_state(value, lo, hi, slop).or(if force < 0.0 {
        Some(true)
    } else if force > 0.0 {
        Some(false)
    } else {
        None
    })
}

/// Relax only pre-existing penetration, not free travel inside a limit.
/// Correcting all old error instantly converts residual error into rebound.
pub(super) fn regularized_limit(value: f32, initial: f32, bound: f32, lower: bool) -> f32 {
    let old = initial - bound;
    let violation = if lower { old.min(0.0) } else { old.max(0.0) };
    value - bound - ALPHA * violation
}

/// World-space inertia tensor from a body-frame diagonal and orientation.
pub(super) fn world_inertia(inertia: Vec3, rot: Quat) -> [[f32; 3]; 3] {
    let r = Mat3::from_quat(rot);
    let cols = [r.x_axis, r.y_axis, r.z_axis];
    let mut m = [[0.0f32; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let ca = [cols[0][a], cols[1][a], cols[2][a]];
            let cb = [cols[0][b], cols[1][b], cols[2][b]];
            m[a][b] =
                ca[0] * inertia.x * cb[0] + ca[1] * inertia.y * cb[1] + ca[2] * inertia.z * cb[2];
        }
    }
    m
}

/// Analytic inverse of a symmetric 3x3 (torque path). Falls back to the
/// guarded diagonal when near-singular.
pub(super) fn inverse_symmetric(m: [[f32; 3]; 3], diag: Vec3) -> [[f32; 3]; 3] {
    let (a, b, c) = (m[0][0], m[0][1], m[0][2]);
    let (d, e, f) = (m[1][1], m[1][2], m[2][2]);
    let det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    if det.abs() < 1e-12 {
        let inv = Vec3::new(
            if diag.x > 0.0 { 1.0 / diag.x } else { 0.0 },
            if diag.y > 0.0 { 1.0 / diag.y } else { 0.0 },
            if diag.z > 0.0 { 1.0 / diag.z } else { 0.0 },
        );
        return [[inv.x, 0.0, 0.0], [0.0, inv.y, 0.0], [0.0, 0.0, inv.z]];
    }
    let id = 1.0 / det;
    [
        [
            (d * f - e * e) * id,
            (c * e - b * f) * id,
            (b * e - c * d) * id,
        ],
        [
            (c * e - b * f) * id,
            (a * f - c * c) * id,
            (b * c - a * e) * id,
        ],
        [
            (b * e - c * d) * id,
            (b * c - a * e) * id,
            (a * d - b * b) * id,
        ],
    ]
}

pub(super) fn mat3_vec(m: [[f32; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

/// Bounding radius for the O(n2) prefilter (exact for the G1-G4 shapes,
/// conservative estimates elsewhere fall back to always-narrowphase).
fn bound_radius(shape: &Shape) -> f32 {
    match shape {
        Shape::Sphere { radius } => *radius,
        Shape::Box { half_extents } => half_extents.length(),
        Shape::Capsule {
            radius,
            half_height,
        } => half_height + radius,
        Shape::Cylinder {
            radius,
            half_height,
        }
        | Shape::Cone {
            radius,
            half_height,
        } => (radius * radius + half_height * half_height).sqrt(),
        Shape::ConvexHull(hull) => hull
            .vertices()
            .iter()
            .map(|v| v.length())
            .fold(0.0f32, f32::max),
        Shape::Heightfield(_) | Shape::TriMesh(_) => f32::INFINITY,
    }
}

/// Local-space corners of a box (empty for every other shape).
fn box_corners(shape: &Shape) -> Vec<Vec3> {
    match shape {
        Shape::Box { half_extents: h } => {
            let mut out = Vec::with_capacity(8);
            for &sx in &[-1.0f32, 1.0] {
                for &sy in &[-1.0f32, 1.0] {
                    for &sz in &[-1.0f32, 1.0] {
                        out.push(Vec3::new(sx * h.x, sy * h.y, sz * h.z));
                    }
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Whether the pair may ever produce force (layer/mask mutual filter).
pub(super) fn pair_allowed(a: &RigidBody, b: &RigidBody) -> bool {
    (a.collision_layer & b.collision_mask) != 0 && (b.collision_layer & a.collision_mask) != 0
}

/// Smallest shape dimension (builtin TOI parity): a body whose step
/// displacement exceeds HALF of this sweeps `cast_shape` instead of moving
/// blindly. Apex-point shapes (cone/hull/heightfield/trimesh) halve again.
pub(super) fn shape_min_dimension(shape: &Shape) -> f32 {
    match shape {
        Shape::Sphere { radius } => *radius,
        Shape::Box { half_extents } => half_extents.min_element(),
        Shape::Capsule { radius, .. } => *radius,
        Shape::Cylinder {
            radius,
            half_height,
        } => radius.min(*half_height),
        Shape::Cone {
            radius,
            half_height,
        } => 0.5 * radius.min(*half_height),
        Shape::ConvexHull(hull) => 0.5 * hull.min_extent(),
        Shape::Heightfield(hf) => 0.5 * hf.cell(),
        Shape::TriMesh(mesh) => 0.5 * mesh.min_feature(),
    }
}

/// Effective inverse mass: only true dynamics participate in the solve.
pub(super) fn eff_inv_mass(b: &RigidBody) -> f32 {
    if b.body_type == BodyType::Dynamic {
        b.inv_mass
    } else {
        0.0
    }
}

/// Keep every nonzero reaction, including a warm multiplier at zero error.
/// Force/torque magnitudes cannot be compared with a position tolerance:
/// the coupled 6x6 mass/inertia system determines their motion. The C-only
/// tolerance is used by dual accumulation and penalty growth, not to erase
/// a holding force (especially on small-inertia bodies).
pub(super) fn row_live(c: f32, f: f32) -> bool {
    c.abs() >= C_EPS || f != 0.0
}

impl AvbdEngine {
    /// Body count above which pair discovery runs on the rayon pool.
    /// Below it the sequential loop is cheaper than task dispatch; both
    /// feed the same ordered merge, so results are identical either way.
    const PAR_DISCOVERY_BODIES: usize = 256;

    /// Pure per-pair discovery (no `&mut`, safe under rayon): prefilter,
    /// witness distance, signed gap, frame, manifold points and the wake
    /// vote. Returns `None` for pairs that stay unknown this step.
    fn discover_pair(&self, ia: usize, ib: usize) -> Option<Discovered> {
        let (a, b) = (&self.bodies[ia], &self.bodies[ib]);
        if !pair_allowed(a, b) {
            return None;
        }
        // No-collide for pin-jointed bodies (builtin `joint_pairs`
        // parity, narrowed: Ball/Revolute/Prismatic/Distance/Wheel — a
        // hinge pin passes through its mount, so contact friction there
        // is a phantom brake on the joint. Measured: a motor-driven
        // hinge buried 0.2 in its mount never turned — the mount's spin
        // friction saturated the motor. Triggers still report overlap
        // below; gears carry no entry, so geared bodies keep colliding
        // like in the builtin.
        //
        // Fixed/SixDof are EXCLUDED (weld-like assemblies): their tests
        // bury boxes by construction and the joint rows alone do not
        // hold the assembly (measured: SixDof spin-clamp overshoots 4x
        // without its contact) — there the contact is structural, not
        // a brake. (A SixDof with free axes sliding through overlap
        // would want no-collide too; no such scene exists — revisit if
        // one appears.)
        if !a.is_trigger
            && !b.is_trigger
            && self.joints.iter().any(|j| {
                matches!(
                    j.kind,
                    AvbdJointKind::Ball
                        | AvbdJointKind::Revolute
                        | AvbdJointKind::Prismatic
                        | AvbdJointKind::Distance
                        | AvbdJointKind::Wheel
                ) && ((j.a == ia && j.b == ib) || (j.a == ib && j.b == ia))
            })
        {
            return None;
        }
        let ra = bound_radius(&a.shape);
        let rb = bound_radius(&b.shape);
        if (a.position - b.position).length() > ra + rb + GEN_MARGIN {
            return None;
        }
        let d = shape_distance(
            ShapeRef {
                shape: &a.shape,
                pos: a.position,
                rot: a.orientation,
            },
            ShapeRef {
                shape: &b.shape,
                pos: b.position,
                rot: b.orientation,
            },
        );
        // Signed gap: the vertex/face box oracle is UNSIGNED
        // (overlap bottoms out at zero, shallow burial reads back
        // as separation), so Box-Box pairs use the SAT gap here.
        // Every other arm is already signed. All gates below
        // (trigger, creation, shell, wake backstop, touch) read
        // this, never the raw oracle distance.
        let signed = match (&a.shape, &b.shape) {
            (Shape::Box { half_extents: ha }, Shape::Box { half_extents: hb }) => {
                box_box_signed_gap(
                    a.position,
                    a.orientation,
                    *ha,
                    b.position,
                    b.orientation,
                    *hb,
                )
            }
            _ => d.dist,
        };
        if a.is_trigger || b.is_trigger {
            return Some(Discovered {
                ia,
                ib,
                trigger: true,
                exists: false,
                normal: Vec3::Y,
                mu: [0.0; 2],
                signed,
                fresh: Vec::new(),
                wake_vote: false,
            });
        }
        let mut normal = d.point_a - d.point_b;
        // Penetration witnesses point opposite the separating normal. This
        // also resolves coincident-center containment without relying on
        // the otherwise ambiguous center-to-center direction.
        if d.dist < 0.0 {
            normal = -normal;
        }
        if normal.length_squared() < 1e-16 {
            normal = a.position - b.position;
        }
        let mut normal = normal.normalize_or(Vec3::Y);
        // Persistent normal per pair: witness directions flip sign at
        // first touch (separated vs penetrating closest points), which
        // would turn a compressive lambda tensile and catapult the
        // bodies. New pairs orient by body centers (B -> A); live
        // pairs keep their stored direction (spike/official lesson:
        // the contact frame must be stable, like collide() face
        // normals, not a live witness direction).
        //
        // NOTE: no deep-penetration center fallback here. Buried
        // witnesses can be orthogonal garbage (a hinge arm buried 0.2
        // in its mount was born with a +X normal), but snapping them
        // to body centers was ablated: it breaks resting sphere
        // contacts. The garbage only ever mattered for jointed pairs,
        // and those no longer collide (no-collide below), so the
        // fallback's cure was worse than the disease.
        if let Some(old) = self.pairs.iter().find(|p| p.a == ia && p.b == ib) {
            if normal.dot(old.n) < 0.0 {
                normal = -normal;
            }
        } else if normal.dot(a.position - b.position) < 0.0 {
            normal = -normal;
        }
        let exists = self.pairs.iter().any(|p| p.a == ia && p.b == ib);
        // Swept creation: a fast approach covers `approach*dt`
        // this step, so the pair must exist (as a damper shell)
        // before burial — otherwise the body band-skips from "no
        // pair" into a deep spring pair with fresh penalties and
        // tunnels (measured: 10 m/s vs the floor). Slow pairs keep
        // the old near-touch gate bit-identically.
        let approach = (b.velocity - a.velocity).dot(normal).max(0.0);
        let reach = GEN_MARGIN + approach * DT_STEP;
        if signed > reach && !exists {
            // Creation gate: new pairs form near touch.
            return None;
        }
        // Separated live pairs keep an EMPTY shell (official: manifold
        // persists while spheres overlap, zero contacts when apart).
        // Keeping stale points would push bodies apart with dead
        // lambda (slow levitation); dropping the pair would burn
        // warm duals and reload every re-touch (limit cycle).
        let separated = signed > reach;
        // Scalar coefficients only; the t1 direction is recomputed per
        // solve from live orientations.
        let mu = {
            let (_, mu1, mu2) = friction_frame(a, b, normal);
            [mu1.max(0.0), mu2.max(0.0)]
        };
        // Witness point plus box-corner expansion for face stability.
        // Every point is a coincident pair on the witness plane: both
        // anchors are material points that start at the same world
        // position (official rA/rB semantics). Projecting a foreign
        // corner into the other body's frame instead would glue a
        // non-material point and pump energy (spike lesson).
        //
        // The gate is anchored at a CENTER witness (body centers
        // projected onto the contact plane, midpoint shared), NOT at
        // the raw closest-point pair: shape_distance returns CORNER
        // witnesses for box-box (face-face has infinite closest
        // pairs), and gating around a flickering corner admits one
        // corner per step — churning anchors, no support accumulation
        // (boxes fell 82m through the floor). Centers move smoothly,
        // so the pass-set is stable.
        //
        // The gate is split by stability character: the NORMAL gate
        // uses the per-side raw witness (only its PLANE matters, and
        // the face plane is flicker-immune even when the witness
        // slides within it); the TANGENTIAL gate uses the center
        // witness (position-stable). Gating normal distance around
        // the mid-gap center instead would reject gapped faces.
        //
        // The tangential gate is two-sided (a local patch around the
        // center witness, scaled by the smaller body): the old
        // normal-only gate admitted a huge floor's coplanar corners
        // 3.5m away as phantom points — 2.5m levers whose meter-scale
        // Ct torqued every spin to death.
        let patch = ra.min(rb) + MARGIN;
        let pp = (d.point_a + d.point_b) * 0.5;
        // Per-body patch centers: each body's own center projected onto
        // the contact plane. Centers move smoothly (the anti-flicker
        // property the shared midpoint was built for), and each stays
        // under its own body — the shared midpoint of two centers drifts
        // off the contact patch under lateral offset (small box far from
        // a huge floor's center: measured, its true face corners gated
        // out 1.5 m away, single-point catch, 30 m/s retouch tunneled
        // while the centered twin held) while still rejecting the big
        // body's far corners around its own center.
        let pa_c = a.position - normal * (a.position - pp).dot(normal);
        let pb_c = b.position - normal * (b.position - pp).dot(normal);
        // Plane support values: unsigned box witnesses may coincide inside
        // penetration, so reconstruct their actual opposing surface planes.
        let plane = |body: &RigidBody, witness: Vec3, sign: f32| {
            if let Shape::Box { half_extents } = body.shape {
                let extent = (body.orientation.conjugate() * normal)
                    .abs()
                    .dot(half_extents);
                normal.dot(body.position) - sign * extent
            } else {
                normal.dot(witness)
            }
        };
        let plane_a = plane(a, d.point_a, 1.0);
        let plane_b = plane(b, d.point_b, -1.0);
        let mut fresh: Vec<(Vec3, Vec3)> = Vec::new();
        if !separated {
            let inv_a = a.orientation.inverse();
            let inv_b = b.orientation.inverse();
            // Box corners first: stable material support for faces.
            for (h, witness, sign, pc) in [
                (ia, d.point_a, 1.0f32, pa_c),
                (ib, d.point_b, -1.0f32, pb_c),
            ] {
                let body = &self.bodies[h];
                for corner in box_corners(&body.shape) {
                    let world = body.position + body.orientation * corner;
                    let along = (world - witness).dot(sign * normal);
                    if along.abs() > EXPAND_SLOP + (-signed).max(0.0) {
                        continue;
                    }
                    let rel_c = world - pc;
                    let tang = rel_c - normal * rel_c.dot(normal);
                    if tang.length() > patch {
                        continue;
                    }
                    if fresh.len() >= MAX_POINTS {
                        break;
                    }
                    // Keep the incident corner and project its counterpart
                    // onto the reference surface, preserving per-point depth.
                    let (pa, pb) = if h == ia {
                        (world, world + normal * (plane_b - normal.dot(world)))
                    } else {
                        (world + normal * (plane_a - normal.dot(world)), world)
                    };
                    let ra_l = inv_a * (pa - a.position);
                    let rb_l = inv_b * (pb - b.position);
                    if fresh.iter().any(|(ea, eb)| {
                        (ea - ra_l).length() < POINT_MATCH_DIST
                            && (eb - rb_l).length() < POINT_MATCH_DIST
                    }) {
                        continue;
                    }
                    fresh.push((ra_l, rb_l));
                }
            }
        } // end corner expansion.
        // Center point for face-like contacts (2+ corners): the shared
        // projected-center midpoint, as before — it moves smoothly with
        // the bodies even when the admitted corner set flickers, so it
        // never churns the solver (a centroid of the admitted set was
        // tried: it inherits admission flicker and rocks stacks).
        // Under lateral offset it can sit off the admitted patch; the
        // admitted corners still do the catching, this only centers.
        if !separated && fresh.len() >= 2 && fresh.len() < MAX_POINTS {
            let inv_a = a.orientation.inverse();
            let inv_b = b.orientation.inverse();
            // The smaller body's center stays on its footprint even far
            // from a large floor's center; an admitted-corner centroid does
            // not (it flickers when the corner set changes).
            let mid = if ra < rb {
                pa_c
            } else if rb < ra {
                pb_c
            } else {
                (pa_c + pb_c) * 0.5
            };
            let pa = mid + normal * (plane_a - normal.dot(mid));
            let pb = mid + normal * (plane_b - normal.dot(mid));
            let ra_l = inv_a * (pa - a.position);
            let rb_l = inv_b * (pb - b.position);
            if !fresh.iter().any(|(ea, eb)| {
                (ea - ra_l).length() < POINT_MATCH_DIST && (eb - rb_l).length() < POINT_MATCH_DIST
            }) {
                fresh.push((ra_l, rb_l));
            }
        }
        // Witness fallback: the closest-point pair slides across faces
        // (non-material churn that rocks stacks), so it is only used
        // when corners give fewer than 3 points (edge/vertex and
        // non-box contacts).
        if !separated && fresh.len() < 3 {
            let inv_a = a.orientation.inverse();
            let inv_b = b.orientation.inverse();
            let ra_l = inv_a * (d.point_a - a.position);
            let rb_l = inv_b * (d.point_b - b.position);
            if !fresh.iter().any(|(ea, eb)| {
                (ea - ra_l).length() < POINT_MATCH_DIST && (eb - rb_l).length() < POINT_MATCH_DIST
            }) {
                fresh.push((ra_l, rb_l));
            }
        }
        // Wake vote (evaluated by the merge): new pairs wake on
        // genuine impact or deep fresh overlap; live pairs wake on
        // the same 1cm penetration backstop (island-wake parity
        // without islands — a slow burrow never trips the impact
        // gate). All inputs are step-top state, hence schedule-free.
        let wake_vote = if exists {
            (self.asleep[ia] || self.asleep[ib]) && signed < -WAKE_PENETRATION
        } else {
            // Gated wake (builtin `wake_on_impact` parity, threshold
            // 0.5 m/s approach): a NEW pair wakes its sleepers only on
            // a genuine impact or a deep fresh overlap (spawn /
            // teleport driver, blind to the velocity test). Resting
            // micro-jitter and GEN_MARGIN re-gating stay asleep.
            let fresh_pen = signed < -WAKE_PENETRATION;
            let mut impact = fresh_pen;
            if !impact {
                for (s, o, sign) in [(ia, ib, 1.0f32), (ib, ia, -1.0f32)] {
                    if self.asleep[s] {
                        let app =
                            (self.bodies[o].velocity - self.bodies[s].velocity).dot(sign * normal);
                        if app > WAKE_IMPACT_SPEED {
                            impact = true;
                            break;
                        }
                    }
                }
            }
            impact
        };
        Some(Discovered {
            ia,
            ib,
            trigger: false,
            exists,
            normal,
            mu,
            signed,
            fresh,
            wake_vote,
        })
    }

    /// Generate/persist contact pairs for this step. Returns the set of
    /// overlapping trigger pairs (canonical order).
    ///
    /// Two phases: discovery (pure per-pair bundles, sequential below
    /// [`Self::PAR_DISCOVERY_BODIES`] bodies, rayon above) and a
    /// sequential merge in canonical `(ia, ib)` order — the ONLY writer
    /// of pair state. The sweep itself stays single-threaded
    /// (Gauss-Seidel order is load-bearing), so thread count cannot leak
    /// into the simulation anywhere: 1-vs-N threads is bit-identical.
    pub(super) fn generate_pairs(&mut self) -> BTreeSet<(usize, usize)> {
        let n = self.bodies.len();
        let mut discovered: Vec<Discovered> = if n > Self::PAR_DISCOVERY_BODIES {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .flat_map(|ia| {
                    let mut out = Vec::new();
                    for ib in (ia + 1)..n {
                        if let Some(d) = self.discover_pair(ia, ib) {
                            out.push(d);
                        }
                    }
                    out
                })
                .collect()
        } else {
            let mut out = Vec::new();
            for ia in 0..n {
                for ib in (ia + 1)..n {
                    if let Some(d) = self.discover_pair(ia, ib) {
                        out.push(d);
                    }
                }
            }
            out
        };
        discovered.sort_by_key(|d| (d.ia, d.ib));
        let mut seen = vec![false; self.pairs.len()];
        let mut trigger_now = BTreeSet::new();
        // Bodies of brand-new pairs wake in the merge below: a fresh touch
        // is the one disturbance a sleeper must react to — but only a
        // genuine impact (see the discovery vote).
        let mut wake: Vec<(usize, usize)> = Vec::new();
        for d in discovered {
            let (ia, ib) = (d.ia, d.ib);
            if d.trigger {
                if d.signed <= 0.0 {
                    trigger_now.insert((ia, ib));
                }
                continue;
            }
            // Persist duals by matching material anchors (5cm window).
            if d.exists {
                let idx = self
                    .pairs
                    .iter()
                    .position(|p| p.a == ia && p.b == ib)
                    .expect("discovery saw this pair a few microseconds ago on the same state");
                seen[idx] = true;
                let pair = &mut self.pairs[idx];
                pair.n = d.normal;
                pair.mu = d.mu;
                pair.gap = d.signed;
                // Wake propagation through LIVE pairs (island-wake
                // parity without islands): a slow burrow never trips
                // the 0.5 m/s impact gate, so deepening overlap past
                // the 1cm backstop wakes — same rule as the creation
                // backstop, one consistent threshold. Resting contacts
                // sit an order of magnitude shallower (mm), settled
                // piles stay asleep; chains wake ring-by-ring as the
                // push advances (measured: a 0.2 m/s pusher ghosted
                // 40cm through its target with no wake at all).
                if d.wake_vote {
                    wake.push((ia, ib));
                }
                let mut next: Vec<AvbdPoint> = Vec::with_capacity(d.fresh.len());
                for (ra_l, rb_l) in d.fresh {
                    if let Some(old) = pair
                        .points
                        .iter()
                        .find(|p| (p.ra - ra_l).length() < LAMBDA_MATCH_DIST)
                    {
                        // Official merge: matched points carry lambda/penalty;
                        // anchors stay frozen while the grip holds
                        // (`stick`), otherwise they refresh to the live
                        // coincident geometry. PLUS the separation rule:
                        // anchors refresh while the witness gap is open
                        // (`d.dist > 0`), no matter the grip flag — a
                        // frozen band-entry anchor encodes a stale C0
                        // (gap + margin at creation), and the Taylor row
                        // then treats the stale offset as a live
                        // violation: the pair levitates (measured: a
                        // 5 m/s drop stopped +7mm above the floor and
                        // rose on phantom support instead of touching
                        // down) and burrowing drivers meet only the
                        // (1-alpha)-diluted residue (measured: a 2 m/s
                        // kinematic pusher ghosted through its target).
                        // Refreshing keeps C0 honest (live gap +
                        // margin), so the row fires on the full
                        // within-step approach J*dq — the fast-impact
                        // catch survives (it never needed stale
                        // anchors), while phantom support cannot form.
                        // Freeze starts at first touch, like resting
                        // pairs always did (signed gap: the raw oracle
                        // is unsigned for box pairs and would never
                        // freeze them).
                        let touching = d.signed <= 0.0;
                        // No-slip roll still changes the material point at
                        // a sphere's geometric pole. Refresh its patch while
                        // carrying the matched dual, rather than repeatedly
                        // losing support when the frozen point leaves it.
                        let smooth = matches!(self.bodies[ia].shape, Shape::Sphere { .. })
                            || matches!(self.bodies[ib].shape, Shape::Sphere { .. });
                        let (ra, rb) = if old.stuck && touching && !smooth {
                            (old.ra, old.rb)
                        } else {
                            (ra_l, rb_l)
                        };
                        //
                        // No rolling-aware extension: fast spin makes any
                        // anchor stale WITHIN its own step (10 rad/s =
                        // 9.6 deg/step of material carry), so cross-step
                        // refresh cannot save sustained rotation — the
                        // official headless build kills a free spin
                        // 10 -> 0.000 in 600 steps too. This is a
                        // position-level material-anchor limit, not a
                        // refresh-policy bug (finer substeps shrink the
                        // per-step carry; velocity-level rolling rows
                        // ignore anchors entirely).
                        next.push(AvbdPoint {
                            ra,
                            rb,
                            lam: old.lam,
                            pen: old.pen,
                            stuck: old.stuck,
                            roll_lam: old.roll_lam,
                        });
                    } else {
                        next.push(AvbdPoint {
                            ra: ra_l,
                            rb: rb_l,
                            lam: [0.0; 3],
                            pen: [PENALTY_INIT; 3],
                            stuck: true,
                            roll_lam: [0.0; 3],
                        });
                    }
                }
                pair.points = next;
            } else {
                seen.push(true);
                // Gated wake for NEW pairs (vote computed in discovery
                // from step-top state: impact or deep fresh overlap).
                if d.wake_vote {
                    wake.push((ia, ib));
                }
                self.pairs.push(AvbdPair {
                    a: ia,
                    b: ib,
                    n: d.normal,
                    mu: d.mu,
                    gap: d.signed,
                    points: d
                        .fresh
                        .into_iter()
                        .map(|(ra_l, rb_l)| AvbdPoint {
                            ra: ra_l,
                            rb: rb_l,
                            lam: [0.0; 3],
                            pen: [PENALTY_INIT; 3],
                            stuck: true,
                            roll_lam: [0.0; 3],
                        })
                        .collect(),
                });
            }
        }
        for idx in (0..seen.len()).rev() {
            if !seen[idx] {
                self.pairs.swap_remove(idx);
            }
        }
        for (a, b) in wake {
            self.wake_body(a);
            self.wake_body(b);
        }
        trigger_now
    }

    /// Taylor row value `C = C0*(1-alpha) + u.(dA - dB)` plus the live lever
    /// arms, for constraint row direction `u` (A-side).
    pub(super) fn row_c(
        &self,
        pair: &AvbdPair,
        pt: &AvbdPoint,
        u: Vec3,
        c0: f32,
    ) -> (f32, Vec3, Vec3) {
        let a = &self.bodies[pair.a];
        let b = &self.bodies[pair.b];
        let (ra_w, rb_w) = Self::contact_levers(a, b, pair.n, pt);
        self.row_c_levers(pair, u, c0, ra_w, rb_w)
    }

    /// World-space contact levers for a stored point: material anchors,
    /// except a sphere contributes its geometric pole (normal force has
    /// no torque; friction acts at full radius).
    fn contact_levers(a: &RigidBody, b: &RigidBody, n: Vec3, pt: &AvbdPoint) -> (Vec3, Vec3) {
        let ra_w = match a.shape {
            Shape::Sphere { radius } => -n * radius,
            _ => a.orientation * pt.ra,
        };
        let rb_w = match b.shape {
            Shape::Sphere { radius } => n * radius,
            _ => b.orientation * pt.rb,
        };
        (ra_w, rb_w)
    }

    /// Coincident friction levers: the [`contact_levers`] pair with its
    /// normal-direction separation removed (both anchors shifted to their
    /// shared interface midplane). Normal rows keep the material anchors
    /// (per-point depth is the normal signal); friction rows must not see
    /// the depth offset as a torque arm — `stamp_row` builds `t = r × nn`,
    /// so a `d·n` offset turns into a phantom `(d·n)×f_t` torque that grows
    /// with penetration (measured: a buried SixDof spin-clamp escapes its
    /// ±0.2 window to 0.48 on friction levers that carry depth, holds at
    /// 0.07 on coincident ones). Tangential anchor offset is real geometry
    /// and stays. This is NOT the reverted all-rows-midpoint attempt (see
    /// `row_c_levers` NOTE): the normal row keeps material levers, so
    /// sphere poles stay load-bearing.
    pub(super) fn friction_levers(
        a: &RigidBody,
        b: &RigidBody,
        n: Vec3,
        pt: &AvbdPoint,
    ) -> (Vec3, Vec3) {
        let (ra_w, rb_w) = Self::contact_levers(a, b, n, pt);
        let sep = ((a.position + ra_w) - (b.position + rb_w)).dot(n);
        (ra_w - n * (sep * 0.5), rb_w + n * (sep * 0.5))
    }

    /// Residual with explicit world levers (drags the matching stamp
    /// levers along, so C and J can never disagree about the point).
    ///
    /// NOTE (spin-instability bisection): evaluating ALL rows at the
    /// shared interface midpoint was tried here and reverted — it broke
    /// `rolling_sphere_keeps_support_without_gaining_speed` (sphere poles
    /// are load-bearing as levers, not just directions). The helpers stay
    /// for the next attempt: pass explicit levers, get consistent C+J back.
    pub(super) fn row_c_levers(
        &self,
        pair: &AvbdPair,
        u: Vec3,
        c0: f32,
        ra_w: Vec3,
        rb_w: Vec3,
    ) -> (f32, Vec3, Vec3) {
        let a = &self.bodies[pair.a];
        let b = &self.bodies[pair.b];
        let d_a = a.position - self.pos0[pair.a]
            + u * (ra_w
                .cross(u)
                .dot(quat_diff_vec(a.orientation, self.rot0[pair.a])));
        let d_b = b.position - self.pos0[pair.b]
            + u * (rb_w
                .cross(u)
                .dot(quat_diff_vec(b.orientation, self.rot0[pair.b])));
        (c0 * (1.0 - ALPHA) + u.dot(d_a - d_b), ra_w, rb_w)
    }

    /// Clamped contact force triple: push-only normal, elliptical Coulomb
    /// cone on the tangents (`mu` per frame axis). With `mu1 == mu2` this
    /// is bit-identical to the legacy circular cone. A zero-`mu` axis is
    /// a degenerate ellipse (segment/point): that axis locks to zero and
    /// the live axis clamps to its own bound instead of dying with it.
    pub(super) fn contact_force(
        cn: f32,
        pens: [f32; 3],
        lam: [f32; 3],
        ct: [f32; 2],
        mu: [f32; 2],
    ) -> [f32; 3] {
        let fn_c = (pens[0] * cn + lam[0]).min(0.0);
        let ft_raw = [pens[1] * ct[0] + lam[1], pens[2] * ct[1] + lam[2]];
        let b1 = fn_c.abs() * mu[0];
        let b2 = fn_c.abs() * mu[1];
        if b1 <= 0.0 || b2 <= 0.0 {
            let t1 = if b1 <= 0.0 {
                0.0
            } else {
                ft_raw[0].clamp(-b1, b1)
            };
            let t2 = if b2 <= 0.0 {
                0.0
            } else {
                ft_raw[1].clamp(-b2, b2)
            };
            return [fn_c, t1, t2];
        }
        // Elliptical projection: uniform downscale when outside the ellipse.
        let s = (ft_raw[0] / b1) * (ft_raw[0] / b1) + (ft_raw[1] / b2) * (ft_raw[1] / b2);
        let k = if s > 1.0 { 1.0 / s.sqrt() } else { 1.0 };
        [fn_c, ft_raw[0] * k, ft_raw[1] * k]
    }

    /// Stamp one constraint row for a single body side.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stamp_row(
        lhs: &mut [[f32; SPATIAL_DOF]; SPATIAL_DOF],
        rhs: &mut [f32; SPATIAL_DOF],
        axis: Vec3,
        pen: f32,
        f: f32,
        r: Vec3,
        sign: f32,
    ) {
        let nn = sign * axis;
        let t = r.cross(nn);
        let o_nn = outer(nn, nn);
        let o_tt = outer(t, t);
        let o_nt = outer(nn, t);
        for x in 0..3 {
            for y in 0..3 {
                lhs[x][y] += pen * o_nn[x][y];
                lhs[ANGULAR_OFFSET + x][ANGULAR_OFFSET + y] += pen * o_tt[x][y];
                lhs[x][ANGULAR_OFFSET + y] += pen * o_nt[x][y];
                lhs[ANGULAR_OFFSET + x][y] += pen * o_nt[y][x];
            }
        }
        rhs[0] += f * nn.x;
        rhs[1] += f * nn.y;
        rhs[2] += f * nn.z;
        rhs[ANGULAR_OFFSET] += f * t.x;
        rhs[4] += f * t.y;
        rhs[5] += f * t.z;
    }

    /// Stamp a pure angular row in world coordinates.
    pub(super) fn stamp_angular_row(
        lhs: &mut [[f32; SPATIAL_DOF]; SPATIAL_DOF],
        rhs: &mut [f32; SPATIAL_DOF],
        axis: Vec3,
        pen: f32,
        force: f32,
    ) {
        let h = outer(axis, axis);
        for x in 0..3 {
            for y in 0..3 {
                lhs[ANGULAR_OFFSET + x][ANGULAR_OFFSET + y] += pen * h[x][y];
            }
        }
        rhs[ANGULAR_OFFSET] += force * axis.x;
        rhs[4] += force * axis.y;
        rhs[5] += force * axis.z;
    }
}
