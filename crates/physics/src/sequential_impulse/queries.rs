//! Raycast and shapecast queries: conservative advancement, swept-shape
//! overlap tests and exact per-shape ray hits.

use glam::{Quat, Vec3};

use crate::body::RigidBody;
use crate::constants::{CCD_TRAVEL_GATE_FRACTION, NEAR_ZERO, SHAPE_TOUCH};
use crate::distance;
use crate::flags::HitKind;
use crate::math::{Ray, RaycastHit};
use crate::shape::Shape;

use super::SequentialImpulseEngine;
use super::math::vec3_finite;
use super::*;

/// Numerical zero for degenerate guards: denominators, lengths and polynomial
/// coefficients at or below this magnitude are treated as exactly zero
/// (singular solve, zero-length direction, repeated root).
const DEGENERATE_EPS: f32 = crate::constants::DEGENERATE_LEN2;

/// Minimum meaningful segment length: shorter extents are treated as
/// collapsed (no sweep direction, no bound worth keeping).
const MIN_SEGMENT_LENGTH: f32 = NEAR_ZERO;

/// Edge-cross length floor for OBB SAT candidates (m): shorter means
/// nearly parallel edges, so the axis is dropped.
const PARALLEL_EDGE_EPS: f32 = 1e-3;
/// Overlap / SAT margin for swept-shape discrete probes (m).
const OVERLAP_EPS: f32 = 1e-5;
/// Angular-CCD touch band (m): tighter than [`SHAPE_TOUCH`] so binary
/// refine does not stop a hair short of the contact.
const ANGULAR_CCD_TOUCH: f32 = 1e-5;
/// Binary-search refine iterations inside the angular TOI bracket.
const BINARY_REFINE_ITERS: usize = 10;
/// Minimum fractional advance of the angular CA loop.
const CA_FRACTION_EPS: f32 = 1e-4;
/// Explicit BVH walk stack for mesh raycasts (same depth as distance).
const BVH_STACK_CAP: usize = 64;
/// Midpoint / half-span scale for CCD and heightfield grid math.
const HALF: f32 = 0.5;

/// Shared exact ray/shape query for engine implementations: hit distance
/// plus the surface normal in shape-local coordinates, or `None`.
/// [`SequentialImpulseEngine`] and [`crate::avbd::AvbdEngine`] both route
/// through this routine so raycasts agree by construction.
pub(crate) fn raycast_shape_hit(
    shape: &Shape,
    origin: Vec3,
    direction: Vec3,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    match shape {
        Shape::Sphere { radius } => {
            ray_sphere_hit(origin, direction, Vec3::ZERO, *radius, max_dist)
        }
        Shape::Box { half_extents } => ray_obb_hit(origin, direction, *half_extents, max_dist),
        Shape::Capsule {
            radius,
            half_height,
        } => ray_capsule_hit(origin, direction, *radius, *half_height, max_dist),
        Shape::Cylinder {
            radius,
            half_height,
        } => ray_cylinder_hit(origin, direction, *radius, *half_height, max_dist),
        Shape::Cone {
            radius,
            half_height,
        } => ray_cone_hit(origin, direction, *radius, *half_height, max_dist),
        Shape::ConvexHull(hull) => ray_hull_hit(origin, direction, hull, max_dist),
        Shape::Heightfield(hf) => ray_heightfield_hit(origin, direction, hf, max_dist),
        Shape::TriMesh(mesh) => ray_trimesh_hit(origin, direction, mesh, max_dist),
    }
}

/// Candidate returned by the linear or angular continuous collision query.
pub struct ContinuousHit {
    /// Travel fraction at impact (0..1).
    pub fraction: f32,
    /// Impact normal.
    pub normal: Vec3,
    /// Hit body handle.
    pub handle: crate::body::BodyHandle,
    /// Origin of the hit (linear vs angular sweep).
    pub kind: HitKind,
    /// World contact point on the mover at the hit fraction (angular hits
    /// only): the response levers the spin off it instead of killing it.
    pub contact: Option<Vec3>,
}

impl ContinuousHit {
    /// `true` when the hit came from the angular sweep (compat for the
    /// legacy `angular: bool` field).
    pub fn angular(&self) -> bool {
        self.kind.is_angular()
    }
}

pub(crate) fn shape_min_dimension(shape: &Shape) -> f32 {
    match shape {
        Shape::Sphere { radius } => *radius,
        Shape::Box { half_extents } => half_extents.min_element(),
        Shape::Capsule { radius, .. } => *radius,
        Shape::Cylinder {
            radius,
            half_height,
        } => radius.min(*half_height),
        // The apex is a point: arm CCD early (half the box rule).
        Shape::Cone {
            radius,
            half_height,
        } => HALF * radius.min(*half_height),
        Shape::ConvexHull(hull) => HALF * hull.min_extent(),
        Shape::Heightfield(hf) => HALF * hf.cell(),
        Shape::TriMesh(mesh) => HALF * mesh.min_feature(),
    }
}

fn shape_max_radius(shape: &Shape) -> f32 {
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
        } => (radius * radius + half_height * half_height).sqrt(),
        Shape::Cone {
            radius,
            half_height,
        } => (radius * radius + half_height * half_height).sqrt(),
        Shape::ConvexHull(hull) => hull
            .vertices()
            .iter()
            .map(|v| v.length())
            .fold(0.0f32, f32::max),
        Shape::Heightfield(hf) => hf.local_extents().length() + hf.local_center().length(),
        Shape::TriMesh(mesh) => mesh.bound_radius(),
    }
}

fn shape_rotation_sensitive(shape: &Shape) -> bool {
    !matches!(shape, Shape::Sphere { .. })
}

/// Orientation at a fraction of the current substep's angular motion.
pub(crate) fn swept_orientation(body: &RigidBody, sub_dt: f32, fraction: f32) -> Quat {
    (Quat::from_scaled_axis(body.angular_velocity * (sub_dt * fraction)) * body.orientation)
        .normalize()
}

/// Exact shape distance at a pose on the combined linear/angular sweep.
fn swept_distance(
    body: &RigidBody,
    target: distance::ShapeRef<'_>,
    displacement: Vec3,
    sub_dt: f32,
    fraction: f32,
) -> distance::Distance {
    distance::shape_distance(
        distance::ShapeRef {
            shape: &body.shape,
            pos: body.position + displacement * fraction,
            rot: swept_orientation(body, sub_dt, fraction),
        },
        target,
    )
}

/// Conservative overlap predicate for a swept pose. OBB pairs use SAT because
/// the generic OBB distance oracle is unsigned while overlapping boxes need a
/// signed contact decision; the other pairs use their analytic signed distance.
fn swept_shape_overlaps(
    body: &RigidBody,
    target: distance::ShapeRef<'_>,
    displacement: Vec3,
    sub_dt: f32,
    fraction: f32,
) -> bool {
    let position = body.position + displacement * fraction;
    let orientation = swept_orientation(body, sub_dt, fraction);
    match (&body.shape, target.shape) {
        (
            Shape::Box {
                half_extents: half_a,
            },
            Shape::Box {
                half_extents: half_b,
            },
        ) => obb_sat(
            position,
            *half_a,
            orientation,
            target.pos,
            *half_b,
            target.rot,
            OVERLAP_EPS,
        )
        .is_some(),
        _ => {
            let distance = distance::shape_distance(
                distance::ShapeRef {
                    shape: &body.shape,
                    pos: position,
                    rot: orientation,
                },
                target,
            );
            distance.dist <= OVERLAP_EPS
        }
    }
}

/// Signed separation between a mover at `mover_pos` (frozen orientation
/// `mover_rot`) and a target. Box-box uses the SAT separation — the
/// vertex/edge distance oracle is unsigned and cannot see overlap, so the
/// shared `cast_shape` is blind to box-vs-box crossings and the kinematic
/// sweep advances on this instead. Every other pair uses the exact signed
/// distance (same oracle as the linear cast). Positive = separated.
pub fn sweep_gap(
    mover_shape: &Shape,
    mover_pos: Vec3,
    mover_rot: Quat,
    target: distance::ShapeRef<'_>,
) -> f32 {
    if let (Shape::Box { half_extents: ha }, Shape::Box { half_extents: hb }) =
        (mover_shape, target.shape)
    {
        let aa = [
            mover_rot * Vec3::X,
            mover_rot * Vec3::Y,
            mover_rot * Vec3::Z,
        ];
        let ba = [
            target.rot * Vec3::X,
            target.rot * Vec3::Y,
            target.rot * Vec3::Z,
        ];
        // Separation = max over axes of the negated overlap (Ericson,
        // RTCD §5.1.9): positive while apart, negative while penetrating.
        let mut sep = f32::MIN;
        for u in aa.into_iter().chain(ba) {
            let overlap = obb_overlap_on(mover_pos, *ha, mover_rot, target.pos, *hb, target.rot, u);
            sep = sep.max(-overlap);
        }
        for ai in &aa {
            for bi in &ba {
                let c = ai.cross(*bi);
                if c.length() < PARALLEL_EDGE_EPS {
                    continue;
                }
                let overlap = obb_overlap_on(
                    mover_pos,
                    *ha,
                    mover_rot,
                    target.pos,
                    *hb,
                    target.rot,
                    c.normalize(),
                );
                sep = sep.max(-overlap);
            }
        }
        return sep;
    }
    distance::shape_distance(
        distance::ShapeRef {
            shape: mover_shape,
            pos: mover_pos,
            rot: mover_rot,
        },
        target,
    )
    .dist
}

/// Conservative advancement of one kinematic step segment against one
/// target. Tunnel-proof: every advance is bounded by the exact signed gap
/// ([`sweep_gap`]), so no feature can be crossed mid-step; rotation is
/// frozen, like the linear cast. Returns the absolute travel distance and
/// the sweep normal (pointing back toward the mover, `cast_shape`
/// convention). Touching at t=0 reports no hit — resting contact is the
/// discrete solver's job.
pub fn kinematic_cast(
    mover_shape: &Shape,
    mover_rot: Quat,
    from: Vec3,
    displacement: Vec3,
    target: distance::ShapeRef<'_>,
) -> Option<(f32, Vec3)> {
    let len = displacement.length();
    if len < MIN_SEGMENT_LENGTH {
        return None;
    }
    let dir = displacement / len;
    const MAX_ITERS: usize = 32;
    let mut t = 0.0f32;
    for _ in 0..MAX_ITERS {
        let pos = from + dir * t;
        let gap = sweep_gap(mover_shape, pos, mover_rot, target);
        if gap <= SHAPE_TOUCH {
            if t > 0.0 {
                // Witnesses at the touching pose: near-zero gap, so even the
                // unsigned OBB oracle reads a valid contact frame here.
                let d = distance::shape_distance(
                    distance::ShapeRef {
                        shape: mover_shape,
                        pos,
                        rot: mover_rot,
                    },
                    target,
                );
                let n = (d.point_a - d.point_b).normalize_or(-dir);
                return Some((t, n));
            }
            return None;
        }
        t += gap - SHAPE_TOUCH * HALF;
        if t >= len {
            break;
        }
    }
    None
}

/// Energy-neutral cap for a spin correction `delta`: walk back along it to
/// the closed-form neutral point `t = (d·Iω)/E(d)` when the full correction
/// would add rotational energy (stiff anisotropic levers). Pure arithmetic,
/// hence deterministic; degenerate inputs return `omega` unchanged.
fn cap_spin_correction(omega: Vec3, inertia: Vec3, orientation: Quat, delta: Vec3) -> Vec3 {
    let out = omega - delta;
    // Body-frame energies: E(Ω) = ½ΣIᵢwᵢ². Exact correction first; if it
    // would inject energy, walk back along `delta` to the neutral point.
    let qb = orientation.conjugate();
    let wb = qb * omega;
    let db = qb * delta;
    let iw = inertia * wb;
    let e_omega = HALF * iw.dot(wb);
    let e_out = HALF * (inertia * (wb - db)).dot(wb - db);
    if e_out <= e_omega {
        return out;
    }
    let e_d = HALF * (inertia * db).dot(db);
    if !e_d.is_finite() || e_d <= 0.0 {
        return omega;
    }
    let t = db.dot(iw) / e_d;
    if !t.is_finite() || t <= 0.0 {
        return omega;
    }
    omega - delta * t.min(1.0)
}

/// Frictionless spin response for a CCD stop: remove exactly the spin that
/// drives the contact point into the surface, keep the tangential spin.
/// The correction follows a frictionless contact impulse — `Δω = J·I⁻¹m`
/// with the lever `m = r_c×n̂` and `J = −vn/((I⁻¹m)·m)`, `vn = ((ω×r_c)·n̂)`
/// the approach speed (`n̂` points from the target back toward the mover, so
/// approach is `vn < 0`). Consequences: for isotropic inertia this is the
/// minimum-norm projection; for planar motion with an in-plane contact
/// normal it reduces exactly to the old full stop — the fix only changes
/// contacts with a genuine out-of-plane lever (scrapes keep tangential
/// spin instead of dying). No angular restitution here (inelastic) — the
/// restitution-aware path is [`ccd_impact_velocity`]; friction stays with
/// the contact/joint passes.
///
/// The energy-neutral [`cap_spin_correction`] applies (see its docs).
/// Pure arithmetic, hence deterministic. Degenerate lever or a separating
/// contact returns `omega` unchanged (never NaN).
pub fn remove_angular_approach(
    omega: Vec3,
    inertia: Vec3,
    orientation: Quat,
    lever: Vec3,
    normal: Vec3,
) -> Vec3 {
    let m = lever.cross(normal);
    let im = mul_inv_inertia(inertia, orientation, m);
    let denom = im.dot(m);
    if !denom.is_finite() || denom <= DEGENERATE_EPS {
        return omega;
    }
    let vn = omega.cross(lever).dot(normal);
    if !vn.is_finite() || vn >= 0.0 {
        return omega;
    }
    cap_spin_correction(omega, inertia, orientation, im * (vn / denom))
}

/// Unified one-shot impact for an angular CCD hit: the contact-point velocity
/// `v_c = v + ω×r_c` (spin counts toward the approach, not just the center
/// motion), a textbook rigid-body impulse `J = −(1+e)·vn_c/denom` with
/// `denom = inv_mass + (I⁻¹m)·m`, restitution `e` above the shared 1 m/s
/// contact-speed threshold (otherwise inelastic). Tangential surface motion
/// is preserved — friction is the discrete solver's job next substep (it
/// sees a clean touching contact thanks to the clamp+backoff), so CCD never
/// double-applies it. The linear part applies fully (center-mass, bounded);
/// the spin part goes through [`cap_spin_correction`]. Pure arithmetic,
/// hence deterministic.
#[allow(clippy::too_many_arguments)]
pub fn ccd_impact_velocity(
    velocity: Vec3,
    omega: Vec3,
    inv_mass: f32,
    inertia: Vec3,
    orientation: Quat,
    lever: Vec3,
    normal: Vec3,
    restitution: f32,
) -> (Vec3, Vec3) {
    let contact_vel = velocity + omega.cross(lever);
    let vn_c = contact_vel.dot(normal);
    if !vn_c.is_finite() || vn_c >= 0.0 || !inv_mass.is_finite() || inv_mass <= 0.0 {
        return (velocity, omega);
    }
    let e = if vn_c < -1.0 { restitution } else { 0.0 };
    let m = lever.cross(normal);
    let im = mul_inv_inertia(inertia, orientation, m);
    let denom = inv_mass + im.dot(m);
    if !denom.is_finite() || denom <= DEGENERATE_EPS {
        return (velocity, omega);
    }
    let impulse = -(1.0 + e) * vn_c / denom;
    let new_velocity = velocity + normal * (impulse * inv_mass);
    let new_omega = cap_spin_correction(omega, inertia, orientation, im * -impulse);
    (new_velocity, new_omega)
}

fn find_linear_continuous_hit(
    bodies: &[RigidBody],
    mover_index: usize,
    displacement: Vec3,
) -> Option<ContinuousHit> {
    let body = &bodies[mover_index];
    if body.is_trigger {
        return None;
    }
    let length = displacement.length();
    let min_dimension = shape_min_dimension(&body.shape);
    if length <= CCD_TRAVEL_GATE_FRACTION * min_dimension {
        return None;
    }
    let mover_layer = body.collision_layer;
    let mover_mask = body.collision_mask;
    let mover = distance::ShapeRef {
        shape: &body.shape,
        pos: body.position,
        rot: body.orientation,
    };
    let targets = bodies
        .iter()
        .enumerate()
        .filter(move |&(handle, target)| {
            handle != mover_index
                && !target.is_trigger
                && mover_mask & target.collision_layer != 0
                && target.collision_mask & mover_layer != 0
        })
        .map(|(handle, target)| {
            (
                crate::body::BodyHandle::from(handle),
                distance::ShapeRef {
                    shape: &target.shape,
                    pos: target.position,
                    rot: target.orientation,
                },
            )
        });
    distance::cast_shape(mover, displacement, targets).map(|hit| ContinuousHit {
        fraction: (hit.t / length).clamp(0.0, 1.0),
        normal: hit.normal,
        handle: hit.handle,
        kind: HitKind::Linear,
        contact: None,
    })
}

/// Conservative-advancement first overlap for the combined linear+angular
/// sweep. Uses the exact distance at each pose and the uniform bound
/// `|displacement| + max_radius*angle` per unit fraction.
///
/// Why the uniform bound is enough (and deliberately kept): the mover is
/// rigid and the target frozen, so the gap is Lipschitz in the fraction with
/// exactly this constant — the distance cannot change faster than the
/// fastest surface point. A step of `gap/μ` therefore varies the gap by at
/// most `gap`: a pass-through must land interpenetrating, never straddling,
/// so the per-iterate overlap check plus the binary refine catch every
/// crossing for any feature thickness. Any "tighter" witness-based bound
/// would break this (witness switches mid-step), trading a proof for fewer
/// iterations — not worth it; separated pairs already exit in 1–2 oracle
/// calls, only grazing approaches walk the full 32.
fn first_angular_overlap_fraction(
    body: &RigidBody,
    target: distance::ShapeRef<'_>,
    displacement: Vec3,
    sub_dt: f32,
) -> Option<f32> {
    if swept_shape_overlaps(body, target, displacement, sub_dt, 0.0) {
        return None;
    }
    let angle = (body.angular_velocity * sub_dt).length();
    let bound = displacement.length() + shape_max_radius(&body.shape) * angle;
    if bound < MIN_SEGMENT_LENGTH {
        return None;
    }
    const MAX_ITERS: usize = 32;
    let mut f = 0.0f32;
    let mut prev_f = 0.0f32;
    for _ in 0..MAX_ITERS {
        if f >= 1.0 {
            break;
        }
        if swept_shape_overlaps(body, target, displacement, sub_dt, f) {
            if f <= 0.0 {
                return None;
            }
            // Binary refine the bracket [prev_f, f] for sub-sample precision.
            let mut low = prev_f;
            let mut high = f;
            for _ in 0..BINARY_REFINE_ITERS {
                let mid = (low + high) * HALF;
                if swept_shape_overlaps(body, target, displacement, sub_dt, mid) {
                    high = mid;
                } else {
                    low = mid;
                }
            }
            return Some(high);
        }
        let d = swept_distance(body, target, displacement, sub_dt, f);
        // `d.dist` is the exact surface gap (positive = separated). Advance
        // by at most the gap over the worst-case point speed.
        let gap = d.dist - ANGULAR_CCD_TOUCH * HALF;
        if gap <= 0.0 {
            // Numerically touching — treat as overlap at next fraction.
            let next = (f + CA_FRACTION_EPS).min(1.0);
            if swept_shape_overlaps(body, target, displacement, sub_dt, next) {
                return Some(next);
            }
            break;
        }
        let step = (gap / bound).clamp(CA_FRACTION_EPS, 1.0 - f);
        prev_f = f;
        f += step;
        if f <= prev_f {
            break;
        }
    }
    None
}

/// Fully analytic angular CCD: conservative advancement along the screw
/// motion `pos(t)=pos0+disp*t, rot(t)=slerp(angVel*t)` using the exact
/// distance oracle. No 5° sampling — the bound guarantees zero tunneling
/// for any thin feature, any angle.
pub fn find_angular_continuous_hit(
    bodies: &[RigidBody],
    mover_index: usize,
    displacement: Vec3,
    sub_dt: f32,
) -> Option<ContinuousHit> {
    let body = &bodies[mover_index];
    if body.is_trigger || !shape_rotation_sensitive(&body.shape) {
        return None;
    }
    // Travel gate, mirror of the linear one (`0.5 * min_dimension` on
    // displacement): rotation alone cannot defeat the discrete phase unless
    // its fastest surface point moves more than half the thinnest feature
    // within the substep. Thin bodies arm CCD at small angles (they tunnel
    // easily); chunky bodies only at large ones — cheaper than a flat angle
    // for cubes, stricter than one for blades.
    let angle = (body.angular_velocity * sub_dt).length();
    if shape_max_radius(&body.shape) * angle
        <= CCD_TRAVEL_GATE_FRACTION * shape_min_dimension(&body.shape)
    {
        return None;
    }
    let bound = displacement.length() + shape_max_radius(&body.shape) * angle;
    if bound < MIN_SEGMENT_LENGTH {
        return None;
    }
    let mover_layer = body.collision_layer;
    let mover_mask = body.collision_mask;
    let mut best = None;

    for (handle, target) in bodies.iter().enumerate() {
        if handle == mover_index
            || target.is_trigger
            || mover_mask & target.collision_layer == 0
            || target.collision_mask & mover_layer == 0
        {
            continue;
        }
        let target_ref = distance::ShapeRef {
            shape: &target.shape,
            pos: target.position,
            rot: target.orientation,
        };
        let Some(fraction) = first_angular_overlap_fraction(body, target_ref, displacement, sub_dt)
        else {
            continue;
        };
        let distance = swept_distance(body, target_ref, displacement, sub_dt, fraction);
        let position = body.position + displacement * fraction;
        let fallback = (position - target.position).normalize_or(Vec3::Y);
        let normal = (distance.point_a - distance.point_b).normalize_or(fallback);
        let candidate = ContinuousHit {
            fraction,
            normal,
            handle: crate::body::BodyHandle::from(handle),
            kind: HitKind::Angular,
            contact: Some(distance.point_a),
        };
        best = choose_continuous_hit(best, Some(candidate));
    }
    best
}

fn choose_continuous_hit(
    best: Option<ContinuousHit>,
    candidate: Option<ContinuousHit>,
) -> Option<ContinuousHit> {
    match (best, candidate) {
        (None, candidate) => candidate,
        (best, None) => best,
        (Some(best), Some(candidate)) => Some(if candidate.fraction < best.fraction {
            candidate
        } else {
            best
        }),
    }
}

/// Find the earliest linear or angular time of impact for one dynamic body.
pub(crate) fn find_continuous_hit(
    bodies: &[RigidBody],
    mover_index: usize,
    displacement: Vec3,
    sub_dt: f32,
) -> Option<ContinuousHit> {
    let linear = find_linear_continuous_hit(bodies, mover_index, displacement);
    let angular = find_angular_continuous_hit(bodies, mover_index, displacement, sub_dt);
    choose_continuous_hit(linear, angular)
}

/// Ray/sphere intersection in the shape's local frame. The returned normal is
/// also local so callers can rotate it back into world space.
fn ray_sphere_hit(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    radius: f32,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let a = direction.length_squared();
    if a <= DEGENERATE_EPS {
        return None;
    }
    let offset = origin - center;
    let half_b = offset.dot(direction);
    let c = offset.length_squared() - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let mut distance = (-half_b - root) / a;
    if distance < 0.0 {
        distance = (-half_b + root) / a;
    }
    if distance < 0.0 || distance > max_dist {
        return None;
    }
    let point = origin + direction * distance;
    Some((distance, (point - center).normalize_or(Vec3::X)))
}

/// Mutable interval and normal state for a local-space OBB ray query.
struct RayObbState {
    near: f32,
    far: f32,
    near_normal: Vec3,
    far_normal: Vec3,
}

/// Update one slab of a local-space OBB ray intersection.
fn ray_obb_slab(
    origin: f32,
    direction: f32,
    minimum: f32,
    maximum: f32,
    axis: Vec3,
    state: &mut RayObbState,
) -> bool {
    if direction.abs() <= DEGENERATE_EPS {
        return origin >= minimum && origin <= maximum;
    }
    let (entry, entry_normal, exit, exit_normal) = if direction > 0.0 {
        (
            (minimum - origin) / direction,
            -axis,
            (maximum - origin) / direction,
            axis,
        )
    } else {
        (
            (maximum - origin) / direction,
            axis,
            (minimum - origin) / direction,
            -axis,
        )
    };
    if entry > state.near {
        state.near = entry;
        state.near_normal = entry_normal;
    }
    if exit < state.far {
        state.far = exit;
        state.far_normal = exit_normal;
    }
    state.near <= state.far
}

/// Exact local-space ray/OBB intersection using a three-axis slab test.
fn ray_obb_hit(
    origin: Vec3,
    direction: Vec3,
    half_extents: Vec3,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    if direction.length_squared() <= DEGENERATE_EPS {
        return None;
    }
    let mut state = RayObbState {
        near: f32::NEG_INFINITY,
        far: max_dist,
        near_normal: Vec3::ZERO,
        far_normal: Vec3::ZERO,
    };
    if !ray_obb_slab(
        origin.x,
        direction.x,
        -half_extents.x,
        half_extents.x,
        Vec3::X,
        &mut state,
    ) || !ray_obb_slab(
        origin.y,
        direction.y,
        -half_extents.y,
        half_extents.y,
        Vec3::Y,
        &mut state,
    ) || !ray_obb_slab(
        origin.z,
        direction.z,
        -half_extents.z,
        half_extents.z,
        Vec3::Z,
        &mut state,
    ) {
        return None;
    }
    if state.far < 0.0 || state.near > max_dist {
        return None;
    }
    if state.near >= 0.0 {
        Some((state.near, state.near_normal))
    } else {
        Some((state.far, state.far_normal))
    }
}

/// Keep the closest candidate hit in a local-space ray query.
fn keep_closest_hit(
    best: Option<(f32, Vec3)>,
    candidate: Option<(f32, Vec3)>,
) -> Option<(f32, Vec3)> {
    match (best, candidate) {
        (None, candidate) => candidate,
        (best, None) => best,
        (Some(best), Some(candidate)) => Some(if candidate.0 < best.0 {
            candidate
        } else {
            best
        }),
    }
}

/// Exact intersection with the cylindrical side of a local Y-axis capsule.
fn ray_capsule_cylinder_hit(
    origin: Vec3,
    direction: Vec3,
    radius: f32,
    half_height: f32,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let a = direction.x * direction.x + direction.z * direction.z;
    if a <= DEGENERATE_EPS {
        return None;
    }
    let half_b = origin.x * direction.x + origin.z * direction.z;
    let c = origin.x * origin.x + origin.z * origin.z - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let denominator = a;
    let roots = [
        (-half_b - root) / denominator,
        (-half_b + root) / denominator,
    ];
    let mut best = None;
    for distance in roots {
        if distance < 0.0 || distance > max_dist {
            continue;
        }
        let point = origin + direction * distance;
        if point.y < -half_height || point.y > half_height {
            continue;
        }
        let normal = Vec3::new(point.x, 0.0, point.z).normalize_or(Vec3::X);
        best = keep_closest_hit(best, Some((distance, normal)));
    }
    best
}

/// Exact local-space ray/capsule intersection: finite cylinder side plus its
/// two spherical caps. The nearest valid feature is returned.
fn ray_capsule_hit(
    origin: Vec3,
    direction: Vec3,
    radius: f32,
    half_height: f32,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let mut best = ray_capsule_cylinder_hit(origin, direction, radius, half_height, max_dist);
    for center in [
        Vec3::new(0.0, -half_height, 0.0),
        Vec3::new(0.0, half_height, 0.0),
    ] {
        best = keep_closest_hit(
            best,
            ray_sphere_hit(origin, direction, center, radius, max_dist),
        );
    }
    best
}

/// Ray vs a flat-capped cylinder along local +Y (local frame): curved wall
/// quadratic plus two cap disks. Returns distance and the local normal.
fn ray_cylinder_hit(
    origin: Vec3,
    direction: Vec3,
    radius: f32,
    half_height: f32,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    // Curved wall: |o.xz + t*d.xz|^2 = r^2.
    let a = direction.x * direction.x + direction.z * direction.z;
    if a > DEGENERATE_EPS {
        let half_b = origin.x * direction.x + origin.z * direction.z;
        let c = origin.x * origin.x + origin.z * origin.z - radius * radius;
        let disc = half_b * half_b - a * c;
        if disc >= 0.0 {
            let root = disc.sqrt();
            for t in [(-half_b - root) / a, (-half_b + root) / a] {
                if t >= 0.0 && t <= max_dist {
                    let y = origin.y + direction.y * t;
                    if y.abs() <= half_height {
                        let n =
                            Vec3::new(origin.x + direction.x * t, 0.0, origin.z + direction.z * t)
                                .normalize_or(Vec3::X);
                        best = keep_closest_hit(best, Some((t, n)));
                        break; // Nearer root first.
                    }
                }
            }
        }
    }
    // Caps: planes y = ±half_height with a radial check.
    if direction.y.abs() > DEGENERATE_EPS {
        for (plane_y, n) in [(half_height, Vec3::Y), (-half_height, Vec3::NEG_Y)] {
            let t = (plane_y - origin.y) / direction.y;
            if t >= 0.0 && t <= max_dist {
                let px = origin.x + direction.x * t;
                let pz = origin.z + direction.z * t;
                if px * px + pz * pz <= radius * radius {
                    best = keep_closest_hit(best, Some((t, n)));
                }
            }
        }
    }
    best
}

/// Ray vs a solid cone (apex `+half_height`, base disk at `-half_height`,
/// local frame): cone-surface quadratic plus the base cap. The apex hit
/// reports +Y (the tip direction) — the tip normal is undefined.
fn ray_cone_hit(
    origin: Vec3,
    direction: Vec3,
    radius: f32,
    half_height: f32,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    // Surface: x^2 + z^2 = k^2 * (h - y)^2, k = r / (2h).
    let k = if half_height > MIN_SEGMENT_LENGTH {
        radius / (2.0 * half_height)
    } else {
        return None;
    };
    let mut best: Option<(f32, Vec3)> = None;
    let a =
        direction.x * direction.x + direction.z * direction.z - k * k * direction.y * direction.y;
    let h = k * k * (half_height - origin.y) * direction.y
        + origin.x * direction.x
        + origin.z * direction.z;
    let c = origin.x * origin.x + origin.z * origin.z
        - k * k * (half_height - origin.y) * (half_height - origin.y);
    // Quadratic a*t^2 + 2*h*t + c = 0 (linear fallback when a ~ 0).
    let mut roots = [0.0f32; 2];
    let n_roots = if a.abs() > DEGENERATE_EPS {
        let disc = h * h - a * c;
        if disc < 0.0 {
            0
        } else {
            let root = disc.sqrt();
            roots = [(-h - root) / a, (-h + root) / a];
            2
        }
    } else if h.abs() > DEGENERATE_EPS {
        roots = [-c / (2.0 * h), f32::INFINITY];
        1
    } else {
        0
    };
    for t in roots.into_iter().take(n_roots) {
        if t >= 0.0 && t <= max_dist {
            let y = origin.y + direction.y * t;
            if y >= -half_height && y <= half_height {
                let px = origin.x + direction.x * t;
                let pz = origin.z + direction.z * t;
                // Gradient of x^2+z^2-k^2*(h-y)^2, outward.
                let n = Vec3::new(px, k * k * (half_height - y), pz).normalize_or(Vec3::Y);
                best = keep_closest_hit(best, Some((t, n)));
            }
        }
    }
    // Base cap disk at y = -half_height.
    if direction.y.abs() > DEGENERATE_EPS {
        let t = (-half_height - origin.y) / direction.y;
        if t >= 0.0 && t <= max_dist {
            let px = origin.x + direction.x * t;
            let pz = origin.z + direction.z * t;
            if px * px + pz * pz <= radius * radius {
                best = keep_closest_hit(best, Some((t, Vec3::NEG_Y)));
            }
        }
    }
    best
}

/// Ray vs a convex hull (local frame): slab test over the triangulated
/// faces. Empty faces (over-cap hulls) report no hit — documented in
/// [`crate::shape::ConvexHull`].
fn ray_hull_hit(
    origin: Vec3,
    direction: Vec3,
    hull: &crate::shape::ConvexHull,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    const EPS: f32 = MIN_SEGMENT_LENGTH;
    let mut best: Option<(f32, Vec3)> = None;
    for f in &hull.faces {
        let (a, b, c) = (
            hull.vertices[f.0.index()],
            hull.vertices[f.1.index()],
            hull.vertices[f.2.index()],
        );
        let n = (b - a).cross(c - a);
        let denom = n.dot(direction);
        if denom.abs() < EPS {
            continue;
        }
        let t = n.dot(a - origin) / denom;
        if t < 0.0 || t > max_dist {
            continue;
        }
        let p = origin + direction * t;
        // Inside-triangle edge tests (same winding as the outward face).
        let e0 = b - a;
        let e1 = c - b;
        let e2 = a - c;
        if e0.cross(p - a).dot(n) < -EPS
            || e1.cross(p - b).dot(n) < -EPS
            || e2.cross(p - c).dot(n) < -EPS
        {
            continue;
        }
        let normal = if denom < 0.0 {
            n.normalize()
        } else {
            -n.normalize()
        };
        best = keep_closest_hit(best, Some((t, normal)));
    }
    best
}

/// Ray vs a heightfield (local frame): Amanatides–Woo DDA over the grid,
/// each visited cell tested as a solid column box (the union entry is the
/// minimum box entry — a point inside another box would contradict
/// minimality). Bounded walk, deterministic order.
fn ray_heightfield_hit(
    origin: Vec3,
    direction: Vec3,
    hf: &crate::shape::Heightfield,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    // Canonical validity gate: legacy-constructed grids degrade to a miss
    // instead of walking garbage (see `Heightfield::try_height_at`).
    if !hf.is_valid() || max_dist < 0.0 {
        return None;
    }
    // Local grid coordinates (float cell indices).
    let to_cell = |x: f32, n: usize| x / hf.cell + (n - 1) as f32 * HALF;
    let mut cx = to_cell(origin.x, hf.cols).floor() as isize;
    let mut cz = to_cell(origin.z, hf.rows).floor() as isize;
    let step_x = if direction.x > 0.0 {
        1
    } else if direction.x < 0.0 {
        -1
    } else {
        0
    };
    let step_z = if direction.z > 0.0 {
        1
    } else if direction.z < 0.0 {
        -1
    } else {
        0
    };
    // Parametric distance to the next cell boundary per axis.
    let x_origin = -((hf.cols - 1) as f32) * HALF * hf.cell;
    let z_origin = -((hf.rows - 1) as f32) * HALF * hf.cell;
    let mut t_max_x = if step_x == 0 {
        f32::INFINITY
    } else {
        let boundary = x_origin + (if step_x > 0 { cx + 1 } else { cx }) as f32 * hf.cell;
        (boundary - origin.x) / direction.x
    };
    let mut t_max_z = if step_z == 0 {
        f32::INFINITY
    } else {
        let boundary = z_origin + (if step_z > 0 { cz + 1 } else { cz }) as f32 * hf.cell;
        (boundary - origin.z) / direction.z
    };
    let t_delta_x = if step_x == 0 {
        f32::INFINITY
    } else {
        (hf.cell / direction.x).abs()
    };
    let t_delta_z = if step_z == 0 {
        f32::INFINITY
    } else {
        (hf.cell / direction.z).abs()
    };
    let (y_min, _) = hf.height_range();
    // Bounded walk: at most one full grid diagonal plus margin.
    const HF_WALK_DIAG_MUL: usize = 4;
    const HF_WALK_MARGIN: usize = 8;
    let max_steps = HF_WALK_DIAG_MUL * (hf.rows + hf.cols) + HF_WALK_MARGIN;
    let mut best: Option<(f32, Vec3)> = None;
    // Degenerate ray (straight down the Y axis): a single cell owns the
    // whole walk — test it and return.
    if step_x == 0 && step_z == 0 {
        if cx >= 0 && cx < hf.cols as isize && cz >= 0 && cz < hf.rows as isize {
            let h = hf.heights[cz as usize * hf.cols + cx as usize].max(y_min);
            let bmin = Vec3::new(
                x_origin + cx as f32 * hf.cell,
                y_min,
                z_origin + cz as f32 * hf.cell,
            );
            let bmax = Vec3::new(bmin.x + hf.cell, h, bmin.z + hf.cell);
            return ray_aabb_hit(origin, direction, bmin, bmax, max_dist);
        }
        return None;
    }
    for _ in 0..max_steps {
        if cx >= 0 && cx < hf.cols as isize && cz >= 0 && cz < hf.rows as isize {
            let h = hf.heights[cz as usize * hf.cols + cx as usize].max(y_min);
            let bmin = Vec3::new(
                x_origin + cx as f32 * hf.cell,
                y_min,
                z_origin + cz as f32 * hf.cell,
            );
            let bmax = Vec3::new(bmin.x + hf.cell, h, bmin.z + hf.cell);
            if let Some((t, n)) = ray_aabb_hit(origin, direction, bmin, bmax, max_dist) {
                best = keep_closest_hit(best, Some((t, n)));
                // A hit nearer than the next cell boundary is final: later
                // cells start farther along the ray.
                if t <= t_max_x.min(t_max_z) {
                    break;
                }
            }
        }
        // Advance to the next cell.
        if t_max_x < t_max_z {
            if t_max_x > max_dist && best.is_some() {
                break;
            }
            cx += step_x;
            t_max_x += t_delta_x;
        } else {
            if t_max_z > max_dist && best.is_some() {
                break;
            }
            cz += step_z;
            t_max_z += t_delta_z;
        }
        if t_max_x > max_dist && t_max_z > max_dist {
            break;
        }
        if cx < -1 || cx > hf.cols as isize || cz < -1 || cz > hf.rows as isize {
            break;
        }
    }
    best
}

/// Ray vs a triangle mesh (local frame): BVH walk with slab pruning,
/// each surviving triangle tested with the hull face slab. Deterministic
/// leaf order; nearest hit wins.
fn ray_trimesh_hit(
    origin: Vec3,
    direction: Vec3,
    mesh: &crate::shape::TriMesh,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    if mesh.tris.is_empty() || max_dist < 0.0 {
        return None;
    }
    let mut best: Option<(f32, Vec3)> = None;
    let mut limit = max_dist;
    let mut stack = [0u32; BVH_STACK_CAP];
    let mut len = 1usize;
    while len > 0 {
        len -= 1;
        let ni = stack[len] as usize;
        if ni >= mesh.nodes.len() {
            continue;
        }
        let node = &mesh.nodes[ni];
        // Slab prune: skip subtrees entered past the best hit so far.
        if ray_aabb_hit(origin, direction, node.min, node.max, limit).is_none() {
            continue;
        }
        if let Some((start, count)) = node.link.leaf_range() {
            let end = (start + count) as usize;
            for o in start as usize..end.min(mesh.order.len()) {
                let Some(t) = mesh.ordered_triangle(o) else {
                    continue;
                };
                let crate::shape::Shape::ConvexHull(hull) = &mesh.tris[t] else {
                    continue;
                };
                // Triangle verts are centroid-relative: shift the ray.
                let c = mesh.centroids[t];
                if let Some((d, n)) = ray_hull_hit(origin - c, direction, hull, limit)
                    && d < limit
                {
                    limit = d;
                    best = Some((d, n));
                }
            }
        } else if let Some((left, right)) = node.link.children() {
            if len + 2 > BVH_STACK_CAP {
                break; // Depth guard: keep the best hit so far.
            }
            // Near-first order is irrelevant for correctness (best-tracked
            // pruning); push right-then-left so left pops first.
            stack[len] = right;
            stack[len + 1] = left;
            len += 2;
        }
    }
    best
}

/// Ray vs an axis-aligned box given by corners (local frame). Returns
/// distance and the entry-face normal.
fn ray_aabb_hit(
    origin: Vec3,
    direction: Vec3,
    box_min: Vec3,
    box_max: Vec3,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let mut near = 0.0f32;
    let mut far = max_dist;
    let mut normal = Vec3::X;
    for (o, d, mn, mx, neg, pos) in [
        (
            origin.x,
            direction.x,
            box_min.x,
            box_max.x,
            Vec3::NEG_X,
            Vec3::X,
        ),
        (
            origin.y,
            direction.y,
            box_min.y,
            box_max.y,
            Vec3::NEG_Y,
            Vec3::Y,
        ),
        (
            origin.z,
            direction.z,
            box_min.z,
            box_max.z,
            Vec3::NEG_Z,
            Vec3::Z,
        ),
    ] {
        if d.abs() <= DEGENERATE_EPS {
            if o < mn || o > mx {
                return None;
            }
            continue;
        }
        let (t0, t1, n0) = if d > 0.0 {
            ((mn - o) / d, (mx - o) / d, neg)
        } else {
            ((mx - o) / d, (mn - o) / d, pos)
        };
        if t0 > near {
            near = t0;
            normal = n0;
        }
        far = far.min(t1);
        if near > far {
            return None;
        }
    }
    Some((near, normal))
}

impl SequentialImpulseEngine {
    pub(crate) fn raycast_body(
        &self,
        ray: &Ray,
        handle: crate::body::BodyHandle,
        max_dist: f32,
    ) -> Option<RaycastHit> {
        if max_dist.is_nan() || max_dist < 0.0 || !vec3_finite(ray.direction) {
            return None;
        }
        let body = &self.bodies[handle.index()];
        let inverse = body.orientation.inverse();
        let origin = inverse * (ray.origin - body.position);
        let direction = inverse * ray.direction;
        let hit = match &body.shape {
            Shape::Sphere { radius } => {
                ray_sphere_hit(origin, direction, Vec3::ZERO, *radius, max_dist)
            }
            Shape::Box { half_extents } => ray_obb_hit(origin, direction, *half_extents, max_dist),
            Shape::Capsule {
                radius,
                half_height,
            } => ray_capsule_hit(origin, direction, *radius, *half_height, max_dist),
            Shape::Cylinder {
                radius,
                half_height,
            } => ray_cylinder_hit(origin, direction, *radius, *half_height, max_dist),
            Shape::Cone {
                radius,
                half_height,
            } => ray_cone_hit(origin, direction, *radius, *half_height, max_dist),
            Shape::ConvexHull(hull) => ray_hull_hit(origin, direction, hull, max_dist),
            Shape::Heightfield(hf) => ray_heightfield_hit(origin, direction, hf, max_dist),
            Shape::TriMesh(mesh) => ray_trimesh_hit(origin, direction, mesh, max_dist),
        }?;
        let (distance, local_normal) = hit;
        let point = ray.point_at(distance);
        let normal = (body.orientation * local_normal).normalize_or(Vec3::Y);
        Some(RaycastHit {
            handle,
            point,
            normal,
            distance,
        })
    }
}
