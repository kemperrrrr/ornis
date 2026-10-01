//! Narrowphase: analytic contact generation for sphere/box/capsule pairs
//! (SAT manifolds, speculative margins) with the distance-oracle fallback,
//! scheduler-sharded dispatch and the cached-substep entry point.

use glam::{Quat, Vec3};
use ornis_schedule::run_levels;
use rustc_hash::FxHashMap;

use crate::body::{BodyType, RigidBody};
use crate::constants::DEGENERATE_LEN2;
use crate::distance;
use crate::engine::{Contact, Manifold};
use crate::flags::CachePolicy;
use crate::shape::Shape;

use super::*;

// ---- Narrow-phase: world-frame analytic contact tests (oriented shapes) ----

/// Squared separation below which a contact distance is treated as collapsed
/// (coincident centers / zero-length normal). Distinct from the mass-domain
/// [`crate::constants::MIN_EFFECTIVE_MASS`] even though the literal matches.
const DEGENERATE_DIST_SQ: f32 = 1e-10;
/// Midpoint / half-extent scale.
const HALF: f32 = 0.5;
/// Edge-axis length floor for SAT edge-edge candidates (m).
const EDGE_AXIS_EPS: f32 = 1e-3;
/// Pair count above which narrowphase shards across workers.
const NARROW_PARALLEL_MIN_PAIRS: usize = 256;
/// Fallback worker hint when `available_parallelism` is unavailable.
const DEFAULT_WORKER_HINT: usize = 4;
/// Coarse shards per worker (spike-tuned: too few starves, too many overhead).
const SHARDS_PER_WORKER: usize = 4;
/// Upper clamp on narrowphase shard count.
const MAX_NARROW_SHARDS: usize = 64;

/// Sphere-sphere. `margin` (G6 speculative): pairs separated by less than
/// the margin still report a contact with NEGATIVE penetration (= the gap),
/// so the solver can stop approach before any overlap exists.
fn sphere_vs_sphere(
    pos_a: Vec3,
    radius_a: f32,
    pos_b: Vec3,
    radius_b: f32,
    margin: f32,
) -> Option<Contact> {
    let diff = pos_b - pos_a;
    let dist_sq = diff.length_squared();
    let radius_sum = radius_a + radius_b + margin;
    if dist_sq > radius_sum * radius_sum {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = diff.normalize_or(Vec3::X);
    let penetration = radius_sum - dist - margin;
    Some(Contact {
        normal,
        penetration,
        contact_point: pos_a + normal * (radius_a - penetration * HALF),
    })
}

/// Sphere vs an oriented box (OBB), resolved in the box's local frame.
/// `margin`: speculative contact distance (see sphere_vs_sphere).
fn sphere_vs_obb(
    sphere_pos: Vec3,
    sphere_radius: f32,
    box_pos: Vec3,
    half_extents: Vec3,
    box_rot: Quat,
    margin: f32,
) -> Option<Contact> {
    let local = box_rot.inverse() * (sphere_pos - box_pos);
    let clamped = local.clamp(-half_extents, half_extents);
    let delta = clamped - local;
    let dist_sq = delta.length_squared();
    let reach = sphere_radius + margin;
    if dist_sq > reach * reach || dist_sq < DEGENERATE_DIST_SQ {
        return None;
    }
    let dist = dist_sq.sqrt();
    // Normal points from the box toward the sphere, in world space.
    let normal = box_rot * (delta / dist);
    let penetration = sphere_radius - dist;
    // Contact point: the sphere's surface point pushed halfway into the
    // overlap (same convention as `sphere_vs_sphere`). The old code used
    // the midpoint of (center, closest box point), which sits at half the
    // radius depth for a touching sphere — halving every friction lever
    // and torque arm. Symptom (measured): a rolling ball converged to a
    // phantom "half-rolling" equilibrium v = ω·r/2 with live slip, because
    // the solver saw zero slip at the half-depth point.
    let dir = delta / dist; // box frame, sphere center toward box surface
    let contact_point = local + dir * (sphere_radius + penetration * HALF);
    Some(Contact {
        normal,
        penetration,
        contact_point: box_pos + box_rot * contact_point,
    })
}

/// Axis used for the OBB overlap test: returns `radius_a + radius_b - separation`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn obb_overlap_on(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
    axis: Vec3,
) -> f32 {
    let aa = rot_a * Vec3::X;
    let ab = rot_a * Vec3::Y;
    let ac = rot_a * Vec3::Z;
    let ba = rot_b * Vec3::X;
    let bb = rot_b * Vec3::Y;
    let bc = rot_b * Vec3::Z;
    let ra = half_a.x * axis.dot(aa).abs()
        + half_a.y * axis.dot(ab).abs()
        + half_a.z * axis.dot(ac).abs();
    let rb = half_b.x * axis.dot(ba).abs()
        + half_b.y * axis.dot(bb).abs()
        + half_b.z * axis.dot(bc).abs();
    let center_dist = (pos_b - pos_a).dot(axis);
    ra + rb - center_dist.abs()
}

/// Separating-axis test for two oriented boxes: returns the deepest
/// penetration axis and depth, or `None` when separated beyond `margin`.
#[allow(clippy::too_many_arguments)]
pub fn obb_sat(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
    margin: f32,
) -> Option<(Vec3, f32)> {
    // SAT: the 3 face normals of each box plus the cross products of their axes.
    let aa = [rot_a * Vec3::X, rot_a * Vec3::Y, rot_a * Vec3::Z];
    let ba = [rot_b * Vec3::X, rot_b * Vec3::Y, rot_b * Vec3::Z];

    // Face normals first; an edge-edge axis may replace a face axis only if
    // it beats it by a margin. Otherwise micro-tilts at face contacts make
    // SAT pick noisy cross-product axes and the normal flickers.
    const FACE_PREFERENCE: f32 = 1e-3;

    let mut best_overlap = f32::MAX;
    let mut best_axis = Vec3::X;

    for u in aa.into_iter().chain(ba) {
        let overlap = obb_overlap_on(pos_a, half_a, rot_a, pos_b, half_b, rot_b, u);
        // Separated along any axis by more than the speculative margin -> no
        // contact. Within the margin the pair still reports as touching
        // (negative overlap = gap): the manifold never blinks off for a
        // substep, and fast pairs get speculative constraints (G6).
        if overlap <= -margin {
            return None;
        }
        if overlap < best_overlap {
            best_overlap = overlap;
            best_axis = u;
        }
    }
    for ai in &aa {
        for bi in &ba {
            let c = ai.cross(*bi);
            // Near-parallel edge pairs give a numerically noisy axis; the
            // face axes cover that case. A too-small threshold here lets
            // float noise produce false "separated" verdicts on micro-tilts
            // and the manifold blinks on/off — a warm-start energy pump.
            if c.length() < EDGE_AXIS_EPS {
                continue;
            }
            let u = c.normalize();
            let overlap = obb_overlap_on(pos_a, half_a, rot_a, pos_b, half_b, rot_b, u);
            if overlap <= -margin {
                return None;
            }
            if overlap < best_overlap - FACE_PREFERENCE {
                best_overlap = overlap;
                best_axis = u;
            }
        }
    }

    // Orient the normal so it points from A to B.
    let normal = if best_axis.dot(pos_b - pos_a) < 0.0 {
        -best_axis
    } else {
        best_axis
    };
    Some((normal, best_overlap))
}

#[allow(clippy::too_many_arguments)]
fn box_vs_box(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
    margin: f32,
) -> Option<Contact> {
    let (normal, penetration) = obb_sat(pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin)?;
    Some(Contact {
        normal,
        penetration,
        contact_point: (pos_a + pos_b) * HALF,
    })
}

/// Eight world-space corners of an oriented box.
fn obb_corners(pos: Vec3, half: Vec3, rot: Quat) -> [Vec3; 8] {
    let x = rot * (Vec3::X * half.x);
    let y = rot * (Vec3::Y * half.y);
    let z = rot * (Vec3::Z * half.z);
    [
        pos + x + y + z,
        pos + x + y - z,
        pos + x - y + z,
        pos + x - y - z,
        pos - x + y + z,
        pos - x + y - z,
        pos - x - y + z,
        pos - x - y - z,
    ]
}

/// Reference-face contact manifold for two boxes (up to 4 points).
#[allow(clippy::too_many_arguments)]
pub fn box_manifold(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
    margin: f32,
) -> Option<Manifold> {
    let (n, _pen) = obb_sat(pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin)?;

    let aa = [rot_a * Vec3::X, rot_a * Vec3::Y, rot_a * Vec3::Z];
    let ba = [rot_b * Vec3::X, rot_b * Vec3::Y, rot_b * Vec3::Z];

    // Half-width of each box projected onto the contact normal.
    let hwn_a = half_a.x * aa[0].dot(n).abs()
        + half_a.y * aa[1].dot(n).abs()
        + half_a.z * aa[2].dot(n).abs();
    let hwn_b = half_b.x * ba[0].dot(n).abs()
        + half_b.y * ba[1].dot(n).abs()
        + half_b.z * ba[2].dot(n).abs();

    // Contact-region tolerance: a corner counts as touching the opposing face
    // when it is within this distance along the (negated) contact normal.
    // G6: this is the pair's speculative margin (base + approach speed · dt),
    // so fast pairs generate constraints BEFORE any overlap exists; points
    // then carry negative penetration (= the remaining gap).
    let depth_tol = margin;
    // Tangential slack beyond the face rectangle: corners slightly outside the
    // face edge (micro-tilts at face contacts) must still generate points,
    // otherwise the manifold collapses to the single-point fallback and the
    // body starts rocking on a corner.
    let tangent_slack = SPEC_BASE;

    // B's corners touching A's face (the face most anti-parallel to `n`),
    // then A's corners touching B's face.
    let mut cand: Vec<(Vec3, f32)> = Vec::new();
    cand.extend(collect_face_corners(
        &obb_corners(pos_b, half_b, rot_b),
        &FaceProbe {
            pos: pos_a,
            half: half_a,
            rot: rot_a,
            hwn: hwn_a,
            dir: n,
            depth_tol,
            slack: tangent_slack,
        },
    ));
    cand.extend(collect_face_corners(
        &obb_corners(pos_a, half_a, rot_a),
        &FaceProbe {
            pos: pos_b,
            half: half_b,
            rot: rot_b,
            hwn: hwn_b,
            dir: -n,
            depth_tol,
            slack: tangent_slack,
        },
    ));

    // Deduplicate in the tangent plane, then keep the deepest four points.
    let mut uniq = dedupe_contact_points(cand, n);
    uniq.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut points = [ManifoldPoint {
        world_point: Vec3::ZERO,
        penetration: 0.0,
    }; MAX_MANIFOLD_POINTS];
    let mut count = 0;
    for (p, d) in uniq.into_iter().take(MAX_MANIFOLD_POINTS) {
        // Speculative points keep their NEGATIVE depth (= the gap); the
        // velocity solver turns it into an approach-speed limit (G6).
        points[count] = ManifoldPoint {
            world_point: p,
            penetration: d,
        };
        count += 1;
    }

    if count == 0 {
        return box_vs_box(pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin).map(|c| {
            Manifold::single(
                crate::body::BodyHandle::from_raw(0),
                crate::body::BodyHandle::from_raw(0),
                c,
            )
        });
    }
    Manifold::from_parts(
        crate::body::BodyHandle::from_raw(0),
        crate::body::BodyHandle::from_raw(0),
        n,
        points,
        count,
    )
}

/// One opposing face of an OBB to test the other box's corners against.
struct FaceProbe {
    pos: Vec3,
    half: Vec3,
    rot: Quat,
    /// Half-width of the probed face's box along the contact normal.
    hwn: f32,
    /// Direction pointing INTO the face along the contact normal.
    dir: Vec3,
    depth_tol: f32,
    slack: f32,
}

/// Collect the corners of one box that touch the probed face of the other:
/// depth along the normal within tolerance AND tangential containment inside
/// the face rectangle (with slack).
fn collect_face_corners(corners: &[Vec3; 8], probe: &FaceProbe) -> Vec<(Vec3, f32)> {
    let mut out: Vec<(Vec3, f32)> = Vec::new();
    for c in corners {
        let local = probe.rot.inverse() * (*c - probe.pos);
        // Depth of the corner relative to the surface along the normal.
        let d = probe.hwn - (*c - probe.pos).dot(probe.dir);
        if d < -probe.depth_tol {
            continue;
        }
        if local.x.abs() <= probe.half.x + probe.slack
            && local.y.abs() <= probe.half.y + probe.slack
            && local.z.abs() <= probe.half.z + probe.slack
        {
            out.push((*c, d));
        }
    }
    out
}

/// Merge near-coincident contact candidates in the tangent plane: the same
/// contact region appears once from each box's corners, offset along the
/// normal by the penetration depth. Keep the deeper representative — a stable
/// 4-point manifold instead of a flickering mix.
fn dedupe_contact_points(cand: Vec<(Vec3, f32)>, n: Vec3) -> Vec<(Vec3, f32)> {
    let mut uniq: Vec<(Vec3, f32)> = Vec::new();
    for (p, d) in cand {
        let mut merged = false;
        for (q, qd) in uniq.iter_mut() {
            let tangential = (p - *q) - n * (p - *q).dot(n);
            // 5 cm: near-coincident points make the constraint system
            // near-singular and PGS oscillates into runaway impulses.
            if tangential.length() < SPEC_BASE {
                if d > *qd {
                    *q = p;
                    *qd = d;
                }
                merged = true;
                break;
            }
        }
        if !merged {
            uniq.push((p, d));
        }
    }
    uniq
}

/// Sphere vs an oriented capsule: closest point on the capsule's segment.
#[allow(clippy::too_many_arguments)]
fn sphere_vs_capsule(
    sphere_pos: Vec3,
    sphere_radius: f32,
    cap_pos: Vec3,
    cap_radius: f32,
    cap_half_height: f32,
    cap_rot: Quat,
    margin: f32,
) -> Option<Contact> {
    let axis = cap_rot * Vec3::Y;
    let bottom = cap_pos - axis * cap_half_height;
    let seg = axis * (2.0 * cap_half_height);
    let t = (sphere_pos - bottom).dot(seg) / seg.length_squared();
    let t = t.clamp(0.0, 1.0);
    let closest = bottom + seg * t;
    let to_sphere = sphere_pos - closest;
    let d = to_sphere.length();
    let rr = cap_radius + sphere_radius + margin;
    if d >= rr || d < DEGENERATE_DIST_SQ {
        return None;
    }
    // Normal points from the capsule toward the sphere.
    let n = to_sphere / d;
    let penetration = rr - d - margin;
    let contact_point = closest + n * (cap_radius - penetration * HALF);
    Some(Contact {
        normal: n,
        penetration,
        contact_point,
    })
}

/// Box vs capsule via the shared analytic `shape_distance` (G6).
/// Thin wrapper: converts the exact distance witness into a speculative
/// contact (`dist <= margin`), keeping the narrowphase zero-alloc.
#[allow(clippy::too_many_arguments)]
fn box_vs_capsule(
    box_pos: Vec3,
    half_extents: Vec3,
    box_rot: Quat,
    cap_pos: Vec3,
    cap_radius: f32,
    cap_half_height: f32,
    cap_rot: Quat,
    margin: f32,
) -> Option<Contact> {
    let d = crate::distance::shape_distance(
        crate::distance::ShapeRef {
            shape: &Shape::Box { half_extents },
            pos: box_pos,
            rot: box_rot,
        },
        crate::distance::ShapeRef {
            shape: &Shape::Capsule {
                radius: cap_radius,
                half_height: cap_half_height,
            },
            pos: cap_pos,
            rot: cap_rot,
        },
    );
    if d.dist > margin {
        return None;
    }
    let ab = d.point_b - d.point_a;
    let len = ab.length();
    if len < DEGENERATE_DIST_SQ {
        return None;
    }
    let mut normal = ab / len; // box → capsule
    if d.dist < 0.0 {
        normal = -normal;
    }
    // Hemisphere rule (see `distance_contact`): crossed witnesses in
    // penetration would push the pair through each other.
    if normal.dot(cap_pos - box_pos) < 0.0 {
        normal = -normal;
    }
    let penetration = -d.dist;
    let contact_point = (d.point_a + d.point_b) * HALF;
    Some(Contact {
        normal,
        penetration,
        contact_point,
    })
}

/// Capsule collision parameters (keeps `capsule_vs_capsule` within the structural gate's
/// argument-count limit).
struct CapsuleShape {
    pos: Vec3,
    radius: f32,
    half_height: f32,
    rot: Quat,
}

/// Capsule-capsule: both segment axes are rotated by the body orientation.
fn capsule_vs_capsule(a: &CapsuleShape, b: &CapsuleShape, margin: f32) -> Option<Contact> {
    let ax = a.rot * Vec3::Y;
    let bx = b.rot * Vec3::Y;
    let bot_a = a.pos - ax * a.half_height;
    let bot_b = b.pos - bx * b.half_height;

    let seg_a = ax * (2.0 * a.half_height);
    let seg_b = bx * (2.0 * b.half_height);
    let diff = bot_b - bot_a;
    let q = seg_a.dot(seg_a);
    let r = seg_a.dot(seg_b);
    let c = seg_b.dot(seg_b);
    let d = seg_a.dot(diff);
    let e = seg_b.dot(diff);
    let det = q * c - r * r;

    let (t_a, t_b) = if det.abs() < DEGENERATE_DIST_SQ {
        (0.0, if c > 0.0 { e / c } else { 0.0 })
    } else {
        ((r * e - c * d) / det, (q * e - r * d) / det)
    };
    let t_a = clamp01(t_a);
    let t_b = clamp01(t_b);

    let closest_a = bot_a + seg_a * t_a;
    let closest_b = bot_b + seg_b * t_b;
    let diff2 = closest_b - closest_a;
    let dist_sq = diff2.length_squared();
    let radius_sum = a.radius + b.radius + margin;
    if dist_sq > radius_sum * radius_sum || dist_sq < DEGENERATE_DIST_SQ {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = diff2 / dist;
    let penetration = radius_sum - dist - margin;
    Some(Contact {
        normal,
        penetration,
        contact_point: (closest_a + closest_b) * HALF,
    })
}

/// Narrow phase over the broadphase pair list. G6: every pair gets a
/// speculative contact margin = base + approach speed · sub_dt, so contacts
/// exist BEFORE overlap; the velocity solver then caps the approach speed
/// to the remaining gap instead of letting the bodies interpenetrate.
#[allow(dead_code)]
fn detect_collisions(
    bodies: &[RigidBody],
    active: &[(usize, usize)],
    asleep: &[bool],
    sub_dt: f32,
) -> Vec<Manifold> {
    let mut out = Vec::new();
    let mut pool = NarrowShardPool::default();
    detect_collisions_into(
        bodies, active, asleep, sub_dt, &mut out, None, 0, None, &mut pool,
    );
    out
}

/// Generic single contact from a pairwise distance query: covers every
/// shape pair without a dedicated manifold builder (cylinder/cone/hull
/// via GJK/EPA, heightfields via columns). Surface-anchored convention
/// (see `sphere_vs_sphere`): the witness on A's surface pushed halfway
/// into the overlap, so friction levers and torque arms stay full-length.
/// Normal points from A to B; `penetration` is the true signed overlap
/// (negative = speculative gap inside the margin).
fn distance_contact(
    i: usize,
    j: usize,
    a: &RigidBody,
    b: &RigidBody,
    margin: f32,
) -> Option<Manifold> {
    let d = distance::shape_distance(
        distance::ShapeRef {
            shape: &a.shape,
            pos: a.position,
            rot: a.orientation,
        },
        distance::ShapeRef {
            shape: &b.shape,
            pos: b.position,
            rot: b.orientation,
        },
    );
    if d.dist > margin {
        return None;
    }
    let axis = d.point_b - d.point_a;
    let mut normal = if axis.length_squared() > DEGENERATE_LEN2 {
        axis.normalize()
    } else {
        (b.position - a.position).normalize_or(Vec3::Y)
    };
    // Hemisphere rule: in penetration the witnesses can cross (the B-side
    // point ends up above the A-side point), flipping the axis against the
    // separating direction — the solver would then push the bodies through
    // each other. Enforce consistency with the center delta (same class of
    // fix as in `box_vs_capsule` below).
    if normal.dot(b.position - a.position) < 0.0 {
        normal = -normal;
    }
    let penetration = -d.dist;
    Some(Manifold::single(
        crate::body::BodyHandle::from(i),
        crate::body::BodyHandle::from(j),
        Contact {
            normal,
            penetration,
            contact_point: d.point_a - normal * (penetration * HALF),
        },
    ))
}

/// Base speculative margin (m): also the AABB inflation used by the
/// broadphase, so pairs within it are guaranteed to reach narrow phase.
const SPEC_BASE: f32 = 0.05;

/// Maximum relative linear speed (m/s) for the cached SAT/box path and the
/// narrow-phase cache: faster pairs bypass the cache (near-zero hit rate)
/// and run the full manifold build.
const SAT_CACHE_MAX_REL_SPEED: f32 = HALF;

/// Squared angular-speed gate for the same cache (rad²/s²).
///
/// Equal to [`SAT_CACHE_MAX_REL_SPEED`]² so linear 0.5 m/s and angular
/// 0.5 rad/s share one threshold magnitude.
const SAT_CACHE_MAX_ANG_SPEED_SQ: f32 = SAT_CACHE_MAX_REL_SPEED * SAT_CACHE_MAX_REL_SPEED;

/// Shard count rule for scheduler-dispatched narrowphase: enough coarse
/// tasks to feed every worker without starving (the spike showed 8 shards
/// on 8 threads ~50% slower than flat rayon from imbalance, while 32
/// shards ran ~20% faster). Order-preserving concat keeps results
/// deterministic for any shard count.
fn narrow_shard_count(pairs: usize) -> usize {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(DEFAULT_WORKER_HINT);
    (threads * SHARDS_PER_WORKER)
        .clamp(DEFAULT_WORKER_HINT, MAX_NARROW_SHARDS)
        .min(pairs.max(1))
}

/// Per-pair narrowphase kernel shared by the parallel and sequential paths:
/// filters, speculative margin, all 9 shape combos with the unified SAT-cache
/// gate, and canonical body ids on the manifold. Pure over shared borrows
/// (`sat_cache` is lock-free), so any scheduler can shard it — this is the
/// unit the scheduler spike compares against rayon.
#[allow(clippy::too_many_arguments)]
fn narrow_pair(
    bodies: &[RigidBody],
    asleep: &[bool],
    i: usize,
    j: usize,
    body_required: Option<&[u32]>,
    cur_substep: u32,
    sub_dt: f32,
    sat_cache: Option<&SatCache>,
) -> Option<Manifold> {
    let a = &bodies[i];
    let b = &bodies[j];
    if !a.can_collide_with(b) || a.is_trigger || b.is_trigger {
        return None;
    }
    if a.body_type == BodyType::Static && b.body_type == BodyType::Static {
        return None;
    }
    // G7: both-asleep pairs are frozen in place — their relative geometry
    // cannot change, so re-running the narrow phase (SAT!) per substep is
    // pure waste. On a settled scene this IS the frame cost. The island
    // graph keeps them composed via the frozen-asleep union instead.
    if asleep[i] && asleep[j] {
        return None;
    }
    // B: per-body substeps — slow pairs only needed for first
    // MIN substeps; fast pairs need all. Skip extra substeps.
    if let Some(req) = body_required
        && req[i].max(req[j]) <= cur_substep
    {
        return None;
    }
    let rel_speed = (a.velocity - b.velocity).length();
    let margin = SPEC_BASE + rel_speed * sub_dt;
    let manifold = match (&a.shape, &b.shape) {
        (&Shape::Sphere { radius: ra }, &Shape::Sphere { radius: rb }) => {
            sphere_vs_sphere(a.position, ra, b.position, rb, margin).map(|c| {
                Manifold::single(
                    crate::body::BodyHandle::from(i),
                    crate::body::BodyHandle::from(j),
                    c,
                )
            })
        }
        (&Shape::Sphere { radius: ra }, &Shape::Box { half_extents: hb }) => {
            sphere_vs_obb(a.position, ra, b.position, hb, b.orientation, margin).map(|c| {
                Manifold::single(
                    crate::body::BodyHandle::from(i),
                    crate::body::BodyHandle::from(j),
                    c,
                )
            })
        }
        (&Shape::Box { half_extents: ha }, &Shape::Sphere { radius: rb }) => {
            sphere_vs_obb(b.position, rb, a.position, ha, a.orientation, margin).map(|c| {
                Manifold::single(
                    crate::body::BodyHandle::from(i),
                    crate::body::BodyHandle::from(j),
                    Contact {
                        normal: -c.normal,
                        penetration: c.penetration,
                        contact_point: c.contact_point,
                    },
                )
            })
        }
        (&Shape::Box { half_extents: ha }, &Shape::Box { half_extents: hb }) => {
            // SAT cache is lock-free (DashMap): shared by both paths; hits
            // reuse the axis only, contacts rebuild from live geometry.
            let use_sat = sat_cache.is_some()
                && cur_substep == 0
                && rel_speed <= SAT_CACHE_MAX_REL_SPEED
                && a.angular_velocity.length_squared() <= SAT_CACHE_MAX_ANG_SPEED_SQ
                && b.angular_velocity.length_squared() <= SAT_CACHE_MAX_ANG_SPEED_SQ;
            if use_sat {
                box_manifold_cached(
                    a.position,
                    ha,
                    a.orientation,
                    b.position,
                    hb,
                    b.orientation,
                    margin,
                    Some((i, j)),
                    sat_cache,
                    CachePolicy::Cached,
                )
            } else {
                box_manifold(
                    a.position,
                    ha,
                    a.orientation,
                    b.position,
                    hb,
                    b.orientation,
                    margin,
                )
            }
        }
        (
            &Shape::Capsule {
                radius: ra,
                half_height: ha,
            },
            &Shape::Capsule {
                radius: rb,
                half_height: hb,
            },
        ) => capsule_vs_capsule(
            &CapsuleShape {
                pos: a.position,
                radius: ra,
                half_height: ha,
                rot: a.orientation,
            },
            &CapsuleShape {
                pos: b.position,
                radius: rb,
                half_height: hb,
                rot: b.orientation,
            },
            margin,
        )
        .map(|c| {
            Manifold::single(
                crate::body::BodyHandle::from(i),
                crate::body::BodyHandle::from(j),
                c,
            )
        }),
        (
            &Shape::Sphere { radius: r },
            &Shape::Capsule {
                radius: cr,
                half_height: hh,
            },
        ) => sphere_vs_capsule(a.position, r, b.position, cr, hh, b.orientation, margin).map(|c| {
            Manifold::single(
                crate::body::BodyHandle::from(i),
                crate::body::BodyHandle::from(j),
                c,
            )
        }),
        (
            &Shape::Capsule {
                radius: cr,
                half_height: hh,
            },
            &Shape::Sphere { radius: r },
        ) => sphere_vs_capsule(b.position, r, a.position, cr, hh, a.orientation, margin).map(|c| {
            Manifold::single(
                crate::body::BodyHandle::from(i),
                crate::body::BodyHandle::from(j),
                Contact {
                    normal: -c.normal,
                    penetration: c.penetration,
                    contact_point: c.contact_point,
                },
            )
        }),
        (
            &Shape::Box { half_extents: ha },
            &Shape::Capsule {
                radius: cr,
                half_height: hh,
            },
        ) => box_vs_capsule(
            a.position,
            ha,
            a.orientation,
            b.position,
            cr,
            hh,
            b.orientation,
            margin,
        )
        .map(|c| {
            Manifold::single(
                crate::body::BodyHandle::from(i),
                crate::body::BodyHandle::from(j),
                c,
            )
        }),
        (
            &Shape::Capsule {
                radius: cr,
                half_height: hh,
            },
            &Shape::Box { half_extents: ha },
        ) => box_vs_capsule(
            b.position,
            ha,
            b.orientation,
            a.position,
            cr,
            hh,
            a.orientation,
            margin,
        )
        .map(|c| {
            Manifold::single(
                crate::body::BodyHandle::from(i),
                crate::body::BodyHandle::from(j),
                Contact {
                    normal: -c.normal,
                    penetration: c.penetration,
                    contact_point: c.contact_point,
                },
            )
        }),
        // Every other pair (cylinder/cone/hull via GJK/EPA, heightfields
        // via columns): generic single contact from the distance query.
        // The analytic arms above keep their dedicated manifolds.
        _ => distance_contact(i, j, a, b, margin),
    };
    manifold.map(|mut m| {
        m.body_a = crate::body::BodyHandle::from(i);
        m.body_b = crate::body::BodyHandle::from(j);
        m
    })
}

/// Broadphase candidate pairs into contact manifolds, with SAT-cache and
/// shard-pool acceleration for large scenes. Appends to `out`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::collapsible_if)]
pub fn detect_collisions_into(
    bodies: &[RigidBody],
    active: &[(usize, usize)],
    asleep: &[bool],
    sub_dt: f32,
    out: &mut Vec<Manifold>,
    body_required: Option<&[u32]>,
    cur_substep: u32,
    sat_cache: Option<&SatCache>,
    pool: &mut NarrowShardPool,
) {
    out.clear();
    // Parallel narrowphase for large candidate sets: SAT/box_manifold is heavy,
    // and bodies/asleep are read-only. Threshold keeps small scenes sequential.
    if active.len() > NARROW_PARALLEL_MIN_PAIRS {
        // Scheduler dispatch (one level, K coarse shards): same kernel, same
        // order-preserving concat as the flat rayon path it replaces, so
        // results are identical for any shard count. Shard buffers come from
        // the engine-owned pool — no allocation on the hot path.
        let shards = narrow_shard_count(active.len());
        pool.ensure(shards);
        let bufs = &pool.bufs;
        run_levels(&pool.level, shards, true, |shard| {
            let lo = shard * active.len() / shards;
            let hi = (shard + 1) * active.len() / shards;
            let mut guard = bufs[shard].lock().unwrap_or_else(|e| e.into_inner());
            for &(i, j) in &active[lo..hi] {
                if let Some(m) = narrow_pair(
                    bodies,
                    asleep,
                    i,
                    j,
                    body_required,
                    cur_substep,
                    sub_dt,
                    sat_cache,
                ) {
                    guard.push(m);
                }
            }
        });
        out.reserve(active.len());
        for b in &pool.bufs {
            out.extend(b.lock().unwrap_or_else(|e| e.into_inner()).drain(..));
        }
        return;
    }
    for &(i, j) in active {
        let a = &bodies[i];
        let b = &bodies[j];
        if !a.can_collide_with(b) || a.is_trigger || b.is_trigger {
            continue;
        }
        if a.body_type == BodyType::Static && b.body_type == BodyType::Static {
            continue;
        }
        // G7: both-asleep pairs are frozen in place — their relative geometry
        // cannot change, so re-running the narrow phase (SAT!) per substep is
        // pure waste. On a settled scene this IS the frame cost. The island
        // graph keeps them composed via the frozen-asleep union instead.
        if asleep[i] && asleep[j] {
            continue;
        }
        if let Some(req) = body_required
            && req[i].max(req[j]) <= cur_substep
        {
            continue;
        }
        let rel_speed = (a.velocity - b.velocity).length();
        let margin = SPEC_BASE + rel_speed * sub_dt;

        let manifold =
            match (&a.shape, &b.shape) {
                (&Shape::Sphere { radius: ra }, &Shape::Sphere { radius: rb }) => {
                    sphere_vs_sphere(a.position, ra, b.position, rb, margin).map(|c| {
                        Manifold::single(
                            crate::body::BodyHandle::from(i),
                            crate::body::BodyHandle::from(j),
                            c,
                        )
                    })
                }
                (&Shape::Sphere { radius: ra }, &Shape::Box { half_extents: hb }) => {
                    sphere_vs_obb(a.position, ra, b.position, hb, b.orientation, margin).map(|c| {
                        Manifold::single(
                            crate::body::BodyHandle::from(i),
                            crate::body::BodyHandle::from(j),
                            c,
                        )
                    })
                }
                (&Shape::Box { half_extents: ha }, &Shape::Sphere { radius: rb }) => {
                    sphere_vs_obb(b.position, rb, a.position, ha, a.orientation, margin).map(|c| {
                        Manifold::single(
                            crate::body::BodyHandle::from(i),
                            crate::body::BodyHandle::from(j),
                            Contact {
                                normal: -c.normal,
                                penetration: c.penetration,
                                contact_point: c.contact_point,
                            },
                        )
                    })
                }
                (&Shape::Box { half_extents: ha }, &Shape::Box { half_extents: hb }) => {
                    // SAT cache is lock-free (DashMap): shared by both paths; hits
                    // reuse the axis only, contacts rebuild from live geometry.
                    let use_sat = sat_cache.is_some()
                        && cur_substep == 0
                        && (a.velocity - b.velocity).length() <= SAT_CACHE_MAX_REL_SPEED
                        && a.angular_velocity.length_squared() <= SAT_CACHE_MAX_ANG_SPEED_SQ
                        && b.angular_velocity.length_squared() <= SAT_CACHE_MAX_ANG_SPEED_SQ;
                    if use_sat {
                        box_manifold_cached(
                            a.position,
                            ha,
                            a.orientation,
                            b.position,
                            hb,
                            b.orientation,
                            margin,
                            Some((i, j)),
                            sat_cache,
                            CachePolicy::Cached,
                        )
                    } else {
                        box_manifold(
                            a.position,
                            ha,
                            a.orientation,
                            b.position,
                            hb,
                            b.orientation,
                            margin,
                        )
                    }
                }
                (
                    &Shape::Capsule {
                        radius: ra,
                        half_height: ha,
                    },
                    &Shape::Capsule {
                        radius: rb,
                        half_height: hb,
                    },
                ) => capsule_vs_capsule(
                    &CapsuleShape {
                        pos: a.position,
                        radius: ra,
                        half_height: ha,
                        rot: a.orientation,
                    },
                    &CapsuleShape {
                        pos: b.position,
                        radius: rb,
                        half_height: hb,
                        rot: b.orientation,
                    },
                    margin,
                )
                .map(|c| {
                    Manifold::single(
                        crate::body::BodyHandle::from(i),
                        crate::body::BodyHandle::from(j),
                        c,
                    )
                }),
                (
                    &Shape::Sphere { radius: r },
                    &Shape::Capsule {
                        radius: cr,
                        half_height: hh,
                    },
                ) => sphere_vs_capsule(a.position, r, b.position, cr, hh, b.orientation, margin)
                    .map(|c| {
                        Manifold::single(
                            crate::body::BodyHandle::from(i),
                            crate::body::BodyHandle::from(j),
                            c,
                        )
                    }),
                (
                    &Shape::Capsule {
                        radius: cr,
                        half_height: hh,
                    },
                    &Shape::Sphere { radius: r },
                ) => sphere_vs_capsule(b.position, r, a.position, cr, hh, a.orientation, margin)
                    .map(|c| {
                        Manifold::single(
                            crate::body::BodyHandle::from(i),
                            crate::body::BodyHandle::from(j),
                            Contact {
                                normal: -c.normal,
                                penetration: c.penetration,
                                contact_point: c.contact_point,
                            },
                        )
                    }),
                (
                    &Shape::Box { half_extents: ha },
                    &Shape::Capsule {
                        radius: cr,
                        half_height: hh,
                    },
                ) => box_vs_capsule(
                    a.position,
                    ha,
                    a.orientation,
                    b.position,
                    cr,
                    hh,
                    b.orientation,
                    margin,
                )
                .map(|c| {
                    Manifold::single(
                        crate::body::BodyHandle::from(i),
                        crate::body::BodyHandle::from(j),
                        c,
                    )
                }),
                (
                    &Shape::Capsule {
                        radius: cr,
                        half_height: hh,
                    },
                    &Shape::Box { half_extents: ha },
                ) => box_vs_capsule(
                    b.position,
                    ha,
                    b.orientation,
                    a.position,
                    cr,
                    hh,
                    a.orientation,
                    margin,
                )
                .map(|c| {
                    Manifold::single(
                        crate::body::BodyHandle::from(i),
                        crate::body::BodyHandle::from(j),
                        Contact {
                            normal: -c.normal,
                            penetration: c.penetration,
                            contact_point: c.contact_point,
                        },
                    )
                }),
                // Every other pair (cylinder/cone/hull via GJK/EPA, heightfields
                // via columns): generic single contact from the distance query.
                _ => distance_contact(i, j, a, b, margin),
            };

        if let Some(mut m) = manifold {
            m.body_a = crate::body::BodyHandle::from(i);
            m.body_b = crate::body::BodyHandle::from(j);
            out.push(m);
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::collapsible_if)]
fn obb_sat_cached(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
    margin: f32,
    key: Option<(usize, usize)>,
    sat_cache: Option<&SatCache>,
    policy: CachePolicy,
) -> Option<(Vec3, f32)> {
    if policy.use_cache() {
        if let (Some(k), Some(cache)) = (key, sat_cache) {
            if let Some(entry) = cache.get(&k) {
                if sat_cache_hit(&entry, pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin) {
                    return entry.result;
                }
            }
        }
    }
    let res = obb_sat(pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin);
    if policy.use_cache() {
        if let (Some(k), Some(cache)) = (key, sat_cache) {
            cache.insert(
                k,
                SatCacheEntry {
                    pos_a,
                    half_a,
                    rot_a,
                    pos_b,
                    half_b,
                    rot_b,
                    margin,
                    result: res,
                },
            );
        }
    }
    res
}

#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::collapsible_if)]
fn box_manifold_cached(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
    margin: f32,
    key: Option<(usize, usize)>,
    sat_cache: Option<&SatCache>,
    policy: CachePolicy,
) -> Option<Manifold> {
    let (n, _pen) = obb_sat_cached(
        pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin, key, sat_cache, policy,
    )?;
    // Reuse normal from SAT — remainder is face-corner collection (cheap vs 15-axis SAT).
    let aa = [rot_a * Vec3::X, rot_a * Vec3::Y, rot_a * Vec3::Z];
    let ba = [rot_b * Vec3::X, rot_b * Vec3::Y, rot_b * Vec3::Z];
    let hwn_a = half_a.x * aa[0].dot(n).abs()
        + half_a.y * aa[1].dot(n).abs()
        + half_a.z * aa[2].dot(n).abs();
    let hwn_b = half_b.x * ba[0].dot(n).abs()
        + half_b.y * ba[1].dot(n).abs()
        + half_b.z * ba[2].dot(n).abs();
    let depth_tol = margin;
    let tangent_slack = SPEC_BASE;
    let mut cand: Vec<(Vec3, f32)> = Vec::new();
    cand.extend(collect_face_corners(
        &obb_corners(pos_b, half_b, rot_b),
        &FaceProbe {
            pos: pos_a,
            half: half_a,
            rot: rot_a,
            hwn: hwn_a,
            dir: n,
            depth_tol,
            slack: tangent_slack,
        },
    ));
    cand.extend(collect_face_corners(
        &obb_corners(pos_a, half_a, rot_a),
        &FaceProbe {
            pos: pos_b,
            half: half_b,
            rot: rot_b,
            hwn: hwn_b,
            dir: -n,
            depth_tol,
            slack: tangent_slack,
        },
    ));
    let mut uniq = dedupe_contact_points(cand, n);
    uniq.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut points = [ManifoldPoint {
        world_point: Vec3::ZERO,
        penetration: 0.0,
    }; MAX_MANIFOLD_POINTS];
    let mut count = 0;
    for (p, d) in uniq.into_iter().take(MAX_MANIFOLD_POINTS) {
        points[count] = ManifoldPoint {
            world_point: p,
            penetration: d,
        };
        count += 1;
    }
    if count == 0 {
        return box_vs_box(pos_a, half_a, rot_a, pos_b, half_b, rot_b, margin).map(|c| {
            Manifold::single(
                crate::body::BodyHandle::from_raw(0),
                crate::body::BodyHandle::from_raw(0),
                c,
            )
        });
    }
    Manifold::from_parts(
        crate::body::BodyHandle::from_raw(0),
        crate::body::BodyHandle::from_raw(0),
        n,
        points,
        count,
    )
}

#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::collapsible_if)]
pub(crate) fn detect_collisions_into_with_cache(
    bodies: &[RigidBody],
    active: &[(usize, usize)],
    asleep: &[bool],
    sub_dt: f32,
    out: &mut Vec<Manifold>,
    body_required: Option<&[u32]>,
    cur_substep: u32,
    cache: &mut FxHashMap<(usize, usize), NarrowCacheEntry>,
    sat_cache: Option<&SatCache>,
    pool: &mut NarrowShardPool,
) {
    // Only cache the first substep: later substeps are filtered to fast bodies,
    // hit rate is near zero and the HashMap overhead dominates.
    if cur_substep != 0 {
        detect_collisions_into(
            bodies,
            active,
            asleep,
            sub_dt,
            out,
            body_required,
            cur_substep,
            sat_cache,
            pool,
        );
        return;
    }
    out.clear();
    // Evict stale entries where bodies were removed.
    cache.retain(|(a, b), _| *a < bodies.len() && *b < bodies.len());
    if let Some(sc) = sat_cache {
        sc.retain(|(a, b), _| *a < bodies.len() && *b < bodies.len());
    }
    if active.is_empty() {
        return;
    }
    // Fast path: small active sets bypass cache overhead (same as sequential threshold).
    // For large sets we do two-phase: hits sequential, misses parallel via the original routine.
    let mut misses: Vec<(usize, usize)> = Vec::with_capacity(active.len());
    let mut fast_misses: Vec<(usize, usize)> = Vec::new();
    for &(i, j) in active {
        let a = &bodies[i];
        let b = &bodies[j];
        // Early rejects identical to the original routine — don't cache trivial rejects.
        if !a.can_collide_with(b) || a.is_trigger || b.is_trigger {
            continue;
        }
        if a.body_type == BodyType::Static && b.body_type == BodyType::Static {
            continue;
        }
        if asleep[i] && asleep[j] {
            continue;
        }
        if let Some(req) = body_required
            && req[i].max(req[j]) <= cur_substep
        {
            continue;
        }
        let rel_speed = (a.velocity - b.velocity).length();
        // Fast-moving pairs have near-zero cache hit rate — bypass HashMap lookup and don't cache.
        if rel_speed > SAT_CACHE_MAX_REL_SPEED
            || a.angular_velocity.length_squared() > SAT_CACHE_MAX_ANG_SPEED_SQ
            || b.angular_velocity.length_squared() > SAT_CACHE_MAX_ANG_SPEED_SQ
        {
            fast_misses.push((i, j));
            continue;
        }
        let margin = SPEC_BASE + rel_speed * sub_dt;
        let key = (i, j);
        if let Some(entry) = cache.get(&key) {
            if narrow_cache_hit(entry, a, b, margin) {
                if let Some(m) = &entry.manifold {
                    out.push(m.clone());
                }
                continue;
            }
        }
        misses.push((i, j));
    }
    if misses.is_empty() && fast_misses.is_empty() {
        return;
    }
    // Fast-moving pairs: compute directly without caching.
    if !fast_misses.is_empty() {
        let mut fast_tmp: Vec<Manifold> = Vec::new();
        detect_collisions_into(
            bodies,
            &fast_misses,
            asleep,
            sub_dt,
            &mut fast_tmp,
            body_required,
            cur_substep,
            sat_cache,
            pool,
        );
        out.extend(fast_tmp);
    }
    if misses.is_empty() {
        return;
    }
    // Compute misses with the original (potentially parallel) routine into a temp vec.
    let mut tmp: Vec<Manifold> = Vec::new();
    detect_collisions_into(
        bodies,
        &misses,
        asleep,
        sub_dt,
        &mut tmp,
        body_required,
        cur_substep,
        sat_cache,
        pool,
    );
    // Populate cache for misses.
    let mut tmp_map: FxHashMap<(usize, usize), Option<Manifold>> = FxHashMap::default();
    for m in &tmp {
        tmp_map.insert((m.body_a.index(), m.body_b.index()), Some(m.clone()));
    }
    for &(i, j) in &misses {
        let key = (i, j);
        if tmp_map.contains_key(&key) {
            continue;
        }
        tmp_map.insert(key, None);
    }
    for &(i, j) in &misses {
        let a = &bodies[i];
        let b = &bodies[j];
        let rel_speed = (a.velocity - b.velocity).length();
        let margin = SPEC_BASE + rel_speed * sub_dt;
        let manifold = tmp_map.get(&(i, j)).and_then(|o| o.clone());
        cache.insert(
            (i, j),
            NarrowCacheEntry {
                pos_a: a.position,
                pos_b: b.position,
                rot_a: a.orientation,
                rot_b: b.orientation,
                margin,
                manifold: manifold.clone(),
            },
        );
        if let Some(m) = manifold {
            out.push(m);
        }
    }
}
