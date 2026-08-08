use std::collections::HashMap;

use glam::{Quat, Vec3};

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::math::{AABB, Ray, RaycastHit};
use crate::shape::Shape;

pub trait PhysicsEngine: Send + Sync {
    fn step(&mut self, dt: f32);
    fn add_body(&mut self, body: RigidBody) -> BodyHandle;
    fn remove_body(&mut self, handle: BodyHandle);
    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody>;
    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody>;
    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit>;
    fn shapecast(&self, shape: &Shape, from: Vec3, to: Vec3) -> Option<RaycastHit>;
}

struct Contact {
    normal: Vec3,
    penetration: f32,
    contact_point: Vec3,
}

/// A single contact point inside a manifold (G2).
#[derive(Clone, Copy, Debug)]
struct ManifoldPoint {
    world_point: Vec3,
    penetration: f32,
}

/// Contact manifold: one normal + up to 4 points per body pair.
#[derive(Clone, Debug)]
struct Manifold {
    body_a: BodyHandle,
    body_b: BodyHandle,
    normal: Vec3,
    point_count: usize,
    points: [ManifoldPoint; 4],
}

impl Manifold {
    fn single(body_a: BodyHandle, body_b: BodyHandle, c: Contact) -> Self {
        let mut points = [ManifoldPoint {
            world_point: Vec3::ZERO,
            penetration: 0.0,
        }; 4];
        points[0] = ManifoldPoint {
            world_point: c.contact_point,
            penetration: c.penetration,
        };
        Self {
            body_a,
            body_b,
            normal: c.normal,
            point_count: 1,
            points,
        }
    }
}

#[inline]
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

#[inline]
fn inv_inertia_axis(i: f32) -> f32 {
    if i > 0.0 { 1.0 / i } else { 0.0 }
}

/// Apply the inverse world-space inertia tensor (body axes rotated to world).
fn mul_inv_inertia(inertia: Vec3, orientation: glam::Quat, v: Vec3) -> Vec3 {
    let body = orientation.inverse() * v;
    Vec3::new(
        inv_inertia_axis(inertia.x) * body.x,
        inv_inertia_axis(inertia.y) * body.y,
        inv_inertia_axis(inertia.z) * body.z,
    )
}

fn compute_aabb(body: &RigidBody) -> AABB {
    body.shape.aabb(body.position, body.orientation)
}

// ---- Narrow-phase: world-frame analytic contact tests (oriented shapes) ----

fn sphere_vs_sphere(pos_a: Vec3, radius_a: f32, pos_b: Vec3, radius_b: f32) -> Option<Contact> {
    let diff = pos_b - pos_a;
    let dist_sq = diff.length_squared();
    let radius_sum = radius_a + radius_b;
    if dist_sq > radius_sum * radius_sum || dist_sq < 1e-10 {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = diff / dist;
    let penetration = radius_sum - dist;
    Some(Contact {
        normal,
        penetration,
        contact_point: pos_a + normal * (radius_a - penetration * 0.5),
    })
}

/// Sphere vs an oriented box (OBB), resolved in the box's local frame.
fn sphere_vs_obb(
    sphere_pos: Vec3,
    sphere_radius: f32,
    box_pos: Vec3,
    half_extents: Vec3,
    box_rot: Quat,
) -> Option<Contact> {
    let local = box_rot.inverse() * (sphere_pos - box_pos);
    let clamped = local.clamp(-half_extents, half_extents);
    let delta = clamped - local;
    let dist_sq = delta.length_squared();
    if dist_sq > sphere_radius * sphere_radius || dist_sq < 1e-10 {
        return None;
    }
    let dist = dist_sq.sqrt();
    // Normal points from the box toward the sphere, in world space.
    let normal = box_rot * (delta / dist);
    let penetration = sphere_radius - dist;
    // Contact point: where the sphere touches the box (midway on the normal).
    let contact_point = (local + clamped) * 0.5;
    Some(Contact {
        normal,
        penetration,
        contact_point: box_pos + box_rot * contact_point,
    })
}

/// Axis used for the OBB overlap test: returns `radius_a + radius_b - separation`.
#[allow(clippy::too_many_arguments)]
fn obb_overlap_on(
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

#[allow(clippy::too_many_arguments)]
fn obb_sat(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
) -> Option<(Vec3, f32)> {
    // SAT: the 3 face normals of each box plus the cross products of their axes.
    let aa = [rot_a * Vec3::X, rot_a * Vec3::Y, rot_a * Vec3::Z];
    let ba = [rot_b * Vec3::X, rot_b * Vec3::Y, rot_b * Vec3::Z];

    let mut best_overlap = f32::MAX;
    let mut best_axis = Vec3::X;

    // Candidate axes: face normals first, then edge-edge cross products.
    let mut axes: Vec<Vec3> = Vec::with_capacity(15);
    for i in 0..3 {
        axes.push(aa[i]);
        axes.push(ba[i]);
    }
    for ai in &aa {
        for bi in &ba {
            let c = ai.cross(*bi);
            if c.length() < 1e-6 {
                continue;
            }
            axes.push(c.normalize());
        }
    }

    for u in axes {
        let overlap = obb_overlap_on(pos_a, half_a, rot_a, pos_b, half_b, rot_b, u);
        // Separated along any axis -> no contact.
        if overlap <= 0.0 {
            return None;
        }
        if overlap < best_overlap {
            best_overlap = overlap;
            best_axis = u;
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
) -> Option<Contact> {
    let (normal, penetration) = obb_sat(pos_a, half_a, rot_a, pos_b, half_b, rot_b)?;
    Some(Contact {
        normal,
        penetration,
        contact_point: (pos_a + pos_b) * 0.5,
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

/// Multi-point manifold for OBB-OBB: keeps up to 4 vertices that lie inside the
/// opposing box's face (vertex-face contacts), falling back to a single point.
#[allow(clippy::too_many_arguments)]
fn box_manifold(
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
) -> Option<Manifold> {
    let (n, _pen) = obb_sat(pos_a, half_a, rot_a, pos_b, half_b, rot_b)?;

    let aa = [rot_a * Vec3::X, rot_a * Vec3::Y, rot_a * Vec3::Z];
    let ba = [rot_b * Vec3::X, rot_b * Vec3::Y, rot_b * Vec3::Z];
    // Half-width of each box projected onto the contact normal.
    let hwn_a = half_a.x * aa[0].dot(n).abs()
        + half_a.y * aa[1].dot(n).abs()
        + half_a.z * aa[2].dot(n).abs();
    let hwn_b = half_b.x * ba[0].dot(n).abs()
        + half_b.y * ba[1].dot(n).abs()
        + half_b.z * ba[2].dot(n).abs();

    let mut cand: Vec<(Vec3, f32)> = Vec::new();
    // B's corners inside A's face.
    let eps = 1e-3;
    for c in obb_corners(pos_b, half_b, rot_b) {
        let local = rot_a.inverse() * (c - pos_a);
        if local.x.abs() <= half_a.x + eps
            && local.y.abs() <= half_a.y + eps
            && local.z.abs() <= half_a.z + eps
        {
            let d = hwn_a - (c - pos_a).dot(n);
            if d > -eps {
                cand.push((c, d));
            }
        }
    }
    // A's corners inside B's box.
    for c in obb_corners(pos_a, half_a, rot_a) {
        let local = rot_b.inverse() * (c - pos_b);
        if local.x.abs() <= half_b.x + eps
            && local.y.abs() <= half_b.y + eps
            && local.z.abs() <= half_b.z + eps
        {
            let d = hwn_b - (c - pos_b).dot(-n);
            if d > -eps {
                cand.push((c, d));
            }
        }
    }

    // Deduplicate near-identical points, sort by depth, keep up to 4.
    let mut uniq: Vec<(Vec3, f32)> = Vec::new();
    for (p, d) in cand {
        if uniq.iter().any(|&(q, _)| (p - q).length() < 1e-3) {
            continue;
        }
        uniq.push((p, d));
    }
    uniq.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut points = [ManifoldPoint {
        world_point: Vec3::ZERO,
        penetration: 0.0,
    }; 4];
    let mut count = 0;
    for (p, d) in uniq.into_iter().take(4) {
        points[count] = ManifoldPoint {
            world_point: p,
            penetration: d.max(0.0),
        };
        count += 1;
    }

    if count == 0 {
        return box_vs_box(pos_a, half_a, rot_a, pos_b, half_b, rot_b)
            .map(|c| Manifold::single(0, 0, c));
    }
    Some(Manifold {
        body_a: 0,
        body_b: 0,
        normal: n,
        point_count: count,
        points,
    })
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
) -> Option<Contact> {
    let axis = cap_rot * Vec3::Y;
    let bottom = cap_pos - axis * cap_half_height;
    let seg = axis * (2.0 * cap_half_height);
    let t = (sphere_pos - bottom).dot(seg) / seg.length_squared();
    let t = t.clamp(0.0, 1.0);
    let closest = bottom + seg * t;
    let to_sphere = sphere_pos - closest;
    let d = to_sphere.length();
    let rr = cap_radius + sphere_radius;
    if d >= rr || d < 1e-10 {
        return None;
    }
    // Normal points from the capsule toward the sphere.
    let n = to_sphere / d;
    let penetration = rr - d;
    let contact_point = closest + n * (cap_radius - penetration * 0.5);
    Some(Contact {
        normal: n,
        penetration,
        contact_point,
    })
}

/// Capsule-capsule: both segment axes are rotated by the body orientation.
#[allow(clippy::too_many_arguments)]
fn capsule_vs_capsule(
    pos_a: Vec3,
    radius_a: f32,
    half_height_a: f32,
    rot_a: Quat,
    pos_b: Vec3,
    radius_b: f32,
    half_height_b: f32,
    rot_b: Quat,
) -> Option<Contact> {
    let ax = rot_a * Vec3::Y;
    let bx = rot_b * Vec3::Y;
    let bot_a = pos_a - ax * half_height_a;
    let bot_b = pos_b - bx * half_height_b;

    let seg_a = ax * (2.0 * half_height_a);
    let seg_b = bx * (2.0 * half_height_b);
    let diff = bot_b - bot_a;
    let a = seg_a.dot(seg_a);
    let b = seg_a.dot(seg_b);
    let c = seg_b.dot(seg_b);
    let d = seg_a.dot(diff);
    let e = seg_b.dot(diff);
    let det = a * c - b * b;

    let (t_a, t_b) = if det.abs() < 1e-10 {
        (0.0, if c > 0.0 { e / c } else { 0.0 })
    } else {
        ((b * e - c * d) / det, (a * e - b * d) / det)
    };
    let t_a = clamp01(t_a);
    let t_b = clamp01(t_b);

    let closest_a = bot_a + seg_a * t_a;
    let closest_b = bot_b + seg_b * t_b;
    let diff2 = closest_b - closest_a;
    let dist_sq = diff2.length_squared();
    let radius_sum = radius_a + radius_b;
    if dist_sq > radius_sum * radius_sum || dist_sq < 1e-10 {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = diff2 / dist;
    let penetration = radius_sum - dist;
    Some(Contact {
        normal,
        penetration,
        contact_point: (closest_a + closest_b) * 0.5,
    })
}

fn detect_collisions(bodies: &[RigidBody], active: &[(usize, usize)]) -> Vec<Manifold> {
    let mut manifolds = Vec::new();
    for &(i, j) in active {
        let a = &bodies[i];
        let b = &bodies[j];
        if a.body_type == BodyType::Static && b.body_type == BodyType::Static {
            continue;
        }

        let manifold = match (&a.shape, &b.shape) {
            (&Shape::Sphere { radius: ra }, &Shape::Sphere { radius: rb }) => {
                sphere_vs_sphere(a.position, ra, b.position, rb).map(|c| Manifold::single(i, j, c))
            }
            (&Shape::Sphere { radius: ra }, &Shape::Box { half_extents: hb }) => {
                sphere_vs_obb(a.position, ra, b.position, hb, b.orientation)
                    .map(|c| Manifold::single(i, j, c))
            }
            (&Shape::Box { half_extents: ha }, &Shape::Sphere { radius: rb }) => {
                sphere_vs_obb(b.position, rb, a.position, ha, a.orientation).map(|c| {
                    Manifold::single(
                        i,
                        j,
                        Contact {
                            normal: -c.normal,
                            penetration: c.penetration,
                            contact_point: c.contact_point,
                        },
                    )
                })
            }
            (&Shape::Box { half_extents: ha }, &Shape::Box { half_extents: hb }) => {
                box_manifold(a.position, ha, a.orientation, b.position, hb, b.orientation)
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
                a.position,
                ra,
                ha,
                a.orientation,
                b.position,
                rb,
                hb,
                b.orientation,
            )
            .map(|c| Manifold::single(i, j, c)),
            (
                &Shape::Sphere { radius: r },
                &Shape::Capsule {
                    radius: cr,
                    half_height: hh,
                },
            ) => sphere_vs_capsule(a.position, r, b.position, cr, hh, b.orientation)
                .map(|c| Manifold::single(i, j, c)),
            (
                &Shape::Capsule {
                    radius: cr,
                    half_height: hh,
                },
                &Shape::Sphere { radius: r },
            ) => sphere_vs_capsule(b.position, r, a.position, cr, hh, a.orientation).map(|c| {
                Manifold::single(
                    i,
                    j,
                    Contact {
                        normal: -c.normal,
                        penetration: c.penetration,
                        contact_point: c.contact_point,
                    },
                )
            }),
            _ => None,
        };

        if let Some(m) = manifold {
            manifolds.push(m);
        }
    }
    manifolds
}

struct SweepAndPrune {
    aabbs: Vec<AABB>,
    active: Vec<(usize, usize)>,
    sort_axis: usize,
}

impl SweepAndPrune {
    fn new() -> Self {
        Self {
            aabbs: Vec::new(),
            active: Vec::new(),
            sort_axis: 0,
        }
    }

    fn update(&mut self, bodies: &[RigidBody]) {
        self.aabbs = bodies.iter().map(compute_aabb).collect();
        self.sort_axis = (self.sort_axis + 1) % 3;
        self.active.clear();

        let n = self.aabbs.len();
        let mut starts: Vec<(f32, usize)> = self
            .aabbs
            .iter()
            .enumerate()
            .map(|(i, aabb)| match self.sort_axis {
                0 => (aabb.min.x, i),
                1 => (aabb.min.y, i),
                _ => (aabb.min.z, i),
            })
            .collect();
        starts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for sweep_pos in 0..n {
            let i = starts[sweep_pos].1;
            let sweep_aabb = &self.aabbs[i];
            let end = match self.sort_axis {
                0 => sweep_aabb.max.x,
                1 => sweep_aabb.max.y,
                _ => sweep_aabb.max.z,
            };
            for &(_pos, j) in &starts[(sweep_pos + 1)..] {
                let start_j = match self.sort_axis {
                    0 => self.aabbs[j].min.x,
                    1 => self.aabbs[j].min.y,
                    _ => self.aabbs[j].min.z,
                };
                if start_j > end {
                    break;
                }
                if i < j && sweep_aabb.overlaps(&self.aabbs[j]) {
                    self.active.push((i, j));
                }
            }
        }
    }
}

pub struct BuiltinPhysicsEngine {
    bodies: Vec<RigidBody>,
    broadphase: SweepAndPrune,
    gravity: Vec3,
    substeps: u32,
    /// Accumulated normal impulses per point, keyed by sorted body pair (warm start).
    warm_impulses: HashMap<(usize, usize), [f32; 4]>,
}

impl BuiltinPhysicsEngine {
    pub fn new(gravity: Vec3) -> Self {
        Self {
            bodies: Vec::new(),
            broadphase: SweepAndPrune::new(),
            gravity,
            substeps: 4,
            warm_impulses: HashMap::new(),
        }
    }

    pub fn set_substeps(&mut self, n: u32) {
        self.substeps = n;
    }

    fn integrate(&mut self, dt: f32) {
        for body in &mut self.bodies {
            if body.body_type != BodyType::Dynamic {
                continue;
            }
            // Linear integrate (semi-implicit).
            body.velocity += self.gravity * dt;
            body.position += body.velocity * dt;

            // Angular integrate from applied torque.
            if body.torque != Vec3::ZERO {
                body.angular_velocity +=
                    mul_inv_inertia(body.inertia, body.orientation, body.torque * dt);
                body.torque = Vec3::ZERO;
            }

            // Rotation: exact small-step quaternion update (exp of angular velocity * dt).
            if body.angular_velocity != Vec3::ZERO {
                let dwq = Quat::from_scaled_axis(body.angular_velocity * dt);
                body.orientation = (dwq * body.orientation).normalize();
            }
        }
    }

    fn resolve_manifolds(&mut self, manifolds: &[Manifold]) {
        let mut next: HashMap<(usize, usize), [f32; 4]> = HashMap::new();
        for m in manifolds {
            let (i, j) = (m.body_a, m.body_b);
            let inv_mass_a = self.bodies[i].inv_mass;
            let inv_mass_b = self.bodies[j].inv_mass;
            let total_inv = inv_mass_a + inv_mass_b;
            if total_inv < 1e-10 {
                continue;
            }
            let n = m.normal;
            let key = (i.min(j), i.max(j));
            let warm = self.warm_impulses.get(&key).copied().unwrap_or([0.0; 4]);
            let mut acc = warm;

            // ---- Positional correction per point (incl. rotation) ----
            for k in 0..m.point_count {
                let p = m.points[k].world_point;
                let ra = p - self.bodies[i].position;
                let rb = p - self.bodies[j].position;
                let ia = self.bodies[i].inertia;
                let ib = self.bodies[j].inertia;
                let oa = self.bodies[i].orientation;
                let ob = self.bodies[j].orientation;
                let ra_n = ra.cross(n);
                let rb_n = rb.cross(n);
                let k_pos = total_inv
                    + ra_n.dot(mul_inv_inertia(ia, oa, ra_n))
                    + rb_n.dot(mul_inv_inertia(ib, ob, rb_n));
                if k_pos > 1e-10 {
                    let c = (m.points[k].penetration - 0.02).clamp(0.0, 0.25);
                    let lam = 0.5 * c / k_pos;
                    let jp = n * lam;
                    self.bodies[i].position -= jp * inv_mass_a;
                    self.bodies[j].position += jp * inv_mass_b;
                    let da = mul_inv_inertia(ia, oa, ra.cross(-jp));
                    let db = mul_inv_inertia(ib, ob, rb.cross(jp));
                    if da != Vec3::ZERO {
                        self.bodies[i].orientation =
                            (Quat::from_scaled_axis(da) * self.bodies[i].orientation).normalize();
                    }
                    if db != Vec3::ZERO {
                        self.bodies[j].orientation =
                            (Quat::from_scaled_axis(db) * self.bodies[j].orientation).normalize();
                    }
                }
            }

            // ---- Velocity solve per point with warm starting ----
            for k in 0..m.point_count {
                let p = m.points[k].world_point;
                let ra = p - self.bodies[i].position;
                let rb = p - self.bodies[j].position;
                let ia = self.bodies[i].inertia;
                let ib = self.bodies[j].inertia;
                let oa = self.bodies[i].orientation;
                let ob = self.bodies[j].orientation;
                let ra_n = ra.cross(n);
                let rb_n = rb.cross(n);
                let k_eff = total_inv
                    + ra_n.dot(mul_inv_inertia(ia, oa, ra_n))
                    + rb_n.dot(mul_inv_inertia(ib, ob, rb_n));
                if k_eff < 1e-10 {
                    continue;
                }
                let va = self.bodies[i].velocity + self.bodies[i].angular_velocity.cross(ra);
                let vb = self.bodies[j].velocity + self.bodies[j].angular_velocity.cross(rb);
                let rel = vb - va;
                let vn = rel.dot(n);
                if vn > 0.0 {
                    // Separating: keep the warm impulse, don't add.
                    acc[k] = warm[k].max(0.0);
                    continue;
                }
                let e = self.bodies[i].restitution.min(self.bodies[j].restitution);
                let j_impulse = -(1.0 + e) * vn / k_eff;
                // Warm start: accumulate from the cached impulse, clamp >= 0,
                // apply only the delta.
                let new_impulse = (warm[k] + j_impulse).max(0.0);
                let delta = new_impulse - warm[k];
                acc[k] = new_impulse;
                if delta.abs() < 1e-9 {
                    continue;
                }
                let imp = n * delta;
                self.bodies[i].velocity -= imp * inv_mass_a;
                self.bodies[j].velocity += imp * inv_mass_b;
                self.bodies[i].angular_velocity -= mul_inv_inertia(ia, oa, ra.cross(imp));
                self.bodies[j].angular_velocity += mul_inv_inertia(ib, ob, rb.cross(imp));

                // ---- Friction (Coulomb), bound by the accumulated normal impulse ----
                let tangent = rel - n * vn;
                let tangent_len = tangent.length();
                if tangent_len > 1e-6 {
                    let tangent_dir = tangent / tangent_len;
                    let mu = self.bodies[i].friction.max(self.bodies[j].friction);
                    let max_friction = new_impulse * mu;
                    let f_impulse = (-tangent_len / total_inv).min(max_friction);
                    let jt = tangent_dir * f_impulse;
                    self.bodies[i].velocity -= jt * inv_mass_a;
                    self.bodies[j].velocity += jt * inv_mass_b;
                    self.bodies[i].angular_velocity -= mul_inv_inertia(ia, oa, ra.cross(jt));
                    self.bodies[j].angular_velocity += mul_inv_inertia(ib, ob, rb.cross(jt));
                }
            }
            next.insert(key, acc);
        }
        self.warm_impulses = next;
    }

    fn raycast_body(&self, ray: &Ray, handle: usize, max_dist: f32) -> Option<RaycastHit> {
        let body = &self.bodies[handle];
        let aabb = body.shape.aabb(body.position, body.orientation);
        let inv_dir = Vec3::new(
            1.0 / ray.direction.x,
            1.0 / ray.direction.y,
            1.0 / ray.direction.z,
        );

        let t1 = (aabb.min - ray.origin) * inv_dir;
        let t2 = (aabb.max - ray.origin) * inv_dir;
        let t_min = t1.min(t2);
        let t_max = t1.max(t2);
        let enter = t_min.x.max(t_min.y.max(t_min.z));
        let exit = t_max.x.min(t_max.y.min(t_max.z));

        if enter > exit || exit < 0.0 || enter > max_dist {
            return None;
        }

        let t = enter.max(0.0);
        let point = ray.point_at(t);
        // Approximate surface normal pointing away from the shape centre.
        let normal = (point - body.position).normalize_or_zero();
        if normal.length_squared() < 0.5 {
            return None;
        }
        Some(RaycastHit {
            handle,
            point,
            normal,
            distance: t,
        })
    }
}

impl PhysicsEngine for BuiltinPhysicsEngine {
    fn step(&mut self, dt: f32) {
        let sub_dt = dt / self.substeps as f32;
        for _ in 0..self.substeps {
            self.integrate(sub_dt);
            self.broadphase.update(&self.bodies);
            let manifolds = detect_collisions(&self.bodies, &self.broadphase.active);
            self.resolve_manifolds(&manifolds);
        }
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        let handle = self.bodies.len();
        self.bodies.push(body);
        handle
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        if handle < self.bodies.len() {
            self.bodies.swap_remove(handle);
        }
    }

    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        self.bodies.get(handle)
    }

    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        self.bodies.get_mut(handle)
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit> {
        let mut closest: Option<RaycastHit> = None;
        for handle in 0..self.bodies.len() {
            if let Some(hit) = self.raycast_body(&ray, handle, max_dist) {
                match &closest {
                    Some(best) if hit.distance < best.distance => closest = Some(hit),
                    None => closest = Some(hit),
                    _ => {}
                }
            }
        }
        closest
    }

    fn shapecast(&self, shape: &Shape, from: Vec3, to: Vec3) -> Option<RaycastHit> {
        // Conservative sweep by reusing the narrow-phase over a probe body.
        let delta = to - from;
        if delta.length_squared() < 1e-9 {
            return None;
        }
        const STEPS: usize = 64;
        for s in 1..=STEPS {
            let pos = from + delta * (s as f32 / STEPS as f32);
            let probe = RigidBody {
                position: pos,
                orientation: Quat::IDENTITY,
                velocity: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
                mass: 1.0,
                inv_mass: 1.0,
                inertia: shape.inertia(1.0),
                torque: Vec3::ZERO,
                restitution: 0.0,
                friction: 0.0,
                shape: shape.clone(),
                body_type: BodyType::Dynamic,
            };
            for (handle, body) in self.bodies.iter().enumerate() {
                let pair = [probe.clone(), body.clone()];
                if !detect_collisions(&pair, &[(0, 1)]).is_empty() {
                    return Some(RaycastHit {
                        handle,
                        point: pos,
                        normal: Vec3::ZERO,
                        distance: (pos - from).length(),
                    });
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_falls() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let sphere = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 10.0, 0.0), 1.0, 1.0));
        physics.step(1.0 / 60.0);
        let body = physics.get_body(sphere).unwrap();
        assert!(body.position.y < 10.0);
    }

    #[test]
    fn static_body_does_not_fall() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let ground = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(10.0, 1.0, 10.0),
            0.0,
        ));
        physics.step(1.0 / 60.0);
        let body = physics.get_body(ground).unwrap();
        assert_eq!(body.position.y, -1.0);
    }

    #[test]
    fn sphere_vs_sphere_collision() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let a = physics.add_body(RigidBody::new_sphere(Vec3::new(-0.4, 0.0, 0.0), 0.5, 1.0));
        let b = physics.add_body(RigidBody::new_sphere(Vec3::new(0.4, 0.0, 0.0), 0.5, 1.0));
        physics.step(1.0 / 60.0);
        let body_a = physics.get_body(a).unwrap();
        let body_b = physics.get_body(b).unwrap();
        let dist = (body_a.position - body_b.position).length();
        assert!(dist < 1.1);
    }

    #[test]
    fn raycast_hits_sphere() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.0, -5.0), 1.0, 1.0));
        let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        let hit = physics.raycast(ray, 10.0);
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert!((hit.distance - 4.0).abs() < 0.01);
    }

    #[test]
    fn box_vs_box_collision() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let a = physics.add_body(RigidBody::new_box(
            Vec3::new(-0.4, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            1.0,
        ));
        let b = physics.add_body(RigidBody::new_box(
            Vec3::new(0.4, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            1.0,
        ));
        physics.step(1.0 / 60.0);
        let body_a = physics.get_body(a).unwrap();
        let body_b = physics.get_body(b).unwrap();
        let dist = (body_a.position - body_b.position).length();
        assert!(dist < 1.1);
    }

    // ---- G1: orientation + angular dynamics ----

    #[test]
    fn angular_velocity_rotates_body() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let handle = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0));
        physics
            .get_body_mut(handle)
            .unwrap()
            .set_angular_velocity(Vec3::new(0.0, 2.0, 0.0));
        physics.step(1.0 / 60.0);
        let body = physics.get_body(handle).unwrap();
        // Orientation must have changed and remain a unit quaternion.
        assert!(
            body.orientation.to_axis_angle().1.abs() > 1e-4,
            "should have rotated about Y"
        );
        assert!(
            (body.orientation.length() - 1.0).abs() < 1e-4,
            "unit quaternion preserved"
        );
    }

    #[test]
    fn torque_turns_body() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let sphere = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0));
        // Apply torque around Z -> angular velocity must appear.
        let w_after = {
            physics
                .get_body_mut(sphere)
                .unwrap()
                .apply_torque(Vec3::new(0.0, 0.0, 1.0));
            physics.step(1.0 / 60.0);
            physics.get_body(sphere).unwrap().angular_velocity
        };
        assert!(
            w_after.z.abs() > 1e-5,
            "torque must produce angular velocity, got {w_after:?}"
        );
    }

    #[test]
    fn oriented_boxes_collide() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        // Same center, rotated 45° about Y, box-ish units: OBB-OBB should separate.
        let half = Vec3::new(0.5, 0.5, 0.5);
        let a = physics.add_body(
            RigidBody::new_box(Vec3::new(0.0, 0.0, 0.0), half, 1.0)
                .with_orientation(Quat::from_rotation_z(0.0)),
        );
        let b = physics.add_body(
            RigidBody::new_box(Vec3::new(0.4, 0.0, 0.0), half, 1.0)
                .with_orientation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        );
        physics.step(1.0 / 60.0);
        let body_a = physics.get_body(a).unwrap();
        let body_b = physics.get_body(b).unwrap();
        // Resting separation for two half-0.5 cubes is exactly 1.0 (touching).
        let dist = (body_a.position - body_b.position).length();
        assert!(dist <= 1.05, "oriented boxes should resolve, dist={dist}");
    }

    #[test]
    fn obb_aabb_respects_rotation() {
        let half = Vec3::new(1.0, 1.0, 1.0);
        let q = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let aabb = Shape::Box { half_extents: half }.aabb(Vec3::ZERO, q);
        // The rotated corner (1,1,1) -> (0, 1.414, 1) must be inside the bounding AABB.
        let corner = q * Vec3::splat(1.0);
        assert!(
            aabb.contains_point(corner),
            "corner {corner:?} not inside {aabb:?}"
        );
        // Y half-extent grows to sqrt(2) after the 45° rotation.
        let half_y = (aabb.max.y - aabb.min.y) * 0.5;
        assert!(
            (half_y - 2f32.sqrt()).abs() < 1e-3,
            "OBB->AABB y-extent, got {half_y}"
        );
    }

    #[test]
    fn sphere_capsule_collision() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        let sphere = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.0, 0.0), 0.5, 1.0));
        let capsule = physics.add_body(
            RigidBody::new_capsule(Vec3::new(0.6, 0.0, 0.0), 0.5, 1.0, 1.0)
                .with_orientation(glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        );
        physics.step(1.0 / 60.0);
        let d = (physics.get_body(sphere).unwrap().position
            - physics.get_body(capsule).unwrap().position)
            .length();
        // Sphere radius 0.5 + capsule radius 0.5 -> resting center distance ~1.0.
        assert!(d <= 1.05, "sphere/capsule should resolve on contact, d={d}");
    }

    #[test]
    fn shapecast_hits_body() {
        let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
        physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.0, -5.0), 1.0, 0.0));
        // Cast a small sphere from origin toward the static target.
        let shape = Shape::Sphere { radius: 0.1 };
        let hit = physics.shapecast(&shape, Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0));
        assert!(hit.is_some(), "conservative shapecast should hit");
        let hit = hit.unwrap();
        assert_eq!(hit.handle, 0);
        assert!(
            hit.distance > 3.0 && hit.distance < 10.0,
            "hit distance={}",
            hit.distance
        );
    }

    #[test]
    fn box_manifold_produces_four_points() {
        // Two equal half-0.5 boxes, overlapping by 0.25 along +Y: the resting
        // face yields 4 manifold points (vertex-face contact), not one.
        let half = Vec3::new(0.5, 0.5, 0.5);
        let m = box_manifold(
            Vec3::new(0.0, 0.0, 0.0),
            half,
            Quat::IDENTITY,
            Vec3::new(0.0, 0.75, 0.0),
            half,
            Quat::IDENTITY,
        )
        .expect("boxes overlap");
        assert_eq!(m.point_count, 4, "expected a 4-point manifold");
        assert!((m.normal - Vec3::Y).length() < 1e-3, "normal should be +Y");
        for k in 0..m.point_count {
            assert!(
                m.points[k].penetration > 0.0,
                "point {k} has positive penetration"
            );
        }
    }
}
