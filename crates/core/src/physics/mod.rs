pub mod math;
pub mod body;
pub mod shape;

use glam::Vec3;

use math::{AABB, Ray, RaycastHit};
use body::{BodyHandle, BodyType, RigidBody};
use shape::Shape;

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
    body_a: BodyHandle,
    body_b: BodyHandle,
    normal: Vec3,
    penetration: f32,
    contact_point: Vec3,
}

fn compute_aabb(body: &RigidBody) -> AABB {
    body.shape.aabb(body.position)
}

fn sphere_vs_sphere(
    pos_a: Vec3, radius_a: f32,
    pos_b: Vec3, radius_b: f32,
) -> Option<Contact> {
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
        body_a: 0,
        body_b: 0,
        normal,
        penetration,
        contact_point: pos_a + normal * (radius_a - penetration * 0.5),
    })
}

fn sphere_vs_box(
    sphere_pos: Vec3, sphere_radius: f32,
    box_pos: Vec3, half_extents: Vec3,
) -> Option<Contact> {
    let local = sphere_pos - box_pos;
    let clamped = Vec3::new(
        local.x.clamp(-half_extents.x, half_extents.x),
        local.y.clamp(-half_extents.y, half_extents.y),
        local.z.clamp(-half_extents.z, half_extents.z),
    );
    let closest = clamped + box_pos;
    let diff = sphere_pos - closest;
    let dist_sq = diff.length_squared();
    if dist_sq > sphere_radius * sphere_radius || dist_sq < 1e-10 {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = diff / dist;
    let penetration = sphere_radius - dist;
    Some(Contact {
        body_a: 0,
        body_b: 0,
        normal,
        penetration,
        contact_point: closest,
    })
}

fn box_vs_box(
    pos_a: Vec3, half_a: Vec3,
    pos_b: Vec3, half_b: Vec3,
) -> Option<Contact> {
    let diff = pos_b - pos_a;
    let overlap_x = half_a.x + half_b.x - diff.x.abs();
    if overlap_x <= 0.0 { return None; }
    let overlap_y = half_a.y + half_b.y - diff.y.abs();
    if overlap_y <= 0.0 { return None; }
    let overlap_z = half_a.z + half_b.z - diff.z.abs();
    if overlap_z <= 0.0 { return None; }

    let (penetration, normal) = if overlap_x < overlap_y && overlap_x < overlap_z {
        (overlap_x, Vec3::new(diff.x.signum(), 0.0, 0.0))
    } else if overlap_y < overlap_z {
        (overlap_y, Vec3::new(0.0, diff.y.signum(), 0.0))
    } else {
        (overlap_z, Vec3::new(0.0, 0.0, diff.z.signum()))
    };

    let contact_point = (pos_a + pos_b) * 0.5;
    Some(Contact {
        body_a: 0,
        body_b: 0,
        normal,
        penetration,
        contact_point,
    })
}

fn capsule_vs_capsule(
    pos_a: Vec3, radius_a: f32, half_height_a: f32,
    pos_b: Vec3, radius_b: f32, half_height_b: f32,
) -> Option<Contact> {
    let top_a = pos_a + Vec3::Y * half_height_a;
    let bot_a = pos_a - Vec3::Y * half_height_a;
    let top_b = pos_b + Vec3::Y * half_height_b;
    let bot_b = pos_b - Vec3::Y * half_height_b;

    let seg_a = top_a - bot_a;
    let seg_b = top_b - bot_b;
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
    let t_a = t_a.clamp(0.0, 1.0);
    let t_b = t_b.clamp(0.0, 1.0);

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
        body_a: 0,
        body_b: 0,
        normal,
        penetration,
        contact_point: (closest_a + closest_b) * 0.5,
    })
}

fn detect_collisions(bodies: &[RigidBody], active: &[(usize, usize)]) -> Vec<Contact> {
    let mut contacts = Vec::new();
    for &(i, j) in active {
        let a = &bodies[i];
        let b = &bodies[j];
        if a.body_type == BodyType::Static && b.body_type == BodyType::Static {
            continue;
        }

        let contact = match (&a.shape, &b.shape) {
            (&Shape::Sphere { radius: ra }, &Shape::Sphere { radius: rb }) =>
                sphere_vs_sphere(a.position, ra, b.position, rb),
            (&Shape::Sphere { radius: ra }, &Shape::Box { half_extents: hb }) =>
                sphere_vs_box(a.position, ra, b.position, hb),
            (&Shape::Box { half_extents: ha }, &Shape::Sphere { radius: rb }) =>
                sphere_vs_box(b.position, rb, a.position, ha).map(|c| Contact {
                    body_a: c.body_a, body_b: c.body_b, normal: -c.normal,
                    penetration: c.penetration, contact_point: c.contact_point,
                }),
            (&Shape::Box { half_extents: ha }, &Shape::Box { half_extents: hb }) =>
                box_vs_box(a.position, ha, b.position, hb),
            (&Shape::Capsule { radius: ra, half_height: hha },
             &Shape::Capsule { radius: rb, half_height: hhb }) =>
                capsule_vs_capsule(a.position, ra, hha, b.position, rb, hhb),
            _ => None,
        };

        if let Some(mut c) = contact {
            c.body_a = i;
            c.body_b = j;
            contacts.push(c);
        }
    }
    contacts
}

struct SweepAndPrune {
    aabbs: Vec<AABB>,
    active: Vec<(usize, usize)>,
    sort_axis: usize,
}

impl SweepAndPrune {
    fn new() -> Self {
        Self { aabbs: Vec::new(), active: Vec::new(), sort_axis: 0 }
    }

    fn update(&mut self, bodies: &[RigidBody]) {
        self.aabbs = bodies.iter().map(compute_aabb).collect();
        self.sort_axis = (self.sort_axis + 1) % 3;
        self.active.clear();

        let n = self.aabbs.len();
        let mut starts: Vec<(f32, usize)> = self.aabbs.iter().enumerate()
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
                if start_j > end { break; }
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
}

impl BuiltinPhysicsEngine {
    pub fn new(gravity: Vec3) -> Self {
        Self {
            bodies: Vec::new(),
            broadphase: SweepAndPrune::new(),
            gravity,
            substeps: 4,
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
            body.velocity += self.gravity * dt;
            body.position += body.velocity * dt;
        }
    }

    fn resolve_contacts(&mut self, contacts: &[Contact]) {
        for contact in contacts {
            let (i, j) = (contact.body_a, contact.body_b);
            let inv_mass_a = self.bodies[i].inv_mass;
            let inv_mass_b = self.bodies[j].inv_mass;
            let total_inv = inv_mass_a + inv_mass_b;
            if total_inv < 1e-10 { continue; }

            let correction = contact.normal * (contact.penetration / total_inv);
            self.bodies[i].position -= correction * inv_mass_a;
            self.bodies[j].position += correction * inv_mass_b;

            let rel_vel = self.bodies[j].velocity - self.bodies[i].velocity;
            let vel_along_normal = rel_vel.dot(contact.normal);
            if vel_along_normal > 0.0 { continue; }

            let e = self.bodies[i].restitution.min(self.bodies[j].restitution);
            let j_impulse = -(1.0 + e) * vel_along_normal / total_inv;
            let impulse = contact.normal * j_impulse;
            self.bodies[i].velocity -= impulse * inv_mass_a;
            self.bodies[j].velocity += impulse * inv_mass_b;

            let tangent = rel_vel - contact.normal * vel_along_normal;
            let tangent_len = tangent.length();
            if tangent_len > 1e-6 {
                let tangent_dir = tangent / tangent_len;
                let mu = self.bodies[i].friction.max(self.bodies[j].friction);
                let max_friction = j_impulse * mu;
                let friction_impulse = (-tangent_len / total_inv).min(max_friction);
                self.bodies[i].velocity -= tangent_dir * friction_impulse * inv_mass_a;
                self.bodies[j].velocity += tangent_dir * friction_impulse * inv_mass_b;
            }
        }
    }

    fn raycast_body(&self, ray: &Ray, handle: usize, max_dist: f32) -> Option<RaycastHit> {
        let body = &self.bodies[handle];
        let aabb = body.shape.aabb(body.position);
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
        let normal = (point - body.position).normalize_or_zero();
        if normal.length_squared() < 0.5 {
            return None;
        }
        Some(RaycastHit { handle, point, normal, distance: t })
    }
}

impl PhysicsEngine for BuiltinPhysicsEngine {
    fn step(&mut self, dt: f32) {
        let sub_dt = dt / self.substeps as f32;
        for _ in 0..self.substeps {
            self.integrate(sub_dt);
            self.broadphase.update(&self.bodies);
            let contacts = detect_collisions(&self.bodies, &self.broadphase.active);
            self.resolve_contacts(&contacts);
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

    fn shapecast(&self, _shape: &Shape, _from: Vec3, _to: Vec3) -> Option<RaycastHit> {
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
        let ground = physics.add_body(RigidBody::new_box(Vec3::new(0.0, -1.0, 0.0), Vec3::new(10.0, 1.0, 10.0), 0.0));
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
        let a = physics.add_body(RigidBody::new_box(Vec3::new(-0.4, 0.0, 0.0), Vec3::new(0.5, 0.5, 0.5), 1.0));
        let b = physics.add_body(RigidBody::new_box(Vec3::new(0.4, 0.0, 0.0), Vec3::new(0.5, 0.5, 0.5), 1.0));
        physics.step(1.0 / 60.0);
        let body_a = physics.get_body(a).unwrap();
        let body_b = physics.get_body(b).unwrap();
        let dist = (body_a.position - body_b.position).length();
        assert!(dist < 1.1);
    }
}
