use glam::{Quat, Vec3};

use crate::shape::Shape;

/// Stable index of a body inside its owning [`BuiltinPhysicsEngine`](crate::engine::BuiltinPhysicsEngine).
///
/// Handles stay valid until the body is explicitly removed; removal shifts
/// subsequent handles because this is a plain vector index, so callers should
/// not cache handles across removals.
pub type BodyHandle = usize;

/// How a body participates in simulation.
///
/// Determines which solver terms apply: static bodies have zero inverse mass
/// and never integrate, kinematic bodies integrate velocity but push dynamic
/// bodies with infinite effective mass, dynamic bodies respond to contacts
/// and gravity in full.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BodyType {
    /// Never moves (level geometry); treated as having infinite mass.
    Static,
    /// Fully simulated under forces, gravity, contacts and joints.
    Dynamic,
    /// Moved by setting [`RigidBody::velocity`] directly; collides with and
    /// pushes dynamic bodies but is unaffected by them.
    Kinematic,
}

/// A single rigid body: pose, motion state, material properties and shape.
///
/// Mass properties are derived from [`Shape::inertia`] at construction; when
/// mutating `mass` directly, keep `inv_mass` consistent (`1/mass` for
/// dynamics, `0` for statics) — the solver reads only the inverse quantities.
#[derive(Debug, Clone)]
pub struct RigidBody {
    /// World-space position of the body's center of mass.
    pub position: Vec3,
    /// Rotation of the body, stored as a unit quaternion.
    pub orientation: Quat,
    /// Linear velocity of the center of mass (m/s).
    pub velocity: Vec3,
    /// Angular velocity in world space (rad/s), about the center of mass.
    pub angular_velocity: Vec3,
    /// Total mass (kg). Zero mass means static/infinite-mass behavior.
    pub mass: f32,
    /// Cached `1 / mass` (0 for statics) — what the solver actually uses.
    pub inv_mass: f32,
    /// Diagonal (body-frame) inertia tensor.
    pub inertia: Vec3,
    /// Accumulated external torque (N·m), consumed and cleared each step.
    pub torque: Vec3,
    /// Coefficient of restitution in `[0, 1]`: how much normal velocity
    /// survives a bounce (0 = dead stop, 1 = perfectly elastic).
    pub restitution: f32,
    /// Coulomb friction coefficient ≥ 0 used by the contact solver.
    pub friction: f32,
    /// Collision primitive; also drives the derived inertia tensor.
    pub shape: Shape,
    /// Bit identifying the collision layer this body belongs to.
    ///
    /// A zero layer is valid and makes the body ineligible for all pairs;
    /// ordinary layers are represented by one or more set bits.
    pub collision_layer: u32,
    /// Bit mask of layers this body is allowed to collide with.
    ///
    /// A pair collides only when both bodies' masks include the other
    /// body's layer. The default is all layers, preserving pre-filter
    /// behavior.
    pub collision_mask: u32,
    /// Simulation role; derived from mass at construction, settable after.
    pub body_type: BodyType,
}

impl RigidBody {
    // Internal constructor: derives derived quantities from mass and shape.
    fn build(position: Vec3, mass: f32, restitution: f32, friction: f32, shape: Shape) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            orientation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass,
            inv_mass,
            inertia: shape.inertia(mass),
            torque: Vec3::ZERO,
            restitution,
            friction,
            shape,
            collision_layer: 1,
            collision_mask: u32::MAX,
            body_type: if mass > 0.0 {
                BodyType::Dynamic
            } else {
                BodyType::Static
            },
        }
    }

    /// Sphere body with default material (restitution 0.5, friction 0.3).
    /// Mass > 0 yields a dynamic body; mass 0 a static one.
    pub fn new_sphere(position: Vec3, radius: f32, mass: f32) -> Self {
        Self::build(position, mass, 0.5, 0.3, Shape::Sphere { radius })
    }

    /// Axis-aligned box body (half-extents per axis), restitution 0.3,
    /// friction 0.5.
    pub fn new_box(position: Vec3, half_extents: Vec3, mass: f32) -> Self {
        Self::build(position, mass, 0.3, 0.5, Shape::Box { half_extents })
    }

    /// Capsule body aligned to the local +Y axis (`half_height` is the
    /// cylinder half-length excluding the caps), restitution/friction 0.4.
    pub fn new_capsule(position: Vec3, radius: f32, half_height: f32, mass: f32) -> Self {
        Self::build(
            position,
            mass,
            0.4,
            0.4,
            Shape::Capsule {
                radius,
                half_height,
            },
        )
    }

    /// Builder-style collision-layer and mask configuration.
    ///
    /// A pair is eligible only when both directions agree: `self`'s mask
    /// contains `other`'s layer and vice versa. This keeps filtering
    /// symmetric even when a body uses a restrictive mask.
    pub fn with_collision_filter(mut self, layer: u32, mask: u32) -> Self {
        self.set_collision_filter(layer, mask);
        self
    }

    /// Changes the collision layer and mask in place.
    pub fn set_collision_filter(&mut self, layer: u32, mask: u32) {
        self.collision_layer = layer;
        self.collision_mask = mask;
    }

    /// Returns whether this body and `other` pass their mutual layer masks.
    pub fn can_collide_with(&self, other: &Self) -> bool {
        self.collision_mask & other.collision_layer != 0
            && other.collision_mask & self.collision_layer != 0
    }

    /// Builder-style variant of [`RigidBody::set_orientation`].
    pub fn with_orientation(mut self, orientation: Quat) -> Self {
        self.orientation = orientation;
        self
    }

    /// Overwrite the rotation; must remain a unit quaternion.
    pub fn set_orientation(&mut self, orientation: Quat) {
        self.orientation = orientation;
    }

    /// Directly set the world-space angular velocity (rad/s).
    pub fn set_angular_velocity(&mut self, w: Vec3) {
        self.angular_velocity = w;
    }

    /// Apply a torque (N·m) to the body; takes effect on the next step.
    pub fn apply_torque(&mut self, torque: Vec3) {
        self.torque += torque;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for a mutation-testing zombie (`cargo mutants`, night gate,
    /// 2026-08-24): `replace RigidBody::set_orientation with ()` survived —
    /// no test asserted that the setter actually mutates `orientation`.
    #[test]
    fn set_orientation_mutates_the_body() {
        let mut body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0);
        assert_eq!(body.orientation, Quat::IDENTITY);
        let target = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        body.set_orientation(target);
        assert_eq!(body.orientation, target);
    }

    /// `with_orientation` is the builder counterpart of `set_orientation`;
    /// same zombie risk, covered separately since it consumes/returns `self`
    /// rather than mutating in place.
    #[test]
    fn with_orientation_sets_the_body_on_construction() {
        let target = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0).with_orientation(target);
        assert_eq!(body.orientation, target);
    }

    /// `apply_torque` accumulates (`+=`); a mutant flipping it to `-=` or
    /// dropping the call entirely must be caught. Two calls in the same
    /// direction must sum, not overwrite or cancel.
    #[test]
    fn apply_torque_accumulates() {
        let mut body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0);
        assert_eq!(body.torque, Vec3::ZERO);
        body.apply_torque(Vec3::new(1.0, 0.0, 0.0));
        body.apply_torque(Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(body.torque, Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn collision_filter_defaults_to_all_layers() {
        let body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0);
        assert_eq!(body.collision_layer, 1);
        assert_eq!(body.collision_mask, u32::MAX);
        assert!(body.can_collide_with(&body));
    }

    #[test]
    fn collision_filter_requires_mutual_mask_match() {
        let a = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0).with_collision_filter(0b0001, 0b0010);
        let b = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0).with_collision_filter(0b0010, 0b0001);
        assert!(a.can_collide_with(&b));

        let blocked = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0)
            .with_collision_filter(0b0100, 0b0001);
        assert!(!a.can_collide_with(&blocked));
        assert!(!blocked.can_collide_with(&a));
    }

    #[test]
    fn collision_filter_setter_updates_layer_and_mask() {
        let mut body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0);
        body.set_collision_filter(0b1000, 0b0100);
        assert_eq!(body.collision_layer, 0b1000);
        assert_eq!(body.collision_mask, 0b0100);
    }

    /// `set_angular_velocity` is a straight setter with the same zombie
    /// shape as `set_orientation` — assert it actually takes effect.
    #[test]
    fn set_angular_velocity_mutates_the_body() {
        let mut body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0);
        assert_eq!(body.angular_velocity, Vec3::ZERO);
        let w = Vec3::new(0.0, 1.0, 2.0);
        body.set_angular_velocity(w);
        assert_eq!(body.angular_velocity, w);
    }

    /// `build`'s `inv_mass` derivation (`1.0 / mass`, guarded for statics):
    /// a mutant flipping `/` to `*` or dropping the `mass > 0.0` guard must
    /// be caught on both the dynamic and static branches.
    #[test]
    fn build_derives_inverse_mass_for_dynamic_bodies() {
        let body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 2.0);
        assert_eq!(body.mass, 2.0);
        assert!((body.inv_mass - 0.5).abs() < 1e-6, "inv_mass = 1/mass");
        assert_eq!(body.body_type, BodyType::Dynamic);
    }

    /// Static bodies (mass == 0) must have `inv_mass == 0`, not `1/0`
    /// (infinity) or a silently wrong non-zero value.
    #[test]
    fn build_static_body_has_zero_inverse_mass() {
        let body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 0.0);
        assert_eq!(body.inv_mass, 0.0);
        assert_eq!(body.body_type, BodyType::Static);
    }
}
