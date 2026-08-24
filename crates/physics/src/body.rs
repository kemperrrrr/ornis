use glam::{Quat, Vec3};

use crate::shape::Shape;

pub type BodyHandle = usize;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BodyType {
    Static,
    Dynamic,
    Kinematic,
}

#[derive(Debug, Clone)]
pub struct RigidBody {
    pub position: Vec3,
    /// Rotation of the body, stored as a unit quaternion.
    pub orientation: Quat,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub mass: f32,
    pub inv_mass: f32,
    /// Diagonal (body-frame) inertia tensor.
    pub inertia: Vec3,
    pub torque: Vec3,
    pub restitution: f32,
    pub friction: f32,
    pub shape: Shape,
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
            body_type: if mass > 0.0 {
                BodyType::Dynamic
            } else {
                BodyType::Static
            },
        }
    }

    pub fn new_sphere(position: Vec3, radius: f32, mass: f32) -> Self {
        Self::build(position, mass, 0.5, 0.3, Shape::Sphere { radius })
    }

    pub fn new_box(position: Vec3, half_extents: Vec3, mass: f32) -> Self {
        Self::build(position, mass, 0.3, 0.5, Shape::Box { half_extents })
    }

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

    pub fn with_orientation(mut self, orientation: Quat) -> Self {
        self.orientation = orientation;
        self
    }

    pub fn set_orientation(&mut self, orientation: Quat) {
        self.orientation = orientation;
    }

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
