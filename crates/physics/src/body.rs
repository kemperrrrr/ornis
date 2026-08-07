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
