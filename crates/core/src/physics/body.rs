use glam::Vec3;

use super::shape::Shape;

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
    pub velocity: Vec3,
    pub mass: f32,
    pub inv_mass: f32,
    pub restitution: f32,
    pub friction: f32,
    pub shape: Shape,
    pub body_type: BodyType,
}

impl RigidBody {
    pub fn new_sphere(position: Vec3, radius: f32, mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            velocity: Vec3::ZERO,
            mass,
            inv_mass,
            restitution: 0.5,
            friction: 0.3,
            shape: Shape::Sphere { radius },
            body_type: if mass > 0.0 {
                BodyType::Dynamic
            } else {
                BodyType::Static
            },
        }
    }

    pub fn new_box(position: Vec3, half_extents: Vec3, mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            velocity: Vec3::ZERO,
            mass,
            inv_mass,
            restitution: 0.3,
            friction: 0.5,
            shape: Shape::Box { half_extents },
            body_type: if mass > 0.0 {
                BodyType::Dynamic
            } else {
                BodyType::Static
            },
        }
    }

    pub fn new_capsule(position: Vec3, radius: f32, half_height: f32, mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            velocity: Vec3::ZERO,
            mass,
            inv_mass,
            restitution: 0.4,
            friction: 0.4,
            shape: Shape::Capsule {
                radius,
                half_height,
            },
            body_type: if mass > 0.0 {
                BodyType::Dynamic
            } else {
                BodyType::Static
            },
        }
    }
}
