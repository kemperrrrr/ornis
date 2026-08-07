use glam::{Quat, Vec3};

use crate::math::AABB;

#[derive(Debug, Clone)]
pub enum Shape {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Capsule { radius: f32, half_height: f32 },
}

impl Shape {
    /// World-space AABB of the shape at `position` with `orientation`.
    pub fn aabb(&self, position: Vec3, orientation: Quat) -> AABB {
        match self {
            Shape::Sphere { radius } => {
                let r = Vec3::splat(*radius);
                AABB::new(position - r, position + r)
            }
            Shape::Box { half_extents } => {
                // OBB -> AABB: axis-aligned extent is the rotated |half_extents|.
                let r = orientation.mul_vec3(*half_extents).abs();
                AABB::new(position - r, position + r)
            }
            Shape::Capsule {
                radius,
                half_height,
            } => {
                // Local +Y axis, rotated by orientation; plus an isotropic radius shell.
                let axis = orientation * Vec3::Y;
                let e = axis.abs() * *half_height + Vec3::splat(*radius);
                AABB::new(position - e, position + e)
            }
        }
    }

    /// Diagonal (body-frame) inertia tensor for a given mass.
    /// One entry per principal axis; a sphere is isotropic, a box and a
    /// capsule are symmetric about their local +Y.
    pub fn inertia(&self, mass: f32) -> Vec3 {
        match self {
            Shape::Sphere { radius } => {
                let i = 0.4 * mass * radius * radius;
                Vec3::splat(i)
            }
            Shape::Box { half_extents } => {
                let (x, y, z) = (
                    right2(half_extents.x),
                    right2(half_extents.y),
                    right2(half_extents.z),
                );
                Vec3::new(
                    (mass / 12.0) * (y + z),
                    (mass / 12.0) * (z + x),
                    (mass / 12.0) * (x + y),
                )
            }
            Shape::Capsule {
                radius,
                half_height,
            } => {
                // h = total half-length along the axis (excluding radius), like a cylinder.
                let h = half_height;
                let r = radius;
                // Uniform about the axis:
                let i_y = 0.5 * mass * r * r;
                // Perpendicular, approximating a cylinder + sphere caps.
                let i_xz = 0.25 * mass * (r * r) + (mass / 3.0) * h * r + 0.25 * mass * h * h;
                Vec3::new(i_xz, i_y, i_xz)
            }
        }
    }
}

#[inline]
fn right2(v: f32) -> f32 {
    v * v
}
