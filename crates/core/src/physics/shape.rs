use glam::Vec3;

use super::math::AABB;

#[derive(Debug, Clone)]
pub enum Shape {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Capsule { radius: f32, half_height: f32 },
}

impl Shape {
    pub fn aabb(&self, position: Vec3) -> AABB {
        match self {
            Shape::Sphere { radius } => AABB::new(
                position - Vec3::splat(*radius),
                position + Vec3::splat(*radius),
            ),
            Shape::Box { half_extents } => {
                AABB::new(position - *half_extents, position + *half_extents)
            }
            Shape::Capsule {
                radius,
                half_height,
            } => {
                let half = Vec3::new(*radius, *half_height + *radius, *radius);
                AABB::new(position - half, position + half)
            }
        }
    }
}
