use glam::{Mat4, Quat, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_identity() {
        let t = Transform::default();
        assert_eq!(t.position, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn from_position_keeps_identity_rotation_and_unit_scale() {
        let t = Transform::from_position(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn new_stores_components() {
        let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let t = Transform::new(Vec3::new(4.0, 5.0, 6.0), rotation, Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(t.position, Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(t.rotation, rotation);
        assert_eq!(t.scale, Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn identity_matrix_is_identity() {
        assert_eq!(Transform::default().matrix(), Mat4::IDENTITY);
    }

    #[test]
    fn matrix_round_trips_to_components() {
        let t = Transform::new(
            Vec3::new(1.0, -2.0, 3.5),
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
            Vec3::new(2.0, 0.5, 1.0),
        );
        let (scale, rotation, translation) = t.matrix().to_scale_rotation_translation();
        assert!(translation.abs_diff_eq(t.position, 1e-6));
        assert!(scale.abs_diff_eq(t.scale, 1e-6));
        // q and -q describe the same rotation; compare via transformed axis.
        let v = Vec3::X;
        assert!(
            (rotation * v).abs_diff_eq(t.rotation * v, 1e-6),
            "rotation mismatch"
        );
    }

    #[test]
    fn matrix_applies_scale_rotation_translation_in_order() {
        // 90° about Y maps +X to -Z; scale 2 doubles first, translation shifts last.
        let t = Transform::new(
            Vec3::new(10.0, 0.0, 0.0),
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            Vec3::splat(2.0),
        );
        let p = t.matrix().transform_point3(Vec3::X);
        assert!(p.abs_diff_eq(Vec3::new(10.0, 0.0, -2.0), 1e-5), "got {p}");
    }
}
