//! Convex collision primitives: sphere, box, capsule.
//!
//! All [`Shape`] variants are centered on the body origin (box and capsule
//! are symmetric about local +Y) and provide the two queries the pipeline
//! relies on: an exact world-space AABB projection for the broadphase and a
//! diagonal inertia tensor for the solver.
//!
//! The box↔capsule pair (G1 remainder) is now a first-class discrete contact
//! via `crate::distance::shape_distance` and `engine::box_vs_capsule`
//! (both `detect_collisions_into` paths, speculative `margin`, analytic TOI
//! through `distance::cast_shape`).

use glam::{Quat, Vec3};

use crate::math::AABB;

/// Convex collision primitives supported by the builtin engine.
///
/// All shapes are centered on the body origin; a box and a capsule are
/// symmetric about the body's local +Y axis. Every variant must provide an
/// AABB projection (broadphase) and a diagonal inertia tensor (solver).
#[derive(Debug, Clone)]
pub enum Shape {
    /// Uniform ball: rotation-invariant, isotropic inertia.
    Sphere {
        /// Distance from center to surface.
        radius: f32,
    },
    /// Oriented box (OBB) with half-extents along each local axis.
    Box {
        /// Half-size of the box along its local X/Y/Z axes.
        half_extents: Vec3,
    },
    /// Cylinder of `2 * half_height` along local +Y with hemispherical caps
    /// of `radius`; used for characters and rounded bars.
    Capsule {
        /// Radius of the cylinder and the spherical caps.
        radius: f32,
        /// Half-length of the cylindrical segment, excluding the caps.
        half_height: f32,
    },
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
                // OBB -> AABB: the world-axis half-extent along axis i is
                // sum_j |R_ij| * half_extents_j (Ericson, RTCD §4.2.6),
                // i.e. |R| @ half_extents with an ELEMENT-WISE absolute
                // value of the rotation matrix — NOT |R @ half_extents|.
                // The two only coincide at 0/90/180/270° rotations; at
                // e.g. 45° the naive `orientation.mul_vec3(..).abs()`
                // under-reports the AABB (measured: a unit-cube box
                // rotated 45° about Z needs a sqrt(2) half-extent on x
                // AND y, but the naive formula zeroes the x component).
                // An under-sized AABB can miss broadphase pairs entirely.
                let basis_x = (orientation * Vec3::X).abs() * half_extents.x;
                let basis_y = (orientation * Vec3::Y).abs() * half_extents.y;
                let basis_z = (orientation * Vec3::Z).abs() * half_extents.z;
                let r = basis_x + basis_y + basis_z;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `Shape` has no unit tests at all (night gate, 2026-08-24: 50 missed
    /// mutants in `Shape::inertia` alone — every arithmetic op and shape
    /// arm was free to mutate). These are golden reference values computed
    /// independently from the closed-form formulas, not "run the same code
    /// with a tolerance" — they catch `*`<->`/`, `+`<->`-`, coefficient
    /// swaps (0.4/0.5/0.25/(1/12)/(1/3)) and axis-mismatch mutations.
    const EPS: f32 = 1e-5;

    fn assert_vec3_close(got: Vec3, want: Vec3) {
        assert!((got - want).length() < EPS, "got {got:?}, want {want:?}");
    }

    #[test]
    fn sphere_inertia_is_isotropic_and_matches_formula() {
        // I = (2/5) m r^2, same on all three axes.
        let shape = Shape::Sphere { radius: 3.0 };
        let i = shape.inertia(2.0);
        assert_vec3_close(i, Vec3::splat(7.2));
    }

    #[test]
    fn box_inertia_matches_closed_form_per_axis() {
        // I_x = (m/12)(h_y^2 + h_z^2), and cyclic; asymmetric half-extents
        // so a swapped axis or coefficient cannot hide behind symmetry.
        let shape = Shape::Box {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let i = shape.inertia(6.0);
        assert_vec3_close(i, Vec3::new(6.5, 5.0, 2.5));
    }

    #[test]
    fn capsule_inertia_matches_closed_form_and_is_symmetric_about_axis() {
        // Symmetric about the local +Y axis: i_x == i_z != i_y.
        let shape = Shape::Capsule {
            radius: 1.0,
            half_height: 2.0,
        };
        let i = shape.inertia(3.0);
        assert_vec3_close(i, Vec3::new(5.75, 1.5, 5.75));
        assert_eq!(i.x, i.z, "capsule inertia must be symmetric about +Y");
    }

    #[test]
    fn sphere_aabb_is_centered_cube() {
        let shape = Shape::Sphere { radius: 2.0 };
        let aabb = shape.aabb(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        assert_vec3_close(aabb.min, Vec3::new(-1.0, 0.0, 1.0));
        assert_vec3_close(aabb.max, Vec3::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn box_aabb_at_identity_matches_half_extents() {
        let shape = Shape::Box {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let aabb = shape.aabb(Vec3::ZERO, Quat::IDENTITY);
        assert_vec3_close(aabb.min, Vec3::new(-1.0, -2.0, -3.0));
        assert_vec3_close(aabb.max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn box_aabb_grows_when_rotated_45_degrees() {
        // A unit cube rotated 45° about Z: the AABB half-extent on x/y
        // grows to half_extent * sqrt(2), z unchanged. Exact known value
        // pins the rotation being applied to the extents at all (a mutant
        // dropping the rotation entirely would keep the AABB at (1, 1, 1)).
        let shape = Shape::Box {
            half_extents: Vec3::splat(1.0),
        };
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let aabb = shape.aabb(Vec3::ZERO, rot);
        let expected = std::f32::consts::SQRT_2;
        assert!((aabb.max.x - expected).abs() < 1e-4, "{:?}", aabb.max);
        assert!((aabb.max.y - expected).abs() < 1e-4, "{:?}", aabb.max);
        assert!((aabb.max.z - 1.0).abs() < 1e-4, "{:?}", aabb.max);
    }

    #[test]
    fn capsule_aabb_extends_along_local_y_plus_radius() {
        let shape = Shape::Capsule {
            radius: 0.5,
            half_height: 2.0,
        };
        let aabb = shape.aabb(Vec3::ZERO, Quat::IDENTITY);
        // Along Y: half_height + radius. On X/Z: just the radius shell.
        assert_vec3_close(aabb.min, Vec3::new(-0.5, -2.5, -0.5));
        assert_vec3_close(aabb.max, Vec3::new(0.5, 2.5, 0.5));
    }
}
