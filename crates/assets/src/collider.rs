//! Explicit collider recipes.
//!
//! [`ColliderDesc`] is the editor/authoring answer to "what collides here":
//! an explicit recipe component, never inferred from render geometry.
//! Transport alone must not invent colliders — the old binary helper built
//! a sphere for every sphere mesh and silently skipped the rest. Exact
//! analytic recipes ([`ColliderDesc::Sphere`]/`Box`/`Cylinder`) derive from
//! [`MeshDesc`](crate::scene::MeshDesc) via [`collider_for`]; everything
//! else needs an explicit [`ColliderDesc`] lane entry, and entities
//! without one simply have no body. The physics crate turns recipes into
//! solver bodies; this crate never names physics types.

use serde::{Deserialize, Serialize};

use crate::scene::MeshDesc;

/// Collision recipe for one entity, stored as an ECS component.
///
/// Register under `"Collider"` to make it editor-addressable. Entities
/// without this lane fall back to [`collider_for`] over their mesh.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColliderDesc {
    /// Exact sphere (also the [`collider_for`] recipe for sphere meshes).
    Sphere {
        /// Radius in world units.
        radius: f32,
    },
    /// Exact axis-aligned box in local space (also the recipe for box
    /// meshes, with `half = size / 2`).
    Box {
        /// Half extents per axis in world units.
        half: [f32; 3],
    },
    /// Exact cylinder around local `+Y` (also the recipe for cylinder
    /// meshes).
    Cylinder {
        /// Radius in world units.
        radius: f32,
        /// Full height along `+Y` in world units.
        height: f32,
    },
    /// Exact triangle soup: the entity's own [`MeshDesc::Custom`] data.
    /// Only meaningful alongside a `Custom` mesh; concave-vs-concave
    /// contact is undefined in the solver, so prefer this for static
    /// level geometry, not dynamic bodies.
    TriMesh,
    /// Explicitly no collider: suppresses even the exact auto recipes.
    /// This is how a render-only sphere opts out of physics.
    None,
}

/// Exact auto recipe for `mesh`, or `None` when no exact recipe exists.
///
/// Sphere/Box/Cylinder meshes map to their analytic collider; Plane and
/// Custom meshes map to `None` — a transport mesh is not a collision
/// promise. Pair with an explicit [`ColliderDesc`] lane entry for those
/// (`TriMesh` validates the `Custom` soup through [`MeshDesc::as_triangles`]
/// at the physics projection).
pub fn collider_for(mesh: &MeshDesc) -> Option<ColliderDesc> {
    match mesh {
        MeshDesc::Sphere { radius, .. } => Some(ColliderDesc::Sphere {
            radius: radius.get(),
        }),
        MeshDesc::Box { size } => Some(ColliderDesc::Box {
            half: [
                size[0].get() / 2.0,
                size[1].get() / 2.0,
                size[2].get() / 2.0,
            ],
        }),
        MeshDesc::Cylinder { radius, height, .. } => Some(ColliderDesc::Cylinder {
            radius: radius.get(),
            height: height.get(),
        }),
        MeshDesc::Plane { .. } | MeshDesc::Custom { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_core::units::PositiveF32;

    #[test]
    fn exact_recipes_derive_from_meshes() {
        assert_eq!(
            collider_for(&MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(2.0),
                segments: 16,
                rings: 8,
            }),
            Some(ColliderDesc::Sphere { radius: 2.0 })
        );
        assert_eq!(
            collider_for(&MeshDesc::Box {
                size: [
                    PositiveF32::expect_valid(2.0),
                    PositiveF32::expect_valid(4.0),
                    PositiveF32::expect_valid(6.0),
                ]
            }),
            Some(ColliderDesc::Box {
                half: [1.0, 2.0, 3.0]
            })
        );
        assert_eq!(
            collider_for(&MeshDesc::Cylinder {
                radius: PositiveF32::expect_valid(1.5),
                height: PositiveF32::expect_valid(7.0),
                radial_segments: 12,
            }),
            Some(ColliderDesc::Cylinder {
                radius: 1.5,
                height: 7.0
            })
        );
    }

    #[test]
    fn plane_and_custom_have_no_auto_recipe() {
        assert_eq!(
            collider_for(&MeshDesc::Plane {
                size: [
                    PositiveF32::expect_valid(3.0),
                    PositiveF32::expect_valid(5.0),
                ]
            }),
            None
        );
        assert_eq!(
            collider_for(&MeshDesc::Custom {
                positions: vec![[0.0, 0.0, 0.0]],
                indices: vec![0, 0, 0],
            }),
            None
        );
    }
}
