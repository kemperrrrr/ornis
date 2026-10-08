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
    /// Rigid union of placed child recipes (P5): each child owns a local
    /// [`CompoundChild`] placement. `None` children are skipped (a
    /// render-only part inside a compound); an all-`None`/empty union is a
    /// projection error, not a silent empty body.
    Compound {
        /// Child recipes with their compound-local placements.
        children: Vec<CompoundChild>,
    },
    /// Convex recipe dilated by `border_radius` (P5): contacts form a full
    /// radius earlier than the inner surface. The inner `None` recipe is a
    /// projection error (a rounded nothing is meaningless).
    Round {
        /// Inner recipe to dilate.
        inner: Box<ColliderDesc>,
        /// Dilation radius in world units (must be finite and `> 0`).
        border_radius: f32,
    },
    /// Infinite static plane through the entity position with outward
    /// `normal` (P5): the floor primitive. Static-only — projecting with a
    /// dynamic mass is a projection error, never a silent grounding.
    HalfSpace {
        /// Outward plane normal in local space (need not be unit).
        normal: [f32; 3],
    },
    /// Explicitly no collider: suppresses even the exact auto recipes.
    /// This is how a render-only sphere opts out of physics.
    None,
}

/// One compound child: a collider recipe with its compound-local placement
/// (translation plus unit quaternion `[x, y, z, w]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompoundChild {
    /// Child collider recipe (`None` skips this part).
    pub shape: ColliderDesc,
    /// Child origin in the compound frame.
    pub translation: [f32; 3],
    /// Child orientation in the compound frame (`[x, y, z, w]`).
    pub rotation: [f32; 4],
}

/// Exact auto recipe for `mesh`, or `None` when no exact recipe exists.
///
/// Sphere/Box/Cylinder meshes map to their analytic collider; Plane, Quad
/// and Custom meshes map to `None` — a transport mesh is not a collision
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
        MeshDesc::Plane { .. } | MeshDesc::Quad { .. } | MeshDesc::Custom { .. } => None,
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
            collider_for(&MeshDesc::Quad {
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
