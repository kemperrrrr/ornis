//! Mesh/collider recipes → solver bodies.
//!
//! [`body_for`] is the single place where authoring geometry becomes a
//! [`RigidBody`]: an explicit [`ColliderDesc`] lane entry wins, otherwise
//! the exact auto recipe ([`collider_for`]) applies. Entities without a
//! buildable recipe get no body (`None`) — transport never invents
//! colliders. The editor calls this on spawn and on explicit collider
//! edits; per-step pose sync stays in the engine runtime.

use glam::{Quat, Vec3};
use ornis_assets::{
    collider::{ColliderDesc, collider_for},
    scene::{MeshDesc, TransformDesc},
};

use crate::body::RigidBody;

/// Builds a solver body for an entity, or returns `None` when it has no
/// collider.
///
/// Precedence: explicit [`ColliderDesc`] first (`None` suppresses even
/// exact auto recipes), then the auto recipe for the mesh. `TriMesh`
/// needs a `Custom` soup with triangular, in-range indices — anything
/// else yields `None`. Pose comes from `transform` (translation +
/// quaternion); `mass` follows the [`RigidBody`] convention (0 = static).
pub fn body_for(
    transform: &TransformDesc,
    mesh: &MeshDesc,
    collider: Option<&ColliderDesc>,
    mass: f32,
) -> Option<RigidBody> {
    let recipe = match collider {
        Some(recipe) => Some(recipe.clone()),
        None => collider_for(mesh),
    };
    let recipe = recipe?;
    if matches!(recipe, ColliderDesc::None) {
        return None;
    }
    let position = Vec3::from_array(transform.translation);
    let rotation = transform.rotation;
    let orientation = Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
    let mut body = match recipe {
        ColliderDesc::Sphere { radius } => RigidBody::new_sphere(position, radius, mass),
        ColliderDesc::Box { half } => RigidBody::new_box(position, Vec3::from_array(half), mass),
        ColliderDesc::Cylinder { radius, height } => {
            RigidBody::new_cylinder(position, radius, height / 2.0, mass)
        }
        ColliderDesc::TriMesh => {
            let (positions, indices) = mesh.as_custom()?;
            let vertices: Vec<Vec3> = positions.iter().map(|p| Vec3::from_array(*p)).collect();
            let triangles = validated_triangles(&vertices, indices)?;
            RigidBody::new_trimesh(position, &vertices, &triangles, mass)
        }
        ColliderDesc::None => return None,
    };
    body.orientation = orientation;
    Some(body)
}

/// Chunks a flat soup index list into triangles, rejecting anything the
/// solver cannot consume: non-triangular length or out-of-range indices.
fn validated_triangles(vertices: &[Vec3], indices: &[u32]) -> Option<Vec<[u32; 3]>> {
    if !indices.len().is_multiple_of(3) {
        return None;
    }
    let triangles: Vec<[u32; 3]> = indices
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    if triangles
        .iter()
        .any(|t| t.iter().any(|&i| (i as usize) >= vertices.len()))
    {
        return None;
    }
    Some(triangles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform() -> TransformDesc {
        TransformDesc {
            translation: [1.0, 2.0, 3.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn auto_recipes_build_exact_bodies() {
        let transform = transform();
        let body = body_for(
            &transform,
            &MeshDesc::Sphere {
                radius: 2.0,
                segments: 16,
                rings: 8,
            },
            None,
            0.0,
        )
        .expect("sphere recipe builds");
        assert_eq!(body.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(body.orientation, Quat::IDENTITY);

        let body = body_for(
            &transform,
            &MeshDesc::Box {
                size: [2.0, 4.0, 6.0],
            },
            None,
            0.0,
        )
        .expect("box recipe builds");
        assert_eq!(body.position, Vec3::new(1.0, 2.0, 3.0));

        assert!(body_for(&transform, &MeshDesc::Plane { size: [3.0, 5.0] }, None, 0.0).is_none());
    }

    #[test]
    fn explicit_collider_wins_and_none_suppresses() {
        let transform = transform();
        let mesh = MeshDesc::Sphere {
            radius: 2.0,
            segments: 16,
            rings: 8,
        };
        // Explicit box overrides the sphere auto recipe.
        let body = body_for(
            &transform,
            &mesh,
            Some(&ColliderDesc::Box {
                half: [1.0, 1.0, 1.0],
            }),
            0.0,
        )
        .expect("explicit recipe builds");
        assert_eq!(body.position, Vec3::new(1.0, 2.0, 3.0));
        // Explicit None suppresses even exact recipes.
        assert!(body_for(&transform, &mesh, Some(&ColliderDesc::None), 0.0).is_none());
    }

    #[test]
    fn trimesh_needs_a_valid_custom_soup() {
        let transform = transform();
        let soup = MeshDesc::Custom {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![0, 1, 2],
        };
        assert!(body_for(&transform, &soup, None, 0.0).is_none());
        assert!(body_for(&transform, &soup, Some(&ColliderDesc::TriMesh), 0.0,).is_some());
        let bad = MeshDesc::Custom {
            positions: vec![[0.0, 0.0, 0.0]],
            indices: vec![0, 1, 2],
        };
        assert!(body_for(&transform, &bad, Some(&ColliderDesc::TriMesh), 0.0).is_none());
    }
}
