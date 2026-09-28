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
use crate::errors::ColliderError;

/// Builds a solver body for an entity: `Ok(None)` means "no collider"
/// (no recipe, explicit `None`, or a non-`Custom` soup without an explicit
/// `TriMesh` recipe); `Err` means "broken collider" (bad soup).
///
/// Precedence: explicit [`ColliderDesc`] first (`None` suppresses even
/// exact auto recipes), then the auto recipe for the mesh. `TriMesh`
/// needs a `Custom` soup with triangular, in-range indices. Pose comes
/// from `transform` (translation + quaternion); `mass` follows the
/// [`RigidBody`] convention (0 = static).
///
/// # Errors
///
/// [`ColliderError`] when an explicit `TriMesh` soup is malformed
/// (non-triangular length or out-of-range indices).
pub fn body_for(
    transform: &TransformDesc,
    mesh: &MeshDesc,
    collider: Option<&ColliderDesc>,
    mass: f32,
) -> Result<Option<RigidBody>, ColliderError> {
    let recipe = match collider {
        Some(recipe) => Some(recipe.clone()),
        None => collider_for(mesh),
    };
    let Some(recipe) = recipe else {
        return Ok(None);
    };
    if matches!(recipe, ColliderDesc::None) {
        return Ok(None);
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
            let Some((positions, indices)) = mesh.as_custom() else {
                return Ok(None);
            };
            let vertices: Vec<Vec3> = positions.iter().map(|p| Vec3::from_array(*p)).collect();
            let triangles = validated_triangles(&vertices, indices)?;
            RigidBody::try_new_trimesh(position, &vertices, &triangles, mass)
                .map_err(ColliderError::InvalidMesh)?
        }
        ColliderDesc::None => return Ok(None),
    };
    body.orientation = orientation;
    Ok(Some(body))
}

/// Chunks a flat soup index list into typed triangles.
///
/// # Errors
///
/// [`ColliderError::BadIndexCount`] on non-triangular length,
/// [`ColliderError::IndexOutOfRange`] on dangling indices.
fn validated_triangles(
    vertices: &[Vec3],
    indices: &[u32],
) -> Result<Vec<crate::shape::Triangle>, ColliderError> {
    if !indices.len().is_multiple_of(3) {
        return Err(ColliderError::BadIndexCount { len: indices.len() });
    }
    let triangles: Vec<crate::shape::Triangle> = indices
        .chunks_exact(3)
        .map(|c| crate::shape::Triangle::from_raw([c[0], c[1], c[2]]))
        .collect();
    for (t, tri) in triangles.iter().enumerate() {
        for raw in tri.as_u32() {
            if (raw as usize) >= vertices.len() {
                let _ = t;
                return Err(ColliderError::IndexOutOfRange {
                    index: raw,
                    vertices: vertices.len(),
                });
            }
        }
    }
    Ok(triangles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_core::units::PositiveF32;

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
                radius: PositiveF32::expect_valid(2.0),
                segments: 16,
                rings: 8,
            },
            None,
            0.0,
        )
        .expect("sphere recipe infallible")
        .expect("sphere recipe builds");
        assert_eq!(body.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(body.orientation, Quat::IDENTITY);

        let body = body_for(
            &transform,
            &MeshDesc::Box {
                size: [
                    PositiveF32::expect_valid(2.0),
                    PositiveF32::expect_valid(4.0),
                    PositiveF32::expect_valid(6.0),
                ],
            },
            None,
            0.0,
        )
        .expect("box recipe infallible")
        .expect("box recipe builds");
        assert_eq!(body.position, Vec3::new(1.0, 2.0, 3.0));

        assert!(
            body_for(
                &transform,
                &MeshDesc::Plane {
                    size: [
                        PositiveF32::expect_valid(3.0),
                        PositiveF32::expect_valid(5.0)
                    ]
                },
                None,
                0.0
            )
            .expect("plane infallible")
            .is_none()
        );
    }

    #[test]
    fn explicit_collider_wins_and_none_suppresses() {
        let transform = transform();
        let mesh = MeshDesc::Sphere {
            radius: PositiveF32::expect_valid(2.0),
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
        .expect("explicit recipe infallible")
        .expect("explicit recipe builds");
        assert_eq!(body.position, Vec3::new(1.0, 2.0, 3.0));
        // Explicit None suppresses even exact recipes.
        assert!(
            body_for(&transform, &mesh, Some(&ColliderDesc::None), 0.0)
                .expect("none infallible")
                .is_none()
        );
    }

    #[test]
    fn trimesh_needs_a_valid_custom_soup() {
        let transform = transform();
        let soup = MeshDesc::Custom {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![0, 1, 2],
        };
        assert!(
            body_for(&transform, &soup, None, 0.0)
                .expect("auto soup infallible")
                .is_none()
        );
        assert!(
            body_for(&transform, &soup, Some(&ColliderDesc::TriMesh), 0.0,)
                .expect("explicit soup infallible")
                .is_some()
        );
        let bad = MeshDesc::Custom {
            positions: vec![[0.0, 0.0, 0.0]],
            indices: vec![0, 1, 2],
        };
        // Broken soup is now a typed error, not a silent `None`.
        assert!(matches!(
            body_for(&transform, &bad, Some(&ColliderDesc::TriMesh), 0.0),
            Err(ColliderError::IndexOutOfRange { .. })
        ));
    }

    #[test]
    fn trimesh_bad_index_count_is_typed() {
        let transform = transform();
        let bad = MeshDesc::Custom {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            indices: vec![0, 1],
        };
        assert!(matches!(
            body_for(&transform, &bad, Some(&ColliderDesc::TriMesh), 0.0),
            Err(ColliderError::BadIndexCount { .. })
        ));
    }
}
