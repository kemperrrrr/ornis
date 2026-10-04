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
    collider::{ColliderDesc, CompoundChild, collider_for},
    scene::{MeshDesc, TransformDesc},
};

use crate::body::RigidBody;
use crate::errors::ColliderError;
use crate::shape::{Pose, Shape};

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
    let position = transform.translation;
    let orientation = transform.rotation.get();
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
        ColliderDesc::Compound { children } => {
            let mut parts = Vec::with_capacity(children.len());
            for child in &children {
                let Some(shape) = shape_for(&child.shape, mesh)? else {
                    continue; // `None` skips that part (render-only geometry).
                };
                parts.push((shape, child_pose(child)));
            }
            RigidBody::try_new_compound(position, parts, mass)?
        }
        ColliderDesc::Round {
            inner,
            border_radius,
        } => {
            let Some(inner_shape) = shape_for(inner.as_ref(), mesh)? else {
                return Err(ColliderError::EmptyRecipe {
                    detail: "rounded collider needs an inner recipe",
                });
            };
            RigidBody::try_new_round(position, inner_shape, border_radius, mass)?
        }
        ColliderDesc::HalfSpace { normal } => {
            RigidBody::try_new_halfspace(position, Vec3::from_array(normal), mass)?
        }
        ColliderDesc::None => return Ok(None),
    };
    body.orientation = orientation;
    Ok(Some(body))
}

/// Authoring recipe plus entity soup into solver geometry (`None` for
/// [`ColliderDesc::None`] — the caller decides skip-vs-error). Recursive
/// over compound/rounded recipes; triangle soup validates like
/// [`body_for`].
fn shape_for(desc: &ColliderDesc, mesh: &MeshDesc) -> Result<Option<Shape>, ColliderError> {
    match desc {
        ColliderDesc::Sphere { radius } => Ok(Some(Shape::Sphere { radius: *radius })),
        ColliderDesc::Box { half } => Ok(Some(Shape::Box {
            half_extents: Vec3::from_array(*half),
        })),
        ColliderDesc::Cylinder { radius, height } => Ok(Some(Shape::Cylinder {
            radius: *radius,
            half_height: height / 2.0,
        })),
        ColliderDesc::TriMesh => {
            let Some((positions, indices)) = mesh.as_custom() else {
                return Ok(None);
            };
            let vertices: Vec<Vec3> = positions.iter().map(|p| Vec3::from_array(*p)).collect();
            let triangles = validated_triangles(&vertices, indices)?;
            Ok(Some(Shape::TriMesh(crate::shape::TriMesh::from_triangles(
                &vertices, &triangles,
            )?)))
        }
        ColliderDesc::Compound { children } => {
            let mut parts = Vec::with_capacity(children.len());
            for child in children {
                if let Some(shape) = shape_for(&child.shape, mesh)? {
                    parts.push((shape, child_pose(child)));
                }
            }
            Ok(Some(Shape::try_compound(parts)?))
        }
        ColliderDesc::Round {
            inner,
            border_radius,
        } => {
            let Some(inner_shape) = shape_for(inner.as_ref(), mesh)? else {
                return Err(ColliderError::EmptyRecipe {
                    detail: "rounded collider needs an inner recipe",
                });
            };
            Ok(Some(Shape::try_round(inner_shape, *border_radius).ok_or(
                crate::errors::ShapeError::BadBorderRadius {
                    radius: *border_radius,
                },
            )?))
        }
        ColliderDesc::HalfSpace { normal } => Ok(Some(
            Shape::try_halfspace(Vec3::from_array(*normal))
                .ok_or(crate::errors::ShapeError::BadNormal)?,
        )),
        ColliderDesc::None => Ok(None),
    }
}

/// Compound-local placement of one authoring child (translation plus
/// `[x, y, z, w]` quaternion; degenerate rotations fall back at query
/// time via [`Pose::world_rot`], so projection never fails here).
fn child_pose(child: &CompoundChild) -> Pose {
    Pose::new(
        Vec3::from_array(child.translation),
        Quat::from_xyzw(
            child.rotation[0],
            child.rotation[1],
            child.rotation[2],
            child.rotation[3],
        ),
    )
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
    /// Vertices per triangle index triple.
    const TRI_VERTS: usize = 3;
    if !indices.len().is_multiple_of(TRI_VERTS) {
        return Err(ColliderError::BadIndexCount { len: indices.len() });
    }
    let triangles: Vec<crate::shape::Triangle> = indices
        .chunks_exact(TRI_VERTS)
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
            translation: Vec3::new(1.0, 2.0, 3.0),
            ..TransformDesc::IDENTITY
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

    #[test]
    fn compound_round_and_halfspace_recipes_project() {
        use ornis_assets::collider::CompoundChild;
        let transform = transform();
        let mesh = MeshDesc::Box {
            size: [
                PositiveF32::expect_valid(2.0),
                PositiveF32::expect_valid(2.0),
                PositiveF32::expect_valid(2.0),
            ],
        };
        // Compound of two explicit boxes.
        let recipe = ColliderDesc::Compound {
            children: vec![
                CompoundChild {
                    shape: ColliderDesc::Box {
                        half: [1.0, 1.0, 1.0],
                    },
                    translation: [-1.0, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                },
                CompoundChild {
                    shape: ColliderDesc::Box {
                        half: [1.0, 1.0, 1.0],
                    },
                    translation: [1.0, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                },
            ],
        };
        let body = body_for(&transform, &mesh, Some(&recipe), 1.0)
            .expect("compound recipe infallible")
            .expect("compound recipe builds");
        assert!(matches!(body.shape, crate::shape::Shape::Compound { .. }));
        // Rounded box recipe.
        let recipe = ColliderDesc::Round {
            inner: Box::new(ColliderDesc::Box {
                half: [1.0, 1.0, 1.0],
            }),
            border_radius: 0.25,
        };
        let body = body_for(&transform, &mesh, Some(&recipe), 1.0)
            .expect("round recipe infallible")
            .expect("round recipe builds");
        assert!(matches!(body.shape, crate::shape::Shape::Round { .. }));
        // Half-space recipe builds a static floor.
        let recipe = ColliderDesc::HalfSpace {
            normal: [0.0, 1.0, 0.0],
        };
        let body = body_for(&transform, &mesh, Some(&recipe), 0.0)
            .expect("half-space recipe infallible")
            .expect("half-space recipe builds");
        assert!(matches!(body.shape, crate::shape::Shape::HalfSpace { .. }));
        assert_eq!(body.body_type, crate::body::BodyType::Static);
    }

    #[test]
    fn new_recipes_reject_degenerates_loudly() {
        use ornis_assets::collider::CompoundChild;
        let transform = transform();
        let mesh = MeshDesc::Box {
            size: [
                PositiveF32::expect_valid(2.0),
                PositiveF32::expect_valid(2.0),
                PositiveF32::expect_valid(2.0),
            ],
        };
        // Dynamic half-space is an explicit error, never a silent grounding.
        assert!(matches!(
            body_for(
                &transform,
                &mesh,
                Some(&ColliderDesc::HalfSpace {
                    normal: [0.0, 1.0, 0.0]
                }),
                1.0,
            ),
            Err(ColliderError::InvalidShape(
                crate::errors::ShapeError::DynamicHalfSpace { .. }
            ))
        ));
        // Zero normal is a typed error.
        assert!(matches!(
            body_for(
                &transform,
                &mesh,
                Some(&ColliderDesc::HalfSpace {
                    normal: [0.0, 0.0, 0.0]
                }),
                0.0,
            ),
            Err(ColliderError::InvalidShape(
                crate::errors::ShapeError::BadNormal
            ))
        ));
        // Empty compound is a typed error.
        assert!(matches!(
            body_for(
                &transform,
                &mesh,
                Some(&ColliderDesc::Compound { children: vec![] }),
                0.0,
            ),
            Err(ColliderError::InvalidShape(
                crate::errors::ShapeError::EmptyCompound
            ))
        ));
        // `None` children are skipped; an all-`None` compound still errors.
        let all_none = ColliderDesc::Compound {
            children: vec![CompoundChild {
                shape: ColliderDesc::None,
                translation: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            }],
        };
        assert!(matches!(
            body_for(&transform, &mesh, Some(&all_none), 0.0),
            Err(ColliderError::InvalidShape(
                crate::errors::ShapeError::EmptyCompound
            ))
        ));
        // Rounded nothing is a typed error.
        let no_inner = ColliderDesc::Round {
            inner: Box::new(ColliderDesc::None),
            border_radius: 0.25,
        };
        assert!(matches!(
            body_for(&transform, &mesh, Some(&no_inner), 0.0),
            Err(ColliderError::EmptyRecipe { .. })
        ));
    }
}
