//! Mesh/collider recipes → solver bodies.
//!
//! [`body_for`](crate::colliders::body_for) is the single place where authoring geometry becomes a
//! [`RigidBody`]: an explicit [`ColliderDesc`](ornis_assets::collider::ColliderDesc) lane entry wins, otherwise
//! the exact auto recipe ([`collider_for`](ornis_assets::collider::collider_for)) applies. Entities without a
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

/// Relative tolerance for treating per-axis scales as uniform.
const SCALE_TOLERANCE: f32 = 1e-6;

/// Validated positive, finite transform scale.
///
/// The renderer applies scale in body-local axes before rotation
/// (`T·R·S`), so collider geometry must be scaled the same way. A
/// non-positive or non-finite scale has no collider interpretation and
/// becomes [`ColliderError::InvalidScale`].
#[derive(Debug, Clone, Copy)]
struct Scale(Vec3);

impl Scale {
    fn try_new(raw: Vec3) -> Result<Self, ColliderError> {
        let positive = raw.is_finite() && raw.x > 0.0 && raw.y > 0.0 && raw.z > 0.0;
        if !positive {
            return Err(ColliderError::InvalidScale {
                scale: raw.to_array(),
            });
        }
        Ok(Self(raw))
    }

    fn raw(self) -> Vec3 {
        self.0
    }

    fn max_component(self) -> f32 {
        self.0.max_element()
    }

    /// Uniform factor when all axes agree within [`SCALE_TOLERANCE`],
    /// using the largest axis so the collider contains the scaled mesh.
    fn uniform_for(self, shape: &'static str) -> Result<f32, ColliderError> {
        let max = self.0.max_element();
        let min = self.0.min_element();
        if max - min <= SCALE_TOLERANCE * max.max(1.0) {
            Ok(max)
        } else {
            Err(ColliderError::NonUniformScale {
                shape,
                scale: self.0.to_array(),
            })
        }
    }

    /// Cylinder radius under scale: `radius * max(sx, sz)`. Non-uniform
    /// XZ is not representable by a circular cross-section and is a typed
    /// error.
    fn cylinder_radius(self, radius: f32) -> Result<f32, ColliderError> {
        let radial = self.0.x.max(self.0.z);
        if (self.0.x - self.0.z).abs() > SCALE_TOLERANCE * radial.max(1.0) {
            return Err(ColliderError::NonUniformScale {
                shape: "cylinder",
                scale: self.0.to_array(),
            });
        }
        Ok(radius * radial)
    }
}

/// Builds a solver body for an entity: `Ok(None)` means "no collider"
/// (no recipe, explicit `None`, or a non-`Custom` soup without an explicit
/// `TriMesh` recipe); `Err` means "broken collider" (bad soup or scale).
///
/// Precedence: explicit [`ColliderDesc`] first (`None` suppresses even
/// exact auto recipes), then the auto recipe for the mesh. `TriMesh`
/// needs a `Custom` soup with triangular, in-range indices. Pose comes
/// from `transform` (translation + quaternion); `mass` follows the
/// [`RigidBody`] convention (0 = static). `transform.scale` is applied in
/// body-local axes to match render extraction: boxes scale per axis,
/// spheres use the largest axis, cylinders require equal X/Z scale, and
/// compounds, rounded shapes and half-spaces require uniform scale.
///
/// # Errors
///
/// [`ColliderError`] when an explicit `TriMesh` soup is malformed
/// (non-triangular length or out-of-range indices), the scale is
/// non-positive/non-finite, or a shape cannot represent the non-uniform
/// scale.
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
    let scale = Scale::try_new(transform.scale)?;
    let position = transform.translation;
    let orientation = transform.rotation.get();
    let mut body = match recipe {
        ColliderDesc::Sphere { radius } => {
            RigidBody::new_sphere(position, radius * scale.max_component(), mass)
        }
        ColliderDesc::Box { half } => {
            RigidBody::new_box(position, Vec3::from_array(half) * scale.raw(), mass)
        }
        ColliderDesc::Cylinder { radius, height } => {
            let radius = scale.cylinder_radius(radius)?;
            RigidBody::new_cylinder(position, radius, height / 2.0 * scale.raw().y, mass)
        }
        ColliderDesc::TriMesh => {
            let Some((positions, indices)) = mesh.as_custom() else {
                return Ok(None);
            };
            let vertices: Vec<Vec3> = positions
                .iter()
                .map(|p| Vec3::from_array(*p) * scale.raw())
                .collect();
            let triangles = validated_triangles(&vertices, indices)?;
            RigidBody::try_new_trimesh(position, &vertices, &triangles, mass)
                .map_err(ColliderError::InvalidMesh)?
        }
        ColliderDesc::Compound { children } => {
            let factor = scale.uniform_for("compound")?;
            let child_scale = Scale(Vec3::splat(factor));
            let mut parts = Vec::with_capacity(children.len());
            for child in &children {
                let Some(shape) = shape_for(&child.shape, mesh, child_scale)? else {
                    continue; // `None` skips that part (render-only geometry).
                };
                let position = Vec3::from_array(child.translation) * factor;
                parts.push((shape, child_pose(child, position)));
            }
            RigidBody::try_new_compound(position, parts, mass)?
        }
        ColliderDesc::Round {
            inner,
            border_radius,
        } => {
            let factor = scale.uniform_for("rounded collider")?;
            let Some(inner_shape) = shape_for(inner.as_ref(), mesh, Scale(Vec3::splat(factor)))?
            else {
                return Err(ColliderError::EmptyRecipe {
                    detail: "rounded collider needs an inner recipe",
                });
            };
            RigidBody::try_new_round(position, inner_shape, border_radius * factor, mass)?
        }
        ColliderDesc::HalfSpace { normal } => {
            scale.uniform_for("half-space")?;
            RigidBody::try_new_halfspace(position, Vec3::from_array(normal), mass)?
        }
        ColliderDesc::None => return Ok(None),
    };
    body.orientation = orientation;
    Ok(Some(body))
}

/// Authoring recipe plus entity soup into solver geometry, with
/// body-local scale applied (`None` for [`ColliderDesc::None`] — the
/// caller decides skip-vs-error). Recursive over compound/rounded
/// recipes; triangle soup validates like [`body_for`].
fn shape_for(
    desc: &ColliderDesc,
    mesh: &MeshDesc,
    scale: Scale,
) -> Result<Option<Shape>, ColliderError> {
    match desc {
        ColliderDesc::Sphere { radius } => Ok(Some(Shape::Sphere {
            radius: radius * scale.max_component(),
        })),
        ColliderDesc::Box { half } => Ok(Some(Shape::Box {
            half_extents: Vec3::from_array(*half) * scale.raw(),
        })),
        ColliderDesc::Cylinder { radius, height } => Ok(Some(Shape::Cylinder {
            radius: scale.cylinder_radius(*radius)?,
            half_height: height / 2.0 * scale.raw().y,
        })),
        ColliderDesc::TriMesh => {
            let Some((positions, indices)) = mesh.as_custom() else {
                return Ok(None);
            };
            let vertices: Vec<Vec3> = positions
                .iter()
                .map(|p| Vec3::from_array(*p) * scale.raw())
                .collect();
            let triangles = validated_triangles(&vertices, indices)?;
            Ok(Some(Shape::TriMesh(crate::shape::TriMesh::from_triangles(
                &vertices, &triangles,
            )?)))
        }
        ColliderDesc::Compound { children } => {
            let factor = scale.uniform_for("compound")?;
            let child_scale = Scale(Vec3::splat(factor));
            let mut parts = Vec::with_capacity(children.len());
            for child in children {
                if let Some(shape) = shape_for(&child.shape, mesh, child_scale)? {
                    let position = Vec3::from_array(child.translation) * factor;
                    parts.push((shape, child_pose(child, position)));
                }
            }
            Ok(Some(Shape::try_compound(parts)?))
        }
        ColliderDesc::Round {
            inner,
            border_radius,
        } => {
            let factor = scale.uniform_for("rounded collider")?;
            let Some(inner_shape) = shape_for(inner.as_ref(), mesh, Scale(Vec3::splat(factor)))?
            else {
                return Err(ColliderError::EmptyRecipe {
                    detail: "rounded collider needs an inner recipe",
                });
            };
            Ok(Some(
                Shape::try_round(inner_shape, *border_radius * factor).ok_or(
                    crate::errors::ShapeError::BadBorderRadius {
                        radius: *border_radius * factor,
                    },
                )?,
            ))
        }
        ColliderDesc::HalfSpace { normal } => {
            scale.uniform_for("half-space")?;
            Ok(Some(
                Shape::try_halfspace(Vec3::from_array(*normal))
                    .ok_or(crate::errors::ShapeError::BadNormal)?,
            ))
        }
        ColliderDesc::None => Ok(None),
    }
}

/// Compound-local placement of one authoring child (`position` plus its
/// `[x, y, z, w]` quaternion; degenerate rotations fall back at query
/// time via [`Pose::world_rot`], so projection never fails here).
fn child_pose(child: &CompoundChild, position: Vec3) -> Pose {
    Pose::new(
        position,
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

    fn scaled(scale: Vec3) -> TransformDesc {
        TransformDesc {
            scale,
            ..TransformDesc::IDENTITY
        }
    }

    fn unit_box_mesh() -> MeshDesc {
        MeshDesc::Box {
            size: [
                PositiveF32::expect_valid(2.0),
                PositiveF32::expect_valid(2.0),
                PositiveF32::expect_valid(2.0),
            ],
        }
    }

    #[test]
    fn box_and_sphere_scale_in_body_local_axes() {
        let body = body_for(
            &scaled(Vec3::new(2.0, 3.0, 4.0)),
            &unit_box_mesh(),
            None,
            0.0,
        )
        .expect("scaled box infallible")
        .expect("scaled box builds");
        match body.shape {
            Shape::Box { half_extents } => assert_eq!(half_extents, Vec3::new(2.0, 3.0, 4.0)),
            _ => panic!("expected box"),
        }

        let sphere = MeshDesc::Sphere {
            radius: PositiveF32::expect_valid(2.0),
            segments: 16,
            rings: 8,
        };
        let body = body_for(&scaled(Vec3::new(2.0, 3.0, 1.5)), &sphere, None, 0.0)
            .expect("scaled sphere infallible")
            .expect("scaled sphere builds");
        match body.shape {
            Shape::Sphere { radius } => assert_eq!(radius, 6.0),
            _ => panic!("expected sphere"),
        }
    }

    #[test]
    fn cylinder_uses_radial_max_and_rejects_non_circular_xz() {
        let mesh = unit_box_mesh();
        let recipe = ColliderDesc::Cylinder {
            radius: 2.0,
            height: 4.0,
        };
        let body = body_for(&scaled(Vec3::new(2.0, 3.0, 2.0)), &mesh, Some(&recipe), 0.0)
            .expect("scaled cylinder infallible")
            .expect("scaled cylinder builds");
        match body.shape {
            Shape::Cylinder {
                radius,
                half_height,
            } => {
                assert_eq!(radius, 4.0);
                assert_eq!(half_height, 6.0);
            }
            _ => panic!("expected cylinder"),
        }
        assert!(matches!(
            body_for(&scaled(Vec3::new(2.0, 3.0, 1.0)), &mesh, Some(&recipe), 0.0),
            Err(ColliderError::NonUniformScale {
                shape: "cylinder",
                ..
            })
        ));
    }

    #[test]
    fn half_space_accepts_uniform_scale_only() {
        let mesh = unit_box_mesh();
        let recipe = ColliderDesc::HalfSpace {
            normal: [0.0, 1.0, 0.0],
        };
        let body = body_for(&scaled(Vec3::splat(2.0)), &mesh, Some(&recipe), 0.0)
            .expect("scaled half-space infallible")
            .expect("scaled half-space builds");
        match body.shape {
            Shape::HalfSpace { normal } => assert_eq!(normal.get(), Vec3::Y),
            _ => panic!("expected half-space"),
        }
        assert!(matches!(
            body_for(&scaled(Vec3::new(2.0, 1.0, 1.0)), &mesh, Some(&recipe), 0.0),
            Err(ColliderError::NonUniformScale {
                shape: "half-space",
                ..
            })
        ));
    }

    #[test]
    fn invalid_scale_is_typed() {
        let mesh = unit_box_mesh();
        for scale in [
            Vec3::ZERO,
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(f32::NAN, 1.0, 1.0),
            Vec3::new(1.0, f32::INFINITY, 1.0),
        ] {
            assert!(
                matches!(
                    body_for(&scaled(scale), &mesh, None, 0.0),
                    Err(ColliderError::InvalidScale { .. })
                ),
                "scale {scale:?} must be rejected"
            );
        }
    }

    #[test]
    fn trimesh_vertices_follow_scale() {
        let soup = MeshDesc::Custom {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            indices: vec![0, 1, 2],
        };
        let body = body_for(
            &scaled(Vec3::new(2.0, 3.0, 4.0)),
            &soup,
            Some(&ColliderDesc::TriMesh),
            0.0,
        )
        .expect("scaled soup infallible")
        .expect("scaled soup builds");
        match body.shape {
            Shape::TriMesh(mesh) => {
                assert_eq!(mesh.local_min, Vec3::ZERO);
                assert_eq!(mesh.local_max, Vec3::new(2.0, 6.0, 0.0));
            }
            _ => panic!("expected trimesh"),
        }
    }

    #[test]
    fn uniform_scale_propagates_through_compound_and_round() {
        use ornis_assets::collider::CompoundChild;
        let mesh = unit_box_mesh();
        let compound = ColliderDesc::Compound {
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
        let body = body_for(&scaled(Vec3::splat(2.0)), &mesh, Some(&compound), 1.0)
            .expect("scaled compound infallible")
            .expect("scaled compound builds");
        let children = body.shape.compound_children().expect("compound shape");
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].1.position, Vec3::new(-2.0, 0.0, 0.0));
        match children[1].0 {
            Shape::Box { half_extents } => assert_eq!(half_extents, Vec3::splat(2.0)),
            _ => panic!("expected child box"),
        }

        let round = ColliderDesc::Round {
            inner: Box::new(ColliderDesc::Box {
                half: [1.0, 1.0, 1.0],
            }),
            border_radius: 0.25,
        };
        let body = body_for(&scaled(Vec3::splat(2.0)), &mesh, Some(&round), 1.0)
            .expect("scaled round infallible")
            .expect("scaled round builds");
        match body.shape {
            Shape::Round {
                inner,
                border_radius,
            } => {
                assert_eq!(border_radius, 0.5);
                match *inner {
                    Shape::Box { half_extents } => assert_eq!(half_extents, Vec3::splat(2.0)),
                    _ => panic!("expected rounded box"),
                }
            }
            _ => panic!("expected round"),
        }
    }

    #[test]
    fn compound_and_round_reject_non_uniform_scale() {
        use ornis_assets::collider::CompoundChild;
        let mesh = unit_box_mesh();
        let compound = ColliderDesc::Compound {
            children: vec![CompoundChild {
                shape: ColliderDesc::Box {
                    half: [1.0, 1.0, 1.0],
                },
                translation: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            }],
        };
        assert!(matches!(
            body_for(
                &scaled(Vec3::new(2.0, 1.0, 1.0)),
                &mesh,
                Some(&compound),
                1.0
            ),
            Err(ColliderError::NonUniformScale {
                shape: "compound",
                ..
            })
        ));
        let round = ColliderDesc::Round {
            inner: Box::new(ColliderDesc::Box {
                half: [1.0, 1.0, 1.0],
            }),
            border_radius: 0.25,
        };
        assert!(matches!(
            body_for(&scaled(Vec3::new(2.0, 1.0, 1.0)), &mesh, Some(&round), 1.0),
            Err(ColliderError::NonUniformScale {
                shape: "rounded collider",
                ..
            })
        ));
    }
}
