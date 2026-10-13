//! K1 planar physics: 2D as degenerate 3D (Unreal Paper2D approach).
//!
//! A 2D body is an ordinary thin 3D [`RigidBody`] locked to a [`MotionPlane`]
//! by a [`planar_joint_kind`] [`JointKind::SixDof`] joint against a dedicated
//! static [`PlaneAnchor`]: translation is free in the plane and locked along
//! its normal, rotation is locked except the yaw about the normal (the
//! two-locked-axes swing hold of the sequential-impulse position pass). No
//! new solver and no new body mode — planar dynamics are ordinary 3D dynamics
//! under that lock.
//!
//! 2D shapes are thin 3D shapes: [`PlanarShape::Rect`] is a flat box,
//! [`PlanarShape::Circle`] a short cylinder whose axis is turned onto the
//! plane normal, [`PlanarShape::Polygon`] a prism hull recentered on its area
//! centroid. Supported solver: SequentialImpulse (directly or
//! [`Engine`](crate::Engine) with [`RoutingKind::Single`](crate::RoutingKind)).
//! AVBD accepts the joint but is a known limit (see the crate tests), XPBD
//! rejects it, and `Islands` routing can move a plane-locked island to AVBD —
//! all outside the K1 scope.
//!
//! # Example
//!
//! ```
//! use glam::{Vec2, Vec3};
//! use ornis_physics::{
//!     add_plane_anchor, constrain_to_plane, planar_body, Engine, MotionPlane, PhysicsEngine,
//!     PlanarPose, PlanarShape, RigidBody, SolverKind, PLANAR_THICKNESS,
//! };
//!
//! let mut engine = Engine::new(SolverKind::SequentialImpulse, Vec3::new(0.0, -9.81, 0.0));
//! engine.add_body(RigidBody::new_box(Vec3::new(0.0, -1.0, 0.0), Vec3::new(30.0, 1.0, 2.0), 0.0));
//! let plane = MotionPlane::XY;
//! let anchor = add_plane_anchor(&mut engine, plane);
//! let crate_box = planar_body(
//!     &plane,
//!     &PlanarShape::Rect { half_extents: Vec2::new(0.5, 0.5) },
//!     PlanarPose { position: Vec2::new(0.0, 2.0), angle: 0.3 },
//!     PLANAR_THICKNESS,
//!     1.0,
//! )?;
//! let h = engine.add_body(crate_box);
//! constrain_to_plane(&mut engine, anchor, h)?;
//! let wheel = planar_body(&plane, &PlanarShape::Circle { radius: 0.3 },
//!     PlanarPose { position: Vec2::new(1.0, 3.0), angle: 0.0 }, PLANAR_THICKNESS, 1.0)?;
//! let w = engine.add_body(wheel);
//! constrain_to_plane(&mut engine, anchor, w)?;
//! # Ok::<(), ornis_physics::PlanarError>(())
//! ```

use core::f32::consts::FRAC_PI_2;
use glam::{Quat, Vec2, Vec3};

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::engine::PhysicsEngine;
use crate::errors::PlanarError;
use crate::invariants::UnitVec3;
use crate::joint::{AxisConfig, JointHandle, JointKind};

/// Default depth of a 2D shape along the plane normal (m). Planar dynamics do not depend on it:
/// the inertia about the normal of a box, cylinder or prism does not depend on depth, and the
/// tilt axes are locked. It only sets the contact surface width.
pub const PLANAR_THICKNESS: f32 = 0.1;

/// Max vertex count of a convex 2D polygon: 2·N prism vertices must fit in
/// `ConvexHull::FACE_CAP` (64), so the hull keeps faces (exact inertia).
pub const MAX_POLYGON_VERTICES: usize = 32;

/// Radius of the plane anchor's static sphere (m). It never collides (layer/mask 0) and only
/// carries the SixDof frame.
pub const PLANE_ANCHOR_RADIUS: f32 = 1.0e-3;

/// Denominator scale of the polygon area-centroid moment formula
/// (`3 · Σ cross = 6 · A`).
const CENTROID_MOMENT_SCALE: f32 = 3.0;

/// Smallest polygon outline (triangle).
const MIN_POLYGON_VERTICES: usize = 3;

/// Signed doubled area below which a polygon outline counts as degenerate
/// (zero area).
const POLYGON_AREA_EPS: f32 = 1.0e-8;

/// Plane of 2D motion: a point plus an orthonormal frame whose local +Z is the normal and whose
/// local X/Y are the in-plane axes u/v.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionPlane {
    origin: Vec3,
    frame: Quat,
}

impl MotionPlane {
    /// XY plane through the world origin, normal +Z (side view, gravity −Y).
    pub const XY: Self = Self {
        origin: Vec3::ZERO,
        frame: Quat::IDENTITY,
    };

    /// Plane through `origin` with normal `normal`;
    /// `frame = Quat::from_rotation_arc(Vec3::Z, normal.get())`.
    pub fn new(origin: Vec3, normal: UnitVec3) -> Self {
        Self {
            origin,
            frame: Quat::from_rotation_arc(Vec3::Z, normal.get()),
        }
    }

    /// Point on the plane (the plane anchor's position).
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Plane frame: local X/Y → u/v, local Z → normal.
    pub fn frame(&self) -> Quat {
        self.frame
    }

    /// Unit normal `frame * Vec3::Z`.
    pub fn normal(&self) -> Vec3 {
        self.frame * Vec3::Z
    }

    /// Plane point (u, v) → world: `origin + frame * (u, v, 0)`.
    pub fn to_world(&self, p: Vec2) -> Vec3 {
        self.origin + self.frame * lift_to_3d(p)
    }

    /// World → plane coordinates (normal component dropped).
    // qual:api — public entry; in-tree callers are the unit tests.
    pub fn to_plane(&self, p: Vec3) -> Vec2 {
        let local = self.frame.conjugate() * (p - self.origin);
        Vec2::new(local.x, local.y)
    }

    /// Signed distance to the plane along the normal.
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        (p - self.origin).dot(self.frame * Vec3::Z)
    }

    /// Body orientation for 2D angle `angle` (rad, CCW about the normal):
    /// `frame * Quat::from_rotation_z(angle)`.
    pub fn orientation(&self, angle: f32) -> Quat {
        self.frame * yaw_rotation(angle)
    }
}

/// Lift of a plane point into body-local 3D: `(u, v, 0)`.
fn lift_to_3d(p: Vec2) -> Vec3 {
    Vec3::new(p.x, p.y, 0.0)
}

/// CCW yaw by `angle` (rad) about the local Z axis.
fn yaw_rotation(angle: f32) -> Quat {
    Quat::from_rotation_z(angle)
}

/// Quarter turn about the local X axis (local +Y onto local +Z).
fn x_quarter_turn() -> Quat {
    Quat::from_rotation_x(FRAC_PI_2)
}

/// 2D outline in body-local plane coordinates (u, v).
#[derive(Debug, Clone, PartialEq)]
pub enum PlanarShape {
    /// Rectangle → flat box `Shape::Box { half_extents: (hx, hy, thickness/2) }`.
    Rect {
        /// Half extents (u, v) of the rectangle.
        half_extents: Vec2,
    },
    /// Circle → short cylinder `Shape::Cylinder { radius, half_height: thickness/2 }`, whose axis
    /// (local +Y) is turned onto the normal by an extra `Quat::from_rotation_x(FRAC_PI_2)`.
    Circle {
        /// Radius of the circle.
        radius: f32,
    },
    /// Convex polygon (either winding, 3..=MAX_POLYGON_VERTICES vertices) → prism hull with
    /// vertices (x, y, ±thickness/2). Vertices are recentered on the area centroid, so the
    /// body position is the polygon centroid.
    Polygon {
        /// Outline vertices in (u, v), either winding.
        vertices: Vec<Vec2>,
    },
}

/// 2D pose: position in plane coordinates (the shape center or centroid) plus angle about the normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarPose {
    /// Center (Rect/Circle) or area centroid (Polygon) in (u, v).
    pub position: Vec2,
    /// Angle about the normal (rad, CCW).
    pub angle: f32,
}

/// Half of the planar depth: the half-extent of a 2D shape along the normal.
fn half_thickness(thickness: f32) -> f32 {
    thickness / 2.0
}

/// Flat-box half extents of a rectangle: `(hx, hy, thickness/2)`.
fn rect_half_extents(half_extents: Vec2, thickness: f32) -> Vec3 {
    Vec3::new(half_extents.x, half_extents.y, half_thickness(thickness))
}

/// Cylinder-frame offset that turns the engine's local +Y cylinder axis onto
/// the plane normal: `frame * Quat::from_rotation_x(FRAC_PI_2)`.
fn cylinder_axis_to_normal(frame: Quat) -> Quat {
    frame * x_quarter_turn()
}

/// Shared scalar input gate: thickness, mass and pose finiteness and sign.
fn check_common_inputs(pose: PlanarPose, thickness: f32, mass: f32) -> Result<(), PlanarError> {
    if !thickness.is_finite() || thickness <= 0.0 {
        return Err(PlanarError::BadShape {
            detail: "thickness must be finite and positive",
        });
    }
    if !mass.is_finite() || mass < 0.0 {
        return Err(PlanarError::BadShape {
            detail: "mass must be finite and non-negative",
        });
    }
    if !pose.position.is_finite() || !pose.angle.is_finite() {
        return Err(PlanarError::BadShape {
            detail: "pose must be finite",
        });
    }
    Ok(())
}

/// Per-shape extent gate: rectangle half extents and circle radius must be
/// finite and positive (the polygon outline is checked separately).
fn check_shape_extent(shape: &PlanarShape) -> Result<(), PlanarError> {
    match shape {
        PlanarShape::Rect { half_extents } => {
            if !half_extents.is_finite() || half_extents.x <= 0.0 || half_extents.y <= 0.0 {
                return Err(PlanarError::BadShape {
                    detail: "half extents must be finite and positive",
                });
            }
        }
        PlanarShape::Circle { radius } => {
            if !radius.is_finite() || *radius <= 0.0 {
                return Err(PlanarError::BadShape {
                    detail: "radius must be finite and positive",
                });
            }
        }
        PlanarShape::Polygon { .. } => {}
    }
    Ok(())
}

/// Validate a 2D outline (count, finiteness, convexity by 2D-cross signs —
/// either winding, non-zero area) and return its area centroid.
fn polygon_centroid(vertices: &[Vec2]) -> Result<Vec2, PlanarError> {
    let n = vertices.len();
    if !(MIN_POLYGON_VERTICES..=MAX_POLYGON_VERTICES).contains(&n) {
        return Err(PlanarError::BadPolygon {
            detail: "vertex count must be in 3..=MAX_POLYGON_VERTICES",
        });
    }
    if vertices.iter().any(|v| !v.is_finite()) {
        return Err(PlanarError::BadPolygon {
            detail: "non-finite vertex",
        });
    }
    let mut turn_sign = 0i8;
    let mut area2 = 0.0;
    let mut moment = Vec2::ZERO;
    for i in 0..n {
        let p = vertices[i];
        let q = vertices[(i + 1) % n];
        let r = vertices[(i + 2) % n];
        let cross = (q.x - p.x) * (r.y - q.y) - (q.y - p.y) * (r.x - q.x);
        if cross != 0.0 {
            let sign = if cross > 0.0 { 1i8 } else { -1i8 };
            if turn_sign == 0 {
                turn_sign = sign;
            } else if turn_sign != sign {
                return Err(PlanarError::BadPolygon {
                    detail: "outline is not convex",
                });
            }
        }
        let wedge = p.x * q.y - q.x * p.y;
        area2 += wedge;
        moment += (p + q) * wedge;
    }
    if area2.abs() < POLYGON_AREA_EPS {
        return Err(PlanarError::BadPolygon {
            detail: "zero area",
        });
    }
    Ok(moment / (CENTROID_MOMENT_SCALE * area2))
}

/// Prism vertices `(x - cx, y - cy, ±thickness/2)` of a recentered outline.
fn prism_vertices(outline: &[Vec2], centroid: Vec2, thickness: f32) -> Vec<Vec3> {
    let half = half_thickness(thickness);
    outline
        .iter()
        .flat_map(|v| {
            let x = v.x - centroid.x;
            let y = v.y - centroid.y;
            [Vec3::new(x, y, half), Vec3::new(x, y, -half)]
        })
        .collect()
}

/// Thin 3D body of a rectangle outline (materials of [`RigidBody::new_box`]).
fn rect_body(
    plane: &MotionPlane,
    half_extents: Vec2,
    pose: PlanarPose,
    thickness: f32,
    mass: f32,
) -> RigidBody {
    RigidBody::new_box(
        plane.to_world(pose.position),
        rect_half_extents(half_extents, thickness),
        mass,
    )
    .with_orientation(plane.orientation(pose.angle))
}

/// Thin 3D body of a circle outline (materials of [`RigidBody::new_cylinder`],
/// axis turned onto the normal).
fn circle_body(
    plane: &MotionPlane,
    radius: f32,
    pose: PlanarPose,
    thickness: f32,
    mass: f32,
) -> RigidBody {
    RigidBody::new_cylinder(
        plane.to_world(pose.position),
        radius,
        half_thickness(thickness),
        mass,
    )
    .with_orientation(cylinder_axis_to_normal(plane.orientation(pose.angle)))
}

/// Thin 3D prism body of a polygon outline (hull materials of
/// [`RigidBody::try_new_convex_hull`]).
fn polygon_body(
    plane: &MotionPlane,
    outline: &[Vec2],
    pose: PlanarPose,
    thickness: f32,
    mass: f32,
) -> Result<RigidBody, PlanarError> {
    let centroid = polygon_centroid(outline)?;
    let local = prism_vertices(outline, centroid, thickness);
    let body = RigidBody::try_new_convex_hull(plane.to_world(pose.position), local, mass)?;
    Ok(body.with_orientation(plane.orientation(pose.angle)))
}

/// Build a thin 3D body for a 2D shape: position `plane.to_world(pose.position)`, orientation
/// `plane.orientation(pose.angle)` (plus the `Rx(+90°)` offset for Circle). Materials match the
/// underlying constructors: `new_box` / `new_cylinder` / `try_new_convex_hull` (with hull
/// rolling/torsion defaults). `mass == 0` gives a static 2D body (level geometry); a static body
/// needs no plane lock.
///
/// # Errors
///
/// [`PlanarError::BadShape`] for non-finite or ≤ 0 thickness, half extents or radius, non-finite
/// or negative mass, or non-finite pose; [`PlanarError::BadPolygon`] for < 3 or
/// \> MAX_POLYGON_VERTICES vertices, non-finite vertices, a non-convex polygon or zero area;
/// [`PlanarError::Mesh`] passes through `try_new_convex_hull` errors.
pub fn planar_body(
    plane: &MotionPlane,
    shape: &PlanarShape,
    pose: PlanarPose,
    thickness: f32,
    mass: f32,
) -> Result<RigidBody, PlanarError> {
    check_common_inputs(pose, thickness, mass)?;
    check_shape_extent(shape)?;
    match shape {
        PlanarShape::Rect { half_extents } => {
            Ok(rect_body(plane, *half_extents, pose, thickness, mass))
        }
        PlanarShape::Circle { radius } => Ok(circle_body(plane, *radius, pose, thickness, mass)),
        PlanarShape::Polygon { vertices } => polygon_body(plane, vertices, pose, thickness, mass),
    }
}

/// Static anchor of a plane inside a concrete engine. One anchor per plane is shared by all of
/// that plane's bodies. The handle follows the engine's swap-remove rule (`remove_body` moves the
/// last body into the freed slot).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneAnchor {
    plane: MotionPlane,
    body: BodyHandle,
}

impl PlaneAnchor {
    /// Plane of this anchor.
    pub fn plane(&self) -> MotionPlane {
        self.plane
    }

    /// Handle of the anchor's static body (e.g. for raycast filtering).
    pub fn body(&self) -> BodyHandle {
        self.body
    }
}

/// Add the plane anchor: `RigidBody::new_sphere(plane.origin(), PLANE_ANCHOR_RADIUS, 0.0)` with
/// orientation `plane.frame()` and `set_collision_filter(0, 0)`. Raycast does not filter by layer:
/// a ray through the plane origin can hit the 1 mm anchor (documented; filter by `body()`).
pub fn add_plane_anchor<E: PhysicsEngine + ?Sized>(
    engine: &mut E,
    plane: MotionPlane,
) -> PlaneAnchor {
    let mut body = RigidBody::new_sphere(plane.origin(), PLANE_ANCHOR_RADIUS, 0.0)
        .with_orientation(plane.frame());
    body.set_collision_filter(0, 0);
    let handle = engine.add_body(body);
    PlaneAnchor {
        plane,
        body: handle,
    }
}

/// Planar lock in the anchor frame (A = anchor, B = body):
/// `SixDof { local_anchor_a: Vec3::ZERO, local_anchor_b: Vec3::ZERO,
///           linear: [Free, Free, Locked], angular: [Locked, Locked, Free] }`.
pub const fn planar_joint_kind() -> JointKind {
    JointKind::SixDof {
        local_anchor_a: Vec3::ZERO,
        local_anchor_b: Vec3::ZERO,
        linear: [AxisConfig::Free, AxisConfig::Free, AxisConfig::Locked],
        angular: [AxisConfig::Locked, AxisConfig::Locked, AxisConfig::Free],
    }
}

/// First snap of a body onto the plane: project the position onto the
/// plane, drop the normal velocity component and reduce the angular
/// velocity to its normal component. Only dynamic bodies take the plane
/// lock.
fn snap_to_plane(
    plane: &MotionPlane,
    body: &mut RigidBody,
    handle: BodyHandle,
) -> Result<(), PlanarError> {
    if body.body_type != BodyType::Dynamic {
        return Err(PlanarError::NotDynamic {
            handle: handle.index(),
        });
    }
    let normal = plane.normal();
    let depth = plane.signed_distance(body.position);
    body.position -= normal * depth;
    body.velocity -= normal * body.velocity.dot(normal);
    body.angular_velocity = normal * body.angular_velocity.dot(normal);
    Ok(())
}

/// Put a body in «plane» mode. First snap: position projected onto the plane, the normal velocity
/// component removed, angular velocity reduced to its normal component. Then add the
/// `planar_joint_kind()` joint between `anchor.body()` and `body`. Orientation is not corrected:
/// the tilt at call time becomes the lock's zero, so bodies from `planar_body` are planar
/// already. Supported solver: SequentialImpulse (directly or `Engine` with `RoutingKind::Single`).
/// AVBD accepts the joint but is a known limit (see tests); XPBD rejects it; `Islands` routing can
/// move an island to AVBD, which is not supported in K1.
///
/// # Errors
///
/// [`PlanarError::InvalidBody`] for an invalid handle; [`PlanarError::NotDynamic`] for a non-dynamic
/// body; [`PlanarError::Joint`] passes through `add_joint` errors (on XPBD — `JointError::Unsupported`).
pub fn constrain_to_plane<E: PhysicsEngine + ?Sized>(
    engine: &mut E,
    anchor: PlaneAnchor,
    body: BodyHandle,
) -> Result<JointHandle, PlanarError> {
    let plane = anchor.plane();
    let target = engine.get_body_mut(body).ok_or(PlanarError::InvalidBody {
        handle: body.index(),
    })?;
    snap_to_plane(&plane, target, body)?;
    Ok(engine.add_joint(anchor.body(), body, planar_joint_kind())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    const CUSTOM_ORIGIN: Vec3 = Vec3::new(1.0, 2.0, 3.0);

    fn area_centroid_or_panic(vertices: &[Vec2]) -> Vec2 {
        polygon_centroid(vertices).expect("valid outline")
    }

    #[test]
    fn motion_plane_xy_roundtrip() {
        let plane = MotionPlane::XY;
        assert_eq!(plane.origin(), Vec3::ZERO);
        assert_eq!(plane.frame(), Quat::IDENTITY);
        assert_eq!(plane.normal(), Vec3::Z);
        for p in [
            Vec2::new(1.5, -2.0),
            Vec2::ZERO,
            Vec2::new(-0.25, 4.0),
            Vec2::new(0.0, 0.0),
        ] {
            let w = plane.to_world(p);
            assert!((plane.to_plane(w) - p).length() < 1e-6);
            assert!(plane.signed_distance(w).abs() < 1e-6);
        }
    }

    #[test]
    fn motion_plane_custom_normal_roundtrip() {
        // Includes the antiparallel -Z normal: `from_rotation_arc` handles it.
        for n in [
            Vec3::X,
            Vec3::NEG_Z,
            Vec3::Y,
            Vec3::new(0.3, -0.5, 0.8).normalize(),
        ] {
            let normal = UnitVec3::try_from(n).expect("unit normal");
            let plane = MotionPlane::new(CUSTOM_ORIGIN, normal);
            assert!((plane.normal() - n).length() < 1e-5);
            assert!((plane.normal().length() - 1.0).abs() < 1e-6);
            for p in [Vec2::new(0.5, -1.5), Vec2::new(-3.0, 2.0)] {
                let w = plane.to_world(p);
                assert!((plane.to_plane(w) - p).length() < 1e-5);
                assert!(plane.signed_distance(w).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn planar_body_rect_and_circle_frames() {
        let plane = MotionPlane::new(
            CUSTOM_ORIGIN,
            UnitVec3::try_from(Vec3::new(0.2, -0.3, 0.9).normalize()).expect("unit normal"),
        );
        let n = plane.normal();
        let pose = PlanarPose {
            position: Vec2::new(1.0, -2.0),
            angle: 0.4,
        };
        let rect = planar_body(
            &plane,
            &PlanarShape::Rect {
                half_extents: Vec2::new(0.5, 0.3),
            },
            pose,
            PLANAR_THICKNESS,
            1.0,
        )
        .expect("rect builds");
        assert!((rect.orientation * Vec3::Z - n).length() < 1e-5);
        assert!(plane.signed_distance(rect.position).abs() < 1e-6);
        assert!((rect.position - plane.to_world(pose.position)).length() < 1e-6);
        let circle = planar_body(
            &plane,
            &PlanarShape::Circle { radius: 0.3 },
            pose,
            PLANAR_THICKNESS,
            1.0,
        )
        .expect("circle builds");
        assert!((circle.orientation * Vec3::Y - n).length() < 1e-5);
        assert!(plane.signed_distance(circle.position).abs() < 1e-6);
        assert!((circle.position - plane.to_world(pose.position)).length() < 1e-6);
    }

    #[test]
    fn planar_polygon_is_recentered_on_centroid() {
        let plane = MotionPlane::XY;
        // Asymmetric convex quad: its centroid is not at the vertex average.
        let outline = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.5),
            Vec2::new(1.5, 2.0),
            Vec2::new(-0.5, 1.0),
        ];
        let pose = PlanarPose {
            position: Vec2::new(0.7, -1.3),
            angle: 0.0,
        };
        let body = planar_body(
            &plane,
            &PlanarShape::Polygon {
                vertices: outline.clone(),
            },
            pose,
            PLANAR_THICKNESS,
            1.0,
        )
        .expect("polygon builds");
        assert!((body.position - plane.to_world(pose.position)).length() < 1e-6);
        let Shape::ConvexHull(hull) = &body.shape else {
            panic!("polygon outline must build a convex hull");
        };
        let top: Vec<Vec2> = hull
            .vertices()
            .iter()
            .filter(|v| v.z > 0.0)
            .map(|v| Vec2::new(v.x, v.y))
            .collect();
        assert_eq!(top.len(), outline.len());
        let centroid = area_centroid_or_panic(&top);
        assert!(
            centroid.length() < 1e-5,
            "hull outline must be centered on the area centroid, got {centroid:?}"
        );
    }

    #[test]
    fn planar_body_rejects_bad_input() {
        let plane = MotionPlane::XY;
        let pose = PlanarPose {
            position: Vec2::ZERO,
            angle: 0.0,
        };
        let rect = PlanarShape::Rect {
            half_extents: Vec2::ONE,
        };
        for thickness in [0.0, f32::NAN, -1.0] {
            assert!(matches!(
                planar_body(&plane, &rect, pose, thickness, 1.0),
                Err(PlanarError::BadShape { .. })
            ));
        }
        assert!(matches!(
            planar_body(
                &plane,
                &PlanarShape::Circle { radius: 0.0 },
                pose,
                PLANAR_THICKNESS,
                1.0
            ),
            Err(PlanarError::BadShape { .. })
        ));
        assert!(matches!(
            planar_body(&plane, &rect, pose, PLANAR_THICKNESS, f32::NAN),
            Err(PlanarError::BadShape { .. })
        ));
        let two = PlanarShape::Polygon {
            vertices: vec![Vec2::ZERO, Vec2::X],
        };
        assert!(matches!(
            planar_body(&plane, &two, pose, PLANAR_THICKNESS, 1.0),
            Err(PlanarError::BadPolygon { .. })
        ));
        let too_many = PlanarShape::Polygon {
            vertices: (0..MAX_POLYGON_VERTICES + 1)
                .map(|i| {
                    let a = FRAC_PI_2 * i as f32;
                    Vec2::new(a.cos(), a.sin())
                })
                .collect(),
        };
        assert!(matches!(
            planar_body(&plane, &too_many, pose, PLANAR_THICKNESS, 1.0),
            Err(PlanarError::BadPolygon { .. })
        ));
        let dent = PlanarShape::Polygon {
            vertices: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(2.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(2.0, 2.0),
                Vec2::new(0.0, 2.0),
            ],
        };
        assert!(matches!(
            planar_body(&plane, &dent, pose, PLANAR_THICKNESS, 1.0),
            Err(PlanarError::BadPolygon { .. })
        ));
        let flat = PlanarShape::Polygon {
            vertices: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(2.0, 0.0),
            ],
        };
        assert!(matches!(
            planar_body(&plane, &flat, pose, PLANAR_THICKNESS, 1.0),
            Err(PlanarError::BadPolygon { .. })
        ));
    }

    #[test]
    fn planar_joint_kind_is_plane_lock() {
        match planar_joint_kind() {
            JointKind::SixDof {
                local_anchor_a,
                local_anchor_b,
                linear,
                angular,
            } => {
                assert_eq!(local_anchor_a, Vec3::ZERO);
                assert_eq!(local_anchor_b, Vec3::ZERO);
                assert_eq!(
                    linear,
                    [AxisConfig::Free, AxisConfig::Free, AxisConfig::Locked]
                );
                assert_eq!(
                    angular,
                    [AxisConfig::Locked, AxisConfig::Locked, AxisConfig::Free]
                );
            }
            other => panic!("expected a six-DOF joint, got {other:?}"),
        }
    }
}
