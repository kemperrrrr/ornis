//! Rapier-style read-only scene queries: ray / point / AABB / shape casts.
//!
//! [`QueryPipeline`] is the game's read-only window into the physics scene:
//! closest-ray hits with normals, all-hits ray intersections, point and AABB
//! overlap lists, closest-point projection with a shape feature id, linear
//! shape sweeps and shape-overlap tests. Every method takes a body snapshot
//! (`&[RigidBody]`) plus a [`QueryFilter`] and returns without touching the
//! engine: no state is mutated, nothing sleeps or wakes, solver behavior is
//! bit-identical before and after any query.
//!
//! Scope (P1): the built-in [`Shape`](crate::shape::Shape) variants (plus
//! the P5 compound/rounded/half-space shapes, evaluated by the same
//! per-shape kernels),
//! evaluated by exact per-shape kernels shared with the solver (the
//! single-body raycast, `Shape::closest_point`, the analytic pairwise
//! distance and the conservative-advancement sweep). Unsupported pairs
//! (terrain-vs-terrain, mesh-vs-mesh) report separation, so they never
//! produce cast hits or shape overlaps — same standing as the solver.
//! Motion (CCD) filters stay in
//! [`hooks`](crate::sequential_impulse::hooks): this pipeline never
//! filters contacts, it only reads poses.
//!
//! Determinism: bodies are visited in handle order; [`QueryPipeline::cast_ray`]
//! keeps the first minimum (ties resolve to the lowest handle) and
//! [`QueryPipeline::intersect_ray`] is stable-sorted by distance, so ties
//! keep handle order too. The pipeline is stateless (no BVH cache yet);
//! per-query cost is linear in the body count — broadphase acceleration is
//! an explicit follow-up, as is a [`PhysicsEngine`](crate::engine::PhysicsEngine)
//! trait-level surface (today the binding lives on
//! [`SequentialImpulseEngine`](crate::sequential_impulse::SequentialImpulseEngine),
//! the main simulation path, via the `query_*` methods below).

use glam::{Quat, Vec2, Vec3};

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::broadphase_tree::DynamicAabbTree;
use crate::constants::{NEAR_ZERO, SHAPE_TOUCH};
use crate::distance::{ShapeRef, cast_shape as swept_cast, shape_distance};
use crate::errors::{QueryError, check_ray_input};
use crate::math::{AABB, Ray, RaycastHit};
use crate::sequential_impulse::{SequentialImpulseEngine, raycast_body_hit, raycast_shape_hit};
use crate::shape::{ConvexHull, Heightfield, Pose, Shape, TriMesh};

/// Boundary tolerance (m) for point/shape containment tests.
const CONTAIN_EPS: f32 = 1e-4;
/// Ray-t floor (m) for the triangle-mesh parity walk: hits at or below this
/// are the query point's own surface, not a crossing.
const PARITY_EPS: f32 = 1e-4;
/// Fixed oblique local direction for the mesh parity walk: non-axis-aligned
/// so axis-placed soup rarely crosses exactly on a shared triangle edge
/// (which would double-count one crossing).
const PARITY_DIR: Vec3 = Vec3::new(1.0, 0.37, 0.73);
/// Heightfield column treated as flat when the height span is below this (m).
const HF_FLAT_EPS: f32 = 1e-4;
/// Floor for the heightfield skirt depth (m).
const HF_MIN_CELL: f32 = 1e-3;

/// Per-body query veto: return `false` to exclude the body from the hit
/// list. Runs after every other [`QueryFilter`] switch, so it sees only
/// bodies that already passed the type/sensor/layer tests.
pub type QueryPredicate<'a> = &'a dyn Fn(BodyHandle, &RigidBody) -> bool;

/// Which bodies a query may hit, in the spirit of Rapier's `QueryFilter`.
///
/// The body-type and sensor/solid vetoes are independent switches; the
/// layer/mask pair is the mutual test from [`RigidBody::can_collide_with`]
/// against the *query's* membership (`layer`) and interest (`mask`): a body
/// passes only when `mask & body.layer != 0` and
/// `body.mask & layer != 0`. The default hits every body with a non-zero
/// layer and mask. `predicate` is the final escape hatch (for example,
/// excluding one handle) and runs last.
#[derive(Clone, Copy)]
pub struct QueryFilter<'a> {
    /// Skip [`BodyType::Static`] bodies when true.
    pub exclude_fixed: bool,
    /// Skip [`BodyType::Kinematic`] bodies when true.
    pub exclude_kinematic: bool,
    /// Skip [`BodyType::Dynamic`] bodies when true.
    pub exclude_dynamic: bool,
    /// Skip sensor bodies (`is_trigger`) when true.
    pub exclude_sensors: bool,
    /// Skip solid bodies (`!is_trigger`) when true.
    pub exclude_solids: bool,
    /// Query membership layer, tested against each body's `collision_mask`.
    pub layer: u32,
    /// Query interest mask, tested against each body's `collision_layer`.
    pub mask: u32,
    /// Extra per-body veto; runs after every other test.
    pub predicate: Option<QueryPredicate<'a>>,
}

impl<'a> std::fmt::Debug for QueryFilter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryFilter")
            .field("exclude_fixed", &self.exclude_fixed)
            .field("exclude_kinematic", &self.exclude_kinematic)
            .field("exclude_dynamic", &self.exclude_dynamic)
            .field("exclude_sensors", &self.exclude_sensors)
            .field("exclude_solids", &self.exclude_solids)
            .field("layer", &self.layer)
            .field("mask", &self.mask)
            .field("predicate", &self.predicate.map(|_| "..."))
            .finish()
    }
}

impl<'a> Default for QueryFilter<'a> {
    /// Permissive filter: every body type, sensors and solids, all layers.
    fn default() -> Self {
        Self {
            exclude_fixed: false,
            exclude_kinematic: false,
            exclude_dynamic: false,
            exclude_sensors: false,
            exclude_solids: false,
            layer: u32::MAX,
            mask: u32::MAX,
            predicate: None,
        }
    }
}

impl<'a> QueryFilter<'a> {
    /// Whether `body` under `handle` passes every switch of this filter.
    pub fn matches(&self, handle: BodyHandle, body: &RigidBody) -> bool {
        match body.body_type {
            BodyType::Static if self.exclude_fixed => return false,
            BodyType::Kinematic if self.exclude_kinematic => return false,
            BodyType::Dynamic if self.exclude_dynamic => return false,
            _ => {}
        }
        if self.exclude_sensors && body.is_trigger {
            return false;
        }
        if self.exclude_solids && !body.is_trigger {
            return false;
        }
        if self.mask & body.collision_layer == 0 {
            return false;
        }
        if body.collision_mask & self.layer == 0 {
            return false;
        }
        if self.predicate.is_some_and(|keep| !keep(handle, body)) {
            return false;
        }
        true
    }
}

/// Closest surface projection of a query point onto one body.
///
/// Returned by [`QueryPipeline::project_point`]: `point` is the nearest
/// surface point in world space, `is_inside` tells whether the query point
/// was inside the solid (boundary counts as inside, like [`AABB`)).
/// `feature_id` names the winning shape feature — sphere `0`; box face
/// `0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z`; capsule `0` wall `1` bottom cap `2` top
/// cap; cylinder `0` wall `1` top cap `2` bottom cap; cone `0` wall `1` base
/// `2` apex; hull face index into [`ConvexHull::faces`]; heightfield home
/// column `row * cols + col`; mesh triangle ordinal into [`TriMesh::tris`].
/// Degraded shapes (empty hull faces, invalid heightfield, empty mesh)
/// report [`PointProjection::UNKNOWN_FEATURE`].
#[derive(Debug, Clone, Copy)]
pub struct PointProjection {
    /// Body the projected point lies on.
    pub handle: BodyHandle,
    /// Closest surface point in world space.
    pub point: Vec3,
    /// Whether the query point was inside the solid.
    pub is_inside: bool,
    /// Winning shape feature id (see the type docs).
    pub feature_id: u32,
}

impl PointProjection {
    /// Sentinel [`PointProjection::feature_id`] for shapes that cannot name
    /// a feature (empty hull faces, invalid heightfield, empty mesh).
    pub const UNKNOWN_FEATURE: u32 = u32::MAX;
}

/// Stateless read-only query pipeline over a body snapshot.
///
/// All methods take `&self` plus `bodies: &[RigidBody]` (the engine
/// bindings pass their live tables by shared reference) and never mutate
/// anything: poses, sleep state and warm caches are untouched, and no body
/// is woken. See the module docs for scope and determinism.
#[derive(Debug, Clone, Copy, Default)]
pub struct QueryPipeline {}

impl QueryPipeline {
    /// Empty pipeline (stateless: construction allocates nothing).
    pub fn new() -> Self {
        Self::default()
    }

    /// Closest ray hit against the bodies passing `filter`, with the
    /// surface normal (see [`RaycastHit`]). `Ok(None)` is a clean miss.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidInput`] for a non-finite/zero direction or a
    /// bad `max_dist` (same gate as [`PhysicsEngine::raycast`](crate::engine::PhysicsEngine::raycast)).
    pub fn cast_ray(
        &self,
        bodies: &[RigidBody],
        ray: &Ray,
        max_dist: f32,
        filter: &QueryFilter,
    ) -> Result<Option<RaycastHit>, QueryError> {
        check_ray_input(ray.origin, ray.direction, max_dist)?;
        let mut best: Option<RaycastHit> = None;
        for (index, body) in bodies.iter().enumerate() {
            let handle = BodyHandle::from(index);
            if !filter.matches(handle, body) {
                continue;
            }
            if let Some(hit) = raycast_body_hit(body, handle, ray, max_dist)
                && best.is_none_or(|known| hit.distance < known.distance)
            {
                best = Some(hit);
            }
        }
        Ok(best)
    }

    /// Every ray hit against the bodies passing `filter`, stable-sorted by
    /// distance (ties keep handle order). Bodies containing the ray origin
    /// report their exit hit, like the per-shape kernels do.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidInput`] for a non-finite/zero direction or a
    /// bad `max_dist` (same gate as [`QueryPipeline::cast_ray`]).
    pub fn intersect_ray(
        &self,
        bodies: &[RigidBody],
        ray: &Ray,
        max_dist: f32,
        filter: &QueryFilter,
    ) -> Result<Vec<RaycastHit>, QueryError> {
        check_ray_input(ray.origin, ray.direction, max_dist)?;
        let mut hits = Vec::new();
        for (index, body) in bodies.iter().enumerate() {
            let handle = BodyHandle::from(index);
            if !filter.matches(handle, body) {
                continue;
            }
            if let Some(hit) = raycast_body_hit(body, handle, ray, max_dist) {
                hits.push(hit);
            }
        }
        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }

    /// Handles of every body passing `filter` whose solid contains `point`
    /// (boundary-inclusive), in handle order. A non-finite point matches
    /// nothing.
    pub fn intersect_point(
        &self,
        bodies: &[RigidBody],
        point: Vec3,
        filter: &QueryFilter,
    ) -> Vec<BodyHandle> {
        if !point.is_finite() {
            return Vec::new();
        }
        bodies
            .iter()
            .enumerate()
            .filter(|(index, body)| {
                let handle = BodyHandle::from(*index);
                filter.matches(handle, body)
                    && shape_contains(&body.shape, body.position, body.orientation, point)
            })
            .map(|(index, _)| BodyHandle::from(index))
            .collect()
    }

    /// Handles of every body passing `filter` whose world AABB overlaps
    /// `query` (the span query: touching faces count), in handle order.
    pub fn intersect_aabb(
        &self,
        bodies: &[RigidBody],
        query: &AABB,
        filter: &QueryFilter,
    ) -> Vec<BodyHandle> {
        bodies
            .iter()
            .enumerate()
            .filter(|(index, body)| {
                let handle = BodyHandle::from(*index);
                filter.matches(handle, body)
                    && body
                        .shape
                        .aabb(body.position, body.orientation)
                        .overlaps(query)
            })
            .map(|(index, _)| BodyHandle::from(index))
            .collect()
    }

    /// Closest surface projection of `point` over the bodies passing
    /// `filter` (see [`PointProjection`]). A non-finite point projects to
    /// nothing.
    pub fn project_point(
        &self,
        bodies: &[RigidBody],
        point: Vec3,
        filter: &QueryFilter,
    ) -> Option<PointProjection> {
        if !point.is_finite() {
            return None;
        }
        let mut best: Option<(f32, PointProjection)> = None;
        for (index, body) in bodies.iter().enumerate() {
            let handle = BodyHandle::from(index);
            if !filter.matches(handle, body) {
                continue;
            }
            let (closest, is_inside, feature_id) =
                project_shape(&body.shape, body.position, body.orientation, point);
            if !closest.is_finite() {
                continue;
            }
            let dist2 = (point - closest).length_squared();
            if !dist2.is_finite() {
                continue;
            }
            if best.is_none_or(|(known, _)| dist2 < known) {
                best = Some((
                    dist2,
                    PointProjection {
                        handle,
                        point: closest,
                        is_inside,
                        feature_id,
                    },
                ));
            }
        }
        best.map(|(_, projection)| projection)
    }

    /// Linear sweep of `shape` from `position`/`orientation` along
    /// `displacement` (conservative advancement over the exact pairwise
    /// distance): the
    /// first body passing `filter` that the sweep touches, with the travel
    /// distance in [`RaycastHit::distance`] and the outward normal. A mover
    /// already touching at t=0 reports no hit for that body (resting
    /// contact is the discrete solver's job). `None` on a degenerate
    /// (zero/non-finite) sweep.
    pub fn cast_shape(
        &self,
        bodies: &[RigidBody],
        shape: &Shape,
        position: Vec3,
        orientation: Quat,
        displacement: Vec3,
        filter: &QueryFilter,
    ) -> Option<RaycastHit> {
        if !position.is_finite() || !orientation.is_finite() || !displacement.is_finite() {
            return None;
        }
        let mover = ShapeRef {
            shape,
            pos: position,
            rot: orientation,
        };
        let targets = bodies.iter().enumerate().filter_map(|(index, body)| {
            let handle = BodyHandle::from(index);
            filter.matches(handle, body).then_some((
                handle,
                ShapeRef {
                    shape: &body.shape,
                    pos: body.position,
                    rot: body.orientation,
                },
            ))
        });
        swept_cast(mover, displacement, targets).map(|hit| RaycastHit {
            handle: hit.handle,
            point: hit.point,
            normal: hit.normal,
            distance: hit.t,
        })
    }

    /// Handles of every body passing `filter` that overlaps `shape` at
    /// `position`/`orientation`, in handle order. Overlap means the exact
    /// surface distance is within the sweep touch band ([`SHAPE_TOUCH`],
    /// 1 mm — the same band [`QueryPipeline::cast_shape`] stops at), so
    /// kissing shapes count as intersecting.
    pub fn intersect_shape(
        &self,
        bodies: &[RigidBody],
        shape: &Shape,
        position: Vec3,
        orientation: Quat,
        filter: &QueryFilter,
    ) -> Vec<BodyHandle> {
        if !position.is_finite() || !orientation.is_finite() {
            return Vec::new();
        }
        let mover = ShapeRef {
            shape,
            pos: position,
            rot: orientation,
        };
        bodies
            .iter()
            .enumerate()
            .filter(|(index, body)| {
                let handle = BodyHandle::from(*index);
                filter.matches(handle, body)
                    && shape_distance(
                        mover,
                        ShapeRef {
                            shape: &body.shape,
                            pos: body.position,
                            rot: body.orientation,
                        },
                    )
                    .dist
                        <= SHAPE_TOUCH
            })
            .map(|(index, _)| BodyHandle::from(index))
            .collect()
    }
}

/// Whether the placed shape's solid contains `point` (boundary-inclusive).
fn shape_contains(shape: &Shape, pos: Vec3, rot: Quat, point: Vec3) -> bool {
    let local = rot.conjugate() * (point - pos);
    match shape {
        Shape::Sphere { radius } => local.length() <= radius + CONTAIN_EPS,
        Shape::Box { half_extents } => {
            local.x.abs() <= half_extents.x + CONTAIN_EPS
                && local.y.abs() <= half_extents.y + CONTAIN_EPS
                && local.z.abs() <= half_extents.z + CONTAIN_EPS
        }
        Shape::Capsule {
            radius,
            half_height,
        } => {
            let core = local.y.clamp(-*half_height, *half_height);
            (local - Vec3::new(0.0, core, 0.0)).length() <= radius + CONTAIN_EPS
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => {
            local.y.abs() <= half_height + CONTAIN_EPS
                && Vec2::new(local.x, local.z).length() <= radius + CONTAIN_EPS
        }
        Shape::Cone {
            radius,
            half_height,
        } => cone_contains(*radius, *half_height, local),
        Shape::ConvexHull(hull) => hull_contains(hull, local),
        Shape::Heightfield(hf) => heightfield_contains(hf, local),
        Shape::TriMesh(mesh) => trimesh_contains(mesh, local),
        // P5 compat (new-shape arms only): compound contains when any child
        // does; rounded shapes add the skin around the inner surface;
        // half-spaces contain the closed below-plane side.
        Shape::Compound { shapes } => shapes.iter().any(|(child, pose)| {
            shape_contains(child, pos + rot * pose.position, pose.world_rot(rot), point)
        }),
        Shape::Round {
            inner,
            border_radius,
        } => {
            shape_contains(inner, pos, rot, point)
                || (inner.closest_point(pos, rot, point) - point).length()
                    <= border_radius + CONTAIN_EPS
        }
        Shape::HalfSpace { normal } => local.dot(normal.get()) <= CONTAIN_EPS,
    }
}

/// Cone containment (local frame: apex `+half_height`, base at `-half_height`).
fn cone_contains(radius: f32, half_height: f32, local: Vec3) -> bool {
    if half_height <= NEAR_ZERO {
        return local.y.abs() <= CONTAIN_EPS
            && Vec2::new(local.x, local.z).length() <= radius + CONTAIN_EPS;
    }
    if local.y < -half_height - CONTAIN_EPS || local.y > half_height + CONTAIN_EPS {
        return false;
    }
    let wall_at = radius * (half_height - local.y.min(half_height)) / (2.0 * half_height);
    Vec2::new(local.x, local.z).length() <= wall_at + CONTAIN_EPS
}

/// Convex-hull containment (local frame): inside every outward face
/// half-space. Over-cap hulls ship no faces — conservative AABB fallback
/// (documented in [`PointProjection`]).
fn hull_contains(hull: &ConvexHull, local: Vec3) -> bool {
    if hull.faces.is_empty() {
        if hull.vertices.is_empty() {
            return false;
        }
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for v in &hull.vertices {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        return local.x >= lo.x - CONTAIN_EPS
            && local.x <= hi.x + CONTAIN_EPS
            && local.y >= lo.y - CONTAIN_EPS
            && local.y <= hi.y + CONTAIN_EPS
            && local.z >= lo.z - CONTAIN_EPS
            && local.z <= hi.z + CONTAIN_EPS;
    }
    for face in &hull.faces {
        let (a, b, c) = (
            hull.vertices[face.0.index()],
            hull.vertices[face.1.index()],
            hull.vertices[face.2.index()],
        );
        let normal = (b - a).cross(c - a);
        if normal.length_squared() < 1e-12 {
            continue;
        }
        if (local - a).dot(normal) > CONTAIN_EPS * normal.length() {
            return false;
        }
    }
    true
}

/// Heightfield containment (local frame): inside the home column's solid
/// box (same columns as `Shape::closest_point`).
fn heightfield_contains(hf: &Heightfield, local: Vec3) -> bool {
    if !hf.is_valid() {
        return false;
    }
    let (rows, cols, cell) = (hf.rows(), hf.cols(), hf.cell());
    let col_f = (local.x / cell + (cols - 1) as f32 * 0.5).floor();
    let row_f = (local.z / cell + (rows - 1) as f32 * 0.5).floor();
    if col_f < 0.0 || col_f >= cols as f32 || row_f < 0.0 || row_f >= rows as f32 {
        return false;
    }
    let (col, row) = (col_f as usize, row_f as usize);
    let h = hf.heights()[row * cols + col];
    let (y_min, _) = hf.height_range();
    let y_low = if h - y_min >= HF_FLAT_EPS {
        y_min
    } else {
        h - cell.max(HF_MIN_CELL)
    };
    let x_origin = -((cols - 1) as f32) * 0.5 * cell;
    let z_origin = -((rows - 1) as f32) * 0.5 * cell;
    let x0 = x_origin + col as f32 * cell;
    let z0 = z_origin + row as f32 * cell;
    local.x >= x0 - CONTAIN_EPS
        && local.x <= x0 + cell + CONTAIN_EPS
        && local.z >= z0 - CONTAIN_EPS
        && local.z <= z0 + cell + CONTAIN_EPS
        && local.y >= y_low.min(h) - CONTAIN_EPS
        && local.y <= y_low.max(h) + CONTAIN_EPS
}

/// Mesh containment (mesh-local frame): on-surface counts as inside, else a
/// ray-parity walk along [`PARITY_DIR`] (odd crossings = inside). Exact for
/// closed soup; planar/open soup reads as outside on both sides.
fn trimesh_contains(mesh: &TriMesh, local: Vec3) -> bool {
    if mesh.tris().is_empty() {
        return false;
    }
    if trimesh_closest2(mesh, local) <= CONTAIN_EPS * CONTAIN_EPS {
        return true;
    }
    let mut crossings = 0u32;
    for (tri, center) in mesh.tris().iter().zip(mesh.centroids().iter()) {
        if !matches!(tri, Shape::ConvexHull(_)) {
            continue;
        }
        if let Some((t, _)) = raycast_body_hit_triangle(tri, *center, local)
            && t > PARITY_EPS
        {
            crossings += 1;
        }
    }
    crossings % 2 == 1
}

/// Ray/triangle-primitive hit in mesh-local space: the triangle's verts are
/// centroid-relative, so the ray is shifted by the centroid first.
fn raycast_body_hit_triangle(tri: &Shape, center: Vec3, origin: Vec3) -> Option<(f32, Vec3)> {
    raycast_shape_hit(tri, origin - center, PARITY_DIR, f32::INFINITY)
}

/// Squared mesh-local distance from `p` to the soup (infinity when no face
/// exists at all).
fn trimesh_closest2(mesh: &TriMesh, p: Vec3) -> f32 {
    let mut best = f32::INFINITY;
    for (tri, center) in mesh.tris().iter().zip(mesh.centroids().iter()) {
        let Shape::ConvexHull(hull) = tri else {
            continue;
        };
        for face in &hull.faces {
            let (a, b, c) = (
                hull.vertices[face.0.index()] + *center,
                hull.vertices[face.1.index()] + *center,
                hull.vertices[face.2.index()] + *center,
            );
            let (q, _, _, _) = crate::gjk::closest_triangle(a - p, b - p, c - p);
            best = best.min(q.length_squared());
        }
    }
    best
}

/// Closest surface point, containment flag and winning feature id for one
/// placed shape (all in world space on return; feature ids are documented
/// on [`PointProjection`]).
fn project_shape(shape: &Shape, pos: Vec3, rot: Quat, point: Vec3) -> (Vec3, bool, u32) {
    let closest = shape.closest_point(pos, rot, point);
    let inside = shape_contains(shape, pos, rot, point);
    let local_p = rot.conjugate() * (point - pos);
    let local_q = rot.conjugate() * (closest - pos);
    (closest, inside, shape_feature(shape, local_p, local_q))
}

/// Winning feature id for a closest-point pair (local frame).
fn shape_feature(shape: &Shape, local_p: Vec3, local_q: Vec3) -> u32 {
    match shape {
        Shape::Sphere { .. } => 0,
        Shape::Box { half_extents } => box_face_id(*half_extents, local_q),
        Shape::Capsule { half_height, .. } => {
            if local_p.y < -*half_height {
                1
            } else if local_p.y > *half_height {
                2
            } else {
                0
            }
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => cylinder_feature(*radius, *half_height, local_p),
        Shape::Cone {
            radius,
            half_height,
        } => cone_feature(*radius, *half_height, local_p),
        Shape::ConvexHull(hull) => hull_feature(hull, local_p),
        Shape::Heightfield(hf) => heightfield_feature(hf, local_p),
        Shape::TriMesh(mesh) => trimesh_feature(mesh, local_p),
        // P5 compat (new-shape arms only): compounds name the winning child
        // ordinal; rounded shapes name the inner feature; the plane is a
        // single feature.
        Shape::Compound { shapes } => compound_feature(shapes, local_p),
        Shape::Round { inner, .. } => shape_feature(inner, local_p, local_q),
        Shape::HalfSpace { .. } => 0,
    }
}

/// Box face ordinal for a projected point: `0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z`.
/// The face owns the axis with the largest point-to-half-extent ratio;
/// ties resolve X, then Y, then Z (strict `>` keeps the first maximum).
fn box_face_id(half: Vec3, local_q: Vec3) -> u32 {
    let ratio = |q: f32, h: f32| {
        if h > 0.0 { q.abs() / h } else { -1.0 }
    };
    let (nx, ny, nz) = (
        ratio(local_q.x, half.x),
        ratio(local_q.y, half.y),
        ratio(local_q.z, half.z),
    );
    let mut axis = 0u32;
    let mut best = nx;
    if ny > best {
        best = ny;
        axis = 1;
    }
    if nz > best {
        axis = 2;
    }
    let positive = [local_q.x >= 0.0, local_q.y >= 0.0, local_q.z >= 0.0][axis as usize];
    axis * 2 + u32::from(!positive)
}

/// Cylinder feature mirror of `closest_cylinder_point` in `shape.rs`:
/// the same three candidates with the same strict tie order, so the id
/// always names the face the closest point lies on.
fn cylinder_feature(radius: f32, half_height: f32, p: Vec3) -> u32 {
    let radial = Vec2::new(p.x, p.z);
    let len = radial.length();
    let wall = if len > NEAR_ZERO {
        Vec3::new(
            radial.x / len * radius,
            p.y.clamp(-half_height, half_height),
            radial.y / len * radius,
        )
    } else {
        Vec3::new(radius, p.y.clamp(-half_height, half_height), 0.0)
    };
    let cap = |y: f32| {
        let s = if len > NEAR_ZERO {
            (radius / len).min(1.0)
        } else {
            0.0
        };
        Vec3::new(p.x * s, y, p.z * s)
    };
    let (top, bottom) = (cap(half_height), cap(-half_height));
    let (wall_d2, top_d2, bottom_d2) = (
        (p - wall).length_squared(),
        (p - top).length_squared(),
        (p - bottom).length_squared(),
    );
    if wall_d2 < top_d2 && wall_d2 < bottom_d2 {
        0
    } else if top_d2 < bottom_d2 {
        1
    } else {
        2
    }
}

/// Cone feature mirror of `closest_cone_point` in `shape.rs`: wall `0`,
/// base `1`, apex `2`, same strict tie order as the closest point.
fn cone_feature(radius: f32, half_height: f32, p: Vec3) -> u32 {
    if half_height <= NEAR_ZERO {
        return 1;
    }
    let radial = Vec2::new(p.x, p.z);
    let len = radial.length();
    let y_c = p.y.clamp(-half_height, half_height);
    let wall_at = radius * (half_height - y_c) / (2.0 * half_height);
    let wall = if len > NEAR_ZERO {
        Vec3::new(radial.x / len * wall_at, y_c, radial.y / len * wall_at)
    } else {
        Vec3::new(wall_at, y_c, 0.0)
    };
    let s = if len > NEAR_ZERO {
        (radius / len).min(1.0)
    } else {
        0.0
    };
    let base = Vec3::new(p.x * s, -half_height, p.z * s);
    let apex = Vec3::new(0.0, half_height, 0.0);
    let mut feature = 0;
    let mut best = (p - wall).length_squared();
    for (candidate, id) in [(base, 1), (apex, 2)] {
        let d2 = (p - candidate).length_squared();
        if d2 < best {
            best = d2;
            feature = id;
        }
    }
    feature
}

/// Winning hull face index into [`ConvexHull::faces`]
/// ([`PointProjection::UNKNOWN_FEATURE`] when there are no faces).
fn hull_feature(hull: &ConvexHull, local_p: Vec3) -> u32 {
    let mut best = f32::INFINITY;
    let mut feature = PointProjection::UNKNOWN_FEATURE;
    for (index, face) in hull.faces.iter().enumerate() {
        let (a, b, c) = (
            hull.vertices[face.0.index()],
            hull.vertices[face.1.index()],
            hull.vertices[face.2.index()],
        );
        let (q, _, _, _) = crate::gjk::closest_triangle(a - local_p, b - local_p, c - local_p);
        let d2 = q.length_squared();
        if d2 < best {
            best = d2;
            feature = index as u32;
        }
    }
    feature
}

/// Home column `row * cols + col` of a heightfield query
/// ([`PointProjection::UNKNOWN_FEATURE`] for an invalid grid).
fn heightfield_feature(hf: &Heightfield, local_p: Vec3) -> u32 {
    if !hf.is_valid() {
        return PointProjection::UNKNOWN_FEATURE;
    }
    let (rows, cols, cell) = (hf.rows(), hf.cols(), hf.cell());
    let col = ((local_p.x / cell + (cols - 1) as f32 * 0.5).floor() as isize)
        .clamp(0, cols as isize - 1) as usize;
    let row = ((local_p.z / cell + (rows - 1) as f32 * 0.5).floor() as isize)
        .clamp(0, rows as isize - 1) as usize;
    (row * cols + col) as u32
}

/// P5 compat: winning compound child ordinal for a body-frame query
/// point ([`PointProjection::UNKNOWN_FEATURE`] when the union is empty).
fn compound_feature(shapes: &[(Shape, Pose)], local_p: Vec3) -> u32 {
    let mut best = f32::INFINITY;
    let mut feature = PointProjection::UNKNOWN_FEATURE;
    for (index, (child, pose)) in shapes.iter().enumerate() {
        let child_rot = pose.world_rot(Quat::IDENTITY);
        let query = child_rot.inverse() * (local_p - pose.position);
        let q = child.closest_point(Vec3::ZERO, Quat::IDENTITY, query);
        let d2 = (query - q).length_squared();
        if d2 < best {
            best = d2;
            feature = index as u32;
        }
    }
    feature
}

/// Winning mesh triangle ordinal into [`TriMesh::tris`]
/// ([`PointProjection::UNKNOWN_FEATURE`] when the soup is empty).
fn trimesh_feature(mesh: &TriMesh, local_p: Vec3) -> u32 {
    let mut best = f32::INFINITY;
    let mut feature = PointProjection::UNKNOWN_FEATURE;
    for (index, (tri, center)) in mesh.tris().iter().zip(mesh.centroids().iter()).enumerate() {
        let Shape::ConvexHull(hull) = tri else {
            continue;
        };
        for face in &hull.faces {
            let (a, b, c) = (
                hull.vertices[face.0.index()] + *center,
                hull.vertices[face.1.index()] + *center,
                hull.vertices[face.2.index()] + *center,
            );
            let (q, _, _, _) = crate::gjk::closest_triangle(a - local_p, b - local_p, c - local_p);
            let d2 = q.length_squared();
            if d2 < best {
                best = d2;
                feature = index as u32;
            }
        }
    }
    feature
}

/// Read-only query view over the live broadphase tree (R5).
///
/// Binds the *same* [`DynamicAabbTree`](crate::broadphase_tree) the pair
/// pipeline maintains — no second tree is built. Traversal mirrors the
/// Box3D `b3DynamicTree_RayCast` / `b3DynamicTree_BoxCast` / `QueryClosest`
/// family (segment-box prefilter, slab pruning on fat boxes, closest-first
/// walk) but is implemented independently over this engine's node layout;
/// the tree only *prunes*, every hit decision runs the exact per-shape
/// kernels shared with [`QueryPipeline`] over candidates in ascending
/// handle order with the same strict-minimum / handle tie-break rules, so
/// tree results are identical and only the visit order differs.
///
/// Read-only invariant: the view holds `tree: &DynamicAabbTree` (shared
/// borrow — no insert, remove or rebalance can run during a pass) and
/// records [`DynamicAabbTree::generation`] at bind time to name the tree
/// state. Freshness beyond the counter comes from
/// [`DynamicAabbTree::is_fresh_for`]: every current base AABB must still sit
/// inside its proxy fat. [`QueryTreeView::try_bind`] returns `None` on any
/// staleness (bodies added/removed/moved/flipped without a broadphase
/// update, degenerate NaN bounds), and each method falls back to the
/// [`QueryPipeline`] brute-force kernel when its own query region is
/// degenerate — both fallbacks are explicit branches, never silent skips.
#[derive(Clone, Copy)]
pub(crate) struct QueryTreeView<'a> {
    bodies: &'a [RigidBody],
    tree: &'a DynamicAabbTree,
    /// Tree generation seen at bind time (names the tree state for the pass;
    /// the borrow plus the fat-containment probe carry the actual guarantee).
    generation: u64,
}

impl<'a> QueryTreeView<'a> {
    /// Bind a view over `bodies` and the live `tree`. Returns `None` — the
    /// explicit brute-force fallback — when the topology changed since the
    /// last broadphase update or any current base AABB escaped its proxy
    /// fat (see [`DynamicAabbTree::is_fresh_for`]).
    pub(crate) fn try_bind(bodies: &'a [RigidBody], tree: &'a DynamicAabbTree) -> Option<Self> {
        if tree.topology_len() != bodies.len() {
            return None;
        }
        if !tree.is_fresh_for(bodies) {
            return None;
        }
        Some(Self {
            bodies,
            tree,
            generation: tree.generation(),
        })
    }

    /// Pins the no-mutation invariant for a pass: the shared borrow already
    /// forbids any insert/rebalance during traversal, so the generation
    /// recorded at bind time must still hold here (debug builds fail loudly
    /// instead of querying a half-mutated tree).
    fn check_generation(&self) {
        debug_assert_eq!(
            self.tree.generation(),
            self.generation,
            "live broadphase tree mutated during a query pass"
        );
    }

    /// Closest ray hit (same contract as [`QueryPipeline::cast_ray`]).
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidInput`] for a non-finite/zero direction or a
    /// bad `max_dist`.
    pub(crate) fn cast_ray(
        &self,
        ray: &Ray,
        max_dist: f32,
        filter: &QueryFilter,
    ) -> Result<Option<RaycastHit>, QueryError> {
        check_ray_input(ray.origin, ray.direction, max_dist)?;
        self.check_generation();
        let mut ids = Vec::new();
        self.tree
            .query_ray_into(ray.origin, ray.direction, max_dist, &mut ids);
        sort_dedup(&mut ids);
        let mut best: Option<RaycastHit> = None;
        // Shrinking the exact-test limit to the best hit so far is sound:
        // farther hits would lose the strict minimum anyway, and a hit at
        // exactly the limit still reports (inclusive far plane) but keeps
        // the earlier — smaller — handle via the strict comparison.
        let mut limit = max_dist;
        for index in ids {
            let Some(body) = self.bodies.get(index) else {
                continue;
            };
            let handle = BodyHandle::from(index);
            if !filter.matches(handle, body) {
                continue;
            }
            if let Some(hit) = raycast_body_hit(body, handle, ray, limit)
                && best.is_none_or(|known| hit.distance < known.distance)
            {
                limit = hit.distance;
                best = Some(hit);
            }
        }
        Ok(best)
    }

    /// Every ray hit, stable-sorted by distance (same contract as
    /// [`QueryPipeline::intersect_ray`]).
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidInput`] for a non-finite/zero direction or a
    /// bad `max_dist`.
    pub(crate) fn intersect_ray(
        &self,
        ray: &Ray,
        max_dist: f32,
        filter: &QueryFilter,
    ) -> Result<Vec<RaycastHit>, QueryError> {
        check_ray_input(ray.origin, ray.direction, max_dist)?;
        self.check_generation();
        let mut ids = Vec::new();
        self.tree
            .query_ray_into(ray.origin, ray.direction, max_dist, &mut ids);
        sort_dedup(&mut ids);
        let mut hits = Vec::new();
        for index in ids {
            let Some(body) = self.bodies.get(index) else {
                continue;
            };
            let handle = BodyHandle::from(index);
            if !filter.matches(handle, body) {
                continue;
            }
            if let Some(hit) = raycast_body_hit(body, handle, ray, max_dist) {
                hits.push(hit);
            }
        }
        // Brute force stable-sorts by distance over handle-order hits; the
        // explicit handle tie-break below is that order, stated plainly.
        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.handle.index().cmp(&b.handle.index()))
        });
        Ok(hits)
    }

    /// Bodies containing `point` (same contract as
    /// [`QueryPipeline::intersect_point`]).
    pub(crate) fn intersect_point(&self, point: Vec3, filter: &QueryFilter) -> Vec<BodyHandle> {
        if !point.is_finite() {
            return Vec::new();
        }
        self.check_generation();
        let target = AABB::from_point(point);
        if !DynamicAabbTree::aabb_is_queryable(&target) {
            return QueryPipeline::new().intersect_point(self.bodies, point, filter);
        }
        let mut ids = Vec::new();
        self.tree.query_aabb_into(&target, &mut ids);
        sort_dedup(&mut ids);
        ids.into_iter()
            .filter(|index| {
                self.bodies.get(*index).is_some_and(|body| {
                    let handle = BodyHandle::from(*index);
                    filter.matches(handle, body)
                        && shape_contains(&body.shape, body.position, body.orientation, point)
                })
            })
            .map(BodyHandle::from)
            .collect()
    }

    /// Bodies whose AABB overlaps `query` (same contract as
    /// [`QueryPipeline::intersect_aabb`]).
    pub(crate) fn intersect_aabb(&self, query: &AABB, filter: &QueryFilter) -> Vec<BodyHandle> {
        self.check_generation();
        if !DynamicAabbTree::aabb_is_queryable(query) {
            return QueryPipeline::new().intersect_aabb(self.bodies, query, filter);
        }
        let mut ids = Vec::new();
        self.tree.query_aabb_into(query, &mut ids);
        sort_dedup(&mut ids);
        ids.into_iter()
            .filter(|index| {
                self.bodies.get(*index).is_some_and(|body| {
                    let handle = BodyHandle::from(*index);
                    filter.matches(handle, body)
                        && body
                            .shape
                            .aabb(body.position, body.orientation)
                            .overlaps(query)
                })
            })
            .map(BodyHandle::from)
            .collect()
    }

    /// Closest surface projection (same contract as
    /// [`QueryPipeline::project_point`]).
    pub(crate) fn project_point(
        &self,
        point: Vec3,
        filter: &QueryFilter,
    ) -> Option<PointProjection> {
        if !point.is_finite() {
            return None;
        }
        self.check_generation();
        let mut best: Option<(f32, PointProjection)> = None;
        let mut best_handle = usize::MAX;
        let mut best_d2 = f32::INFINITY;
        self.tree
            .query_closest(point, &mut best_d2, |index, current| {
                let Some(body) = self.bodies.get(index) else {
                    return current;
                };
                let handle = BodyHandle::from(index);
                if !filter.matches(handle, body) {
                    return current;
                }
                let (closest, is_inside, feature_id) =
                    project_shape(&body.shape, body.position, body.orientation, point);
                if !closest.is_finite() {
                    return current;
                }
                let dist2 = (point - closest).length_squared();
                if !dist2.is_finite() {
                    return current;
                }
                // Strict minimum in traversal order plus an explicit
                // smaller-handle tie-break: the brute-force first-minimum over
                // handle order, regardless of visit order.
                if dist2 < current || (dist2 == current && index < best_handle) {
                    best = Some((
                        dist2,
                        PointProjection {
                            handle,
                            point: closest,
                            is_inside,
                            feature_id,
                        },
                    ));
                    best_handle = index;
                    dist2
                } else {
                    current
                }
            });
        best.map(|(_, projection)| projection)
    }

    /// Linear shape sweep (same contract as [`QueryPipeline::cast_shape`]).
    pub(crate) fn cast_shape(
        &self,
        shape: &Shape,
        position: Vec3,
        orientation: Quat,
        displacement: Vec3,
        filter: &QueryFilter,
    ) -> Option<RaycastHit> {
        if !position.is_finite() || !orientation.is_finite() || !displacement.is_finite() {
            return None;
        }
        self.check_generation();
        let brute = || {
            QueryPipeline::new().cast_shape(
                self.bodies,
                shape,
                position,
                orientation,
                displacement,
                filter,
            )
        };
        let start = shape.aabb(position, orientation);
        let end = shape.aabb(position + displacement, orientation);
        if !DynamicAabbTree::aabb_is_queryable(&start) || !DynamicAabbTree::aabb_is_queryable(&end)
        {
            return brute();
        }
        // Touch-band expansion: overlap means `dist <= SHAPE_TOUCH`, so
        // bodies resting just under the band would not overlap the raw
        // swept box. The pad keeps the candidate set a superset.
        let pad = Vec3::splat(SHAPE_TOUCH);
        let swept = AABB::new(start.min.min(end.min) - pad, start.max.max(end.max) + pad);
        let mut ids = Vec::new();
        self.tree.query_aabb_into(&swept, &mut ids);
        sort_dedup(&mut ids);
        let mover = ShapeRef {
            shape,
            pos: position,
            rot: orientation,
        };
        let targets = ids.into_iter().filter_map(|index| {
            let body = self.bodies.get(index)?;
            let handle = BodyHandle::from(index);
            filter.matches(handle, body).then_some((
                handle,
                ShapeRef {
                    shape: &body.shape,
                    pos: body.position,
                    rot: body.orientation,
                },
            ))
        });
        swept_cast(mover, displacement, targets).map(|hit| RaycastHit {
            handle: hit.handle,
            point: hit.point,
            normal: hit.normal,
            distance: hit.t,
        })
    }

    /// Bodies overlapping `shape` at the pose (same contract as
    /// [`QueryPipeline::intersect_shape`]).
    pub(crate) fn intersect_shape(
        &self,
        shape: &Shape,
        position: Vec3,
        orientation: Quat,
        filter: &QueryFilter,
    ) -> Vec<BodyHandle> {
        if !position.is_finite() || !orientation.is_finite() {
            return Vec::new();
        }
        self.check_generation();
        let base = shape.aabb(position, orientation);
        if !DynamicAabbTree::aabb_is_queryable(&base) {
            return QueryPipeline::new().intersect_shape(
                self.bodies,
                shape,
                position,
                orientation,
                filter,
            );
        }
        let pad = Vec3::splat(SHAPE_TOUCH);
        let query = AABB::new(base.min - pad, base.max + pad);
        let mut ids = Vec::new();
        self.tree.query_aabb_into(&query, &mut ids);
        sort_dedup(&mut ids);
        let mover = ShapeRef {
            shape,
            pos: position,
            rot: orientation,
        };
        ids.into_iter()
            .filter(|index| {
                self.bodies.get(*index).is_some_and(|body| {
                    let handle = BodyHandle::from(*index);
                    filter.matches(handle, body)
                        && shape_distance(
                            mover,
                            ShapeRef {
                                shape: &body.shape,
                                pos: body.position,
                                rot: body.orientation,
                            },
                        )
                        .dist
                            <= SHAPE_TOUCH
                })
            })
            .map(BodyHandle::from)
            .collect()
    }
}

/// Ascending-handle candidate order with duplicates removed (a body owns
/// exactly one leaf, so duplicates only arise defensively).
fn sort_dedup(ids: &mut Vec<usize>) {
    ids.sort_unstable();
    ids.dedup();
}

impl SequentialImpulseEngine {
    /// Live-tree query view when the broadphase is tree-backed and fresh;
    /// `None` is the explicit brute-force path (non-tree backend, stale
    /// tree, topology change).
    fn query_tree_view(&self) -> Option<QueryTreeView<'_>> {
        let tree = self.broadphase_tree_for_query()?;
        QueryTreeView::try_bind(&self.bodies, tree)
    }

    /// Closest ray hit with the surface normal (see [`RaycastHit`]);
    /// read-only: poses, sleep state and caches are untouched, nothing wakes.
    ///
    /// Served by the live broadphase tree when it is tree-backed and
    /// fresh, otherwise by the brute-force [`QueryPipeline`] — identical
    /// results either way (the tree only prunes).
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidInput`] for a non-finite/zero direction or a
    /// bad `max_dist`.
    pub fn query_cast_ray(
        &self,
        ray: &Ray,
        max_dist: f32,
        filter: &QueryFilter,
    ) -> Result<Option<RaycastHit>, QueryError> {
        if let Some(view) = self.query_tree_view() {
            return view.cast_ray(ray, max_dist, filter);
        }
        QueryPipeline::new().cast_ray(&self.bodies, ray, max_dist, filter)
    }

    /// Every ray hit, stable-sorted by distance; read-only (see
    /// [`QueryPipeline::intersect_ray`]). Tree-accelerated when the live
    /// broadphase is tree-backed and fresh, brute-force otherwise.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidInput`] for a non-finite/zero direction or a
    /// bad `max_dist`.
    pub fn query_intersect_ray(
        &self,
        ray: &Ray,
        max_dist: f32,
        filter: &QueryFilter,
    ) -> Result<Vec<RaycastHit>, QueryError> {
        if let Some(view) = self.query_tree_view() {
            return view.intersect_ray(ray, max_dist, filter);
        }
        QueryPipeline::new().intersect_ray(&self.bodies, ray, max_dist, filter)
    }

    /// Handles of every body containing `point`; read-only (see
    /// [`QueryPipeline::intersect_point`]). Tree-accelerated when the live
    /// broadphase is tree-backed and fresh, brute-force otherwise.
    pub fn query_intersect_point(&self, point: Vec3, filter: &QueryFilter) -> Vec<BodyHandle> {
        if let Some(view) = self.query_tree_view() {
            return view.intersect_point(point, filter);
        }
        QueryPipeline::new().intersect_point(&self.bodies, point, filter)
    }

    /// Handles of every body whose AABB overlaps `query`; read-only (see
    /// [`QueryPipeline::intersect_aabb`]). Tree-accelerated when the live
    /// broadphase is tree-backed and fresh, brute-force otherwise.
    pub fn query_intersect_aabb(&self, query: &AABB, filter: &QueryFilter) -> Vec<BodyHandle> {
        if let Some(view) = self.query_tree_view() {
            return view.intersect_aabb(query, filter);
        }
        QueryPipeline::new().intersect_aabb(&self.bodies, query, filter)
    }

    /// Closest surface projection of `point`; read-only (see
    /// [`QueryPipeline::project_point`]). Tree-accelerated when the live
    /// broadphase is tree-backed and fresh, brute-force otherwise.
    pub fn query_project_point(
        &self,
        point: Vec3,
        filter: &QueryFilter,
    ) -> Option<PointProjection> {
        if let Some(view) = self.query_tree_view() {
            return view.project_point(point, filter);
        }
        QueryPipeline::new().project_point(&self.bodies, point, filter)
    }

    /// Linear shape sweep along `displacement`; read-only (see
    /// [`QueryPipeline::cast_shape`]). Tree-accelerated when the live
    /// broadphase is tree-backed and fresh, brute-force otherwise.
    pub fn query_cast_shape(
        &self,
        shape: &Shape,
        position: Vec3,
        orientation: Quat,
        displacement: Vec3,
        filter: &QueryFilter,
    ) -> Option<RaycastHit> {
        if let Some(view) = self.query_tree_view() {
            return view.cast_shape(shape, position, orientation, displacement, filter);
        }
        QueryPipeline::new().cast_shape(
            &self.bodies,
            shape,
            position,
            orientation,
            displacement,
            filter,
        )
    }

    /// Handles of every body overlapping `shape` at the given pose;
    /// read-only (see [`QueryPipeline::intersect_shape`]).
    /// Tree-accelerated when the live broadphase is tree-backed and fresh,
    /// brute-force otherwise.
    pub fn query_intersect_shape(
        &self,
        shape: &Shape,
        position: Vec3,
        orientation: Quat,
        filter: &QueryFilter,
    ) -> Vec<BodyHandle> {
        if let Some(view) = self.query_tree_view() {
            return view.intersect_shape(shape, position, orientation, filter);
        }
        QueryPipeline::new().intersect_shape(&self.bodies, shape, position, orientation, filter)
    }
}
