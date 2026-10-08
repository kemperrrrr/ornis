//! Viewport picking for the editor domain (PLAN §i, slice E1).
//!
//! A viewport click becomes a [`PickRay`] built from the client-side
//! [`ViewportCamera`], intersected against the CPU-side render lanes
//! (`TransformDesc`/`MeshDesc`, read directly from the [`SmartStore`] — the
//! same direct-read canon the render extraction uses, never a second
//! world). The interim core is [`ray_triangle_hit`] (Möller–Trumbore) plus
//! an analytic sphere fast path; the nearest hit wins as [`PickHit`].
//!
//! The API is shaped for the future `geometry`-crate move (PLAN B3): the
//! ray/hit stay opaque newtypes ([`PickRay`]/[`PickHit`], never raw
//! arrays), so only these internals relocate — signatures are stable.
//! Procedural proxies mirror the renderer's shared unit meshes (unit sphere
//! `r = 1`, unit box `±0.5`, unit plane `1×1` in XZ, unit cylinder `r = 1`,
//! `h = 1`); sizes stay baked in the model matrix exactly like the
//! extraction does. Cost is `O(entities × triangles)` per query — fine for
//! click/hover rates, the BVH arrives with the `geometry` crate.
//!
//! [`SmartStore`]: ornis_core::SmartStore

use glam::{Mat4, Vec3};
use ornis_assets::scene::{MeshDesc, TransformDesc};
use ornis_core::{Entity, PositiveF32, SmartStore, UnitVec3};
use thiserror::Error;

/// Squared length below which a direction is treated as degenerate.
const DEGENERATE_LEN2: f32 = 1e-12;
/// Half-extent of the unit picking proxies (box corners, plane quad).
const HALF_EXTENT: f32 = 0.5;
/// Bounding radius of the unit-box proxy (half-diagonal `√3/2`).
const UNIT_BOX_BOUND: f32 = 0.866_025_4;
/// Bounding radius of the unit-plane proxy (half-diagonal `√2/2`).
const UNIT_PLANE_BOUND: f32 = std::f32::consts::FRAC_1_SQRT_2;
/// Bounding radius of the unit-quad proxy (same `1×1` half-diagonal).
const UNIT_QUAD_BOUND: f32 = std::f32::consts::FRAC_1_SQRT_2;
/// Bounding radius of the unit-cylinder proxy (`r = 1`, `h = 1`: `√(1+1/4)`).
const UNIT_CYLINDER_BOUND: f32 = 1.118_034;
/// Side/cap fan segments of the interim unit-cylinder proxy.
const CYLINDER_SEGMENTS: usize = 12;

/// Corners of the unit-box proxy (`±0.5`, matches the renderer's unit box).
const BOX_CORNERS: [[f32; 3]; 8] = [
    [-HALF_EXTENT, -HALF_EXTENT, -HALF_EXTENT],
    [HALF_EXTENT, -HALF_EXTENT, -HALF_EXTENT],
    [HALF_EXTENT, HALF_EXTENT, -HALF_EXTENT],
    [-HALF_EXTENT, HALF_EXTENT, -HALF_EXTENT],
    [-HALF_EXTENT, -HALF_EXTENT, HALF_EXTENT],
    [HALF_EXTENT, -HALF_EXTENT, HALF_EXTENT],
    [HALF_EXTENT, HALF_EXTENT, HALF_EXTENT],
    [-HALF_EXTENT, HALF_EXTENT, HALF_EXTENT],
];

/// Quad corner indices per box face (`-z`, `+z`, `-x`, `+x`, `-y`, `+y`).
const BOX_QUADS: [[usize; 4]; 6] = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 4, 7, 3],
    [1, 5, 6, 2],
    [0, 1, 5, 4],
    [3, 2, 6, 7],
];

/// Corners of the unit-plane proxy (`1×1` in XZ, matches the renderer).
const PLANE_CORNERS: [[f32; 3]; 4] = [
    [-HALF_EXTENT, 0.0, -HALF_EXTENT],
    [HALF_EXTENT, 0.0, -HALF_EXTENT],
    [HALF_EXTENT, 0.0, HALF_EXTENT],
    [-HALF_EXTENT, 0.0, HALF_EXTENT],
];

/// Corners of the unit-quad proxy (`1×1` in XY, matches the renderer).
const QUAD_CORNERS: [[f32; 3]; 4] = [
    [-HALF_EXTENT, HALF_EXTENT, 0.0],
    [HALF_EXTENT, HALF_EXTENT, 0.0],
    [HALF_EXTENT, -HALF_EXTENT, 0.0],
    [-HALF_EXTENT, -HALF_EXTENT, 0.0],
];

/// Tolerances for ray intersection (Möller–Trumbore core + sphere path).
///
/// Named thresholds instead of magic numbers at the call sites: `parallel`
/// floors the triangle determinant (near-parallel rays miss), `min_distance`
/// floors the hit distance (the ray origin itself never self-hits).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PickEpsilon {
    parallel: f32,
    min_distance: f32,
}

impl PickEpsilon {
    /// Default tolerances: determinant floor `1e-8`, distance floor `1e-6`.
    pub const DEFAULT: Self = Self {
        parallel: 1e-8,
        min_distance: 1e-6,
    };

    /// Checked constructor: `None` unless both thresholds are finite and
    /// positive.
    #[must_use]
    pub fn try_new(parallel: f32, min_distance: f32) -> Option<Self> {
        if parallel.is_finite() && parallel > 0.0 && min_distance.is_finite() && min_distance > 0.0
        {
            Some(Self {
                parallel,
                min_distance,
            })
        } else {
            None
        }
    }

    /// Determinant floor: `|det|` below this means a parallel miss.
    #[must_use]
    pub fn parallel(&self) -> f32 {
        self.parallel
    }

    /// Distance floor: hits at or below this are discarded.
    #[must_use]
    pub fn min_distance(&self) -> f32 {
        self.min_distance
    }
}

impl Default for PickEpsilon {
    /// Default tolerances (see [`PickEpsilon::DEFAULT`]).
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Client-side viewport camera picking rays are built from.
///
/// View state, not authoritative scene: the server world stays
/// camera-agnostic, while each viewport side (native window, WASM replica)
/// holds its own camera and converts clicks locally. Build from an orbit
/// camera via [`ViewportCamera::from_view_parameters`], which takes the
/// look-at frame as an explicit `(position, target, up, fov, near, far)`
/// tuple — the same components the render orbit camera publishes, without
/// a render dependency here by design.
///
/// Pointer convention: pixels, origin top-left, `y` down (browser canvas).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportCamera {
    position: Vec3,
    target: Vec3,
    up: Vec3,
    fov_y_deg: f32,
    near: f32,
    far: f32,
    viewport_px: [f32; 2],
}

impl ViewportCamera {
    /// Minimum viewport dimension in pixels (degenerate surfaces rejected).
    const MIN_VIEWPORT_PX: f32 = 1.0;
    /// Narrowest accepted vertical field of view in degrees.
    const MIN_FOV_DEG: f32 = 0.01;
    /// Widest accepted vertical field of view in degrees.
    const MAX_FOV_DEG: f32 = 179.0;

    /// Checked constructor: `None` on non-finite vectors, a coincident
    /// eye/target, an up vector parallel to the view direction, an
    /// out-of-range field of view, a non-positive near plane, `far <= near`,
    /// or a sub-pixel viewport.
    #[must_use]
    pub fn new(
        position: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y_deg: f32,
        near: f32,
        far: f32,
        viewport_px: [f32; 2],
    ) -> Option<Self> {
        if !position.is_finite() || !target.is_finite() || !up.is_finite() {
            return None;
        }
        let view = target - position;
        if !view.is_finite() || view.length_squared() < DEGENERATE_LEN2 {
            return None;
        }
        if view.normalize().cross(up).length_squared() < DEGENERATE_LEN2 {
            return None;
        }
        if !(Self::MIN_FOV_DEG..=Self::MAX_FOV_DEG).contains(&fov_y_deg) {
            return None;
        }
        if !near.is_finite() || near <= 0.0 || !far.is_finite() || far <= near {
            return None;
        }
        let [w, h] = viewport_px;
        if !w.is_finite()
            || !h.is_finite()
            || w < Self::MIN_VIEWPORT_PX
            || h < Self::MIN_VIEWPORT_PX
        {
            return None;
        }
        Some(Self {
            position,
            target,
            up,
            fov_y_deg,
            near,
            far,
            viewport_px,
        })
    }

    /// Builds a picking camera from orbit view parameters and a viewport
    /// size in pixels.
    ///
    /// `view` is the `(position, target, up, fov, near, far)` look-at frame
    /// the orbit camera publishes (as fields, not as this tuple), so
    /// viewport sides convert without depending on the render crate.
    /// Same validation as [`Self::new`].
    #[must_use]
    pub fn from_view_parameters(
        view: (Vec3, Vec3, Vec3, f32, f32, f32),
        viewport_px: [f32; 2],
    ) -> Option<Self> {
        let (position, target, up, fov_y_deg, near, far) = view;
        Self::new(position, target, up, fov_y_deg, near, far, viewport_px)
    }

    /// Eye position in world units.
    #[must_use]
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Look-at target in world units.
    #[must_use]
    pub fn target(&self) -> Vec3 {
        self.target
    }

    /// Camera up vector (world units, non-parallel to the view direction).
    #[must_use]
    pub fn up(&self) -> Vec3 {
        self.up
    }

    /// Vertical field of view in degrees.
    #[must_use]
    pub fn fov_y_deg(&self) -> f32 {
        self.fov_y_deg
    }

    /// Near clip plane distance in world units (positive).
    #[must_use]
    pub fn near(&self) -> f32 {
        self.near
    }

    /// Far clip plane distance in world units (greater than `near`).
    #[must_use]
    pub fn far(&self) -> f32 {
        self.far
    }

    /// Viewport size in pixels (`[width, height]`).
    #[must_use]
    pub fn viewport_px(&self) -> [f32; 2] {
        self.viewport_px
    }
}

/// Viewport pick ray: origin plus a unit direction (invariant in the type).
///
/// Built by [`pick_ray`]; [`UnitVec3`] guarantees callers never handle an
/// unnormalized or zero direction. Survives the `geometry`-crate move
/// unchanged (B3): future backends keep consuming this newtype.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PickRay {
    origin: Vec3,
    direction: UnitVec3,
}

impl PickRay {
    /// Checked constructor: `None` on a non-finite origin or a zero /
    /// non-finite direction.
    #[must_use]
    pub fn try_new(origin: Vec3, direction: Vec3) -> Option<Self> {
        if !origin.is_finite() {
            return None;
        }
        Some(Self {
            origin,
            direction: UnitVec3::normalize(direction)?,
        })
    }

    /// Ray origin in world units.
    #[must_use]
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Unit ray direction in world units.
    #[must_use]
    pub fn direction(&self) -> UnitVec3 {
        self.direction
    }

    /// Point at ray distance `t` in world units.
    #[must_use]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + Vec3::from(self.direction) * t
    }
}

/// Nearest-hit result: the picked entity plus its positive world distance.
///
/// Nearest wins across candidates; ties resolve to the first candidate in
/// lane order (deterministic). Survives the `geometry`-crate move unchanged
/// (B3), like [`PickRay`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PickHit {
    entity: Entity,
    distance: PositiveF32,
}

impl PickHit {
    /// Creates a hit result (distance must already be positive-checked).
    #[must_use]
    pub fn new(entity: Entity, distance: PositiveF32) -> Self {
        Self { entity, distance }
    }

    /// The picked entity.
    #[must_use]
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// World distance from the ray origin (always positive).
    #[must_use]
    pub fn distance(&self) -> PositiveF32 {
        self.distance
    }
}

/// Editor-chrome picking policy (enum, never a bool flag).
///
/// E1 decision, fixed here: [`ChromePolicy::SkipChrome`] is the default —
/// gizmo/grid chrome replicates for drawing but is not clickable; only
/// scene content selects. Gizmo picking arrives with E3, when handles become
/// first-class selectable targets with their own drag semantics. The pure
/// query takes the policy explicitly so tests cover both sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChromePolicy {
    /// `EditorOnly` entities never win a pick (E1 behaviour).
    #[default]
    SkipChrome,
    /// `EditorOnly` entities compete like scene content (E3 preview seam).
    IncludeChrome,
}

impl ChromePolicy {
    /// Whether chrome entities participate in picking.
    #[must_use]
    pub fn includes_chrome(self) -> bool {
        matches!(self, Self::IncludeChrome)
    }
}

/// Pure pick-query outcome: a nearest hit or a clean miss.
///
/// A miss is data, not an error: the click pipeline clears the selection on
/// [`PickOutcome::Miss`]. Errors ([`PickError`]) are reserved for queries
/// that could not run at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PickOutcome {
    /// Nearest candidate hit.
    Hit(PickHit),
    /// Ran over a non-empty scene, nothing intersected.
    Miss,
}

impl PickOutcome {
    /// The hit, if any.
    #[must_use]
    pub fn hit(self) -> Option<PickHit> {
        match self {
            Self::Hit(hit) => Some(hit),
            Self::Miss => None,
        }
    }
}

/// Typed picking failures: queries that could not run return these instead
/// of panicking. A clean miss over a pickable scene is [`PickOutcome::Miss`],
/// not an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PickError {
    /// No [`ViewportCamera`] resource: the authoritative world is
    /// camera-agnostic, so picking without a viewport side is a no-op.
    #[error("no viewport camera: the authoritative world is camera-agnostic")]
    NoCamera,
    /// No input-state or store resource to pick against.
    #[error("no input state or store resource to pick against")]
    NoInput,
    /// No pickable entities (missing lanes or every candidate filtered).
    #[error("no pickable entities in the store")]
    EmptyScene,
    /// Degenerate view or pointer input: no ray can be built.
    #[error("degenerate view or pointer input: ray cannot be built")]
    DegenerateView,
    /// Pointer lies outside the viewport: no selection change.
    #[error("pointer is outside the viewport")]
    OutsideViewport,
}

/// Builds the pick ray through viewport pixel `viewport_xy`.
///
/// Origin top-left, `y` down (browser canvas convention); edges inclusive.
/// Out-of-viewport pointers are [`PickError::OutsideViewport`] (the click
/// pipeline ignores them without touching the selection), non-finite input
/// or a degenerate camera frame is [`PickError::DegenerateView`].
pub fn pick_ray(camera: &ViewportCamera, viewport_xy: [f32; 2]) -> Result<PickRay, PickError> {
    let [px, py] = viewport_xy;
    if !px.is_finite() || !py.is_finite() {
        return Err(PickError::DegenerateView);
    }
    let [w, h] = camera.viewport_px();
    if px < 0.0 || py < 0.0 || px > w || py > h {
        return Err(PickError::OutsideViewport);
    }
    let forward = camera.target() - camera.position();
    if !forward.is_finite() || forward.length_squared() < DEGENERATE_LEN2 {
        return Err(PickError::DegenerateView);
    }
    let forward = forward.normalize();
    let right = forward.cross(camera.up());
    if !right.is_finite() || right.length_squared() < DEGENERATE_LEN2 {
        return Err(PickError::DegenerateView);
    }
    let right = right.normalize();
    let up = right.cross(forward);
    let half_tan = (camera.fov_y_deg().to_radians() * 0.5).tan();
    let aspect = w / h;
    let x = (2.0 * px / w - 1.0) * aspect * half_tan;
    let y = (1.0 - 2.0 * py / h) * half_tan;
    let direction = x * right + y * up + forward;
    PickRay::try_new(camera.position(), direction).ok_or(PickError::DegenerateView)
}

/// Analytic ray/sphere hit: nearest positive distance, if any.
///
/// `radius` must be finite and positive (degenerate spheres never hit).
/// A ray starting inside the sphere reports the exit distance (the camera
/// inside an object still selects it). Non-finite centers fail closed.
#[must_use]
pub fn ray_sphere_hit(
    ray: &PickRay,
    center: Vec3,
    radius: f32,
    eps: PickEpsilon,
) -> Option<PositiveF32> {
    if !radius.is_finite() || radius <= 0.0 || !center.is_finite() {
        return None;
    }
    let direction = Vec3::from(ray.direction());
    let oc = ray.origin() - center;
    let half_b = direction.dot(oc);
    let c = oc.length_squared() - radius * radius;
    let discriminant = half_b * half_b - c;
    if !discriminant.is_finite() || discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let min = eps.min_distance();
    let first = -half_b - root;
    if first > min {
        return PositiveF32::try_new(first);
    }
    let second = -half_b + root;
    if second > min {
        return PositiveF32::try_new(second);
    }
    None
}

/// Interim Möller–Trumbore ray/triangle hit: nearest positive distance.
///
/// Double-sided (either winding hits — thin proxies stay clickable from
/// both sides); near-parallel rays (`|det|` below the epsilon) and hits at
/// or below the distance floor miss. Non-finite vertices fail closed
/// (comparisons reject `NaN`). Moves to the `geometry` crate with B3 —
/// callers keep calling this signature.
#[must_use]
pub fn ray_triangle_hit(
    ray: &PickRay,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    eps: PickEpsilon,
) -> Option<PositiveF32> {
    let direction = Vec3::from(ray.direction());
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let p = direction.cross(edge2);
    let det = edge1.dot(p);
    if !det.is_finite() || det.abs() < eps.parallel() {
        return None;
    }
    let inv_det = 1.0 / det;
    let s = ray.origin() - v0;
    let u = s.dot(p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(edge1);
    let v = direction.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(q) * inv_det;
    if t > eps.min_distance() {
        PositiveF32::try_new(t)
    } else {
        None
    }
}

/// Nearest pick over the store's render lanes with default tolerances.
///
/// Reads `TransformDesc`/`MeshDesc` directly (the extraction canon), skips
/// chrome unless `policy` includes it, and returns the nearest hit.
/// `Ok(None)` is a clean miss; `Err(EmptyScene)` means no pickable
/// candidate existed at all (missing lanes count as empty, never panic).
pub fn pick_closest(
    store: &SmartStore,
    ray: &PickRay,
    policy: ChromePolicy,
) -> Result<Option<PickHit>, PickError> {
    pick_closest_with_epsilon(store, ray, policy, PickEpsilon::DEFAULT)
}

/// Nearest pick over the store's render lanes with explicit tolerances.
///
/// Same contract as [`pick_closest`]; the epsilon override exists for
/// tests and for the `geometry`-crate handover measurements.
pub fn pick_closest_with_epsilon(
    store: &SmartStore,
    ray: &PickRay,
    policy: ChromePolicy,
    eps: PickEpsilon,
) -> Result<Option<PickHit>, PickError> {
    let Some(transforms) = store.read_lane::<TransformDesc>() else {
        return Err(PickError::EmptyScene);
    };
    let Some(meshes) = store.read_lane::<MeshDesc>() else {
        return Err(PickError::EmptyScene);
    };
    let chrome = store.read_lane::<crate::EditorOnly>();
    let mut best: Option<PickHit> = None;
    let mut candidates = 0u32;
    for (entity, transform) in transforms.entities.iter().zip(transforms.data.iter()) {
        let Some(mesh) = meshes.get(*entity) else {
            continue;
        };
        if !policy.includes_chrome() && chrome.as_ref().is_some_and(|lane| lane.contains(*entity)) {
            continue;
        }
        candidates += 1;
        if let Some(distance) = intersect_entity(mesh, transform, ray, eps)
            && best.is_none_or(|hit: PickHit| distance.get() < hit.distance.get())
        {
            best = Some(PickHit::new(*entity, distance));
        }
    }
    if candidates == 0 {
        return Err(PickError::EmptyScene);
    }
    Ok(best)
}

/// Intersects one renderable entity: analytic sphere, triangulated unit
/// proxies for box/plane/cylinder (sizes baked in the model matrix, like
/// the extraction), exact soup for custom meshes. `None` on degenerate
/// transforms or misses — never panics.
fn intersect_entity(
    mesh: &MeshDesc,
    transform: &TransformDesc,
    ray: &PickRay,
    eps: PickEpsilon,
) -> Option<PositiveF32> {
    match mesh {
        MeshDesc::Sphere { radius, .. } => {
            let scale = transform.scale;
            let center = transform.translation;
            if !scale.is_finite() || !center.is_finite() {
                return None;
            }
            // Unit sphere (`r = 1`): world radius is the longest model axis
            // (same canon as the extraction's bounding sphere). Non-uniform
            // scale over-selects off-axis — interim, documented.
            let world_radius = scale.abs().max_element() * radius.get();
            ray_sphere_hit(ray, center, world_radius, eps)
        }
        MeshDesc::Box { size } => {
            let size_scale = scaled_size(
                transform.scale,
                Vec3::new(size[0].get(), size[1].get(), size[2].get()),
            )?;
            let model = model_matrix(transform, size_scale)?;
            proxy_hit(ray, &model, &box_triangles(), UNIT_BOX_BOUND, eps)
        }
        MeshDesc::Plane { size } => {
            let size_scale = scaled_size(
                transform.scale,
                Vec3::new(size[0].get(), 1.0, size[1].get()),
            )?;
            let model = model_matrix(transform, size_scale)?;
            proxy_hit(ray, &model, &plane_triangles(), UNIT_PLANE_BOUND, eps)
        }
        MeshDesc::Quad { size } => {
            let size_scale = scaled_size(
                transform.scale,
                Vec3::new(size[0].get(), size[1].get(), 1.0),
            )?;
            let model = model_matrix(transform, size_scale)?;
            proxy_hit(ray, &model, &quad_triangles(), UNIT_QUAD_BOUND, eps)
        }
        MeshDesc::Cylinder { radius, height, .. } => {
            let size_scale = scaled_size(
                transform.scale,
                Vec3::new(radius.get(), height.get(), radius.get()),
            )?;
            let model = model_matrix(transform, size_scale)?;
            proxy_hit(ray, &model, &cylinder_triangles(), UNIT_CYLINDER_BOUND, eps)
        }
        MeshDesc::Custom { positions, indices } => {
            let scale = transform.scale;
            if !scale.is_finite() {
                return None;
            }
            let model = model_matrix(transform, scale)?;
            let mut best: Option<PositiveF32> = None;
            for tri in indices.chunks_exact(3) {
                let (Some(&a), Some(&b), Some(&c)) = (
                    positions.get(tri[0] as usize),
                    positions.get(tri[1] as usize),
                    positions.get(tri[2] as usize),
                ) else {
                    continue;
                };
                let hit = ray_triangle_hit(
                    ray,
                    model.transform_point3(Vec3::from_array(a)),
                    model.transform_point3(Vec3::from_array(b)),
                    model.transform_point3(Vec3::from_array(c)),
                    eps,
                );
                if let Some(t) = hit
                    && best.is_none_or(|prev: PositiveF32| t.get() < prev.get())
                {
                    best = Some(t);
                }
            }
            best
        }
    }
}

/// Model matrix with the primitive size baked in the scale (extraction
/// canon). `None` on non-finite translation or scale.
fn model_matrix(transform: &TransformDesc, size_scale: Vec3) -> Option<Mat4> {
    let translation = transform.translation;
    if !translation.is_finite() || !size_scale.is_finite() {
        return None;
    }
    Some(Mat4::from_scale_rotation_translation(
        size_scale,
        transform.rotation.get(),
        translation,
    ))
}

/// Base scale times the primitive size, checked finite.
fn scaled_size(base: Vec3, mul: Vec3) -> Option<Vec3> {
    let scaled = base * mul;
    scaled.is_finite().then_some(scaled)
}

/// Longest model-matrix axis (same canon as the extraction bounding
/// sphere: the shared unit meshes carry size in the scale).
fn max_axis_len(model: &Mat4) -> f32 {
    model
        .x_axis
        .length()
        .max(model.y_axis.length())
        .max(model.z_axis.length())
}

/// Proxy intersection: bounding-sphere pre-test, then Möller–Trumbore over
/// the world-transformed triangles, nearest wins.
fn proxy_hit(
    ray: &PickRay,
    model: &Mat4,
    tris: &[[Vec3; 3]],
    bound: f32,
    eps: PickEpsilon,
) -> Option<PositiveF32> {
    let center = model.w_axis.truncate();
    ray_sphere_hit(ray, center, max_axis_len(model) * bound, eps)?;
    let mut best: Option<PositiveF32> = None;
    for tri in tris {
        let hit = ray_triangle_hit(
            ray,
            model.transform_point3(tri[0]),
            model.transform_point3(tri[1]),
            model.transform_point3(tri[2]),
            eps,
        );
        if let Some(t) = hit
            && best.is_none_or(|prev: PositiveF32| t.get() < prev.get())
        {
            best = Some(t);
        }
    }
    best
}

/// Twelve triangles of the unit box (`±0.5` corners, double-sided).
fn box_triangles() -> Vec<[Vec3; 3]> {
    let corners = BOX_CORNERS.map(Vec3::from_array);
    let mut tris = Vec::with_capacity(BOX_QUADS.len() * 2);
    for quad in BOX_QUADS {
        let (a, b, c, d) = (
            corners[quad[0]],
            corners[quad[1]],
            corners[quad[2]],
            corners[quad[3]],
        );
        tris.push([a, b, c]);
        tris.push([a, c, d]);
    }
    tris
}

/// Two triangles of the unit plane (`1×1` in XZ, double-sided).
fn plane_triangles() -> Vec<[Vec3; 3]> {
    let corners = PLANE_CORNERS.map(Vec3::from_array);
    vec![
        [corners[0], corners[1], corners[2]],
        [corners[0], corners[2], corners[3]],
    ]
}

/// Two triangles of the unit sprite quad (`1×1` in XY, double-sided).
fn quad_triangles() -> Vec<[Vec3; 3]> {
    let corners = QUAD_CORNERS.map(Vec3::from_array);
    vec![
        [corners[0], corners[1], corners[2]],
        [corners[0], corners[2], corners[3]],
    ]
}

/// Side quads plus cap fans of the unit cylinder (`r = 1`, `h = 1`,
/// double-sided). Tessellation is fixed interim (ignores the desc's
/// `radial_segments`); exact staging moves with the `geometry` crate.
fn cylinder_triangles() -> Vec<[Vec3; 3]> {
    use std::f32::consts::TAU;
    let mut tris = Vec::with_capacity(CYLINDER_SEGMENTS * 4);
    for j in 0..CYLINDER_SEGMENTS {
        let a0 = j as f32 / CYLINDER_SEGMENTS as f32 * TAU;
        let a1 = (j + 1) as f32 / CYLINDER_SEGMENTS as f32 * TAU;
        let (bottom0, top0) = (
            Vec3::new(a0.cos(), -HALF_EXTENT, a0.sin()),
            Vec3::new(a0.cos(), HALF_EXTENT, a0.sin()),
        );
        let (bottom1, top1) = (
            Vec3::new(a1.cos(), -HALF_EXTENT, a1.sin()),
            Vec3::new(a1.cos(), HALF_EXTENT, a1.sin()),
        );
        tris.push([bottom0, top0, top1]);
        tris.push([bottom0, top1, bottom1]);
        tris.push([Vec3::new(0.0, HALF_EXTENT, 0.0), top0, top1]);
        tris.push([Vec3::new(0.0, -HALF_EXTENT, 0.0), bottom1, bottom0]);
    }
    tris
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_core::Engine;

    /// Test camera: eye `(0, 0, 6)`, target origin, `200×200` viewport.
    fn camera() -> ViewportCamera {
        ViewportCamera::new(
            Vec3::new(0.0, 0.0, 6.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0,
            0.1,
            100.0,
            [200.0, 200.0],
        )
        .expect("valid test camera")
    }

    /// Store with render lanes registered for picking tests.
    fn store_with_lanes(engine: &mut Engine) -> &mut SmartStore {
        let store = engine.world_mut().store_mut().expect("world store");
        store.register::<TransformDesc>();
        store.register::<MeshDesc>();
        store.register::<crate::EditorOnly>();
        store
    }

    /// Spawns a unit sphere at `translation` with the given radius.
    fn spawn_sphere(store: &mut SmartStore, translation: [f32; 3], radius: f32) -> Entity {
        let entity = store.create_entity();
        store.insert(
            entity,
            TransformDesc::from_translation(glam::Vec3::from_array(translation)),
        );
        store.insert(
            entity,
            MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(radius),
                segments: 16,
                rings: 8,
            },
        );
        entity
    }

    /// A ray through the viewport center hits a known sphere at the front
    /// face distance.
    #[test]
    fn ray_through_known_sphere_hits() {
        let mut engine = Engine::new();
        let entity = spawn_sphere(store_with_lanes(&mut engine), [0.0, 0.0, 0.0], 1.0);
        let store = engine.world().store().expect("world store");
        let ray = pick_ray(&camera(), [100.0, 100.0]).expect("center ray");
        let hit = pick_closest(store, &ray, ChromePolicy::SkipChrome)
            .expect("pickable scene")
            .expect("sphere hit");
        assert_eq!(hit.entity(), entity);
        assert!(
            (hit.distance.get() - 5.0).abs() < 1e-3,
            "got {}",
            hit.distance.get()
        );
    }

    /// A corner ray over a non-empty scene is a clean miss, not an error.
    #[test]
    fn ray_missing_sphere_returns_miss() {
        let mut engine = Engine::new();
        spawn_sphere(store_with_lanes(&mut engine), [0.0, 0.0, 0.0], 1.0);
        let store = engine.world().store().expect("world store");
        let ray = pick_ray(&camera(), [199.0, 199.0]).expect("corner ray");
        assert_eq!(
            pick_closest(store, &ray, ChromePolicy::SkipChrome).expect("pickable scene"),
            None
        );
    }

    /// Two spheres on the ray: the nearest wins.
    #[test]
    fn nearest_of_two_wins() {
        let mut engine = Engine::new();
        let store = store_with_lanes(&mut engine);
        let near = spawn_sphere(store, [0.0, 0.0, 0.0], 1.0);
        spawn_sphere(store, [0.0, 0.0, -5.0], 1.0);
        let store = engine.world().store().expect("world store");
        let ray = pick_ray(&camera(), [100.0, 100.0]).expect("center ray");
        let hit = pick_closest(store, &ray, ChromePolicy::SkipChrome)
            .expect("pickable scene")
            .expect("sphere hit");
        assert_eq!(hit.entity(), near);
    }

    /// Chrome in front is ignored by default but wins when included.
    #[test]
    fn chrome_policy_decides_selectability() {
        let mut engine = Engine::new();
        let store = store_with_lanes(&mut engine);
        let gizmo = spawn_sphere(store, [0.0, 0.0, 3.0], 1.0);
        store.insert(gizmo, crate::EditorOnly);
        let scene = spawn_sphere(store, [0.0, 0.0, 0.0], 1.0);
        let store = engine.world().store().expect("world store");
        let ray = pick_ray(&camera(), [100.0, 100.0]).expect("center ray");
        let skipped = pick_closest(store, &ray, ChromePolicy::SkipChrome)
            .expect("pickable scene")
            .expect("scene hit");
        assert_eq!(skipped.entity(), scene);
        let included = pick_closest(store, &ray, ChromePolicy::IncludeChrome)
            .expect("pickable scene")
            .expect("chrome hit");
        assert_eq!(included.entity(), gizmo);
    }

    /// Empty stores and lane-less stores error instead of panicking.
    #[test]
    fn empty_scene_errors_not_panics() {
        let engine = Engine::new();
        let fallback_ray = PickRay::try_new(Vec3::new(0.0, 0.0, 6.0), Vec3::new(0.0, 0.0, -1.0))
            .expect("valid test ray");
        let store = engine.world().store().expect("world store");
        assert_eq!(
            pick_closest(store, &fallback_ray, ChromePolicy::SkipChrome),
            Err(PickError::EmptyScene)
        );
        let mut engine = Engine::new();
        store_with_lanes(&mut engine);
        let store = engine.world().store().expect("world store");
        assert_eq!(
            pick_closest(store, &fallback_ray, ChromePolicy::SkipChrome),
            Err(PickError::EmptyScene)
        );
    }

    /// Out-of-viewport pointers and degenerate inputs are typed errors.
    #[test]
    fn viewport_and_input_validation() {
        assert_eq!(
            pick_ray(&camera(), [201.0, 100.0]),
            Err(PickError::OutsideViewport)
        );
        assert_eq!(
            pick_ray(&camera(), [-1.0, 100.0]),
            Err(PickError::OutsideViewport)
        );
        assert_eq!(
            pick_ray(&camera(), [f32::NAN, 100.0]),
            Err(PickError::DegenerateView)
        );
        assert!(
            ViewportCamera::new(
                Vec3::new(0.0, 0.0, 6.0),
                Vec3::ZERO,
                Vec3::Y,
                60.0,
                0.1,
                100.0,
                [0.0, 200.0]
            )
            .is_none()
        );
        assert!(
            ViewportCamera::new(
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::Y,
                60.0,
                0.1,
                100.0,
                [200.0, 200.0]
            )
            .is_none()
        );
        assert!(PickEpsilon::try_new(0.0, 1e-6).is_none());
        assert!(PickEpsilon::try_new(1e-8, f32::NAN).is_none());
    }

    /// Box proxy hits the front face; a custom soup hits its triangle.
    #[test]
    fn box_and_custom_proxies_hit() {
        let mut engine = Engine::new();
        let store = store_with_lanes(&mut engine);
        let cube = store.create_entity();
        store.insert(cube, TransformDesc::IDENTITY);
        store.insert(
            cube,
            MeshDesc::Box {
                size: [
                    PositiveF32::expect_valid(2.0),
                    PositiveF32::expect_valid(2.0),
                    PositiveF32::expect_valid(2.0),
                ],
            },
        );
        let soup = store.create_entity();
        store.insert(
            soup,
            TransformDesc::from_translation(glam::Vec3::new(10.0, 0.0, 0.0)),
        );
        store.insert(
            soup,
            MeshDesc::Custom {
                positions: vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
                indices: vec![0, 1, 2, 99, 99, 99],
            },
        );
        let store = engine.world().store().expect("world store");
        let ray = pick_ray(&camera(), [100.0, 100.0]).expect("center ray");
        let hit = pick_closest(store, &ray, ChromePolicy::SkipChrome)
            .expect("pickable scene")
            .expect("box hit");
        assert_eq!(hit.entity(), cube);
        assert!(
            (hit.distance.get() - 5.0).abs() < 1e-3,
            "got {}",
            hit.distance.get()
        );
        // Malformed triangle indices fail closed per triangle: the valid
        // triangle still reports its own hit.
        let soup_ray = PickRay::try_new(Vec3::new(10.0, 0.0, 6.0), Vec3::new(0.0, 0.0, -1.0))
            .expect("valid soup ray");
        let soup_hit = pick_closest(store, &soup_ray, ChromePolicy::SkipChrome)
            .expect("pickable scene")
            .expect("soup hit");
        assert_eq!(soup_hit.entity(), soup);
    }

    /// Cylinder proxy intersects; the sphere pre-test rejects clear misses.
    #[test]
    fn cylinder_proxy_hits_and_misses() {
        let mut engine = Engine::new();
        let store = store_with_lanes(&mut engine);
        let rod = store.create_entity();
        store.insert(rod, TransformDesc::IDENTITY);
        store.insert(
            rod,
            MeshDesc::Cylinder {
                radius: PositiveF32::expect_valid(0.5),
                height: PositiveF32::expect_valid(2.0),
                radial_segments: 12,
            },
        );
        let store = engine.world().store().expect("world store");
        let ray = pick_ray(&camera(), [100.0, 100.0]).expect("center ray");
        let hit = pick_closest(store, &ray, ChromePolicy::SkipChrome)
            .expect("pickable scene")
            .expect("cylinder hit");
        assert_eq!(hit.entity(), rod);
        assert!(
            (hit.distance.get() - 5.5).abs() < 1e-2,
            "got {}",
            hit.distance.get()
        );
        let side = pick_ray(&camera(), [199.0, 100.0]).expect("side ray");
        assert_eq!(
            pick_closest(store, &side, ChromePolicy::SkipChrome).expect("pickable scene"),
            None
        );
    }
}
