//! Procedural convex-hull tooling (R4): Box3D-style builders over [`ConvexHull`].
//!
//! Thin, validated construction helpers in the spirit of Box3D's
//! `b3CreateCylinder` / `b3CreateCone` / `b3CreateRock` / `b3CreateHull` and
//! `b3CloneAndTransformHull` (see also parry's
//! `ConvexPolyhedron::from_convex_hull`, which likewise refuses degenerate
//! input instead of building a silent no-op): analytic shapes emit minimal
//! fan triangulations with exact volume/inertia, while the strict
//! [`ConvexHull::from_points`] wrapper adds caller budgets and explicit
//! degeneracy errors on top of [`ConvexHull::from_vertices`]. Narrow phase,
//! solver, sleep, CCD and broadphase are untouched — this module only
//! *builds* hulls the existing GJK/EPA path already consumes.
//!
//! # Limits (GJK path)
//!
//! * [`ConvexHull::MAX_VERTICES`]: hard tooling ceiling (256 vertices).
//!   GJK support itself is an `O(n)` vertex scan with fixed iteration caps
//!   (64 GJK + 32 EPA, polytope capped at 128 vertices), so larger hulls
//!   would only burn time — anything over the ceiling is
//!   [`MeshError::TooManyVertices`](crate::errors::MeshError), never a
//!   silent truncation.
//! * [`ConvexHull::FACE_CAP`]: faces triangulate only up to 64 vertices
//!   (pre-existing `O(n^4)` triple test). Builder hulls always ship explicit
//!   fans and are never faceless; [`ConvexHull::from_points`] hulls past the
//!   cap stay support-only (GJK/EPA exact, `try_inertia` falls back to the
//!   local box, face raycasts miss) — documented here, not hidden.
//! * Weld tolerance 1e-6 m (inherited from `from_vertices`); volume needs a
//!   fourth point 1e-6 m off the plane of the first three, else
//!   [`MeshError::DegenerateHull`](crate::errors::MeshError).
//!
//! # Follow-up (deliberately out of scope)
//!
//! `ColliderDesc::Hull` (convex recipe from authoring data) plus the glTF
//! `MeshDesc::Custom` convex projection in `colliders.rs` (`shape_for` /
//! `body_for`). Needs an assets-crate variant plus two projection arms —
//! more than the one-liner this change allows — so hosts keep building hull
//! bodies via `RigidBody::try_new_convex_hull` until then.

use std::f32::consts::TAU;

use glam::{Quat, Vec3};

use super::shape::{ConvexHull, Triangle};
use crate::constants::{DEGENERATE_LEN2, TET_VOLUME_DIVISOR};
use crate::errors::MeshError;

/// Minimum ring vertices for the analytic builders (a triangular prism is
/// the coarsest closed solid).
const MIN_SIDES: usize = 3;
/// Maximum cylinder ring resolution: `2 * sides` vertices must keep the
/// explicit fan within [`ConvexHull::FACE_CAP`], so builder hulls always
/// carry faces (exact volume/inertia, no support-only fallback).
const MAX_CYLINDER_SIDES: usize = 32;
/// Maximum cone ring resolution: `slices + 1` vertices within
/// [`ConvexHull::FACE_CAP`] for the same reason.
const MAX_CONE_SLICES: usize = 63;
/// Rock skeleton points (axis extremes, guarantee volume for every seed).
const ROCK_SKELETON: usize = 6;
/// Rock shell points (jittered directions for the rocky silhouette).
const ROCK_SHELL: usize = 20;
/// Minimum unique points for a closed hull (a tetrahedron).
const MIN_HULL_POINTS: usize = 4;
/// Squared feature floor for the volume probe: (1e-6 m)^2, the same weld
/// scale [`ConvexHull::from_vertices`] deduplicates at.
const THIN_EPS2: f32 = 1e-12;

/// Deterministic 64-bit LCG stream (Knuth constants): integer-only state,
/// so every seed maps to one bit-stable `f32` sequence on any platform
/// (IEEE round-to-nearest is deterministic; no libm calls on the path).
struct Lcg(u64);

impl Lcg {
    /// Next `u32` sample (high bits of the wrapped state).
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Next sample in `[0, 1]` (deterministic float division, same bits
    /// everywhere for the same integer stream).
    fn next_unit(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

/// True when the points span non-zero volume: some pair clears the weld
/// floor, some third point clears their line, some fourth clears the plane.
/// Index-ordered scan, no allocation, deterministic.
fn has_volume(vertices: &[Vec3]) -> bool {
    let Some(&a) = vertices.first() else {
        return false;
    };
    let mut second = None;
    for v in &vertices[1..] {
        if (*v - a).length_squared() > THIN_EPS2 {
            second = Some(*v);
            break;
        }
    }
    let Some(b) = second else {
        return false; // All coincident.
    };
    let ab = b - a;
    let ab_len2 = ab.length_squared();
    let mut third = None;
    for v in vertices {
        if (*v - a).cross(ab).length_squared() > THIN_EPS2 * ab_len2 {
            third = Some(*v);
            break;
        }
    }
    let Some(c) = third else {
        return false; // Collinear.
    };
    let normal = ab.cross(c - a);
    let n_len2 = normal.length_squared();
    for v in vertices {
        let gap = (*v - a).dot(normal);
        if gap * gap > THIN_EPS2 * n_len2 {
            return true;
        }
    }
    false // Coplanar.
}

impl ConvexHull {
    /// Hard vertex ceiling for hull tooling (GJK-path budget): GJK support
    /// is an `O(n)` vertex scan and the EPA polytope caps at 128 vertices,
    /// so inputs past this are refused with
    /// [`MeshError::TooManyVertices`](crate::errors::MeshError) instead of
    /// silently truncated. Enforced by [`ConvexHull::from_points`] and every
    /// builder below (which additionally keep their output within
    /// [`ConvexHull::FACE_CAP`] so builder hulls always carry faces).
    pub const MAX_VERTICES: usize = 256;

    /// Strict hull constructor: [`ConvexHull::from_vertices`] plus a caller
    /// vertex budget and explicit degeneracy errors. Thin wrapper — no new
    /// hull math, only admission checks.
    ///
    /// Coplanar-heavy input (grids, caps) triangulates with overlapping
    /// cover: GJK support stays exact, but [`ConvexHull::volume`] and
    /// `try_inertia` inherit the overlap bias — prefer the analytic
    /// builders ([`ConvexHull::cylinder`], [`ConvexHull::cone`]) for
    /// prismatic shapes.
    ///
    /// # Errors
    ///
    /// [`MeshError::TooManyVertices`](crate::errors::MeshError) when
    /// `points.len()` exceeds `max_vertices` (or the hard
    /// [`ConvexHull::MAX_VERTICES`]), [`MeshError::NonFiniteVertex`](crate::errors::MeshError)
    /// on non-finite input (from `from_vertices`),
    /// [`MeshError::DegenerateHull`](crate::errors::MeshError) on fewer
    /// than 4 unique points or zero enclosed volume.
    pub fn from_points(points: Vec<Vec3>, max_vertices: usize) -> Result<Self, MeshError> {
        if points.len() > max_vertices {
            return Err(MeshError::TooManyVertices {
                count: points.len(),
                max: max_vertices,
            });
        }
        if points.len() > Self::MAX_VERTICES {
            return Err(MeshError::TooManyVertices {
                count: points.len(),
                max: Self::MAX_VERTICES,
            });
        }
        let hull = Self::from_vertices(points)?;
        if hull.vertices.len() < MIN_HULL_POINTS {
            return Err(MeshError::DegenerateHull);
        }
        if hull.vertices.len() > Self::MAX_VERTICES {
            return Err(MeshError::TooManyVertices {
                count: hull.vertices.len(),
                max: Self::MAX_VERTICES,
            });
        }
        if !has_volume(&hull.vertices) {
            return Err(MeshError::DegenerateHull);
        }
        Ok(hull)
    }

    /// Solid cylinder hull along local +Y (`2 * sides` vertices, `4 * sides
    /// - 4` explicit faces: cap fans plus side quads). Origin at mid-height.
    ///
    /// Minimal fan triangulation, not the naive triple test: volume and
    /// `try_inertia` are exact (up to `f32` rounding), with none of the
    /// overlapping cover coplanar caps would otherwise produce.
    ///
    /// # Errors
    ///
    /// [`MeshError::BadHullParams`](crate::errors::MeshError) on
    /// non-finite/non-positive `height`/`radius` or `sides` outside
    /// `3..=32` (the upper bound keeps `2 * sides` within
    /// [`ConvexHull::FACE_CAP`]).
    pub fn cylinder(height: f32, radius: f32, sides: usize) -> Result<Self, MeshError> {
        if !height.is_finite() || height <= 0.0 {
            return Err(MeshError::BadHullParams {
                detail: "cylinder height must be finite and > 0",
            });
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(MeshError::BadHullParams {
                detail: "cylinder radius must be finite and > 0",
            });
        }
        if !(MIN_SIDES..=MAX_CYLINDER_SIDES).contains(&sides) {
            return Err(MeshError::BadHullParams {
                detail: "cylinder sides must be 3..=32",
            });
        }
        let half = height * 0.5;
        let mut vertices = Vec::with_capacity(2 * sides);
        for cap_y in [half, -half] {
            for i in 0..sides {
                let angle = TAU * i as f32 / sides as f32;
                vertices.push(Vec3::new(radius * angle.cos(), cap_y, radius * angle.sin()));
            }
        }
        let top = |i: usize| i;
        let bottom = |i: usize| sides + i;
        let mut faces = Vec::with_capacity(4 * sides - 4);
        for i in 1..sides - 1 {
            faces.push(Triangle::from_raw([
                top(0) as u32,
                top(i + 1) as u32,
                top(i) as u32,
            ]));
        }
        for i in 1..sides - 1 {
            faces.push(Triangle::from_raw([
                bottom(0) as u32,
                bottom(i) as u32,
                bottom(i + 1) as u32,
            ]));
        }
        for i in 0..sides {
            let j = (i + 1) % sides;
            faces.push(Triangle::from_raw([
                top(i) as u32,
                bottom(j) as u32,
                bottom(i) as u32,
            ]));
            faces.push(Triangle::from_raw([
                top(i) as u32,
                top(j) as u32,
                bottom(j) as u32,
            ]));
        }
        Ok(Self { vertices, faces })
    }

    /// Solid cone hull along local +Y: apex at `+height / 2`, base ring of
    /// `radius` at `-height / 2` (`slices + 1` vertices, `2 * slices - 2`
    /// explicit faces). Origin at mid-height.
    ///
    /// Minimal fan triangulation (exact volume/inertia), same standing as
    /// [`ConvexHull::cylinder`].
    ///
    /// # Errors
    ///
    /// [`MeshError::BadHullParams`](crate::errors::MeshError) on
    /// non-finite/non-positive `height`/`radius` or `slices` outside
    /// `3..=63` (the upper bound keeps `slices + 1` within
    /// [`ConvexHull::FACE_CAP`]).
    pub fn cone(height: f32, radius: f32, slices: usize) -> Result<Self, MeshError> {
        if !height.is_finite() || height <= 0.0 {
            return Err(MeshError::BadHullParams {
                detail: "cone height must be finite and > 0",
            });
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(MeshError::BadHullParams {
                detail: "cone radius must be finite and > 0",
            });
        }
        if !(MIN_SIDES..=MAX_CONE_SLICES).contains(&slices) {
            return Err(MeshError::BadHullParams {
                detail: "cone slices must be 3..=63",
            });
        }
        let half = height * 0.5;
        let mut vertices = Vec::with_capacity(slices + 1);
        vertices.push(Vec3::new(0.0, half, 0.0));
        for i in 0..slices {
            let angle = TAU * i as f32 / slices as f32;
            vertices.push(Vec3::new(radius * angle.cos(), -half, radius * angle.sin()));
        }
        let ring = |i: usize| 1 + i;
        let mut faces = Vec::with_capacity(2 * slices - 2);
        for i in 0..slices {
            let j = (i + 1) % slices;
            faces.push(Triangle::from_raw([0, ring(j) as u32, ring(i) as u32]));
        }
        for i in 1..slices - 1 {
            faces.push(Triangle::from_raw([
                ring(0) as u32,
                ring(i) as u32,
                ring(i + 1) as u32,
            ]));
        }
        Ok(Self { vertices, faces })
    }

    /// Deterministic rocky hull inside `radius` (26 vertices: a 6-point axis
    /// skeleton guaranteeing volume for every seed, plus a 20-point jittered
    /// shell). Same `seed` builds bitwise-identical vertices on every run
    /// and platform: the stream is integer LCG arithmetic with no libm calls
    /// on the path (directions come from normalized cube samples, radii from
    /// scaled unit samples).
    ///
    /// Shell points sit in generic position (no 4 coplanar by construction
    /// intent), so the naive triangulation covers facets uniformly and
    /// [`ConvexHull::volume`]/`try_inertia` stay exact up to `f32` rounding.
    ///
    /// # Errors
    ///
    /// [`MeshError::BadHullParams`](crate::errors::MeshError) on
    /// non-finite/non-positive `radius`; the [`ConvexHull::from_points`]
    /// errors on pathological weld merges (practically unreachable: samples
    /// spread over the full ball).
    pub fn rock(radius: f32, seed: u64) -> Result<Self, MeshError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(MeshError::BadHullParams {
                detail: "rock radius must be finite and > 0",
            });
        }
        let mut rng = Lcg(seed);
        let mut points = Vec::with_capacity(ROCK_SKELETON + ROCK_SHELL);
        for axis in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ] {
            points.push(axis * (radius * (0.8 + 0.2 * rng.next_unit())));
        }
        for i in 0..ROCK_SHELL {
            let mut dir = Vec3::new(
                2.0 * rng.next_unit() - 1.0,
                2.0 * rng.next_unit() - 1.0,
                2.0 * rng.next_unit() - 1.0,
            );
            if dir.length_squared() < THIN_EPS2 {
                dir = match i % 3 {
                    0 => Vec3::X,
                    1 => Vec3::Y,
                    _ => Vec3::Z,
                };
            }
            points.push(dir.normalize() * (radius * (0.7 + 0.3 * rng.next_unit())));
        }
        Self::from_points(points, Self::MAX_VERTICES)
    }

    /// Clone with a rigid placement plus (possibly non-uniform) scale:
    /// `pos + rot * (v * scale)` per vertex. Faces survive (topology is
    /// scale-invariant); mirrored scales (odd negative count) swap face
    /// winding back outward, so planes/normals derive correctly from the
    /// transformed positions instead of inheriting stale orientations.
    ///
    /// # Errors
    ///
    /// [`MeshError::BadHullParams`](crate::errors::MeshError) on
    /// non-finite `pos`/`scale`, a zero scale component (collapses volume),
    /// or a non-finite/zero-length `rot`;
    /// [`MeshError::DegenerateHull`](crate::errors::MeshError) when the
    /// source hull has no vertices.
    pub fn clone_and_transform(
        hull: &ConvexHull,
        pos: Vec3,
        rot: Quat,
        scale: Vec3,
    ) -> Result<Self, MeshError> {
        if !pos.is_finite() {
            return Err(MeshError::BadHullParams {
                detail: "transform position must be finite",
            });
        }
        if !scale.is_finite() {
            return Err(MeshError::BadHullParams {
                detail: "transform scale must be finite",
            });
        }
        if scale.x == 0.0 || scale.y == 0.0 || scale.z == 0.0 {
            return Err(MeshError::BadHullParams {
                detail: "transform scale must be non-zero per axis",
            });
        }
        if !rot.is_finite() || rot.length_squared() <= DEGENERATE_LEN2 {
            return Err(MeshError::BadHullParams {
                detail: "transform rotation must be a finite non-zero quaternion",
            });
        }
        if hull.vertices.is_empty() {
            return Err(MeshError::DegenerateHull);
        }
        let rot = rot.normalize();
        let vertices: Vec<Vec3> = hull
            .vertices
            .iter()
            .map(|v| pos + rot * (*v * scale))
            .collect();
        let mut faces = hull.faces.clone();
        if scale.x * scale.y * scale.z < 0.0 {
            for face in &mut faces {
                let raw = face.as_u32();
                *face = Triangle::from_raw([raw[0], raw[2], raw[1]]);
            }
        }
        Ok(Self { vertices, faces })
    }

    /// Enclosed volume (m^3) from the face triangulation (origin-fan
    /// tetrahedra, the same sum `try_inertia` integrates): exact for closed
    /// hulls with non-overlapping faces (every analytic builder below) up to
    /// `f32` rounding, `0.0` for faceless (over-cap) hulls. Naive
    /// [`ConvexHull::from_points`] triangulations with coplanar point sets
    /// overlap in cover (the unit cube sums to 48, a uniform 6x cover), so
    /// this query over-reports there — `try_inertia` stays exact regardless
    /// (mass normalizes the cover out).
    pub fn volume(&self) -> f32 {
        if self.faces.is_empty() {
            return 0.0;
        }
        let mut volume = 0.0f32;
        for face in &self.faces {
            let raw = face.as_u32();
            let (a, b, c) = (
                self.vertices[raw[0] as usize],
                self.vertices[raw[1] as usize],
                self.vertices[raw[2] as usize],
            );
            volume += a.dot(b.cross(c)) / TET_VOLUME_DIVISOR;
        }
        volume.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::ShapeRef;
    use crate::gjk::convex_distance;
    use crate::shape::Shape;

    /// Outward-facing check shared by the builder tests: every face normal
    /// must point away from the face centroid (same rule as the cube
    /// triangulation test in `gjk`).
    fn assert_faces_outward(hull: &ConvexHull) {
        assert!(!hull.faces.is_empty(), "builder hulls always carry faces");
        for face in hull.faces() {
            let raw = face.as_u32();
            let (a, b, c) = (
                hull.vertices[raw[0] as usize],
                hull.vertices[raw[1] as usize],
                hull.vertices[raw[2] as usize],
            );
            let normal = (b - a).cross(c - a).normalize();
            let center = (a + b + c) / 3.0;
            assert!(
                normal.dot(center) > 0.0,
                "face must point outward: {face:?} n={normal}"
            );
        }
    }

    #[test]
    fn cylinder_counts_aabb_and_volume() {
        let hull = ConvexHull::cylinder(2.0, 1.0, 8).expect("valid cylinder builds");
        assert_eq!(hull.vertices.len(), 16);
        assert_eq!(hull.faces.len(), 28);
        assert_faces_outward(&hull);
        let aabb = hull.aabb(Vec3::ZERO, Quat::IDENTITY);
        for (got, want) in [
            (aabb.min.x, -1.0),
            (aabb.min.y, -1.0),
            (aabb.min.z, -1.0),
            (aabb.max.x, 1.0),
            (aabb.max.y, 1.0),
            (aabb.max.z, 1.0),
        ] {
            assert!((got - want).abs() < 1e-5, "aabb mismatch: {got} vs {want}");
        }
        // Prism volume vs the smooth analytic PI*r^2*h: 8-gon cover is
        // within 5%; the fine 32-gon must hit the required +-2%.
        let analytic = std::f32::consts::PI * 2.0;
        assert!(
            (hull.volume() - 5.656854).abs() < 1e-3,
            "got {}",
            hull.volume()
        );
        let fine = ConvexHull::cylinder(2.0, 1.0, 32).expect("fine cylinder builds");
        assert_eq!(fine.vertices.len(), 64);
        assert_eq!(fine.faces.len(), 124);
        let rel = (fine.volume() - analytic).abs() / analytic;
        assert!(rel < 0.02, "cylinder volume {rel} off analytic {analytic}");
    }

    #[test]
    fn cone_counts_aabb_and_volume() {
        let hull = ConvexHull::cone(2.0, 1.0, 8).expect("valid cone builds");
        assert_eq!(hull.vertices.len(), 9);
        assert_eq!(hull.faces.len(), 14);
        assert_faces_outward(&hull);
        // Apex at +half_height, base ring at -half_height.
        assert!((hull.vertices[0] - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
        let aabb = hull.aabb(Vec3::ZERO, Quat::IDENTITY);
        assert!((aabb.min.y + 1.0).abs() < 1e-6);
        assert!((aabb.max.y - 1.0).abs() < 1e-6);
        assert!((aabb.max.x - 1.0).abs() < 1e-5);
        assert!((aabb.min.x + 1.0).abs() < 1e-5);
        // 32-gon base vs the smooth (1/3)*PI*r^2*h within +-2%.
        let fine = ConvexHull::cone(3.0, 1.0, 32).expect("fine cone builds");
        assert_eq!(fine.vertices.len(), 33);
        assert_eq!(fine.faces.len(), 62);
        let analytic = std::f32::consts::PI;
        let rel = (fine.volume() - analytic).abs() / analytic;
        assert!(rel < 0.02, "cone volume {rel} off analytic {analytic}");
    }

    #[test]
    fn rock_is_deterministic_and_bounded() {
        let seed = 0x1234_5678_9ABC_DEF0;
        let a = ConvexHull::rock(1.5, seed).expect("valid rock builds");
        let b = ConvexHull::rock(1.5, seed).expect("same seed rebuilds");
        assert_eq!(a.vertices.len(), 26);
        assert!(!a.faces.is_empty());
        assert_faces_outward(&a);
        // Bitwise determinism per seed (the LCG path is integer-only).
        assert_eq!(a.vertices.len(), b.vertices.len());
        for (va, vb) in a.vertices.iter().zip(b.vertices.iter()) {
            assert_eq!((va.x.to_bits(), va.y.to_bits(), va.z.to_bits()), {
                (vb.x.to_bits(), vb.y.to_bits(), vb.z.to_bits())
            });
        }
        // A different seed builds a different rock.
        let other = ConvexHull::rock(1.5, seed ^ 0xFFFF).expect("other seed builds");
        assert!(
            a.vertices
                .iter()
                .zip(other.vertices.iter())
                .any(|(va, vb)| va.to_array() != vb.to_array()),
            "distinct seeds must differ"
        );
        // Bounded by the radius, with real volume inside the ball.
        let aabb = a.aabb(Vec3::ZERO, Quat::IDENTITY);
        assert!(aabb.max.x <= 1.5 && aabb.max.y <= 1.5 && aabb.max.z <= 1.5);
        assert!(aabb.min.x >= -1.5 && aabb.min.y >= -1.5 && aabb.min.z >= -1.5);
        let ball = 4.0 / 3.0 * std::f32::consts::PI * 1.5f32.powi(3);
        assert!(
            a.volume() > 0.1,
            "rock must enclose volume, got {}",
            a.volume()
        );
        assert!(a.volume() < ball, "rock stays inside its ball");
    }

    #[test]
    fn clone_transform_matches_manual_build_in_narrow_phase() {
        let base = ConvexHull::cylinder(2.0, 1.0, 8).expect("valid cylinder builds");
        // Non-unit quaternion exercises the normalize path; non-uniform
        // scale the axis-wise mapping.
        let rot = Quat::from_rotation_y(0.6) * 2.0;
        let pos = Vec3::new(0.3, -0.1, 0.2);
        let scale = Vec3::new(1.5, 0.5, 2.0);
        let via_tool = ConvexHull::clone_and_transform(&base, pos, rot, scale)
            .expect("valid transform clones");
        let norm = rot.normalize();
        let manual_pts: Vec<Vec3> = base
            .vertices()
            .iter()
            .map(|v| pos + norm * (*v * scale))
            .collect();
        let manual = ConvexHull {
            vertices: manual_pts,
            faces: base.faces().to_vec(),
        };
        assert_eq!(via_tool.vertices, manual.vertices);
        assert_eq!(via_tool.faces, manual.faces);
        // Narrow parity: identical inputs hit GJK identically (separated).
        let probe = Shape::Sphere { radius: 0.5 };
        let query = |hull: &Shape| {
            convex_distance(
                ShapeRef {
                    shape: hull,
                    pos: Vec3::ZERO,
                    rot: Quat::IDENTITY,
                },
                ShapeRef {
                    shape: &probe,
                    pos: Vec3::new(4.0, 0.5, 0.0),
                    rot: Quat::IDENTITY,
                },
            )
        };
        let (tool_shape, manual_shape) = (Shape::ConvexHull(via_tool), Shape::ConvexHull(manual));
        let (d_tool, d_manual) = (query(&tool_shape), query(&manual_shape));
        assert!(d_tool.dist > 0.0, "test geometry must separate");
        assert_eq!(d_tool.dist, d_manual.dist);
        assert_eq!(d_tool.normal, d_manual.normal);
        assert_eq!(d_tool.point_a, d_manual.point_a);
        assert_eq!(d_tool.point_b, d_manual.point_b);
        // And in penetration (EPA path takes over below zero).
        let deep = |hull: &Shape| {
            convex_distance(
                ShapeRef {
                    shape: hull,
                    pos: Vec3::ZERO,
                    rot: Quat::IDENTITY,
                },
                ShapeRef {
                    shape: &probe,
                    pos: Vec3::new(1.0, 0.0, 0.0),
                    rot: Quat::IDENTITY,
                },
            )
        };
        let (p_tool, p_manual) = (deep(&tool_shape), deep(&manual_shape));
        assert!(p_tool.dist < 0.0, "test geometry must overlap");
        assert_eq!(p_tool.dist, p_manual.dist);
        assert_eq!(p_tool.normal, p_manual.normal);
    }

    #[test]
    fn clone_transform_mirror_keeps_outward_faces() {
        let base = ConvexHull::cone(2.0, 1.0, 8).expect("valid cone builds");
        let mirrored = ConvexHull::clone_and_transform(
            &base,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(-1.0, 1.0, 1.0),
        )
        .expect("mirror clones");
        assert_eq!(mirrored.vertices.len(), base.vertices.len());
        assert_faces_outward(&mirrored);
    }

    #[test]
    fn degenerates_are_typed_errors() {
        // Two points: not a hull.
        assert_eq!(
            ConvexHull::from_points(vec![Vec3::ZERO, Vec3::X], 8).expect_err("2 points refused"),
            MeshError::DegenerateHull
        );
        // Duplicates weld down to one point.
        assert_eq!(
            ConvexHull::from_points(vec![Vec3::Y; 5], 8).expect_err("duplicates refused"),
            MeshError::DegenerateHull
        );
        // Collinear and coplanar input has no volume.
        let line: Vec<Vec3> = (0..6).map(|i| Vec3::new(i as f32, 0.0, 0.0)).collect();
        assert_eq!(
            ConvexHull::from_points(line, 8).expect_err("collinear refused"),
            MeshError::DegenerateHull
        );
        let plane = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ];
        assert_eq!(
            ConvexHull::from_points(plane, 8).expect_err("coplanar refused"),
            MeshError::DegenerateHull
        );
        // Over the caller budget.
        let cube = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        assert_eq!(
            ConvexHull::from_points(cube.clone(), 3).expect_err("over budget refused"),
            MeshError::TooManyVertices { count: 8, max: 3 }
        );
        // The cube itself is fine, with cover-invariant exact inertia
        // (mass normalizes the uniform 6x overlap out: m=2 box I=4/3/axis).
        let hull = ConvexHull::from_points(cube, 64).expect("cube builds");
        let inertia = hull
            .try_inertia(2.0)
            .expect("closed cube has exact inertia");
        for got in [inertia.x, inertia.y, inertia.z] {
            assert!((got - 4.0 / 3.0).abs() < 1e-3, "got {inertia}");
        }
        // Non-finite input keeps the from_vertices error.
        assert_eq!(
            ConvexHull::from_points(vec![Vec3::ZERO, Vec3::NAN], 8)
                .expect_err("non-finite refused"),
            MeshError::NonFiniteVertex { index: 1 }
        );
        // Bad builder params.
        assert_eq!(
            ConvexHull::cylinder(0.0, 1.0, 8).expect_err("zero height refused"),
            MeshError::BadHullParams {
                detail: "cylinder height must be finite and > 0"
            }
        );
        assert_eq!(
            ConvexHull::cylinder(2.0, 1.0, 2).expect_err("2 sides refused"),
            MeshError::BadHullParams {
                detail: "cylinder sides must be 3..=32"
            }
        );
        assert_eq!(
            ConvexHull::cone(2.0, f32::NAN, 8).expect_err("NaN radius refused"),
            MeshError::BadHullParams {
                detail: "cone radius must be finite and > 0"
            }
        );
        assert_eq!(
            ConvexHull::rock(0.0, 7).expect_err("zero rock refused"),
            MeshError::BadHullParams {
                detail: "rock radius must be finite and > 0"
            }
        );
        assert_eq!(
            ConvexHull::clone_and_transform(
                &hull,
                Vec3::ZERO,
                Quat::IDENTITY,
                Vec3::new(1.0, 0.0, 1.0)
            )
            .expect_err("zero scale refused"),
            MeshError::BadHullParams {
                detail: "transform scale must be non-zero per axis"
            }
        );
    }
}
