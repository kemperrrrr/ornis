//! Convex collision primitives: sphere, box, capsule, cylinder, cone,
//! convex hull and heightfield.
//!
//! All [`Shape`] variants are centered on the body origin (box, capsule,
//! cylinder and cone are symmetric about local +Y; the cone's apex sits at
//! `+half_height`) and provide the two queries the pipeline relies on: an
//! exact world-space AABB projection for the broadphase and a diagonal
//! inertia tensor for the solver.
//!
//! The box↔capsule pair (G1 remainder) is now a first-class discrete contact
//! via `crate::distance::shape_distance` and `engine::box_vs_capsule`
//! (both `detect_collisions_into` paths, speculative `margin`, analytic TOI
//! through `distance::cast_shape`). Pairs involving cylinder, cone and
//! convex hull resolve through the GJK/EPA fallback in `crate::gjk` (single
//! contacts, same standing as the capsule paths); heightfields collide
//! column-wise and never go through GJK (they are not convex).

use glam::{Quat, Vec2, Vec3};

use crate::constants::{COINCIDENT_LEN2, DEGENERATE_LEN2, NEAR_ZERO, TET_VOLUME_DIVISOR};
use crate::errors::MeshError;
use crate::math::AABB;

// ---- Analytic inertia coefficients (standard rigid-body formulas) ----

/// Solid-sphere inertia factor about a diameter: `I = (2/5) m r²`.
const SPHERE_INERTIA_FACTOR: f32 = 0.4;
/// Uniform box inertia divisor: `I_xx = m/12 · (y² + z²)` on full side lengths.
const BOX_INERTIA_DIVISOR: f32 = 12.0;
/// Thin-disk / solid-cylinder inertia about the symmetry axis: `½ m r²`.
const DISK_AXIS_INERTIA: f32 = 0.5;
/// Midpoint / half-span scale used by heightfield grid math.
const HALF: f32 = 0.5;
/// Capsule transverse disk term: `¼ m r²`.
const CAPSULE_DISK_TRANSVERSE: f32 = 0.25;
/// Capsule stem coupling in the transverse inertia: `(m/3) · h · r`.
const CAPSULE_STEM_FACTOR: f32 = 3.0;
/// Solid cone inertia about the symmetry axis: `(3/10) m r²`.
const CONE_AXIS_INERTIA: f32 = 0.3;
/// Cone transverse inertia divisor in `m (3r² + 8h²) / 20`.
const CONE_TRANSVERSE_DIVISOR: f32 = 20.0;
/// Cone transverse `r²` coefficient inside that numerator.
const CONE_TRANSVERSE_R2: f32 = 3.0;
/// Cone transverse `h²` coefficient inside that numerator.
const CONE_TRANSVERSE_H2: f32 = 8.0;
/// Cylinder transverse `r²` coefficient in `m (3r² + 4h²) / 12`.
const CYLINDER_TRANSVERSE_R2: f32 = 3.0;
/// Cylinder transverse `h²` coefficient in `m (3r² + 4h²) / 12`.
const CYLINDER_TRANSVERSE_H2: f32 = 4.0;

/// Mirtich second-moment factor over an origin-based tet: `∫x² = V/10 · (…)`.
const TET_SECOND_MOMENT_DIVISOR: f32 = 10.0;

/// Vertices per triangle (index validation / centroid average).
const TRI_VERTS: usize = 3;
/// Minimum vertices for a usable convex hull (a tetrahedron).
const MIN_HULL_VERTS: usize = 4;
/// Corners of an AABB/OBB.
const BOX_CORNERS: usize = 8;
/// AABB-corner bit selecting the max endpoint on X / Y / Z.
const AABB_BIT_X: usize = 4;
const AABB_BIT_Y: usize = 2;
const AABB_BIT_Z: usize = 1;
/// Heightfield column treated as flat when height span is below this (m).
const HEIGHTFIELD_FLAT_EPS: f32 = 1e-4;
/// Floor for heightfield cell size when building a skirt (m).
const HEIGHTFIELD_MIN_CELL: f32 = 1e-3;

/// Convex collision primitives supported by the sequential-impulse engine.
///
/// All shapes are centered on the body origin; a box, a capsule, a cylinder
/// and a cone are symmetric about the body's local +Y axis. Every variant
/// must provide an AABB projection (broadphase) and a diagonal inertia
/// tensor (solver).
#[derive(Debug, Clone)]
pub enum Shape {
    /// Uniform ball: rotation-invariant, isotropic inertia.
    Sphere {
        /// Distance from center to surface.
        radius: f32,
    },
    /// Oriented box (OBB) with half-extents along each local axis.
    Box {
        /// Half-size of the box along its local X/Y/Z axes.
        half_extents: Vec3,
    },
    /// Cylinder of `2 * half_height` along local +Y with hemispherical caps
    /// of `radius`; used for characters and rounded bars.
    Capsule {
        /// Radius of the cylinder and the spherical caps.
        radius: f32,
        /// Half-length of the cylindrical segment, excluding the caps.
        half_height: f32,
    },
    /// Solid cylinder of `2 * half_height` along local +Y with FLAT caps
    /// (unlike [`Shape::Capsule`]).
    Cylinder {
        /// Radius of the flat end disks.
        radius: f32,
        /// Half-height of the cylinder along local +Y.
        half_height: f32,
    },
    /// Solid right circular cone along local +Y: apex at `+half_height`,
    /// base disk of `radius` at `-half_height`, origin at mid-height.
    Cone {
        /// Radius of the base disk.
        radius: f32,
        /// Half-height from the origin to the apex (and to the base).
        half_height: f32,
    },
    /// Explicit convex polyhedron in body-local space.
    ConvexHull(ConvexHull),
    /// Static heightfield terrain in body-local space (X/Z grid, +Y up).
    /// Intended for static bodies; colliding two heightfields reports no
    /// contact (terrain-vs-terrain is undefined).
    Heightfield(Heightfield),
    /// Triangle-soup mesh collider in body-local space (concave meshes
    /// welcome). Built with [`TriMesh::from_triangles`] (flat
    /// [`TriMesh::from_indexed`]): each triangle becomes a prebuilt convex
    /// primitive under a median-split AABB BVH, so narrow phase reuses the
    /// exact GJK/EPA path per triangle.
    /// Mesh-vs-mesh reports no contact (concave-concave is undefined —
    /// split compound colliders with `Fixed` joints instead).
    TriMesh(TriMesh),
}

/// Vertex index into a mesh vertex list.
///
/// Newtype over raw `u32` soup so vertex indices never mix with triangle
/// ordinals or body handles at the type level. Layout is `repr(transparent)`
/// over `u32` (12 bytes per [`Triangle`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct TriIndex(pub u32);

impl TriIndex {
    /// Wraps a raw vertex index without validation.
    pub const fn from_raw(index: u32) -> Self {
        Self(index)
    }

    /// Raw `u32` vertex index (for GPU/upload transports).
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Vertex position in a slice (`as usize`).
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for TriIndex {
    fn from(index: u32) -> Self {
        Self::from_raw(index)
    }
}

/// One triangle as three vertex indices (CCW from outside).
///
/// Stored as three [`TriIndex`] (12 bytes, `repr(C)`); use
/// [`Triangle::from_raw`]/[`Triangle::as_u32`] at transport boundaries and
/// [`Triangle::index`] for checked-position access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Triangle(pub TriIndex, pub TriIndex, pub TriIndex);

impl Triangle {
    /// Wraps three raw vertex indices without validation.
    pub const fn from_raw(indices: [u32; 3]) -> Self {
        Self(
            TriIndex(indices[0]),
            TriIndex(indices[1]),
            TriIndex(indices[2]),
        )
    }

    /// Raw `[u32; 3]` triple (for GPU/upload transports).
    pub const fn as_u32(self) -> [u32; 3] {
        [self.0.0, self.1.0, self.2.0]
    }

    /// `i`-th corner (`0..3`) as a vertex index. Out-of-range indices
    /// clamp to the last corner (same bit pattern as a saturated read).
    pub const fn index(self, i: usize) -> TriIndex {
        match i {
            0 => self.0,
            1 => self.1,
            _ => self.2,
        }
    }
}

impl From<[u32; 3]> for Triangle {
    fn from(indices: [u32; 3]) -> Self {
        Self::from_raw(indices)
    }
}

/// Explicit convex polyhedron: deduplicated local vertices plus outward
/// faces triangulated at construction. Built with
/// [`ConvexHull::from_vertices`]; faces exist only for small hulls (see the
/// constructor cap) — GJK support queries need vertices alone.
///
/// Construction invariant (finite vertices; faces valid triangle indices)
/// is enforced by the checked [`ConvexHull::from_vertices`]. The fields
/// stay `pub` for solver-adjacent compat (reads in `gjk`, `broadphase`,
/// `queries`, `avbd` and tests exceed the privatization budget), so prefer
/// the [`ConvexHull::vertices`]/[`ConvexHull::faces`] accessors and the
/// `try_` queries in new code instead of reaching into the fields.
#[derive(Debug, Clone)]
pub struct ConvexHull {
    /// Deduplicated vertices in body-local space.
    pub vertices: Vec<Vec3>,
    /// Outward-oriented triangles as vertex indices. Empty when the input
    /// exceeded the triangulation cap (support/GJK still work; face-slab
    /// raycasts report no hit).
    pub faces: Vec<Triangle>,
}

/// Heightfield terrain: a regular X/Z grid of heights, centered on the
/// body origin (+Y up). `heights[row * cols + col]`, `rows` along +Z,
/// `cols` along +X, uniform `cell` spacing.
///
/// Construction invariant (`heights.len() == rows * cols`, non-empty grid,
/// positive finite `cell`, finite samples) is enforced by the checked
/// [`Heightfield::new`] (plus [`Heightfield::validate`] /
/// [`Heightfield::is_valid`] for re-checks); the fields stay `pub` for
/// solver-adjacent compat (reads in `distance`, `queries` and tests exceed
/// the privatization budget — kept deliberately, not by oversight), so
/// queries keep defensive guards for legacy-constructed values instead of
/// assuming validity. New code should construct via [`Heightfield::new`] /
/// [`Heightfield::try_new_units`] / [`TryFrom`] and read via the
/// [`Heightfield::heights`]/[`Heightfield::rows`]/[`Heightfield::cols`]/
/// [`Heightfield::cell`] accessors.
#[derive(Debug, Clone)]
pub struct Heightfield {
    /// Row-major heights (`rows * cols` entries).
    pub heights: Vec<f32>,
    /// Sample count along +Z.
    pub rows: usize,
    /// Sample count along +X.
    pub cols: usize,
    /// Uniform grid spacing along X and Z (m).
    pub cell: f32,
}

/// Triangle-soup mesh collider: indexed triangles in body-local space
/// with a prebuilt AABB BVH. Each surviving triangle is stored as a
/// centroid-relative [`Shape::ConvexHull`] (zero per-query allocation in
/// narrow phase) under `centroids`; `nodes` accelerates both the narrow
/// walk and raycasts. Built with [`TriMesh::from_triangles`] (flat
/// [`TriMesh::from_indexed`]).
///
/// Derived aggregates (`local_min`/`local_max`, `bound_radius`,
/// `min_feature`) are computed at construction; the fields stay `pub` for
/// solver-adjacent compat (reads in `broadphase`, `queries`, `avbd` and
/// tests exceed the privatization budget), so new code should read via the
/// [`TriMesh::tris`]/[`TriMesh::centroids`]/[`TriMesh::bound_radius`]/
/// [`TriMesh::min_feature`]/[`TriMesh::local_bounds`] accessors.
#[derive(Debug, Clone)]
pub struct TriMesh {
    /// One convex primitive per surviving triangle, vertices relative to
    /// the matching entry of `centroids`.
    pub tris: Vec<Shape>,
    /// Triangle centroids in mesh-local space (placement origins).
    pub centroids: Vec<Vec3>,
    /// Median-split AABB tree over `tris` in mesh-local space.
    pub(crate) nodes: Vec<BvhNode>,
    /// Median-split permutation: leaves own `order[start..start+count]`
    /// ranges into `tris`/`centroids`.
    pub(crate) order: Vec<u32>,
    /// Mesh-local bounding box (precomputed; transformed per query).
    pub local_min: Vec3,
    /// Mesh-local bounding box max (precomputed).
    pub local_max: Vec3,
    /// Support radius about the body origin (max vertex length).
    pub bound_radius: f32,
    /// Smallest triangle edge over the soup (CCD travel gate input).
    pub min_feature: f32,
}

/// Closed contact-pair support: the loud marker for the two undefined
/// (concave-concave) pairs. See [`Shape::pair_support`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairSupport {
    /// The pair produces contacts (and cast hits) through
    /// `distance::shape_distance`.
    Supported,
    /// The pair is undefined and reports separation (no contact, no cast
    /// hit ever): terrain-vs-terrain or mesh-vs-mesh. Hosts must handle it
    /// explicitly — never a silent skip.
    UnsupportedPair,
}

/// Link of one [`BvhNode`]: either a leaf owning a triangle range or an
/// internal node branching to two children.
///
/// Replaces the legacy `left == u32::MAX` sentinel so leaf/internal
/// confusion is a type error, not a silent wrong traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BvhLink {
    /// Leaf owning `order[start..start+count]`.
    Leaf {
        /// First triangle in [`TriMesh::order`].
        start: u32,
        /// Triangle count.
        count: u32,
    },
    /// Internal node branching to two children.
    Node {
        /// Left child node index.
        left: u32,
        /// Right child node index.
        right: u32,
    },
}

impl BvhLink {
    /// Empty leaf owning no triangles.
    pub(crate) const fn empty_leaf() -> Self {
        Self::Leaf { start: 0, count: 0 }
    }

    /// Leaf range as `(start, count)`, or `None` for internal nodes.
    pub(crate) const fn leaf_range(self) -> Option<(u32, u32)> {
        match self {
            Self::Leaf { start, count } => Some((start, count)),
            Self::Node { .. } => None,
        }
    }

    /// Child indices, or `None` for leaves.
    pub(crate) const fn children(self) -> Option<(u32, u32)> {
        match self {
            Self::Node { left, right } => Some((left, right)),
            Self::Leaf { .. } => None,
        }
    }
}

/// One median-split AABB BVH node in mesh-local space. Internal nodes
/// branch via [`BvhLink::Node`]; leaves own their triangle range via
/// [`BvhLink::Leaf`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct BvhNode {
    /// Node bounding-box min.
    pub min: Vec3,
    /// Node bounding-box max.
    pub max: Vec3,
    /// Leaf vs internal link (replaces the `left == u32::MAX` sentinel).
    pub link: BvhLink,
}

impl Shape {
    /// Checked sphere: `None` unless `radius` is finite and `> 0`.
    pub fn try_sphere(radius: ornis_core::units::Meters) -> Option<Self> {
        let r = radius.get();
        if r.is_finite() && r > 0.0 {
            Some(Self::Sphere { radius: r })
        } else {
            None
        }
    }

    /// Checked capsule: `None` unless both `radius` and `half_height` are
    /// finite and `> 0`.
    pub fn try_capsule(
        radius: ornis_core::units::Meters,
        half_height: ornis_core::units::Meters,
    ) -> Option<Self> {
        let (r, h) = (radius.get(), half_height.get());
        if r.is_finite() && r > 0.0 && h.is_finite() && h > 0.0 {
            Some(Self::Capsule {
                radius: r,
                half_height: h,
            })
        } else {
            None
        }
    }

    /// Checked cylinder: `None` unless both `radius` and `half_height` are
    /// finite and `> 0`.
    pub fn try_cylinder(
        radius: ornis_core::units::Meters,
        half_height: ornis_core::units::Meters,
    ) -> Option<Self> {
        let (r, h) = (radius.get(), half_height.get());
        if r.is_finite() && r > 0.0 && h.is_finite() && h > 0.0 {
            Some(Self::Cylinder {
                radius: r,
                half_height: h,
            })
        } else {
            None
        }
    }

    /// Checked cone: `None` unless both `radius` and `half_height` are
    /// finite and `> 0`.
    pub fn try_cone(
        radius: ornis_core::units::Meters,
        half_height: ornis_core::units::Meters,
    ) -> Option<Self> {
        let (r, h) = (radius.get(), half_height.get());
        if r.is_finite() && r > 0.0 && h.is_finite() && h > 0.0 {
            Some(Self::Cone {
                radius: r,
                half_height: h,
            })
        } else {
            None
        }
    }

    /// Checked heightfield over a typed cell spacing: `None` unless the
    /// grid description satisfies [`Heightfield::new`].
    pub fn try_heightfield(
        heights: Vec<f32>,
        rows: usize,
        cols: usize,
        cell: ornis_core::units::Meters,
    ) -> Option<Self> {
        Heightfield::new(heights, rows, cols, cell.get())
            .ok()
            .map(Self::Heightfield)
    }

    /// Sphere radius in meters, or `None` for non-sphere shapes.
    pub fn sphere_radius(&self) -> Option<ornis_core::units::Meters> {
        match self {
            Self::Sphere { radius } => Some(ornis_core::units::Meters::new(*radius)),
            _ => None,
        }
    }

    /// World-space AABB of the shape at `position` with `orientation`.
    pub fn aabb(&self, position: Vec3, orientation: Quat) -> AABB {
        match self {
            Shape::Sphere { radius } => {
                let r = Vec3::splat(*radius);
                AABB::new(position - r, position + r)
            }
            Shape::Box { half_extents } => {
                // OBB -> AABB: the world-axis half-extent along axis i is
                // sum_j |R_ij| * half_extents_j (Ericson, RTCD §4.2.6),
                // i.e. |R| @ half_extents with an ELEMENT-WISE absolute
                // value of the rotation matrix — NOT |R @ half_extents|.
                // The two only coincide at 0/90/180/270° rotations; at
                // e.g. 45° the naive `orientation.mul_vec3(..).abs()`
                // under-reports the AABB (measured: a unit-cube box
                // rotated 45° about Z needs a sqrt(2) half-extent on x
                // AND y, but the naive formula zeroes the x component).
                // An under-sized AABB can miss broadphase pairs entirely.
                let basis_x = (orientation * Vec3::X).abs() * half_extents.x;
                let basis_y = (orientation * Vec3::Y).abs() * half_extents.y;
                let basis_z = (orientation * Vec3::Z).abs() * half_extents.z;
                let r = basis_x + basis_y + basis_z;
                AABB::new(position - r, position + r)
            }
            Shape::Capsule {
                radius,
                half_height,
            } => {
                // Local +Y axis, rotated by orientation; plus an isotropic radius shell.
                let axis = orientation * Vec3::Y;
                let e = axis.abs() * *half_height + Vec3::splat(*radius);
                AABB::new(position - e, position + e)
            }
            Shape::Cylinder {
                radius,
                half_height,
            } => {
                // Same axis shell as the capsule (flat caps change nothing
                // about the extents: the rim reaches radius in the plane).
                let axis = orientation * Vec3::Y;
                let e = axis.abs() * *half_height + Vec3::splat(*radius);
                AABB::new(position - e, position + e)
            }
            Shape::Cone {
                radius,
                half_height,
            } => {
                // Apex at +half_height, base rim at -half_height: the rim
                // needs the full radius shell only on the base side. The
                // axis-symmetric shell below is exact along the axis and
                // conservative radially (apex side over-covers by `radius`
                // in the plane — thin, and always sound for broadphase).
                let axis = orientation * Vec3::Y;
                let e = axis.abs() * *half_height + Vec3::splat(*radius);
                AABB::new(position - e, position + e)
            }
            Shape::ConvexHull(hull) => hull.aabb(position, orientation),
            Shape::Heightfield(hf) => hf.aabb(position, orientation),
            Shape::TriMesh(mesh) => mesh.aabb(position, orientation),
        }
    }

    /// Fallible diagonal (body-frame) inertia tensor for a given mass.
    /// `Ok` on every shape with exact inertia (sphere/box/capsule/
    /// cylinder/cone, plus hull/mesh soup with usable volume); `Err` when
    /// no exact inertia exists — hull/mesh [`MeshError::DegenerateMesh`]
    /// (empty faces/soup, non-positive mass, near-zero volume) or the
    /// heightfield box fallback below. The solver-facing total query is
    /// [`Shape::inertia`], which substitutes the bounding-box fallback.
    ///
    /// # Errors
    ///
    /// [`MeshError::DegenerateMesh`] for hull/mesh soup without usable
    /// volume, and for heightfields (terrain has no exact inertia — use
    /// the bounding box like [`Shape::inertia`] does).
    pub fn try_inertia(&self, mass: f32) -> Result<Vec3, MeshError> {
        match self {
            Self::Sphere { radius } => {
                if !radius.is_finite() || *radius <= 0.0 || !mass.is_finite() || mass <= 0.0 {
                    return Err(MeshError::DegenerateMesh);
                }
                Ok(Vec3::splat(SPHERE_INERTIA_FACTOR * mass * radius * radius))
            }
            _ => self
                .convex_try_inertia(mass)
                .or_else(|_| self.soup_try_inertia(mass)),
        }
    }

    /// Exact inertia for the analytic convex arms (box/capsule/cylinder/
    /// cone); `Err` for everything else (sphere is handled by the caller,
    /// hull/mesh/heightfield go through [`Shape::soup_try_inertia`]).
    fn convex_try_inertia(&self, mass: f32) -> Result<Vec3, MeshError> {
        if !mass.is_finite() || mass <= 0.0 {
            return Err(MeshError::DegenerateMesh);
        }
        let inertia = match self {
            Shape::Box { half_extents } => {
                if !half_extents.is_finite() || *half_extents == Vec3::ZERO {
                    return Err(MeshError::DegenerateMesh);
                }
                let sides = *half_extents * 2.0;
                let (x, y, z) = (right2(sides.x), right2(sides.y), right2(sides.z));
                Vec3::new(
                    (mass / BOX_INERTIA_DIVISOR) * (y + z),
                    (mass / BOX_INERTIA_DIVISOR) * (z + x),
                    (mass / BOX_INERTIA_DIVISOR) * (x + y),
                )
            }
            Shape::Capsule {
                radius,
                half_height,
            } => {
                if !radius.is_finite() || !half_height.is_finite() {
                    return Err(MeshError::DegenerateMesh);
                }
                let (h, r) = (half_height, radius);
                let i_y = DISK_AXIS_INERTIA * mass * r * r;
                let i_xz = CAPSULE_DISK_TRANSVERSE * mass * (r * r)
                    + (mass / CAPSULE_STEM_FACTOR) * h * r
                    + CAPSULE_DISK_TRANSVERSE * mass * h * h;
                Vec3::new(i_xz, i_y, i_xz)
            }
            Shape::Cylinder {
                radius,
                half_height,
            } => {
                if !radius.is_finite() || !half_height.is_finite() {
                    return Err(MeshError::DegenerateMesh);
                }
                let (r, h) = (radius, half_height);
                let i_y = DISK_AXIS_INERTIA * mass * r * r;
                let i_xz = mass * (CYLINDER_TRANSVERSE_R2 * r * r + CYLINDER_TRANSVERSE_H2 * h * h)
                    / BOX_INERTIA_DIVISOR;
                Vec3::new(i_xz, i_y, i_xz)
            }
            Shape::Cone {
                radius,
                half_height,
            } => {
                if !radius.is_finite() || !half_height.is_finite() {
                    return Err(MeshError::DegenerateMesh);
                }
                let (r, h) = (radius, half_height);
                let i_y = CONE_AXIS_INERTIA * mass * r * r;
                let i_xz = mass * (CONE_TRANSVERSE_R2 * r * r + CONE_TRANSVERSE_H2 * h * h)
                    / CONE_TRANSVERSE_DIVISOR;
                Vec3::new(i_xz, i_y, i_xz)
            }
            _ => return Err(MeshError::DegenerateMesh),
        };
        if inertia.is_finite() {
            Ok(inertia)
        } else {
            Err(MeshError::DegenerateMesh)
        }
    }

    /// Exact inertia for the soup arms: hull/mesh delegate to their
    /// [`ConvexHull::try_inertia`]/[`TriMesh::try_inertia`]; heightfields
    /// have no exact inertia and always report `DegenerateMesh` (the total
    /// [`Shape::inertia`] substitutes the bounding box).
    fn soup_try_inertia(&self, mass: f32) -> Result<Vec3, MeshError> {
        match self {
            Shape::ConvexHull(hull) => hull.try_inertia(mass),
            Shape::TriMesh(mesh) => mesh.try_inertia(mass),
            _ => Err(MeshError::DegenerateMesh),
        }
    }

    /// Diagonal (body-frame) inertia tensor for a given mass.
    /// One entry per principal axis; a sphere is isotropic, a box and a
    /// capsule are symmetric about their local +Y.
    ///
    /// Total (never fails) solver query over [`Shape::try_inertia`]:
    /// degenerate hull/mesh soup and heightfields silently use the local
    /// bounding box so a degenerate dynamic body still tumbles instead of
    /// dividing by zero. New code that must distinguish exact from
    /// fallback inertia should match on `try_inertia` directly.
    /// Deprecated as an exactness claim — kept as the total solver query.
    pub fn inertia(&self, mass: f32) -> Vec3 {
        if let Ok(exact) = self.try_inertia(mass) {
            return exact;
        }
        // Fallback site (single): bounding-box inertia for degenerate
        // soup and terrain. Terrain is static in practice; a dynamic
        // heightfield at least tumbles like its bounds. Non-soup shapes
        // only reach here on non-positive/non-finite mass and report zero.
        match self {
            Shape::ConvexHull(hull) => Shape::Box {
                half_extents: hull.local_box(),
            }
            .convex_try_inertia(mass)
            .unwrap_or(Vec3::ZERO),
            Shape::Heightfield(hf) => Shape::Box {
                half_extents: hf.local_extents(),
            }
            .convex_try_inertia(mass)
            .unwrap_or(Vec3::ZERO),
            Shape::TriMesh(mesh) => mesh.fallback_inertia(mass),
            _ => Vec3::ZERO,
        }
    }

    /// Whether the shape answers GJK support queries (`gjk` module).
    ///
    /// Only convex primitives answer directly: heightfields collide
    /// column-wise and triangle meshes resolve per triangle (see
    /// `distance::shape_distance`), so both report `false` here and their
    /// `support` arms are unreachable-by-construction fallbacks. A custom
    /// render soup has no implicit collider: hosts must build it explicitly
    /// with [`TriMesh::from_triangles`] ([`RigidBody::try_new_trimesh`] for bodies)
    /// — physics never substitutes a sphere placeholder.
    pub fn has_gjk_support(&self) -> bool {
        !matches!(self, Shape::Heightfield(_) | Shape::TriMesh(_))
    }

    /// Closed supported-pair list for discrete contacts: every pair EXCEPT
    /// the two variants below produces contacts through
    /// `distance::shape_distance` (heightfields column-wise, triangle meshes
    /// per triangle — both dispatched before GJK, so [`has_gjk_support`](Self::has_gjk_support)
    /// staying `false` for them is not a gap).
    ///
    /// The two [`PairSupport::UnsupportedPair`] cases (terrain-vs-terrain,
    /// mesh-vs-mesh) are concave-concave and undefined: the query reports
    /// separation, so no contact and no cast hit ever forms. This marker is
    /// the loud counterpart of that silent separation — hosts bridging
    /// per-entity custom colliders must check it up front (a custom render
    /// soup arrives as [`TriMesh::from_triangles`], which pairs with every
    /// convex shape and with heightfields) and either split the compound
    /// with `Fixed` joints or fail loudly instead of expecting contacts
    /// that never come. The solver behavior is unchanged by this query.
    pub fn pair_support(&self, other: &Shape) -> PairSupport {
        match (self, other) {
            (Shape::Heightfield(_), Shape::Heightfield(_))
            | (Shape::TriMesh(_), Shape::TriMesh(_)) => PairSupport::UnsupportedPair,
            _ => PairSupport::Supported,
        }
    }

    /// Closest surface point (world) to `p` for a placed shape. Exact for
    /// sphere/box/capsule/cylinder; cone picks the nearest of the wall,
    /// base-disk and apex candidates; hull scans its triangles; the
    /// heightfield clamps into the home column's solid box. Interior query
    /// points project to the nearest boundary face (never return the query
    /// itself): witness repair shifts queries inside the other solid by
    /// construction, and an interior "closest point" would freeze the
    /// iteration a full depth below the surface. Used to repair GJK/EPA
    /// tangential witnesses: EPA reports the right plane but its preimage
    /// blend collapses onto box corners when the other side's supports
    /// slide (rim circle), offsetting the contact meters sideways with a
    /// correct normal — a phantom-torque sink.
    pub(crate) fn closest_point(&self, pos: Vec3, rot: Quat, p: Vec3) -> Vec3 {
        let local = rot.conjugate() * (p - pos);
        let q = match self {
            Shape::Sphere { radius } => local.normalize_or(Vec3::X) * *radius,
            Shape::Box { half_extents } => closest_box_point(*half_extents, local),
            Shape::Capsule {
                radius,
                half_height,
            } => {
                let core = local.y.clamp(-*half_height, *half_height);
                let axis = Vec3::new(0.0, core, 0.0);
                axis + (local - axis).normalize_or(Vec3::X) * *radius
            }
            Shape::Cylinder {
                radius,
                half_height,
            } => closest_cylinder_point(*radius, *half_height, local),
            Shape::Cone {
                radius,
                half_height,
            } => closest_cone_point(*radius, *half_height, local),
            Shape::ConvexHull(hull) => closest_hull_point(hull, local),
            Shape::Heightfield(hf) => closest_heightfield_point(hf, local),
            Shape::TriMesh(mesh) => mesh.closest_point_local(local),
        };
        pos + rot * q
    }
}

/// Closest boundary point on a box (local frame): clamp for exterior
/// queries, nearest-face projection for interior ones.
fn closest_box_point(half_extents: Vec3, p: Vec3) -> Vec3 {
    let c = p.clamp(-half_extents, half_extents);
    if (c - p).length_squared() > COINCIDENT_LEN2 {
        return c; // Exterior: the clamp is the surface point.
    }
    // Interior: snap to the nearest face (deterministic x, then y, then z
    // on ties — strict comparisons keep the first minimum).
    let dx = half_extents.x - p.x.abs();
    let dy = half_extents.y - p.y.abs();
    let dz = half_extents.z - p.z.abs();
    if dx < dy && dx < dz {
        Vec3::new(half_extents.x.copysign(p.x), p.y, p.z)
    } else if dy < dz {
        Vec3::new(p.x, half_extents.y.copysign(p.y), p.z)
    } else {
        Vec3::new(p.x, p.y, half_extents.z.copysign(p.z))
    }
}

/// Closest point on a flat-capped cylinder (local frame, axis +Y).
fn closest_cylinder_point(radius: f32, half_height: f32, p: Vec3) -> Vec3 {
    let radial = Vec2::new(p.x, p.z);
    let len = radial.length();
    // Wall candidate: clamp height, snap the radius.
    let wall = if len > NEAR_ZERO {
        Vec3::new(
            radial.x / len * radius,
            p.y.clamp(-half_height, half_height),
            radial.y / len * radius,
        )
    } else {
        Vec3::new(radius, p.y.clamp(-half_height, half_height), 0.0)
    };
    // Cap candidates: clamp the radius at each cap plane.
    let cap = |y: f32| {
        let s = if len > NEAR_ZERO {
            (radius / len).min(1.0)
        } else {
            0.0
        };
        Vec3::new(p.x * s, y, p.z * s)
    };
    let top = cap(half_height);
    let bottom = cap(-half_height);
    if (p - wall).length_squared() < (p - top).length_squared()
        && (p - wall).length_squared() < (p - bottom).length_squared()
    {
        wall
    } else if (p - top).length_squared() < (p - bottom).length_squared() {
        top
    } else {
        bottom
    }
}

/// Closest point on a solid cone (local frame: apex `+half_height`, base
/// disk at `-half_height`). Nearest of wall/base/apex candidates.
fn closest_cone_point(radius: f32, half_height: f32, p: Vec3) -> Vec3 {
    if half_height <= NEAR_ZERO {
        return Vec3::new(p.x.clamp(-radius, radius), 0.0, p.z.clamp(-radius, radius));
    }
    let radial = Vec2::new(p.x, p.z);
    let len = radial.length();
    let y_c = p.y.clamp(-half_height, half_height);
    // Wall radius at the clamped height: r * (h - y) / (2h).
    let r_at = radius * (half_height - y_c) / (2.0 * half_height);
    let wall = if len > NEAR_ZERO {
        Vec3::new(radial.x / len * r_at, y_c, radial.y / len * r_at)
    } else {
        // On the axis: the wall circle is equidistant — pick +X
        // deterministically.
        Vec3::new(r_at, y_c, 0.0)
    };
    let s = if len > NEAR_ZERO {
        (radius / len).min(1.0)
    } else {
        0.0
    };
    let base = Vec3::new(p.x * s, -half_height, p.z * s);
    let apex = Vec3::new(0.0, half_height, 0.0);
    let (mut best, mut bd) = (wall, (p - wall).length_squared());
    for c in [base, apex] {
        let d = (p - c).length_squared();
        if d < bd {
            best = c;
            bd = d;
        }
    }
    best
}

/// Closest point on a convex hull (local frame): nearest triangle.
/// Falls back to the vertex average for empty faces (over-cap hulls).
fn closest_hull_point(hull: &ConvexHull, p: Vec3) -> Vec3 {
    if hull.faces.is_empty() {
        if hull.vertices.is_empty() {
            return Vec3::ZERO;
        }
        return hull.vertices.iter().sum::<Vec3>() / hull.vertices.len() as f32;
    }
    let mut best = hull.vertices[hull.faces[0].0.index()];
    let mut bd = (p - best).length_squared();
    for f in &hull.faces {
        let (a, b, c) = (
            hull.vertices[f.0.index()],
            hull.vertices[f.1.index()],
            hull.vertices[f.2.index()],
        );
        let (q, _, _, _) = crate::gjk::closest_triangle(a - p, b - p, c - p);
        let q = q + p;
        let d = (p - q).length_squared();
        if d < bd {
            best = q;
            bd = d;
        }
    }
    best
}

/// Closest point on a heightfield (local frame): clamp into the home
/// column's solid box (global minimum to the column top, with the same
/// one-cell skirt as the collision columns so flat plains have volume).
/// Off-grid points clamp into the nearest edge column.
fn closest_heightfield_point(hf: &Heightfield, p: Vec3) -> Vec3 {
    if hf.rows == 0 || hf.cols == 0 || !hf.cell.is_sign_positive() || hf.heights.is_empty() {
        return p;
    }
    let col = ((p.x / hf.cell + (hf.cols - 1) as f32 * HALF).floor() as isize)
        .clamp(0, hf.cols as isize - 1) as usize;
    let row = ((p.z / hf.cell + (hf.rows - 1) as f32 * HALF).floor() as isize)
        .clamp(0, hf.rows as isize - 1) as usize;
    let h = hf.heights[row * hf.cols + col];
    let (y_min, _) = hf.height_range();
    let y_low = if h - y_min >= HEIGHTFIELD_FLAT_EPS {
        y_min
    } else {
        h - hf.cell.max(HEIGHTFIELD_MIN_CELL)
    };
    let x_origin = -((hf.cols - 1) as f32) * HALF * hf.cell;
    let z_origin = -((hf.rows - 1) as f32) * HALF * hf.cell;
    let c = Vec3::new(
        p.x.clamp(
            x_origin + col as f32 * hf.cell,
            x_origin + (col + 1) as f32 * hf.cell,
        ),
        p.y.clamp(y_low.min(h), y_low.max(h)),
        p.z.clamp(
            z_origin + row as f32 * hf.cell,
            z_origin + (row + 1) as f32 * hf.cell,
        ),
    );
    if (c - p).length_squared() > COINCIDENT_LEN2 {
        return c; // Exterior: the clamp is the surface point.
    }
    // Interior: nearest-face projection of the column box (same rule as
    // `closest_box_point`; the column is thin in y on cliffs, wide in
    // x/z — ties keep the first minimum, x before y before z).
    let lo = Vec3::new(
        x_origin + col as f32 * hf.cell,
        y_low.min(h),
        z_origin + row as f32 * hf.cell,
    );
    let hi = Vec3::new(
        x_origin + (col + 1) as f32 * hf.cell,
        y_low.max(h),
        z_origin + (row + 1) as f32 * hf.cell,
    );
    let mid = (lo + hi) * HALF;
    let half = (hi - lo) * HALF;
    let q = p - mid;
    let dx = half.x - q.x.abs();
    let dy = half.y - q.y.abs();
    let dz = half.z - q.z.abs();
    if dx < dy && dx < dz {
        Vec3::new(mid.x + half.x.copysign(q.x), p.y, p.z)
    } else if dy < dz {
        Vec3::new(p.x, mid.y + half.y.copysign(q.y), p.z)
    } else {
        Vec3::new(p.x, p.y, mid.z + half.z.copysign(q.z))
    }
}

#[inline]
fn right2(v: f32) -> f32 {
    v * v
}

impl ConvexHull {
    /// Face-triangulation cap: the naive triple test is O(n^4), fast for
    /// game-size hulls, hopeless for scanned meshes. Above the cap the hull
    /// keeps its vertices (support/GJK work) but ships no faces (face-slab
    /// raycasts miss, inertia falls back to the local box).
    pub const FACE_CAP: usize = 64;

    /// Weld tolerance for vertex dedup (m).
    const WELD_EPS: f32 = 1e-6;

    /// Build a hull from local vertices: dedupes (weld), then triangulates
    /// outward faces with the naive triple test (deterministic index
    /// order). Degenerate input (fewer than 4 non-coplanar points) yields a
    /// hull with empty faces — collisions degrade to the vertex cloud, they
    /// never panic.
    ///
    /// # Errors
    ///
    /// [`MeshError::NonFiniteVertex`] when any input vertex is non-finite.
    pub fn from_vertices(points: Vec<Vec3>) -> Result<Self, MeshError> {
        for (index, p) in points.iter().enumerate() {
            if !p.is_finite() {
                return Err(MeshError::NonFiniteVertex { index });
            }
        }
        let mut vertices: Vec<Vec3> = Vec::with_capacity(points.len());
        for p in points {
            if !vertices
                .iter()
                .any(|q| (*q - p).length_squared() < Self::WELD_EPS * Self::WELD_EPS)
            {
                vertices.push(p);
            }
        }
        let faces = if vertices.len() >= MIN_HULL_VERTS && vertices.len() <= Self::FACE_CAP {
            Self::triangulate(&vertices)
        } else {
            Vec::new()
        };
        Ok(Self { vertices, faces })
    }

    /// Outward triangles: every ordered triple whose plane keeps all other
    /// points on the non-positive side (coplanar points tolerated, so quad
    /// faces triangulate into overlapping-but-harmless triangles sharing
    /// one normal). O(n^4), index-ordered, deterministic.
    fn triangulate(vertices: &[Vec3]) -> Vec<Triangle> {
        const COPLANAR_EPS: f32 = 1e-5;
        let n = vertices.len();
        let mut faces = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if j == i {
                    continue;
                }
                for k in 0..n {
                    if k == i || k == j {
                        continue;
                    }
                    let normal = (vertices[j] - vertices[i]).cross(vertices[k] - vertices[i]);
                    if normal.length_squared() < DEGENERATE_LEN2 {
                        continue; // Collinear triple, no plane.
                    }
                    let mut outside = false;
                    for (m, p) in vertices.iter().enumerate() {
                        if m == i || m == j || m == k {
                            continue;
                        }
                        if (p - vertices[i]).dot(normal) > COPLANAR_EPS {
                            outside = true;
                            break;
                        }
                    }
                    if !outside {
                        faces.push(Triangle::from_raw([i as u32, j as u32, k as u32]));
                    }
                }
            }
        }
        faces
    }

    /// Deduplicated local vertices (accessor over the compat `pub` field;
    /// see the struct docs for the privatization note).
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// Outward-oriented triangle indices (accessor over the compat `pub`
    /// field; empty past the triangulation cap).
    pub fn faces(&self) -> &[Triangle] {
        &self.faces
    }

    /// World-space AABB over the rotated vertices.
    pub fn aabb(&self, position: Vec3, orientation: Quat) -> AABB {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &self.vertices {
            let w = position + orientation * *v;
            min = min.min(w);
            max = max.max(w);
        }
        if self.vertices.is_empty() {
            min = position;
            max = position;
        }
        AABB::new(min, max)
    }

    /// Diagonal inertia from the face triangulation (tetrahedral fan about
    /// the origin, exact for closed hulls; off-diagonal products are
    /// dropped — the solver stores a body-frame diagonal). Falls back to
    /// the local bounding box when faces are missing.
    ///
    /// Legacy wrapper over [`ConvexHull::try_inertia`]: degenerate input
    /// silently uses the bounding box so existing scenes are bit-identical.
    /// Deprecated — do not use in new code, kept only for compat; match on
    /// `try_inertia` (exact) or [`Shape::inertia`] (total solver query).
    pub fn inertia(&self, mass: f32) -> Vec3 {
        self.try_inertia(mass).unwrap_or_else(|_| {
            let e = self.local_box();
            Shape::Box { half_extents: e }.inertia(mass)
        })
    }

    /// Exact diagonal inertia, or [`MeshError::DegenerateMesh`] when the
    /// hull has no usable faces/volume for the Mirtich sum.
    ///
    /// # Errors
    ///
    /// `DegenerateMesh` on empty faces, non-positive mass, or near-zero volume.
    pub fn try_inertia(&self, mass: f32) -> Result<Vec3, MeshError> {
        if self.faces.is_empty() || mass <= 0.0 {
            return Err(MeshError::DegenerateMesh);
        }
        // Signed volumes and second moments of tetrahedra (origin, a, b, c)
        // summed over faces (Mirtich-style, diagonal only).
        let mut vol = 0.0f32;
        let mut exx = 0.0f32;
        let mut eyy = 0.0f32;
        let mut ezz = 0.0f32;
        for f in &self.faces {
            let (a, b, c) = (
                self.vertices[f.0.index()],
                self.vertices[f.1.index()],
                self.vertices[f.2.index()],
            );
            let v = a.dot(b.cross(c)) / TET_VOLUME_DIVISOR;
            vol += v;
            // ∫x² over tet (origin,a,b,c) = v/10 * (xa²+xb²+xc²+xa·xb+...);
            // diagonal-only accumulation:
            exx += v / TET_SECOND_MOMENT_DIVISOR
                * (a.x * a.x + b.x * b.x + c.x * c.x + a.x * b.x + b.x * c.x + c.x * a.x);
            eyy += v / TET_SECOND_MOMENT_DIVISOR
                * (a.y * a.y + b.y * b.y + c.y * c.y + a.y * b.y + b.y * c.y + c.y * a.y);
            ezz += v / TET_SECOND_MOMENT_DIVISOR
                * (a.z * a.z + b.z * b.z + c.z * c.z + a.z * b.z + b.z * c.z + c.z * a.z);
        }
        if vol.abs() < NEAR_ZERO {
            return Err(MeshError::DegenerateMesh);
        }
        let density = mass / vol.abs();
        // I_xx = ∫(y²+z²), cyclic. Signed accumulation keeps orientation
        // consistent; density takes the absolute volume.
        Ok(Vec3::new(
            density * (eyy + ezz).abs(),
            density * (ezz + exx).abs(),
            density * (exx + eyy).abs(),
        ))
    }

    /// Local-space bounding half-extents (for inertia fallback).
    fn local_box(&self) -> Vec3 {
        let mut hi = Vec3::ZERO;
        for v in &self.vertices {
            hi = hi.max(v.abs());
        }
        hi
    }

    /// Smallest local bounding extent (for the CCD travel gate): thin hulls
    /// arm continuous collision at small motion, like thin boxes.
    pub(crate) fn min_extent(&self) -> f32 {
        if self.vertices.is_empty() {
            return 0.0;
        }
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for v in &self.vertices {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        (hi - lo).min_element().max(0.0)
    }
}

impl Heightfield {
    /// Checked constructor: `heights.len() == rows * cols`, non-empty grid,
    /// positive finite `cell` and finite samples.
    pub fn new(
        heights: Vec<f32>,
        rows: usize,
        cols: usize,
        cell: f32,
    ) -> Result<Self, crate::invariants::HeightfieldError> {
        crate::invariants::validate_heightfield(&heights, rows, cols, cell)?;
        Ok(Self {
            heights,
            rows,
            cols,
            cell,
        })
    }

    /// Whether this heightfield satisfies the construction invariant.
    pub fn is_valid(&self) -> bool {
        crate::invariants::validate_heightfield(&self.heights, self.rows, self.cols, self.cell)
            .is_ok()
    }

    /// Row-major height samples (accessor over the compat `pub` field).
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// Sample count along +Z (accessor over the compat `pub` field).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Sample count along +X (accessor over the compat `pub` field).
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Uniform grid spacing along X and Z in meters (accessor over the
    /// compat `pub` field).
    pub fn cell(&self) -> f32 {
        self.cell
    }

    /// Explicit validation of the construction invariant.
    pub fn validate(&self) -> Result<(), crate::invariants::HeightfieldError> {
        crate::invariants::validate_heightfield(&self.heights, self.rows, self.cols, self.cell)
    }

    /// Explicit fallible height query: `None` for an invalid heightfield
    /// instead of the silent `0.0` of [`Heightfield::height_at`].
    pub fn try_height_at(&self, x: f32, z: f32) -> Option<f32> {
        if !self.is_valid() || !x.is_finite() || !z.is_finite() {
            return None;
        }
        Some(self.height_at(x, z))
    }

    /// Bilinear height at local (x, z), clamped to the grid edge.
    /// Deterministic: pure arithmetic over the stored samples.
    ///
    /// Legacy wrapper: a broken grid (length mismatch, empty dims) silently
    /// reports `0.0`. Deprecated — do not use in new code, kept only for
    /// compat; use [`Heightfield::try_height_at`] (explicit `None` for
    /// invalid grids) instead.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        if self.heights.len() != self.rows * self.cols || self.rows == 0 || self.cols == 0 {
            return 0.0;
        }
        let fx = (x / self.cell + (self.cols - 1) as f32 * HALF).clamp(0.0, (self.cols - 1) as f32);
        let fz = (z / self.cell + (self.rows - 1) as f32 * HALF).clamp(0.0, (self.rows - 1) as f32);
        let x0 = (fx as usize).min(self.cols - 1);
        let z0 = (fz as usize).min(self.rows - 1);
        let x1 = (x0 + 1).min(self.cols - 1);
        let z1 = (z0 + 1).min(self.rows - 1);
        let tx = fx - x0 as f32;
        let tz = fz - z0 as f32;
        let h00 = self.heights[z0 * self.cols + x0];
        let h10 = self.heights[z0 * self.cols + x1];
        let h01 = self.heights[z1 * self.cols + x0];
        let h11 = self.heights[z1 * self.cols + x1];
        h00 * (1.0 - tx) * (1.0 - tz)
            + h10 * tx * (1.0 - tz)
            + h01 * (1.0 - tx) * tz
            + h11 * tx * tz
    }

    /// Local-space vertical range over all samples.
    pub fn height_range(&self) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for h in &self.heights {
            lo = lo.min(*h);
            hi = hi.max(*h);
        }
        if self.heights.is_empty() {
            lo = 0.0;
            hi = 0.0;
        }
        (lo, hi)
    }

    /// Local bounding half-extents (x/z from the grid footprint, y from the
    /// sample range).
    pub fn local_extents(&self) -> Vec3 {
        let (lo, hi) = self.height_range();
        Vec3::new(
            (self.cols.max(1) - 1) as f32 * HALF * self.cell,
            ((hi - lo) * HALF).max(0.0),
            (self.rows.max(1) - 1) as f32 * HALF * self.cell,
        )
    }

    /// Local bounding center (x/z centered, y at mid-range).
    pub fn local_center(&self) -> Vec3 {
        let (lo, hi) = self.height_range();
        Vec3::new(0.0, (lo + hi) * HALF, 0.0)
    }

    /// Typed cell-spacing entry point: same checks as [`Heightfield::new`]
    /// with the spacing carried as [`ornis_core::units::Meters`].
    pub fn try_new_units(
        heights: Vec<f32>,
        rows: usize,
        cols: usize,
        cell: ornis_core::units::Meters,
    ) -> Result<Self, crate::invariants::HeightfieldError> {
        Self::new(heights, rows, cols, cell.get())
    }

    /// Cell spacing in meters.
    pub fn cell_units(&self) -> ornis_core::units::Meters {
        ornis_core::units::Meters::new(self.cell)
    }

    /// World-space AABB: the oriented bounding box of the local bounds.
    /// Conservative like the box arm (element-wise rotated extents).
    pub fn aabb(&self, position: Vec3, orientation: Quat) -> AABB {
        let e = self.local_extents();
        let c = position + orientation * self.local_center();
        let basis_x = (orientation * Vec3::X).abs() * e.x;
        let basis_y = (orientation * Vec3::Y).abs() * e.y;
        let basis_z = (orientation * Vec3::Z).abs() * e.z;
        let r = basis_x + basis_y + basis_z;
        AABB::new(c - r, c + r)
    }
}

impl TriMesh {
    /// Leaf capacity of the median-split BVH (tuned for narrow-phase
    /// walks: few leaves visited, little per-leaf GJK fan-out).
    const LEAF_TRIS: usize = 4;

    /// Build a mesh collider from a vertex soup and typed triangles.
    /// Degenerate triangles (area² below 1e-12) are dropped, like the
    /// hull weld drops degenerate input.
    /// Deterministic: median splits with a stable index sort, no RNG.
    ///
    /// # Errors
    ///
    /// [`MeshError::OutOfRangeIndex`] for dangling indices,
    /// [`MeshError::NonFiniteVertex`] for non-finite vertices. An
    /// all-degenerate soup is `Ok` with zero triangles (empty mesh).
    pub fn from_triangles(vertices: &[Vec3], triangles: &[Triangle]) -> Result<Self, MeshError> {
        for (index, v) in vertices.iter().enumerate() {
            if !v.is_finite() {
                return Err(MeshError::NonFiniteVertex { index });
            }
        }
        let mut tris = Vec::with_capacity(triangles.len());
        let mut centroids = Vec::with_capacity(triangles.len());
        let mut local_min = Vec3::splat(f32::INFINITY);
        let mut local_max = Vec3::splat(f32::NEG_INFINITY);
        let mut bound_radius = 0.0f32;
        let mut min_feature = f32::INFINITY;
        for (t, tri) in triangles.iter().enumerate() {
            let raw = tri.as_u32();
            let v = [
                *vertices
                    .get(raw[0] as usize)
                    .ok_or(MeshError::OutOfRangeIndex {
                        triangle: t,
                        index: raw[0],
                        vertices: vertices.len(),
                    })?,
                *vertices
                    .get(raw[1] as usize)
                    .ok_or(MeshError::OutOfRangeIndex {
                        triangle: t,
                        index: raw[1],
                        vertices: vertices.len(),
                    })?,
                *vertices
                    .get(raw[2] as usize)
                    .ok_or(MeshError::OutOfRangeIndex {
                        triangle: t,
                        index: raw[2],
                        vertices: vertices.len(),
                    })?,
            ];
            if (v[1] - v[0]).cross(v[2] - v[0]).length_squared() < DEGENERATE_LEN2 {
                continue; // Degenerate: zero-area sliver, no collision value.
            }
            for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
                min_feature = min_feature.min((b - a).length());
            }
            for p in v {
                local_min = local_min.min(p);
                local_max = local_max.max(p);
                bound_radius = bound_radius.max(p.length());
            }
            let c = (v[0] + v[1] + v[2]) / TRI_VERTS as f32;
            centroids.push(c);
            // `from_vertices` only triangulates 4+ vertices, so a lone
            // triangle would get no faces (dead raycast, centroid-fallback
            // closest point). Set both windings explicitly: raycasts flip
            // the normal toward the ray and closest-point is winding-free.
            // Vertices are validated finite above; skip if hull construction
            // still rejects the triple (defensive — should not happen).
            let Ok(mut hull) = ConvexHull::from_vertices(vec![v[0] - c, v[1] - c, v[2] - c]) else {
                continue;
            };
            hull.faces = vec![Triangle::from_raw([0, 1, 2]), Triangle::from_raw([0, 2, 1])];
            tris.push(Shape::ConvexHull(hull));
        }
        if tris.is_empty() {
            local_min = Vec3::ZERO;
            local_max = Vec3::ZERO;
            min_feature = 0.0;
        }
        let (nodes, order) = Self::build_bvh(&tris, &centroids);
        Ok(Self {
            tris,
            centroids,
            nodes,
            order,
            local_min,
            local_max,
            bound_radius,
            min_feature,
        })
    }

    /// Build a mesh collider from a vertex soup and a flat index list.
    ///
    /// The flat list is chunked into [`Triangle`]s via
    /// [`Triangle::from_raw`]; degenerate triangles are dropped by
    /// [`TriMesh::from_triangles`].
    ///
    /// # Errors
    ///
    /// [`MeshError::BadIndexCount`] when `indices.len() % 3 != 0`,
    /// plus the [`TriMesh::from_triangles`] errors for dangling or
    /// non-finite input.
    pub fn from_indexed(vertices: &[Vec3], indices: &[u32]) -> Result<Self, MeshError> {
        if !indices.len().is_multiple_of(TRI_VERTS) {
            return Err(MeshError::BadIndexCount { len: indices.len() });
        }
        let triangles: Vec<Triangle> = indices
            .chunks_exact(TRI_VERTS)
            .map(|c| Triangle::from_raw([c[0], c[1], c[2]]))
            .collect();
        Self::from_triangles(vertices, &triangles)
    }

    /// One convex primitive per surviving triangle (accessor over the
    /// compat `pub` field).
    pub fn tris(&self) -> &[Shape] {
        &self.tris
    }

    /// Triangle centroids in mesh-local space (accessor over the compat
    /// `pub` field).
    pub fn centroids(&self) -> &[Vec3] {
        &self.centroids
    }

    /// Precomputed mesh-local bounding box (accessor over the compat `pub`
    /// fields).
    pub fn local_bounds(&self) -> (Vec3, Vec3) {
        (self.local_min, self.local_max)
    }

    /// Support radius about the body origin (accessor over the compat
    /// `pub` field).
    pub fn bound_radius(&self) -> f32 {
        self.bound_radius
    }

    /// Smallest triangle edge over the soup (accessor over the compat
    /// `pub` field).
    pub fn min_feature(&self) -> f32 {
        self.min_feature
    }

    /// Triangle ordinal at BVH permutation position `pos`, or `None` when
    /// out of range. Centralizes the `order[o] as usize` walk so narrow
    /// phase never indexes blindly.
    pub(crate) fn ordered_triangle(&self, pos: usize) -> Option<usize> {
        self.order
            .get(pos)
            .map(|id| *id as usize)
            .filter(|t| *t < self.tris.len() && *t < self.centroids.len())
    }

    /// Median-split AABB BVH over the triangle bounds (mesh-local space).
    /// Splits the longest axis at the centroid median; all-equal centroids
    /// become a leaf (no empty children, no infinite recursion).
    fn build_bvh(tris: &[Shape], centroids: &[Vec3]) -> (Vec<BvhNode>, Vec<u32>) {
        // Per-triangle local bounds from the centroid-relative verts.
        let mut bounds = Vec::with_capacity(tris.len());
        for (tri, c) in tris.iter().zip(centroids.iter()) {
            let Shape::ConvexHull(hull) = tri else {
                continue; // Unreachable: built above as hulls.
            };
            let mut lo = Vec3::splat(f32::INFINITY);
            let mut hi = Vec3::splat(f32::NEG_INFINITY);
            for v in &hull.vertices {
                let w = *v + *c;
                lo = lo.min(w);
                hi = hi.max(w);
            }
            bounds.push((lo, hi));
        }
        let mut nodes = Vec::new();
        let mut order: Vec<u32> = (0..bounds.len() as u32).collect();
        if !order.is_empty() {
            Self::split(&bounds, centroids, &mut order, 0, &mut nodes);
        }
        if nodes.is_empty() {
            // Degenerate mesh (no surviving triangles): one empty leaf so
            // traversals visit nothing instead of indexing nothing.
            nodes.push(BvhNode {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
                link: BvhLink::empty_leaf(),
            });
        }
        (nodes, order)
    }

    /// Recursive median split over `order` (`base` = its offset in the root
    /// permutation); appends nodes depth-first and returns the node index.
    /// Partitioned in place with a stable centroid sort — deterministic for
    /// fixed input. Leaves own `order[start..start+count]` ranges.
    fn split(
        bounds: &[(Vec3, Vec3)],
        centroids: &[Vec3],
        order: &mut [u32],
        base: usize,
        nodes: &mut Vec<BvhNode>,
    ) -> u32 {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for &t in order.iter() {
            let (a, b) = bounds[t as usize];
            lo = lo.min(a);
            hi = hi.max(b);
        }
        let here = nodes.len() as u32;
        nodes.push(BvhNode {
            min: lo,
            max: hi,
            link: BvhLink::empty_leaf(),
        });
        if order.len() <= Self::LEAF_TRIS {
            nodes[here as usize].link = BvhLink::Leaf {
                start: base as u32,
                count: order.len() as u32,
            };
            return here;
        }
        let e = hi - lo;
        let axis = if e.x >= e.y && e.x >= e.z {
            0
        } else if e.y >= e.z {
            1
        } else {
            2
        };
        order.sort_by(|&a, &b| {
            centroids[a as usize][axis]
                .partial_cmp(&centroids[b as usize][axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = order.len() / 2;
        // All-equal centroids: any split is empty on one side — leaf out.
        let (first, last) = (
            centroids[order[0] as usize][axis],
            centroids[order[order.len() - 1] as usize][axis],
        );
        if first == last {
            nodes[here as usize].link = BvhLink::Leaf {
                start: base as u32,
                count: order.len() as u32,
            };
            return here;
        }
        let (left_order, right_order) = order.split_at_mut(mid);
        let left = Self::split(bounds, centroids, left_order, base, nodes);
        let right = Self::split(bounds, centroids, right_order, base + mid, nodes);
        nodes[here as usize].link = BvhLink::Node { left, right };
        here
    }

    /// World-space AABB from the precomputed local box (8 corners).
    pub fn aabb(&self, position: Vec3, orientation: Quat) -> AABB {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for i in 0..BOX_CORNERS {
            let corner = Vec3::new(
                if i & AABB_BIT_X == 0 {
                    self.local_min.x
                } else {
                    self.local_max.x
                },
                if i & AABB_BIT_Y == 0 {
                    self.local_min.y
                } else {
                    self.local_max.y
                },
                if i & AABB_BIT_Z == 0 {
                    self.local_min.z
                } else {
                    self.local_max.z
                },
            );
            let w = position + orientation * corner;
            lo = lo.min(w);
            hi = hi.max(w);
        }
        AABB::new(lo, hi)
    }

    /// Diagonal inertia from the triangle soup (Mirtich-style signed
    /// tetrahedra about the body origin, same math as
    /// [`ConvexHull::inertia`]): exact for closed, consistently wound
    /// meshes; falls back to the local bounding box on degenerate input.
    /// Inconsistent winding averages out instead of failing loudly — keep
    /// soup clean (outward, manifold) for dynamic bodies.
    ///
    /// Legacy wrapper over [`TriMesh::try_inertia`]: degenerate input
    /// silently uses the bounding box so existing scenes are bit-identical.
    /// Deprecated — do not use in new code, kept only for compat; match on
    /// `try_inertia` (exact) or [`Shape::inertia`] (total solver query).
    pub fn inertia(&self, mass: f32) -> Vec3 {
        self.try_inertia(mass)
            .unwrap_or_else(|_| self.fallback_inertia(mass))
    }

    /// Exact soup inertia, or [`MeshError::DegenerateMesh`] on empty soup,
    /// non-positive mass, or near-zero signed volume.
    ///
    /// # Errors
    ///
    /// `DegenerateMesh` when no exact inertia exists (use the bounding box).
    pub fn try_inertia(&self, mass: f32) -> Result<Vec3, MeshError> {
        if self.tris.is_empty() || mass <= 0.0 {
            return Err(MeshError::DegenerateMesh);
        }
        let mut vol = 0.0f32;
        let mut exx = 0.0f32;
        let mut eyy = 0.0f32;
        let mut ezz = 0.0f32;
        for (tri, c) in self.tris.iter().zip(self.centroids.iter()) {
            let Shape::ConvexHull(hull) = tri else {
                continue;
            };
            if hull.vertices.len() < TRI_VERTS {
                continue;
            }
            // Mesh-frame triangle = centroid-relative verts + centroid.
            let (a, b, cc) = (
                hull.vertices[0] + *c,
                hull.vertices[1] + *c,
                hull.vertices[2] + *c,
            );
            let v = a.dot(b.cross(cc)) / TET_VOLUME_DIVISOR;
            vol += v;
            exx += v / TET_SECOND_MOMENT_DIVISOR
                * (a.x * a.x + b.x * b.x + cc.x * cc.x + a.x * b.x + b.x * cc.x + cc.x * a.x);
            eyy += v / TET_SECOND_MOMENT_DIVISOR
                * (a.y * a.y + b.y * b.y + cc.y * cc.y + a.y * b.y + b.y * cc.y + cc.y * a.y);
            ezz += v / TET_SECOND_MOMENT_DIVISOR
                * (a.z * a.z + b.z * b.z + cc.z * cc.z + a.z * b.z + b.z * cc.z + cc.z * a.z);
        }
        if vol.abs() < NEAR_ZERO {
            return Err(MeshError::DegenerateMesh);
        }
        let density = mass / vol.abs();
        Ok(Vec3::new(
            density * (eyy + ezz).abs(),
            density * (ezz + exx).abs(),
            density * (exx + eyy).abs(),
        ))
    }

    /// Bounding-box inertia fallback (degenerate/empty soup).
    fn fallback_inertia(&self, mass: f32) -> Vec3 {
        Shape::Box {
            half_extents: ((self.local_max - self.local_min) * HALF).max(Vec3::ZERO),
        }
        .inertia(mass)
    }

    /// Closest surface point (mesh-local): brute force over triangles.
    /// O(T) — the hot narrow path refines per winning triangle instead
    /// (see `distance::trimesh_convex`); this serves cold callers.
    pub(crate) fn closest_point_local(&self, p: Vec3) -> Vec3 {
        let mut best = p;
        let mut bd = f32::INFINITY;
        for (tri, c) in self.tris.iter().zip(self.centroids.iter()) {
            let Shape::ConvexHull(hull) = tri else {
                continue;
            };
            if hull.faces.is_empty() {
                continue;
            }
            for f in &hull.faces {
                let (a, b, cc) = (
                    hull.vertices[f.0.index()] + *c,
                    hull.vertices[f.1.index()] + *c,
                    hull.vertices[f.2.index()] + *c,
                );
                let (q, _, _, _) = crate::gjk::closest_triangle(a - p, b - p, cc - p);
                let q = q + p;
                let d = (p - q).length_squared();
                if d < bd {
                    best = q;
                    bd = d;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Half-extent of the unit cube used by mesh tests.
    const HALF: f32 = 0.5;

    /// Unit-cube mesh (outward wound, edge 1): shared by mesh tests.
    fn cube_mesh() -> TriMesh {
        let v = [
            Vec3::new(-HALF, -HALF, -HALF),
            Vec3::new(HALF, -HALF, -HALF),
            Vec3::new(-HALF, HALF, -HALF),
            Vec3::new(HALF, HALF, -HALF),
            Vec3::new(-HALF, -HALF, HALF),
            Vec3::new(HALF, -HALF, HALF),
            Vec3::new(-HALF, HALF, HALF),
            Vec3::new(HALF, HALF, HALF),
        ];
        let raw = [
            [0, 3, 1],
            [0, 2, 3],
            [4, 5, 7],
            [4, 7, 6],
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 7],
            [1, 7, 5],
            [0, 1, 5],
            [0, 5, 4],
            [2, 7, 3],
            [2, 6, 7],
        ];
        let triangles: Vec<Triangle> = raw.iter().map(|t| Triangle::from_raw(*t)).collect();
        TriMesh::from_triangles(&v, &triangles).expect("valid cube mesh")
    }

    #[test]
    fn trimesh_builds_bvh_and_derived_data() {
        let mesh = cube_mesh();
        assert_eq!(mesh.tris.len(), 12);
        assert_eq!(mesh.order.len(), 12);
        assert!(!mesh.nodes.is_empty());
        // Root bounds cover the cube.
        assert_vec3_close(mesh.nodes[0].min, Vec3::splat(-HALF));
        assert_vec3_close(mesh.nodes[0].max, Vec3::splat(HALF));
        assert_vec3_close(mesh.local_min, Vec3::splat(-HALF));
        assert_vec3_close(mesh.local_max, Vec3::splat(HALF));
        // Support radius = half space diagonal; min feature = edge.
        assert!(
            (mesh.bound_radius - 0.8660254).abs() < 1e-5,
            "{}",
            mesh.bound_radius
        );
        assert!(
            (mesh.min_feature - 1.0).abs() < 1e-5,
            "{}",
            mesh.min_feature
        );
        // Every triangle is centroid-relative: centroid + verts recover it.
        for (tri, _) in mesh.tris.iter().zip(mesh.centroids.iter()) {
            let Shape::ConvexHull(hull) = tri else {
                panic!("mesh tris are hulls");
            };
            assert_eq!(hull.vertices.len(), 3);
            assert_eq!(hull.faces.len(), 2);
        }
    }

    #[test]
    fn trimesh_cube_inertia_matches_box_golden() {
        // Closed cube, mass 2: I = (m/12)(1²+1²) = 1/3 per axis (Mirtich).
        let mesh = cube_mesh();
        assert_vec3_close(mesh.inertia(2.0), Vec3::splat(1.0 / 3.0));
    }

    #[test]
    fn trimesh_drops_degenerate_triangles() {
        let v = [
            Vec3::ZERO,
            Vec3::X,
            Vec3::new(2.0, 0.0, 0.0), // Collinear with the first two.
            Vec3::Y,
        ];
        let triangles = [Triangle::from_raw([0, 1, 2]), Triangle::from_raw([0, 1, 3])];
        let mesh = TriMesh::from_triangles(&v, &triangles).expect("degenerate drops, not errors");
        assert_eq!(mesh.tris.len(), 1);
    }

    #[test]
    fn trimesh_rejects_out_of_range_indices() {
        use crate::errors::MeshError;
        let v = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let err = TriMesh::from_triangles(&v, &[Triangle::from_raw([0, 1, 9])])
            .expect_err("dangling index errors");
        assert_eq!(
            err,
            MeshError::OutOfRangeIndex {
                triangle: 0,
                index: 9,
                vertices: 3,
            }
        );
    }

    #[test]
    fn trimesh_indexed_rejects_bad_count_and_delegates_range() {
        use crate::errors::MeshError;
        let v = [Vec3::ZERO, Vec3::X, Vec3::Y];
        assert_eq!(
            TriMesh::from_indexed(&v, &[0, 1]).expect_err("odd count errors"),
            MeshError::BadIndexCount { len: 2 }
        );
        let err = TriMesh::from_indexed(&v, &[0, 1, 9]).expect_err("dangling flat errors");
        assert_eq!(
            err,
            MeshError::OutOfRangeIndex {
                triangle: 0,
                index: 9,
                vertices: 3,
            }
        );
        let mesh = TriMesh::from_indexed(&v, &[0, 1, 2]).expect("flat triple builds");
        assert_eq!(mesh.tris.len(), 1);
    }

    #[test]
    fn triangle_round_trips_raw_and_indexes_corners() {
        let tri = Triangle::from_raw([2, 5, 7]);
        assert_eq!(tri.as_u32(), [2, 5, 7]);
        assert_eq!(tri.index(0), TriIndex::from_raw(2));
        assert_eq!(tri.index(1).as_u32(), 5);
        assert_eq!(tri.index(2).index(), 7);
        assert_eq!(TriIndex::from_raw(9).as_u32(), 9);
        assert_eq!(TriIndex::from_raw(9).index(), 9);
        assert_eq!(std::mem::size_of::<Triangle>(), 12);
    }

    #[test]
    fn trimesh_rejects_non_finite_vertices() {
        use crate::errors::MeshError;
        let v = [Vec3::ZERO, Vec3::X, Vec3::NAN];
        let err = TriMesh::from_triangles(&v, &[Triangle::from_raw([0, 1, 2])])
            .expect_err("non-finite errors");
        assert_eq!(err, MeshError::NonFiniteVertex { index: 2 });
    }

    #[test]
    fn convex_hull_rejects_non_finite_vertices() {
        use crate::errors::MeshError;
        let err = ConvexHull::from_vertices(vec![Vec3::ZERO, Vec3::NAN])
            .expect_err("non-finite hull errors");
        assert_eq!(err, MeshError::NonFiniteVertex { index: 1 });
    }

    #[test]
    fn hull_and_mesh_try_inertia_report_degenerate() {
        use crate::errors::MeshError;
        let flat =
            ConvexHull::from_vertices(vec![Vec3::ZERO, Vec3::X, Vec3::Y]).expect("flat builds");
        assert_eq!(flat.try_inertia(1.0), Err(MeshError::DegenerateMesh));
        assert!(flat.inertia(1.0).is_finite());
        let empty = TriMesh::from_indexed(&[], &[]).expect("empty soup builds");
        assert_eq!(empty.try_inertia(1.0), Err(MeshError::DegenerateMesh));
        assert!(empty.inertia(1.0).is_finite());
    }

    #[test]
    fn heightfield_try_height_at_rejects_broken_grid() {
        let broken = Heightfield {
            heights: vec![0.0],
            rows: 2,
            cols: 2,
            cell: 1.0,
        };
        assert!(broken.try_height_at(0.0, 0.0).is_none());
        assert_eq!(broken.height_at(0.0, 0.0), 0.0);
    }

    /// `Shape` has no unit tests at all (night gate, 2026-08-24: 50 missed
    /// mutants in `Shape::inertia` alone — every arithmetic op and shape
    /// arm was free to mutate). These are golden reference values computed
    /// independently from the closed-form formulas, not "run the same code
    /// with a tolerance" — they catch `*`<->`/`, `+`<->`-`, coefficient
    /// swaps (0.4/0.5/0.25/(1/12)/(1/3)) and axis-mismatch mutations.
    const EPS: f32 = 1e-5;

    fn assert_vec3_close(got: Vec3, want: Vec3) {
        assert!((got - want).length() < EPS, "got {got:?}, want {want:?}");
    }

    #[test]
    fn sphere_inertia_is_isotropic_and_matches_formula() {
        // I = (2/5) m r^2, same on all three axes.
        let shape = Shape::Sphere { radius: 3.0 };
        let i = shape.inertia(2.0);
        assert_vec3_close(i, Vec3::splat(7.2));
    }

    #[test]
    fn box_inertia_matches_closed_form_per_axis() {
        // I_x = (m/3)(h_y^2 + h_z^2), and cyclic; asymmetric half-extents
        // so a swapped axis or coefficient cannot hide behind symmetry.
        let shape = Shape::Box {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let i = shape.inertia(6.0);
        assert_vec3_close(i, Vec3::new(26.0, 20.0, 10.0));
    }

    #[test]
    fn capsule_inertia_matches_closed_form_and_is_symmetric_about_axis() {
        // Symmetric about the local +Y axis: i_x == i_z != i_y.
        let shape = Shape::Capsule {
            radius: 1.0,
            half_height: 2.0,
        };
        let i = shape.inertia(3.0);
        assert_vec3_close(i, Vec3::new(5.75, 1.5, 5.75));
        assert_eq!(i.x, i.z, "capsule inertia must be symmetric about +Y");
    }

    #[test]
    fn sphere_aabb_is_centered_cube() {
        let shape = Shape::Sphere { radius: 2.0 };
        let aabb = shape.aabb(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        assert_vec3_close(aabb.min, Vec3::new(-1.0, 0.0, 1.0));
        assert_vec3_close(aabb.max, Vec3::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn box_aabb_at_identity_matches_half_extents() {
        let shape = Shape::Box {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let aabb = shape.aabb(Vec3::ZERO, Quat::IDENTITY);
        assert_vec3_close(aabb.min, Vec3::new(-1.0, -2.0, -3.0));
        assert_vec3_close(aabb.max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn box_aabb_grows_when_rotated_45_degrees() {
        // A unit cube rotated 45° about Z: the AABB half-extent on x/y
        // grows to half_extent * sqrt(2), z unchanged. Exact known value
        // pins the rotation being applied to the extents at all (a mutant
        // dropping the rotation entirely would keep the AABB at (1, 1, 1)).
        let shape = Shape::Box {
            half_extents: Vec3::splat(1.0),
        };
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let aabb = shape.aabb(Vec3::ZERO, rot);
        let expected = std::f32::consts::SQRT_2;
        assert!((aabb.max.x - expected).abs() < 1e-4, "{:?}", aabb.max);
        assert!((aabb.max.y - expected).abs() < 1e-4, "{:?}", aabb.max);
        assert!((aabb.max.z - 1.0).abs() < 1e-4, "{:?}", aabb.max);
    }

    #[test]
    fn capsule_aabb_extends_along_local_y_plus_radius() {
        let shape = Shape::Capsule {
            radius: HALF,
            half_height: 2.0,
        };
        let aabb = shape.aabb(Vec3::ZERO, Quat::IDENTITY);
        // Along Y: half_height + radius. On X/Z: just the radius shell.
        assert_vec3_close(aabb.min, Vec3::new(-HALF, -2.5, -HALF));
        assert_vec3_close(aabb.max, Vec3::new(HALF, 2.5, HALF));
    }

    #[test]
    fn custom_mesh_has_no_implicit_collider_contract() {
        // Convex primitives answer GJK support directly; heightfields and
        // triangle meshes opt out (they dispatch before GJK — see
        // `Shape::has_gjk_support`). A custom render soup must be built
        // explicitly: an empty soup is an empty mesh (no contact), never a
        // fallback sphere.
        for shape in [
            Shape::Sphere { radius: HALF },
            Shape::Box {
                half_extents: Vec3::splat(HALF),
            },
            Shape::Capsule {
                radius: 0.3,
                half_height: HALF,
            },
            Shape::Cylinder {
                radius: 0.3,
                half_height: HALF,
            },
            Shape::Cone {
                radius: 0.3,
                half_height: HALF,
            },
            Shape::ConvexHull(
                ConvexHull::from_vertices(vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z])
                    .expect("valid test hull"),
            ),
        ] {
            assert!(shape.has_gjk_support(), "{shape:?} must answer GJK support");
        }
        let terrain = Shape::Heightfield(
            Heightfield::new(vec![0.0], 1, 1, 1.0).expect("valid test heightfield"),
        );
        assert!(!terrain.has_gjk_support());
        let empty = TriMesh::from_indexed(&[], &[]).expect("empty soup builds");
        assert!(
            empty.tris.is_empty(),
            "empty soup builds an empty mesh, never a placeholder"
        );
        assert!(!Shape::TriMesh(empty).has_gjk_support());
    }

    /// Closed supported-pair list: every pair is `Supported` except the two
    /// concave-concave cases, which must read back as the loud
    /// `UnsupportedPair` marker (never a silent separation). One
    /// representative per variant, both orders.
    #[test]
    fn pair_support_lists_supported_pairs_and_marks_undefined() {
        use super::PairSupport;
        let sphere = Shape::Sphere { radius: HALF };
        let boxed = Shape::Box {
            half_extents: Vec3::splat(HALF),
        };
        let capsule = Shape::Capsule {
            radius: 0.3,
            half_height: HALF,
        };
        let cylinder = Shape::Cylinder {
            radius: 0.3,
            half_height: HALF,
        };
        let cone = Shape::Cone {
            radius: 0.3,
            half_height: HALF,
        };
        let hull = Shape::ConvexHull(
            ConvexHull::from_vertices(vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z])
                .expect("valid test hull"),
        );
        let terrain = Shape::Heightfield(
            Heightfield::new(vec![0.0; 4], 2, 2, 1.0).expect("valid test heightfield"),
        );
        let mesh = Shape::TriMesh(cube_mesh());
        let convex = [&sphere, &boxed, &capsule, &cylinder, &cone, &hull];
        // Convex-convex, convex-mesh, convex-terrain, mesh-terrain: supported.
        let all: Vec<&Shape> = convex.iter().copied().chain([&mesh, &terrain]).collect();
        for &a in &all {
            for &b in &all {
                // Mesh-vs-mesh and terrain-vs-terrain are the only gaps.
                let undefined = matches!(
                    (a, b),
                    (Shape::TriMesh(_), Shape::TriMesh(_))
                        | (Shape::Heightfield(_), Shape::Heightfield(_))
                );
                assert_eq!(
                    a.pair_support(b),
                    if undefined {
                        PairSupport::UnsupportedPair
                    } else {
                        PairSupport::Supported
                    },
                    "{a:?} vs {b:?}"
                );
            }
        }
        // Symmetry: the marker does not depend on argument order.
        assert_eq!(mesh.pair_support(&terrain), PairSupport::Supported);
        assert_eq!(terrain.pair_support(&mesh), PairSupport::Supported);
    }
}
