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

use crate::math::AABB;

/// Convex collision primitives supported by the builtin engine.
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
}

/// Explicit convex polyhedron: deduplicated local vertices plus outward
/// faces triangulated at construction. Built with
/// [`ConvexHull::from_vertices`]; faces exist only for small hulls (see the
/// constructor cap) — GJK support queries need vertices alone.
#[derive(Debug, Clone)]
pub struct ConvexHull {
    /// Deduplicated vertices in body-local space.
    pub vertices: Vec<Vec3>,
    /// Outward-oriented triangles as vertex indices. Empty when the input
    /// exceeded the triangulation cap (support/GJK still work; face-slab
    /// raycasts report no hit).
    pub faces: Vec<[u32; 3]>,
}

/// Heightfield terrain: a regular X/Z grid of heights, centered on the
/// body origin (+Y up). `heights[row * cols + col]`, `rows` along +Z,
/// `cols` along +X, uniform `cell` spacing.
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

impl Shape {
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
        }
    }

    /// Diagonal (body-frame) inertia tensor for a given mass.
    /// One entry per principal axis; a sphere is isotropic, a box and a
    /// capsule are symmetric about their local +Y.
    pub fn inertia(&self, mass: f32) -> Vec3 {
        match self {
            Shape::Sphere { radius } => {
                let i = 0.4 * mass * radius * radius;
                Vec3::splat(i)
            }
            Shape::Box { half_extents } => {
                let (x, y, z) = (
                    right2(half_extents.x),
                    right2(half_extents.y),
                    right2(half_extents.z),
                );
                Vec3::new(
                    (mass / 12.0) * (y + z),
                    (mass / 12.0) * (z + x),
                    (mass / 12.0) * (x + y),
                )
            }
            Shape::Capsule {
                radius,
                half_height,
            } => {
                // h = total half-length along the axis (excluding radius), like a cylinder.
                let h = half_height;
                let r = radius;
                // Uniform about the axis:
                let i_y = 0.5 * mass * r * r;
                // Perpendicular, approximating a cylinder + sphere caps.
                let i_xz = 0.25 * mass * (r * r) + (mass / 3.0) * h * r + 0.25 * mass * h * h;
                Vec3::new(i_xz, i_y, i_xz)
            }
            Shape::Cylinder {
                radius,
                half_height,
            } => {
                // Solid cylinder, height H = 2 * half_height: standard
                // I_y = M*R^2/2, I_xz = M*(3*R^2 + H^2)/12.
                let (r, h) = (radius, half_height);
                let i_y = 0.5 * mass * r * r;
                let i_xz = mass * (3.0 * r * r + 4.0 * h * h) / 12.0;
                Vec3::new(i_xz, i_y, i_xz)
            }
            Shape::Cone {
                radius,
                half_height,
            } => {
                // Solid cone about the geometric center (mid-height), not
                // the center of mass (which sits half_height/2 toward the
                // base): axial I_y = 3/10*M*R^2; transverse folds the
                // parallel-axis shift in —
                // I_xz = 3*M/80*(4*R^2 + H^2) + M*(H/4)^2, H = 2*h.
                // Both formulas verified by Monte-Carlo integration
                // (see the cone check in the P3 work notes).
                let (r, h) = (radius, half_height);
                let i_y = 0.3 * mass * r * r;
                let i_xz = mass * (3.0 * r * r + 8.0 * h * h) / 20.0;
                Vec3::new(i_xz, i_y, i_xz)
            }
            Shape::ConvexHull(hull) => hull.inertia(mass),
            Shape::Heightfield(hf) => {
                // Terrain is static in practice; report the inertia of the
                // bounding box so a dynamic heightfield at least tumbles
                // like its bounds instead of dividing by zero.
                let e = hf.local_extents();
                Shape::Box { half_extents: e }.inertia(mass)
            }
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
        };
        pos + rot * q
    }
}

/// Closest boundary point on a box (local frame): clamp for exterior
/// queries, nearest-face projection for interior ones.
fn closest_box_point(half_extents: Vec3, p: Vec3) -> Vec3 {
    let c = p.clamp(-half_extents, half_extents);
    if (c - p).length_squared() > 1e-18 {
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
    let wall = if len > 1e-9 {
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
        let s = if len > 1e-9 {
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
    if half_height <= 1e-9 {
        return Vec3::new(p.x.clamp(-radius, radius), 0.0, p.z.clamp(-radius, radius));
    }
    let radial = Vec2::new(p.x, p.z);
    let len = radial.length();
    let y_c = p.y.clamp(-half_height, half_height);
    // Wall radius at the clamped height: r * (h - y) / (2h).
    let r_at = radius * (half_height - y_c) / (2.0 * half_height);
    let wall = if len > 1e-9 {
        Vec3::new(radial.x / len * r_at, y_c, radial.y / len * r_at)
    } else {
        // On the axis: the wall circle is equidistant — pick +X
        // deterministically.
        Vec3::new(r_at, y_c, 0.0)
    };
    let s = if len > 1e-9 {
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
    let mut best = hull.vertices[hull.faces[0][0] as usize];
    let mut bd = (p - best).length_squared();
    for f in &hull.faces {
        let (a, b, c) = (
            hull.vertices[f[0] as usize],
            hull.vertices[f[1] as usize],
            hull.vertices[f[2] as usize],
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
    let col = ((p.x / hf.cell + (hf.cols - 1) as f32 * 0.5).floor() as isize)
        .clamp(0, hf.cols as isize - 1) as usize;
    let row = ((p.z / hf.cell + (hf.rows - 1) as f32 * 0.5).floor() as isize)
        .clamp(0, hf.rows as isize - 1) as usize;
    let h = hf.heights[row * hf.cols + col];
    let (y_min, _) = hf.height_range();
    let y_low = if h - y_min >= 1e-4 {
        y_min
    } else {
        h - hf.cell.max(1e-3)
    };
    let x_origin = -((hf.cols - 1) as f32) * 0.5 * hf.cell;
    let z_origin = -((hf.rows - 1) as f32) * 0.5 * hf.cell;
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
    if (c - p).length_squared() > 1e-18 {
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
    let mid = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
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
    pub fn from_vertices(points: Vec<Vec3>) -> Self {
        let mut vertices: Vec<Vec3> = Vec::with_capacity(points.len());
        for p in points {
            if !vertices
                .iter()
                .any(|q| (*q - p).length_squared() < Self::WELD_EPS * Self::WELD_EPS)
            {
                vertices.push(p);
            }
        }
        let faces = if vertices.len() >= 4 && vertices.len() <= Self::FACE_CAP {
            Self::triangulate(&vertices)
        } else {
            Vec::new()
        };
        Self { vertices, faces }
    }

    /// Outward triangles: every ordered triple whose plane keeps all other
    /// points on the non-positive side (coplanar points tolerated, so quad
    /// faces triangulate into overlapping-but-harmless triangles sharing
    /// one normal). O(n^4), index-ordered, deterministic.
    fn triangulate(vertices: &[Vec3]) -> Vec<[u32; 3]> {
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
                    if normal.length_squared() < 1e-12 {
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
                        faces.push([i as u32, j as u32, k as u32]);
                    }
                }
            }
        }
        faces
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
    pub fn inertia(&self, mass: f32) -> Vec3 {
        if self.faces.is_empty() || mass <= 0.0 {
            let e = self.local_box();
            return Shape::Box { half_extents: e }.inertia(mass);
        }
        // Signed volumes and second moments of tetrahedra (origin, a, b, c)
        // summed over faces (Mirtich-style, diagonal only).
        let mut vol = 0.0f32;
        let mut exx = 0.0f32;
        let mut eyy = 0.0f32;
        let mut ezz = 0.0f32;
        for f in &self.faces {
            let (a, b, c) = (
                self.vertices[f[0] as usize],
                self.vertices[f[1] as usize],
                self.vertices[f[2] as usize],
            );
            let v = a.dot(b.cross(c)) / 6.0;
            vol += v;
            // ∫x² over tet (origin,a,b,c) = v/10 * (xa²+xb²+xc²+xa·xb+...);
            // diagonal-only accumulation:
            exx +=
                v / 10.0 * (a.x * a.x + b.x * b.x + c.x * c.x + a.x * b.x + b.x * c.x + c.x * a.x);
            eyy +=
                v / 10.0 * (a.y * a.y + b.y * b.y + c.y * c.y + a.y * b.y + b.y * c.y + c.y * a.y);
            ezz +=
                v / 10.0 * (a.z * a.z + b.z * b.z + c.z * c.z + a.z * b.z + b.z * c.z + c.z * a.z);
        }
        if vol.abs() < 1e-9 {
            let e = self.local_box();
            return Shape::Box { half_extents: e }.inertia(mass);
        }
        let density = mass / vol.abs();
        // I_xx = ∫(y²+z²), cyclic. Signed accumulation keeps orientation
        // consistent; density takes the absolute volume.
        Vec3::new(
            density * (eyy + ezz).abs(),
            density * (ezz + exx).abs(),
            density * (exx + eyy).abs(),
        )
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
    /// Bilinear height at local (x, z), clamped to the grid edge.
    /// Deterministic: pure arithmetic over the stored samples.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        if self.heights.len() != self.rows * self.cols || self.rows == 0 || self.cols == 0 {
            return 0.0;
        }
        let fx = (x / self.cell + (self.cols - 1) as f32 * 0.5).clamp(0.0, (self.cols - 1) as f32);
        let fz = (z / self.cell + (self.rows - 1) as f32 * 0.5).clamp(0.0, (self.rows - 1) as f32);
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
            (self.cols.max(1) - 1) as f32 * 0.5 * self.cell,
            ((hi - lo) * 0.5).max(0.0),
            (self.rows.max(1) - 1) as f32 * 0.5 * self.cell,
        )
    }

    /// Local bounding center (x/z centered, y at mid-range).
    pub fn local_center(&self) -> Vec3 {
        let (lo, hi) = self.height_range();
        Vec3::new(0.0, (lo + hi) * 0.5, 0.0)
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

#[cfg(test)]
mod tests {
    use super::*;

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
        // I_x = (m/12)(h_y^2 + h_z^2), and cyclic; asymmetric half-extents
        // so a swapped axis or coefficient cannot hide behind symmetry.
        let shape = Shape::Box {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let i = shape.inertia(6.0);
        assert_vec3_close(i, Vec3::new(6.5, 5.0, 2.5));
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
            radius: 0.5,
            half_height: 2.0,
        };
        let aabb = shape.aabb(Vec3::ZERO, Quat::IDENTITY);
        // Along Y: half_height + radius. On X/Z: just the radius shell.
        assert_vec3_close(aabb.min, Vec3::new(-0.5, -2.5, -0.5));
        assert_vec3_close(aabb.max, Vec3::new(0.5, 2.5, 0.5));
    }
}
