//! The six half-space planes of one cluster, and the sphere test against them.
//!
//! Named fields, not an array: an array of planes becomes an `alloca` in the
//! LLVM IR, which the translator rejects. Six scalars stay in SSA.

use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::exp;

use crate::CameraView;
use crate::DEPTH_NEAR;
use crate::FAR_LIMIT;
use crate::LOG_RATIO;
use crate::SLICES;
use crate::TILES_X;
use crate::TILES_Y;

/// One plane `(n.xyz, d)`.
#[derive(Clone, Copy)]
pub(crate) struct Plane {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

/// The six planes, in the same order as the hand-written WGSL.
#[derive(Clone, Copy)]
pub(crate) struct Planes {
    /// Left.
    pub(crate) p0: Plane,
    /// Right.
    pub(crate) p1: Plane,
    /// Bottom.
    pub(crate) p2: Plane,
    /// Top.
    pub(crate) p3: Plane,
    /// Near.
    pub(crate) p4: Plane,
    /// Far.
    pub(crate) p5: Plane,
}

/// Far view-depth of slice `index`. The last slice runs out to [`FAR_LIMIT`].
#[inline(always)]
fn slice_far(index: u32) -> f32 {
    if index + 1 >= SLICES {
        FAR_LIMIT
    } else {
        let along = (index as f32) / ((SLICES - 1) as f32);
        DEPTH_NEAR * exp(along * LOG_RATIO)
    }
}

/// Index of the `w` component, and of the `w` column of `view_proj`.
const LANE_W: usize = 3;

/// Lane `LANE` of every `view_proj` column: one row of the matrix.
#[inline(always)]
fn row<const LANE: usize>(camera: &Uniform<CameraView, 0, 0>) -> Plane {
    Plane {
        x: camera.vec4_lane::<0, LANE>(),
        y: camera.vec4_lane::<1, LANE>(),
        z: camera.vec4_lane::<2, LANE>(),
        w: camera.vec4_lane::<LANE_W, LANE>(),
    }
}

#[inline(always)]
fn mix(left: Plane, scale_left: f32, right: Plane, scale_right: f32) -> Plane {
    Plane {
        x: scale_left * left.x + scale_right * right.x,
        y: scale_left * left.y + scale_right * right.y,
        z: scale_left * left.z + scale_right * right.z,
        w: scale_left * left.w + scale_right * right.w,
    }
}

/// Near and far NDC edges of tile `index` in a row of `tiles`.
#[inline(always)]
fn ndc_edges(index: u32, tiles: u32) -> (f32, f32) {
    let tiles = tiles as f32;
    let start = -1.0 + 2.0 * (index as f32) / tiles;
    let end = -1.0 + 2.0 * ((index + 1) as f32) / tiles;
    (start, end)
}

/// Planes of the cluster at tile `(tx, ty)` and slice `tz`.
// qual:allow(dry, boilerplate) reason: "six named plane fields stay scalars; an array of planes becomes an alloca the translator rejects"
#[inline(always)]
pub(crate) fn cluster_planes(
    camera: &Uniform<CameraView, 0, 0>,
    tx: u32,
    ty: u32,
    tz: u32,
) -> Planes {
    let (x0, x1) = ndc_edges(tx, TILES_X);
    let (y0, y1) = ndc_edges(ty, TILES_Y);
    let row_x = row::<0>(camera);
    let row_y = row::<1>(camera);
    let row_w = row::<LANE_W>(camera);
    let z_lo = if tz >= 1 { slice_far(tz - 1) } else { 0.0 };
    let z_hi = slice_far(tz);
    Planes {
        p0: mix(row_x, 1.0, row_w, -x0),
        p1: mix(row_w, x1, row_x, -1.0),
        p2: mix(row_y, 1.0, row_w, -y0),
        p3: mix(row_w, y1, row_y, -1.0),
        p4: Plane {
            x: row_w.x,
            y: row_w.y,
            z: row_w.z,
            w: row_w.w - z_lo,
        },
        p5: Plane {
            x: -row_w.x,
            y: -row_w.y,
            z: -row_w.z,
            w: z_hi - row_w.w,
        },
    }
}

#[inline(always)]
fn beyond(plane: &Plane, center: Vec3, radius: f32) -> bool {
    let normal = Vec3::new(plane.x, plane.y, plane.z);
    normal.dot(center) + plane.w < -(radius * normal.length())
}

/// `true` when the sphere is fully outside at least one plane.
#[inline(always)]
pub(crate) fn outside(center: Vec3, radius: f32, planes: &Planes) -> bool {
    beyond(&planes.p0, center, radius)
        || beyond(&planes.p1, center, radius)
        || beyond(&planes.p2, center, radius)
        || beyond(&planes.p3, center, radius)
        || beyond(&planes.p4, center, radius)
        || beyond(&planes.p5, center, radius)
}
