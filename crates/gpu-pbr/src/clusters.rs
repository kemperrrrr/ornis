//! The clustered-light lookup: fragment → cluster cell.
//!
//! The grid is 16 × 9 tiles of the screen and 24 exponential depth slices. The
//! fragment projects its world position, maps the NDC tile to a cell and the
//! view depth to a slice, and the caller then reads that cell's `(offset,
//! count)` span. Keeping the index here means the caller needs one `u32` and no
//! second copy of the constants.

use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::ceil;
use ornis_gpu_shader::floor;
use ornis_gpu_shader::ln;

use crate::layout::CAMERA_BINDING;
use crate::layout::CAMERA_VIEW_PROJ;
use crate::layout::CLUSTER_DEPTH_NEAR;
use crate::layout::CLUSTER_SLICE_LOOKUP_EPS;
use crate::layout::CLUSTER_SLICE_SCALE;
use crate::layout::CLUSTERS_X;
use crate::layout::CLUSTERS_Y;
use crate::layout::CLUSTERS_Z;
use crate::layout::Camera;
use crate::layout::GROUP;
use crate::layout::LAST_COLUMN;
use crate::layout::NDC_TO_UV_HALF;
use crate::layout::W_LANE_U;
use crate::layout::X_LANE_U;
use crate::layout::Y_LANE_U;

/// Index of the cluster cell a world position falls into.
///
/// The depth slice is exponential — `log(w / near)` — so the cells stay roughly
/// the same size in screen space as they recede. A position behind the near
/// plane clamps into the first tile and the first slice instead of wrapping: the
/// GPU conversion saturates, which is what the reference clamps after.
// qual:allow(test_quality, untested) reason: "reads the camera uniform; the host build has no uniform, and the deferred lighting device gate covers the cluster lookup"
#[inline(always)]
pub fn cluster_index(camera: &Uniform<Camera, GROUP, CAMERA_BINDING>, world_pos: Vec3) -> u32 {
    let clip_x = clip_lane::<X_LANE_U>(camera, world_pos);
    let clip_y = clip_lane::<Y_LANE_U>(camera, world_pos);
    let clip_w = clip_lane::<W_LANE_U>(camera, world_pos);
    let w = if clip_w > CLUSTER_DEPTH_NEAR {
        clip_w
    } else {
        CLUSTER_DEPTH_NEAR
    };
    let tile_x = (clip_x / w) * NDC_TO_UV_HALF + NDC_TO_UV_HALF;
    let tile_y = (clip_y / w) * NDC_TO_UV_HALF + NDC_TO_UV_HALF;
    let x = clamp_index((floor(tile_x * CLUSTERS_X as f32)) as u32, CLUSTERS_X);
    let y = clamp_index((floor(tile_y * CLUSTERS_Y as f32)) as u32, CLUSTERS_Y);
    let slice = clamp_index(
        (ceil(CLUSTER_SLICE_SCALE * ln(w / CLUSTER_DEPTH_NEAR) - CLUSTER_SLICE_LOOKUP_EPS)) as u32,
        CLUSTERS_Z,
    );
    x + y * CLUSTERS_X + slice * CLUSTERS_X * CLUSTERS_Y
}

/// `index` clamped into `[0, count - 1]`.
#[inline(always)]
fn clamp_index(index: u32, count: u32) -> u32 {
    if index >= count { count - 1 } else { index }
}

/// Lane `LANE` of `view_proj · (x, y, z, 1)`.
///
/// The matrix is a uniform, so each column is read by its constant lane and
/// scaled by that column's coordinate — reading the columns as rows would
/// transpose the projection.
#[inline(always)]
fn clip_lane<const LANE: usize>(
    camera: &Uniform<Camera, GROUP, CAMERA_BINDING>,
    world_pos: Vec3,
) -> f32 {
    let c0 = camera.vec4_lane::<CAMERA_VIEW_PROJ, LANE>();
    let c1 = camera.vec4_lane::<{ CAMERA_VIEW_PROJ + 1 }, LANE>();
    let c2 = camera.vec4_lane::<{ CAMERA_VIEW_PROJ + 2 }, LANE>();
    let c3 = camera.vec4_lane::<{ CAMERA_VIEW_PROJ + LAST_COLUMN }, LANE>();
    c0 * world_pos.x + c1 * world_pos.y + c2 * world_pos.z + c3
}

#[cfg(test)]
mod tests {
    use super::clamp_index;

    /// Cluster lookup clamps instead of wrapping: an index past the end lands
    /// on the last slice, and everything inside is untouched.
    #[test]
    fn clamp_index_saturates_at_the_last_slice() {
        assert_eq!(clamp_index(0, 4), 0);
        assert_eq!(clamp_index(3, 4), 3);
        assert_eq!(clamp_index(4, 4), 3);
        assert_eq!(clamp_index(u32::MAX, 4), 3);
        assert_eq!(clamp_index(0, 1), 0);
    }
}
