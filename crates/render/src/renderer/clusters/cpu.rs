//! CPU reference of the cluster assignment (#130) — test-only, like
//! the extraction culler's CPU mirror: the GPU parity gate compares the
//! compute-pass output against it. Mirrors the WGSL pass exactly: same
//! plane math, same bound spheres, same per-cluster cap, spans claimed
//! in cluster order (the GPU claims through one atomic counter —
//! content per cluster is identical, span placement is not).
use super::*;
use crate::shaders::helpers::NDC_TO_UV_HALF;

/// Upper view depth of `slice`: `near * (far/near)^(slice/(S-1))`, with
/// the last slice extending to [`CLUSTER_FAR_LIMIT`].
pub(crate) fn cluster_slice_z_max(slice: u32) -> f32 {
    if slice + 1 >= CLUSTER_SLICE_COUNT {
        return CLUSTER_FAR_LIMIT;
    }
    let step = slice as f32 / (CLUSTER_SLICE_COUNT as f32 - 1.0);
    CLUSTER_DEPTH_NEAR * (step * cluster_log_ratio()).exp()
}

/// Lower view depth of `slice` (0 for slice 0).
pub(crate) fn cluster_slice_z_min(slice: u32) -> f32 {
    if slice == 0 {
        0.0
    } else {
        cluster_slice_z_max(slice - 1)
    }
}

/// Cluster index of a world position: NDC tile × view-depth slice. The
/// CPU mirror of the `cluster_light_span` WGSL lookup (behind-eye and
/// NaN inputs fall into tile 0 / slice 0 exactly like the shader).
pub(crate) fn cluster_index_for(world_pos: glam::Vec3, view_proj: &Mat4) -> u32 {
    let clip = *view_proj * world_pos.extend(1.0);
    // Behind-eye/NaN-safe depth: `max` drops NaNs and negatives onto the
    // near plane, so tile/slice math stays in range.
    let w = clip.w.max(CLUSTER_DEPTH_NEAR);
    let ndc = clip.truncate() / w;
    let tile_f = (ndc.truncate() * NDC_TO_UV_HALF + NDC_TO_UV_HALF)
        * glam::Vec2::new(CLUSTER_TILE_COUNT_X as f32, CLUSTER_TILE_COUNT_Y as f32);
    let tile = glam::UVec2::new(tile_f.x.floor() as u32, tile_f.y.floor() as u32).clamp(
        glam::UVec2::ZERO,
        glam::UVec2::new(CLUSTER_TILE_COUNT_X - 1, CLUSTER_TILE_COUNT_Y - 1),
    );
    let slice = (cluster_slice_scale() * (w / CLUSTER_DEPTH_NEAR).ln())
        .ceil()
        .clamp(0.0, CLUSTER_SLICE_COUNT as f32 - 1.0) as u32;
    tile.x + tile.y * CLUSTER_TILE_COUNT_X + slice * CLUSTER_TILE_COUNT_X * CLUSTER_TILE_COUNT_Y
}

/// (tile_x, tile_y, slice) encoded in a cluster index.
pub(crate) fn cluster_tile_slice(cluster: u32) -> (u32, u32, u32) {
    let tile_x = cluster % CLUSTER_TILE_COUNT_X;
    let tile_y = (cluster / CLUSTER_TILE_COUNT_X) % CLUSTER_TILE_COUNT_Y;
    let slice = cluster / (CLUSTER_TILE_COUNT_X * CLUSTER_TILE_COUNT_Y);
    (tile_x, tile_y, slice)
}

/// Six cluster half-space planes packed as `(n.xyz, d)` rows: NDC tile
/// edges from the clip-matrix rows, view-depth bounds from the `w` row.
/// A sphere touches the cluster iff `|n·c + d| <= r·|n|` on every plane.
pub(crate) fn cluster_planes(tile_x: u32, tile_y: u32, slice: u32, view_proj: &Mat4) -> [Vec4; 6] {
    let x0 = -1.0 + 2.0 * tile_x as f32 / CLUSTER_TILE_COUNT_X as f32;
    let x1 = -1.0 + 2.0 * (tile_x + 1) as f32 / CLUSTER_TILE_COUNT_X as f32;
    let y0 = -1.0 + 2.0 * tile_y as f32 / CLUSTER_TILE_COUNT_Y as f32;
    let y1 = -1.0 + 2.0 * (tile_y + 1) as f32 / CLUSTER_TILE_COUNT_Y as f32;
    let m = view_proj.to_cols_array_2d();
    let row = |i: usize| glam::Vec4::new(m[0][i], m[1][i], m[2][i], m[3][i]);
    let row_x = row(0);
    let row_y = row(1);
    let row_w = row(3);
    let z_lo = cluster_slice_z_min(slice);
    let z_hi = cluster_slice_z_max(slice);
    [
        row_x - x0 * row_w,
        x1 * row_w - row_x,
        row_y - y0 * row_w,
        y1 * row_w - row_y,
        glam::Vec4::new(row_w.x, row_w.y, row_w.z, row_w.w - z_lo),
        -glam::Vec4::new(row_w.x, row_w.y, row_w.z, row_w.w - z_hi),
    ]
}

/// Sphere (`center`, `radius`) vs the six cluster planes.
pub(crate) fn sphere_touches_planes(center: glam::Vec3, radius: f32, planes: &[Vec4; 6]) -> bool {
    planes.iter().all(|plane| {
        let n = plane.truncate();
        (n.dot(center) + plane.w).abs() <= radius * n.length()
    })
}

/// Conservative bounding sphere of a packed light: point lights use
/// `(position, range)`; spots use the minimal cone bounding sphere
/// `center = apex + axis·R`, `R = range·(1 + tan²θ)/2` (falls back to
/// the point bound for cones wider than
/// `acos(SPOT_BOUND_COS_FLOOR)`).
pub(crate) fn light_bound_sphere(light: &GpuLight) -> (glam::Vec3, f32) {
    let center = glam::Vec3::from_array([
        light.position[0],
        light.position[1],
        light.position[2],
    ]);
    let range = light.params[0];
    if light.kind[0] < LIGHT_KIND_SPOT {
        return (center, range);
    }
    let cos_outer = light.params[2];
    if cos_outer < SPOT_BOUND_COS_FLOOR {
        return (center, range);
    }
    let tan_sq = (1.0 - cos_outer * cos_outer) / (cos_outer * cos_outer);
    let radius = range * (1.0 + tan_sq) * 0.5;
    let axis = glam::Vec3::from_array([
        light.direction[0],
        light.direction[1],
        light.direction[2],
    ])
    .try_normalize()
    .unwrap_or(glam::Vec3::Z);
    (center + axis * radius, radius)
}

/// CPU reference output: the per-cluster light lists plus the shared
/// index-list overflow count (span placement is atomic-order on the
/// GPU and intentionally unchecked here — content per cluster is the
/// comparable part).
pub(crate) struct ClusterAssignment {
    /// Per-cluster light indices (ascending storage order).
    pub(crate) lists: Vec<Vec<u32>>,
    /// Index-list words that did not fit (reported, never fatal).
    pub(crate) overflow: u32,
}

/// Run the CPU reference for packed `records` under `view_proj`.
/// Lights below `head_count` are the uniform-window records and never
/// enter the lists.
pub(crate) fn cpu_assign_lights_to_clusters(
    records: &[GpuLight],
    head_count: u32,
    view_proj: &Mat4,
) -> ClusterAssignment {
    let mut lists = vec![Vec::new(); CLUSTER_COUNT as usize];
    for (cluster, list) in lists.iter_mut().enumerate() {
        let (tile_x, tile_y, slice) = cluster_tile_slice(cluster as u32);
        let planes = cluster_planes(tile_x, tile_y, slice, view_proj);
        for (index, light) in records.iter().enumerate().skip(head_count as usize) {
            let (center, radius) = light_bound_sphere(light);
            if sphere_touches_planes(center, radius, &planes) {
                list.push(index as u32);
            }
        }
        list.truncate(MAX_LIGHTS_PER_CLUSTER as usize);
    }
    // The GPU claims list spans through one atomic counter and counts
    // everything past the shared capacity as overflow; the lists above
    // are already per-cluster capped at MAX_LIGHTS_PER_CLUSTER.
    let total: u32 = lists.iter().map(|list| list.len() as u32).sum();
    let overflow = total.saturating_sub(CLUSTER_INDEX_LIST_LEN);
    ClusterAssignment { lists, overflow }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    /// Perspective frame matching the culling tests.
    fn view_proj() -> Mat4 {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 2.5, 9.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(60.0f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        proj * view
    }

    #[test]
    fn grid_consts_agree_with_design() {
        assert_eq!(CLUSTER_COUNT, 3456);
        assert_eq!(CLUSTER_WORKGROUPS, 54);
        assert_eq!(CLUSTER_INDEX_LIST_LEN, 442368);
        // Workgroups cover the grid exactly (no remainder thread).
        assert_eq!(CLUSTER_COUNT % CLUSTER_WORKGROUP_SIZE, 0);
    }

    #[test]
    fn slice_boundaries_are_monotonic_and_scale_inverts_them() {
        let mut previous = 0.0f32;
        for slice in 0..CLUSTER_SLICE_COUNT {
            let lo = cluster_slice_z_min(slice);
            let hi = cluster_slice_z_max(slice);
            assert!(lo < hi, "slice {slice} is empty");
            assert!(lo >= previous, "slice {slice} overlaps its neighbor");
            previous = hi;
        }
        assert_eq!(cluster_slice_z_max(0), CLUSTER_DEPTH_NEAR);
        assert_eq!(cluster_slice_z_max(CLUSTER_SLICE_COUNT - 1), CLUSTER_FAR_LIMIT);
        // The lookup scale inverts the boundary map: every boundary maps
        // back onto its own slice (ceil semantics: the boundary belongs
        // to the slice it closes).
        for slice in 0..CLUSTER_SLICE_COUNT - 1 {
            let depth = cluster_slice_z_max(slice);
            let looked_up = cluster_slice_scale() * (depth / CLUSTER_DEPTH_NEAR).ln();
            assert_eq!(
                looked_up.ceil() as u32,
                slice,
                "boundary of slice {slice} maps to {}",
                looked_up.ceil()
            );
        }
    }

    #[test]
    fn cluster_index_for_partitions_the_view() {
        let vp = view_proj();
        // In front of the camera and behind it: both map to valid
        // clusters (behind-eye falls into slice 0).
        let front = cluster_index_for(glam::Vec3::new(0.0, 0.0, 0.0), &vp);
        assert!(front < CLUSTER_COUNT);
        let behind = cluster_index_for(glam::Vec3::new(0.0, 0.0, 20.0), &vp);
        assert!(behind < CLUSTER_COUNT);
        // Depth ordering: the straight-ahead center point walks through
        // slices as it recedes, and far past the limit clamps to the
        // last slice.
        let near = cluster_tile_slice(cluster_index_for(glam::Vec3::new(0.0, 0.0, 7.0), &vp));
        let far = cluster_tile_slice(cluster_index_for(glam::Vec3::new(0.0, 0.0, -90.0), &vp));
        assert!(far.2 > near.2, "depth must grow the slice");
        let limit = cluster_tile_slice(cluster_index_for(glam::Vec3::new(0.0, 0.0, -1.0e8), &vp));
        assert_eq!(limit.2, CLUSTER_SLICE_COUNT - 1);
    }

    #[test]
    fn assignment_lists_a_center_light_in_a_few_clusters() {
        let vp = view_proj();
        let mut light = GpuLight::zeroed();
        light.kind = [LIGHT_KIND_POINT, 0.0, 0.0, 0.0];
        light.position = [0.0, 0.0, 0.0, 1.0];
        light.params = [2.0, 0.0, 0.0, -1.0];
        // The head window never enters the lists.
        let head = cpu_assign_lights_to_clusters(&[light], 1, &vp);
        assert!(
            head.lists.iter().all(|list| list.is_empty()),
            "head records are not clustered"
        );
        // A small sphere in the middle of the view touches a handful of
        // clusters, and the pixel at the light's own position finds it.
        let assignment = cpu_assign_lights_to_clusters(&[light], 0, &vp);
        let lit = assignment
            .lists
            .iter()
            .filter(|list| !list.is_empty())
            .count();
        assert!(lit > 0, "the light must light something");
        assert!(lit < 64, "a 2 m light spans few clusters");
        let cluster = cluster_index_for(glam::Vec3::ZERO, &vp);
        assert!(assignment.lists[cluster as usize].contains(&0));
    }

    #[test]
    fn spot_bound_sphere_is_tighter_than_the_point_bound() {
        let mut spot = GpuLight::zeroed();
        spot.kind = [LIGHT_KIND_SPOT, 0.0, 0.0, 0.0];
        spot.direction = [0.0, 0.0, -1.0, 0.0];
        spot.position = [0.0, 0.0, 9.0, 1.0];
        // 30° outer cone: cos ≈ 0.866.
        spot.params = [8.0, 0.9, 0.866, -1.0];
        let (center, radius) = light_bound_sphere(&spot);
        assert!(radius < 8.0, "narrow cone must shrink the bound");
        assert!(radius > 4.0, "bound still covers the axis extent");
        // The bound contains the cone rim: the halfway-out rim point
        // stays inside the sphere.
        let rim = glam::Vec3::new(8.0 * 0.5 * 0.5, 0.0, 9.0 - 8.0 * 0.5);
        assert!((rim - center).length() <= radius);
    }

    #[test]
    fn cluster_layouts_are_derive_pinned() {
        assert_eq!(std::mem::size_of::<ClusterSpan>(), 8);
        assert_eq!(std::mem::size_of::<ClusterParams>(), 16);
        assert_eq!(
            ClusterParams::FIELD_NAMES,
            &["light_count", "head_count", "clustered", "pad"]
        );
        assert!(ClusterSpan::WGSL_SOURCE.contains("struct ClusterSpan"));
        assert!(ClusterParams::WGSL_SOURCE.contains("struct ClusterParams"));
        // The uniform block is a 16-byte multiple (WGSL uniform rule).
        assert_eq!(std::mem::size_of::<ClusterParams>() % 16, 0);
    }
}
