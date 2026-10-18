//! Compute-shader source of the cluster assignment pass (#130),
//! assembled from the shared derives and the same Rust constants the
//! CPU reference and the fragment consts block use, so no layout can
//! drift between the three.
use super::{
    CLUSTER_COUNT, CLUSTER_DEPTH_NEAR, CLUSTER_FAR_LIMIT, CLUSTER_INDEX_LIST_LEN,
    CLUSTER_SLICE_COUNT, CLUSTER_TILE_COUNT_X, CLUSTER_TILE_COUNT_Y, CLUSTER_WORKGROUP_SIZE,
    ClusterParams, ClusterSpan, MAX_LIGHTS_PER_CLUSTER, SPOT_BOUND_COS_FLOOR,
    cluster_log_ratio, cluster_slice_scale,
};
use crate::renderer::{CameraUniform, GpuLight, LIGHT_KIND_SPOT};

/// Cluster-pass binding numbers (private to this pass).
pub(super) const BINDING_CAMERA: u32 = 0;
pub(super) const BINDING_PARAMS: u32 = 1;
pub(super) const BINDING_LIGHTS: u32 = 2;
pub(super) const BINDING_GRID: u32 = 3;
pub(super) const BINDING_INDICES: u32 = 4;
pub(super) const BINDING_COUNTERS: u32 = 5;

/// Compute shader source for the cluster assignment: one thread per
/// cluster builds its six frustum planes from the camera `view_proj`,
/// tests every packed light's bounding sphere against them and claims a
/// span of the shared index list through one atomic counter.
pub(crate) fn cluster_compute_wgsl() -> String {
    let f = crate::shaders::f32_lit;
    [
        crate::shaders::wgsl_decl(CameraUniform::WGSL_SOURCE),
        crate::shaders::wgsl_decl(ClusterParams::WGSL_SOURCE),
        crate::shaders::wgsl_decl(GpuLight::WGSL_SOURCE),
        crate::shaders::wgsl_decl(ClusterSpan::WGSL_SOURCE),
        format!(
            "const CLUSTERS_X: u32 = {CLUSTER_TILE_COUNT_X}u;\n\
             const CLUSTERS_Y: u32 = {CLUSTER_TILE_COUNT_Y}u;\n\
             const CLUSTERS_Z: u32 = {CLUSTER_SLICE_COUNT}u;\n\
             const CLUSTER_COUNT: u32 = {CLUSTER_COUNT}u;\n\
             const MAX_LIGHTS_PER_CLUSTER: u32 = {MAX_LIGHTS_PER_CLUSTER}u;\n\
             const CLUSTER_INDEX_LIST_LEN: u32 = {CLUSTER_INDEX_LIST_LEN}u;\n\
             const CLUSTER_WORKGROUP_SIZE: u32 = {CLUSTER_WORKGROUP_SIZE}u;\n\
             const CLUSTER_DEPTH_NEAR: f32 = {near};\n\
             const CLUSTER_FAR_LIMIT: f32 = {far};\n\
             const CLUSTER_LOG_RATIO: f32 = {ratio};\n\
             const CLUSTER_SLICE_SCALE: f32 = {scale};\n\
             const LIGHT_KIND_SPOT: f32 = {spot_kind};\n\
             const SPOT_BOUND_COS_FLOOR: f32 = {floor};\n",
            near = f(CLUSTER_DEPTH_NEAR),
            far = f(CLUSTER_FAR_LIMIT),
            ratio = f(cluster_log_ratio()),
            scale = f(cluster_slice_scale()),
            spot_kind = f(LIGHT_KIND_SPOT),
            floor = f(SPOT_BOUND_COS_FLOOR),
        ),
        "@group(0) @binding(0) var<uniform> camera: Camera;\n\
         @group(0) @binding(1) var<uniform> params: ClusterParams;\n\
         @group(0) @binding(2) var<storage, read> lights: array<Light>;\n\
         @group(0) @binding(3) var<storage, read_write> grid: array<ClusterSpan>;\n\
         @group(0) @binding(4) var<storage, read_write> indices: array<u32>;\n\
         @group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>, 2>;\n"
            .to_string(),
        // Six cluster half-space planes packed as (n.xyz, d).
        "struct ClusterPlanes {\n    p: array<vec4<f32>, 6>,\n};\n".to_string(),
        r#"
fn slice_z_max(i: u32) -> f32 {
    if (i + 1u >= CLUSTERS_Z) {
        return CLUSTER_FAR_LIMIT;
    }
    return CLUSTER_DEPTH_NEAR * exp((f32(i) / f32(CLUSTERS_Z - 1u)) * CLUSTER_LOG_RATIO);
}

fn cluster_planes(tx: u32, ty: u32, tz: u32) -> ClusterPlanes {
    let x0 = -1.0 + 2.0 * f32(tx) / f32(CLUSTERS_X);
    let x1 = -1.0 + 2.0 * f32(tx + 1u) / f32(CLUSTERS_X);
    let y0 = -1.0 + 2.0 * f32(ty) / f32(CLUSTERS_Y);
    let y1 = -1.0 + 2.0 * f32(ty + 1u) / f32(CLUSTERS_Y);
    let m = camera.view_proj;
    let row_x = vec4<f32>(m[0].x, m[1].x, m[2].x, m[3].x);
    let row_y = vec4<f32>(m[0].y, m[1].y, m[2].y, m[3].y);
    let row_w = vec4<f32>(m[0].w, m[1].w, m[2].w, m[3].w);
    var z_lo = 0.0;
    if (tz >= 1u) {
        z_lo = slice_z_max(tz - 1u);
    }
    let z_hi = slice_z_max(tz);
    var planes: ClusterPlanes;
    planes.p[0] = row_x - x0 * row_w;
    planes.p[1] = x1 * row_w - row_x;
    planes.p[2] = row_y - y0 * row_w;
    planes.p[3] = y1 * row_w - row_y;
    planes.p[4] = vec4<f32>(row_w.xyz, row_w.w - z_lo);
    planes.p[5] = vec4<f32>(-(row_w.xyz), z_hi - row_w.w);
    return planes;
}

fn sphere_in_cluster(center: vec3<f32>, radius: f32, planes: ClusterPlanes) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = planes.p[i];
        let n = plane.xyz;
        if (abs(dot(n, center) + plane.w) > radius * length(n)) {
            return false;
        }
    }
    return true;
}

fn light_bound_sphere(light: Light) -> vec4<f32> {
    if (light.kind.x < LIGHT_KIND_SPOT) {
        return vec4<f32>(light.position.xyz, light.params.x);
    }
    let cos_outer = light.params.z;
    if (cos_outer < SPOT_BOUND_COS_FLOOR) {
        return vec4<f32>(light.position.xyz, light.params.x);
    }
    let tan_sq = (1.0 - cos_outer * cos_outer) / (cos_outer * cos_outer);
    let radius = light.params.x * (1.0 + tan_sq) * 0.5;
    let center = light.position.xyz + normalize(light.direction.xyz) * radius;
    return vec4<f32>(center, radius);
}

@compute @workgroup_size(CLUSTER_WORKGROUP_SIZE)
fn assign_lights_to_clusters(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cluster = gid.x;
    if (cluster >= CLUSTER_COUNT) {
        return;
    }
    let tx = cluster % CLUSTERS_X;
    let ty = (cluster / CLUSTERS_X) % CLUSTERS_Y;
    let tz = cluster / (CLUSTERS_X * CLUSTERS_Y);
    let planes = cluster_planes(tx, ty, tz);
    var count = 0u;
    for (var li = params.head_count; li < params.light_count; li = li + 1u) {
        let bound = light_bound_sphere(lights[li]);
        if (sphere_in_cluster(bound.xyz, bound.w, planes)) {
            count = count + 1u;
        }
    }
    count = min(count, MAX_LIGHTS_PER_CLUSTER);
    let base = atomicAdd(&counters[0], count);
    var span_count = count;
    if (base + count > CLUSTER_INDEX_LIST_LEN) {
        if (base >= CLUSTER_INDEX_LIST_LEN) {
            span_count = 0u;
        } else {
            span_count = CLUSTER_INDEX_LIST_LEN - base;
        }
        atomicAdd(&counters[1], count - span_count);
    }
    grid[cluster] = ClusterSpan(base, span_count);
    var j = 0u;
    for (var li = params.head_count; li < params.light_count; li = li + 1u) {
        if (j >= span_count) {
            break;
        }
        let bound = light_bound_sphere(lights[li]);
        if (sphere_in_cluster(bound.xyz, bound.w, planes)) {
            indices[base + j] = li;
            j = j + 1u;
        }
    }
}
"#
        .to_string(),
    ]
    .concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validate the assembled compute shader with naga — layout drift
    /// between the derives, the consts and the pass body fails here.
    #[test]
    fn cluster_compute_shader_validates_with_naga() {
        let source = cluster_compute_wgsl();
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("cluster compute must parse: {e}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("cluster compute must validate");
        assert!(source.contains("fn assign_lights_to_clusters"));
        assert!(source.contains("array<atomic<u32>, 2>"));
    }
}
