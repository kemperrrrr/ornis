//! GTAO slice march for the raw AO pass (XeGTAO / Bevy 0.14).
//!
//! Split out of [`super::ao_generated`] so that module stays inside the
//! file-length gate. SSAO never includes these strings: the horizon helper
//! would be dead code in that kernel.

/// XeGTAO arc (`gtao_arc`, the same closed form as
/// [`crate::ao::gtao_sample`]) and one horizon sample.
///
/// A miss, or a hit at/past `radius`, returns `-1` rather than the
/// hemisphere floor. That floor is the horizon's starting cosine
/// (`min_cos` in the march). Returning it from a miss would make the
/// start dead: deleting the clamp would not change a pixel, because every
/// empty tap would put the floor back. `max` with the start still ignores
/// `-1`, so a correct march is unchanged.
pub(super) const AO_GTAO_HELPER: &str = r#"
fn gtao_arc(horizon: f32, normal_angle: f32) -> f32 {
    let h = horizon;
    let n = normal_angle;
    return 0.25 * (-cos(2.0 * h - n) + cos(n) + 2.0 * h * sin(n));
}

fn gtao_horizon_cos(uv: vec2<f32>, dims: vec2<f32>, center: vec3<f32>, view_vec: vec3<f32>, min_cos: f32, radius: f32) -> f32 {
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
        return -1.0;
    }
    let dims_u = vec2<u32>(dims);
    let texel = min(vec2<u32>(uv * dims), dims_u - vec2<u32>(1u));
    let sdepth = textureLoad(depth_tex, texel, 0);
    if (sdepth >= CLEAR_DEPTH) {
        return -1.0;
    }
    let sample_pos = reconstruct_world_pos(uv, sdepth, camera);
    let delta = sample_pos - center;
    let r = length(delta);
    if (r <= EPS || r >= radius) {
        return -1.0;
    }
    let falloff = clamp(1.0 - r / max(radius, EPS), 0.0, 1.0);
    let cos_h = dot(delta / r, view_vec);
    return mix(min_cos, cos_h, falloff);
}
"#;

/// Two slices by four steps, each step sampling both directions. The normal
/// is projected onto the slice and both horizons feed the arc above.
/// Matches Bevy 0.14 `gtao.wgsl` / Intel XeGTAO (no bent normals, no depth
/// mips — the g-buffer is a single hardware depth).
pub(super) const AO_GTAO_BODY: &str = r#"
    let to_eye = camera.camera_pos.xyz - center;
    let view_vec = to_eye / max(length(to_eye), EPS);
    let texel_x = vec2<f32>(1.0 / dims.x, 0.0);
    let texel_y = vec2<f32>(0.0, 1.0 / dims.y);
    let screen_x = normalize(reconstruct_world_pos(input.uv + texel_x, depth, camera) - center + vec3<f32>(EPS, 0.0, 0.0));
    let screen_y = normalize(reconstruct_world_pos(input.uv + texel_y, depth, camera) - center + vec3<f32>(0.0, EPS, 0.0));
    let edge = center + screen_x * ao_params.radius;
    let edge_clip = camera.view_proj * vec4<f32>(edge, 1.0);
    let edge_w = max(edge_clip.w, EPS);
    let edge_uv = vec2<f32>(edge_clip.x / edge_w * 0.5 + 0.5, 0.5 - edge_clip.y / edge_w * 0.5);
    let uv_radius = max(length(edge_uv - input.uv), 1.0 / dims.x);
    let slice_base = ao_interleaved_gradient(coord) * 3.14159265359;
    var visibility = 0.0;
    for (var s = 0u; s < 2u; s = s + 1u) {
        let phi = slice_base + f32(s) * 1.57079632679;
        let cos_phi = cos(phi);
        let sin_phi = sin(phi);
        let slice_dir = normalize(screen_x * cos_phi + screen_y * sin_phi);
        let axis = normalize(cross(slice_dir, view_vec) + vec3<f32>(EPS, 0.0, 0.0));
        let projected = n - axis * dot(n, axis);
        let proj_len = max(length(projected), EPS);
        let cos_norm = clamp(dot(projected, view_vec) / proj_len, 0.0, 1.0);
        let normal_angle = sign(dot(slice_dir, projected)) * acos(cos_norm);
        let min_cos_pos = cos(normal_angle + 1.57079632679);
        let min_cos_neg = cos(normal_angle - 1.57079632679);
        var horizon_cos_pos = min_cos_pos;
        var horizon_cos_neg = min_cos_neg;
        let slice_uv = vec2<f32>(cos_phi, sin_phi);
        for (var j = 1u; j <= 4u; j = j + 1u) {
            let off = slice_uv * (uv_radius * f32(j) / 4.0);
            horizon_cos_pos = max(horizon_cos_pos, gtao_horizon_cos(input.uv + off, dims, center, view_vec, min_cos_pos, ao_params.radius));
            horizon_cos_neg = max(horizon_cos_neg, gtao_horizon_cos(input.uv - off, dims, center, view_vec, min_cos_neg, ao_params.radius));
        }
        let h_pos = acos(clamp(horizon_cos_pos, -1.0, 1.0));
        let h_neg = -acos(clamp(horizon_cos_neg, -1.0, 1.0));
        visibility += proj_len * (gtao_arc(h_pos, normal_angle) + gtao_arc(h_neg, normal_angle));
    }
    let ao = clamp(mix(1.0, visibility / 2.0, ao_params.intensity), 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
"#;
