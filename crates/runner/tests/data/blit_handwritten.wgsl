// Frozen reference: the hand-written stats-window blit that `ornis-gpu-blit`
// replaces. Kept verbatim so `tests/blit_parity.rs` can compare the translated
// module with it pixel for pixel. Not a build input.
@group(0) @binding(0) var panel: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(panel));
    return textureLoad(panel, min(vec2<i32>(p.xy), size - 1), 0);
}
