// Screen-space AA. Motion is previous UV minus current UV; z/w are
// current/previous linear depth. A negative w explicitly rejects history.
struct Params { mode: u32, valid: u32, pad: vec2<u32> }
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var history: texture_2d<f32>;
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var auxiliary: texture_2d<f32>;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var<uniform> params: Params;
struct VertexOut { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VertexOut;
    out.position = vec4<f32>(p * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = p;
    return out;
}
fn color(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(source, linear_sampler, uv, 0.0).rgb;
}
fn luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.299, 0.587, 0.114)); }
fn edge_at(p: vec2<i32>) -> vec2<f32> {
    let extent = vec2<i32>(textureDimensions(source));
    return textureLoad(source, clamp(p, vec2<i32>(0), extent - 1), 0).rg;
}
// SMAA 1x: local contrast adaptation suppresses edges beside stronger edges.
@fragment fn edges(in: VertexOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(source));
    let c = luma(color(in.uv));
    let l = luma(color(in.uv - vec2<f32>(texel.x, 0.0)));
    let t = luma(color(in.uv - vec2<f32>(0.0, texel.y)));
    let d = abs(vec2<f32>(c - l, c - t));
    let r = abs(c - luma(color(in.uv + vec2<f32>(texel.x, 0.0))));
    let b = abs(c - luma(color(in.uv + vec2<f32>(0.0, texel.y))));
    let ll = abs(l - luma(color(in.uv - vec2<f32>(2.0 * texel.x, 0.0))));
    let tt = abs(t - luma(color(in.uv - vec2<f32>(0.0, 2.0 * texel.y))));
    let strongest = max(max(max(d.x, d.y), max(r, b)), max(ll, tt));
    let e = step(vec2<f32>(0.05), d) * step(vec2<f32>(strongest), 2.0 * d);
    return vec4<f32>(e, 0.0, 0.0);
}
// Bounded orthogonal searches and analytic crossing-area approximation.
// No external area/search lookup textures; diagonal/corner refinement is omitted.
fn area(p: vec2<i32>, along: vec2<i32>, component: u32) -> f32 {
    let extent = vec2<i32>(textureDimensions(source));
    var distances = vec2<f32>(0.0);
    for (var side = 0u; side < 2u; side++) {
        let direction = select(-1, 1, side == 1u);
        for (var i = 1; i <= 16; i++) {
            let q = p + along * i * direction;
            if (any(q < vec2<i32>(0)) || any(q >= extent)) { break; }
            if (edge_at(q)[component] < 0.5) { break; }
            distances[side] += 1.0;
        }
    }
    return 0.5 * (min(distances.x, distances.y) + 1.0) / (distances.x + distances.y + 2.0);
}
@fragment fn weights(in: VertexOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.position.xy);
    let e = edge_at(p);
    var w = vec2<f32>(0.0);
    if (e.x > 0.0) { w.x = area(p, vec2<i32>(0, 1), 0u); }
    if (e.y > 0.0) { w.y = area(p, vec2<i32>(1, 0), 1u); }
    return vec4<f32>(w, 0.0, 0.0);
}
fn fxaa(uv: vec2<f32>) -> vec3<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(source));
    let c = color(uv);
    let nw = luma(color(uv + t * vec2<f32>(-1.0, -1.0)));
    let ne = luma(color(uv + t * vec2<f32>(1.0, -1.0)));
    let sw = luma(color(uv + t * vec2<f32>(-1.0, 1.0)));
    let se = luma(color(uv + t * vec2<f32>(1.0, 1.0)));
    let lo = min(luma(c), min(min(nw, ne), min(sw, se)));
    let hi = max(luma(c), max(max(nw, ne), max(sw, se)));
    if (hi - lo < max(0.0312, hi * 0.125)) { return c; }
    var direction = vec2<f32>(-(nw + ne - sw - se), nw + sw - ne - se);
    let reduce = max((nw + ne + sw + se) * 0.03125, 0.0078125);
    direction = clamp(direction / (min(abs(direction.x), abs(direction.y)) + reduce), vec2<f32>(-8.0), vec2<f32>(8.0)) * t;
    let a = 0.5 * (color(uv + direction * (-1.0 / 6.0)) + color(uv + direction * (1.0 / 6.0)));
    let b = a * 0.5 + 0.25 * (color(uv - direction * 0.5) + color(uv + direction * 0.5));
    return select(b, a, luma(b) < lo || luma(b) > hi);
}
struct ResolveOut { @location(0) color: vec4<f32>, @location(1) history: vec4<f32> }
@fragment fn resolve(in: VertexOut) -> ResolveOut {
    let t = 1.0 / vec2<f32>(textureDimensions(source));
    let p = vec2<i32>(in.position.xy);
    let velocity = textureLoad(motion, p, 0);
    var c = color(in.uv);
    if (params.mode == 2u) { c = fxaa(in.uv); }
    if (params.mode == 3u) {
        let w = textureLoad(auxiliary, p, 0).rg;
        let right = textureSampleLevel(auxiliary, linear_sampler, in.uv + vec2<f32>(t.x, 0.0), 0.0).r;
        let bottom = textureSampleLevel(auxiliary, linear_sampler, in.uv + vec2<f32>(0.0, t.y), 0.0).g;
        let total = w.x + w.y + right + bottom;
        if (total > 0.0) {
            let blended = color(in.uv - vec2<f32>(t.x, 0.0)) * w.x
                + color(in.uv - vec2<f32>(0.0, t.y)) * w.y
                + color(in.uv + vec2<f32>(t.x, 0.0)) * right
                + color(in.uv + vec2<f32>(0.0, t.y)) * bottom;
            c = mix(c, blended / total, min(total, 1.0));
        }
    }
    if (params.mode == 4u && params.valid != 0u) {
        let uv = in.uv + velocity.xy;
        let inside = all(uv >= t * 0.5) && all(uv <= vec2<f32>(1.0) - t * 0.5);
        let old = textureSampleLevel(history, linear_sampler, uv, 0.0);
        // Depth is point sampled: interpolation across silhouettes hides disocclusion.
        let hp = clamp(vec2<i32>(uv * vec2<f32>(textureDimensions(history))), vec2<i32>(0), vec2<i32>(textureDimensions(history)) - 1);
        let old_depth = textureLoad(history, hp, 0).a;
        if (inside && velocity.w > 0.0 && abs(old_depth - velocity.w) <= max(0.01, velocity.w * 0.01)) {
            var lo = c;
            var hi = c;
            for (var y = -1; y <= 1; y++) {
                for (var x = -1; x <= 1; x++) {
                    let s = color(in.uv + vec2<f32>(f32(x), f32(y)) * t);
                    lo = min(lo, s); hi = max(hi, s);
                }
            }
            let speed = length(velocity.xy / t);
            let feedback = mix(0.9, 0.5, clamp(speed / 8.0, 0.0, 1.0));
            c = mix(c, clamp(old.rgb, lo, hi), feedback);
        }
    }
    var out: ResolveOut;
    out.color = vec4<f32>(c, 1.0);
    out.history = vec4<f32>(c, velocity.z);
    return out;
}
