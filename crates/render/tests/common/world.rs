//! Master's `view.xyz / view.w` reconstruction for AO tests.
//!
//! Shared so the parity scene and the CPU GTAO march stay bit-identical
//! and rustqual does not count them as a duplicate pair.

/// Reconstruct the world position of a depth sample, dividing by raw `w`.
pub fn reconstruct(inv: glam::Mat4, uv: glam::Vec2, depth: f32) -> glam::Vec3 {
    let ndc = glam::Vec4::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = inv * ndc;
    world.truncate() / world.w
}
