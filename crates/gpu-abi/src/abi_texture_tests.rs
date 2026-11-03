//! Explicit-LOD texture ABI tests.

use crate::GpuVec2;
use crate::GpuVec4;
use crate::HostMemory;
use crate::Sampler;
use crate::Texture2d;
use crate::with_host;

#[test]
fn sample_lod_uses_the_host_shadow_base_level() {
    let mut memory = HostMemory::new();
    let texel = GpuVec4 {
        x: 0.25,
        y: 0.5,
        z: 0.75,
        w: 1.0,
    };
    memory.bind_texture::<0, 0>(1, 1, &[texel]);
    memory.bind_sampler::<0, 1>();
    let uv = GpuVec2 { x: 0.5, y: 0.5 };
    let sampled = with_host(&mut memory, || {
        Texture2d::<0, 0>::bind().sample_lod(&Sampler::<0, 1>::bind(), uv, 2.0)
    });
    assert_eq!(sampled.x.to_bits(), texel.x.to_bits());
    assert_eq!(sampled.y.to_bits(), texel.y.to_bits());
    assert_eq!(sampled.z.to_bits(), texel.z.to_bits());
    assert_eq!(sampled.w.to_bits(), texel.w.to_bits());
}
