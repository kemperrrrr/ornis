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
    assert_eq!(sampled.x.to_bits(), 0.25_f32.to_bits());
    assert_eq!(sampled.y.to_bits(), 0.5_f32.to_bits());
    assert_eq!(sampled.z.to_bits(), 0.75_f32.to_bits());
    assert_eq!(sampled.w.to_bits(), 1.0_f32.to_bits());
}

#[test]
#[should_panic(expected = "texture coordinate out of bounds")]
fn texture_load_shadow_rejects_negative_coordinates() {
    let mut memory = HostMemory::new();
    memory.bind_texture::<0, 0>(
        1,
        1,
        &[GpuVec4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }],
    );
    with_host(&mut memory, || Texture2d::<0, 0>::bind().load(-1, 0));
}
