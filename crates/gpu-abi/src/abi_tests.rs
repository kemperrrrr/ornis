//! Safe-API tests. The host shadow's byte copies are the `unsafe` boundary:
//! a missed check is UB, and these cases hit the checks (and the success path
//! Miri executes).

use std::panic::catch_unwind;

use crate::GpuSafeLayout;
use crate::HostMemory;
use crate::Sampler;
use crate::Storage;
use crate::Texture2d;
use crate::Uniform;
use crate::gpu_struct;
use crate::layout::F32_ALIGN;
use crate::layout::FieldLayout;
use crate::layout::GpuVec2;
use crate::layout::GpuVec3;
use crate::layout::GpuVec4;
use crate::layout::VEC3_ALIGN;
use crate::layout::VEC3_SIZE;
use crate::layout::place_fields;
use crate::symbols;
use crate::with_host;

gpu_struct! {
    struct FrameTint {
        color: GpuVec3,
        intensity: f32,
    }
}

gpu_struct! {
    struct TwoVec {
        a: GpuVec3,
        pad: [u8; 4],
        b: GpuVec3,
    }
}

fn rgba(x: f32, y: f32, z: f32, w: f32) -> GpuVec4 {
    GpuVec4 { x, y, z, w }
}

fn image_texels(last: f32) -> [GpuVec4; 4] {
    [
        rgba(1.0, 0.0, 0.0, 1.0),
        rgba(0.0, 1.0, 0.0, 1.0),
        rgba(0.0, 0.0, 1.0, 1.0),
        rgba(last, last, last, 1.0),
    ]
}

fn bind_image<const G: u32, const B: u32, const SG: u32, const SB: u32>(
    mem: &mut HostMemory,
    last: f32,
) -> [GpuVec4; 4] {
    let texels = image_texels(last);
    mem.bind_texture::<G, B>(2, 2, &texels);
    mem.bind_sampler::<SG, SB>();
    texels
}

fn sample_uv<const TG: u32, const TB: u32, const SG: u32, const SB: u32>(
    mem: &mut HostMemory,
    uv: GpuVec2,
) -> GpuVec4 {
    with_host(mem, || {
        Texture2d::<TG, TB>::bind().sample(&Sampler::<SG, SB>::bind(), uv)
    })
}

fn place_pair(first: FieldLayout, second: FieldLayout) -> ([u32; 2], u32) {
    let fields = [first, second];
    let mut offsets = [0u32; 2];
    let span = place_fields(&fields, true, &mut offsets);
    (offsets, span)
}

#[test]
fn gpu_ref_is_an_offset() {
    let uniform = Uniform::<0, 0, f32>::bind();
    let reference = uniform.as_ref();
    assert!(size_of_val(&reference) == size_of::<u32>());
}

#[test]
fn frame_tint_matches_wgsl_vec3_then_f32() {
    assert!(FrameTint::FIELD_OFFSETS[0] == 0);
    assert!(FrameTint::FIELD_OFFSETS[1] == VEC3_SIZE);
    assert!(core::hint::black_box(FrameTint::SIZE) == VEC3_ALIGN);
    assert!(core::hint::black_box(FrameTint::ALIGN) == VEC3_ALIGN);
    let (offsets, span) = place_pair(
        FieldLayout {
            size: GpuVec3::SIZE,
            align: GpuVec3::ALIGN,
        },
        FieldLayout {
            size: f32::SIZE,
            align: f32::ALIGN,
        },
    );
    assert!(offsets[0] == FrameTint::FIELD_OFFSETS[0]);
    assert!(offsets[1] == FrameTint::FIELD_OFFSETS[1]);
    assert!(span == FrameTint::SIZE);
}

#[test]
fn two_vec3_fields_are_sixteen_apart() {
    let vec3 = FieldLayout {
        size: GpuVec3::SIZE,
        align: GpuVec3::ALIGN,
    };
    let (offsets, span) = place_pair(vec3, vec3);
    assert!(offsets[1] == VEC3_ALIGN);
    assert!(span == VEC3_ALIGN * 2);
    assert!(TwoVec::FIELD_OFFSETS[2] == VEC3_ALIGN);
    assert!(TwoVec::SIZE == span);
}

#[test]
fn uniform_roundtrip_and_field_read() {
    let mut mem = HostMemory::new();
    mem.bind_uniform::<0, 1, FrameTint>();
    let tint = FrameTint {
        color: GpuVec3 {
            x: 0.25,
            y: 0.5,
            z: 0.75,
        },
        intensity: 2.0,
    };
    mem.write_uniform::<0, 1, FrameTint>(&tint);
    let got = with_host(&mut mem, || {
        let uniform = Uniform::<0, 1, FrameTint>::bind();
        let whole = uniform.read();
        let intensity = uniform
            .as_ref()
            .field::<f32>(FrameTint::FIELD_OFFSETS[1])
            .read();
        let color = uniform.as_ref().field::<GpuVec3>(0).index_uniform(0).read();
        (whole, intensity, color)
    });
    assert!(got.0 == tint);
    assert!(got.1 == tint.intensity);
    assert!(got.2 == tint.color);
    assert!(mem.read_uniform::<0, 1, FrameTint>() == tint);
}

// The signature is the proof: `field` / `index_storage` must keep these consts.
fn indexed<const G: u32, const B: u32>(
    reference: crate::GpuRef<G, B, f32, crate::StorageSpace>,
) -> crate::GpuRef<G, B, f32, crate::StorageSpace> {
    reference.index_storage(1)
}

#[test]
fn field_and_index_keep_group_and_binding() {
    let mut mem = HostMemory::new();
    mem.bind_storage::<3, 5, [f32; 2]>();
    mem.write_storage::<3, 5, [f32; 2]>(&[1.0, 9.0]);
    let got = with_host(&mut mem, || {
        let storage = Storage::<3, 5, [f32; 2]>::bind();
        let base = storage.as_ref();
        indexed::<3, 5>(base.field::<f32>(0)).read()
    });
    assert!(got == 9.0);
    assert!(size_of::<crate::GpuRef<3, 5, f32, crate::StorageSpace>>() == size_of::<u32>());
}

#[test]
fn storage_index_write_does_not_touch_the_neighbor() {
    let mut mem = HostMemory::new();
    mem.bind_storage::<0, 2, [f32; 2]>();
    mem.write_storage::<0, 2, [f32; 2]>(&[1.0, 1.0]);
    with_host(&mut mem, || {
        let storage = Storage::<0, 2, [f32; 2]>::bind();
        storage.as_ref().field::<f32>(0).index_storage(1).write(3.0);
    });
    let got = mem.read_storage::<0, 2, [f32; 2]>();
    assert!(got[0] == 1.0);
    assert!(got[1] == 3.0);
}

#[test]
fn texture_sample_is_not_a_linear_load() {
    let mut mem = HostMemory::new();
    mem.bind_uniform::<0, 0, GpuVec4>();
    let marker = rgba(9.0, 9.0, 9.0, 9.0);
    mem.write_uniform::<0, 0, GpuVec4>(&marker);
    let texels = bind_image::<1, 0, 1, 1>(&mut mem, 1.0);
    let sampled = sample_uv::<1, 0, 1, 1>(&mut mem, GpuVec2 { x: 0.0, y: 0.0 });
    assert!(sampled == texels[0]);
    assert!(mem.read_uniform::<0, 0, GpuVec4>() == marker);
}

#[test]
fn nearest_corners_and_clamp() {
    let mut mem = HostMemory::new();
    let texels = bind_image::<0, 0, 0, 1>(&mut mem, 0.0);
    let corner = sample_uv::<0, 0, 0, 1>(&mut mem, GpuVec2 { x: 0.9, y: 0.1 });
    let clamped = sample_uv::<0, 0, 0, 1>(&mut mem, GpuVec2 { x: -1.0, y: 2.0 });
    assert!(corner == texels[1]);
    assert!(clamped == texels[2]);
}

#[test]
fn nested_with_host_restores_the_outer_shadow() {
    let mut outer = HostMemory::new();
    outer.bind_uniform::<0, 0, f32>();
    outer.write_uniform::<0, 0, f32>(&1.0);
    let mut inner = HostMemory::new();
    inner.bind_uniform::<0, 0, f32>();
    inner.write_uniform::<0, 0, f32>(&2.0);
    let seen = with_host(&mut outer, || {
        let inside = with_host(&mut inner, || Uniform::<0, 0, f32>::bind().read());
        let after = Uniform::<0, 0, f32>::bind().read();
        (inside, after)
    });
    assert!(seen.0 == 2.0);
    assert!(seen.1 == 1.0);
    assert!(inner.read_uniform::<0, 0, f32>() == 2.0);
}

#[test]
fn panic_inside_with_host_restores_memory() {
    let mut mem = HostMemory::new();
    mem.bind_storage::<0, 0, f32>();
    mem.write_storage::<0, 0, f32>(&1.0);
    let panicked = catch_unwind(core::panic::AssertUnwindSafe(|| {
        with_host(&mut mem, || {
            Storage::<0, 0, f32>::bind().write(4.0);
            panic!("shader");
        });
    }));
    assert!(panicked.is_err());
    assert!(mem.read_storage::<0, 0, f32>() == 4.0);
}

#[test]
fn threads_do_not_share_the_shadow() {
    let mut mem = HostMemory::new();
    mem.bind_uniform::<0, 0, f32>();
    mem.write_uniform::<0, 0, f32>(&1.0);
    let saw = with_host(&mut mem, || {
        std::thread::spawn(|| catch_unwind(|| Uniform::<0, 0, f32>::bind().read()))
            .join()
            .expect("thread")
    });
    assert!(saw.is_err());
    assert!(mem.read_uniform::<0, 0, f32>() == 1.0);
}

#[test]
fn import_names_stay_unpacked() {
    assert!(symbols::is_toy_import(symbols::RAW_READ));
    assert!(symbols::is_toy_import(symbols::RAW_READ_F32));
    assert!(symbols::is_toy_import(symbols::RAW_WRITE));
    assert!(symbols::is_toy_import(symbols::STORAGE_READ));
    assert!(symbols::is_toy_import(symbols::STORAGE_WRITE));
    assert!(symbols::is_toy_import(symbols::TEXTURE_SAMPLE_2D));
    assert!(!symbols::is_toy_import("i32.load"));
    assert!(symbols::MODULE == "ornis_gpu");
    assert!(symbols::VS_MAIN == "vs_main");
    assert!(symbols::FS_MAIN == "fs_main");
}

#[test]
#[should_panic]
fn read_outside_with_host_panics() {
    let mut mem = HostMemory::new();
    mem.bind_uniform::<0, 0, f32>();
    mem.write_uniform::<0, 0, f32>(&1.0);
    let _ = mem;
    Uniform::<0, 0, f32>::bind().read();
}

#[test]
#[should_panic]
fn out_of_bounds_panics_before_the_copy() {
    let mut mem = HostMemory::new();
    mem.bind_uniform::<0, 0, f32>();
    mem.write_uniform::<0, 0, f32>(&1.0);
    with_host(&mut mem, || {
        Uniform::<0, 0, f32>::bind()
            .as_ref()
            .field::<f32>(F32_ALIGN)
            .read();
    });
}

#[test]
#[should_panic]
fn misaligned_offset_panics_before_the_copy() {
    let mut mem = HostMemory::new();
    mem.bind_uniform::<0, 0, f32>();
    mem.write_uniform::<0, 0, f32>(&1.0);
    with_host(&mut mem, || {
        Uniform::<0, 0, f32>::bind().as_ref().field::<f32>(1).read();
    });
}

#[test]
#[should_panic]
fn index_overflow_panics() {
    let mut mem = HostMemory::new();
    mem.bind_storage::<0, 0, f32>();
    with_host(&mut mem, || {
        Storage::<0, 0, f32>::bind()
            .as_ref()
            .index_storage(u32::MAX)
            .read();
    });
}

#[test]
#[should_panic]
fn shader_write_rejects_a_uniform_binding() {
    let mut mem = HostMemory::new();
    mem.bind_uniform::<0, 0, f32>();
    mem.write_uniform::<0, 0, f32>(&1.0);
    with_host(&mut mem, || {
        Storage::<0, 0, f32>::bind().write(2.0);
    });
}

#[test]
#[should_panic]
fn duplicate_binding_panics() {
    let mut mem = HostMemory::new();
    mem.bind_uniform::<0, 0, f32>();
    mem.bind_storage::<0, 0, f32>();
}

#[test]
#[should_panic]
fn sample_without_sampler_panics() {
    let mut mem = HostMemory::new();
    mem.bind_texture::<0, 0>(1, 1, &[rgba(0.0, 0.0, 0.0, 1.0)]);
    with_host(&mut mem, || {
        Texture2d::<0, 0>::bind().sample(&Sampler::<0, 1>::bind(), GpuVec2 { x: 0.0, y: 0.0 });
    });
}

#[test]
#[should_panic]
fn empty_binding_panics() {
    let mut mem = HostMemory::new();
    mem.bind_uniform::<0, 0, [f32; 0]>();
}
