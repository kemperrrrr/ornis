//! Host `gpu_struct!` span versus naga's layout of lowered uniform structs.
//!
//! Short structs (`f32`, two `f32`s, `vec2`) are padded to the host stride
//! of 16. The checks run naga's validator and layouter, not only the parser.

use ornis_gpu_abi::GpuSafeLayout;
use ornis_gpu_abi::GpuVec3;
use ornis_gpu_abi::gpu_struct;
use ornis_gpu_abi::symbols;

#[path = "support/uniform.rs"]
mod support;
use support::*;

gpu_struct! {
    struct OneF32 {
        value: f32,
    }
}

gpu_struct! {
    struct TwoF32 {
        a: f32,
        b: f32,
    }
}

gpu_struct! {
    struct Vec2Struct {
        value: ornis_gpu_abi::GpuVec2,
    }
}

gpu_struct! {
    struct Vec3Struct {
        value: GpuVec3,
    }
}

gpu_struct! {
    struct Wide {
        a: ornis_gpu_abi::GpuVec4,
        b: ornis_gpu_abi::GpuVec4,
    }
}

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

#[test]
fn small_struct_arrays_and_whole_reads_match_the_host() {
    check_struct::<OneF32>(Read::Indexed);
    check_struct::<TwoF32>(Read::Indexed);
    check_struct::<Vec2Struct>(Read::Indexed);
    check_struct::<Vec3Struct>(Read::Indexed);
    check_struct::<FrameTint>(Read::Indexed);
    check_struct::<Wide>(Read::Indexed);
    check_struct::<TwoVec>(Read::Indexed);
    check_struct::<OneF32>(Read::Whole);
    check_struct::<TwoF32>(Read::Whole);
    check_struct::<Vec2Struct>(Read::Whole);
    check_struct::<Vec3Struct>(Read::Whole);
}

#[test]
fn vec2_and_vec3_uniform_arrays_keep_stride_16() {
    check_vec(
        symbols::UNIFORM_READ_VEC2,
        "f32 f32",
        "f32.const 0 f32.const 1",
        ornis_gpu_abi::GpuVec2::UNIFORM_STRIDE,
        ornis_gpu_abi::GpuVec2::UNIFORM_STRIDE,
    );
    check_vec(
        symbols::UNIFORM_READ_VEC3,
        "f32 f32 f32",
        "f32.const 1",
        GpuVec3::SIZE,
        GpuVec3::UNIFORM_STRIDE,
    );
}

#[test]
fn storage_write_of_a_short_struct_composes_the_tail() {
    let got = wgsl(
        &section(&OneF32::FIELD_TAGS),
        &import(symbols::STORAGE_WRITE_STRUCT, "(param i32 i32 i32 i32 f32)"),
        &format!(
            "(param f32 f32) (result f32 f32 f32 f32)
             i32.const 0 i32.const 1 i32.const 0 i32.const 0 f32.const 1 call ${write}
             {ret}",
            write = symbols::STORAGE_WRITE_STRUCT,
            ret = rgba(),
        ),
    );
    let (offsets, span) = struct_layout(&got, 1);
    assert!(offsets[0] == OneF32::FIELD_OFFSETS[0]);
    assert!(span == OneF32::SIZE);
}

#[derive(Clone, Copy)]
enum Read {
    Whole,
    Indexed,
}

fn check_struct<T>(read: Read)
where
    T: ornis_gpu_abi::GpuSafeLayout + HostStruct,
{
    let lanes = field_lanes(T::FIELD_TAGS);
    let name = match read {
        Read::Whole => symbols::UNIFORM_READ_STRUCT,
        Read::Indexed => symbols::UNIFORM_INDEX_STRUCT,
    };
    let params = match read {
        Read::Whole => "(param i32 i32 i32)",
        Read::Indexed => "(param i32 i32 i32 i32 i32)",
    };
    let got = wgsl(
        &section(T::FIELD_TAGS),
        &import(name, &format!("{params} (result {})", result_words(lanes))),
        &body(read, name, lanes),
    );
    let module = parsed(&got);
    let want = wasm_offsets(T::FIELD_TAGS, T::FIELD_OFFSETS);
    let handle = find_struct(&module, &want);
    let (size, stride) = laid(&module, handle);
    assert!(size == T::SIZE, "{got}");
    assert!(stride == T::UNIFORM_STRIDE, "{got}");
    if matches!(read, Read::Indexed) {
        let (base, array_stride, len) = uniform_array(&module);
        assert!(base == handle, "{got}");
        assert!(array_stride == T::UNIFORM_STRIDE, "{got}");
        assert!(len == 4, "{got}");
        assert!(got.contains('['), "{got}");
    }
}

trait HostStruct {
    const FIELD_TAGS: &'static [u8];
    const FIELD_OFFSETS: &'static [u32];
}

macro_rules! host_struct {
    ($name:ty) => {
        impl HostStruct for $name {
            const FIELD_TAGS: &'static [u8] = &<$name>::FIELD_TAGS;
            const FIELD_OFFSETS: &'static [u32] = &<$name>::FIELD_OFFSETS;
        }
    };
}

host_struct!(OneF32);
host_struct!(TwoF32);
host_struct!(Vec2Struct);
host_struct!(Vec3Struct);
host_struct!(FrameTint);
host_struct!(Wide);
host_struct!(TwoVec);

fn check_vec(name: &str, results: &str, tail: &str, size: u32, stride: u32) {
    let got = wgsl(
        "",
        &import(name, &format!("(param i32 i32 i32 i32) (result {results})")),
        &format!(
            "(param f32 f32 i32) (result f32 f32 f32 f32)
             i32.const 0 i32.const 1 i32.const 4 local.get 2 call ${name}
             {tail}"
        ),
    );
    let module = parsed(&got);
    let (base, array_stride, len) = uniform_array(&module);
    let (laid_size, laid_stride) = laid(&module, base);
    assert!(laid_size == size, "{got}");
    assert!(laid_stride == stride, "{got}");
    assert!(array_stride == stride, "{got}");
    assert!(len == 4, "{got}");
    assert!(got.contains('['), "{got}");
}

fn section(tags: &[u8]) -> String {
    let fields = wasm_tags(tags);
    let bytes = ornis_gpu_abi::encode_types_section(&[&fields]).expect("types section");
    let mut body = String::new();
    for byte in &bytes {
        body.push_str(&format!("\\{byte:02x}"));
    }
    format!(
        "(@custom \"{name}\" \"{body}\")",
        name = symbols::TYPES_SECTION,
    )
}

fn wasm_tags(tags: &[u8]) -> Vec<u8> {
    tags.iter().copied().filter(|tag| *tag != 0).collect()
}

fn wasm_offsets(tags: &[u8], offsets: &[u32]) -> Vec<u32> {
    tags.iter()
        .zip(offsets)
        .filter(|(tag, _)| **tag != 0)
        .map(|(_, offset)| *offset)
        .collect()
}

fn field_lanes(tags: &[u8]) -> u32 {
    let mut lanes = 0u32;
    for tag in tags {
        lanes += match *tag {
            ornis_gpu_abi::TYPE_TAG_VEC2 => 2,
            ornis_gpu_abi::TYPE_TAG_VEC3 => 3,
            ornis_gpu_abi::TYPE_TAG_VEC4 => 4,
            0 => 0,
            _ => 1,
        };
    }
    lanes
}

fn result_words(lanes: u32) -> String {
    let mut words = String::new();
    let mut index = 0u32;
    while index < lanes {
        words.push_str("f32 ");
        index += 1;
    }
    words
}

fn body(read: Read, name: &str, lanes: u32) -> String {
    let call = match read {
        Read::Whole => format!("i32.const 0 i32.const 0 i32.const 0 call ${name}"),
        Read::Indexed => {
            format!("i32.const 0 i32.const 0 i32.const 4 local.get 2 i32.const 0 call ${name}")
        }
    };
    let params = match read {
        Read::Whole => "f32 f32",
        Read::Indexed => "f32 f32 i32",
    };
    format!(
        "(param {params}) (result f32 f32 f32 f32) {call} {}",
        tail(lanes)
    )
}

fn tail(lanes: u32) -> String {
    let mut text = String::new();
    if lanes > 4 {
        let mut extra = 4u32;
        while extra < lanes {
            text.push_str("drop ");
            extra += 1;
        }
    } else {
        let mut have = lanes;
        while have < 4 {
            text.push_str("f32.const 0 ");
            have += 1;
        }
    }
    text
}

fn find_struct(module: &naga::Module, offsets: &[u32]) -> naga::Handle<naga::Type> {
    for (handle, ty) in module.types.iter() {
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            continue;
        };
        if members.len() < offsets.len() || !data_struct(module, members) {
            continue;
        }
        let mut matched = true;
        let mut index = 0usize;
        while index < offsets.len() {
            if members[index].offset != offsets[index] {
                matched = false;
            }
            index += 1;
        }
        if matched {
            return handle;
        }
    }
    panic!("struct offsets {offsets:?} missing");
}

fn laid(module: &naga::Module, handle: naga::Handle<naga::Type>) -> (u32, u32) {
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).expect("layout");
    let layout = &layouter[handle];
    (layout.size, layout.to_stride())
}

fn uniform_array(module: &naga::Module) -> (naga::Handle<naga::Type>, u32, u32) {
    for (_, global) in module.global_variables.iter() {
        if global.space != naga::AddressSpace::Uniform {
            continue;
        }
        let naga::TypeInner::Struct { members, .. } = &module.types[global.ty].inner else {
            continue;
        };
        let Some(member) = members.first() else {
            continue;
        };
        let naga::TypeInner::Array { base, stride, size } = &module.types[member.ty].inner else {
            continue;
        };
        let naga::ArraySize::Constant(len) = size else {
            continue;
        };
        return (*base, *stride, len.get());
    }
    panic!("no uniform array");
}
