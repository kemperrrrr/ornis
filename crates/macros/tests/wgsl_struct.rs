//! Integration tests for `#[derive(WgslStruct)]`: the derive generates a
//! WGSL struct declaration from the Rust layout and asserts (at compile
//! time) that the `repr(C)` offsets match WGSL layout rules. A layout
//! mismatch would fail this test crate's compilation.

use ornis_macros::WgslStruct;

/// Mirrors a typical GPU body state: two vec3s with the explicit padding
/// that WGSL's 16-byte vec3 alignment requires.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, WgslStruct)]
struct BodyState {
    velocity: [f32; 3],
    pad_v: f32,
    angular: [f32; 3],
    pad_w: f32,
}

/// Mixed vectors, matrix rows and scalars, all aligned like WGSL storage.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, WgslStruct)]
struct Contact {
    nx: [f32; 4],
    ny: [f32; 4],
    uv: [f32; 2],
    pad0: [f32; 2],
    indices: [u32; 4],
    count: u32,
    pad1: u32,
}

#[test]
fn body_state_wgsl_source() {
    let src = BodyState::WGSL_SOURCE;
    assert!(src.contains("struct BodyState"));
    assert!(src.contains("velocity: vec3<f32>"));
    assert!(src.contains("pad_v: f32"));
    assert!(src.contains("angular: vec3<f32>"));
    assert!(src.contains("pad_w: f32"));
}

#[test]
fn contact_wgsl_source() {
    let src = Contact::WGSL_SOURCE;
    assert!(src.contains("struct Contact"));
    assert!(src.contains("nx: vec4<f32>"));
    assert!(src.contains("uv: vec2<f32>"));
    assert!(src.contains("indices: vec4<u32>"));
    assert!(src.contains("count: u32"));
}

/// Renamed mirror: `WGSL_NAME` carries the override, `WGSL_SOURCE` uses it.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, WgslStruct)]
#[wgsl(name = "Renamed")]
struct Original {
    a: [f32; 4],
}

#[test]
fn wgsl_name_matches_override() {
    assert_eq!(Original::WGSL_NAME, "Renamed");
    assert!(Original::WGSL_SOURCE.contains("struct Renamed"));
    assert_eq!(BodyState::WGSL_NAME, "BodyState");
}

/// The derived `naga_add_type` builds the same layout in naga IR: one
/// member per field, same names and offsets, validated by naga itself.
#[test]
fn naga_ir_matches_wgsl_source() {
    let mut module = naga::Module::default();
    let handle = BodyState::naga_add_type(&mut module);
    let ty = &module.types[handle];
    assert_eq!(ty.name.as_deref(), Some("BodyState"));
    let naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("BodyState must lower to a naga struct");
    };
    assert_eq!(*span, 32);
    let names: Vec<_> = members
        .iter()
        .map(|m| (m.name.as_deref().unwrap(), m.offset))
        .collect();
    assert_eq!(
        names,
        [
            ("velocity", 0),
            ("pad_v", 12),
            ("angular", 16),
            ("pad_w", 28)
        ]
    );
    // The IR module round-trips through naga validation.
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("declaration-only module must validate");
    let wgsl =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .expect("naga WGSL writer must print the struct");
    assert!(wgsl.contains("struct BodyState"));
    assert!(wgsl.contains("velocity: vec3<f32>"));
}

#[test]
fn layout_matches_wgsl_rules() {
    // These offsets are enforced at compile time by the derive; re-assert
    // them here to document the contract.
    assert_eq!(std::mem::size_of::<BodyState>(), 32);
    assert_eq!(std::mem::offset_of!(BodyState, velocity), 0);
    assert_eq!(std::mem::offset_of!(BodyState, angular), 16);
    assert_eq!(std::mem::size_of::<Contact>(), 80);
    assert_eq!(std::mem::offset_of!(Contact, uv), 32);
    assert_eq!(std::mem::offset_of!(Contact, count), 64);
}

/// Native Rust `bool` is rejected; this explicit wrapper has a stable u32
/// representation and is emitted as host-shareable WGSL `u32`.
#[repr(C)]
#[derive(Clone, Copy, Debug, WgslStruct)]
struct Flags {
    enabled: ornis_core::GpuBool,
}

#[test]
fn gpu_bool_has_explicit_wgsl_u32_representation() {
    assert_eq!(std::mem::size_of::<ornis_core::GpuBool>(), 4);
    assert_eq!(ornis_core::GpuBool::new(true).as_u32(), 1);
    assert!(!ornis_core::GpuBool::from_u32(0).get());
    assert!(ornis_core::GpuBool::from_u32(7).get());
    assert!(Flags::WGSL_SOURCE.contains("enabled: u32"));
}
