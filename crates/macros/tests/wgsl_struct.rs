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
