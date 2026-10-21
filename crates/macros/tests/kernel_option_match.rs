//! RAT-36 feature 3: `Option`, fieldless `u32` tags and `match` in the
//! `#[kernel]` DSL (issue #110, part 3 of 3).
//!
//! Every kernel below is checked twice: CPU `eval` runs the real Rust
//! body, and the generated WGSL is parse- and validated by naga. A
//! translation that only "looks right" cannot pass — the same source
//! has to be executable on both sides. `Some`/`None` lower to a
//! generated two-field struct (`Option_f32 { has: u32, value: f32 }`)
//! and `match` lowers to an `if`/`else` chain, so the CPU and GPU
//! semantics are the same branch tree.

use ornis_macros::kernel;

fn validate_module(name: &str, source: &str) {
    let module =
        naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("{name} must parse: {e}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{name} must validate: {e:?}"));
}

/// `match` over `Option<f32>` in tail position: the payload binding
/// comes from the `has` discriminant, `None` falls through to a
/// literal arm.
#[kernel]
fn opt_match(x: Option<f32>) -> f32 {
    match x {
        Some(v) => v * 2.0,
        None => 0.0,
    }
}

/// `Option` payloads are not limited to `f32`: a glam vector carries
/// through the same generated struct.
#[kernel]
fn opt_vec(v: Option<glam::Vec3>) -> f32 {
    match v {
        Some(p) => p.x + p.y,
        None => 1.0,
    }
}

/// A fieldless enum has no WGSL spelling of its own — the DSL spells it
/// as a `u32` tag and matches the tag with literal arms. This is the
/// `enum` half of issue #110's `Option`/`enum`/`match` item.
#[kernel]
fn tag_match(tag: u32) -> u32 {
    match tag {
        0u32 => 10u32,
        1u32 => 20u32,
        _ => 0u32,
    }
}

/// `if let Some(v) = e`: the same discriminant test as `match`, spelled
/// as a conditional binding.
#[kernel]
fn if_let_opt(x: Option<u32>) -> u32 {
    let mut out = 7u32;
    if let Some(v) = x {
        out = v + 1u32;
    }
    out
}

/// `Some`/`None` as values: the annotation on the `let` is the type
/// context that names the generated struct.
#[kernel]
fn option_roundtrip(x: f32) -> f32 {
    let a: Option<f32> = Some(x + 1.0);
    let b: Option<f32> = None;
    let picked = match a {
        Some(v) => v,
        None => 0.0,
    };
    let mut fallback = 0.5;
    if let Some(v) = b {
        fallback = v;
    }
    picked + fallback
}

/// Reading an `Option` payload without `match`: `if let` binds it.
#[kernel]
fn value_or(o: Option<f32>, fallback: f32) -> f32 {
    let mut out = fallback;
    if let Some(v) = o {
        out = v;
    }
    out
}

/// `return match …` — the same value form, with the `return` folded
/// into the trailing binding.
#[kernel]
fn return_match(x: u32) -> u32 {
    return match x {
        1u32 => 100u32,
        2u32 => 200u32,
        _ => 300u32,
    };
}

/// A `None` return: the function's return type is the only context the
/// tail expression needs.
#[kernel]
fn make_none(flag: u32) -> Option<f32> {
    if flag == 0u32 {
        return None;
    }
    Some(1.5)
}

#[test]
fn option_match_eval_matches_rust_semantics() {
    assert_eq!(opt_match::eval(Some(3.0)), 6.0);
    assert_eq!(opt_match::eval(Some(-1.5)), -3.0);
    assert_eq!(opt_match::eval(None), 0.0);
}

#[test]
fn option_match_wgsl_validates() {
    let src = opt_match::wgsl_source();
    assert!(src.starts_with("struct Option_f32 {"), "{src}");
    assert!(src.contains("has: u32"), "{src}");
    assert!(src.contains("value: f32"), "{src}");
    assert!(src.contains("x.has != 0u"), "{src}");
    assert!(src.contains("x.value"), "{src}");
    validate_module(opt_match::label(), src);
}

#[test]
fn option_vec_payload_carries_through() {
    assert_eq!(opt_vec::eval(Some(glam::Vec3::new(1.0, 2.0, 3.0))), 3.0);
    assert_eq!(opt_vec::eval(None), 1.0);
    let src = opt_vec::wgsl_source();
    assert!(src.contains("struct Option_vec3_f32 {"), "{src}");
    assert!(src.contains("value: vec3<f32>"), "{src}");
    validate_module(opt_vec::label(), src);
}

#[test]
fn fieldless_enum_lowers_to_u32_tag_match() {
    assert_eq!(tag_match::eval(0), 10);
    assert_eq!(tag_match::eval(1), 20);
    assert_eq!(tag_match::eval(2), 0);
    let src = tag_match::wgsl_source();
    assert!(src.contains("tag == 0u"), "{src}");
    assert!(src.contains("tag == 1u"), "{src}");
    validate_module(tag_match::label(), src);
}

#[test]
fn if_let_binds_payload() {
    assert_eq!(if_let_opt::eval(Some(41)), 42);
    assert_eq!(if_let_opt::eval(None), 7);
    let src = if_let_opt::wgsl_source();
    assert!(src.contains("x.has != 0u"), "{src}");
    assert!(src.contains("x.value"), "{src}");
    validate_module(if_let_opt::label(), src);
}

#[test]
fn some_and_none_lower_from_the_annotation() {
    assert_eq!(option_roundtrip::eval(1.0), 2.5);
    assert_eq!(option_roundtrip::eval(0.0), 1.5);
    let src = option_roundtrip::wgsl_source();
    assert!(src.contains("Option_f32(1u,"), "{src}");
    assert!(src.contains("Option_f32(0u, 0.0)"), "{src}");
    validate_module(option_roundtrip::label(), src);
}

#[test]
fn value_or_reads_payload_without_match() {
    assert_eq!(value_or::eval(Some(9.0), 1.0), 9.0);
    assert_eq!(value_or::eval(None, 1.0), 1.0);
    validate_module(value_or::label(), value_or::wgsl_source());
}

#[test]
fn return_match_expands_like_the_tail_form() {
    assert_eq!(return_match::eval(1), 100);
    assert_eq!(return_match::eval(2), 200);
    assert_eq!(return_match::eval(9), 300);
    let src = return_match::wgsl_source();
    assert!(src.contains("ornis_match_result"), "{src}");
    validate_module(return_match::label(), src);
}

#[test]
fn none_return_uses_the_function_return_type() {
    assert_eq!(make_none::eval(0), None);
    assert_eq!(make_none::eval(1), Some(1.5));
    let src = make_none::wgsl_source();
    assert!(src.contains("Option_f32(0u, 0.0)"), "{src}");
    assert!(src.contains("Option_f32(1u, 1.5)"), "{src}");
    validate_module(make_none::label(), src);
}
