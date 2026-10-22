//! RAT-36 feature 1: `Entity` handles and read-only foreign-field
//! gathers in the `#[kernel]` DSL (issue #110, part 1 of 3).
//!
//! Every kernel below is checked twice: CPU `eval` runs the real Rust
//! body against `&[T]` lane slices, and the generated WGSL is parse-
//! and validated by naga. A translation that only "looks right" cannot
//! pass — the same source has to be executable on both sides. `t.field`
//! on an entity binding lowers to a per-lane gather helper over a
//! module-scope storage array; `eval` mirrors it with `_ornis_gather`,
//! the same index-bounds condition yielding the element zero.

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

/// A basic annotated gather: `t.position` reads lane slot `t.index`.
#[kernel]
fn gather_position(t: Entity) -> f32 {
    let p: glam::Vec3 = t.position;
    p.x
}

/// Scalar lanes work like vector lanes; the annotation is the storage
/// array's element type.
#[kernel]
fn gather_hp(t: Entity) -> u32 {
    let hp: u32 = t.hp;
    hp
}

/// Two fields on one binding get two lanes, numbered in sorted
/// (binding, field) order — `hp` at binding 0, `position` at 1.
#[kernel]
fn two_fields(t: Entity) -> f32 {
    let p: glam::Vec3 = t.position;
    let hp: u32 = t.hp;
    p.x + hp as f32
}

/// `t.index` / `t.generation` are real struct fields on both sides —
/// no lane, no annotation.
#[kernel]
fn index_passthrough(t: Entity) -> u32 {
    t.index
}

/// `Option<Entity>`: the generated struct mirrors the handle, and
/// `match` binds entity values from the payload.
#[kernel]
fn option_entity(oe: Option<Entity>) -> u32 {
    match oe {
        Some(e) => e.index + e.generation,
        None => 0u32,
    }
}

/// A gather through an `if let` binding: the pattern binding is the
/// entity, its field read is an ordinary annotated gather.
#[kernel]
fn if_let_gather(oe: Option<Entity>) -> f32 {
    let mut out = 0.0;
    if let Some(e) = oe {
        let p: glam::Vec3 = e.position;
        out = p.y;
    }
    out
}

/// Entity-typed lane: a gathered owner handle, checked through its
/// generation field.
#[kernel]
fn owner_lane(t: Entity) -> u32 {
    let o: Entity = t.owner;
    o.index
}

/// The one un-`let` gather position: the function tail, typed by the
/// return type.
#[kernel]
fn tail_read(t: Entity) -> glam::Vec3 {
    t.position
}

/// Constructing a handle: the struct literal is a positional
/// constructor in WGSL and the emitted CPU struct in `eval`.
#[kernel]
fn make_entity(i: u32) -> Entity {
    Entity {
        index: i,
        generation: 1u32,
    }
}

/// Comparing passthrough fields (`u32` vs `u32`) is ordinary DSL
/// arithmetic — only whole-handle `==` is rejected.
#[kernel]
fn compare_index(t: Entity, other: Entity) -> u32 {
    if t.index == other.index { 1u32 } else { 0u32 }
}

#[test]
fn gather_position_eval_reads_the_lane() {
    let lane = [
        glam::Vec3::new(1.5, 2.0, 3.0),
        glam::Vec3::new(-4.0, 5.0, 6.0),
    ];
    let e0 = gather_position::Entity {
        index: 0,
        generation: 1,
    };
    let e1 = gather_position::Entity {
        index: 1,
        generation: 1,
    };
    assert_eq!(gather_position::eval(e0, &lane), 1.5);
    assert_eq!(gather_position::eval(e1, &lane), -4.0);
    // Out of range: the element zero, same as the WGSL helper.
    let oob = gather_position::Entity {
        index: 9,
        generation: 1,
    };
    assert_eq!(gather_position::eval(oob, &lane), 0.0);
}

#[test]
fn gather_position_wgsl_validates() {
    let src = gather_position::wgsl_source();
    assert!(src.starts_with("struct Entity {"), "{src}");
    assert!(src.contains("index: u32"), "{src}");
    assert!(src.contains("generation: u32"), "{src}");
    assert!(
        src.contains(
            "@group(0) @binding(0) var<storage, read> _ornis_lane_t__position: array<vec3<f32>>;"
        ),
        "{src}"
    );
    assert!(
        src.contains("fn _ornis_gather_t__position(e: Entity) -> vec3<f32>"),
        "{src}"
    );
    assert!(
        src.contains("arrayLength(&_ornis_lane_t__position)"),
        "{src}"
    );
    assert!(
        src.contains("return _ornis_lane_t__position[e.index];"),
        "{src}"
    );
    assert!(src.contains("_ornis_gather_t__position(t)"), "{src}");
    validate_module(gather_position::label(), src);
}

#[test]
fn scalar_lane_carries_u32() {
    assert_eq!(
        gather_hp::eval(
            gather_hp::Entity {
                index: 1,
                generation: 1
            },
            &[10, 42]
        ),
        42
    );
    let src = gather_hp::wgsl_source();
    assert!(
        src.contains("var<storage, read> _ornis_lane_t__hp: array<u32>;"),
        "{src}"
    );
    assert!(
        src.contains("fn _ornis_gather_t__hp(e: Entity) -> u32"),
        "{src}"
    );
    assert!(src.contains("return 0u;"), "{src}");
    validate_module(gather_hp::label(), src);
}

#[test]
fn two_fields_get_sorted_lanes() {
    let e = two_fields::Entity {
        index: 0,
        generation: 1,
    };
    assert_eq!(
        two_fields::eval(e, &[7u32], &[glam::Vec3::new(2.0, 0.0, 0.0)]),
        9.0
    );
    let src = two_fields::wgsl_source();
    // Sorted (binding, field) order: hp before position.
    let hp = src.find("_ornis_lane_t__hp").expect("hp lane");
    let position = src.find("_ornis_lane_t__position").expect("position lane");
    assert!(hp < position, "{src}");
    assert!(
        src.contains("@group(0) @binding(0) var<storage, read> _ornis_lane_t__hp: array<u32>;"),
        "{src}"
    );
    assert!(
        src.contains("@group(0) @binding(1) var<storage, read> _ornis_lane_t__position"),
        "{src}"
    );
    validate_module(two_fields::label(), src);
}

#[test]
fn index_passthrough_needs_no_lane() {
    let e = index_passthrough::Entity {
        index: 3,
        generation: 1,
    };
    assert_eq!(index_passthrough::eval(e), 3);
    let src = index_passthrough::wgsl_source();
    assert!(src.contains("struct Entity {"), "{src}");
    assert!(!src.contains("var<storage"), "{src}");
    assert!(src.contains("t.index"), "{src}");
    validate_module(index_passthrough::label(), src);
}

#[test]
fn option_entity_matches_on_the_payload() {
    let e = option_entity::Entity {
        index: 2,
        generation: 5,
    };
    assert_eq!(option_entity::eval(Some(e)), 7);
    assert_eq!(option_entity::eval(None), 0);
    let src = option_entity::wgsl_source();
    assert!(src.contains("struct Option_Entity {"), "{src}");
    assert!(src.contains("value: Entity"), "{src}");
    validate_module(option_entity::label(), src);
}

#[test]
fn if_let_gathers_through_the_binding() {
    let e = if_let_gather::Entity {
        index: 1,
        generation: 1,
    };
    let lane = [glam::Vec3::ZERO, glam::Vec3::new(0.0, 9.0, 0.0)];
    assert_eq!(if_let_gather::eval(Some(e), &lane), 9.0);
    assert_eq!(if_let_gather::eval(None, &lane), 0.0);
    let src = if_let_gather::wgsl_source();
    assert!(
        src.contains("_ornis_lane_e__position"),
        "lane binds to the pattern name: {src}"
    );
    validate_module(if_let_gather::label(), src);
}

#[test]
fn owner_lane_gathers_entity_values() {
    let e = owner_lane::Entity {
        index: 0,
        generation: 1,
    };
    let lane = [owner_lane::Entity {
        index: 4,
        generation: 2,
    }];
    assert_eq!(owner_lane::eval(e, &lane), 4);
    // Out of range yields the null handle; `.index` is 0.
    let oob = owner_lane::Entity {
        index: 7,
        generation: 1,
    };
    assert_eq!(owner_lane::eval(oob, &lane), 0);
    let src = owner_lane::wgsl_source();
    assert!(
        src.contains("var<storage, read> _ornis_lane_t__owner: array<Entity>;"),
        "{src}"
    );
    assert!(src.contains("return Entity(0u, 0u);"), "{src}");
    validate_module(owner_lane::label(), src);
}

#[test]
fn tail_read_is_return_typed() {
    let e = tail_read::Entity {
        index: 0,
        generation: 1,
    };
    assert_eq!(
        tail_read::eval(e, &[glam::Vec3::new(4.0, 5.0, 6.0)]),
        glam::Vec3::new(4.0, 5.0, 6.0)
    );
    let src = tail_read::wgsl_source();
    assert!(
        src.contains("return _ornis_gather_t__position(t);"),
        "{src}"
    );
    validate_module(tail_read::label(), src);
}

#[test]
fn make_entity_is_a_positional_constructor() {
    let e = make_entity::eval(3);
    assert_eq!(
        e,
        make_entity::Entity {
            index: 3,
            generation: 1
        }
    );
    let src = make_entity::wgsl_source();
    assert!(
        src.contains("Entity(i, 1u) /* index, generation */"),
        "{src}"
    );
    validate_module(make_entity::label(), src);
}

#[test]
fn compare_index_stays_u32_comparison() {
    let a = compare_index::Entity {
        index: 1,
        generation: 1,
    };
    let b = compare_index::Entity {
        index: 1,
        generation: 9,
    };
    let c = compare_index::Entity {
        index: 2,
        generation: 1,
    };
    assert_eq!(compare_index::eval(a, b), 1);
    assert_eq!(compare_index::eval(a, c), 0);
    validate_module(compare_index::label(), compare_index::wgsl_source());
}
