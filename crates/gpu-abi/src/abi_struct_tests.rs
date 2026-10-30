//! Safe-API passage for structs and arrays of structs.
//!
//! `gpu_struct!` publishes `FIELD_OFFSETS` and the `GpuSafeLayout`
//! constants; a token walks an array binding by `index_storage` over the
//! struct stride and lands on fields with `field`. The tests pin the
//! mixed-width layout (vec3 slot, scalar at 12, span padded to the
//! 32-byte element stride), whole-struct reads and writes through
//! tokens, and the refusals: an unaligned raw field offset dies while
//! the token is built, an index past the binding dies at the window —
//! never a silent neighbour write.

use crate::GpuSafeLayout;
use crate::GpuVec3;
use crate::HostMemory;
use crate::Storage;
use crate::gpu_struct;
use crate::with_host;

gpu_struct! {
    struct SceneSphere {
        center: GpuVec3,
        radius: f32,
    }
}

gpu_struct! {
    struct MixedStruct {
        center: GpuVec3,
        kind: u32,
        weight: f32,
    }
}

fn sample(radius: f32) -> SceneSphere {
    SceneSphere {
        center: GpuVec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        },
        radius,
    }
}

#[test]
fn mixed_fields_land_where_std430_puts_them() {
    // vec3 owns a 16-byte slot: `kind` sits at 12, `weight` at 16, and
    // the 16-alignment pads the span to 32 — the element stride.
    assert!(MixedStruct::FIELD_OFFSETS[0] == 0);
    assert!(MixedStruct::FIELD_OFFSETS[1] == 12);
    assert!(MixedStruct::FIELD_OFFSETS[2] == 16);
    assert!(core::hint::black_box(MixedStruct::SIZE) == 32);
    assert!(core::hint::black_box(MixedStruct::STORAGE_STRIDE) == 32);
    // Each published offset is aligned for its own field type, so the
    // tokens the offsets build pass the construction gates.
    let storage = Storage::<0, 5, MixedStruct>::bind();
    let _kind = storage.as_ref().field::<u32>(MixedStruct::FIELD_OFFSETS[1]);
    let _weight = storage.as_ref().field::<f32>(MixedStruct::FIELD_OFFSETS[2]);
}

#[test]
fn array_of_structs_roundtrips_through_tokens() {
    let mut mem = HostMemory::new();
    mem.bind_storage::<0, 6, [SceneSphere; 3]>();
    let start = [sample(0.5), sample(1.5), sample(2.5)];
    mem.write_storage::<0, 6, [SceneSphere; 3]>(&start);
    with_host(&mut mem, || {
        let storage = Storage::<0, 6, [SceneSphere; 3]>::bind();
        let elements = storage.as_ref().field::<SceneSphere>(0);
        // Whole-struct read: element 1 at its 16-byte stride.
        let middle = elements.index_storage(1).read();
        assert!(middle.radius == 1.5);
        assert!(middle.center.z == 3.0);
        // Whole-struct write: element 0 replaced through the token.
        elements.index_storage(0).write(sample(4.0));
        // Field write: element 2's radius only, neighbours untouched.
        elements
            .index_storage(2)
            .field::<f32>(SceneSphere::FIELD_OFFSETS[1])
            .write(9.5);
    });
    let got = mem.read_storage::<0, 6, [SceneSphere; 3]>();
    assert!(got[0].radius == 4.0);
    assert!(got[1].radius == 1.5);
    assert!(got[2].radius == 9.5);
    assert!(got[2].center.x == 1.0);
}

#[test]
#[should_panic(expected = "GpuRef field offset is misaligned")]
fn an_unaligned_field_offset_on_a_struct_token_is_a_contract_breach() {
    let storage = Storage::<0, 7, MixedStruct>::bind();
    // Two is not a multiple of u32's alignment — the token never exists.
    let _ = storage.as_ref().field::<u32>(2);
}

#[test]
#[should_panic(expected = "gpu access out of bounds")]
fn an_index_past_the_struct_array_is_refused_at_the_window() {
    let mut mem = HostMemory::new();
    mem.bind_storage::<0, 8, [SceneSphere; 3]>();
    with_host(&mut mem, || {
        let storage = Storage::<0, 8, [SceneSphere; 3]>::bind();
        let _ = storage
            .as_ref()
            .field::<SceneSphere>(0)
            .index_storage(3)
            .read();
    });
}
