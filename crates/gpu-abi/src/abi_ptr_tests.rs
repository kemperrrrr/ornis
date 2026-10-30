//! Pointer-path token gates: the address `GpuRef` hands to `load` / `store`
//! must be aligned for `T`, so a bad stride or field offset is a contract
//! breach while the token is built — never a read of the neighbouring
//! element through a misaligned dynamic index.

use crate::Storage;

#[test]
#[should_panic(expected = "GpuRef index stride is misaligned")]
fn unaligned_dynamic_index_is_a_contract_breach() {
    let storage = Storage::<0, 0, f32>::bind();
    let stride = core::hint::black_box(3u32);
    let _ = storage.as_ref().index(core::hint::black_box(1), stride).read();
}

#[test]
#[should_panic(expected = "GpuRef field offset is misaligned")]
fn misaligned_field_offset_is_a_contract_breach() {
    let storage = Storage::<0, 0, f32>::bind();
    let _ = storage.as_ref().field::<f32>(2).read();
}
