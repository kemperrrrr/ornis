//! The guest's `#[shader::*]` macros leave entry metadata in the wasm module.
//!
//! The lifter and translator rely on that section for stage and workgroup
//! size. This test reads the section from the real `wasm32` build so a lost
//! `#[used]` static or a renamed section fails here, not as a quiet fallback
//! to the ABI default.

use ornis_gpu_abi::entry_meta::EntryPoint;
use ornis_gpu_abi::entry_meta::Stage;
use ornis_gpu_abi::entry_meta::decode;
use ornis_gpu_abi::symbols;
use wasmparser::Parser;
use wasmparser::Payload;

fn entries() -> Vec<EntryPoint> {
    let wasm = super::compile_wasm();
    entries_of(&wasm)
}

fn entries_of(wasm: &[u8]) -> Vec<EntryPoint> {
    let mut entries = Vec::new();
    for payload in Parser::new(0).parse_all(wasm) {
        let payload = payload.expect("wasm");
        if let Payload::CustomSection(section) = payload
            && section.name() == symbols::ENTRY_SECTION
        {
            let text = std::str::from_utf8(section.data()).expect("entry metadata utf8");
            entries.extend(decode(text).expect("entry metadata records"));
        }
    }
    entries
}

fn entry(entries: &[EntryPoint], export: &str) -> EntryPoint {
    entries
        .iter()
        .find(|entry| entry.export == export)
        .unwrap_or_else(|| panic!("{export} has no entry metadata"))
        .clone()
}

#[test]
fn every_guest_entry_declares_its_stage_and_workgroup() {
    let entries = entries();
    for export in ["compute", "cull", "ptr_copy", "vs_main", "ae_adapt"] {
        assert_eq!(entry(&entries, export).stage, Stage::Compute, "{export}");
        assert_eq!(entry(&entries, export).workgroup, [4, 1, 1], "{export}");
    }
    assert_eq!(entry(&entries, "ae_log_luma").stage, Stage::Compute);
    assert_eq!(entry(&entries, "ae_histogram").stage, Stage::Compute);
    assert_eq!(entry(&entries, "shade").stage, Stage::Fragment);
    assert_eq!(entry(&entries, "shade").workgroup, [1, 1, 1]);
}

#[test]
fn the_lifter_carries_metadata_into_the_toy_module() {
    let wasm = super::compile_wasm();
    let wat = super::compute_lift::lift_wat(&wasm, "cull");
    let lifted = wat::parse_str(&wat).expect("lifted wat");
    let lifted = entries_of(&lifted);
    let cs_main = entry(&lifted, symbols::CS_MAIN);
    assert_eq!(cs_main.stage, Stage::Compute);
    assert_eq!(cs_main.workgroup, [4, 1, 1]);
}
