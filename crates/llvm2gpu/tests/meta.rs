//! Metadata record reader for the `#[gpu_entry]` record.
//!
//! The record is what lets `xtask build-shaders` drop its `--workgroup` and
//! `--types` flags, so the reader has to accept both the current
//! `uniform=<id>:<tags>` form and the `types=<tags>` form the first attribute
//! version wrote, and it has to reject a record it only half understands: a
//! guessed workgroup or tag list would silently change the shader.

use ornis_llvm2gpu::meta;
use ornis_llvm2gpu::{extract_meta, parse_meta};

/// An IR module that only carries a record.
fn module(record: &str) -> String {
    format!(
        "@ornis_gpu_shader_meta = private constant [{len} x i8] c\"{record}\", section \
         \"ornis_gpu.shader_meta\", align 1\n\
         define dso_local noundef float @cs_main(i32 noundef %id) unnamed_addr #1 {{ start: ret float 0.000000e+00 }}",
        len = record.len() + 1,
    )
}

#[test]
fn current_record_lists_one_struct_per_uniform_binding() {
    let ir = module("entry=cs_main;workgroup=8,4,1;uniform=0:6666662,1:11");
    let meta = extract_meta(&ir)
        .expect("record parses")
        .expect("record present");
    assert_eq!(meta.entry, "cs_main");
    assert_eq!(meta.workgroup, [8, 4, 1]);
    assert_eq!(meta.tags(0), Some([6, 6, 6, 6, 6, 6, 2].as_slice()));
    assert_eq!(meta.tags(1), Some([1, 1].as_slice()));
    assert_eq!(meta.tags(2), None);
}

#[test]
fn nested_struct_tag_consumes_the_following_id_digit() {
    let meta = parse_meta("entry=cs_main;workgroup=1,1,1;uniform=0:671,1:1").expect("record");
    assert_eq!(
        meta.tags(0),
        Some([6, ornis_gpu_abi::TYPE_TAG_NESTED + 1].as_slice())
    );
    assert_eq!(meta.tags(1), Some([1].as_slice()));
}

#[test]
fn older_types_record_still_reads() {
    let ir = module("entry=cs_main;workgroup=64,1,1;types=6,2");
    let meta = extract_meta(&ir)
        .expect("record parses")
        .expect("record present");
    assert_eq!(meta.tags(0), Some([6, 2].as_slice()));
}

#[test]
fn table_keeps_struct_ids_stable_when_an_id_is_missing() {
    // The lowerer indexes structs by id, so id 2 must not slide into slot 1.
    let meta = parse_meta("entry=cs_main;workgroup=1,1,1;uniform=0:62,2:11").expect("record");
    let table = meta.table();
    assert_eq!(table.len(), 3);
    assert_eq!(table[0], [6, 2]);
    assert!(table[1].is_empty());
    assert_eq!(table[2], [1, 1]);
}

#[test]
fn a_shader_without_a_record_is_not_an_error() {
    let ir = "define dso_local void @cs_main(i32 noundef %id) unnamed_addr #0 { start: ret void }";
    let found = meta::extract(ir).expect("no record");
    assert_eq!(found, None);
}

#[test]
fn malformed_records_are_rejected() {
    for record in [
        "entry=cs_main;workgroup=0,1,1",
        "entry=cs_main;workgroup=8",
        "entry=cs_main",
        "entry=cs_main;workgroup=8,1,1;uniform=0:6a",
        "entry=cs_main;workgroup=8,1,1;uniform=0:67",
        "entry=cs_main;workgroup=8,1,1;uniform=0",
        "entry=cs_main;workgroup=8,1,1;types=",
    ] {
        let parsed = meta::parse(record);
        assert!(parsed.is_err(), "record `{record}` must not parse");
    }
}
