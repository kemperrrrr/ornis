//! Shared wat helpers for the uniform integration tests.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::TranslateError;
use ornis_wasm2gpu::translate;

pub(super) fn vertex() -> String {
    format!(
        r#"(func (export "{vs}") (param f32 f32 f32 f32) (result f32 f32 f32 f32 f32 f32)
            local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)"#,
        vs = symbols::VS_MAIN,
    )
}

pub(super) fn wasm(custom: &str, imports: &str, fs: &str) -> Vec<u8> {
    wat::parse_str(format!(
        "(module {custom} {imports} {vs} (func (export \"{name}\") {fs}))",
        vs = vertex(),
        name = symbols::FS_MAIN,
    ))
    .expect("wat")
}

pub(super) fn import(name: &str, sig: &str) -> String {
    format!(
        "(import \"{m}\" \"{name}\" (func ${name} {sig}))",
        m = symbols::MODULE,
    )
}

pub(super) fn wgsl(custom: &str, imports: &str, fs: &str) -> String {
    translate(&wasm(custom, imports, fs))
        .expect("translate")
        .wgsl
}

#[allow(dead_code)]
pub(super) fn err(custom: &str, imports: &str, fs: &str) -> TranslateError {
    translate(&wasm(custom, imports, fs)).expect_err("reject")
}

#[allow(dead_code)]
pub(super) fn types(body: &str) -> String {
    format!(
        "(@custom \"{name}\" \"{body}\")",
        name = symbols::TYPES_SECTION,
    )
}

pub(super) fn rgba() -> &'static str {
    "f32.const 0 f32.const 0 f32.const 0 f32.const 1"
}

#[allow(dead_code)]
pub(super) fn type_msg(err: TranslateError) -> String {
    match err {
        TranslateError::UnsupportedType(msg) => msg,
        other => panic!("unexpected {other}"),
    }
}

pub(super) fn parsed(wgsl: &str) -> naga::Module {
    let module = naga::front::wgsl::parse_str(wgsl).expect("parse lowered wgsl");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|err| panic!("naga validate: {err}\n{wgsl}"));
    module
}

pub(super) fn struct_layout(wgsl: &str, fields: usize) -> (Vec<u32>, u32) {
    let module = parsed(wgsl);
    for (_, ty) in module.types.iter() {
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            continue;
        };
        if members.len() < fields || !data_struct(&module, members) {
            continue;
        }
        let offsets = members
            .iter()
            .take(fields)
            .map(|member| member.offset)
            .collect();
        return (offsets, *span);
    }
    panic!("no struct with {fields} fields in {wgsl}");
}

pub(super) fn data_struct(module: &naga::Module, members: &[naga::StructMember]) -> bool {
    if members.iter().any(|member| member.binding.is_some()) {
        return false;
    }
    !array_wrapper(module, members)
}

pub(super) fn array_wrapper(module: &naga::Module, members: &[naga::StructMember]) -> bool {
    members.first().is_some_and(|member| {
        matches!(module.types[member.ty].inner, naga::TypeInner::Array { .. })
    })
}
