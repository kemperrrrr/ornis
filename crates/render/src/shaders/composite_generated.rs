//! Composite shader generated from Rust (Render path 2).
//!
//! Канонический источник — Rust-код этого модуля; WGSL выводится сборкой
//! строки из констант + `srgb_to_linear::wgsl_source()` (ядро из
//! `crates/render/src/shaders/math.rs` через `#[kernel]`). Рукописный
//! `shaders/wgsl/composite.wgsl` остаётся как reference/legacy, но
//! `composite.rs` (LegacyCompositePass) уже использует только этот модуль.

use crate::shaders::math::srgb_to_linear;

/// WGSL bindings + quad constants + vertex/fragment entry points.
///
/// Собирается в runtime как `String`, но источник — Rust: константы и
/// `srgb_to_linear::wgsl_source()` — единственный `srgb_to_linear` в
/// системе. Это убирает дублирование WGSL-литерала из `composite.rs`.
fn composite_wgsl_body() -> String {
    // Header: bindings, VertexOutput, QUAD/UVS, vertex entry.
    // Формат идентичен `shaders/wgsl/composite.wgsl`; имена entry points `vs`/`fs`
    // сохранены для совместимости с `CompositePass::new`.
    let header = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var pbr_tex: texture_2d<f32>;
@group(0) @binding(1) var pbr_sampler: sampler;
@group(0) @binding(2) var ui_tex: texture_2d<f32>;
@group(0) @binding(3) var ui_sampler: sampler;

const QUAD: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0,  1.0, 0.0, 1.0),
    vec4<f32>( 1.0, -1.0, 0.0, 1.0),
    vec4<f32>( 1.0,  1.0, 0.0, 1.0),
);
const UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0),
);

@vertex
fn vs(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = QUAD[idx];
    out.uv = UVS[idx];
    return out;
}
"#;

    // Fragment entry: sampling + sRGB decode + mix. Использует `srgb_to_linear`
    // из kernel (в WGSL имя совпадает).
    let fragment = r#"
@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let bg = textureSampleLevel(pbr_tex, pbr_sampler, input.uv, 0.0);
    let ui = textureSampleLevel(ui_tex, ui_sampler, input.uv, 0.0);
    let ui_linear = srgb_to_linear(ui.rgb);
    return vec4<f32>(mix(bg.rgb, ui_linear, ui.a), 1.0);
}
"#;

    // Kernel WGSL уже содержит `fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> { ... }`
    let kernel = srgb_to_linear::wgsl_source();
    format!("{header}\n{kernel}\n{fragment}\n")
}

/// Полный WGSL источник composite-пасса, собранный из Rust.
pub fn wgsl_source() -> String {
    composite_wgsl_body()
}

/// Статический вид для naga-валидации в тестах (клонируется из `wgsl_source()`).
/// Используется также для детерминированного snapshot-теста.
pub fn wgsl_source_static() -> String {
    wgsl_source()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_valid_wgsl(name: &str, source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name} must parse: {e}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} must validate: {e}"));
    }

    #[test]
    fn composite_generated_validates_with_naga() {
        assert_valid_wgsl("composite_generated", &wgsl_source());
    }

    #[test]
    fn composite_generated_contains_expected_bindings() {
        let src = wgsl_source();
        assert!(src.contains("@group(0) @binding(0) var pbr_tex"));
        assert!(src.contains("@group(0) @binding(1) var pbr_sampler"));
        assert!(src.contains("@group(0) @binding(2) var ui_tex"));
        assert!(src.contains("@group(0) @binding(3) var ui_sampler"));
        assert!(src.contains("fn vs("));
        assert!(src.contains("fn fs("));
        assert!(src.contains("fn srgb_to_linear"));
    }
}
