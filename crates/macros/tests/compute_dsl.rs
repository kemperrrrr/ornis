//! Solver-grade compute DSL: `mat3x3<f32>` in the type registry, local
//! fixed-size scratch arrays, and helper inclusion. Translation features
//! go through `#[wgsl_fn]` (its surface needs no wgpu); the stitched
//! module (helpers + entry, every line macro-generated) must
//! naga-validate — exactly the shape `#[gpu_pipeline]` assembles with
//! `helpers(...)`.

use ornis_macros::wgsl_fn;

/// Mat3 in every type position maps to `mat3x3<f32>`.
#[wgsl_fn]
fn mat3_types(m: Mat3) -> Mat3 {
    let t = m.transpose();
    return t;
}

/// Column-major construction from three column vectors.
#[wgsl_fn]
fn mat3_assembly(c0: Vec3, c1: Vec3, c2: Vec3) -> Mat3 {
    let m = Mat3::from_cols(c0, c1, c2);
    return m;
}

/// Zero/identity matrix constants.
#[wgsl_fn]
fn mat3_consts() -> Mat3 {
    let z = Mat3::ZERO;
    let _ = z;
    let ident = Mat3::IDENTITY;
    return ident;
}

/// Local fixed-size scratch: declaration, repeat init, index read/write.
#[wgsl_fn]
fn array_scratch(x: f32) -> f32 {
    let mut d: [f32; 6] = [0.0; 6];
    d[0] = x;
    d[1] += 2.0;
    return d[0] + d[1];
}

/// Nested arrays for Hessian-style scratch.
#[wgsl_fn]
fn nested_scratch() -> f32 {
    let mut h: [[f32; 3]; 3] = [[0.0; 3]; 3];
    h[0][0] = 2.0;
    h[1][1] = 3.0;
    return h[0][0] * h[1][1];
}

/// A 3x3 LDL solve without pivoting — the AVBD per-body kernel shape:
/// fixed scratch arrays, a mat3 assembled from columns, indexed
/// row/column access, plain `for` loops. Must naga-validate as a helper.
#[wgsl_fn]
fn ldl_solve_3x3(c0: Vec3, c1: Vec3, c2: Vec3, rhs: Vec3) -> Vec3 {
    let mut l: [[f32; 3]; 3] = [[0.0; 3]; 3];
    let mut d: [f32; 3] = [0.0; 3];
    let m = Mat3::from_cols(c0, c1, c2);
    for i in 0u32..3u32 {
        let mut s = m[i][i];
        for k in 0u32..3u32 {
            if k < i {
                s = s - l[i][k] * d[k] * l[i][k];
            }
        }
        d[i] = s;
        l[i][i] = 1.0;
    }
    let out = Vec3::new(d[0], d[1], d[2]);
    return out;
}

#[test]
fn mat3_type_maps_everywhere() {
    let src = mat3_types::wgsl_source();
    assert!(
        src.starts_with("fn mat3_types(m: mat3x3<f32>) -> mat3x3<f32>"),
        "{src}"
    );
    assert!(src.contains("transpose(m)"), "{src}");
}

#[test]
fn mat3_from_cols_constructs_columns() {
    let src = mat3_assembly::wgsl_source();
    assert!(
        src.contains("mat3x3<f32>(c0, c1, c2)"),
        "columns in order, got: {src}"
    );
}

#[test]
fn mat3_identity_zero_constants() {
    let src = mat3_consts::wgsl_source();
    assert!(
        src.contains("mat3x3<f32>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)"),
        "zero matrix, got: {src}"
    );
    assert!(
        src.contains("mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)"),
        "identity matrix, got: {src}"
    );
}

#[test]
fn fixed_array_decl_init_index() {
    let src = array_scratch::wgsl_source();
    assert!(
        src.contains("var d: array<f32, 6> = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);"),
        "repeat expands, got: {src}"
    );
    assert!(src.contains("d[0] = x;"), "{src}");
    assert!(src.contains("d[1] += 2.0;"), "{src}");
}

#[test]
fn nested_array_for_hessian_scratch() {
    let src = nested_scratch::wgsl_source();
    assert!(src.contains("var h: array<array<f32, 3>, 3> = "), "{src}");
    assert!(src.contains("h[0][0] = 2.0;"), "{src}");
}

/// Entry-shaped caller: the `main` side of the stitch, written in the
/// DSL like every helper — no hand-written WGSL anywhere in this file.
#[wgsl_fn]
fn ldl_entry() -> Vec3 {
    let c0 = Vec3::new(1.0, 0.0, 0.0);
    let c1 = Vec3::new(0.0, 1.0, 0.0);
    let c2 = Vec3::new(0.0, 0.0, 1.0);
    let rhs = Vec3::new(1.0, 1.0, 1.0);
    let r = ldl_solve_3x3(c0, c1, c2, rhs);
    return r;
}

fn validate_module(name: &str, source: &str) {
    let module =
        naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("{name} must parse: {e}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{name} must validate: {e}"));
}

#[test]
fn stitched_helpers_plus_entry_validate_with_naga() {
    // The exact assembly `#[gpu_pipeline]` produces with
    // `helpers(mat3_assembly, ldl_solve_3x3)`: helper sources first,
    // entry last. Every line below comes from a macro, none hand-written.
    let stitched = format!(
        "{}\n{}\n{}",
        mat3_assembly::wgsl_source(),
        ldl_solve_3x3::wgsl_source(),
        ldl_entry::wgsl_source(),
    );
    validate_module("stitched ldl helpers", &stitched);
}
