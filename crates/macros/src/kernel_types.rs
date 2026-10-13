//! `#[kernel]` type allowlist: which Rust types keep their values into WGSL.
//!
//! One home for the check shared by parameter, return-type and
//! `let`-annotation validation in [`crate::kernel`], so the call sites
//! cannot drift apart. Callers use `kernel_types::kernel_type_error`
//! paths; the helper itself is pure logic over external calls only.

use quote::quote;
use syn::Type;

/// Classifies one type named in a `#[kernel]` signature or `let`
/// annotation: `None` when lowering preserves it exactly, otherwise the
/// error message for the caller's span. `what` names the position
/// (`"type"`, `"return type"`, `"let annotation"`).
///
/// Single home for both name lists — heap types (rejected as before)
/// and supported spellings (`f32`/`u32`/`i32`, verbatim `bool`, and the
/// vector/matrix registry on the last path segment, so `glam::Vec3`
/// works). The vector list mirrors
/// [`ornis_shader_lang::ShaderType::from_rust`]; the unit test below
/// pins them together.
///
/// Integer and float literal suffixes are checked separately
/// (`KernelValidator::visit_lit_int`/`visit_lit_float`): only plain,
/// `u32`/`i32` and `f32`-suffixed literals keep their values into WGSL.
pub(crate) fn kernel_type_error(ty: &Type, what: &str, fn_name: &str) -> Option<String> {
    // Descend through array nesting with a loop: a recursive call would
    // read as a project-local call next to branching.
    let mut current = ty;
    loop {
        match current {
            Type::Path(tp) => {
                let name = tp
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                if matches!(
                    name.as_str(),
                    "Vec"
                        | "String"
                        | "Box"
                        | "Rc"
                        | "Arc"
                        | "HashMap"
                        | "HashSet"
                        | "VecDeque"
                        | "LinkedList"
                        | "BTreeMap"
                        | "BTreeSet"
                ) {
                    return Some(format!(
                        "{what} `{}` is not allowed in GPU kernel `{fn_name}`; use f32, i32, u32, or glam vectors instead",
                        quote!(#ty),
                    ));
                }
                if !(matches!(name.as_str(), "f32" | "u32" | "i32" | "bool")
                    || matches!(
                        name.as_str(),
                        "Vec2"
                            | "vec2"
                            | "Vec3"
                            | "vec3"
                            | "Vec2A"
                            | "Vec3A"
                            | "Vec4"
                            | "vec4"
                            | "Quat"
                            | "Mat3"
                            | "mat3"
                            | "Mat4"
                            | "mat4"
                            | "UVec2"
                            | "UVec3"
                            | "UVec4"
                            | "IVec2"
                            | "IVec3"
                            | "IVec4"
                            | "BVec2"
                            | "BVec3"
                            | "BVec4"
                            | "MaterialIdx"
                            | "TextureHandle"
                    ))
                {
                    return Some(format!(
                        "{what} `{}` is not supported in GPU kernel `{fn_name}`; supported: f32, u32, i32, bool and glam vectors/matrices",
                        quote!(#ty),
                    ));
                }
                return None;
            }
            // Fixed-size scratch (`[f32; 6]`): the length is documentation,
            // the element type must be allowed on its own.
            Type::Array(arr) => current = &arr.elem,
            _ => {
                return Some(format!(
                    "{what} `{}` is not supported in GPU kernel `{fn_name}`; supported: f32, u32, i32, bool and glam vectors/matrices",
                    quote!(#ty),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::kernel_type_error;
    use syn::Type;

    fn type_error(ty: &str) -> Option<String> {
        let parsed: Type = syn::parse_str(ty).expect("test type must parse");
        kernel_type_error(&parsed, "type", "test_kernel")
    }

    #[test]
    fn supported_types_pass_validation() {
        for ty in [
            "f32",
            "u32",
            "i32",
            "bool",
            "Vec3",
            "glam::Vec3",
            "Vec2",
            "Mat4",
            "UVec2",
            "IVec3",
            "BVec4",
            "Quat",
            "MaterialIdx",
            "[f32; 6]",
        ] {
            assert_eq!(type_error(ty), None, "{ty}");
        }
    }

    #[test]
    fn unsupported_types_fail_validation() {
        for ty in ["u8", "i16", "u64", "usize", "f64", "MyStruct", "&f32"] {
            assert!(type_error(ty).is_some(), "{ty}");
        }
    }

    /// The validator's vector list must stay in sync with the shader
    /// registry: every [`ornis_shader_lang::ShaderType`] spelling lowers
    /// exactly and must validate clean.
    #[test]
    fn vector_list_matches_shader_registry() {
        for rust in [
            "Vec2",
            "vec2",
            "Vec2A",
            "Vec3",
            "vec3",
            "Vec3A",
            "Vec4",
            "vec4",
            "Quat",
            "Mat3",
            "mat3",
            "Mat4",
            "mat4",
            "UVec2",
            "UVec3",
            "UVec4",
            "IVec2",
            "IVec3",
            "IVec4",
            "BVec2",
            "BVec3",
            "BVec4",
            "MaterialIdx",
            "TextureHandle",
        ] {
            assert!(
                ornis_shader_lang::ShaderType::from_rust(rust).is_some(),
                "registry must know {rust}"
            );
            assert_eq!(type_error(rust), None, "{rust}");
        }
    }
}
