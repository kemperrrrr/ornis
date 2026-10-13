//! `#[kernel]` type allowlist: which Rust types keep their values into WGSL.
//!
//! One home for the checks shared by parameter, return-type,
//! `let`-annotation and path validation in [`crate::kernel`], so the
//! call sites cannot drift apart. Callers use
//! `kernel_types::kernel_type_error` and
//! `kernel_types::is_scalar_namespaced` paths; the helpers themselves
//! are pure logic over external calls only.

use quote::quote;
use syn::Type;

/// Vector, matrix and transparent-newtype spellings the kernel keeps
/// as-is, matched on the last path segment so qualified `glam::Vec3`
/// works. Mirrors [`ornis_shader_lang::ShaderType::from_rust`]; the
/// unit test below pins them together.
fn is_kernel_vector_ty(name: &str) -> bool {
    matches!(
        name,
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
    )
}

/// Scalar type names whose `::` paths have no WGSL spelling.
fn is_scalar_head(name: &str) -> bool {
    matches!(
        name,
        "u8" | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "f16"
            | "f32"
            | "f64"
            | "f128"
            | "bool"
            | "char"
            | "str"
    )
}

/// Whether a type is a bare scalar name (`u8`, `f32`, …).
fn is_bare_scalar_ty(ty: &Type) -> bool {
    if let Type::Path(tp) = ty
        && tp.qself.is_none()
        && tp.path.segments.len() == 1
        && let Some(seg) = tp.path.segments.first()
    {
        return is_scalar_head(&seg.ident.to_string());
    }
    false
}

/// Whether an expression path is scalar-namespaced (`u8::MAX`,
/// `core::u8::MAX`, `std::primitive::u8::MAX`, `std::f32::consts::PI`,
/// `<u8>::MAX`): scalar associated constants and calls through a scalar
/// type have no WGSL spelling — the writer would join them verbatim and
/// naga would fail only at module creation. Every segment but the last
/// is checked, so a scalar hidden below `core`/`std` cannot slip
/// through. Vector constants (`glam::Vec3::ZERO`) and constructors
/// (`Vec3::new`) resolve in the writer and stay allowed.
pub(crate) fn is_scalar_namespaced(path: &syn::Path, qself: Option<&syn::QSelf>) -> bool {
    if let Some(qself) = qself {
        return is_bare_scalar_ty(&qself.ty);
    }
    path.segments
        .iter()
        .take(path.segments.len().saturating_sub(1))
        .any(|seg| is_scalar_head(&seg.ident.to_string()))
}

/// Renders a type the way Rust spells it (`foo::u32`, `Vec<u8>`,
/// `[f32; 6]`): [`quote!`] separates tokens with spaces (`foo :: u32`),
/// which reads wrong inside error messages.
pub(crate) fn type_string(ty: &Type) -> String {
    match ty {
        Type::Path(tp) if tp.qself.is_none() => path_string(&tp.path, None),
        Type::Array(arr) => {
            let len = &arr.len;
            format!("[{}; {}]", type_string(&arr.elem), quote!(#len))
        }
        Type::Reference(r) => {
            let lifetime = r
                .lifetime
                .as_ref()
                .map(|l| format!("'{} ", l.ident))
                .unwrap_or_default();
            let mutability = if r.mutability.is_some() { "mut " } else { "" };
            format!("&{lifetime}{mutability}{}", type_string(&r.elem))
        }
        _ => quote!(#ty).to_string(),
    }
}

/// Renders an expression path the way Rust spells it (`u8::MAX`,
/// `<u8>::MAX`, `glam::Vec3::new`), without the token-stream spaces
/// [`quote!`] inserts around `::`.
pub(crate) fn path_string(path: &syn::Path, qself: Option<&syn::QSelf>) -> String {
    let Some(qself) = qself else {
        let leading = if path.leading_colon.is_some() {
            "::"
        } else {
            ""
        };
        return format!("{leading}{}", segments_string(path.segments.iter()));
    };
    let pos = qself.position.min(path.segments.len());
    let mut out = format!("<{}", type_string(&qself.ty));
    if qself.as_token.is_some() {
        out.push_str(" as ");
        out.push_str(&segments_string(path.segments.iter().take(pos)));
    }
    out.push_str(">::");
    out.push_str(&segments_string(path.segments.iter().skip(pos)));
    out
}

/// Joins path segments with `::` (`u8::MAX`, `glam::Vec3::new`).
fn segments_string<'a>(segments: impl Iterator<Item = &'a syn::PathSegment>) -> String {
    segments.map(segment_string).collect::<Vec<_>>().join("::")
}

/// Renders one path segment, keeping generic arguments readable
/// (`Vec<u8>`, `Fn(u8) -> bool`).
fn segment_string(segment: &syn::PathSegment) -> String {
    match &segment.arguments {
        syn::PathArguments::None => segment.ident.to_string(),
        syn::PathArguments::AngleBracketed(args) => {
            let args = args
                .args
                .iter()
                .map(generic_argument_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}<{args}>", segment.ident)
        }
        syn::PathArguments::Parenthesized(args) => {
            let inputs = args
                .inputs
                .iter()
                .map(named_argument_string)
                .collect::<Vec<_>>()
                .join(", ");
            let output = match &args.output {
                syn::ReturnType::Default => String::new(),
                syn::ReturnType::Type(_, ty) => format!(" -> {}", type_string(ty)),
            };
            format!("{}({inputs}){output}", segment.ident)
        }
    }
}

/// Renders one parenthesized argument (`u8` in `Fn(u8)`, `x: u8` in
/// `Fn(x: u8)`).
fn named_argument_string(arg: &syn::NamedArg) -> String {
    match &arg.name {
        Some((name, _)) => format!("{name}: {}", type_string(&arg.ty)),
        None => type_string(&arg.ty),
    }
}

/// Renders one generic argument (`u8` in `Vec<u8>`, `'a` in `Cow<'a>`).
fn generic_argument_string(arg: &syn::GenericArgument) -> String {
    match arg {
        syn::GenericArgument::Type(ty) => type_string(ty),
        syn::GenericArgument::Lifetime(lifetime) => format!("'{}", lifetime.ident),
        other => quote!(#other).to_string(),
    }
}

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
///
/// Scalar spellings must be bare (`u32`, not `foo::u32`): casts reject
/// qualified targets too (`cast_target_ty`), so signatures match them.
/// Vector, matrix and newtype spellings stay last-segment (qualified
/// `glam::Vec3` keeps working).
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
                        type_string(ty),
                    ));
                }
                // Scalars keep their values only when bare (`u32`, not
                // `foo::u32`); vectors, matrices and newtypes stay
                // last-segment so qualified `glam::Vec3` keeps working.
                let bare = tp.qself.is_none() && tp.path.segments.len() == 1;
                let scalar = bare && matches!(name.as_str(), "f32" | "u32" | "i32" | "bool");
                if !(scalar || is_kernel_vector_ty(name.as_str())) {
                    return Some(format!(
                        "{what} `{}` is not supported in GPU kernel `{fn_name}`; supported: f32, u32, i32, bool and glam vectors/matrices",
                        type_string(ty),
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
                    type_string(ty),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{is_scalar_namespaced, kernel_type_error, path_string, type_string};
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
            "foo::Vec3",
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
        for ty in [
            "u8",
            "i16",
            "u64",
            "usize",
            "f64",
            "foo::u32",
            "foo::bool",
            "[foo::u32; 4]",
            "MyStruct",
            "&f32",
        ] {
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

    /// A scalar hidden below `core`/`std` is namespaced too: every path
    /// segment but the last is checked, not just the first.
    #[test]
    fn namespaced_scalar_paths_hide_below_core_std() {
        for path in [
            "core::u8::MAX",
            "std::primitive::u8::MAX",
            "std::f32::consts::PI",
        ] {
            let expr: syn::Expr = syn::parse_str(path).expect("test path must parse");
            let syn::Expr::Path(expr_path) = expr else {
                panic!("{path} must parse as a path");
            };
            assert!(
                is_scalar_namespaced(&expr_path.path, expr_path.qself.as_ref()),
                "{path}"
            );
        }
    }

    /// Error messages spell paths the way Rust does — `u8::MAX`, not
    /// the `u8 :: MAX` token stream [`quote!`] prints.
    #[test]
    fn error_spellings_have_no_token_stream_spaces() {
        for (ty, expected) in [
            ("foo::u32", "foo::u32"),
            ("[foo::u32; 4]", "[foo::u32; 4]"),
            ("Vec<u8>", "Vec<u8>"),
            ("HashMap<u32, f32>", "HashMap<u32, f32>"),
            ("&mut f32", "&mut f32"),
        ] {
            let parsed: Type = syn::parse_str(ty).expect("test type must parse");
            assert_eq!(type_string(&parsed), expected, "{ty}");
        }
        for (path, expected) in [("u8::MAX", "u8::MAX"), ("<u8>::MAX", "<u8>::MAX")] {
            let expr: syn::Expr = syn::parse_str(path).expect("test path must parse");
            let syn::Expr::Path(expr_path) = expr else {
                panic!("{path} must parse as a path");
            };
            assert_eq!(
                path_string(&expr_path.path, expr_path.qself.as_ref()),
                expected,
                "{path}"
            );
        }
    }
}
