//! Rust AST → WGSL source translation for `#[kernel]` functions.
//!
//! Two phases with a structural boundary between them: `syn` lowers to
//! IR ([`wgsl_lower`], no WGSL text), then the shared
//! `ornis-shader-lang` [`writer`](ornis_shader_lang::writer) renders IR
//! to WGSL. [`WgslGen`] is the thin façade over both; unsupported
//! constructs lower to loud markers naga rejects. Rust casts are
//! dropped — WGSL infers types from context.

use syn::{FnArg, ItemFn};

#[allow(dead_code)]
struct WgslGen;

impl WgslGen {
    fn expr(e: &syn::Expr) -> String {
        ornis_shader_lang::writer::print_expr(&crate::wgsl_lower::lower_expr(e))
    }
}

#[allow(dead_code)]
pub fn rust_to_wgsl(expr: &syn::Expr) -> String {
    WgslGen::expr(expr)
}

/// WGSL type for an uninitialized `var` declaration. Scalars and glam
/// vectors map to their WGSL spellings; any other named type passes
/// through verbatim (varying/uniform struct names).
pub(crate) fn var_type(ty: &syn::Type) -> String {
    if let syn::Type::Path(tp) = ty
        && let Some(last) = tp.path.segments.last().map(|s| s.ident.to_string())
    {
        // Bare `bool` stays verbatim (not a shader primitive); the rest
        // resolves through the registry, mirrors fall back to verbatim.
        if last == "bool" {
            return last;
        }
        if let Some(mapped) = ornis_shader_lang::ShaderType::from_rust(&last) {
            return mapped.wgsl().to_string();
        }
        return last;
    }
    rust_type_to_wgsl(ty)
}

/// Map a known scalar/glam type name to WGSL; `None` for anything else
/// (interface/layout mirrors, whose Rust names match WGSL by convention).
pub fn glam_type_to_wgsl(name: &str) -> Option<String> {
    // Bare `bool` keeps its previous passthrough (not a shader primitive).
    if name == "bool" {
        return Some(name.to_string());
    }
    wgsl_type(name)
}

fn wgsl_type(ty: &str) -> Option<String> {
    ornis_shader_lang::ShaderType::from_rust(ty).map(|t| t.wgsl().to_string())
}

pub fn rust_type_to_wgsl(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(tp) => {
            let segs: Vec<_> = tp
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            let last = segs.last().map(|s| s.as_str());
            match last {
                // Bare `bool` keeps its previous spelling (not a primitive).
                Some("bool") => "bool".to_string(),
                _ => {
                    let name = last.unwrap_or("f32");
                    ornis_shader_lang::ShaderType::from_rust(name)
                        .map(|t| t.wgsl().to_string())
                        .unwrap_or_else(|| "f32".to_string())
                }
            }
        }
        _ => "f32".to_string(),
    }
}

fn param_name(pat: &syn::Pat) -> String {
    match pat {
        syn::Pat::Ident(i) => i.ident.to_string(),
        _ => "arg".to_string(),
    }
}

fn extract_and_convert_body(func: &ItemFn) -> String {
    let stmts = &func.block.stmts;
    if let Some(syn::Stmt::Expr(expr, _)) = stmts.last() {
        return rust_to_wgsl(expr);
    }
    "0.0".to_string()
}

/// Convert the entire function body to WGSL statements
fn convert_body_to_wgsl(func: &ItemFn) -> String {
    ornis_shader_lang::writer::print_block(&crate::wgsl_lower::lower_body(&func.block.stmts))
}

/// Generate a standalone WGSL function definition (not a compute shader).
/// Returns the function body as WGSL.
// Helper wrapper, used by this module's unit tests.
#[allow(dead_code)]
pub fn wgsl_body_from_fn(func: &ItemFn) -> String {
    convert_body_to_wgsl(func)
}

/// Generate just the statement list of a function body as WGSL. Used by
/// `#[gpu_pipeline]` in full-shader mode, where the body is embedded into a
/// generated `fn main(...) { ... }` compute entry point.
pub fn wgsl_main_body(func: &ItemFn) -> String {
    convert_body_to_wgsl(func)
}

/// Generate the full WGSL function signature + body as a string.
pub fn wgsl_fn_source(func: &ItemFn) -> String {
    let fn_name = func.sig.ident.to_string();
    let params: Vec<_> = func
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_ty) = arg {
                Some((pat_ty.pat.as_ref(), pat_ty.ty.as_ref()))
            } else {
                None
            }
        })
        .collect();

    let return_ty = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => rust_type_to_wgsl(ty),
        syn::ReturnType::Default => "void".to_string(),
    };

    let param_wgsl: Vec<String> = params
        .iter()
        .map(|(pat, ty)| format!("{}: {}", param_name(pat), rust_type_to_wgsl(ty)))
        .collect();

    let body = convert_body_to_wgsl(func);

    format!(
        "fn {}({}) -> {} {{ {} }}",
        fn_name,
        param_wgsl.join(", "),
        return_ty,
        body
    )
}

pub fn wgsl_source_from_fn(func: &ItemFn) -> String {
    let params: Vec<_> = func
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_ty) = arg {
                Some((pat_ty.pat.as_ref(), pat_ty.ty.as_ref()))
            } else {
                None
            }
        })
        .collect();

    let return_ty = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => rust_type_to_wgsl(ty),
        syn::ReturnType::Default => "f32".to_string(),
    };

    let param_names: Vec<String> = params.iter().map(|(pat, _)| param_name(pat)).collect();
    let param_types: Vec<String> = params.iter().map(|(_, ty)| rust_type_to_wgsl(ty)).collect();
    let body_wgsl = extract_and_convert_body(func);

    let mut wgsl = String::new();
    for (i, (pname, ptype)) in param_names.iter().zip(param_types.iter()).enumerate() {
        wgsl.push_str(&format!(
            "@group(0) @binding({i}) var<storage, read> {pname}: array<{ptype}>;\n"
        ));
    }
    let output_idx = param_names.len();
    wgsl.push_str(&format!(
        "@group(0) @binding({output_idx}) var<storage, read_write> output: array<{return_ty}>;\n\n"
    ));
    wgsl.push_str(
        "@compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n\
         let i = id.x;\n\
         output[i] = ",
    );
    wgsl.push_str(&body_wgsl);
    wgsl.push_str(";\n}\n");
    wgsl
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn simple_arithmetic() {
        let expr: syn::Expr = parse_quote!(a + b * c);
        assert_eq!(rust_to_wgsl(&expr), "a + b * c");
    }

    #[test]
    fn parens() {
        let expr: syn::Expr = parse_quote!((a + b) * c);
        assert_eq!(rust_to_wgsl(&expr), "(a + b) * c");
    }

    #[test]
    fn function_call() {
        let expr: syn::Expr = parse_quote!(sin(x));
        assert_eq!(rust_to_wgsl(&expr), "sin(x)");
    }

    #[test]
    fn vec3_construction() {
        let expr: syn::Expr = parse_quote!(glam::Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(rust_to_wgsl(&expr), "vec3<f32>(1.0, 2.0, 3.0)");
    }

    #[test]
    fn struct_literal_is_positional_constructor() {
        let expr: syn::Expr = parse_quote!(VertexOutput {
            clip_position: QUAD[idx],
            uv: UVS[idx]
        });
        assert_eq!(
            rust_to_wgsl(&expr),
            "VertexOutput(QUAD[idx], UVS[idx]) /* clip_position, uv */"
        );
    }

    #[test]
    fn for_range_loop_shape() {
        let func: syn::ItemFn = parse_quote!(
            fn f() {
                for i in 0u..n {
                    continue;
                }
            }
        );
        let body = wgsl_main_body(&func);
        assert!(
            body.contains("for (var i: u32 = 0; i < n; i = i + 1)"),
            "{body}"
        );
        assert!(body.contains("continue;"), "{body}");
    }

    #[test]
    fn unsupported_statement_is_loud() {
        let func: syn::ItemFn = parse_quote!(
            fn f() {
                for x in items {}
            }
        );
        let body = wgsl_main_body(&func);
        assert!(
            body.contains("__wgsl_dsl_unsupported_statement__"),
            "{body}"
        );
    }

    #[test]
    fn fields_access_path() {
        let expr: syn::Expr = parse_quote!(pos.x);
        assert_eq!(rust_to_wgsl(&expr), "pos.x");
    }

    #[test]
    fn method_call() {
        let expr: syn::Expr = parse_quote!(a.dot(b));
        assert_eq!(rust_to_wgsl(&expr), "dot(a, b)");
    }

    #[test]
    fn nested_math() {
        let expr: syn::Expr = parse_quote!(dot(normalize(a), cross(b, c)));
        assert_eq!(rust_to_wgsl(&expr), "dot(normalize(a), cross(b, c))");
    }

    #[test]
    fn field_chain() {
        let expr: syn::Expr = parse_quote!(a.b.c);
        assert_eq!(rust_to_wgsl(&expr), "a.b.c");
    }

    #[test]
    fn powi_2_expands_to_multiplication() {
        let expr: syn::Expr = parse_quote!(x.powi(2));
        assert_eq!(rust_to_wgsl(&expr), "x * x");
    }

    #[test]
    fn powi_3_expands_to_multiplication() {
        let expr: syn::Expr = parse_quote!(x.powi(3));
        assert_eq!(rust_to_wgsl(&expr), "x * x * x");
    }

    #[test]
    fn signum_maps_to_sign() {
        let expr: syn::Expr = parse_quote!(x.signum());
        assert_eq!(rust_to_wgsl(&expr), "sign(x)");
    }

    #[test]
    fn if_expr_translates() {
        let expr: syn::Expr = parse_quote!(if a > b { c } else { d });
        // WGSL requires return in blocks used as expressions
        assert_eq!(
            rust_to_wgsl(&expr),
            "if (a > b) { return c; } else { return d; }"
        );
    }

    #[test]
    fn let_stmt_translates() {
        let func: ItemFn = parse_quote! {
            fn test(a: f32) -> f32 {
                let x = a + 1.0;
                return x;
            }
        };
        let body = wgsl_body_from_fn(&func);
        assert!(body.contains("let x = a + 1.0;"));
        assert!(body.contains("return x;"));
    }

    #[test]
    fn fn_source_generates_wgsl_function() {
        let func: ItemFn = parse_quote! {
            fn ggx_distribution(NoH: f32, alpha: f32) -> f32 {
                let alpha2 = alpha * alpha;
                let denom = 3.14159 * (NoH * NoH * (alpha2 - 1.0) + 1.0) * (NoH * NoH * (alpha2 - 1.0) + 1.0);
                return alpha2 / denom;
            }
        };
        let source = wgsl_fn_source(&func);
        assert!(source.starts_with("fn ggx_distribution"));
        assert!(source.contains("NoH: f32"));
        assert!(source.contains("alpha: f32"));
        assert!(source.contains("-> f32"));
        assert!(source.contains("let alpha2 = alpha * alpha;"));
        assert!(source.contains("return alpha2 / denom;"));
    }

    #[test]
    fn if_else_in_fn_body() {
        let func: ItemFn = parse_quote! {
            fn test(x: f32) -> f32 {
                let y = if x > 0.0 { x } else { -x };
                return y;
            }
        };
        let source = wgsl_fn_source(&func);
        // let y = if/else → var y = else_val; if (cond) { y = if_val; }
        assert!(source.contains("var y = -x;"));
        assert!(source.contains("if (x > 0.0)"));
        assert!(source.contains("y = x;"));
        assert!(source.contains("return y;"));
    }

    #[test]
    fn vec_type_path() {
        let expr: syn::Expr = parse_quote!(glam::Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(rust_to_wgsl(&expr), "vec3<f32>(1.0, 2.0, 3.0)");
    }

    #[test]
    fn pi_constant_passes_through() {
        let expr: syn::Expr = parse_quote!(PI);
        assert_eq!(rust_to_wgsl(&expr), "PI");
    }

    #[test]
    fn index_expr_translates() {
        let expr: syn::Expr = parse_quote!(buf[i]);
        assert_eq!(rust_to_wgsl(&expr), "buf[i]");
    }

    #[test]
    fn nested_index_field_translates() {
        let expr: syn::Expr = parse_quote!(batch_buf[gid.x].acc[l]);
        assert_eq!(rust_to_wgsl(&expr), "batch_buf[gid.x].acc[l]");
    }

    #[test]
    fn cast_is_dropped() {
        let expr: syn::Expr = parse_quote!(x as u32);
        assert_eq!(rust_to_wgsl(&expr), "x");
    }

    #[test]
    fn assign_op_translates() {
        let expr: syn::Expr = parse_quote!(ba.velocity -= delta * n);
        assert_eq!(rust_to_wgsl(&expr), "ba.velocity -= delta * n");
    }

    #[test]
    fn logical_and_or_translate_to_short_circuit() {
        let and: syn::Expr = parse_quote!(a > 0 && b < 1);
        assert_eq!(rust_to_wgsl(&and), "a > 0 && b < 1");
        let or: syn::Expr = parse_quote!(a > 0 || b < 1);
        assert_eq!(rust_to_wgsl(&or), "a > 0 || b < 1");
    }

    #[test]
    fn int_suffix_maps_to_wgsl() {
        let u: syn::Expr = parse_quote!(0u32);
        assert_eq!(rust_to_wgsl(&u), "0u");
        let i: syn::Expr = parse_quote!(7i32);
        assert_eq!(rust_to_wgsl(&i), "7i");
        let plain: syn::Expr = parse_quote!(42);
        assert_eq!(rust_to_wgsl(&plain), "42");
    }

    #[test]
    fn mut_local_becomes_var() {
        let func: ItemFn = parse_quote! {
            fn test() {
                let mut x = 1.0;
                x += 2.0;
            }
        };
        let body = wgsl_body_from_fn(&func);
        assert!(body.contains("var x = 1.0;"));
        assert!(body.contains("x += 2.0;"));
    }

    #[test]
    fn early_return_in_if_block() {
        let func: ItemFn = parse_quote! {
            fn test(l: u32, count: u32) {
                if l >= count { return; }
                let y = 1.0;
                y
            }
        };
        let body = wgsl_body_from_fn(&func);
        assert!(body.contains("if (l >= count) { return; }"));
        assert!(body.contains("let y = 1.0;"));
    }

    #[test]
    fn length_sq_maps_to_dot() {
        let expr: syn::Expr = parse_quote!(length_sq(v));
        assert_eq!(rust_to_wgsl(&expr), "dot(v, v)");
    }
}
