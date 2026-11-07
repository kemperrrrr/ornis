//! Stable shader entry attribute for the llvm2gpu path.
//!
//! [`gpu_entry`] keeps the annotated function and emits a `#[used]`
//! metadata record in a known LLVM global, so the offline builder reads
//! the entry name, workgroup size, and uniform field tags from the shader
//! crate itself instead of CLI flags.

#![warn(missing_docs)]

use proc_macro::TokenStream;
use quote::quote;
use syn::Expr;
use syn::Lit;
use syn::parse::Parser;
use syn::punctuated::Punctuated;

/// Declare a stable GPU entry point.
///
/// Syntax follows rust-gpu's `#[spirv(compute(threads(N)))]` shape:
///
/// ```ignore
/// #[gpu_entry(compute(threads(64)), uniform_tags = [6, 6, 6, 6, 6, 6, 2])]
/// #[unsafe(no_mangle)]
/// pub extern "C" fn cs_main(id: u32) { /* ... */ }
/// ```
///
/// `uniform_tags` is optional and lists the `ornis_gpu.types` field tags
/// of the uniform struct (one struct id, order preserved). The metadata
/// global is named `ornis_gpu_shader_meta` and carries a compact ASCII
/// record: `entry=...;workgroup=x,y,z;types=...`.
#[proc_macro_attribute]
pub fn gpu_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn path_ident(expr: &Expr) -> Option<&syn::Ident> {
    match expr {
        Expr::Path(path) => path.path.get_ident(),
        _ => None,
    }
}

fn expand(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let function: syn::ItemFn = syn::parse2(item)?;
    let name = function.sig.ident.to_string();
    let parser = Punctuated::<Expr, syn::Token![,]>::parse_terminated;
    let exprs = parser.parse2(attr)?;
    let mut threads = None;
    let mut tags = Vec::new();
    for expr in exprs {
        match expr {
            Expr::Call(call) => {
                if path_ident(&call.func).is_none_or(|ident| ident != "compute") {
                    return Err(syn::Error::new_spanned(
                        call.func,
                        "expected `compute(threads(N))`",
                    ));
                }
                let Some(Expr::Call(inner)) = call.args.first() else {
                    return Err(syn::Error::new_spanned(call.args, "expected `threads(N)`"));
                };
                if path_ident(&inner.func).is_none_or(|ident| ident != "threads") {
                    return Err(syn::Error::new_spanned(
                        &inner.func,
                        "expected `threads(N)`",
                    ));
                }
                let Some(Expr::Lit(lit)) = inner.args.first() else {
                    return Err(syn::Error::new_spanned(
                        &inner.args,
                        "threads size must be an integer",
                    ));
                };
                let Lit::Int(value) = &lit.lit else {
                    return Err(syn::Error::new_spanned(
                        lit,
                        "threads size must be an integer",
                    ));
                };
                threads = Some(value.base10_parse::<u32>()?);
            }
            Expr::Assign(assign) => {
                if path_ident(&assign.left).is_none_or(|ident| ident != "uniform_tags") {
                    return Err(syn::Error::new_spanned(
                        assign.left,
                        "expected `uniform_tags = [..]`",
                    ));
                }
                let Expr::Array(array) = *assign.right else {
                    return Err(syn::Error::new_spanned(
                        assign.right,
                        "expected `uniform_tags = [..]`",
                    ));
                };
                for element in array.elems {
                    let Expr::Lit(lit) = element else {
                        return Err(syn::Error::new_spanned(
                            element,
                            "uniform tag must be an integer",
                        ));
                    };
                    let Lit::Int(value) = lit.lit else {
                        return Err(syn::Error::new_spanned(
                            lit,
                            "uniform tag must be an integer",
                        ));
                    };
                    tags.push(value.base10_parse::<u8>()?);
                }
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "unsupported gpu_entry argument",
                ));
            }
        }
    }
    let threads = threads
        .ok_or_else(|| syn::Error::new_spanned(&function.sig, "missing compute(threads(N))"))?;
    let mut meta = format!("entry={name};workgroup={threads},1,1");
    if !tags.is_empty() {
        let tags = tags.iter().map(u8::to_string).collect::<Vec<_>>().join(",");
        meta.push_str(&format!(";types={tags}"));
    }
    let len = meta.len();
    let bytes = syn::LitByteStr::new(meta.as_bytes(), proc_macro2::Span::call_site());
    Ok(quote! {
        #function

        #[used]
        #[cfg_attr(
            target_arch = "wasm32",
            unsafe(link_section = "ornis_gpu.shader_meta")
        )]
        #[unsafe(export_name = "ornis_gpu_shader_meta")]
        static ORNIS_GPU_SHADER_META: [u8; #len] = *#bytes;
    })
}
