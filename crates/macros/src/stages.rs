//! `#[stage]` — translate a stage-entry Rust function to a WGSL entry point.
//!
//! A stage body is written in Rust syntax against the shader's interface
//! mirrors and free binding identifiers, and the macro emits the `@vertex` /
//! `@fragment` entry wrapper with the translated body:
//!
//! ```text
//! #[stage(vertex, entry = "vs_main", returns = "CompositeVertexOutput")]
//! fn composite_vertex_entry(
//!     #[wgsl(builtin = "vertex_index")] idx: u32,
//! ) -> HdrVertexOutput {
//!     return HdrVertexOutput { clip_position: QUAD[idx], uv: UVS[idx] };
//! }
//! ```
//!
//! DSL-only, like [`crate::gpu_pipeline`] full-shader bodies: the input
//! function is replaced by a `pub mod` exposing `wgsl_source()` and is never
//! compiled as Rust, so free binding identifiers (`QUAD`, `per_objects`,
//! …) and mirror-typed locals need no Rust declarations. The trade (same as
//! physics `contact_solver`) is CPU-testability for expressiveness: entry
//! correctness is pinned by stage parity tests, naga validation and the
//! pixel probes, not by evaluating the same code on CPU.
//!
//! Conventions (checked where the macro can see them):
//! - parameter and return types are spelled with their Rust idents; the WGSL
//!   spelling is identical by convention (interface mirrors keep WGSL names).
//!   A renamed mirror (Rust `HdrVertexOutput` → WGSL `CompositeVertexOutput`)
//!   needs `returns = "..."`, which also renames constructor calls to that
//!   type inside the body.
//! - parameter interface attributes ride on `#[wgsl(...)]` (stripped from the
//!   output, never name-resolved): `builtin = "vertex_index"` or
//!   `location = 0`. Scalar and glam types map as in [`crate::wgsl`]; any
//!   other named type passes through verbatim (varying structs).
//! - bodies require an explicit `return` (a tail value would also translate,
//!   but entries must not look like value functions).

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::Parser;
use syn::visit_mut::VisitMut;
use syn::{parse_macro_input, punctuated::Punctuated, token::Comma};

/// `#[stage]` attribute arguments.
struct StageArgs {
    stage: String,
    entry: String,
    returns: Option<String>,
}

fn parse_args(args: TokenStream) -> syn::Result<StageArgs> {
    let items = Punctuated::<syn::Meta, Comma>::parse_terminated.parse2(args.into())?;
    let mut stage = None;
    let mut entry = None;
    let mut returns = None;
    for item in items {
        match item {
            syn::Meta::Path(p) if p.is_ident("vertex") || p.is_ident("fragment") => {
                stage = Some(p.segments.last().expect("segment").ident.to_string());
            }
            syn::Meta::NameValue(nv) if nv.path.is_ident("entry") => {
                if let syn::Expr::Lit(l) = &nv.value
                    && let syn::Lit::Str(s) = &l.lit
                {
                    entry = Some(s.value());
                } else {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "stage: `entry` expects a string literal",
                    ));
                }
            }
            syn::Meta::NameValue(nv) if nv.path.is_ident("returns") => {
                if let syn::Expr::Lit(l) = &nv.value
                    && let syn::Lit::Str(s) = &l.lit
                {
                    returns = Some(s.value());
                } else {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "stage: `returns` expects a string literal",
                    ));
                }
            }
            other => {
                return Err(syn::Error::new_spanned(
                    &other,
                    "stage: expected `vertex`|`fragment`, `entry = \"...\"`, `returns = \"...\"`",
                ));
            }
        }
    }
    match (stage, entry) {
        (Some(stage), Some(entry)) => Ok(StageArgs {
            stage,
            entry,
            returns,
        }),
        _ => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "stage: `vertex`|`fragment` and `entry = \"...\"` are required",
        )),
    }
}

/// Parameter interface prefix from `#[wgsl(builtin = "...")]` /
/// `#[wgsl(location = N)]` (at most one of each; both compose).
fn param_prefix(attrs: &[syn::Attribute]) -> syn::Result<String> {
    let mut builtin = None;
    let mut location = None;
    for attr in attrs {
        if !attr.path().is_ident("wgsl") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("builtin") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                builtin = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("location") {
                let lit: syn::LitInt = meta.value()?.parse()?;
                location = Some(lit.base10_parse::<u32>()?);
                Ok(())
            } else {
                Err(meta.error("stage: parameter option is `builtin = \"...\"` or `location = N`"))
            }
        })?;
    }
    let mut out = String::new();
    if let Some(b) = builtin {
        out.push_str(&format!("@builtin({b}) "));
    }
    if let Some(l) = location {
        out.push_str(&format!("@location({l}) "));
    }
    Ok(out)
}

/// Rename constructor calls of the Rust return type to the WGSL `returns`
/// name: the translator only sees the Rust ident.
struct StructRenamer {
    from: String,
    to: String,
}

impl syn::visit_mut::VisitMut for StructRenamer {
    fn visit_expr_struct_mut(&mut self, s: &mut syn::ExprStruct) {
        if let Some(seg) = s.path.segments.last_mut()
            && seg.ident == self.from
        {
            seg.ident = syn::Ident::new(&self.to, seg.ident.span());
        }
        for field in &mut s.fields {
            self.visit_expr_mut(&mut field.expr);
        }
    }
}

/// WGSL spelling of a parameter/return type: scalars and glam vectors map as
/// in [`crate::wgsl`]; any other named type passes through verbatim
/// (interface mirrors keep WGSL names by convention).
fn named_type(ty: &syn::Type) -> String {
    if let syn::Type::Path(tp) = ty
        && let Some(last) = tp.path.segments.last()
    {
        let name = last.ident.to_string();
        if let Some(mapped) = crate::wgsl::glam_type_to_wgsl(&name) {
            return mapped;
        }
        return name;
    }
    crate::wgsl::rust_type_to_wgsl(ty)
}

pub fn stage(args: TokenStream, input: TokenStream) -> TokenStream {
    let StageArgs {
        stage,
        entry,
        returns,
    } = match parse_args(args) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    let mut func = parse_macro_input!(input as syn::ItemFn);
    let fn_name = func.sig.ident.clone();

    // Return type: Rust ident verbatim, or the explicit WGSL override.
    let rust_ret = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => named_type(ty),
        syn::ReturnType::Default => "void".to_string(),
    };
    let wgsl_ret = returns.clone().unwrap_or_else(|| rust_ret.clone());
    // Constructor calls spell the Rust type; rewrite them to the WGSL name.
    if let Some(to) = returns {
        let mut renamer = StructRenamer { from: rust_ret, to };
        renamer.visit_item_fn_mut(&mut func);
    }

    let mut params = Vec::new();
    for arg in &func.sig.inputs {
        let syn::FnArg::Typed(pat_ty) = arg else {
            return syn::Error::new_spanned(
                arg,
                "stage: `self` receivers are not supported in entries",
            )
            .to_compile_error()
            .into();
        };
        let syn::Pat::Ident(pi) = pat_ty.pat.as_ref() else {
            return syn::Error::new_spanned(
                &pat_ty.pat,
                "stage: entry parameters must be plain `name: Type` bindings",
            )
            .to_compile_error()
            .into();
        };
        // Interface attributes live on the argument (`#[wgsl(...)] idx: u32`);
        // syn may attach them to the `PatType` or the inner `Pat::Ident`.
        let mut all_attrs = pat_ty.attrs.clone();
        all_attrs.extend(pi.attrs.iter().cloned());
        let prefix = match param_prefix(&all_attrs) {
            Ok(p) => p,
            Err(e) => return e.to_compile_error().into(),
        };
        params.push(format!(
            "{}{}: {}",
            prefix,
            pi.ident,
            named_type(&pat_ty.ty)
        ));
    }

    let body = crate::wgsl::wgsl_main_body(&func);
    let entry_wgsl = format!(
        "@{stage}\nfn {entry}({}) -> {wgsl_ret} {{\n{body}\n}}\n",
        params.join(", ")
    );
    let entry_lit = proc_macro2::Literal::string(&entry_wgsl);

    TokenStream::from(quote! {
        #[doc(hidden)]
        pub mod #fn_name {
            /// The translated WGSL entry point (signature + body).
            #[allow(dead_code)]
            pub fn wgsl_source() -> &'static str {
                #entry_lit
            }

            /// Entry-point name this stage was translated for.
            #[allow(dead_code)]
            pub fn entry_point() -> &'static str {
                #entry
            }
        }
    })
}
