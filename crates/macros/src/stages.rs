//! `#[stage]` — translate a stage-entry Rust function to a WGSL entry point.
//!
//! A stage body is written in Rust syntax against the shader's interface
//! mirrors and free binding identifiers, and the macro emits the `@vertex` /
//! `@fragment` entry wrapper with the translated body:
//!
//! ```text
//! use super::interface::HdrVertexOutput as CompositeVertexOutput;
//!
//! #[stage(vertex, entry = "vs_main")]
//! fn composite_vertex_entry(
//!     #[wgsl(builtin = "vertex_index")] idx: u32,
//! ) -> CompositeVertexOutput {
//!     return CompositeVertexOutput { clip_position: QUAD[idx], uv: UVS[idx] };
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
//!   spelling is identical by convention. A renamed mirror (Rust
//!   `HdrVertexOutput` → WGSL `CompositeVertexOutput`) is spelled through a
//!   Rust import alias (`use HdrVertexOutput as CompositeVertexOutput;`) at
//!   the use site — the macro then never needs a name mapping, and the alias
//!   keeps rust-analyzer honest.
//! - `returns = "..."` overrides the full WGSL return specification; it
//!   exists for located value returns (`@location(0) vec4<f32>`), which have
//!   no Rust spelling.
//! - parameter interface attributes ride on `#[wgsl(...)]` (stripped from the
//!   output, never name-resolved): `builtin = "vertex_index"` or
//!   `location = 0`. Module globals the entry reads are declared the same
//!   way — `#[wgsl(global = "QUAD")] quad: ...`: the parameter leaves the
//!   WGSL signature, every use is renamed to the global, and the names are
//!   re-exported via `globals()` so tests can pin them against the assembled
//!   shader. Scalar and glam types map as in [`crate::wgsl`]; any
//!   other named type passes through verbatim (varying structs).
//! - bodies require an explicit `return` (a tail value would also translate,
//!   but entries must not look like value functions).

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::Parser;
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

/// `#[wgsl(global = "NAME")]` on an entry parameter: the parameter is not a
/// WGSL function parameter but a module-global resource (`QUAD`, `camera`,
/// …). Returns the global name when present.
fn param_global(attrs: &[syn::Attribute]) -> syn::Result<Option<String>> {
    let mut global: Option<String> = None;
    let mut saw_interface = false;
    let mut first_wgsl: Option<&syn::Attribute> = None;
    for attr in attrs {
        if !attr.path().is_ident("wgsl") {
            continue;
        }
        first_wgsl = first_wgsl.or(Some(attr));
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("global") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                global = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("builtin") || meta.path.is_ident("location") {
                // Not ours — but the value must still be consumed, or the
                // outer meta parser stalls expecting `,`.
                meta.value()?.parse::<syn::Lit>()?;
                saw_interface = true;
                Ok(())
            } else {
                Err(meta.error("stage: parameter option is `builtin = \"...\"`, `location = N` or `global = \"...\"`"))
            }
        })?;
    }
    if global.is_some()
        && saw_interface
        && let Some(attr) = first_wgsl
    {
        return Err(syn::Error::new_spanned(
            attr,
            "stage: `global` cannot combine with `builtin`/`location` — globals are not function parameters",
        ));
    }
    Ok(global)
}

/// Rename every `from` identifier to `to` in `block` (paths and bindings;
/// field-member names are left alone). Global resource params are spelled
/// in Rust however reads best (`quad`) but must emit the WGSL global name
/// (`QUAD`) at every use.
struct GlobalRenamer<'a> {
    map: &'a std::collections::HashMap<String, String>,
}

impl<'a> syn::visit_mut::VisitMut for GlobalRenamer<'a> {
    fn visit_ident_mut(&mut self, ident: &mut proc_macro2::Ident) {
        if let Some(to) = self.map.get(&ident.to_string()) {
            *ident = proc_macro2::Ident::new(to, ident.span());
        }
    }

    fn visit_member_mut(&mut self, _member: &mut syn::Member) {
        // Struct field names (`a.quad`) are not global references.
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
            } else if meta.path.is_ident("global") {
                // Handled by `param_global`; not an interface prefix — but the
                // value must still be consumed here.
                meta.value()?.parse::<syn::LitStr>()?;
                Ok(())
            } else {
                Err(meta.error("stage: parameter option is `builtin = \"...\"`, `location = N` or `global = \"...\"`"))
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
    let func = parse_macro_input!(input as syn::ItemFn);
    let fn_name = func.sig.ident.clone();

    // Return type: Rust ident verbatim (renamed mirrors are spelled through
    // import aliases at the use site), or the explicit WGSL override for
    // located value returns.
    let wgsl_ret = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => {
            let rust = named_type(ty);
            returns.unwrap_or(rust)
        }
        syn::ReturnType::Default => "void".to_string(),
    };

    let mut params = Vec::new();
    // Global resource params (`#[wgsl(global = "QUAD")] quad: ...`): excluded
    // from the WGSL signature, renamed to the global at every use, and
    // re-exported via `globals()` so tests can pin the contract.
    let mut renames = std::collections::HashMap::new();
    let mut declared_globals = Vec::new();
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
        let global = match param_global(&all_attrs) {
            Ok(g) => g,
            Err(e) => return e.to_compile_error().into(),
        };
        if let Some(name) = global {
            renames.insert(pi.ident.to_string(), name.clone());
            declared_globals.push(name);
            continue;
        }
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

    let mut body_func = func.clone();
    if !renames.is_empty() {
        use syn::visit_mut::VisitMut;
        GlobalRenamer { map: &renames }.visit_block_mut(&mut body_func.block);
    }
    let body = crate::wgsl::wgsl_main_body(&body_func);
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

            /// Module-global resources this entry reads, declared via
            /// `#[wgsl(global = "NAME")]` params. Tests pin each name against
            /// the assembled shader so a typo can never pass silently.
            #[allow(dead_code)]
            pub fn globals() -> &'static [&'static str] {
                &[#(#declared_globals),*]
            }
        }
    })
}

/// Translate a plain WGSL helper function from Rust: `#[wgsl_fn] fn name(
/// params ) -> Ret { body }` is replaced by a `pub mod` exposing
/// `wgsl_source()` with `fn name(params) -> Ret { … }`.
///
/// Same translation as [`stage`](stage), minus the `@vertex`/`@fragment`
/// wrapper and the parameter interface attributes: parameter and return
/// types map through [`named_type`] (glam vectors included, any other named
/// type passes through verbatim — e.g. `mat: OpenPBRMaterial`). The Rust
/// body is never compiled, so helper bodies may name shader-side values
/// freely, exactly like stage entries.
pub fn wgsl_fn(_args: TokenStream, input: TokenStream) -> TokenStream {
    let func = parse_macro_input!(input as syn::ItemFn);
    let fn_name = func.sig.ident.clone();
    let wgsl_ret = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => named_type(ty),
        syn::ReturnType::Default => "void".to_string(),
    };

    let mut params = Vec::new();
    for arg in &func.sig.inputs {
        let syn::FnArg::Typed(pat_ty) = arg else {
            return syn::Error::new_spanned(
                arg,
                "wgsl_fn: `self` receivers are not supported in helpers",
            )
            .to_compile_error()
            .into();
        };
        let syn::Pat::Ident(pi) = pat_ty.pat.as_ref() else {
            return syn::Error::new_spanned(
                &pat_ty.pat,
                "wgsl_fn: helper parameters must be plain `name: Type` bindings",
            )
            .to_compile_error()
            .into();
        };
        params.push(format!("{}: {}", pi.ident, named_type(&pat_ty.ty)));
    }

    let body = crate::wgsl::wgsl_main_body(&func);
    let helper_wgsl = format!(
        "fn {fn_name}({}) -> {wgsl_ret} {{\n{body}\n}}\n",
        params.join(", "),
        fn_name = func.sig.ident,
    );
    let helper_lit = proc_macro2::Literal::string(&helper_wgsl);

    TokenStream::from(quote! {
        #[doc(hidden)]
        pub mod #fn_name {
            /// The translated WGSL helper function (signature + body).
            #[allow(dead_code)]
            pub fn wgsl_source() -> &'static str {
                #helper_lit
            }
        }
    })
}
