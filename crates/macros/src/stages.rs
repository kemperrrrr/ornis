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
//!   way — `#[wgsl(global = "camera")] camera: ...`: the parameter leaves the
//!   WGSL signature, every use is renamed to the global, and the names are
//!   re-exported via `globals()` so tests can pin them against the assembled
//!   shader. Several globals share one bundle instead:
//!   `ctx: Context<LightingContext>` lowers every `ctx.field` to the
//!   global `field` (field name === global name; the bundle struct plus
//!   `#[derive(ShaderContext)]` is the contract). The `Context<...>`
//!   wrapper — not an attribute — is what marks a bundle: a bare struct
//!   param (`input: VertexInput`) stays a real WGSL parameter. Builtin
//!   indices need no attribute either (`vertex_index: VertexIndex`).
//!   Scalar and glam types map
//!   as in [`crate::wgsl`]; any
//!   other named type passes through verbatim (varying structs).
//! - bodies require an explicit `return` (a tail value would also translate,
//!   but entries must not look like value functions).

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::Parser;
use syn::{parse_macro_input, punctuated::Punctuated, token::Comma};

/// `#[stage]` attribute arguments. `entry` is an optional override; without
/// it the WGSL entry name is the Rust function ident (`fn vs_main` →
/// `fn vs_main`) — one identifier, no second spelling.
struct StageArgs {
    stage: String,
    entry: Option<String>,
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
        (Some(stage), entry) => Ok(StageArgs {
            stage,
            entry,
            returns,
        }),
        _ => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "stage: `vertex`|`fragment` is required (`entry = \"...\"` optionally overrides the Rust fn name)",
        )),
    }
}

/// Context-bundle wrapper (`ctx: Context<LightingContext>`): the parameter
/// is a resource bundle, not a WGSL function parameter. Detected locally —
/// the last path segment is `Context` with exactly one generic argument —
/// so no cross-item reading is needed and signatures stay attribute-free.
/// A bare struct param (`input: VertexInput`) is still a real WGSL
/// parameter; the wrapper carries the one disambiguating bit.
fn context_wrapper(ty: &syn::Type) -> syn::Result<bool> {
    if let syn::Type::Path(path) = ty
        && path.qself.is_none()
        && let Some(seg) = path.path.segments.last()
        && seg.ident == "Context"
    {
        match &seg.arguments {
            syn::PathArguments::AngleBracketed(args) if args.args.len() == 1 => Ok(true),
            _ => Err(syn::Error::new_spanned(
                ty,
                "stage: `Context` bundle takes exactly one type argument (`ctx: Context<Bundle>`)",
            )),
        }
    } else {
        Ok(false)
    }
}

/// Located-value return (`-> Location<0, glam::Vec4>` → `-> @location(0)
/// vec4<f32>`): a bare Rust return type cannot spell a WGSL location
/// attribute, so the number and the real inner type travel in the wrapper.
/// Matches the last path segment; combining it with an explicit
/// `returns = "…"` is two sources of truth.
fn location_return(ty: &syn::Type) -> syn::Result<Option<String>> {
    let syn::Type::Path(path) = ty else {
        return Ok(None);
    };
    if path.qself.is_some() {
        return Ok(None);
    }
    let Some(seg) = path.path.segments.last() else {
        return Ok(None);
    };
    if seg.ident != "Location" {
        return Ok(None);
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return Err(syn::Error::new_spanned(
            ty,
            "stage: `Location` return takes a number and a type (`-> Location<0, glam::Vec4>`)",
        ));
    };
    let mut args = args.args.iter();
    let (Some(first), Some(second), None) = (args.next(), args.next(), args.next()) else {
        return Err(syn::Error::new_spanned(
            ty,
            "stage: `Location` return takes a number and a type (`-> Location<0, glam::Vec4>`)",
        ));
    };
    let syn::GenericArgument::Const(syn::Expr::Lit(lit)) = first else {
        return Err(syn::Error::new_spanned(
            first,
            "stage: `Location` number must be an integer literal (`Location<0, …>`)",
        ));
    };
    let syn::Lit::Int(n) = &lit.lit else {
        return Err(syn::Error::new_spanned(
            first,
            "stage: `Location` number must be an integer literal (`Location<0, …>`)",
        ));
    };
    let n: u32 = n
        .base10_parse()
        .map_err(|_| syn::Error::new_spanned(first, "stage: `Location` number must fit in u32"))?;
    let syn::GenericArgument::Type(inner) = second else {
        return Err(syn::Error::new_spanned(
            second,
            "stage: `Location` inner must be a type (`Location<0, glam::Vec4>`)",
        ));
    };
    Ok(Some(format!("@location({n}) {}", named_type(inner))))
}

/// Builtin-index newtypes (`VertexIndex` → `@builtin(vertex_index)`,
/// `InstanceIndex` → `@builtin(instance_index)`): a bare `u32` cannot say
/// which index it is, so the meaning travels in the type and no attribute
/// is needed. Matches the last path segment, so `super::VertexIndex`
/// works; combining a newtype with any `#[wgsl(...)]` option is an error.
fn builtin_newtype(ty: &syn::Type) -> Option<&'static str> {
    if let syn::Type::Path(path) = ty
        && path.qself.is_none()
        && let Some(seg) = path.path.segments.last()
    {
        return match seg.ident.to_string().as_str() {
            "VertexIndex" => Some("vertex_index"),
            "InstanceIndex" => Some("instance_index"),
            _ => None,
        };
    }
    None
}

/// `#[wgsl(global = "NAME")]` on an entry parameter: the parameter is not a
/// WGSL function parameter but a module-global resource (`camera`, …).
/// Returns the global name when present.
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

/// Strip a context-bundle prefix: `ctx.field` → `field` for every bundle
/// ident in `contexts`. The bundle struct (via `#[derive(ShaderContext)]`)
/// guarantees `field` names a WGSL global, so the lowered body spells the
/// global directly. Single level only — deeper paths (`ctx.a.b` lower to
/// `a.b`, which must itself resolve) fail loudly in naga if misused; a
/// bare `ctx` (no field) is left alone and fails the same way.
struct ContextStripper<'a> {
    contexts: &'a std::collections::HashSet<String>,
}

impl<'a> syn::visit_mut::VisitMut for ContextStripper<'a> {
    fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
        if let syn::Expr::Field(field) = expr
            && let syn::Expr::Path(base) = field.base.as_ref()
            && base.qself.is_none()
            && base.path.segments.len() == 1
            && let syn::Member::Named(member) = &field.member
            && self
                .contexts
                .contains(&base.path.segments[0].ident.to_string())
        {
            *expr = syn::Expr::Path(syn::ExprPath {
                attrs: Vec::new(),
                qself: None,
                path: member.clone().into(),
            });
            return;
        }
        syn::visit_mut::visit_expr_mut(self, expr);
    }
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
    // Without an explicit override the WGSL entry name is the Rust fn name.
    let entry = entry.unwrap_or_else(|| fn_name.to_string());

    // Return type: Rust ident verbatim (renamed mirrors are spelled through
    // import aliases at the use site), a `Location<N, T>` wrapper for
    // located value returns, or the explicit WGSL override for legacy
    // string `returns`.
    let wgsl_ret = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => match location_return(ty) {
            Ok(Some(located)) => {
                if returns.is_some() {
                    return syn::Error::new_spanned(
                        ty,
                        "stage: `Location<…>` return already carries the location; drop `returns = \"…\"`",
                    )
                    .to_compile_error()
                    .into();
                }
                located
            }
            Ok(None) => {
                let rust = named_type(ty);
                returns.unwrap_or(rust)
            }
            Err(e) => return e.to_compile_error().into(),
        },
        syn::ReturnType::Default => "void".to_string(),
    };

    let mut params = Vec::new();
    // Global resource params (`#[wgsl(global = "camera")] camera: ...`) and
    // context bundles (`#[wgsl(context)] ctx: LightingContext`): excluded
    // from the WGSL signature, lowered in the body (rename / `ctx.` strip),
    // and re-exported via `globals()` so tests can pin the contract.
    let mut renames = std::collections::HashMap::new();
    let mut contexts = std::collections::HashSet::new();
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
        // Context bundles (`ctx: Context<Bundle>`) and builtin-index
        // newtypes carry their meaning in the type: no attribute. A bundle
        // combined with any `#[wgsl(...)]` option is two sources of truth.
        if match context_wrapper(&pat_ty.ty) {
            Ok(c) => c,
            Err(e) => return e.to_compile_error().into(),
        } {
            if all_attrs.iter().any(|a| a.path().is_ident("wgsl")) {
                return syn::Error::new_spanned(
                    &pat_ty.ty,
                    "stage: `Context<Bundle>` is already a bundle; drop the attribute",
                )
                .to_compile_error()
                .into();
            }
            contexts.insert(pi.ident.to_string());
            continue;
        }
        // Builtin-index newtypes carry their meaning in the type: no
        // attribute, and combining one with any `#[wgsl(...)]` option is
        // two sources of truth.
        if let Some(builtin) = builtin_newtype(&pat_ty.ty) {
            if all_attrs.iter().any(|a| a.path().is_ident("wgsl")) {
                return syn::Error::new_spanned(
                    &pat_ty.ty,
                    format!(
                        "stage: type `{builtin}` already implies `@builtin({builtin})`; drop the attribute"
                    ),
                )
                .to_compile_error()
                .into();
            }
            params.push(format!("@builtin({builtin}) {}: u32", pi.ident));
            continue;
        }
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
    use syn::visit_mut::VisitMut;
    if !contexts.is_empty() {
        ContextStripper {
            contexts: &contexts,
        }
        .visit_block_mut(&mut body_func.block);
    }
    if !renames.is_empty() {
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
