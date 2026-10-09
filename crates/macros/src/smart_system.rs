//! Implementation of the `#[smart_system]` attribute macro (PLAN m, path 3).
//!
//! A `#[smart_system]` function is an ordinary free function taking
//! `&Resources` plus plain parameters:
//!
//! ```ignore
//! #[smart_system]
//! fn player_input(resources: &Resources, speed: f32) {
//!     let store = resources.get::<SmartStore>().expect("store");
//!     let mut vel = store.write_lane::<Velocity>().expect("vel");
//!     // ...
//! }
//! ```
//!
//! The macro emits, next to the (loop-rewritten, [`smart_pipeline`]-style)
//! function:
//!
//! - `struct PlayerInputSystem { speed: f32 }` — one field per non-`Resources`
//!   parameter, in order (a `&T` parameter becomes an owned `T` field passed
//!   as `&self.field`, so no per-frame clone; other parameters are stored
//!   as-is and passed via `.clone()`, hence must be `Clone`);
//! - `impl ornis_core::System` — `name()` (from `name = "..."`, defaulting to
//!   the function name), `access()` derived from the body
//!   (`resources.get::<T>()` → `reads::<T>()`, lane bindings →
//!   `reads_lane`/`writes_lane`, a write covering a read), and `run()`
//!   forwarding to the function with parameters in declaration order.
//!
//! Because `access()` is generated from the same bindings the body uses, the
//! declaration cannot desync from the implementation. What cannot be derived
//! is rejected or declared explicitly:
//!
//! - `#[smart_system(reads(A, B), writes(C))]` adds explicit entries
//!   (a `writes` entry suppresses the same type in `reads`);
//! - `reads_lane(P)` / `writes_lane(Q)` declare lane accesses the body hides
//!   (e.g. a store passed to a helper under `opaque`);
//! - a resource type with interior mutability (`Mutex`, `RwLock`, `Atomic*`,
//!   `Cell`, `RefCell`, `OnceLock`, including wrapped forms like
//!   `Arc<Mutex<T>>`) must be listed explicitly, otherwise the
//!   macro errors — a silent `reads` would mislead the level scheduler;
//! - `resources.get()` without turbofish, any other use of the `resources`
//!   ident (passing it to a helper, aliasing it, capturing it), any
//!   non-lane use of a store ident, and `resources`/store idents inside
//!   opaque macro invocations are compile errors unless `opaque` is set, in
//!   which case the author lists every touched resource and lane explicitly.
//!
//! The analysis is syntactic, not type-directed: newtypes and type aliases
//! around interior-mutability primitives are invisible to the macro and must
//! be declared explicitly, as must accesses hidden inside unknown macros.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    FnArg, Ident, ItemFn, Pat, PatType, ReturnType, Type, parse_macro_input, visit::Visit,
    visit_mut::VisitMut,
};

use crate::smart_pipeline::{
    LaneCollector, LoopRewriter, ParamKind, classify_param, combine_errors_tokens,
    send_sync_assert_tokens, warning_tokens, wrap_body,
};
use crate::smart_system_checks::{SmartSystemArgs, check_derivable};
use crate::smart_system_emit::{Emission, access_tokens, system_struct_ident};
use syn::ext::IdentExt;

/// A non-`Resources` parameter: owned field type plus how `run()` passes it.
pub(crate) struct FieldParam {
    pub(crate) name: Ident,
    /// Type stored in the struct.
    pub(crate) field_ty: Type,
    /// Tokens passed at the call site (`&self.x` or `self.x.clone()`).
    pub(crate) pass: TokenStream2,
}

/// Validates the signature: free function, exactly one `&Resources`, unit
/// return, no generics/async/unsafe. Returns the resources ident.
fn validate_signature(sig: &syn::Signature) -> syn::Result<Ident> {
    if sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            &sig.ident,
            "#[smart_system]: async system functions are not supported",
        ));
    }
    if matches!(sig.safety, syn::Safety::Unsafe(_)) {
        return Err(syn::Error::new_spanned(
            &sig.ident,
            "#[smart_system]: unsafe system functions are not supported",
        ));
    }
    if !sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &sig.ident,
            "#[smart_system]: generic system functions are not supported",
        ));
    }
    if !matches!(sig.output, ReturnType::Default) {
        return Err(syn::Error::new_spanned(
            &sig.ident,
            "#[smart_system]: system functions must return `()`",
        ));
    }
    let mut resources: Option<Ident> = None;
    for arg in &sig.inputs {
        let FnArg::Typed(PatType { pat, ty, .. }) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "#[smart_system]: system functions must be free functions",
            ));
        };
        let Pat::Ident(pat_ident) = &**pat else {
            return Err(syn::Error::new_spanned(
                pat,
                "#[smart_system]: system parameters must be plain identifiers",
            ));
        };
        if classify_param(ty) == ParamKind::Resources {
            if resources.is_some() {
                return Err(syn::Error::new_spanned(
                    pat,
                    "#[smart_system]: expected exactly one `&Resources` parameter",
                ));
            }
            if let Type::Reference(type_ref) = &**ty
                && type_ref.mutability.is_some()
            {
                return Err(syn::Error::new_spanned(
                    ty,
                    "#[smart_system]: `&mut Resources` is not supported — `run` receives `&Resources`",
                ));
            }
            resources = Some(pat_ident.ident.clone());
        }
    }
    resources.ok_or_else(|| {
        syn::Error::new_spanned(
            &sig.ident,
            "#[smart_system]: expected exactly one `&Resources` parameter",
        )
    })
}

/// Builds one struct field: a `&T` parameter becomes an owned `T` field
/// passed as `&self.field` (no per-frame clone); any other parameter is
/// stored as-is and passed via `.clone()`. `&mut` parameters, explicit
/// lifetimes and unsized targets are rejected.
fn field_param(name: Ident, ty: &Type) -> syn::Result<FieldParam> {
    let (pass, field_ty) = match ty {
        Type::Reference(type_ref) => ref_words(type_ref, &name)?,
        _ => (quote! { self.#name.clone() }, ty.clone()),
    };
    Ok(FieldParam {
        pass,
        field_ty,
        name,
    })
}

/// Pass tokens plus stored type for a `&T` parameter: `&self.x` over an
/// owned `T` field, so no per-frame clone happens.
fn ref_words(type_ref: &syn::TypeReference, name: &Ident) -> syn::Result<(TokenStream2, Type)> {
    if type_ref.mutability.is_some() {
        return Err(syn::Error::new_spanned(
            &type_ref.elem,
            "#[smart_system]: `&mut` parameters cannot be stored in the system struct; pass owned values instead",
        ));
    }
    if type_ref.lifetime.is_some() {
        return Err(syn::Error::new_spanned(
            &type_ref.elem,
            "#[smart_system]: explicit lifetimes on parameters are not supported",
        ));
    }
    match &*type_ref.elem {
        Type::Slice(_) => Err(syn::Error::new_spanned(
            &type_ref.elem,
            "#[smart_system]: unsized targets cannot be stored in the system struct",
        )),
        Type::Path(path) if path.path.segments.last().is_some_and(|s| s.ident == "str") => {
            Err(syn::Error::new_spanned(
                &type_ref.elem,
                "#[smart_system]: `&str` parameters cannot be stored in the system struct; use an owned `String`",
            ))
        }
        _ => Ok((quote! { &self.#name }, (*type_ref.elem).clone())),
    }
}

/// Collects struct fields from non-`Resources` parameters, in order.
fn collect_fields(sig: &syn::Signature, resources: &Ident) -> syn::Result<Vec<FieldParam>> {
    let mut fields = Vec::new();
    for arg in &sig.inputs {
        let FnArg::Typed(PatType { pat, ty, .. }) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "#[smart_system]: system functions must be free functions",
            ));
        };
        let Pat::Ident(pat_ident) = &**pat else {
            return Err(syn::Error::new_spanned(
                pat,
                "#[smart_system]: system parameters must be plain identifiers",
            ));
        };
        if pat_ident.ident == *resources {
            continue;
        }
        fields.push(field_param(pat_ident.ident.clone(), ty)?);
    }
    Ok(fields)
}

/// Attribute entry point and bridge-free expansion core. `access()` and
/// struct emission live in [`crate::smart_system_emit`], the derivability
/// gate in [`crate::smart_system_checks`].
pub fn attribute(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as SmartSystemArgs);
    let input = parse_macro_input!(item as ItemFn);
    expand(args, input).into()
}

/// Bridge-free expansion core (unit-testable): everything below takes and
/// returns plain syntax trees, so tests exercise the real pipeline without
/// the procedural-macro bridge.
fn expand(args: SmartSystemArgs, mut input: ItemFn) -> TokenStream2 {
    let resources = match validate_signature(&input.sig) {
        Ok(resources) => resources,
        Err(error) => return error.to_compile_error(),
    };
    let fields = match collect_fields(&input.sig, &resources) {
        Ok(fields) => fields,
        Err(error) => return error.to_compile_error(),
    };

    let mut collector = LaneCollector::new("#[smart_system]");
    collector.visit_item_fn(&input);

    if !collector.errors.is_empty() {
        return combine_errors_tokens(collector.errors);
    }
    let mut store_idents = Vec::new();
    if let Some(store) = &collector.store_param {
        store_idents.push(store.clone());
    }
    store_idents.extend(collector.store_aliases.iter().cloned());
    if let Err(tokens) = check_derivable(&input, &resources, store_idents, &args) {
        return tokens;
    }
    let access = match access_tokens(
        &collector,
        &args.reads,
        &args.writes,
        &args.reads_lane,
        &args.writes_lane,
    ) {
        Ok(access) => access,
        Err(error) => return error.to_compile_error(),
    };

    let mut rewriter = LoopRewriter {
        macro_name: "#[smart_system]",
        lanes: &collector.lanes,
        warnings: Vec::new(),
    };
    rewriter.visit_block_mut(&mut input.block);
    let warnings = warning_tokens(&rewriter.warnings, "#[smart_system]");

    let send_sync_assert = send_sync_assert_tokens(&collector.lane_accesses);
    let struct_ident = system_struct_ident(&input.sig.ident);
    let system_name = args
        .name
        .clone()
        .unwrap_or_else(|| input.sig.ident.unraw().to_string());
    let attrs = &input.attrs;
    let vis = &input.vis;
    let sig = &input.sig;
    let body = wrap_body(&input.block.stmts);
    let generated = Emission {
        struct_ident,
        fn_ident: &input.sig.ident,
        vis,
        system_name,
        sig: &input.sig,
        resources: &resources,
        fields: &fields,
        access,
    }
    .struct_tokens();

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            ornis_core::pipeline_enter();
            #send_sync_assert
            #(#warnings)*
            #body
            ornis_core::pipeline_exit();
            smart_pipeline_result
        }
        #generated
    };
    expanded
}

#[cfg(test)]
mod smart_system_unit_tests {
    use super::*;
    use crate::smart_system_checks::SmartSystemArgs;
    use quote::ToTokens;

    fn parse_item(src: &str) -> ItemFn {
        syn::parse_str(src).expect("fn parses")
    }

    fn resources_ident() -> Ident {
        syn::parse_str("resources").expect("ident")
    }

    #[test]
    fn field_param_owned_and_borrowed() {
        let owned = field_param(
            syn::parse_str("speed").expect("ident"),
            &syn::parse_str("f32").expect("type"),
        )
        .expect("owned ok");
        assert_eq!(owned.pass.to_string(), "self . speed . clone ()");
        let borrowed = field_param(
            syn::parse_str("table").expect("ident"),
            &syn::parse_str("&Vec<u8>").expect("type"),
        )
        .expect("borrowed ok");
        assert_eq!(borrowed.pass.to_string(), "& self . table");
        assert_eq!(
            borrowed.field_ty.to_token_stream().to_string(),
            "Vec < u8 >"
        );
        assert!(
            field_param(
                syn::parse_str::<Ident>("s").expect("ident"),
                &syn::parse_str("&mut Vec<u8>").expect("type"),
            )
            .is_err()
        );
        assert!(
            field_param(
                syn::parse_str::<Ident>("s").expect("ident"),
                &syn::parse_str("&str").expect("type"),
            )
            .is_err()
        );
    }

    #[test]
    fn validate_signature_accepts_and_rejects() {
        let item = parse_item("fn ok(resources: &Resources, speed: f32) {}");
        assert_eq!(
            validate_signature(&item.sig).expect("valid").to_string(),
            "resources"
        );
        for src in [
            "fn no_res(speed: f32) {}",
            "fn two(a: &Resources, b: &Resources) {}",
            "fn ret(resources: &Resources) -> usize { 0 }",
            "async fn asy(resources: &Resources) {}",
            "unsafe fn uns(resources: &Resources) {}",
            "fn gen<T>(resources: &Resources, v: T) {}",
        ] {
            assert!(validate_signature(&parse_item(src).sig).is_err(), "{src}");
        }
    }

    #[test]
    fn collect_fields_skips_resources_in_order() {
        let item = parse_item("fn f(speed: f32, resources: &Resources, n: u32) {}");
        let fields = collect_fields(&item.sig, &resources_ident()).expect("fields");
        let names: Vec<String> = fields.iter().map(|f| f.name.to_string()).collect();
        assert_eq!(names, vec!["speed".to_string(), "n".to_string()]);
    }

    #[test]
    fn attribute_entry_emits_system() {
        let args = SmartSystemArgs {
            name: None,
            reads: Vec::new(),
            writes: Vec::new(),
            reads_lane: Vec::new(),
            writes_lane: Vec::new(),
            opaque: false,
        };
        let item: ItemFn = syn::parse_str(
            "fn direct(resources: &Resources, speed: f32) { let _ = (resources.len(), speed); }",
        )
        .expect("fn parses");
        let rendered = expand(args, item).to_string();
        assert!(rendered.contains("DirectSystem"), "{rendered}");
        assert!(rendered.contains("\"direct\""), "{rendered}");
        assert!(rendered.contains("impl ornis_core :: System"), "{rendered}");
    }
}

#[cfg(test)]
mod ref_words_tests {
    use super::*;
    use quote::ToTokens;

    fn ident(name: &str) -> Ident {
        syn::parse_str(name).expect("ident parses")
    }

    #[test]
    fn ref_words_rejects_and_builds() {
        let (pass, ty) = ref_words(
            &match syn::parse_str::<Type>("&Vec<u8>").expect("type") {
                Type::Reference(type_ref) => type_ref,
                _ => panic!("expected reference"),
            },
            &ident("table"),
        )
        .expect("borrowed ok");
        assert_eq!(pass.to_string(), "& self . table");
        assert_eq!(ty.to_token_stream().to_string(), "Vec < u8 >");
        for src in ["&mut Vec<u8>", "&'a Vec<u8>", "&[u8]", "&str"] {
            let ty: Type = syn::parse_str(src).expect("type parses");
            let Type::Reference(type_ref) = ty else {
                panic!("expected reference");
            };
            assert!(ref_words(&type_ref, &ident("x")).is_err(), "{src}");
        }
    }
}
