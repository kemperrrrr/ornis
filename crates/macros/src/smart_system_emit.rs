//! `access()` and struct emission for `#[smart_system]` (PLAN m, path 3).
//!
//! [`access_tokens`] builds the `access()` body: explicit entries first,
//! then derived ones in canonical order (sorted resource reads with explicit
//! `writes` suppressing the same type in `reads`, then sorted read lanes,
//! then sorted write lanes; a write covers a read). Resource types with
//! interior mutability — including wrapped forms like `Arc<Mutex<T>>` —
//! must be listed explicitly, otherwise the macro errors. [`Emission`]
//! bundles one function's emission context; its [`Emission::struct_tokens`]
//! builds the struct definition, `::new()`, and the `System` impl.

use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::ext::IdentExt;
use syn::{FnArg, Ident, Pat, PatType, Type};

use crate::smart_pipeline::{LaneCollector, access_entries};
use crate::smart_system::FieldParam;

/// `player_input` → `PlayerInputSystem`. Raw identifiers are unrawed first;
/// the struct span comes from the function name.
pub(crate) fn system_struct_ident(fn_ident: &Ident) -> Ident {
    let plain = fn_ident.unraw().to_string();
    let mut camel = String::new();
    for part in plain.split('_').filter(|part| !part.is_empty()) {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            camel.extend(first.to_uppercase());
            camel.push_str(chars.as_str());
        }
    }
    camel.push_str("System");
    Ident::new(&camel, fn_ident.span())
}

/// Renders a type without `TokenStream` spacing (`Mutex<Counter>`, not
/// `Mutex < Counter >`), for diagnostics and dedup keys.
fn pretty_type(ty: &Type) -> String {
    ty.to_token_stream()
        .to_string()
        .replace(" < ", "<")
        .replace(" >", ">")
        .replace(" :: ", "::")
        .replace(" , ", ", ")
}

/// Unwraps transparent wrappers one level at a time: `&T`, `Arc<T>`, `Rc<T>`,
/// `Box<T>`, `Option<T>`, `Pin<P>`. Anything else (including a newtype or a
/// type alias, which are syntactically opaque) stops the walk.
fn unwrap_wrapper(mut ty: &Type) -> &Type {
    loop {
        if let Type::Reference(reference) = ty {
            ty = &reference.elem;
            continue;
        }
        if let Type::Path(path) = ty
            && let Some(segment) = path.path.segments.last()
            && matches!(
                segment.ident.to_string().as_str(),
                "Arc" | "Rc" | "Box" | "Option" | "Pin"
            )
            && let syn::PathArguments::AngleBracketed(bracketed) = &segment.arguments
            && let Some(syn::GenericArgument::Type(inner)) = bracketed.args.first()
        {
            ty = inner;
            continue;
        }
        return ty;
    }
}

/// Whether the resource type has interior mutability: a silent `reads` would
/// mislead the level scheduler, so it must be declared explicitly. Wrapped
/// forms (`Arc<Mutex<T>>`, `Box<RwLock<T>>`, `&AtomicU32`, …) count too;
/// newtypes and aliases around them are syntactically invisible and must
/// likewise be declared explicitly (see the attribute docs).
fn has_interior_mutability(ty: &Type) -> bool {
    match LaneCollector::last_segment(unwrap_wrapper(ty)).as_deref() {
        Some("Mutex" | "RwLock" | "Cell" | "RefCell" | "OnceLock" | "OnceCell") => true,
        Some(name) if name.starts_with("Atomic") => true,
        _ => false,
    }
}

/// Builds the `access()` body tokens: explicit entries first, then derived
/// ones. Canonical order: sorted resource reads (explicit `writes` suppress
/// the same type in `reads`), then sorted read lanes, then sorted write
/// lanes (explicit lane entries merge with derived ones; a write covers a
/// read). Interior-mutability types (including wrapped ones like
/// `Arc<Mutex<T>>`) must be listed explicitly.
pub(crate) fn access_tokens(
    collector: &LaneCollector,
    explicit_reads: &[Type],
    explicit_writes: &[Type],
    explicit_lane_reads: &[Type],
    explicit_lane_writes: &[Type],
) -> syn::Result<TokenStream2> {
    let mut read_tys: Vec<Type> = explicit_reads.to_vec();
    for ty in &collector.resource_reads {
        if !read_tys
            .iter()
            .any(|kept| pretty_type(kept) == pretty_type(ty))
        {
            read_tys.push(ty.clone());
        }
    }
    let mut write_tys: Vec<Type> = explicit_writes.to_vec();
    read_tys.retain(|ty| {
        !write_tys
            .iter()
            .any(|written| pretty_type(written) == pretty_type(ty))
    });
    let mut error: Option<syn::Error> = None;
    for ty in read_tys.iter().chain(write_tys.iter()) {
        if has_interior_mutability(ty)
            && !explicit_reads
                .iter()
                .chain(explicit_writes.iter())
                .any(|listed| pretty_type(listed) == pretty_type(ty))
        {
            let next = syn::Error::new_spanned(
                ty,
                format!(
                    "#[smart_system]: resource `{}` has interior mutability — declare it with `writes(...)` or `reads(...)`",
                    pretty_type(ty)
                ),
            );
            match &mut error {
                Some(first) => first.combine(next),
                None => error = Some(next),
            }
        }
    }
    if let Some(error) = error {
        return Err(error);
    }

    read_tys.sort_by_key(pretty_type);
    write_tys.sort_by_key(pretty_type);
    let mut lane_accesses = collector.lane_accesses.clone();
    for ty in explicit_lane_reads {
        lane_accesses.push((ty.clone(), false));
    }
    for ty in explicit_lane_writes {
        lane_accesses.push((ty.clone(), true));
    }
    let lane_entries = access_entries(&lane_accesses);
    let mut lane_reads: Vec<&Type> = Vec::new();
    let mut lane_writes: Vec<&Type> = Vec::new();
    for (ty, write) in &lane_entries {
        if *write {
            lane_writes.push(ty);
        } else {
            lane_reads.push(ty);
        }
    }
    Ok(quote! {
        ornis_core::SystemAccess::new()
            #(.reads::<#read_tys>())*
            #(.writes::<#write_tys>())*
            #(.reads_lane::<#lane_reads>())*
            #(.writes_lane::<#lane_writes>())*
    })
}

/// Emission context for one `#[smart_system]` function: everything
/// [`Emission::struct_tokens`] needs, so the generator itself stays small.
pub(crate) struct Emission<'a> {
    pub(crate) struct_ident: Ident,
    pub(crate) fn_ident: &'a Ident,
    pub(crate) vis: &'a syn::Visibility,
    pub(crate) system_name: String,
    pub(crate) sig: &'a syn::Signature,
    pub(crate) resources: &'a Ident,
    pub(crate) fields: &'a [FieldParam],
    pub(crate) access: TokenStream2,
}

impl Emission<'_> {
    /// Builds the struct definition, its `::new()`, and the `System` impl.
    /// Call arguments follow the declaration order: `resources` in place,
    /// fields via their pass tokens.
    pub(crate) fn struct_tokens(&self) -> TokenStream2 {
        let Emission {
            struct_ident,
            fn_ident,
            vis,
            system_name,
            sig,
            resources,
            fields,
            access,
        } = self;
        let field_names: Vec<&Ident> = fields.iter().map(|f| &f.name).collect();
        let field_tys: Vec<&Type> = fields.iter().map(|f| &f.field_ty).collect();
        let mut call_args: Vec<TokenStream2> = Vec::new();
        for arg in &sig.inputs {
            if let FnArg::Typed(PatType { pat, .. }) = arg
                && let Pat::Ident(pat_ident) = &**pat
            {
                if pat_ident.ident == **resources {
                    call_args.push(quote! { resources });
                } else if let Some(field) = fields.iter().find(|f| f.name == pat_ident.ident) {
                    let pass = &field.pass;
                    call_args.push(quote! { #pass });
                }
            }
        }
        let fn_name = fn_ident.to_string();
        let doc_struct = format!("System generated by `#[smart_system]` from [`{fn_name}`].");
        let doc_new =
            format!("Creates the system generated by `#[smart_system]` from [`{fn_name}`].");
        let (struct_def, struct_ctor) = if field_names.is_empty() {
            (
                quote! { #vis struct #struct_ident; },
                quote! {
                    impl #struct_ident {
                        #[doc = #doc_new]
                        pub fn new() -> Self {
                            Self
                        }
                    }
                },
            )
        } else {
            (
                quote! {
                    #vis struct #struct_ident {
                        #( #field_names : #field_tys ),*
                    }
                },
                quote! {
                    impl #struct_ident {
                        #[doc = #doc_new]
                        pub fn new( #( #field_names : #field_tys ),* ) -> Self {
                            Self { #( #field_names ),* }
                        }
                    }
                },
            )
        };
        quote! {
            #[doc = #doc_struct]
            #struct_def
            #struct_ctor
            impl ornis_core::System for #struct_ident {
                fn name(&self) -> &'static str {
                    #system_name
                }
                fn access(&self) -> ornis_core::SystemAccess {
                    #access
                }
                fn run(&self, resources: &ornis_core::Resources) {
                    #fn_ident(#(#call_args),*);
                }
            }
        }
    }
}

#[cfg(test)]
mod smart_system_emit_tests {
    use super::*;
    use crate::smart_pipeline::LaneCollector;

    fn ident(name: &str) -> Ident {
        syn::parse_str(name).expect("ident parses")
    }

    fn ty(src: &str) -> Type {
        syn::parse_str(src).expect("type parses")
    }

    fn resources_ident() -> Ident {
        syn::parse_str("resources").expect("ident")
    }

    #[test]
    fn struct_ident_camel_cases() {
        assert_eq!(
            system_struct_ident(&ident("drive")).to_string(),
            "DriveSystem"
        );
        assert_eq!(
            system_struct_ident(&ident("player_input")).to_string(),
            "PlayerInputSystem"
        );
        assert_eq!(system_struct_ident(&ident("_x")).to_string(), "XSystem");
        assert_eq!(system_struct_ident(&ident("a__b")).to_string(), "ABSystem");
    }

    #[test]
    fn struct_ident_unraws_keywords() {
        let raw: Ident = syn::parse_str("r#loop").expect("raw ident parses");
        assert_eq!(system_struct_ident(&raw).to_string(), "LoopSystem");
    }

    #[test]
    fn interior_mutability_set() {
        for name in [
            "Mutex<C>",
            "RwLock<C>",
            "AtomicBool",
            "AtomicU64",
            "Cell<C>",
            "RefCell<C>",
            "OnceLock<C>",
            "std::sync::Arc<Mutex<C>>",
            "Arc<Mutex<C>>",
            "Box<RwLock<C>>",
            "Option<Mutex<C>>",
            "Option<Box<RwLock<C>>>",
            "&AtomicU32",
            "&Mutex<C>",
        ] {
            assert!(has_interior_mutability(&ty(name)), "{name}");
        }
        for name in [
            "Cfg",
            "SmartStore",
            "Vec<u8>",
            "MutexLike",
            "Arc<Cfg>",
            "Option<Cfg>",
        ] {
            assert!(!has_interior_mutability(&ty(name)), "{name}");
        }
    }

    #[test]
    fn pretty_type_has_no_token_spaces() {
        assert_eq!(pretty_type(&ty("Mutex<Counter>")), "Mutex<Counter>");
        assert_eq!(
            pretty_type(&ty("std::sync::Arc<Mutex<Counter>>")),
            "std::sync::Arc<Mutex<Counter>>"
        );
    }

    #[test]
    fn access_tokens_merges_explicit_and_derived() {
        let mut collector = LaneCollector::new("#[smart_system]");
        collector.resources_param = Some(resources_ident());
        let pos: Type = syn::parse_str("Pos").expect("type");
        let vel: Type = syn::parse_str("Vel").expect("type");
        collector.lane_accesses.push((pos, true));
        collector.lane_accesses.push((vel, false));
        let tokens = access_tokens(&collector, &[], &[], &[], &[]).expect("access");
        let rendered = tokens.to_string();
        assert!(rendered.contains("reads_lane :: < Vel >"), "{rendered}");
        assert!(rendered.contains("writes_lane :: < Pos >"), "{rendered}");
    }

    #[test]
    fn access_tokens_merges_explicit_lanes() {
        let collector = LaneCollector::new("#[smart_system]");
        let hidden: Type = syn::parse_str("Hidden").expect("type");
        let tokens = access_tokens(&collector, &[], &[], &[hidden], &[]).expect("access");
        assert!(
            tokens.to_string().contains("Hidden"),
            "{:?}",
            tokens.to_string()
        );
    }
}
