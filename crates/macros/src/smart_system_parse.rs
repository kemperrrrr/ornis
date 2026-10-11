//! Signature validation and parameter collection for `#[smart_system]`
//! (IDEAS §32).
//!
//! [`check_signature`] pins the function shape (sync, safe, non-generic,
//! unit return, at least one input); [`collect_params`] splits the inputs
//! into the entity parameter plus typed parameters — parsed in
//! [`crate::smart_system_parse_params`] — and rejects reserved `__smart_*`
//! names and conflicting duplicate declarations. [`config_struct_parts`]
//! and [`combine_errors`] feed the emitter with the `#[config]` fields and
//! the aggregated diagnostics.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{FnArg, Ident, ItemFn, Type};

use crate::smart_system::{ParamKind, RESERVED_PREFIX, TypedParam};
use crate::smart_system_parse_params::{parse_first_param, parse_param};

pub(crate) fn config_struct_parts(
    params: &[TypedParam],
) -> (Vec<TokenStream2>, Vec<TokenStream2>, Vec<TokenStream2>) {
    let mut fields = Vec::new();
    let mut ctor_params = Vec::new();
    let mut ctor_inits = Vec::new();
    for param in params {
        if matches!(param.kind, ParamKind::Config) {
            let name = &param.name;
            let ty = &param.inner;
            fields.push(quote! { pub #name: #ty, });
            ctor_params.push(quote! { #name: #ty });
            ctor_inits.push(quote! { #name, });
        }
    }
    (fields, ctor_params, ctor_inits)
}

pub(crate) fn combine_errors(errors: Vec<syn::Error>) -> syn::Error {
    let mut iter = errors.into_iter();
    match iter.next() {
        Some(first) => {
            let mut combined = first;
            for err in iter {
                combined.combine(err);
            }
            combined
        }
        // Unreachable in practice (callers check non-empty first); kept
        // total instead of panicking so error aggregation stays clean.
        None => syn::Error::new(
            proc_macro2::Span::call_site(),
            "smart_system: internal error",
        ),
    }
}

// ---------------------------------------------------------------------------
// signature validation + parameter collection (kept here so `expand` stays
// a thin orchestrator; pure `syn` checks, no project calls)
// ---------------------------------------------------------------------------

/// Validates the system signature shape: sync, safe, non-generic, unit
/// return, at least one input.
pub(crate) fn check_signature(func: &ItemFn) -> Result<(), syn::Error> {
    if func.sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            func.sig.fn_token,
            "smart_system: async systems are not supported",
        ));
    }
    if matches!(func.sig.safety, syn::Safety::Unsafe(_)) {
        return Err(syn::Error::new_spanned(
            func.sig.fn_token,
            "smart_system: unsafe systems are not supported",
        ));
    }
    if !func.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &func.sig.generics,
            "smart_system: generic systems are not supported in v2",
        ));
    }
    if let syn::ReturnType::Type(_, ty) = &func.sig.output
        && !matches!(**ty, Type::Tuple(ref t) if t.elems.is_empty())
    {
        return Err(syn::Error::new_spanned(
            &func.sig.output,
            "smart_system: systems must return `()` — bare `return` skips the entity",
        ));
    }
    if func.sig.inputs.is_empty() {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "smart_system: the first parameter must be `u: &T` / `u: &mut T` with `T: Pack`",
        ));
    }
    Ok(())
}

/// Splits inputs into the entity parameter plus typed parameters,
/// rejecting reserved `__smart_*` names.
pub(crate) fn collect_params<'a, I>(
    inputs: I,
) -> Result<(Ident, Type, bool, Vec<TypedParam>), syn::Error>
where
    I: Iterator<Item = &'a FnArg>,
{
    let mut inputs = inputs;
    let Some(first) = inputs.next() else {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "smart_system: the first parameter must be `u: &T` / `u: &mut T` with `T: Pack`",
        ));
    };
    let (u_ident, pack_ty, is_mut) = parse_first_param(first)?;

    let mut params: Vec<TypedParam> = Vec::new();
    for arg in inputs {
        params.push(parse_param(arg)?);
    }

    reject_conflicting_params(&params)?;

    for param in &params {
        if param.name.to_string().starts_with(RESERVED_PREFIX) {
            return Err(syn::Error::new_spanned(
                &param.name,
                "smart_system: parameter names starting with `__smart_` are reserved",
            ));
        }
    }
    Ok((u_ident, pack_ty, is_mut, params))
}

/// Rejects two parameters that address the same resource when at least one
/// of them declares a write: `Res<C>` + `ResMut<C>`, two `ResMut<C>`,
/// `EventWriter<E>` + `EventReader<E>`. Two read-only `Res<C>` stay legal —
/// the scheduler needs no ordering between them. Keys are compared as
/// written, so differently spelled aliases are not detected here (the
/// `ResMutResource`/`PlainResource` bounds still gate what they can name).
fn reject_conflicting_params(params: &[TypedParam]) -> Result<(), syn::Error> {
    for (index, later) in params.iter().enumerate() {
        let Some(family) = later.kind.resource_family() else {
            continue;
        };
        let later_key = resource_key(later);
        for earlier in &params[..index] {
            if earlier.kind.resource_family() != Some(family)
                || resource_key(earlier) != later_key
                || !(earlier.kind.declares_write() || later.kind.declares_write())
            {
                continue;
            }
            return Err(syn::Error::new_spanned(
                &later.name,
                format!(
                    "smart_system: `{}` declares `{}` a second time — declare each \
                     resource/queue through one parameter (`Res` reads, `ResMut` writes, \
                     `EventWriter`/`EventReader` events)",
                    later.name, later_key
                ),
            ));
        }
    }
    Ok(())
}

/// Written spelling of a parameter's resource key: the inner type for
/// `Res`/`ResMut` (the resource) and for the event roles (the payload of
/// the `EventStore<E>` singleton).
fn resource_key(param: &TypedParam) -> String {
    let inner = &param.inner;
    quote!(#inner).to_string()
}
