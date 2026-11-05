//! Parameter parsing for `#[smart_system]` (IDEAS §32).
//!
//! Turns one `FnArg` into a [`TypedParam`]: the entity parameter
//! (`u: &T` / `u: &mut T`, `T: Pack`), the typed wrappers
//! (`Res`/`ResMut`/`EventWriter`/`EventReader`/`StructuralWriter`), and the
//! attribute parameters (`#[config]`, `#[entity]`). Forbidden shapes get
//! dedicated diagnostics here; signature-level checks and the aggregate
//! collection live in [`crate::smart_system_parse`].

use syn::{FnArg, GenericArgument, Ident, Pat, PathArguments, Type};

use crate::smart_system::{
    ParamKind, TypedParam, is_param_wrapper, is_smart_store_ty, path_last_ident,
};

pub(crate) fn parse_first_param(arg: &FnArg) -> Result<(Ident, Type, bool), syn::Error> {
    let typed = match arg {
        FnArg::Receiver(r) => {
            return Err(syn::Error::new_spanned(
                r,
                "smart_system: systems are free functions — the first parameter must be \
                 `u: &T` / `u: &mut T` with `T: Pack`",
            ));
        }
        FnArg::Typed(t) => t,
    };
    let name = match typed.pat.as_ref() {
        Pat::Ident(id) => id.ident.clone(),
        _ => {
            return Err(syn::Error::new_spanned(
                &typed.pat,
                "smart_system: the first parameter must be a plain binding `u: &T` / `u: &mut T`",
            ));
        }
    };
    match typed.ty.as_ref() {
        Type::Reference(r) => {
            let is_mut = r.mutability.is_some();
            Ok((name, (*r.elem).clone(), is_mut))
        }
        _ => Err(syn::Error::new_spanned(
            &typed.ty,
            "smart_system: the first parameter must be `u: &T` / `u: &mut T` with `T: Pack` \
             — owned values and other shapes are not supported",
        )),
    }
}
pub(crate) fn parse_param(arg: &FnArg) -> Result<TypedParam, syn::Error> {
    let typed = match arg {
        FnArg::Receiver(r) => {
            return Err(syn::Error::new_spanned(
                r,
                "smart_system: unexpected `self` — only the first parameter is the entity, \
                 the rest are `Res<T>` / `ResMut<T>` / `EventWriter<E>` / `EventReader<E>` / \
                 `StructuralWriter` / `#[config]`",
            ));
        }
        FnArg::Typed(t) => t,
    };
    let name = match typed.pat.as_ref() {
        Pat::Ident(id) => id.ident.clone(),
        _ => {
            return Err(syn::Error::new_spanned(
                &typed.pat,
                "smart_system: parameters must be plain bindings (`time: Res<FixedTime>`)",
            ));
        }
    };
    let is_config = typed.attrs.iter().any(|a| a.path().is_ident("config"));
    if is_config {
        let ty = (*typed.ty).clone();
        return Ok(TypedParam::new(name, ParamKind::Config, ty.clone(), ty));
    }
    let is_entity = typed.attrs.iter().any(|a| a.path().is_ident("entity"));
    if is_entity {
        if path_last_ident(&typed.ty).as_deref() != Some("Entity") {
            return Err(syn::Error::new_spanned(
                &typed.ty,
                "smart_system: `#[entity]` parameters must have type `Entity`",
            ));
        }
        let ty = (*typed.ty).clone();
        return Ok(TypedParam::new(
            name,
            ParamKind::EntityHandle,
            ty.clone(),
            ty,
        ));
    }
    match typed.ty.as_ref() {
        Type::Reference(_) => Err(syn::Error::new_spanned(
            &typed.ty,
            "smart_system: raw `&Resources` is forbidden — declare typed parameters \
             (`Res<T>`, `ResMut<T>`, `EventWriter<E>` / `EventReader<E>`, `StructuralWriter`) \
             so access follows from types",
        )),
        Type::Path(tp) => parse_wrapper_param(name, &typed.ty, tp),
        _ => Err(syn::Error::new_spanned(
            &typed.ty,
            "smart_system: parameters must be `Res<T>` / `ResMut<T>` / `EventWriter<E>` / \
             `EventReader<E>` / `StructuralWriter` (or `#[config] name: T`)",
        )),
    }
}
/// Validates one `Res<T>` / `ResMut<T>` / `EventWriter<E>` / `EventReader<E>` /
/// `StructuralWriter` wrapper parameter.
pub(crate) fn parse_wrapper_param(
    name: Ident,
    ty: &Type,
    tp: &syn::TypePath,
) -> Result<TypedParam, syn::Error> {
    let Some(seg) = tp.path.segments.last() else {
        return Err(syn::Error::new_spanned(
            ty,
            "smart_system: parameters must be `Res<T>` / `ResMut<T>` / `EventWriter<E>` / \
             `EventReader<E>` / `StructuralWriter` (or `#[config] name: T`)",
        ));
    };
    let Some(kind) = is_param_wrapper(seg) else {
        if seg.ident == "SmartStore" {
            return Err(syn::Error::new_spanned(
                ty,
                "smart_system: raw `SmartStore` is forbidden — the first `&T`/`&mut T` \
                 (`T: Pack`) parameter already selects the entity set",
            ));
        }
        return Err(syn::Error::new_spanned(
            ty,
            "smart_system: parameters must be `Res<T>` / `ResMut<T>` / `EventWriter<E>` / \
             `EventReader<E>` / `StructuralWriter` (or `#[config] name: T`)",
        ));
    };
    // `StructuralWriter` carries a lifetime and a queue-kind argument, not
    // a resource type: handled before the single-type-argument checks below.
    if matches!(
        kind,
        ParamKind::StructuralWriterFixed | ParamKind::StructuralWriterFrame
    ) {
        return parse_structural_writer(name, ty, seg);
    }
    let PathArguments::AngleBracketed(args) = &seg.arguments else {
        return Err(syn::Error::new_spanned(
            ty,
            "smart_system: wrapper needs its inner type (`Res<FixedTime>`)",
        ));
    };
    if args.args.len() != 1 {
        return Err(syn::Error::new_spanned(
            ty,
            "smart_system: wrapper takes exactly one type (`Res<FixedTime>`)",
        ));
    }
    let GenericArgument::Type(inner) = &args.args[0] else {
        return Err(syn::Error::new_spanned(
            ty,
            "smart_system: wrapper takes exactly one type (`Res<FixedTime>`)",
        ));
    };
    reject_forbidden_inner(&kind, inner)?;
    if matches!(kind, ParamKind::Res) {
        check_plain_shape(inner)?;
    }
    Ok(TypedParam::new(name, kind, inner.clone(), (*ty).clone()))
}

/// Accepts the `StructuralWriter` shapes and derives the queue kind from the
/// type argument.
///
/// - bare `StructuralWriter` / `StructuralWriter<'_>` → `FixedStep` (the
///   fixed plan, applied after every tick). This is the default because
///   simulation code is the common case and it is the safe choice: a fixed
///   writer is validated against the stage at registration.
/// - `StructuralWriter<FrameStep>` / `StructuralWriter<'_, FrameStep>` → the
///   frame queue, applied once per frame before `Stage::Last`.
///
/// Anything else (`StructuralWriter<u32>`, two type arguments) is rejected:
/// the only meaningful parameter is the queue kind.
fn parse_structural_writer(
    name: Ident,
    ty: &Type,
    seg: &syn::PathSegment,
) -> Result<TypedParam, syn::Error> {
    let mut kind = ParamKind::StructuralWriterFixed;
    match &seg.arguments {
        PathArguments::None => {}
        PathArguments::AngleBracketed(args) => {
            // Accepted forms: `<'_>`, `<Kind>`, `<'_, Kind>` — at most one
            // lifetime and at most one known kind, nothing else.
            let lifetimes = args
                .args
                .iter()
                .filter(|arg| matches!(arg, GenericArgument::Lifetime(_)))
                .count();
            let mut kinds = args.args.iter().filter_map(|arg| match arg {
                GenericArgument::Type(Type::Path(tp)) => tp.path.segments.last(),
                _ => None,
            });
            let first = kinds.next();
            let extra_kinds = kinds.count();
            let known = match first.as_ref().map(|seg| seg.ident.to_string()).as_deref() {
                Some("FixedStep") => Some(ParamKind::StructuralWriterFixed),
                Some("FrameStep") => Some(ParamKind::StructuralWriterFrame),
                _ => None,
            };
            let type_args = args
                .args
                .iter()
                .filter(|arg| matches!(arg, GenericArgument::Type(_)))
                .count();
            match known {
                Some(found) if extra_kinds == 0 && type_args == 1 && lifetimes <= 1 => kind = found,
                None if type_args == 0 && lifetimes <= 1 => {}
                _ => {
                    return Err(syn::Error::new_spanned(
                        ty,
                        "smart_system: `StructuralWriter` takes only a queue-kind argument — \
                         write `StructuralWriter` / `StructuralWriter<'_, FixedStep>` for the \
                         fixed plan (applied after every tick, `Stage::Gameplay`) or \
                         `StructuralWriter<'_, FrameStep>` for the once-per-frame plan",
                    ));
                }
            }
        }
        PathArguments::Parenthesized(_) => {
            return Err(syn::Error::new_spanned(
                ty,
                "smart_system: `StructuralWriter` takes only a queue-kind argument — \
                 write `StructuralWriter` / `StructuralWriter<'_, FixedStep>` for the \
                 fixed plan (applied after every tick, `Stage::Gameplay`) or \
                 `StructuralWriter<'_, FrameStep>` for the once-per-frame plan",
            ));
        }
    }
    Ok(TypedParam::new(name, kind, ty.clone(), ty.clone()))
}

/// Rejects `Res`/`ResMut` of a type that has a dedicated parameter: the
/// `SmartStore`/`Resources` handles and the `EventStore`/`StructuralQueue`
/// singletons. Naming them as a plain resource would bypass the role-based
/// declaration, so duplicate and conflicting parameters would go undetected.
fn reject_forbidden_inner(kind: &ParamKind, inner: &Type) -> Result<(), syn::Error> {
    let last = path_last_ident(inner);
    if matches!(kind, ParamKind::Res) && is_smart_store_ty(inner) {
        return Err(syn::Error::new_spanned(
            inner,
            "smart_system: `Res<SmartStore>` is forbidden — reach components through \
             the first `&T`/`&mut T` (`T: Pack`) parameter",
        ));
    }
    if matches!(kind, ParamKind::ResMut)
        && matches!(last.as_deref(), Some("SmartStore" | "Resources"))
    {
        return Err(syn::Error::new_spanned(
            inner,
            "smart_system: `ResMut<SmartStore>` and `ResMut<Resources>` are forbidden — \
             use `Pack`/typed parameters so scheduler access stays explicit",
        ));
    }
    let dedicated = match (kind, last.as_deref()) {
        (ParamKind::Res | ParamKind::ResMut, Some("EventStore")) => {
            "smart_system: `Res<EventStore<_>>` / `ResMut<EventStore<_>>` are forbidden — \
             declare queue access with `EventWriter<E>` / `EventReader<E>` so duplicate \
             and conflicting declarations stay detectable"
        }
        (ParamKind::Res | ParamKind::ResMut, Some("StructuralQueue")) => {
            "smart_system: `Res<StructuralQueue>` / `ResMut<StructuralQueue>` are forbidden — \
             declare deferred ops with `StructuralWriter` so duplicate and conflicting \
             declarations stay detectable"
        }
        _ => return Ok(()),
    };
    Err(syn::Error::new_spanned(inner, dedicated))
}

/// Early syntactic backstop for the `PlainResource` bound: well-known
/// interior-mutability types are rejected with one clean diagnostic
/// instead of a cascade of unsatisfied-bound errors (custom interior
/// types are still caught by the `Res<T: PlainResource>` bound itself).
pub(crate) fn check_plain_shape(inner: &Type) -> Result<(), syn::Error> {
    // Interior-mutability names rejected with one clean diagnostic
    // (one line per family; keep alphabetical).
    const DENIED: &[&str] = &[
        "Mutex",
        "RwLock",
        "Cell",
        "RefCell",
        "UnsafeCell",
        "OnceCell",
        "AtomicBool",
        "AtomicI8",
        "AtomicI16",
        "AtomicI32",
        "AtomicI64",
        "AtomicIsize",
        "AtomicU8",
        "AtomicU16",
        "AtomicU32",
        "AtomicU64",
        "AtomicUsize",
    ];
    if let Some(name) = path_last_ident(inner)
        && DENIED.contains(&name.as_str())
    {
        return Err(syn::Error::new_spanned(
            inner,
            format!(
                "smart_system: `Res<{name}<…>>` holds interior mutability and is not \
                 `PlainResource` — use `ResMut` (declares a write) instead"
            ),
        ));
    }
    Ok(())
}
