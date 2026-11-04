//! Implementation of `#[smart_system]`: per-object kernel (IDEAS §32).
//!
//! A system is an ordinary function for **one** object:
//!
//! ```ignore
//! #[smart_system]
//! fn chase(u: &mut Unit, time: Res<FixedTime>) {
//!     if u.team != Team::Enemy { return; } // filter: bare `return` skips the entity
//!     u.position += u.velocity * time.delta_seconds();
//! }
//! ```
//!
//! The macro generates a `ChaseSystem` struct implementing
//! [`ornis_core::System`](System). The first parameter (`u: &T` / `&mut T`,
//! `T: Pack`) selects the bundle; the body runs once per entity and sees each
//! field it touches by reference (`u.position` → `&T` or `&mut T`), so the
//! access set lists only those lanes, and only the written ones as writes.
//! Field gather and scatter run per entity on lanes guarded once per run.
//!
//! Entities are selected by the touched required (non-`Option`) fields, in
//! entity-id order. A body must touch at least one required field (checked at
//! compile time), because optional lanes list only the entries that exist.
//! Access also follows from parameter types: `Res` → read, `ResMut` → write,
//! `EventWriter`/`EventReader` → exclusive queue access, `StructuralWriter`
//! → deferred structural-queue write (R4: `add`/`remove`/`spawn` enqueue
//! during the frame and apply before `Stage::Last`; raw `Res`/`ResMut`
//! of `StructuralQueue` are rejected in favour of the parameter). Raw
//! `&Resources`,
//! `Res<SmartStore>`, and `ResMut` aliases of `SmartStore`/`Resources` are
//! compile errors, as are writes to another entity (`t.hp -= …` — use
//! [`ornis_core::EventWriter`] instead), passing whole `u` to a free function
//! or using `u` other than as `u.field`, writing `u.field` through `&`, and
//! declaring one resource twice when either declaration writes
//! (`Res<C>` + `ResMut<C>`).
//!
//! [System]: https://docs.rs/ornis-core/latest/ornis_core/schedule/trait.System.html

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::format_ident;
use syn::{GenericArgument, Ident, ItemFn, Lifetime, PathArguments, Type, parse_macro_input};

use crate::smart_system_emit::expand;

/// Reserved prefix for macro-introduced bindings (store handle, entity).
pub(crate) const RESERVED_PREFIX: &str = "__smart_";

pub(crate) enum ParamKind {
    /// `#[config] name: Ty` — owned data stored on the system struct.
    Config,
    /// `#[entity] name: Entity` — the current entity handle.
    EntityHandle,
    Res,
    ResMut,
    EventWriter,
    EventReader,
    /// `StructuralWriter` (optional `<'_>`) — deferred `add`/`remove`/
    /// `spawn` ops for the frame-boundary apply (R4).
    StructuralWriter,
}

impl ParamKind {
    /// Resource family of this role — the namespace its inner type keys:
    /// `Res`/`ResMut` address the resource itself, `EventWriter`/
    /// `EventReader` address the `EventStore<E>` singleton, and
    /// `StructuralWriter` addresses the `StructuralQueue` singleton.
    /// Configs and entity handles are not resources.
    pub(crate) fn resource_family(&self) -> Option<&'static str> {
        match self {
            ParamKind::Res | ParamKind::ResMut => Some("resource"),
            ParamKind::EventWriter | ParamKind::EventReader => Some("event"),
            ParamKind::StructuralWriter => Some("structural"),
            ParamKind::Config | ParamKind::EntityHandle => None,
        }
    }

    /// Whether the role declares a scheduler write: `ResMut`, both
    /// event roles (a reader may drain the queue), and the structural
    /// writer (every queued op mutates the store at apply).
    pub(crate) fn declares_write(&self) -> bool {
        matches!(
            self,
            ParamKind::ResMut
                | ParamKind::EventWriter
                | ParamKind::EventReader
                | ParamKind::StructuralWriter
        )
    }
}

pub(crate) struct TypedParam {
    pub(crate) name: Ident,
    pub(crate) kind: ParamKind,
    /// Inner type (`X` in `Res<X>`; full type for `#[config]`).
    pub(crate) inner: Type,
    /// Full declared type (`Res<X>` as written; same as `inner` for configs).
    pub(crate) wrapper: Type,
}

impl TypedParam {
    /// Builds a parameter record (single construction site for the
    /// overlapping `inner`/`wrapper` fields).
    pub(crate) fn new(name: Ident, kind: ParamKind, inner: Type, wrapper: Type) -> Self {
        Self {
            name,
            kind,
            inner,
            wrapper,
        }
    }
}

pub(crate) fn is_param_wrapper(seg: &syn::PathSegment) -> Option<ParamKind> {
    let name = seg.ident.to_string();
    match name.as_str() {
        "Res" => Some(ParamKind::Res),
        "ResMut" => Some(ParamKind::ResMut),
        "EventWriter" => Some(ParamKind::EventWriter),
        "EventReader" => Some(ParamKind::EventReader),
        "StructuralWriter" => Some(ParamKind::StructuralWriter),
        _ => None,
    }
}

pub(crate) fn path_last_ident(ty: &Type) -> Option<String> {
    if let Type::Path(tp) = ty {
        tp.path.segments.last().map(|s| s.ident.to_string())
    } else {
        None
    }
}

pub(crate) fn is_smart_store_ty(ty: &Type) -> bool {
    path_last_ident(ty).as_deref() == Some("SmartStore")
}

/// Adds an elided lifetime for the generated body function (users write
/// the short form): `Res<X>` → `Res<'_, X>`, and bare `StructuralWriter`
/// → `StructuralWriter<'_>`. An already-lifetime-only
/// `StructuralWriter<'_>` is left untouched.
pub(crate) fn with_elided_lifetime(mut ty: Type) -> Type {
    if let Type::Path(tp) = &mut ty
        && let Some(seg) = tp.path.segments.last_mut()
        && matches!(
            seg.ident.to_string().as_str(),
            "Res" | "ResMut" | "EventWriter" | "EventReader" | "StructuralWriter"
        )
    {
        let elided = || GenericArgument::Lifetime(Lifetime::new("'_", Span::call_site()));
        match &mut seg.arguments {
            PathArguments::None => {
                seg.arguments =
                    PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
                        colon2_token: None,
                        lt_token: Default::default(),
                        args: std::iter::once(elided()).collect(),
                        gt_token: Default::default(),
                    });
            }
            PathArguments::AngleBracketed(args)
                if args.args.len() == 1
                    && matches!(args.args.first(), Some(GenericArgument::Type(_))) =>
            {
                args.args.insert(0, elided());
            }
            _ => {}
        }
    }
    ty
}
// qual:allow(test_quality, untested) reason: "proc-macro entry — exercised by trybuild UI tests and core e2e, both invisible to the call graph"
pub fn attribute(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return syn::Error::new(
            Span::call_site(),
            "smart_system: no arguments expected — `#[smart_system]` takes none",
        )
        .to_compile_error()
        .into();
    }
    let func = parse_macro_input!(item as ItemFn);
    match expand(func) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

pub(crate) fn camel_system_name(fn_name: &Ident) -> Ident {
    let name = fn_name.to_string();
    let mut camel = String::with_capacity(name.len() + "System".len());
    let mut upper_next = true;
    for ch in name.chars() {
        if ch == '_' {
            upper_next = true;
        } else if upper_next {
            camel.extend(ch.to_uppercase());
            upper_next = false;
        } else {
            camel.push(ch);
        }
    }
    camel.push_str("System");
    format_ident!("{}", camel)
}

#[cfg(test)]
mod tests {
    use syn::{Expr, Pat};

    #[test]
    fn link_alias_shape_accepts_handles_only() {
        let path: Expr = syn::parse_quote!(t);
        assert!(crate::smart_system_body_links::is_link_alias_shape(&path));
        let reference: Expr = syn::parse_quote!(&t);
        assert!(crate::smart_system_body_links::is_link_alias_shape(
            &reference
        ));
        let cloned: Expr = syn::parse_quote!(t.clone());
        assert!(crate::smart_system_body_links::is_link_alias_shape(&cloned));
        let computed: Expr = syn::parse_quote!(u.target.map(|t| t));
        assert!(!crate::smart_system_body_links::is_link_alias_shape(
            &computed
        ));
    }

    #[test]
    fn visit_let_registers_link_alias() {
        use crate::smart_system_body_visit::BodyInfo;
        use std::collections::HashSet;
        let mut info = BodyInfo {
            u: "u".to_string(),
            links: HashSet::new(),
            errors: Vec::new(),
        };
        let pat: Pat = syn::parse_quote!(t);
        let init: Expr = syn::parse_quote!(b.target);
        info.visit_let(&pat, &init);
        assert!(!info.links.contains("t"));
        let init_alias: Expr = syn::parse_quote!(t);
        info.links.insert("t".to_string());
        let pat2: Pat = syn::parse_quote!(t2);
        info.visit_let(&pat2, &init_alias);
        assert!(info.links.contains("t2"));
    }
}
