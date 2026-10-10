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
//! [`ornis_core::System`](System): the first parameter (`u: &mut T` /
//! `&T`, `T: Pack`) selects the entity set, the body runs once per
//! entity (gather with `pack_get`, scatter with `pack_put`), and the
//! access set follows from the parameter types (`Res` → read, `ResMut`
//! → write, `Events` → write when sending else read, whole-bundle lanes
//! for `u`). A raw `&Resources` parameter and `Res<SmartStore>` are
//! compile errors, as are writes to another entity (`t.hp -= …` — use
//! [`ornis_core::Events`] instead) and passing whole `u` to a free
//! function. Field-level lane inference (only touched `u.field` lanes)
//! is a follow-up (v2b); v2 declares the whole bundle.
//!
//! [System]: https://docs.rs/ornis-core/latest/ornis_core/schedule/trait.System.html

use std::collections::{HashMap, HashSet};

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{
    Expr, FnArg, GenericArgument, Ident, ItemFn, Lifetime, Pat, PathArguments, Type,
    parse_macro_input,
};

/// Reserved prefix for macro-introduced bindings (store handle, entity).
const RESERVED_PREFIX: &str = "__smart_";

enum ParamKind {
    /// `#[config] name: Ty` — owned data stored on the system struct.
    Config,
    /// `#[entity] name: Entity` — the current entity handle.
    EntityHandle,
    Res,
    ResMut,
    Events,
}

struct TypedParam {
    name: Ident,
    kind: ParamKind,
    /// Inner type (`X` in `Res<X>`; full type for `#[config]`).
    inner: Type,
    /// Full declared type (`Res<X>` as written; same as `inner` for configs).
    wrapper: Type,
}

fn is_param_wrapper(seg: &syn::PathSegment) -> Option<ParamKind> {
    let name = seg.ident.to_string();
    match name.as_str() {
        "Res" => Some(ParamKind::Res),
        "ResMut" => Some(ParamKind::ResMut),
        "Events" => Some(ParamKind::Events),
        _ => None,
    }
}

fn path_last_ident(ty: &Type) -> Option<String> {
    if let Type::Path(tp) = ty {
        tp.path.segments.last().map(|s| s.ident.to_string())
    } else {
        None
    }
}

fn is_smart_store_ty(ty: &Type) -> bool {
    path_last_ident(ty).as_deref() == Some("SmartStore")
}

/// Adds an elided lifetime to `Res<X>` → `Res<'_, X>` for the generated
/// body function (users write the Bevy-style short form).
fn with_elided_lifetime(mut ty: Type) -> Type {
    if let Type::Path(tp) = &mut ty
        && let Some(seg) = tp.path.segments.last_mut()
        && matches!(seg.ident.to_string().as_str(), "Res" | "ResMut" | "Events")
        && let PathArguments::AngleBracketed(args) = &mut seg.arguments
        && args.args.len() == 1
    {
        args.args.insert(
            0,
            GenericArgument::Lifetime(Lifetime::new("'_", Span::call_site())),
        );
    }
    ty
}

/// Root ident of a `a.b.c` member chain, when the base is a plain ident.
fn member_root(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(p) => {
            if p.path.segments.len() == 1 && p.qself.is_none() {
                Some(p.path.segments[0].ident.to_string())
            } else {
                None
            }
        }
        Expr::Field(f) => member_root(&f.base),
        _ => None,
    }
}

/// Whether `expr` contains a member access rooted at `u` (own field) or
/// at a known link binding (another entity's field).
struct RootMention<'a> {
    u: &'a str,
    links: &'a HashSet<String>,
    found: bool,
}

impl<'ast> Visit<'ast> for RootMention<'_> {
    fn visit_expr_field(&mut self, node: &'ast syn::ExprField) {
        if let Some(root) = member_root(&node.base)
            && (root == self.u || self.links.contains(&root))
        {
            self.found = true;
        }
        syn::visit::visit_expr_field(self, node);
    }
}

fn mentions_entity_member(expr: &Expr, u: &str, links: &HashSet<String>) -> bool {
    let mut finder = RootMention {
        u,
        links,
        found: false,
    };
    finder.visit_expr(expr);
    finder.found
}

/// Whether `expr` is an alias of a link binding (`t`, `&t`, `t.clone()`,
/// `Some(t)`) — i.e. the bound value is still the entity handle, not a
/// computed value like `u.target.map(…)`.
fn is_link_alias(expr: &Expr, links: &HashSet<String>) -> bool {
    match expr {
        Expr::Path(p) => {
            p.qself.is_none()
                && p.path.segments.len() == 1
                && links.contains(&p.path.segments[0].ident.to_string())
        }
        Expr::Reference(r) => is_link_alias(&r.expr, links),
        Expr::Call(c) => {
            // `Some(t)` / `Ok(t)`.
            if let Expr::Path(f) = c.func.as_ref() {
                let is_some = f.path.segments.len() == 1
                    && matches!(f.path.segments[0].ident.to_string().as_str(), "Some" | "Ok");
                if is_some && c.args.len() == 1 {
                    return is_link_alias(&c.args[0], links);
                }
            }
            false
        }
        Expr::MethodCall(m) => {
            matches!(m.method.to_string().as_str(), "clone" | "copied")
                && is_link_alias(&m.receiver, links)
        }
        _ => false,
    }
}

fn pat_idents(pat: &Pat, out: &mut Vec<String>) {
    match pat {
        Pat::Ident(id) => out.push(id.ident.to_string()),
        Pat::TupleStruct(ts) => {
            for elem in &ts.elems {
                pat_idents(elem, out);
            }
        }
        Pat::Tuple(t) => {
            for elem in &t.elems {
                pat_idents(elem, out);
            }
        }
        Pat::Struct(s) => {
            for field in &s.fields {
                pat_idents(&field.pat, out);
            }
        }
        Pat::Slice(s) => {
            for elem in &s.elems {
                pat_idents(elem, out);
            }
        }
        Pat::Reference(r) => pat_idents(&r.pat, out),
        Pat::Paren(p) => pat_idents(&p.pat, out),
        _ => {}
    }
}

/// Pre-pass over the body: link bindings (handles to *other* entities),
/// per-`Events`-param send flags, and syntactic ban diagnostics.
struct BodyInfo {
    u: String,
    links: HashSet<String>,
    sends: HashMap<String, bool>,
    errors: Vec<syn::Error>,
}

impl BodyInfo {
    fn note_link_pat(&mut self, pat: &Pat) {
        let mut idents = Vec::new();
        pat_idents(pat, &mut idents);
        self.links.extend(idents);
    }

    /// `let PAT = INIT`: alias-of-link extends the link set.
    fn visit_let(&mut self, pat: &Pat, init: &Expr) {
        if is_link_alias(init, &self.links)
            || mentions_entity_member(init, &self.u, &self.links) && is_link_alias_shape(init)
        {
            self.note_link_pat(pat);
        }
    }
}

/// `let x = EXPR` shapes that preserve the entity handle (not computed
/// values): a bare path, a reference, or `.clone()`/`.copied()`.
fn is_link_alias_shape(expr: &Expr) -> bool {
    match expr {
        Expr::Path(_) | Expr::Reference(_) => true,
        Expr::MethodCall(m) => {
            matches!(m.method.to_string().as_str(), "clone" | "copied")
        }
        _ => false,
    }
}

impl<'ast> Visit<'ast> for BodyInfo {
    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let Some(init) = &node.init {
            // `let PAT = INIT` (+ optional `else` divergent — links in the
            // `else` branch never escape, but marking is harmless).
            self.visit_let(&node.pat, &init.expr);
        }
        syn::visit::visit_local(self, node);
    }

    fn visit_expr_if(&mut self, node: &'ast syn::ExprIf) {
        // `if let PAT = SCRUT`: bindings sourced from `u.field` (or a
        // link) are handles to another entity.
        if let Expr::Let(let_expr) = node.cond.as_ref()
            && (mentions_entity_member(&let_expr.expr, &self.u, &self.links)
                || is_link_alias(&let_expr.expr, &self.links))
        {
            self.note_link_pat(&let_expr.pat);
        }
        syn::visit::visit_expr_if(self, node);
    }

    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        if let Expr::Let(let_expr) = node.cond.as_ref()
            && (mentions_entity_member(&let_expr.expr, &self.u, &self.links)
                || is_link_alias(&let_expr.expr, &self.links))
        {
            self.note_link_pat(&let_expr.pat);
        }
        syn::visit::visit_expr_while(self, node);
    }

    fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
        // `match u.field { Some(t) => … }`: every arm pattern binds links.
        if mentions_entity_member(&node.expr, &self.u, &self.links)
            || is_link_alias(&node.expr, &self.links)
        {
            for arm in &node.arms {
                self.note_link_pat(&arm.pat);
            }
        }
        syn::visit::visit_expr_match(self, node);
    }

    fn visit_expr(&mut self, node: &'ast syn::Expr) {
        match node {
            Expr::Call(call) => {
                // Whole-`u` passed to a free function escapes the kernel:
                // field accesses would hide from access derivation.
                for arg in &call.args {
                    let mut inner = arg;
                    while let Expr::Reference(r) = inner {
                        inner = &r.expr;
                    }
                    if let Expr::Path(p) = inner
                        && p.qself.is_none()
                        && p.path.segments.len() == 1
                        && p.path.segments[0].ident == self.u
                    {
                        self.errors.push(syn::Error::new_spanned(
                            arg,
                            "smart_system: passing whole `u` to a function hides field \
                             accesses — pass `u.field` values instead",
                        ));
                    }
                }
            }
            Expr::Return(ret) => {
                if ret.expr.is_some() {
                    self.errors.push(syn::Error::new_spanned(
                        node,
                        "smart_system: `return <value>` is not allowed — bare `return` \
                         skips the current entity",
                    ));
                }
            }
            Expr::Assign(assign) => self.check_foreign_write(&assign.left, node),
            Expr::Binary(bin) => {
                if matches!(
                    bin.op,
                    syn::BinOp::AddAssign(_)
                        | syn::BinOp::SubAssign(_)
                        | syn::BinOp::MulAssign(_)
                        | syn::BinOp::DivAssign(_)
                        | syn::BinOp::RemAssign(_)
                        | syn::BinOp::BitAndAssign(_)
                        | syn::BinOp::BitOrAssign(_)
                        | syn::BinOp::BitXorAssign(_)
                        | syn::BinOp::ShlAssign(_)
                        | syn::BinOp::ShrAssign(_)
                ) {
                    self.check_foreign_write(&bin.left, node);
                }
            }
            _ => {}
        }
        // A closure argument of a call rooted at `u.field`/link binds links.
        if let Expr::MethodCall(call) = node
            && (mentions_entity_member(&call.receiver, &self.u, &self.links)
                || is_link_alias(&call.receiver, &self.links))
        {
            for arg in &call.args {
                if let Expr::Closure(closure) = arg {
                    for input in &closure.inputs {
                        self.note_link_pat(input);
                    }
                }
            }
        }
        syn::visit::visit_expr(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        // `events.send(…)` / `events.clear()` mark the `Events` param as
        // producing (scheduler-visible write).
        if (node.method == "send" || node.method == "clear")
            && let Expr::Path(base) = node.receiver.as_ref()
            && base.qself.is_none()
            && base.path.segments.len() == 1
        {
            let name = base.path.segments[0].ident.to_string();
            if self.sends.contains_key(&name) {
                self.sends.insert(name, true);
            }
        }
        syn::visit::visit_expr_method_call(self, node);
    }
}

impl BodyInfo {
    fn check_foreign_write(&mut self, lhs: &Expr, node: &Expr) {
        if let Some(root) = member_root(lhs)
            && self.links.contains(&root)
        {
            self.errors.push(syn::Error::new_spanned(
                node,
                "smart_system: cannot write another entity's component — \
                 `t.field` is read-only (gather); send an event via `Events<E>` instead",
            ));
        }
    }
}

/// Rewrites foreign reads `t.field` into a pack gather through the store
/// handle. Descends everywhere, including closures.
struct ForeignRewrite<'a> {
    pack_ty: &'a Type,
    links: &'a HashSet<String>,
    store_ident: Ident,
}

impl VisitMut for ForeignRewrite<'_> {
    fn visit_expr_field_mut(&mut self, node: &mut syn::ExprField) {
        // Recurse first so nested `t.a.b` rewrites inside-out.
        syn::visit_mut::visit_expr_field_mut(self, node);
        if let Expr::Path(base) = node.base.as_ref()
            && base.qself.is_none()
            && base.path.segments.len() == 1
            && self
                .links
                .contains(&base.path.segments[0].ident.to_string())
        {
            let link = &base.path.segments[0].ident;
            let field = &node.member;
            let pack_ty = self.pack_ty;
            let store = &self.store_ident;
            let replacement: Expr = syn::parse_quote! {
                <#pack_ty as ornis_core::Pack>::pack_get(#store, #link)
                    .expect("smart_system: related entity is missing the pack bundle")
                    .#field
            };
            *node = match replacement {
                Expr::Field(f) => f,
                _ => unreachable!("quoted replacement is a field access"),
            };
        }
    }
}

fn camel_system_name(fn_name: &Ident) -> Ident {
    let name = fn_name.to_string();
    let mut camel = String::with_capacity(name.len() + 6);
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

fn expand(func: ItemFn) -> Result<TokenStream2, syn::Error> {
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

    let mut inputs = func.sig.inputs.iter();
    let first = inputs.next().unwrap();
    let (u_ident, pack_ty, is_mut) = parse_first_param(first)?;

    let mut params: Vec<TypedParam> = Vec::new();
    for arg in inputs {
        params.push(parse_param(arg)?);
    }

    for param in &params {
        if param.name.to_string().starts_with(RESERVED_PREFIX) {
            return Err(syn::Error::new_spanned(
                &param.name,
                "smart_system: parameter names starting with `__smart_` are reserved",
            ));
        }
    }

    // Pre-pass: links, sends, ban diagnostics.
    let mut info = BodyInfo {
        u: u_ident.to_string(),
        links: HashSet::new(),
        sends: params
            .iter()
            .filter(|p| matches!(p.kind, ParamKind::Events))
            .map(|p| (p.name.to_string(), false))
            .collect(),
        errors: Vec::new(),
    };
    info.visit_block(&func.block);
    if !info.errors.is_empty() {
        return Err(combine_errors(info.errors));
    }

    // Rewrite foreign reads (everywhere incl. closures). Bare `return`
    // needs no rewrite: the body is a separate function called once per
    // entity, so returning from it naturally skips to the next entity.
    let store_ident = format_ident!("__smart_store");
    let mut foreign = ForeignRewrite {
        pack_ty: &pack_ty,
        links: &info.links,
        store_ident: store_ident.clone(),
    };
    let mut rewritten_stmts = func.block.stmts.clone();
    for stmt in &mut rewritten_stmts {
        foreign.visit_stmt_mut(stmt);
    }

    // Body function: same params with elided lifetimes on wrappers.
    let body_fn_name = format_ident!("__smart_system_body_{}", func.sig.ident);
    let mut body_inputs = Vec::new();
    body_inputs.push({
        let u = &u_ident;
        if is_mut {
            quote! { #u: &mut #pack_ty }
        } else {
            quote! { #u: &#pack_ty }
        }
    });
    for param in &params {
        let name = &param.name;
        match &param.kind {
            ParamKind::Config => {
                let ty = &param.inner;
                body_inputs.push(quote! { #name: #ty });
            }
            ParamKind::EntityHandle => {
                body_inputs.push(quote! { #name: ornis_core::Entity });
            }
            ParamKind::Res => {
                let ty = with_elided_lifetime(param.wrapper.clone());
                body_inputs.push(quote! { #name: #ty });
            }
            ParamKind::ResMut => {
                let ty = with_elided_lifetime(param.wrapper.clone());
                body_inputs.push(quote! { #name: #ty });
            }
            ParamKind::Events => {
                let ty = with_elided_lifetime(param.wrapper.clone());
                body_inputs.push(quote! { #name: #ty });
            }
        }
    }

    // Access contributions per parameter.
    let mut access_stmts = Vec::new();
    for param in &params {
        if matches!(param.kind, ParamKind::Config | ParamKind::EntityHandle) {
            continue;
        }
        let inner = &param.inner;
        match &param.kind {
            ParamKind::Res => {
                access_stmts.push(quote! {
                    __access = __access.combine(ornis_core::Res::<#inner>::access());
                });
            }
            ParamKind::ResMut => {
                access_stmts.push(quote! {
                    __access = __access.combine(ornis_core::ResMut::<#inner>::access());
                });
            }
            ParamKind::Events => {
                let sends = info
                    .sends
                    .get(&param.name.to_string())
                    .copied()
                    .unwrap_or(false);
                access_stmts.push(quote! {
                    __access = __access.combine(ornis_core::Events::<#inner>::access(#sends));
                });
            }
            ParamKind::Config | ParamKind::EntityHandle => unreachable!(),
        }
    }

    // Per-entity fetches + call args.
    let mut fetch_stmts = Vec::new();
    let mut call_args = Vec::new();
    for param in &params {
        let name = &param.name;
        match &param.kind {
            ParamKind::Config => {
                call_args.push(quote! { #name.clone() });
            }
            ParamKind::EntityHandle => {
                call_args.push(quote! { __entity });
            }
            ParamKind::Res => {
                let inner = &param.inner;
                fetch_stmts.push(quote! {
                    let #name = ornis_core::Res::<#inner>::fetch(__resources);
                });
                call_args.push(quote! { #name });
            }
            ParamKind::ResMut => {
                let inner = &param.inner;
                fetch_stmts.push(quote! {
                    let #name = ornis_core::ResMut::<#inner>::fetch(__resources);
                });
                call_args.push(quote! { #name });
            }
            ParamKind::Events => {
                let inner = &param.inner;
                fetch_stmts.push(quote! {
                    let #name = ornis_core::Events::<#inner>::fetch(__resources);
                });
                call_args.push(quote! { #name });
            }
        }
    }
    // Config bindings come from the struct (cloned once per run).
    let mut config_bindings = Vec::new();
    for param in &params {
        if matches!(param.kind, ParamKind::Config) {
            let name = &param.name;
            config_bindings.push(quote! {
                let #name = self.#name.clone();
            });
        }
    }

    let struct_name = camel_system_name(&func.sig.ident);
    let fn_name_str = func.sig.ident.to_string();
    let attrs = func.attrs.clone();

    let (struct_fields, ctor_params, ctor_inits) = config_struct_parts(&params);
    let u_binding = if is_mut {
        quote! { mut __u }
    } else {
        quote! { __u }
    };
    let u_ref = if is_mut {
        quote! { &mut __u }
    } else {
        quote! { &__u }
    };
    let scatter = if is_mut {
        quote! {
            ornis_core::Pack::pack_put(&__u, #store_ident, __entity);
        }
    } else {
        quote! {}
    };

    Ok(quote! {
        #[allow(unused_variables)]
        fn #body_fn_name(#(#body_inputs),*) {
            #(#rewritten_stmts)*
        }

        #(#attrs)*
        #[allow(non_camel_case_types)]
        pub struct #struct_name {
            #(#struct_fields)*
        }

        impl #struct_name {
            /// Creates the system; `#[config]` parameters become constructor arguments.
            pub fn new(#(#ctor_params),*) -> Self {
                Self {
                    #(#ctor_inits)*
                }
            }
        }

        impl ornis_core::System for #struct_name {
            fn name(&self) -> &'static str {
                #fn_name_str
            }

            fn access(&self) -> ornis_core::SystemAccess {
                let mut __access =
                    ornis_core::SystemAccess::new().reads::<ornis_core::SmartStore>();
                for __lane in <#pack_ty as ornis_core::Pack>::pack_lane_ids() {
                    __access = if #is_mut {
                        __access.writes_lane_id(__lane)
                    } else {
                        __access.reads_lane_id(__lane)
                    };
                }
                #(#access_stmts)*
                __access
            }

            fn run(&self, __resources: &ornis_core::Resources) {
                let #store_ident: &ornis_core::SmartStore = __resources
                    .get::<ornis_core::SmartStore>()
                    .expect(concat!("smart_system `", #fn_name_str, "`: missing SmartStore resource"));
                #(#config_bindings)*
                for __entity in <#pack_ty as ornis_core::Pack>::pack_entities(#store_ident) {
                    #(#fetch_stmts)*
                    let Some(#u_binding) =
                        <#pack_ty as ornis_core::Pack>::pack_get(#store_ident, __entity)
                    else {
                        continue;
                    };
                    #body_fn_name(#u_ref, #(#call_args),*);
                    #scatter
                }
            }
        }
    })
}

fn config_struct_parts(
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

fn parse_first_param(arg: &FnArg) -> Result<(Ident, Type, bool), syn::Error> {
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

fn parse_param(arg: &FnArg) -> Result<TypedParam, syn::Error> {
    let typed = match arg {
        FnArg::Receiver(r) => {
            return Err(syn::Error::new_spanned(
                r,
                "smart_system: unexpected `self` — only the first parameter is the entity, \
                 the rest are `Res<T>` / `ResMut<T>` / `Events<E>` / `#[config]`",
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
        return Ok(TypedParam {
            name,
            kind: ParamKind::Config,
            inner: (*typed.ty).clone(),
            wrapper: (*typed.ty).clone(),
        });
    }
    let is_entity = typed.attrs.iter().any(|a| a.path().is_ident("entity"));
    if is_entity {
        if path_last_ident(&typed.ty).as_deref() != Some("Entity") {
            return Err(syn::Error::new_spanned(
                &typed.ty,
                "smart_system: `#[entity]` parameters must have type `Entity`",
            ));
        }
        return Ok(TypedParam {
            name,
            kind: ParamKind::EntityHandle,
            inner: (*typed.ty).clone(),
            wrapper: (*typed.ty).clone(),
        });
    }
    match typed.ty.as_ref() {
        Type::Reference(_) => Err(syn::Error::new_spanned(
            &typed.ty,
            "smart_system: raw `&Resources` is forbidden — declare typed parameters \
             (`Res<T>`, `ResMut<T>`, `Events<E>`) so access follows from types",
        )),
        Type::Path(tp) => {
            let Some(seg) = tp.path.segments.last() else {
                return Err(syn::Error::new_spanned(
                    &typed.ty,
                    "smart_system: parameters must be `Res<T>` / `ResMut<T>` / `Events<E>` \
                     (or `#[config] name: T`)",
                ));
            };
            let Some(kind) = is_param_wrapper(seg) else {
                if seg.ident == "SmartStore" {
                    return Err(syn::Error::new_spanned(
                        &typed.ty,
                        "smart_system: raw `SmartStore` is forbidden — the first `&T`/`&mut T` \
                         (`T: Pack`) parameter already selects the entity set",
                    ));
                }
                return Err(syn::Error::new_spanned(
                    &typed.ty,
                    "smart_system: parameters must be `Res<T>` / `ResMut<T>` / `Events<E>` \
                     (or `#[config] name: T`)",
                ));
            };
            let PathArguments::AngleBracketed(args) = &seg.arguments else {
                return Err(syn::Error::new_spanned(
                    &typed.ty,
                    "smart_system: wrapper needs its inner type (`Res<FixedTime>`)",
                ));
            };
            if args.args.len() != 1 {
                return Err(syn::Error::new_spanned(
                    &typed.ty,
                    "smart_system: wrapper takes exactly one type (`Res<FixedTime>`)",
                ));
            }
            let GenericArgument::Type(inner) = &args.args[0] else {
                return Err(syn::Error::new_spanned(
                    &typed.ty,
                    "smart_system: wrapper takes exactly one type (`Res<FixedTime>`)",
                ));
            };
            if is_smart_store_ty(inner) {
                return Err(syn::Error::new_spanned(
                    inner,
                    "smart_system: `Res<SmartStore>` is forbidden — reach components through \
                     the first `&T`/`&mut T` (`T: Pack`) parameter",
                ));
            }
            if matches!(kind, ParamKind::Res) {
                check_plain_shape(inner)?;
            }
            Ok(TypedParam {
                name,
                kind,
                inner: inner.clone(),
                wrapper: (*typed.ty).clone(),
            })
        }
        _ => Err(syn::Error::new_spanned(
            &typed.ty,
            "smart_system: parameters must be `Res<T>` / `ResMut<T>` / `Events<E>` \
             (or `#[config] name: T`)",
        )),
    }
}

/// Early syntactic backstop for the `PlainResource` bound: well-known
/// interior-mutability types are rejected with one clean diagnostic
/// instead of a cascade of unsatisfied-bound errors (custom interior
/// types are still caught by the `Res<T: PlainResource>` bound itself).
fn check_plain_shape(inner: &Type) -> Result<(), syn::Error> {
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

fn combine_errors(errors: Vec<syn::Error>) -> syn::Error {
    let mut iter = errors.into_iter();
    let mut combined = iter.next().expect("at least one error");
    for err in iter {
        combined.combine(err);
    }
    combined
}
