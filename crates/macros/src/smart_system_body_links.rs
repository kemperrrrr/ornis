//! Link-shape analysis for `#[smart_system]` bodies (IDEAS §32).
//!
//! Free-function predicates over `syn` AST shapes: member-chain roots,
//! link aliases, and pattern bindings. The recursive helpers mix match
//! logic with self calls by construction (same as the baselined
//! `smart_pipeline::collect_pat_idents`); each carries a targeted
//! `qual:allow(iosp)` pin that re-fires if logic or calls grow.

use std::collections::HashSet;

use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{Expr, Ident, Pat, Type};

/// Root ident of a `a.b.c` member chain, when the base is a plain ident.
// qual:allow(iosp) reason: "recursive member-chain walk mixing match logic with a self call — same shape as baselined smart_pipeline::collect_pat_idents"
pub(crate) fn member_root(expr: &Expr) -> Option<String> {
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
    // qual:allow(iosp) reason: "syn visitor override coupling a guard check with descent bookkeeping — same shape as baselined pipeline_config::TypeAnalyzer::visit_type"
    fn visit_expr_field(&mut self, node: &'ast syn::ExprField) {
        if let Some(root) = member_root(&node.base)
            && (root == self.u || self.links.contains(&root))
        {
            self.found = true;
        }
        syn::visit::visit_expr_field(self, node);
    }
}

pub(crate) fn mentions_entity_member(expr: &Expr, u: &str, links: &HashSet<String>) -> bool {
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
// qual:allow(iosp) reason: "recursive shape predicate mixing match logic with self calls — same shape as baselined smart_pipeline::collect_pat_idents"
pub(crate) fn is_link_alias(expr: &Expr, links: &HashSet<String>) -> bool {
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

// qual:allow(iosp) reason: "recursive pattern walk mixing match logic with self calls — same shape as baselined smart_pipeline::collect_pat_idents"
pub(crate) fn pat_idents(pat: &Pat, out: &mut Vec<String>) {
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
/// `let x = EXPR` shapes that preserve the entity handle (not computed
/// values): a bare path, a reference, or `.clone()`/`.copied()`.
pub(crate) fn is_link_alias_shape(expr: &Expr) -> bool {
    match expr {
        Expr::Path(_) | Expr::Reference(_) => true,
        Expr::MethodCall(m) => {
            matches!(m.method.to_string().as_str(), "clone" | "copied")
        }
        _ => false,
    }
}

/// Rewrites foreign reads `t.field` into a pack gather through the store
/// handle. Descends everywhere, including closures.
pub(crate) struct ForeignRewrite<'a> {
    pub(crate) pack_ty: &'a Type,
    pub(crate) links: &'a HashSet<String>,
    pub(crate) store_ident: Ident,
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
                // Unreachable by construction (the quote is a field access);
                // return instead of panicking so error-handling stays clean.
                _ => return,
            };
        }
    }
}
