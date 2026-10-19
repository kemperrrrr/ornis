//! Field-level usage of a `#[smart_system]` body (IDEAS §32, v2b).
//!
//! The body is scanned for `u.field` (own fields) and `t.field` (fields of
//! linked entities, see [`crate::smart_system_body_links`]). Only the lanes
//! the body touches are declared, and only the written ones as writes.
//!
//! A field counts as written when it is an assignment or compound-assignment
//! target, a `&mut` operand, or a method receiver. The receiver rule is
//! conservative: the macro cannot tell `&self` from `&mut self`. Over-counting
//! writes only costs parallelism. Under-counting is a compile error, because
//! a read field reaches the body as `&T`, so a write through it does not
//! type-check.

use std::collections::{BTreeSet, HashSet};

use proc_macro2::Span;
use quote::format_ident;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{BinOp, Block, Expr, ExprField, ExprPath, Member, Type, UnOp};

/// Own and foreign field usage of one body, keyed by field name.
#[derive(Default)]
pub(crate) struct FieldUsage {
    /// Own fields read or written (`u.f`).
    pub(crate) own: BTreeSet<String>,
    /// Own fields written (assignment, `&mut`, method receiver).
    pub(crate) written: BTreeSet<String>,
    /// Fields read through a link binding (`t.f`).
    pub(crate) foreign: BTreeSet<String>,
    /// First use of `u` that is not a field access (`u`, `u.method()`).
    pub(crate) bare: Option<Span>,
}

/// Scans `block` for field usage of the entity parameter `u` and the link
/// bindings `links`.
pub(crate) fn scan_fields(block: &Block, u: &str, links: &HashSet<String>) -> FieldUsage {
    let mut scan = FieldScan {
        u,
        links,
        usage: FieldUsage::default(),
    };
    scan.visit_block(block);
    scan.usage
}

/// Compound-assignment operators (`+=`, `<<=`, …).
fn is_compound_assign(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::AddAssign(_)
            | BinOp::SubAssign(_)
            | BinOp::MulAssign(_)
            | BinOp::DivAssign(_)
            | BinOp::RemAssign(_)
            | BinOp::BitAndAssign(_)
            | BinOp::BitOrAssign(_)
            | BinOp::BitXorAssign(_)
            | BinOp::ShlAssign(_)
            | BinOp::ShrAssign(_)
    )
}

/// Name of a named member (`.field`); tuple members yield `None`.
fn member_name(member: &Member) -> Option<String> {
    match member {
        Member::Named(ident) => Some(ident.to_string()),
        Member::Unnamed(_) => None,
    }
}

/// The single-segment identifier `expr` names, if it is one.
fn ident_of(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(path) if path.qself.is_none() => path.path.get_ident().map(|i| i.to_string()),
        _ => None,
    }
}

struct FieldScan<'a> {
    u: &'a str,
    links: &'a HashSet<String>,
    usage: FieldUsage,
}

impl FieldScan<'_> {
    /// Own field a place expression is rooted at (`u.f`, `u.f.x`, `u.f[i]`,
    /// `(*u.f)`), or `None` when the root is not `u`.
    fn own_root(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Field(field) if ident_of(&field.base).as_deref() == Some(self.u) => {
                member_name(&field.member)
            }
            Expr::Field(field) => self.own_root(&field.base),
            Expr::Index(index) => self.own_root(&index.expr),
            Expr::Paren(paren) => self.own_root(&paren.expr),
            Expr::Unary(unary) if matches!(unary.op, UnOp::Deref(_)) => self.own_root(&unary.expr),
            _ => None,
        }
    }

    fn mark_written(&mut self, expr: &Expr) {
        if let Some(field) = self.own_root(expr) {
            self.usage.own.insert(field.clone());
            self.usage.written.insert(field);
        }
    }
}

impl<'ast> Visit<'ast> for FieldScan<'_> {
    fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {
        self.mark_written(&node.left);
        syn::visit::visit_expr_assign(self, node);
    }

    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        if is_compound_assign(&node.op) {
            self.mark_written(&node.left);
        }
        syn::visit::visit_expr_binary(self, node);
    }

    fn visit_expr_reference(&mut self, node: &'ast syn::ExprReference) {
        if node.mutability.is_some() {
            self.mark_written(&node.expr);
        }
        syn::visit::visit_expr_reference(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        self.mark_written(&node.receiver);
        syn::visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_field(&mut self, node: &'ast ExprField) {
        if let (Some(base), Some(field)) = (ident_of(&node.base), member_name(&node.member)) {
            if base == self.u {
                self.usage.own.insert(field);
                return;
            }
            if self.links.contains(&base) {
                self.usage.foreign.insert(field);
            }
        }
        syn::visit::visit_expr_field(self, node);
    }

    fn visit_expr_path(&mut self, node: &'ast ExprPath) {
        if node.qself.is_none() && node.path.is_ident(self.u) && self.usage.bare.is_none() {
            self.usage.bare = Some(node.span());
        }
        syn::visit::visit_expr_path(self, node);
    }
}

/// `ornis_core::Pack` lane type for `field` of the bundle at `pack_path`:
/// the wrapper `Struct__field__PackLane` next to the struct.
pub(crate) fn lane_type(pack_path: &syn::Path, field: &str) -> Type {
    let mut path = pack_path.clone();
    if let Some(last) = path.segments.last_mut() {
        last.ident = format_ident!("{}__{}__PackLane", last.ident, field);
        last.arguments = syn::PathArguments::None;
    }
    syn::parse_quote!(#path)
}

/// Rewrites field access in the body: `u.f` becomes a dereference of the
/// per-field parameter, and `t.f` becomes a gather from the foreign lane.
/// Inside-out, so chained accesses (`t.a.b`) rewrite their base first.
pub(crate) struct FieldRewrite<'a> {
    pub(crate) u: &'a str,
    pub(crate) pack_path: &'a syn::Path,
    pub(crate) links: &'a HashSet<String>,
}

impl FieldRewrite<'_> {
    fn replacement(&self, field: &ExprField) -> Option<Expr> {
        let base = ident_of(&field.base)?;
        let name = member_name(&field.member)?;
        if base == self.u {
            let local = format_ident!("__smart_u_{}", name, span = Span::mixed_site());
            return Some(syn::parse_quote!((*#local)));
        }
        if !self.links.contains(&base) {
            return None;
        }
        let link = format_ident!("{}", base);
        let lane = lane_type(self.pack_path, &name);
        let store = format_ident!("__smart_lane_{}", name, span = Span::mixed_site());
        let message = format!("smart_system: related entity is missing the `{name}` lane");
        Some(syn::parse_quote!(
            (<#lane as ornis_core::PackLane>::gather(#store, #store.index_of(#link))
                .expect(#message))
        ))
    }
}

impl VisitMut for FieldRewrite<'_> {
    fn visit_expr_mut(&mut self, node: &mut Expr) {
        syn::visit_mut::visit_expr_mut(self, node);
        let Expr::Field(field) = node else {
            return;
        };
        if let Some(replacement) = self.replacement(field) {
            *node = replacement;
        }
    }
}
