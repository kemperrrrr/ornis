//! Body pre-pass and rewriting for `#[smart_system]` (IDEAS §32).
//!
//! [`BodyInfo`] walks one system body collecting entity-link bindings
//! and syntactic ban diagnostics, while
//! [`ForeignRewrite`] turns foreign reads (`t.field`) into pack gathers.
//! The visitor overrides couple guard checks with descent bookkeeping by
//! construction (same as the baselined
//! `pipeline_config::TypeAnalyzer::visit_type` and
//! `gpu_pipeline::parse_binding_group`); each carries a targeted
//! `qual:allow(iosp)` pin that re-fires if logic or calls grow.

use std::collections::HashSet;

use syn::visit::Visit;
use syn::{Expr, Ident, Pat};

use crate::smart_system_body_links::{
    is_link_alias, is_link_alias_shape, member_root, mentions_entity_member, pat_idents,
};

/// Pre-pass over the body: link bindings (handles to *other* entities)
/// and syntactic ban diagnostics.
pub(crate) struct BodyInfo {
    pub(crate) u: String,
    pub(crate) links: HashSet<String>,
    pub(crate) errors: Vec<syn::Error>,
}

impl BodyInfo {
    fn note_link_pat(&mut self, pat: &Pat) {
        let mut idents = Vec::new();
        pat_idents(pat, &mut idents);
        self.links.extend(idents);
    }

    /// `let PAT = INIT`: alias-of-link extends the link set.
    // qual:allow(iosp) reason: "link-binding rule coupling alias-shape checks with set insertion — same shape as baselined gpu_pipeline::parse_binding_group"
    pub(crate) fn visit_let(&mut self, pat: &Pat, init: &Expr) {
        if is_link_alias(init, &self.links)
            || mentions_entity_member(init, &self.u, &self.links) && is_link_alias_shape(init)
        {
            self.note_link_pat(pat);
        }
    }
}
impl<'ast> Visit<'ast> for BodyInfo {
    // qual:allow(iosp) reason: "let-binding dispatch coupling an init-shape check with the alias rule — same shape as baselined gpu_pipeline::parse_binding_group"
    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let Some(init) = &node.init {
            // `let PAT = INIT` (+ optional `else` divergent — links in the
            // `else` branch never escape, but marking is harmless).
            self.visit_let(&node.pat, &init.expr);
        }
        syn::visit::visit_local(self, node);
    }

    // qual:allow(iosp) reason: "if-let binder coupling scrutinee-shape checks with pattern registration — same shape as baselined gpu_pipeline::parse_binding_group"
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

    // qual:allow(iosp) reason: "while-let binder coupling scrutinee-shape checks with pattern registration — same shape as baselined gpu_pipeline::parse_binding_group"
    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        if let Expr::Let(let_expr) = node.cond.as_ref()
            && (mentions_entity_member(&let_expr.expr, &self.u, &self.links)
                || is_link_alias(&let_expr.expr, &self.links))
        {
            self.note_link_pat(&let_expr.pat);
        }
        syn::visit::visit_expr_while(self, node);
    }

    // qual:allow(iosp) reason: "match binder coupling scrutinee-shape checks with arm registration — same shape as baselined gpu_pipeline::parse_binding_group"
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

    // qual:allow(iosp) reason: "ban diagnostics coupling whole-arg scans with error emission — same shape as baselined gpu_pipeline::parse_binding_base_ty"
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
}

impl BodyInfo {
    // qual:allow(iosp) reason: "write-target rule coupling root resolution with diagnostic emission — same shape as baselined pipeline_config::TypeAnalyzer::visit_type"
    fn check_foreign_write(&mut self, lhs: &Expr, node: &Expr) {
        if let Some(root) = member_root(lhs)
            && self.links.contains(&root)
        {
            self.errors.push(syn::Error::new_spanned(
                node,
                "smart_system: cannot write another entity's component — \
                 `t.field` is read-only (gather); send an event via `EventWriter<E>` instead",
            ));
        }
    }
}

impl BodyInfo {
    /// Collects link bindings and an empty error sink for one body.
    /// Straight-line construction (trait/default calls only).
    pub(crate) fn collect(u: &Ident) -> Self {
        Self {
            u: u.to_string(),
            links: HashSet::new(),
            errors: Vec::new(),
        }
    }
}
