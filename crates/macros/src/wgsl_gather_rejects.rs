//! Spanned rejections for illegal `Entity` uses (issue #110): writes,
//! method calls, references, and handle equality. Split out of the
//! scanner so each reporter stays a branchy Operation that mixes with
//! no own calls (rustqual IOSP).

use syn::Expr;
use syn::spanned::Spanned;

use crate::wgsl_entity_shape::{PASSTHROUGH_MEMBERS, field_shape_of, root_ident, strip_parens};
use crate::wgsl_gather_scan::Scanner;

impl Scanner<'_> {
    /// Integration: resolve the written expression's root and field
    /// shape, then hand both to the pure reporter.
    pub(crate) fn reject_entity_write(&mut self, left: &Expr) {
        self.reject_entity_write_shape(left, root_ident(left), field_shape_of(left));
    }

    /// Operation: pure report over the precomputed write shape — the
    /// branchy half never mixes with own calls (IOSP).
    pub(crate) fn reject_entity_write_shape(
        &mut self,
        left: &Expr,
        root: Option<String>,
        field: Option<(String, String)>,
    ) {
        let Some(root) = root.filter(|r| self.plan.entity_names.contains(r)) else {
            return;
        };
        let what = match &field {
            Some((_, member)) => format!("`{root}.{member}`"),
            None => format!("`{root}`"),
        };
        if let Some((_, member)) = &field {
            self.errored.insert((root.clone(), member.clone()));
        }
        self.plan.errors.push(syn::Error::new(
            left.span(),
            format!(
                "{what} is read-only in GPU kernel `{}` (Entity gather); foreign \
                 fields have no WGSL write spelling — mutate through a system on the \
                 CPU side",
                self.fn_name
            ),
        ));
    }

    /// Operation: pure report of a method call on an entity handle or
    /// one of its non-passthrough fields (`t.index.method()` names the
    /// `u32` field, not the handle).
    pub(crate) fn reject_entity_method(
        &mut self,
        node: &syn::ExprMethodCall,
        root: Option<String>,
        field: Option<(String, String)>,
    ) {
        let root = root.filter(|r| self.plan.entity_names.contains(r));
        let hit = match (&root, &field) {
            (Some(_), None) => true,
            (Some(_), Some((_, member))) => !PASSTHROUGH_MEMBERS.contains(&member.as_str()),
            (None, _) => false,
        };
        if hit {
            let root = root.unwrap_or_default();
            self.plan.errors.push(syn::Error::new(
                node.span(),
                format!(
                    "method calls on `{root}` have no WGSL spelling in GPU kernel \
                     `{}`; entity fields are read-only gathers",
                    self.fn_name
                ),
            ));
        }
    }

    /// Operation: pure report of a reference to an entity handle.
    pub(crate) fn reject_entity_reference(
        &mut self,
        node: &syn::ExprReference,
        root: Option<String>,
    ) {
        let Some(root) = root.filter(|r| self.plan.entity_names.contains(r)) else {
            return;
        };
        self.plan.errors.push(syn::Error::new(
            node.span(),
            format!(
                "references to `{root}` have no WGSL spelling in GPU kernel `{}`; \
                 `Entity` handles are copied values",
                self.fn_name
            ),
        ));
    }

    /// Operation: `t == t2` on the handle values has no WGSL spelling
    /// (struct equality); `t.index == 0` compares `u32`s and stays
    /// allowed.
    pub(crate) fn reject_entity_equality(&mut self, node: &syn::ExprBinary) {
        let operand_is_entity_value = |expr: &Expr| {
            let inner = strip_parens(expr);
            matches!(inner, Expr::Path(_))
                && root_ident(inner).is_some_and(|root| self.plan.entity_names.contains(&root))
        };
        if operand_is_entity_value(&node.left) || operand_is_entity_value(&node.right) {
            self.plan.errors.push(syn::Error::new(
                node.span(),
                format!(
                    "Entity `==`/`!=` has no WGSL spelling in GPU kernel `{}`; \
                     compare `.index` (and `.generation`) explicitly",
                    self.fn_name
                ),
            ));
        }
    }
}
