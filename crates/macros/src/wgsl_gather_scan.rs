//! The entity-gather scanner (issue #110): walks one `#[kernel]`
//! `eval` body, validates every `Entity` use position, and records
//! the gather lanes in a [`GatherPlan`].

use std::collections::BTreeSet;

use ornis_shader_lang::ir::IrType;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{Expr, ExprField, FnArg, ItemFn, Local, Pat, PatType, Type};

use crate::wgsl_entity_shape::{
    LaneReject, entity_field_shape, field_shape_of, is_passthrough, lane_elem, lane_elem_of_ir,
    root_ident, strip_parens,
};
use crate::wgsl_gather::{GatherPlan, Lane};

/// Scan one kernel: validate `Entity` usage and record its lanes.
pub(crate) fn scan(func: &ItemFn) -> GatherPlan {
    let fn_name = func.sig.ident.to_string();
    let (return_ty, return_syn_ty) = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => (crate::wgsl::rust_ty_to_ir(ty), Some((**ty).clone())),
        syn::ReturnType::Default => (None, None),
    };
    // Even a `-> Option<Entity> { None }` body references the structs
    // through its return type's generated declaration.
    let return_mentions_entity = return_ty.as_ref().is_some_and(|ty| match ty {
        IrType::Entity => true,
        IrType::Optional(inner) => matches!(**inner, IrType::Entity),
        _ => false,
    });
    let mut scanner = Scanner {
        plan: GatherPlan::default(),
        fn_name: &fn_name,
        return_ty,
        return_syn_ty,
        option_entity_names: BTreeSet::new(),
        errored: BTreeSet::new(),
    };
    if return_mentions_entity {
        scanner.plan.mentions_entity = true;
    }
    for arg in &func.sig.inputs {
        if let FnArg::Typed(pat_ty) = arg {
            scanner.register_pat_type(pat_ty);
        }
    }
    // The function tail (`fn f(t: Entity) -> Vec3 { t.position }`) is
    // the one un-`let` position a gather may sit in; typed by the
    // return type. Without a lane-storable return type the field visit
    // below reports the generic bind-first error instead.
    if let Some(syn::Stmt::Expr(expr, None)) = func.block.stmts.last()
        && let Expr::Field(field) = expr
        && let Some((root, member)) = entity_field_shape(field)
        && !is_passthrough(&member)
        && scanner.plan.entity_names.contains(&root)
    {
        scanner.take_return_read(&root, &member);
    }
    scanner.visit_block(&func.block);
    scanner.plan
}

pub(crate) struct Scanner<'a> {
    pub(crate) plan: GatherPlan,
    pub(crate) fn_name: &'a str,
    pub(crate) return_ty: Option<IrType>,
    pub(crate) return_syn_ty: Option<Type>,
    /// Scrutinees typed `Option<Entity>`: `Some` payloads of a match
    /// or `if let` over these bind entity values.
    pub(crate) option_entity_names: BTreeSet<String>,
    /// Sites that already reported their own error — suppresses the
    /// generic position error for the same (binding, field).
    pub(crate) errored: BTreeSet<(String, String)>,
}

impl Scanner<'_> {
    pub(crate) fn register_pat_type(&mut self, pat_ty: &PatType) {
        let Pat::Ident(pi) = pat_ty.pat.as_ref() else {
            return;
        };
        let name = pi.ident.to_string();
        if crate::kernel_types::is_entity_type(&pat_ty.ty) {
            self.plan.entity_names.insert(name);
            self.plan.mentions_entity = true;
        } else if crate::wgsl_options::option_ir_ty(&pat_ty.ty) == Some(IrType::Entity) {
            self.option_entity_names.insert(name);
            self.plan.mentions_entity = true;
        }
    }

    /// `Some(e)` / plain-ident pattern bindings of a match or `if let`
    /// over an entity-typed scrutinee become entity bindings.
    pub(crate) fn register_scrutinee_bindings<'p>(
        &mut self,
        scrutinee: Option<&Expr>,
        pats: impl Iterator<Item = &'p Pat>,
    ) {
        let Some(root) = scrutinee.and_then(root_ident) else {
            return;
        };
        if !(self.plan.entity_names.contains(&root) || self.option_entity_names.contains(&root)) {
            return;
        }
        for pat in pats {
            let ident = match pat {
                Pat::TupleStruct(ts)
                    if ts.path.segments.last().is_some_and(|s| s.ident == "Some") =>
                {
                    match ts.elems.first() {
                        Some(Pat::Ident(pi)) => Some(&pi.ident),
                        _ => None,
                    }
                }
                Pat::Ident(pi)
                    if pi.ident != "None"
                        && pi.ident != "Some"
                        && pi.subpat.is_none()
                        && pi.by_ref.is_none() =>
                {
                    Some(&pi.ident)
                }
                _ => None,
            };
            if let Some(ident) = ident {
                self.plan.entity_names.insert(ident.to_string());
                self.plan.mentions_entity = true;
            }
        }
    }

    /// `let x: T = t.field;` — the annotation is the lane's element
    /// type. No annotation (or a non-storable one) falls through to
    /// the generic position error from the field visit.
    pub(crate) fn check_bound_read(&mut self, local: &Local) {
        let Some(init) = &local.init else {
            return;
        };
        let Expr::Field(field) = init.expr.as_ref() else {
            return;
        };
        let Some((root, member)) = entity_field_shape(field) else {
            return;
        };
        if is_passthrough(&member) || !self.plan.entity_names.contains(&root) {
            return;
        }
        let Pat::Type(pt) = &local.pat else {
            return;
        };
        match lane_elem(&pt.ty) {
            Ok(elem) => self.record_read(&root, &member, elem, (*pt.ty).clone()),
            Err(LaneReject::Bool) => {
                self.reject(&root, &member);
                self.plan.errors.push(syn::Error::new(
                    pt.ty.span(),
                    format!(
                        "`{root}.{member}` gathers as `bool`, which has no WGSL \
                         storage-array spelling in GPU kernel `{}`; use `u32`",
                        self.fn_name
                    ),
                ));
            }
            Err(LaneReject::Other) => {
                self.reject(&root, &member);
                self.plan.errors.push(syn::Error::new(
                    pt.ty.span(),
                    format!(
                        "gather annotation for `{root}.{member}` in GPU kernel `{}` must be a \
                         registry scalar/vector or `Entity`",
                        self.fn_name
                    ),
                ));
            }
        }
    }

    /// `return t.field;` typed by the function return type.
    pub(crate) fn take_return_read(&mut self, root: &str, member: &str) -> bool {
        let Some(elem) = self.return_ty.clone().filter(lane_elem_of_ir) else {
            return false;
        };
        let Some(elem_ty) = self.return_syn_ty.clone() else {
            return false;
        };
        self.record_read(root, member, elem, elem_ty);
        true
    }

    pub(crate) fn record_read(&mut self, root: &str, member: &str, elem: IrType, elem_ty: Type) {
        let key = (root.to_string(), member.to_string());
        if self.errored.contains(&key) {
            return;
        }
        if let Some(existing) = self.plan.reads.get(&key) {
            if existing.elem != elem {
                self.reject(root, member);
                self.plan.errors.push(syn::Error::new(
                    elem_ty.span(),
                    format!(
                        "conflicting gather annotations for `{root}.{member}` in GPU \
                         kernel `{}`; one field has one lane type",
                        self.fn_name
                    ),
                ));
            }
            return;
        }
        self.plan.mentions_entity = true;
        self.plan.reads.insert(
            key,
            Lane {
                binding: root.to_string(),
                elem,
                elem_ty,
                lane_ident: format!("_ornis_lane_{root}__{member}"),
                helper_ident: format!("_ornis_gather_{root}__{member}"),
            },
        );
    }

    /// Mark a site as self-reported so the generic position error does
    /// not fire on the same (binding, field) a second time.
    pub(crate) fn reject(&mut self, root: &str, member: &str) {
        self.errored.insert((root.to_string(), member.to_string()));
    }
}

impl<'ast> Visit<'ast> for Scanner<'_> {
    fn visit_local(&mut self, local: &'ast Local) {
        if let Pat::Type(pt) = &local.pat {
            self.register_pat_type(pt);
        }
        self.check_bound_read(local);
        visit::visit_local(self, local);
    }

    fn visit_expr_return(&mut self, ret: &'ast syn::ExprReturn) {
        if let Some(expr) = &ret.expr
            && let Expr::Field(field) = expr.as_ref()
            && let Some((root, member)) = entity_field_shape(field)
            && !is_passthrough(&member)
            && self.plan.entity_names.contains(&root)
            && self.take_return_read(&root, &member)
        {
            return;
        }
        visit::visit_expr_return(self, ret);
    }

    fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
        self.register_scrutinee_bindings(Some(&node.expr), node.arms.iter().map(|arm| &arm.pat));
        visit::visit_expr_match(self, node);
    }

    fn visit_expr_if(&mut self, node: &'ast syn::ExprIf) {
        if let Expr::Let(let_expr) = node.cond.as_ref() {
            self.register_scrutinee_bindings(Some(&let_expr.expr), std::iter::once(&*let_expr.pat));
        }
        visit::visit_expr_if(self, node);
    }

    fn visit_expr_field(&mut self, field: &'ast ExprField) {
        if let Some((root, member)) = entity_field_shape(field) {
            let key = (root.clone(), member.clone());
            if self.plan.entity_names.contains(&root)
                && !is_passthrough(&member)
                && !self.plan.reads.contains_key(&key)
                && !self.errored.contains(&key)
            {
                self.reject(&root, &member);
                self.plan.errors.push(syn::Error::new(
                    field.span(),
                    format!(
                        "`{root}.{member}` must be bound through an annotated let in \
                         GPU kernel `{}` (`let p: Vec3 = {root}.{member};`) — the \
                         annotation is the field's type",
                        self.fn_name
                    ),
                ));
            }
            return;
        }
        visit::visit_expr_field(self, field);
    }

    fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {
        self.reject_entity_write(&node.left);
        visit::visit_expr_assign(self, node);
    }

    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        use syn::BinOp;
        let assign = matches!(
            node.op,
            BinOp::AddAssign(_)
                | BinOp::SubAssign(_)
                | BinOp::MulAssign(_)
                | BinOp::DivAssign(_)
                | BinOp::RemAssign(_)
                | BinOp::ShlAssign(_)
                | BinOp::ShrAssign(_)
                | BinOp::BitAndAssign(_)
                | BinOp::BitOrAssign(_)
                | BinOp::BitXorAssign(_)
        );
        if assign {
            self.reject_entity_write(&node.left);
        }
        if matches!(node.op, BinOp::Eq(_) | BinOp::Ne(_)) {
            self.reject_entity_equality(node);
        }
        visit::visit_expr_binary(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let receiver = strip_parens(&node.receiver);
        self.reject_entity_method(node, root_ident(receiver), field_shape_of(receiver));
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_reference(&mut self, node: &'ast syn::ExprReference) {
        self.reject_entity_reference(node, root_ident(&node.expr));
        visit::visit_expr_reference(self, node);
    }

    fn visit_expr_struct(&mut self, node: &'ast syn::ExprStruct) {
        if node
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident == "Entity")
        {
            self.plan.mentions_entity = true;
        }
        visit::visit_expr_struct(self, node);
    }
}
