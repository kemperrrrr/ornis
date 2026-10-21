//! Pure expression-shape helpers for [`crate::wgsl_gather`] (issue #110).
//!
//! Free functions only — every item here is side-effect-free so the
//! scanner can call them from Integration seams without tripping the
//! rustqual IOSP rule that forbids mixing branching with own calls.

use ornis_shader_lang::ShaderType;
use ornis_shader_lang::ir::IrType;
use syn::{Expr, ExprField, Member, Type};

/// `Entity` members that are real struct fields on both sides — never
/// gathers, no annotation needed.
pub(crate) const PASSTHROUGH_MEMBERS: [&str; 2] = ["index", "generation"];

/// Strip source-level parentheses: ` (t).position` is `t.position`.
pub(crate) fn strip_parens(expr: &Expr) -> &Expr {
    let mut expr = expr;
    while let Expr::Paren(paren) = expr {
        expr = &paren.expr;
    }
    expr
}

/// The base binding ident of an entity-rooted expression chain
/// (`t`, `t.position`, `t.position.x` all root to `t`). Pure logic:
/// paren-stripping is inlined so this never mixes with own calls.
pub(crate) fn root_ident(expr: &Expr) -> Option<String> {
    let mut expr = expr;
    loop {
        match expr {
            Expr::Paren(paren) => expr = &paren.expr,
            Expr::Path(p) if p.qself.is_none() && p.path.segments.len() == 1 => {
                return Some(p.path.segments[0].ident.to_string());
            }
            Expr::Field(f) => expr = &f.base,
            Expr::Index(ix) => expr = &ix.expr,
            _ => return None,
        }
    }
}

pub(crate) fn is_passthrough(member: &str) -> bool {
    PASSTHROUGH_MEMBERS.contains(&member)
}

/// The `(binding, field)` shape of a field access whose base is a plain
/// single-segment path — the only gather form the DSL accepts.
pub(crate) fn entity_field_shape(field: &ExprField) -> Option<(String, String)> {
    let Expr::Path(base) = strip_parens(&field.base) else {
        return None;
    };
    if base.qself.is_some() || base.path.segments.len() != 1 {
        return None;
    }
    let Member::Named(member) = &field.member else {
        return None;
    };
    Some((base.path.segments[0].ident.to_string(), member.to_string()))
}

/// [`entity_field_shape`] for an arbitrary expression: `None` unless
/// the expression is itself a field access on a plain path.
pub(crate) fn field_shape_of(expr: &Expr) -> Option<(String, String)> {
    let Expr::Field(field) = strip_parens(expr) else {
        return None;
    };
    entity_field_shape(field)
}

/// Which kind of lane an annotation can back; `bool` gets its own
/// rejection because WGSL storage arrays cannot hold `bool`/`BVecN`.
pub(crate) enum LaneReject {
    Bool,
    Other,
}

pub(crate) fn lane_elem(ty: &Type) -> Result<IrType, LaneReject> {
    if crate::kernel_types::is_entity_type(ty) {
        return Ok(IrType::Entity);
    }
    match crate::wgsl::rust_ty_to_ir(ty) {
        Some(IrType::Scalar(s))
            if !matches!(s, ShaderType::BVec2 | ShaderType::BVec3 | ShaderType::BVec4) =>
        {
            Ok(IrType::Scalar(s))
        }
        Some(IrType::Scalar(_) | IrType::Bool) => Err(LaneReject::Bool),
        _ => Err(LaneReject::Other),
    }
}

pub(crate) fn lane_elem_of_ir(ty: &IrType) -> bool {
    match ty {
        IrType::Entity => true,
        IrType::Scalar(s) => {
            !matches!(s, ShaderType::BVec2 | ShaderType::BVec3 | ShaderType::BVec4)
        }
        _ => false,
    }
}
