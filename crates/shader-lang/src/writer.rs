//! WGSL writer: renders [`IrExpr`]/[`IrStmt`] to WGSL text.
//!
//! The writer owns every WGSL spelling decision (constructor syntax,
//! operator symbols, statement terminators and their trailing spaces,
//! the positional-struct comment). Lowering owns none. Byte-identity
//! with the historical inline formatter is pinned by the unit tests
//! below and by the translator's exact-string suites upstream.

use crate::ShaderType;
use crate::ir::{IrBinOp, IrBlock, IrCallee, IrElse, IrExpr, IrStmt, IrUnOp, UNSUPPORTED_MARKER};

/// Render one expression.
pub fn print_expr(e: &IrExpr) -> String {
    match e {
        IrExpr::Int(s) | IrExpr::Float(s) => s.clone(),
        IrExpr::Bool(b) => b.to_string(),
        IrExpr::Path(segs) => print_path(segs),
        IrExpr::Field { base, member } => format!("{}.{member}", print_expr(base)),
        IrExpr::Index { base, index } => {
            format!("{}[{}]", print_expr(base), print_expr(index))
        }
        IrExpr::Paren(inner) => format!("({})", print_expr(inner)),
        IrExpr::Binary { op, left, right } => {
            format!("{} {} {}", print_expr(left), bin_op(*op), print_expr(right))
        }
        IrExpr::Unary { op, expr } => format!("{}{}", un_op(*op), print_expr(expr)),
        IrExpr::Assign { left, right } => {
            format!("{} = {}", print_expr(left), print_expr(right))
        }
        IrExpr::Call { target, args } => print_call(target, args),
        IrExpr::Method { base, method, args } => format!(
            "{}.{}({})",
            print_expr(base),
            method,
            args.iter().map(print_expr).collect::<Vec<_>>().join(", ")
        ),
        IrExpr::Struct { name, fields } => {
            let args = fields
                .iter()
                .map(|(_, v)| print_expr(v))
                .collect::<Vec<_>>()
                .join(", ");
            let names = fields
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            format!("{name}({args}) /* {names} */")
        }
        IrExpr::If { cond, then, els } => print_if(cond, then, els),
        IrExpr::Block(stmts) => print_block(stmts),
        IrExpr::Return(expr) => match expr {
            Some(e) => format!("return {}", print_expr(e)),
            None => "return".to_string(),
        },
        IrExpr::Continue => "continue".to_string(),
        IrExpr::Break => "break".to_string(),
        IrExpr::Cast(inner) => print_expr(inner),
        IrExpr::Verbatim(s) => s.clone(),
        IrExpr::Unsupported => UNSUPPORTED_MARKER.to_string(),
    }
}

/// Render one statement, including its terminator and trailing space.
pub fn print_stmt(s: &IrStmt) -> String {
    match s {
        IrStmt::Let {
            name,
            mutable,
            init,
            decl_ty,
        } => {
            let kw = if *mutable { "var" } else { "let" };
            if let Some(init_val) = init {
                format!("{kw} {name} = {}; ", print_expr(init_val))
            } else if let Some(ty) = decl_ty {
                format!("{kw} {name}: {ty}; ")
            } else {
                format!("{kw} {name}; ")
            }
        }
        IrStmt::For {
            var,
            start,
            end,
            closed,
            body,
        } => {
            let op = if *closed { "<=" } else { "<" };
            format!(
                "for (var {var}: u32 = {}; {var} {op} {}; {var} = {var} + 1) {{ {} }} ",
                print_expr(start),
                print_expr(end),
                print_block(body)
            )
        }
        IrStmt::Flow { expr, semi } => {
            let e = print_expr(expr);
            if *semi {
                format!("{e}; ")
            } else {
                format!("{e} ")
            }
        }
        IrStmt::Expr {
            expr,
            semi,
            is_tail,
        } => {
            let e = print_expr(expr);
            if matches!(expr, IrExpr::Return(_)) {
                format!("{e}; ")
            } else if matches!(expr, IrExpr::If { .. } | IrExpr::Block(_)) {
                if *semi {
                    format!("{e}; ")
                } else {
                    format!("{e} ")
                }
            } else if *semi || !is_tail {
                format!("{e}; ")
            } else {
                format!("return {e}; ")
            }
        }
        IrStmt::Tail(e) => format!("return {}; ", print_expr(e)),
    }
}

/// Render a statement list: statements joined with a single space.
pub fn print_block(stmts: &IrBlock) -> String {
    stmts.iter().map(print_stmt).collect::<Vec<_>>().join(" ")
}

fn print_call(target: &IrCallee, args: &[IrExpr]) -> String {
    let rendered: Vec<String> = args.iter().map(print_expr).collect();
    match target {
        IrCallee::Constructor { wgsl_ty } => format!("{wgsl_ty}({})", rendered.join(", ")),
        IrCallee::Builtin(b) => b.lower(&rendered),
        IrCallee::Verbatim(callee) => format!("{callee}({})", rendered.join(", ")),
        IrCallee::Expr(callee) => format!("{}({})", print_expr(callee), rendered.join(", ")),
    }
}

fn print_if(cond: &IrExpr, then: &IrBlock, els: &Option<IrElse>) -> String {
    let c = print_expr(cond);
    let then_trimmed = print_block(then).trim().to_string();
    match els {
        Some(IrElse::Block(stmts)) => {
            let else_trimmed = print_block(stmts).trim().to_string();
            format!("if ({c}) {{ {then_trimmed} }} else {{ {else_trimmed} }}")
        }
        Some(IrElse::If(inner)) => {
            let else_trimmed = print_expr(inner).trim().to_string();
            format!("if ({c}) {{ {then_trimmed} }} else {{ {else_trimmed} }}")
        }
        None => format!("if ({c}) {{ {then_trimmed} }}"),
    }
}

/// Resolve a path to WGSL: vector constants (`Vec3::ZERO` →
/// `vec3<f32>(0.0)`), type aliases (`Vec3` → `vec3<f32>`, `PI` → `PI`),
/// everything else joined verbatim.
fn print_path(segs: &[String]) -> String {
    let last = segs.last().map(|s| s.as_str());
    let parent = segs
        .len()
        .checked_sub(2)
        .and_then(|i| segs.get(i))
        .map(|s| s.as_str());
    if let (Some(parent), Some(last)) = (parent, last) {
        let dim = match parent {
            "Vec2" => Some(2),
            "Vec3" => Some(3),
            "Vec4" => Some(4),
            _ => None,
        };
        if let Some(dim) = dim {
            let value = match last {
                "ZERO" => Some("0.0"),
                "ONE" => Some("1.0"),
                _ => None,
            };
            if let Some(value) = value {
                return format!("vec{dim}<f32>({value})");
            }
        }
    }
    if let Some(last) = last {
        if last == "PI" {
            return "PI".to_string();
        }
        if last != "bool"
            && let Some(t) = ShaderType::from_rust(last)
        {
            return t.wgsl().to_string();
        }
    }
    segs.join("::")
}

fn bin_op(op: IrBinOp) -> &'static str {
    match op {
        IrBinOp::Add => "+",
        IrBinOp::Sub => "-",
        IrBinOp::Mul => "*",
        IrBinOp::Div => "/",
        IrBinOp::Rem => "%",
        IrBinOp::And => "&&",
        IrBinOp::Or => "||",
        IrBinOp::Eq => "==",
        IrBinOp::Ne => "!=",
        IrBinOp::Lt => "<",
        IrBinOp::Le => "<=",
        IrBinOp::Gt => ">",
        IrBinOp::Ge => ">=",
        IrBinOp::Shl => "<<",
        IrBinOp::Shr => ">>",
        IrBinOp::BitXor => "^",
        IrBinOp::BitAnd => "&",
        IrBinOp::BitOr => "|",
        IrBinOp::AddAssign => "+=",
        IrBinOp::SubAssign => "-=",
        IrBinOp::MulAssign => "*=",
        IrBinOp::DivAssign => "/=",
        IrBinOp::RemAssign => "%=",
        IrBinOp::ShlAssign => "<<=",
        IrBinOp::ShrAssign => ">>=",
        IrBinOp::BitAndAssign => "&=",
        IrBinOp::BitOrAssign => "|=",
        IrBinOp::BitXorAssign => "^=",
    }
}

fn un_op(op: IrUnOp) -> &'static str {
    match op {
        IrUnOp::Neg => "-",
        IrUnOp::Not => "!",
        IrUnOp::DerefOrUnknown => "*",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ShaderBuiltin;

    fn var(name: &str) -> IrExpr {
        IrExpr::Path(vec![name.to_string()])
    }

    #[test]
    fn arithmetic_prints_infix() {
        let e = IrExpr::Binary {
            op: IrBinOp::Add,
            left: Box::new(var("a")),
            right: Box::new(IrExpr::Binary {
                op: IrBinOp::Mul,
                left: Box::new(var("b")),
                right: Box::new(var("c")),
            }),
        };
        assert_eq!(print_expr(&e), "a + b * c");
    }

    #[test]
    fn parens_are_preserved_not_added() {
        let e = IrExpr::Binary {
            op: IrBinOp::Mul,
            left: Box::new(IrExpr::Paren(Box::new(IrExpr::Binary {
                op: IrBinOp::Add,
                left: Box::new(var("a")),
                right: Box::new(var("b")),
            }))),
            right: Box::new(var("c")),
        };
        assert_eq!(print_expr(&e), "(a + b) * c");
    }

    #[test]
    fn struct_keeps_positional_shape_and_field_comment() {
        let e = IrExpr::Struct {
            name: "VertexOutput".to_string(),
            fields: vec![
                (
                    "clip_position".to_string(),
                    IrExpr::Index {
                        base: Box::new(var("QUAD")),
                        index: Box::new(var("idx")),
                    },
                ),
                (
                    "uv".to_string(),
                    IrExpr::Index {
                        base: Box::new(var("UVS")),
                        index: Box::new(var("idx")),
                    },
                ),
            ],
        };
        assert_eq!(
            print_expr(&e),
            "VertexOutput(QUAD[idx], UVS[idx]) /* clip_position, uv */"
        );
    }

    #[test]
    fn path_resolves_vec_constants_and_aliases() {
        let seg = |s: &[&str]| IrExpr::Path(s.iter().map(|s| s.to_string()).collect());
        assert_eq!(
            print_expr(&seg(&["glam", "Vec3", "ZERO"])),
            "vec3<f32>(0.0)"
        );
        assert_eq!(print_expr(&seg(&["Vec3"])), "vec3<f32>");
        assert_eq!(print_expr(&seg(&["PI"])), "PI");
        assert_eq!(print_expr(&seg(&["quad"])), "quad");
    }

    #[test]
    fn builtin_applies_at_print() {
        let e = IrExpr::Call {
            target: IrCallee::Builtin(ShaderBuiltin::Mix),
            args: vec![var("a"), var("b"), var("t")],
        };
        assert_eq!(print_expr(&e), "mix(a, b, t)");
    }

    #[test]
    fn for_loop_shape_matches_historical() {
        let s = IrStmt::For {
            var: "i".to_string(),
            start: IrExpr::Int("0".to_string()),
            end: var("n"),
            closed: false,
            body: vec![IrStmt::Expr {
                expr: IrExpr::Continue,
                semi: true,
                is_tail: true,
            }],
        };
        assert_eq!(
            print_stmt(&s),
            "for (var i: u32 = 0; i < n; i = i + 1) { continue;  } "
        );
    }

    #[test]
    fn tail_expr_becomes_return() {
        let s = IrStmt::Tail(var("x"));
        assert_eq!(print_stmt(&s), "return x; ");
    }

    #[test]
    fn unsupported_is_loud() {
        assert!(print_expr(&IrExpr::Unsupported).contains("__wgsl_dsl_unsupported_statement__"));
    }
}
