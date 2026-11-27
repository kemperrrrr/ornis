//! Value expressions: operands, literals, casts and arithmetic.

use super::emit::push_emit;
use super::lowerer::Lowerer;
use crate::error::TranslateError;
use crate::parse::Operand;

impl<'a, 'b> Lowerer<'a, 'b> {
    pub(crate) fn assign_binary(
        &mut self,
        dst: &str,
        op: naga::BinaryOperator,
        a: &Operand,
        b: &Operand,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        let left = self.operand(out, a)?;
        let right = self.operand(out, b)?;
        let v = self.expr(out, naga::Expression::Binary { op, left, right });
        self.bind(dst, v, out);
        Ok(())
    }

    /// Signed `i32` binary op: bitcast the operands to `i32`, apply the
    /// operator (WGSL picks `sdiv`/`srem` from the operand type), and cast the
    /// result back to the stored `u32` representation.
    pub(crate) fn assign_signed(
        &mut self,
        dst: &str,
        op: naga::BinaryOperator,
        a: &Operand,
        b: &Operand,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        let left = self.operand(out, a)?;
        let right = self.operand(out, b)?;
        let left = self.bitcast_sint(out, left);
        let right = self.bitcast_sint(out, right);
        let value = self.binary(out, op, left, right);
        let value = self.bitcast_uint(out, value);
        self.bind(dst, value, out);
        Ok(())
    }

    /// Bitcast the stored `u32` to `i32` (`bitcast<i32>(x)`).
    pub(crate) fn bitcast_sint(
        &mut self,
        out: &mut naga::Block,
        value: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.expr(
            out,
            naga::Expression::As {
                expr: value,
                kind: naga::ScalarKind::Sint,
                convert: None,
            },
        )
    }

    /// Bitcast an `i32`-typed result back to the stored `u32` type.
    pub(crate) fn bitcast_uint(
        &mut self,
        out: &mut naga::Block,
        value: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.expr(
            out,
            naga::Expression::As {
                expr: value,
                kind: naga::ScalarKind::Uint,
                convert: None,
            },
        )
    }

    /// A `u32` literal expression.
    pub(crate) fn u32_literal(
        &mut self,
        out: &mut naga::Block,
        value: u32,
    ) -> naga::Handle<naga::Expression> {
        self.expr(out, naga::Expression::Literal(naga::Literal::U32(value)))
    }

    /// Bitcast a stored `u32` word to `f32`.
    pub(crate) fn bitcast_float(
        &mut self,
        out: &mut naga::Block,
        value: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.expr(
            out,
            naga::Expression::As {
                expr: value,
                kind: naga::ScalarKind::Float,
                convert: None,
            },
        )
    }

    pub(crate) fn binary(
        &mut self,
        out: &mut naga::Block,
        op: naga::BinaryOperator,
        left: naga::Handle<naga::Expression>,
        right: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.expr(out, naga::Expression::Binary { op, left, right })
    }

    pub(crate) fn negate(
        &mut self,
        out: &mut naga::Block,
        expr: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.expr(
            out,
            naga::Expression::Unary {
                op: naga::UnaryOperator::LogicalNot,
                expr,
            },
        )
    }

    /// `fcmp`: ordered predicates are WGSL comparisons. Unordered (`u*`)
    /// predicates must also be true when either operand is NaN, so they lower
    /// through [`Self::float_ordered`]; `one` is the ordered not-equal and is
    /// false for NaN.
    ///
    /// Float `!=` is not used: naga's SPIR-V backend emits `OpFOrdNotEqual`
    /// for it, which is false for NaN, so both `une` and the usual `x != x`
    /// NaN test would silently break on Vulkan (Metal treats `!=` as
    /// unordered, which is why the difference shows only on lavapipe).
    pub(crate) fn compare_float(
        &mut self,
        out: &mut naga::Block,
        pred: &str,
        a: &Operand,
        b: &Operand,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let left = self.operand(out, a)?;
        let right = self.operand(out, b)?;
        let value = match pred {
            "oeq" => self.binary(out, naga::BinaryOperator::Equal, left, right),
            "olt" => self.binary(out, naga::BinaryOperator::Less, left, right),
            "ole" => self.binary(out, naga::BinaryOperator::LessEqual, left, right),
            "ogt" => self.binary(out, naga::BinaryOperator::Greater, left, right),
            "oge" => self.binary(out, naga::BinaryOperator::GreaterEqual, left, right),
            "one" => {
                let lt = self.binary(out, naga::BinaryOperator::Less, left, right);
                let gt = self.binary(out, naga::BinaryOperator::Greater, left, right);
                self.binary(out, naga::BinaryOperator::InclusiveOr, lt, gt)
            }
            "une" => {
                let ordered = self.float_ordered(out, left, right);
                let equal = self.binary(out, naga::BinaryOperator::Equal, left, right);
                let ordered_equal =
                    self.binary(out, naga::BinaryOperator::LogicalAnd, ordered, equal);
                self.negate(out, ordered_equal)
            }
            "ueq" => {
                let ordered = self.float_ordered(out, left, right);
                let unordered = self.negate(out, ordered);
                let equal = self.binary(out, naga::BinaryOperator::Equal, left, right);
                self.binary(out, naga::BinaryOperator::LogicalOr, unordered, equal)
            }
            "ult" => {
                let ge = self.binary(out, naga::BinaryOperator::GreaterEqual, left, right);
                self.negate(out, ge)
            }
            "ule" => {
                let gt = self.binary(out, naga::BinaryOperator::Greater, left, right);
                self.negate(out, gt)
            }
            "ugt" => {
                let le = self.binary(out, naga::BinaryOperator::LessEqual, left, right);
                self.negate(out, le)
            }
            "uge" => {
                let lt = self.binary(out, naga::BinaryOperator::Less, left, right);
                self.negate(out, lt)
            }
            "uno" | "ord" => {
                let ordered = self.float_ordered(out, left, right);
                if pred == "ord" {
                    ordered
                } else {
                    self.negate(out, ordered)
                }
            }
            _ => return Err(TranslateError::unsupported()),
        };
        Ok(value)
    }

    /// Whether both operands are comparable (`a <= a && b <= b`), i.e. not
    /// NaN. `<=` is ordered on every backend; `a != a` is not (see
    /// [`Self::compare_float`]).
    pub(crate) fn float_ordered(
        &mut self,
        out: &mut naga::Block,
        a: naga::Handle<naga::Expression>,
        b: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        let left = self.binary(out, naga::BinaryOperator::LessEqual, a, a);
        let right = self.binary(out, naga::BinaryOperator::LessEqual, b, b);
        self.binary(out, naga::BinaryOperator::LogicalAnd, left, right)
    }

    /// `icmp`: `s*` predicates reinterpret the stored `u32` as `i32`;
    /// unsigned and equality predicates compare the bit pattern directly.
    pub(crate) fn compare_int(
        &mut self,
        out: &mut naga::Block,
        pred: &str,
        a: &Operand,
        b: &Operand,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let left = self.operand(out, a)?;
        let right = self.operand(out, b)?;
        let (left, right, op) = match pred {
            "slt" => (
                self.bitcast_sint(out, left),
                self.bitcast_sint(out, right),
                naga::BinaryOperator::Less,
            ),
            "sle" => (
                self.bitcast_sint(out, left),
                self.bitcast_sint(out, right),
                naga::BinaryOperator::LessEqual,
            ),
            "sgt" => (
                self.bitcast_sint(out, left),
                self.bitcast_sint(out, right),
                naga::BinaryOperator::Greater,
            ),
            "sge" => (
                self.bitcast_sint(out, left),
                self.bitcast_sint(out, right),
                naga::BinaryOperator::GreaterEqual,
            ),
            "eq" => (left, right, naga::BinaryOperator::Equal),
            "ne" => (left, right, naga::BinaryOperator::NotEqual),
            "ult" => (left, right, naga::BinaryOperator::Less),
            "ule" => (left, right, naga::BinaryOperator::LessEqual),
            "ugt" => (left, right, naga::BinaryOperator::Greater),
            "uge" => (left, right, naga::BinaryOperator::GreaterEqual),
            _ => return Err(TranslateError::unsupported()),
        };
        Ok(self.binary(out, op, left, right))
    }
}

impl<'a, 'b> Lowerer<'a, 'b> {
    pub(crate) fn operand(
        &mut self,
        out: &mut naga::Block,
        op: &Operand,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let v = match op {
            Operand::Local(name) => self
                .env
                .get(name)
                .copied()
                .ok_or_else(TranslateError::unsupported)?,
            Operand::F32(value) => {
                self.expr(out, naga::Expression::Literal(naga::Literal::F32(*value)))
            }
            Operand::U32(value) => {
                self.expr(out, naga::Expression::Literal(naga::Literal::U32(*value)))
            }
            Operand::Bool(value) => {
                self.expr(out, naga::Expression::Literal(naga::Literal::Bool(*value)))
            }
            // A module symbol is a panic location; the call carrying it is
            // dropped, and a zero keeps the value well-typed if one surfaces.
            Operand::Symbol(_) => self.u32_literal(out, 0),
        };
        Ok(v)
    }

    /// Append an expression and emit it in `out` when Naga requires it.
    pub(crate) fn expr(
        &mut self,
        out: &mut naga::Block,
        expr: naga::Expression,
    ) -> naga::Handle<naga::Expression> {
        let handle = self.func.expressions.append(expr, naga::Span::default());
        push_emit(&mut self.func, out, handle);
        handle
    }
}
