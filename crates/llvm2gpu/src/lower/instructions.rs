//! The per-instruction dispatcher that routes one LLVM instruction.

use super::lowerer::Lowerer;
use crate::error::TranslateError;
use crate::parse::{Instr, Ty};

impl<'a, 'b> Lowerer<'a, 'b> {
    pub(crate) fn emit_instr(
        &mut self,
        instr: &Instr,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        crate::site::enter_instruction(&format!("{instr:?}"));
        match instr {
            Instr::FAdd { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Add, a, b, out)?
            }
            Instr::FMul { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Multiply, a, b, out)?
            }
            Instr::FSub { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Subtract, a, b, out)?
            }
            Instr::FDiv { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Divide, a, b, out)?
            }
            Instr::Add { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Add, a, b, out)?
            }
            Instr::Sub { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Subtract, a, b, out)?
            }
            Instr::Mul { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Multiply, a, b, out)?
            }
            Instr::UDiv { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Divide, a, b, out)?
            }
            Instr::SDiv { dst, a, b } => {
                self.assign_signed(dst, naga::BinaryOperator::Divide, a, b, out)?
            }
            Instr::URem { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Modulo, a, b, out)?
            }
            Instr::SRem { dst, a, b } => {
                self.assign_signed(dst, naga::BinaryOperator::Modulo, a, b, out)?
            }
            Instr::And { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::And, a, b, out)?
            }
            Instr::Or { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::InclusiveOr, a, b, out)?
            }
            Instr::Xor { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::ExclusiveOr, a, b, out)?
            }
            Instr::Shl { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::ShiftLeft, a, b, out)?
            }
            Instr::LShr { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::ShiftRight, a, b, out)?
            }
            Instr::AShr { dst, a, b } => {
                // Arithmetic shift: only the shifted value is signed; WGSL
                // keeps the shift amount unsigned.
                let left = self.operand(out, a)?;
                let right = self.operand(out, b)?;
                let left = self.bitcast_sint(out, left);
                let value = self.expr(
                    out,
                    naga::Expression::Binary {
                        op: naga::BinaryOperator::ShiftRight,
                        left,
                        right,
                    },
                );
                let value = self.bitcast_uint(out, value);
                self.bind(dst, value, out);
            }
            Instr::FNeg { dst, a } => {
                let src = self.operand(out, a)?;
                let v = self.expr(
                    out,
                    naga::Expression::Unary {
                        op: naga::UnaryOperator::Negate,
                        expr: src,
                    },
                );
                self.bind(dst, v, out);
            }
            Instr::Fcmp { dst, pred, a, b } => {
                let v = self.compare_float(out, pred, a, b)?;
                self.bind(dst, v, out);
            }
            Instr::Icmp { dst, pred, a, b } => {
                let v = self.compare_int(out, pred, a, b)?;
                self.bind(dst, v, out);
            }
            Instr::Select {
                dst,
                ty: _,
                c,
                a,
                b,
            } => {
                if let Some(pointer) = self.pointer_select(dst, c, a, b, out)? {
                    let _ = pointer;
                    return Ok(());
                }
                let condition = self.operand(out, c)?;
                let accept = self.operand(out, a)?;
                let reject = self.operand(out, b)?;
                let v = self.expr(
                    out,
                    naga::Expression::Select {
                        condition,
                        accept,
                        reject,
                    },
                );
                self.bind(dst, v, out);
            }
            Instr::Uitofp { dst, a } => {
                let value = self.cast(out, a, naga::ScalarKind::Float)?;
                self.bind(dst, value, out);
            }
            Instr::Sitofp { dst, a } => {
                let value = self.cast_signed(out, a, naga::ScalarKind::Float)?;
                self.bind(dst, value, out);
            }
            Instr::Fptosi { dst, a } => {
                // A signed truncation: convert to `i32`, then keep the bits.
                let value = self.cast(out, a, naga::ScalarKind::Sint)?;
                let value = self.bitcast_uint(out, value);
                self.bind(dst, value, out);
            }
            Instr::Fptoui { dst, a } => {
                let value = self.cast(out, a, naga::ScalarKind::Uint)?;
                self.bind(dst, value, out);
            }
            Instr::Bitcast { dst, from, to, a } => {
                if from == to {
                    let value = self.operand(out, a)?;
                    self.bind(dst, value, out);
                } else {
                    let kind = match (from, to) {
                        (Ty::I32, Ty::Float) => naga::ScalarKind::Float,
                        (Ty::Float, Ty::I32) => naga::ScalarKind::Uint,
                        _ => return Err(TranslateError::unsupported()),
                    };
                    let expr = self.operand(out, a)?;
                    let value = self.expr(
                        out,
                        naga::Expression::As {
                            expr,
                            kind,
                            convert: None,
                        },
                    );
                    self.bind(dst, value, out);
                }
            }
            Instr::Call {
                dst,
                ty: _,
                callee,
                args,
            } => {
                let v = self.call_value(out, callee, args)?;
                self.bind(dst, v, out);
            }
            Instr::CallVoid { callee, args } => self.call_void(out, callee, args)?,
            Instr::Alloca { dst, bytes } => self.alloca(dst, *bytes)?,
            Instr::GetElementPtr {
                dst,
                base,
                stride,
                index,
            } => self.getelementptr(dst, base, *stride, index, out)?,
            Instr::Load { dst, ptr, ty } => {
                let value = self.load(ptr, *ty, out)?;
                self.bind(dst, value, out);
            }
            Instr::Store { ptr, value, ty } => self.store(ptr, value, *ty, out)?,
            Instr::Phi { .. } => return Err(TranslateError::unsupported()),
            Instr::Ret { .. } | Instr::RetVoid => return Err(TranslateError::unsupported()),
        }
        Ok(())
    }
}
