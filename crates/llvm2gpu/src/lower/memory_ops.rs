//! Memory operations: alloca, loads and stores, pointer math and intrinsics.

use super::WORD_BYTES;
use super::lowerer::Lowerer;
use super::memory::{MemoryBase, MemoryPointer, local_label, word_count};
use crate::error::TranslateError;
use crate::parse::{Operand, PtrBase, Ty};

impl<'a, 'b> Lowerer<'a, 'b> {
    /// `alloca`: a function-scope array of 4-byte words.
    pub(crate) fn alloca(&mut self, dst: &str, bytes: u32) -> Result<(), TranslateError> {
        let words = word_count(bytes)?;
        let ty = self.word_array_type(words);
        let var = self.func.local_variables.append(
            naga::LocalVariable {
                name: Some(local_label(dst)),
                ty,
                init: None,
            },
            naga::Span::default(),
        );
        let zero = self.u32_literal(&mut naga::Block::new(), 0);
        self.pointers.insert(
            dst.to_string(),
            MemoryPointer {
                base: MemoryBase::Local(var),
                words: zero,
            },
        );
        Ok(())
    }

    /// `getelementptr`: move a pointer by `index * stride` bytes.
    pub(crate) fn getelementptr(
        &mut self,
        dst: &str,
        base: &PtrBase,
        stride: u32,
        index: &Operand,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        let (base, words) = match base {
            PtrBase::Local(name) => {
                let pointer = self
                    .pointers
                    .get(name)
                    .ok_or_else(TranslateError::unsupported)?;
                (pointer.base, pointer.words)
            }
            PtrBase::Global(symbol) => {
                let (var, _) = self.constant_array(symbol)?;
                let zero = self.u32_literal(out, 0);
                (MemoryBase::Constant(var), zero)
            }
        };
        let step = self.pointer_step(stride, index, out)?;
        let words = self.binary(out, naga::BinaryOperator::Add, words, step);
        self.pointers
            .insert(dst.to_string(), MemoryPointer { base, words });
        Ok(())
    }

    /// Words one `getelementptr` index step covers.
    ///
    /// Move a pointer by `index * stride` bytes.
    ///
    /// A stride that is a multiple of four divides cleanly into the word offset
    /// local memory is addressed by. A smaller stride is how LLVM spells an
    /// access once it has folded the element size into the index — see
    /// `byte_offset` — so it is accepted too: a constant index must be
    /// word-aligned to keep a hand-written IR honest, and a computed one is
    /// scaled by the same division the constant case proves.
    pub(crate) fn pointer_step(
        &mut self,
        stride: u32,
        index: &Operand,
        out: &mut naga::Block,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if stride.is_multiple_of(WORD_BYTES) {
            let index = self.operand(out, index)?;
            let step = stride / WORD_BYTES;
            if step == 1 {
                return Ok(index);
            }
            let step = self.u32_literal(out, step);
            return Ok(self.binary(out, naga::BinaryOperator::Multiply, index, step));
        }
        if let Operand::U32(offset) = index {
            if !offset.is_multiple_of(WORD_BYTES) {
                return Err(TranslateError::unsupported());
            }
            return Ok(self.u32_literal(out, offset / WORD_BYTES));
        }
        // `byte_offset`: LLVM folded the element size into the index, so the
        // byte offset is word-aligned at every access the IR can actually
        // perform. The division is what turns it back into a word index.
        let bytes = self.operand(out, index)?;
        let word = self.u32_literal(out, WORD_BYTES);
        Ok(self.binary(out, naga::BinaryOperator::Divide, bytes, word))
    }

    /// `load`: read one 4-byte word and reinterpret it as `ty`.
    pub(crate) fn load(
        &mut self,
        ptr: &str,
        ty: Ty,
        out: &mut naga::Block,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let pointer = self.pointer(ptr)?;
        let word = self.load_word(out, pointer.base, pointer.words);
        match ty {
            Ty::I32 => Ok(word),
            Ty::Float => Ok(self.bitcast_float(out, word)),
            _ => Err(TranslateError::unsupported()),
        }
    }

    /// `store`: write one 4-byte word.
    pub(crate) fn store(
        &mut self,
        ptr: &str,
        value: &Operand,
        ty: Ty,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        let pointer = self.pointer(ptr)?;
        let value = self.operand(out, value)?;
        let value = match ty {
            Ty::I32 => value,
            Ty::Float => self.bitcast_uint(out, value),
            _ => return Err(TranslateError::unsupported()),
        };
        self.store_word(out, pointer.base, pointer.words, value);
        Ok(())
    }

    /// `select i1 %c, ptr %a, ptr %b`: pick the offset of one of two pointers.
    ///
    /// Both have to start at the same base, the same rule `phi ptr` follows:
    /// Naga has no pointer value, so the selection happens on the word offset.
    /// Returns `None` when the operands are not pointers.
    pub(crate) fn pointer_select(
        &mut self,
        dst: &str,
        cond: &Operand,
        accept: &Operand,
        reject: &Operand,
        out: &mut naga::Block,
    ) -> Result<Option<MemoryPointer>, TranslateError> {
        let (Operand::Local(accept), Operand::Local(reject)) = (accept, reject) else {
            return Ok(None);
        };
        let (Some(accept), Some(reject)) = (
            self.pointers.get(accept).copied(),
            self.pointers.get(reject).copied(),
        ) else {
            return Ok(None);
        };
        if accept.base != reject.base {
            return Err(TranslateError::UnsupportedType(format!(
                "`select` of two pointers from different allocations: `{}` picks between                  memory this translator cannot name at runtime",
                dst
            )));
        }
        let condition = self.operand(out, cond)?;
        let words = self.expr(
            out,
            naga::Expression::Select {
                condition,
                accept: accept.words,
                reject: reject.words,
            },
        );
        let pointer = MemoryPointer {
            base: accept.base,
            words,
        };
        self.pointers.insert(dst.to_string(), pointer);
        Ok(Some(pointer))
    }

    /// Pointer SSA value `name`.
    pub(crate) fn pointer(&self, name: &str) -> Result<MemoryPointer, TranslateError> {
        self.pointers
            .get(name)
            .map(|pointer| MemoryPointer {
                base: pointer.base,
                words: pointer.words,
            })
            .ok_or_else(TranslateError::unsupported)
    }
}
