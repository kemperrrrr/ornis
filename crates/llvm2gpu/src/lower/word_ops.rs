//! Word-granular stores, memory intrinsics and array helpers.

use super::WORD_BYTES;
use super::emit::scalar;
use super::lowerer::Lowerer;
use super::memory::{
    BYTE_MASK, MEMORY_INTRINSIC_ARGS, MemoryBase, MemoryPointer, repeated_byte, word_count,
};
use crate::error::TranslateError;
use crate::parse::Operand;

impl<'a, 'b> Lowerer<'a, 'b> {
    /// Write `word` at `base + words`.
    pub(crate) fn store_word(
        &mut self,
        out: &mut naga::Block,
        base: MemoryBase,
        words: naga::Handle<naga::Expression>,
        word: naga::Handle<naga::Expression>,
    ) {
        let base = self.memory_base(base, out);
        let slot = self.expr(out, naga::Expression::Access { base, index: words });
        out.push(
            naga::Statement::Store {
                pointer: slot,
                value: word,
            },
            naga::Span::default(),
        );
    }

    /// Read the word at `base + words`.
    pub(crate) fn load_word(
        &mut self,
        out: &mut naga::Block,
        base: MemoryBase,
        words: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        let base = self.memory_base(base, out);
        let slot = self.expr(out, naga::Expression::Access { base, index: words });
        self.expr(out, naga::Expression::Load { pointer: slot })
    }

    /// `(destination, fill, byte count)` of a `memset`-shaped call.
    pub(crate) fn intrinsic_target<'o>(
        &self,
        args: &'o [Operand],
    ) -> Result<(MemoryPointer, &'o Operand, u32), TranslateError> {
        if args.len() < MEMORY_INTRINSIC_ARGS {
            return Err(TranslateError::unsupported());
        }
        let Operand::Local(name) = &args[0] else {
            return Err(TranslateError::unsupported());
        };
        let Operand::U32(bytes) = args[2] else {
            return Err(TranslateError::unsupported());
        };
        Ok((self.pointer(name)?, &args[1], bytes))
    }

    /// `llvm.memset(ptr, fill, size, …)`: fill whole words with one byte.
    ///
    /// The size and the fill byte must be immediates and the size a multiple of
    /// four, which is what `[value; N]` initialization and rustc's array
    /// zeroing produce.
    pub(crate) fn memset(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        let (pointer, fill, bytes) = self.intrinsic_target(args)?;
        let byte = match fill {
            Operand::U32(value) => *value & BYTE_MASK,
            Operand::Bool(value) => u32::from(*value),
            Operand::F32(_) | Operand::Local(_) | Operand::Symbol(_) => {
                return Err(TranslateError::unsupported());
            }
        };
        let word = self.u32_literal(out, repeated_byte(byte));
        for offset in 0..word_count(bytes)? {
            let index = self.offset_words(out, pointer.words, offset);
            self.store_word(out, pointer.base, index, word);
        }
        Ok(())
    }

    /// `llvm.memcpy(dst, src, size, …)`: copy whole words.
    pub(crate) fn memcpy(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        if args.len() < MEMORY_INTRINSIC_ARGS {
            return Err(TranslateError::unsupported());
        }
        let Operand::Local(dst) = &args[0] else {
            return Err(TranslateError::unsupported());
        };
        let Operand::Local(src) = &args[1] else {
            return Err(TranslateError::unsupported());
        };
        let Operand::U32(bytes) = args[2] else {
            return Err(TranslateError::unsupported());
        };
        let dst = self.pointer(dst)?;
        let src = self.pointer(src)?;
        for offset in 0..word_count(bytes)? {
            let from = self.offset_words(out, src.words, offset);
            let word = self.load_word(out, src.base, from);
            let to = self.offset_words(out, dst.words, offset);
            self.store_word(out, dst.base, to, word);
        }
        Ok(())
    }

    /// `words + offset`.
    pub(crate) fn offset_words(
        &mut self,
        out: &mut naga::Block,
        words: naga::Handle<naga::Expression>,
        offset: u32,
    ) -> naga::Handle<naga::Expression> {
        if offset == 0 {
            return words;
        }
        let step = self.u32_literal(out, offset);
        self.binary(out, naga::BinaryOperator::Add, words, step)
    }

    /// Pointer expression of a memory base.
    pub(crate) fn memory_base(
        &mut self,
        base: MemoryBase,
        out: &mut naga::Block,
    ) -> naga::Handle<naga::Expression> {
        match base {
            MemoryBase::Local(var) => self.expr(out, naga::Expression::LocalVariable(var)),
            MemoryBase::Constant(var) => self.expr(out, naga::Expression::GlobalVariable(var)),
        }
    }

    /// `array<u32, words>`.
    pub(crate) fn word_array_type(&mut self, words: u32) -> naga::Handle<naga::Type> {
        let base = scalar(&mut self.builders.module, naga::ScalarKind::Uint);
        let size = std::num::NonZeroU32::new(words).unwrap_or(std::num::NonZeroU32::MIN);
        self.builders.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base,
                    size: naga::ArraySize::Constant(size),
                    stride: WORD_BYTES,
                },
            },
            naga::Span::default(),
        )
    }

    /// A module constant as a private `array<u32, N>` global.
    ///
    /// The words are the constant's bytes, so an integer table and a float
    /// table both keep their exact bit patterns; a float load bitcasts.
    pub(crate) fn constant_array(
        &mut self,
        symbol: &str,
    ) -> Result<(naga::Handle<naga::GlobalVariable>, u32), TranslateError> {
        if let Some(found) = self.constants.get(symbol) {
            return Ok(*found);
        }
        let bytes = self
            .builders
            .module_constants
            .get(symbol)
            .ok_or_else(TranslateError::unsupported)?
            .clone();
        let words = word_count(bytes.len() as u32)?;
        let ty = self.word_array_type(words);
        let mut components = Vec::with_capacity(words as usize);
        for chunk in bytes.chunks_exact(WORD_BYTES as usize) {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            components.push(self.builders.module.global_expressions.append(
                naga::Expression::Literal(naga::Literal::U32(word)),
                naga::Span::default(),
            ));
        }
        let init = self.builders.module.global_expressions.append(
            naga::Expression::Compose { ty, components },
            naga::Span::default(),
        );
        let var = self.builders.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(format!("const_{}", symbol.replace(['.', '-'], "_"))),
                space: naga::AddressSpace::Private,
                binding: None,
                ty,
                init: Some(init),
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.constants.insert(symbol.to_string(), (var, words));
        Ok((var, words))
    }
}
