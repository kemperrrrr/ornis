//! Storage copies, void stores and the vector types they write through.

use super::SCALAR_WIDTH;
use super::emit::const_u32;
use super::lowerer::Lowerer;
use super::texture::Elem;
use crate::error::TranslateError;
use crate::parse::Operand;

impl<'a, 'b> Lowerer<'a, 'b> {
    /// Bit-preserving copy of one 16-byte record chunk.
    pub(crate) fn storage_copy(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        if args.len() != 6 {
            return Err(TranslateError::unsupported());
        }
        let src_group = const_u32(&args[0])?;
        let src_binding = const_u32(&args[1])?;
        let src_index = self.operand(out, &args[2])?;
        let dst_group = const_u32(&args[3])?;
        let dst_binding = const_u32(&args[4])?;
        let dst_index = self.operand(out, &args[5])?;
        let src = self
            .builders
            .storage_global(src_group, src_binding, Elem::Vec4U32, false)?;
        let dst = self
            .builders
            .storage_global(dst_group, dst_binding, Elem::Vec4U32, true)?;
        let src_base = self.expr(out, naga::Expression::GlobalVariable(src));
        let src_ptr = self.expr(
            out,
            naga::Expression::Access {
                base: src_base,
                index: src_index,
            },
        );
        let value = self.expr(out, naga::Expression::Load { pointer: src_ptr });
        let dst_base = self.expr(out, naga::Expression::GlobalVariable(dst));
        let dst_ptr = self.expr(
            out,
            naga::Expression::Access {
                base: dst_base,
                index: dst_index,
            },
        );
        out.push(
            naga::Statement::Store {
                pointer: dst_ptr,
                value,
            },
            naga::Span::default(),
        );
        Ok(())
    }

    pub(crate) fn store_void(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        elem: Elem,
    ) -> Result<(), TranslateError> {
        if args.len() != 4 {
            return Err(TranslateError::unsupported());
        }
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let index = self.operand(out, &args[2])?;
        let value = self.operand(out, &args[3])?;
        let global = self.builders.storage_global(group, binding, elem, true)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let pointer = self.expr(out, naga::Expression::Access { base, index });
        out.push(
            naga::Statement::Store { pointer, value },
            naga::Span::default(),
        );
        Ok(())
    }

    pub(crate) fn storage_args(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        n: usize,
    ) -> Result<(u32, u32, naga::Handle<naga::Expression>), TranslateError> {
        if args.len() != n {
            return Err(TranslateError::unsupported());
        }
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let index = self.operand(out, &args[2])?;
        Ok((group, binding, index))
    }

    pub(crate) fn vec4_f32_type(&mut self) -> naga::Handle<naga::Type> {
        self.builders.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    scalar: naga::Scalar {
                        kind: naga::ScalarKind::Float,
                        width: SCALAR_WIDTH,
                    },
                },
            },
            naga::Span::default(),
        )
    }

    pub(crate) fn vec4_u32_type(&mut self) -> naga::Handle<naga::Type> {
        self.builders.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    scalar: naga::Scalar {
                        kind: naga::ScalarKind::Uint,
                        width: SCALAR_WIDTH,
                    },
                },
            },
            naga::Span::default(),
        )
    }

    pub(crate) fn vec3_type(&mut self) -> naga::Handle<naga::Type> {
        self.builders.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Tri,
                    scalar: naga::Scalar {
                        kind: naga::ScalarKind::Float,
                        width: SCALAR_WIDTH,
                    },
                },
            },
            naga::Span::default(),
        )
    }
}
