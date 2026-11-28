//! Uniform field access and plane-component reads.

use super::emit::const_u32;
use super::helpers::{UNIFORM_PREFIX, missing_struct, tag_cursor, uniform_ids};
use super::lowerer::Lowerer;
use crate::error::TranslateError;
use crate::parse::Operand;

impl<'a, 'b> Lowerer<'a, 'b> {
    /// Struct id of the element type of array field `field`.
    pub(crate) fn array_element(&self, sid: u32, field: u32) -> Result<u32, TranslateError> {
        let tags = self
            .builders
            .structs
            .get(sid as usize)
            .ok_or_else(|| missing_struct(sid))?;
        let cursor = tag_cursor(tags, field);
        if tags.get(cursor) != Some(&ornis_gpu_abi::TYPE_TAG_ARRAY) {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform field {field} of struct {sid} is not an array"
            )));
        }
        let element = *tags.get(cursor + 1).ok_or_else(|| {
            TranslateError::UnsupportedType(format!(
                "uniform array field {field} of struct {sid} has no element tag"
            ))
        })?;
        if element < ornis_gpu_abi::TYPE_TAG_NESTED {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform array field {field} of struct {sid} holds scalars, not structs"
            )));
        }
        Ok(u32::from(element - ornis_gpu_abi::TYPE_TAG_NESTED))
    }

    /// One constant `AccessIndex` step of an access path.
    #[inline(always)]
    pub(crate) fn access_field(
        &mut self,
        out: &mut naga::Block,
        base: naga::Handle<naga::Expression>,
        index: u32,
    ) -> naga::Handle<naga::Expression> {
        self.expr(out, naga::Expression::AccessIndex { base, index })
    }

    /// `uniform_field_f32/u32(group, binding, sid, field)`.
    // qual:allow(iosp) reason: "the declared tag is checked against the import, then the field is loaded from the uniform global"
    pub(crate) fn uniform_field(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        expect: u8,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != 4 {
            return Err(TranslateError::unsupported());
        }
        let (group, binding, sid, field) = uniform_ids(args)?;
        let tag = self.field_tag(sid, field)?;
        if tag != expect {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform field {field} of struct {sid} has tag {tag}, expected {expect}"
            )));
        }
        let global = self.builders.uniform_struct_global(group, binding, sid)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let pointer = self.expr(out, naga::Expression::AccessIndex { base, index: field });
        Ok(self.expr(out, naga::Expression::Load { pointer }))
    }

    /// `uniform_field_vec4_plane_dot(group, binding, sid, field, x, y, z)`.
    ///
    /// Loads one `vec4` plane and returns `dot((x, y, z, 1), plane)`, the
    /// signed plane distance. One vector uniform load and one `dot`
    /// replace four scalar uniform loads and the scalar FMA chain.
    // qual:allow(iosp) reason: "the vec4 tag is checked, then one uniform load and a dot replace four scalar loads"
    pub(crate) fn uniform_plane_dot(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != 7 {
            return Err(TranslateError::unsupported());
        }
        let (group, binding, sid, field) = uniform_ids(args)?;
        let tag = self.field_tag(sid, field)?;
        if tag != ornis_gpu_abi::TYPE_TAG_VEC4 {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform field {field} of struct {sid} is not a vec4"
            )));
        }
        let x = self.operand(out, &args[4])?;
        let y = self.operand(out, &args[5])?;
        let z = self.operand(out, &args[6])?;
        let global = self.builders.uniform_struct_global(group, binding, sid)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let field_ptr = self.expr(out, naga::Expression::AccessIndex { base, index: field });
        let plane = self.expr(out, naga::Expression::Load { pointer: field_ptr });
        let one = self.expr(out, naga::Expression::Literal(naga::Literal::F32(1.0)));
        let center_ty = self.vec4_f32_type();
        let center = self.expr(
            out,
            naga::Expression::Compose {
                ty: center_ty,
                components: vec![x, y, z, one],
            },
        );
        Ok(self.expr(
            out,
            naga::Expression::Math {
                fun: naga::MathFunction::Dot,
                arg: center,
                arg1: Some(plane),
                arg2: None,
                arg3: None,
            },
        ))
    }

    /// `uniform_field_vec4_component_f32(group, binding, sid, field, component)`.
    // qual:allow(iosp) reason: "the vec4 tag is checked, then the lane is one AccessIndex on the loaded field"
    pub(crate) fn uniform_field_vec4_component(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let (group, binding, sid, field) = uniform_ids(args)?;
        let component = const_u32(
            args.get(UNIFORM_PREFIX)
                .ok_or_else(TranslateError::unsupported)?,
        )?;
        let tag = self.field_tag(sid, field)?;
        if tag != ornis_gpu_abi::TYPE_TAG_VEC4 {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform field {field} of struct {sid} is not a vec4"
            )));
        }
        let global = self.builders.uniform_struct_global(group, binding, sid)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let vector = self.expr(out, naga::Expression::AccessIndex { base, index: field });
        let lane = self.expr(
            out,
            naga::Expression::AccessIndex {
                base: vector,
                index: component,
            },
        );
        Ok(self.expr(out, naga::Expression::Load { pointer: lane }))
    }
}
