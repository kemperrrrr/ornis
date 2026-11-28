//! Uniform buffer members: nested structs, arrays and their lane reads.

use super::NESTED_ARRAY_VEC4_ARGS;
use super::NESTED_VEC4_ARGS;
use super::emit::const_u32;
use super::helpers::{
    UNIFORM_PREFIX, field_tag_width, missing_field, missing_struct, tag_cursor, uniform_ids,
};
use super::lowerer::Lowerer;
use crate::error::TranslateError;
use crate::parse::Operand;

impl<'a, 'b> Lowerer<'a, 'b> {
    /// `uniform_nested_field_f32/u32(group, binding, sid, field, inner)`.
    // qual:allow(iosp) reason: "the nested tag is checked, then the load is two AccessIndex steps into the uniform global"
    pub(crate) fn uniform_nested_field(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        expect: u8,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let (group, binding, sid, field) = uniform_ids(args)?;
        let inner = const_u32(
            args.get(UNIFORM_PREFIX)
                .ok_or_else(TranslateError::unsupported)?,
        )?;
        let tag = self.field_tag(sid, field)?;
        if tag < ornis_gpu_abi::TYPE_TAG_NESTED {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform field {field} of struct {sid} is not a nested struct"
            )));
        }
        let nested = u32::from(tag - ornis_gpu_abi::TYPE_TAG_NESTED);
        let inner_tag = self.field_tag(nested, inner)?;
        if inner_tag != expect {
            return Err(TranslateError::UnsupportedType(format!(
                "nested field {inner} of struct {nested} has tag {inner_tag}, expected {expect}"
            )));
        }
        let global = self.builders.uniform_struct_global(group, binding, sid)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let outer = self.expr(out, naga::Expression::AccessIndex { base, index: field });
        let pointer = self.expr(
            out,
            naga::Expression::AccessIndex {
                base: outer,
                index: inner,
            },
        );
        Ok(self.expr(out, naga::Expression::Load { pointer }))
    }

    pub(crate) fn field_tag(&self, sid: u32, field: u32) -> Result<u8, TranslateError> {
        let tags = self
            .builders
            .structs
            .get(sid as usize)
            .ok_or_else(|| missing_struct(sid))?;
        // Fields are not one tag each (a nested struct adds its id, an array
        // adds the element tag and the length), so the ordinal has to be
        // walked rather than used as an index.
        let mut cursor = 0usize;
        let mut seen = 0u32;
        while seen < field {
            let width = field_tag_width(tags, cursor).ok_or_else(|| missing_field(sid, field))?;
            cursor += width;
            seen += 1;
        }
        tags.get(cursor)
            .copied()
            .ok_or_else(|| missing_field(sid, field))
    }

    /// `uniform_array_vec4_lane(g, b, sid, field, index, element_field, lane)`.
    ///
    /// One element of a fixed-size uniform array by a runtime index, then one
    /// `vec4` field of that element and one lane of it. The array field and the
    /// element field are immediates; only the element index is a value, which
    /// is exactly what WGSL allows in a uniform array.
    // qual:allow(iosp) reason: "the declared tags decide each step of the access path, then one load returns the lane"
    pub(crate) fn uniform_array_vec4_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != 7 {
            return Err(TranslateError::unsupported());
        }
        let (group, binding, sid, field) = uniform_ids(args)?;
        // Decoding "one value operand between two immediates" is the shape of
        // every indexed import in this file; folding it into a helper would
        // hide which argument is the index.
        // qual:allow(dry, fragment) reason: "the argument decode of an indexed import; a shared helper would hide which operand is the runtime index"
        let index = self.operand(out, &args[4])?;
        let element_field = const_u32(&args[5])?;
        let lane = const_u32(&args[6])?;
        if self.field_tag(sid, field)? != ornis_gpu_abi::TYPE_TAG_ARRAY {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform field {field} of struct {sid} is not an array"
            )));
        }
        let element = self.array_element(sid, field)?;
        if self.field_tag(element, element_field)? != ornis_gpu_abi::TYPE_TAG_VEC4 {
            return Err(TranslateError::UnsupportedType(format!(
                "field {element_field} of the array element of struct {sid} is not a vec4"
            )));
        }
        let global = self.builders.uniform_struct_global(group, binding, sid)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let array = self.access_field(out, base, field);
        let element_ptr = self.expr(out, naga::Expression::Access { base: array, index });
        let field_ptr = self.access_field(out, element_ptr, element_field);
        let lane_ptr = self.access_field(out, field_ptr, lane);
        Ok(self.expr(out, naga::Expression::Load { pointer: lane_ptr }))
    }

    /// `uniform_nested_vec4_lane(g, b, sid, field, inner, component)`.
    pub(crate) fn uniform_nested_vec4_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let (sid, field, inner, component) = self.nested_ids(args)?;
        let global =
            self.builders
                .uniform_struct_global(const_u32(&args[0])?, const_u32(&args[1])?, sid)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let ptr = self.access_nested(out, base, field, inner, component);
        Ok(self.expr(out, naga::Expression::Load { pointer: ptr }))
    }

    /// `(sid, field, inner, component)` of a nested `vec4` lane read, after
    /// checking that `field` is a struct and `inner` one of its `vec4` fields.
    pub(crate) fn nested_ids(
        &self,
        args: &[Operand],
    ) -> Result<(u32, u32, u32, u32), TranslateError> {
        if args.len() != NESTED_VEC4_ARGS {
            return Err(TranslateError::unsupported());
        }
        let (group, binding, sid, field) = uniform_ids(args)?;
        let _ = (group, binding);
        // qual:allow(dry, fragment) reason: "the argument decode of an indexed import; a shared helper would hide which operand is the runtime index"
        let inner = const_u32(&args[4])?;
        let component = const_u32(&args[5])?;
        let struct_id = self.nested_struct_id(sid, field)?;
        if self.field_tag(struct_id, inner)? != ornis_gpu_abi::TYPE_TAG_VEC4 {
            return Err(TranslateError::UnsupportedType(format!(
                "field {inner} of struct {struct_id} is not a vec4"
            )));
        }
        Ok((sid, field, inner, component))
    }

    /// Two constant `AccessIndex` steps into `field`'s struct, then the lane.
    #[inline(always)]
    pub(crate) fn access_nested(
        &mut self,
        out: &mut naga::Block,
        base: naga::Handle<naga::Expression>,
        field: u32,
        inner: u32,
        component: u32,
    ) -> naga::Handle<naga::Expression> {
        let outer = self.access_field(out, base, field);
        let vector = self.access_field(out, outer, inner);
        self.access_field(out, vector, component)
    }

    /// `uniform_nested_array_vec4_lane(g, b, sid, outer, field, index,
    /// element_field, lane)`.
    ///
    /// `outer.inner[index].element_field[lane]`: one more `AccessIndex` than
    /// the flat form, because the array lives in a nested uniform struct.
    // qual:allow(iosp) reason: "the declared tags decide each step of the access path, then one load returns the lane"
    pub(crate) fn uniform_nested_array_vec4_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != NESTED_ARRAY_VEC4_ARGS {
            return Err(TranslateError::unsupported());
        }
        let (group, binding, sid, outer) = uniform_ids(args)?;
        let field = const_u32(&args[4])?;
        // qual:allow(dry, fragment) reason: "the argument decode of an indexed import; a shared helper would hide which operand is the runtime index"
        let index = self.operand(out, &args[5])?;
        let element_field = const_u32(&args[6])?;
        let lane = const_u32(&args[7])?;
        // qual:allow(dry, fragment) reason: "the access path of this arity, stated once; a helper taking the depth as a parameter would hide which step is the dynamic index"
        let inner = self.nested_struct_id(sid, outer)?;
        if self.field_tag(inner, field)? != ornis_gpu_abi::TYPE_TAG_ARRAY {
            return Err(TranslateError::UnsupportedType(format!(
                "nested field {field} of struct {sid} is not an array"
            )));
        }
        let element = self.array_element(inner, field)?;
        if self.field_tag(element, element_field)? != ornis_gpu_abi::TYPE_TAG_VEC4 {
            return Err(TranslateError::UnsupportedType(format!(
                "field {element_field} of the nested array element is not a vec4"
            )));
        }
        let global = self.builders.uniform_struct_global(group, binding, sid)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let outer_ptr = self.access_field(out, base, outer);
        let array = self.access_field(out, outer_ptr, field);
        let element_ptr = self.expr(out, naga::Expression::Access { base: array, index });
        let field_ptr = self.access_field(out, element_ptr, element_field);
        let lane_ptr = self.access_field(out, field_ptr, lane);
        Ok(self.expr(out, naga::Expression::Load { pointer: lane_ptr }))
    }

    /// Struct id of the nested-struct field `field`.
    pub(crate) fn nested_struct_id(&self, sid: u32, field: u32) -> Result<u32, TranslateError> {
        let tags = self
            .builders
            .structs
            .get(sid as usize)
            .ok_or_else(|| missing_struct(sid))?;
        let cursor = tag_cursor(tags, field);
        let tag = *tags.get(cursor).ok_or_else(|| missing_field(sid, field))?;
        if tag < ornis_gpu_abi::TYPE_TAG_NESTED {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform field {field} of struct {sid} is not a struct"
            )));
        }
        Ok(u32::from(tag - ornis_gpu_abi::TYPE_TAG_NESTED))
    }
}
