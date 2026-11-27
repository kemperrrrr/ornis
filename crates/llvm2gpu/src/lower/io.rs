//! Stage IO reads, workgroup access and vector lane arithmetic.

use super::VEC2_LANES;
use super::VEC3_LANES;
use super::emit::{const_u32, scalar};
use super::flow::vec_f32_type;
use super::helpers::{POSITION_LANES, workgroup_index};
use super::lowerer::Lowerer;
use super::stage::IoValue;
use crate::error::TranslateError;
use crate::parse::Operand;

impl<'a, 'b> Lowerer<'a, 'b> {
    /// `cross3(ax, ay, az, bx, by, bz, lane)` — one lane of the cross product.
    pub(crate) fn cross3_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let lanes = VEC3_LANES as usize;
        if args.len() != lanes + lanes + 1 {
            return Err(TranslateError::unsupported());
        }
        let lane = const_u32(&args[lanes + lanes])?;
        if lane >= VEC3_LANES {
            return Err(TranslateError::unsupported());
        }
        let va = self.vec3_of(out, args, 0)?;
        let vb = self.vec3_of(out, args, VEC3_LANES)?;
        let crossed = self.expr(
            out,
            naga::Expression::Math {
                fun: naga::MathFunction::Cross,
                arg: va,
                arg1: Some(vb),
                arg2: None,
                arg3: None,
            },
        );
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: crossed,
                index: lane,
            },
        ))
    }

    /// `normalize3(x, y, z, lane)` — one lane of the unit vector.
    pub(crate) fn normalize3_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let lanes = VEC3_LANES as usize;
        if args.len() != lanes + 1 {
            return Err(TranslateError::unsupported());
        }
        let lane = const_u32(&args[lanes])?;
        if lane >= VEC3_LANES {
            return Err(TranslateError::unsupported());
        }
        let va = self.vec3_of(out, args, 0)?;
        let unit = self.expr(
            out,
            naga::Expression::Math {
                fun: naga::MathFunction::Normalize,
                arg: va,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        );
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: unit,
                index: lane,
            },
        ))
    }

    pub(crate) fn vec2_of(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        at: u32,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let start = at as usize;
        let ty = vec_f32_type(&mut self.builders.module, VEC2_LANES)
            .ok_or_else(TranslateError::unsupported)?;
        let x = self.operand(
            out,
            args.get(start).ok_or_else(TranslateError::unsupported)?,
        )?;
        let y = self.operand(
            out,
            args.get(start + 1)
                .ok_or_else(TranslateError::unsupported)?,
        )?;
        Ok(self.expr(
            out,
            naga::Expression::Compose {
                ty,
                components: vec![x, y],
            },
        ))
    }

    pub(crate) fn vec3_of(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        at: u32,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let start = at as usize;
        let ty = self.vec3_type();
        let x = self.operand(
            out,
            args.get(start).ok_or_else(TranslateError::unsupported)?,
        )?;
        let y = self.operand(
            out,
            args.get(start + 1)
                .ok_or_else(TranslateError::unsupported)?,
        )?;
        let z = self.operand(
            out,
            args.get(start + 2)
                .ok_or_else(TranslateError::unsupported)?,
        )?;
        Ok(self.expr(
            out,
            naga::Expression::Compose {
                ty,
                components: vec![x, y, z],
            },
        ))
    }

    /// Lane of `@builtin(global_invocation_id)`. The axis is a constant `0..3`.
    pub(crate) fn global_invocation_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let axis = const_u32(args.first().ok_or_else(TranslateError::unsupported)?)?;
        if axis >= VEC3_LANES {
            return Err(TranslateError::unsupported());
        }
        let global_id = self.global_id.ok_or_else(TranslateError::unsupported)?;
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: global_id,
                index: axis,
            },
        ))
    }

    /// Lane of a declared `@location(location)` input. Both the location and
    /// the lane are constants, and a lane the declared width does not cover is
    /// rejected instead of reading an undeclared component.
    pub(crate) fn input_lane(
        &mut self,
        out: &mut naga::Block,
        location: u32,
        lane: &Operand,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let lane = const_u32(lane)?;
        let Some(IoValue::Vec(lanes)) = self.io.inputs.get(&location).copied() else {
            return Err(TranslateError::unsupported());
        };
        if lane >= lanes {
            return Err(TranslateError::unsupported());
        }
        let input = *self
            .inputs
            .get(&location)
            .ok_or_else(TranslateError::unsupported)?;
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: input,
                index: lane,
            },
        ))
    }

    /// Lane of `@builtin(position)`. The lane index is a constant `0..4`.
    pub(crate) fn fragment_position(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let lane = const_u32(args.first().ok_or_else(TranslateError::unsupported)?)?;
        if lane >= POSITION_LANES {
            return Err(TranslateError::unsupported());
        }
        let position = self.position.ok_or_else(TranslateError::unsupported)?;
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: position,
                index: lane,
            },
        ))
    }

    /// `workgroup_load_u32` or `workgroup_atomic_load_u32`.
    pub(crate) fn workgroup_load(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        atomic: bool,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let index = workgroup_index(args)?;
        let index = self.operand(out, index)?;
        let pointer = self.workgroup_slot(out, index, atomic)?;
        Ok(self.expr(out, naga::Expression::Load { pointer }))
    }

    /// `workgroup_store_u32` or `workgroup_atomic_store_u32`.
    pub(crate) fn workgroup_store(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        atomic: bool,
    ) -> Result<(), TranslateError> {
        if args.len() != 2 {
            return Err(TranslateError::unsupported());
        }
        let index = self.operand(out, &args[0])?;
        let value = self.operand(out, &args[1])?;
        let pointer = self.workgroup_slot(out, index, atomic)?;
        out.push(
            naga::Statement::Store { pointer, value },
            naga::Span::default(),
        );
        Ok(())
    }

    pub(crate) fn workgroup_slot(
        &mut self,
        out: &mut naga::Block,
        index: naga::Handle<naga::Expression>,
        atomic: bool,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let var = if atomic {
            self.builders.workgroup_atomics()?
        } else {
            self.builders.workgroup_words()?
        };
        let base = self.expr(out, naga::Expression::GlobalVariable(var));
        Ok(self.expr(out, naga::Expression::Access { base, index }))
    }

    /// `workgroup_atomic_add_u32(index, value)` returns the previous word.
    pub(crate) fn workgroup_atomic_add(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != 2 {
            return Err(TranslateError::unsupported());
        }
        let index = self.operand(out, &args[0])?;
        let value = self.operand(out, &args[1])?;
        let pointer = self.workgroup_slot(out, index, true)?;
        let ty = scalar(&mut self.builders.module, naga::ScalarKind::Uint);
        let result = self.func.expressions.append(
            naga::Expression::AtomicResult {
                ty,
                comparison: false,
            },
            naga::Span::default(),
        );
        out.push(
            naga::Statement::Atomic {
                pointer,
                fun: naga::AtomicFunction::Add,
                value,
                result: Some(result),
            },
            naga::Span::default(),
        );
        Ok(result)
    }
}
