//! Stage outputs and storage writes: the output struct and value stores.

use super::VEC4_LANES;
use super::emit::const_u32;
use super::flow::vec_f32_type;
use super::helpers::POSITION_LANES;
use super::lowerer::Lowerer;
use super::stage::{IoImport, IoValue, OutSlot, io_import};
use super::stage_io::location_of;
use super::texture::Elem;
use crate::error::TranslateError;
use crate::parse::Operand;
use crate::symbol;

impl<'a, 'b> Lowerer<'a, 'b> {
    pub(crate) fn call_void(
        &mut self,
        out: &mut naga::Block,
        callee: &str,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        if self.staged_void_call(out, callee, args)? {
            return Ok(());
        }
        if symbol::imported(callee) == "workgroup_atomic_store_u32" {
            return self.workgroup_store(out, args, true);
        }
        if symbol::imported(callee) == "workgroup_store_u32" {
            return self.workgroup_store(out, args, false);
        }
        if symbol::imported(callee) == "fragment_discard" {
            out.push(naga::Statement::Kill, naga::Span::default());
            return Ok(());
        }
        if symbol::imported(callee) == "workgroup_barrier" {
            // Uniformity is not proved: every invocation of a workgroup must
            // reach a barrier together, so any enclosing `if`/`loop`, or a
            // merge that a returning arm can skip, is rejected instead of
            // emitting WGSL Tint refuses (or native UB).
            if self.divergent_depth > 0 || self.diverged {
                return Err(TranslateError::unsupported());
            }
            out.push(
                naga::Statement::ControlBarrier(naga::Barrier::WORK_GROUP),
                naga::Span::default(),
            );
            return Ok(());
        }
        if let Some(import) = io_import(callee) {
            return self.stage_store(out, import, args);
        }
        if symbol::imported(callee) == "storage_write_u32x4" {
            return self.storage_write_vec4(out, args);
        }
        if symbol::imported(callee) == "storage_copy_u32x4" {
            return self.storage_copy(out, args);
        }
        if symbol::imported(callee) == "storage_write_f32" {
            return self.store_void(out, args, Elem::F32);
        }
        if symbol::imported(callee) == "storage_write_u32" {
            return self.store_void(out, args, Elem::U32);
        }
        // Intrinsics that exist for the optimiser's benefit only:
        // `llvm.assume`/`lifetime` say what the code already satisfies, and
        // `llvm.experimental.noalias.scope.decl` marks a scope by metadata.
        if symbol::imported(callee).starts_with("llvm.assume")
            || symbol::imported(callee).starts_with("llvm.lifetime")
            || symbol::imported(callee).starts_with("llvm.experimental.")
        {
            return Ok(());
        }
        Err(TranslateError::unsupported())
    }

    /// `vertex_position(x, y, z, w)` into the clip-space output slot.
    pub(crate) fn position_store(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        let member = self
            .stage_out
            .as_ref()
            .and_then(|stage_out| stage_out.position)
            .ok_or_else(TranslateError::unsupported)?;
        self.store_slot(out, OutSlot::Position, member, args)
    }

    /// Store `lanes` operand lanes into the `@location(location)` slot.
    pub(crate) fn store_lanes(
        &mut self,
        out: &mut naga::Block,
        location: u32,
        lanes: &[Operand],
    ) -> Result<(), TranslateError> {
        let member = match self
            .stage_out
            .as_ref()
            .ok_or_else(TranslateError::unsupported)?
            .locations
            .as_ref()
        {
            Some(members) => members.get(&OutSlot::Location(location)).copied(),
            // The fragment's single target stores into the returned `vec4`.
            None if location == 0 => Some(0),
            None => None,
        }
        .ok_or_else(TranslateError::unsupported)?;
        self.store_slot(out, OutSlot::Location(location), member, lanes)
    }

    /// Compose `lanes` into one `vecN<f32>` and store it into output `slot`.
    pub(crate) fn store_slot(
        &mut self,
        out: &mut naga::Block,
        slot: OutSlot,
        member: u32,
        lanes: &[Operand],
    ) -> Result<(), TranslateError> {
        let Some(stage_out) = &self.stage_out else {
            return Err(TranslateError::unsupported());
        };
        let var = stage_out.var;
        if let OutSlot::Location(location) = slot
            && self.io.outputs.get(&location) != Some(&IoValue::floats(lanes.len() as u32))
        {
            return Err(TranslateError::unsupported());
        }
        let mut components = Vec::with_capacity(lanes.len());
        for lane in lanes {
            components.push(self.operand(out, lane)?);
        }
        let width = components.len() as u32;
        let ty = vec_f32_type(&mut self.builders.module, width)
            .ok_or_else(TranslateError::unsupported)?;
        let value = self.expr(out, naga::Expression::Compose { ty, components });
        self.write_slot(out, var, member, value)
    }

    /// Store one `u32` into output `slot` (a flat varying or an integer target).
    pub(crate) fn store_scalar(
        &mut self,
        out: &mut naga::Block,
        location: u32,
        value: &Operand,
    ) -> Result<(), TranslateError> {
        let slot = OutSlot::Location(location);
        if self.io.outputs.get(&location) != Some(&IoValue::U32) {
            return Err(TranslateError::unsupported());
        }
        let member = self
            .stage_out
            .as_ref()
            .and_then(|stage_out| {
                stage_out
                    .locations
                    .as_ref()
                    .and_then(|members| members.get(&slot).copied())
            })
            .ok_or_else(TranslateError::unsupported)?;
        let value = self.operand(out, value)?;
        let var = self
            .stage_out
            .as_ref()
            .ok_or_else(TranslateError::unsupported)?
            .var;
        self.write_slot(out, var, member, value)
    }

    /// Store `value` into member `member` of the stage-output local `var`.
    pub(crate) fn write_slot(
        &mut self,
        out: &mut naga::Block,
        var: naga::Handle<naga::LocalVariable>,
        member: u32,
        value: naga::Handle<naga::Expression>,
    ) -> Result<(), TranslateError> {
        let base = self.expr(out, naga::Expression::LocalVariable(var));
        let pointer = if self
            .stage_out
            .as_ref()
            .is_some_and(|stage_out| stage_out.locations.is_none())
        {
            base
        } else {
            self.expr(
                out,
                naga::Expression::AccessIndex {
                    base,
                    index: member,
                },
            )
        };
        out.push(
            naga::Statement::Store { pointer, value },
            naga::Span::default(),
        );
        Ok(())
    }

    /// Store one stage-interface output the body writes.
    pub(crate) fn stage_store(
        &mut self,
        out: &mut naga::Block,
        import: IoImport,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        match import {
            IoImport::VertexPosition => {
                if args.len() != POSITION_LANES as usize {
                    return Err(TranslateError::unsupported());
                }
                self.position_store(out, args)
            }
            IoImport::VertexOutputU32 | IoImport::FragmentOutputU32 => {
                if args.len() != 2 {
                    return Err(TranslateError::unsupported());
                }
                self.store_scalar(out, location_of(args)?, &args[1])
            }
            IoImport::FragmentOutput0 => {
                if args.len() != VEC4_LANES as usize {
                    return Err(TranslateError::unsupported());
                }
                self.store_lanes(out, 0, args)
            }
            IoImport::VertexOutput2
            | IoImport::VertexOutput3
            | IoImport::VertexOutput4
            | IoImport::FragmentOutput
            | IoImport::FragmentOutput2 => {
                if args.len() < 2 {
                    return Err(TranslateError::unsupported());
                }
                self.store_lanes(out, location_of(args)?, &args[1..])
            }
            _ => Err(TranslateError::unsupported()),
        }
    }

    /// The scalar `u32` fragment input at `location`.
    pub(crate) fn input_word(
        &mut self,
        location: u32,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if self.io.inputs.get(&location) != Some(&IoValue::U32) {
            return Err(TranslateError::unsupported());
        }
        self.inputs
            .get(&location)
            .copied()
            .ok_or_else(TranslateError::unsupported)
    }

    /// Lane `lane` of the integer `@location(location)` input declared `kind`.
    pub(crate) fn input_word_lane(
        &mut self,
        out: &mut naga::Block,
        location: u32,
        lane: u32,
        kind: IoValue,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if self.io.inputs.get(&location) != Some(&kind) || lane >= VEC4_LANES {
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

    /// `storage_write_u32x4(group, binding, index, x, y, z, w)`.
    pub(crate) fn storage_write_vec4(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        if args.len() != 7 {
            return Err(TranslateError::unsupported());
        }
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let index = self.operand(out, &args[2])?;
        let x = self.operand(out, &args[3])?;
        let y = self.operand(out, &args[4])?;
        let z = self.operand(out, &args[5])?;
        let w = self.operand(out, &args[6])?;
        let vec_ty = self.vec4_u32_type();
        let value = self.expr(
            out,
            naga::Expression::Compose {
                ty: vec_ty,
                components: vec![x, y, z, w],
            },
        );
        let global = self
            .builders
            .storage_global(group, binding, Elem::Vec4U32, true)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let pointer = self.expr(out, naga::Expression::Access { base, index });
        out.push(
            naga::Statement::Store { pointer, value },
            naga::Span::default(),
        );
        Ok(())
    }
}
