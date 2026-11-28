//! Value and void calls: ABI import dispatch and argument decoding.

use super::MAX_UNIFORM_BYTES;
use super::UNIFORM_LANE_BYTES;
use super::VEC2_LANES;
use super::VEC3_LANES;
use super::emit::{const_u32, scalar};
use super::helpers::{
    binary_math, derivative_axis, load_global, load_uniform, min_max, ternary_math, unary_math,
};
use super::lowerer::Lowerer;
use super::stage::IoValue;
use super::stage_io::INPUT_LANE_ARGS;
use super::texture::{Elem, TextureKind, compare_import, sample_import};
use crate::error::TranslateError;
use crate::parse::Operand;
use crate::symbol;
use ornis_gpu_abi::symbols as sym;

impl<'a, 'b> Lowerer<'a, 'b> {
    /// Value-returning call: math or storage read / atomic add.
    /// Handle the void calls that are not ABI imports: a helper of this crate
    /// and the memory intrinsics. Returns whether the call was consumed.
    pub(crate) fn staged_void_call(
        &mut self,
        out: &mut naga::Block,
        callee: &str,
        args: &[Operand],
    ) -> Result<bool, TranslateError> {
        if self.builders.diverging.contains(callee) {
            // The panic path: the caller's own `unreachable` ends the
            // invocation, so the call itself has nothing to translate.
            return Ok(true);
        }
        if let Some(reason) = self.builders.skipped_helpers.get(callee) {
            return Err(TranslateError::UnsupportedType(format!(
                "`{callee}` cannot be lowered ({reason}); mark it `#[inline(always)]` so its body \
                 is compiled into the caller, or keep the pointer out of the helper's signature"
            )));
        }
        if let Some(helper) = self.builders.helpers.get(callee).copied() {
            if self.builders.barriered.contains(callee)
                && (self.divergent_depth > 0 || self.diverged)
            {
                return Err(TranslateError::UnsupportedType(format!(
                    "`{callee}` reaches workgroup_barrier and is called in divergent control flow: \
                     every invocation of a workgroup must reach a barrier together"
                )));
            }
            self.call_helper_void(out, helper, args)?;
            return Ok(true);
        }
        if symbol::imported(callee).starts_with("llvm.memset") {
            self.memset(out, args)?;
            return Ok(true);
        }
        if symbol::imported(callee).starts_with("llvm.memcpy") {
            self.memcpy(out, args)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub(crate) fn call_value(
        &mut self,
        out: &mut naga::Block,
        callee: &str,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if let Some(reason) = self.builders.skipped_helpers.get(callee) {
            return Err(TranslateError::UnsupportedType(format!(
                "`{callee}` cannot be lowered ({reason}); mark it `#[inline(always)]` so its body \
                 is compiled into the caller, or keep the pointer out of the helper's signature"
            )));
        }
        if let Some(helper) = self.builders.helpers.get(callee).copied() {
            return self.call_helper(out, helper, args);
        }
        if symbol::imported(callee).starts_with("llvm.fptosi.sat") {
            if args.len() != 1 {
                return Err(TranslateError::unsupported());
            }
            let value = self.cast(out, &args[0], naga::ScalarKind::Sint)?;
            return Ok(self.bitcast_uint(out, value));
        }
        if symbol::imported(callee).starts_with("llvm.fptoui.sat") {
            if args.len() != 1 {
                return Err(TranslateError::unsupported());
            }
            return self.cast(out, &args[0], naga::ScalarKind::Uint);
        }
        if let Some(fun) = min_max(callee) {
            if args.len() != 2 {
                return Err(TranslateError::unsupported());
            }
            // `llvm.smin`/`smax` compare signed values; WGSL picks the
            // signed overload from the operand type.
            let signed = symbol::imported(callee).starts_with("llvm.smin")
                || symbol::imported(callee).starts_with("llvm.smax");
            let mut arg = self.operand(out, &args[0])?;
            let mut arg1 = self.operand(out, &args[1])?;
            if signed {
                arg = self.bitcast_sint(out, arg);
                arg1 = self.bitcast_sint(out, arg1);
            }
            let value = self.expr(
                out,
                naga::Expression::Math {
                    fun,
                    arg,
                    arg1: Some(arg1),
                    arg2: None,
                    arg3: None,
                },
            );
            return Ok(if signed {
                self.bitcast_uint(out, value)
            } else {
                value
            });
        }
        if symbol::imported(callee) == sym::GLOBAL_INVOCATION_LANE {
            return self.global_invocation_lane(out, args);
        }
        if symbol::imported(callee) == sym::VERTEX_INDEX {
            return self.vertex_index.ok_or_else(TranslateError::unsupported);
        }
        if symbol::imported(callee) == sym::INSTANCE_INDEX {
            return self.instance_index.ok_or_else(TranslateError::unsupported);
        }
        if symbol::imported(callee) == sym::LOCAL_INVOCATION_INDEX {
            return self.local_index.ok_or_else(TranslateError::unsupported);
        }
        if symbol::imported(callee) == sym::FRAGMENT_UV_LANE {
            return self.input_lane(
                out,
                0,
                args.first().ok_or_else(TranslateError::unsupported)?,
            );
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::VERTEX_INPUT_U32X4 {
            if args.len() != 2 {
                return Err(TranslateError::unsupported());
            }
            return self.input_word_lane(
                out,
                const_u32(&args[0])?,
                const_u32(&args[1])?,
                IoValue::U32x4,
            );
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::FRAGMENT_INPUT_U32
            || symbol::imported(callee) == ornis_gpu_abi::symbols::VERTEX_INPUT_U32
        {
            return self.input_word(const_u32(
                args.first().ok_or_else(TranslateError::unsupported)?,
            )?);
        }
        if symbol::imported(callee) == sym::VERTEX_INPUT_LANE
            || symbol::imported(callee) == sym::FRAGMENT_INPUT_LANE
        {
            if args.len() != INPUT_LANE_ARGS {
                return Err(TranslateError::unsupported());
            }
            return self.input_lane(out, const_u32(&args[0])?, &args[2]);
        }
        if symbol::imported(callee) == sym::FRAGMENT_POSITION_LANE {
            return self.fragment_position(out, args);
        }
        if symbol::imported(callee) == sym::WORKGROUP_ATOMIC_ADD_U32 {
            return self.workgroup_atomic_add(out, args);
        }
        if symbol::imported(callee) == sym::WORKGROUP_ATOMIC_LOAD_U32 {
            return self.workgroup_load(out, args, true);
        }
        if symbol::imported(callee) == sym::WORKGROUP_LOAD_U32 {
            return self.workgroup_load(out, args, false);
        }
        if let Some((shape, level)) = sample_import(callee) {
            return self.image_sample_lane(out, args, shape, level);
        }
        if let Some(shape) = compare_import(callee) {
            return self.image_sample_compare(out, args, shape);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::TEXTURE_LOAD_2D_F32X4_SAMPLE {
            return self.texture_load_lane_sample(out, args);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::TEXTURE_LOAD_2D_U32_SAMPLE {
            return self.texture_load_sample_of(out, args, TextureKind::Unsigned, None);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::TEXTURE_LOAD_2D_F32_SAMPLE {
            return self.texture_load_sample_of(out, args, TextureKind::Depth, None);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::TEXTURE_LOAD_2D_U32 {
            return self.texture_load_u32(out, args);
        }
        if symbol::imported(callee) == sym::TEXTURE_LOAD_2D_F32X4 {
            return self.texture_load_lane(out, args);
        }
        if symbol::imported(callee) == sym::TEXTURE_LOAD_2D_F32 {
            return self.texture_load_f32(out, args);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::TEXTURE_WIDTH_2D_F32X4 {
            return self.texture_size(out, args, 0, TextureKind::Sampled);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::TEXTURE_HEIGHT_2D_F32X4 {
            return self.texture_size(out, args, 1, TextureKind::Sampled);
        }
        if symbol::imported(callee) == sym::TEXTURE_WIDTH_2D {
            return self.texture_size(out, args, 0, TextureKind::Depth);
        }
        if symbol::imported(callee) == sym::TEXTURE_HEIGHT_2D {
            return self.texture_size(out, args, 1, TextureKind::Depth);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::UNIFORM_NESTED_VEC4_LANE {
            return self.uniform_nested_vec4_lane(out, args);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::UNIFORM_NESTED_ARRAY_VEC4_LANE {
            return self.uniform_nested_array_vec4_lane(out, args);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::UNIFORM_ARRAY_VEC4_LANE {
            return self.uniform_array_vec4_lane(out, args);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::UNIFORM_NESTED_FIELD_F32 {
            return self.uniform_nested_field(out, args, ornis_gpu_abi::TYPE_TAG_F32);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::UNIFORM_NESTED_FIELD_U32 {
            return self.uniform_nested_field(out, args, ornis_gpu_abi::TYPE_TAG_U32);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::UNIFORM_FIELD_VEC4_PLANE_DOT {
            return self.uniform_plane_dot(out, args);
        }
        if symbol::imported(callee) == ornis_gpu_abi::symbols::UNIFORM_FIELD_VEC4_COMPONENT_F32 {
            return self.uniform_field_vec4_component(out, args);
        }
        if symbol::imported(callee) == sym::UNIFORM_FIELD_F32 {
            return self.uniform_field(out, args, ornis_gpu_abi::TYPE_TAG_F32);
        }
        if symbol::imported(callee) == sym::UNIFORM_FIELD_U32 {
            return self.uniform_field(out, args, ornis_gpu_abi::TYPE_TAG_U32);
        }
        if symbol::imported(callee) == sym::UNIFORM_READ_F32 {
            if args.len() != 4 {
                return Err(TranslateError::unsupported());
            }
            let group = const_u32(&args[0])?;
            let binding = const_u32(&args[1])?;
            let len = const_u32(&args[2])?;
            if len == 0
                || len
                    .checked_mul(UNIFORM_LANE_BYTES)
                    .is_none_or(|b| b > MAX_UNIFORM_BYTES)
            {
                return Err(TranslateError::UnsupportedType(
                    "uniform array exceeds 64KiB".to_string(),
                ));
            }
            let index = self.operand(out, &args[3])?;
            let global = self.builders.uniform_global(group, binding, len);
            return load_uniform(&mut self.func, out, global, index);
        }
        if symbol::imported(callee) == sym::STORAGE_LEN_F32 {
            if args.len() != 2 {
                return Err(TranslateError::unsupported());
            }
            let group = const_u32(&args[0])?;
            let binding = const_u32(&args[1])?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::F32, false)?;
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            return Ok(self.expr(out, naga::Expression::ArrayLength(base)));
        }
        if symbol::imported(callee) == sym::STORAGE_LEN_U32 {
            if args.len() != 2 {
                return Err(TranslateError::unsupported());
            }
            let group = const_u32(&args[0])?;
            let binding = const_u32(&args[1])?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::U32, false)?;
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            return Ok(self.expr(out, naga::Expression::ArrayLength(base)));
        }
        if symbol::imported(callee) == sym::STORAGE_LEN_U32X4 {
            if args.len() != 2 {
                return Err(TranslateError::unsupported());
            }
            let group = const_u32(&args[0])?;
            let binding = const_u32(&args[1])?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::Vec4U32, false)?;
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            return Ok(self.expr(out, naga::Expression::ArrayLength(base)));
        }
        if symbol::imported(callee) == sym::STORAGE_READ_U32_LANE {
            if args.len() != 4 {
                return Err(TranslateError::unsupported());
            }
            let group = const_u32(&args[0])?;
            let binding = const_u32(&args[1])?;
            let index = self.operand(out, &args[2])?;
            let lane = const_u32(&args[3])?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::Vec4U32, false)?;
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            let chunk = self.expr(out, naga::Expression::Access { base, index });
            let pointer = self.expr(
                out,
                naga::Expression::AccessIndex {
                    base: chunk,
                    index: lane,
                },
            );
            return Ok(self.expr(out, naga::Expression::Load { pointer }));
        }
        if symbol::imported(callee) == sym::STORAGE_READ_F32 {
            let (group, binding, index) = self.storage_args(out, args, 3)?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::F32, false)?;
            return load_global(&mut self.func, out, global, index);
        }
        if symbol::imported(callee) == sym::STORAGE_READ_U32 {
            let (group, binding, index) = self.storage_args(out, args, 3)?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::U32, false)?;
            return load_global(&mut self.func, out, global, index);
        }
        if symbol::imported(callee) == sym::STORAGE_ATOMIC_ADD_U32 {
            if args.len() != 4 {
                return Err(TranslateError::unsupported());
            }
            let group = const_u32(&args[0])?;
            let binding = const_u32(&args[1])?;
            let index = self.operand(out, &args[2])?;
            let value = self.operand(out, &args[3])?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::AtomicU32, true)?;
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            let pointer = self.expr(out, naga::Expression::Access { base, index });
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
            return Ok(result);
        }
        if let Some(axis) = derivative_axis(callee) {
            let expr = self.operand(out, &args[0])?;
            return Ok(self.expr(
                out,
                naga::Expression::Derivative {
                    axis,
                    ctrl: naga::DerivativeControl::None,
                    expr,
                },
            ));
        }
        // The SDK spells these `gpu_<op>_<shape>`; the hand-written IR of the
        // tests names the operation bare, so both are accepted.
        use ornis_gpu_abi::symbols as abi;
        if matches!(symbol::imported(callee), "cross3" | abi::GPU_CROSS3_LANE) {
            return self.cross3_lane(out, args);
        }
        if matches!(
            symbol::imported(callee),
            "normalize3" | abi::GPU_NORMALIZE3_LANE
        ) {
            return self.normalize3_lane(out, args);
        }
        if matches!(symbol::imported(callee), "dot2" | abi::GPU_DOT2_XY)
            && args.len() == VEC2_LANES as usize + VEC2_LANES as usize
        {
            let va = self.vec2_of(out, args, 0)?;
            let vb = self.vec2_of(out, args, VEC2_LANES)?;
            return Ok(self.expr(
                out,
                naga::Expression::Math {
                    fun: naga::MathFunction::Dot,
                    arg: va,
                    arg1: Some(vb),
                    arg2: None,
                    arg3: None,
                },
            ));
        }
        if matches!(symbol::imported(callee), "dot3" | abi::GPU_DOT3_XYZ)
            && args.len() == VEC3_LANES as usize + VEC3_LANES as usize
        {
            let va = self.vec3_of(out, args, 0)?;
            let vb = self.vec3_of(out, args, VEC3_LANES)?;
            return Ok(self.expr(
                out,
                naga::Expression::Math {
                    fun: naga::MathFunction::Dot,
                    arg: va,
                    arg1: Some(vb),
                    arg2: None,
                    arg3: None,
                },
            ));
        }
        if matches!(symbol::imported(callee), "length3" | abi::GPU_LENGTH3_XYZ)
            && args.len() == VEC3_LANES as usize
        {
            let va = self.vec3_of(out, args, 0)?;
            return Ok(self.expr(
                out,
                naga::Expression::Math {
                    fun: naga::MathFunction::Length,
                    arg: va,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                },
            ));
        }
        if let Some(fun) = binary_math(callee, args.len()) {
            let arg = self.operand(out, &args[0])?;
            let arg1 = self.operand(out, &args[1])?;
            return Ok(self.expr(
                out,
                naga::Expression::Math {
                    fun,
                    arg,
                    arg1: Some(arg1),
                    arg2: None,
                    arg3: None,
                },
            ));
        }
        if let Some(fun) = ternary_math(callee, args.len()) {
            let arg = self.operand(out, &args[0])?;
            let arg1 = self.operand(out, &args[1])?;
            let arg2 = self.operand(out, &args[2])?;
            return Ok(self.expr(
                out,
                naga::Expression::Math {
                    fun,
                    arg,
                    arg1: Some(arg1),
                    arg2: Some(arg2),
                    arg3: None,
                },
            ));
        }
        if args.len() == 1 {
            let Some(fun) = unary_math(callee) else {
                return Err(TranslateError::unsupported());
            };
            let arg = self.operand(out, &args[0])?;
            return Ok(self.expr(
                out,
                naga::Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                },
            ));
        }
        Err(TranslateError::unsupported())
    }
}
