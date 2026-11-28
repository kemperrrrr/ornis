//! Image loads, texel coordinates and numeric casts.

use super::SCALAR_WIDTH;
use super::VEC4_LANES;
use super::emit::const_u32;
use super::helpers::{UNIFORM_PREFIX, texture_xy};
use super::image_cache::ImageRequest;
use super::lowerer::Lowerer;
use super::texture::TextureKind;
use crate::error::TranslateError;
use crate::parse::Operand;

impl<'a, 'b> Lowerer<'a, 'b> {
    /// The expression `request` already emitted in this basic block, else
    /// `expr` appended now and remembered for the next lane in the block.
    pub(crate) fn image_expr(
        &mut self,
        out: &mut naga::Block,
        request: ImageRequest,
        expr: naga::Expression,
    ) -> naga::Handle<naga::Expression> {
        if let Some(handle) = self.images.get(&request) {
            return *handle;
        }
        let handle = self.expr(out, expr);
        self.images.insert(request, handle);
        handle
    }

    /// `texture_load_2d_f32x4(group, binding, x, y, lane)` at mip 0.
    /// One lane of the `vec4<f32>` a sampled `texture_2d<f32>` returns.
    pub(crate) fn texture_load_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != UNIFORM_PREFIX + 1 {
            return Err(TranslateError::unsupported());
        }
        let lane = const_u32(&args[UNIFORM_PREFIX])?;
        if lane >= VEC4_LANES {
            return Err(TranslateError::unsupported());
        }
        self.texture_load_lane_of(out, &args[..UNIFORM_PREFIX], TextureKind::Sampled, lane)
    }

    /// `textureLoad` of a `kind` image at mip 0, spelled by the arguments.
    pub(crate) fn image_load_args(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        kind: TextureKind,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let (group, binding, x, y) = texture_xy(args)?;
        let image = self.builders.texture(group, binding, kind)?;
        let multisampled = self.builders.sampling.multisampled_at(group, binding);
        self.image_load(out, image, multisampled, x, y, None)
    }

    /// `textureLoad` at mip 0, lane `lane` of the loaded texel.
    pub(crate) fn texture_load_lane_of(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        kind: TextureKind,
        lane: u32,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let texel = self.image_load_args(out, args, kind)?;
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: texel,
                index: lane,
            },
        ))
    }

    /// `textureLoad` at mip 0. Depth images yield `f32`; sampled images yield `vec4<f32>`.
    ///
    /// `sample` names the sample to read; `None` reads sample zero, which is
    /// what a pass that only needs one sample of a 4x target binds.
    pub(crate) fn image_load(
        &mut self,
        out: &mut naga::Block,
        image: naga::Handle<naga::GlobalVariable>,
        multisampled: bool,
        x: &Operand,
        y: &Operand,
        sample: Option<&Operand>,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let coordinate = self.texel_coord(out, x, y)?;
        let (sample, level) = if multisampled {
            // A multisampled image has samples, not mip levels.
            let sample = match sample {
                Some(sample) => self.operand(out, sample)?,
                None => self.expr(out, naga::Expression::Literal(naga::Literal::U32(0))),
            };
            (Some(sample), None)
        } else {
            let level = self.expr(out, naga::Expression::Literal(naga::Literal::I32(0)));
            (None, Some(level))
        };
        let handle = self.image_expr(
            out,
            ImageRequest::Global(image),
            naga::Expression::GlobalVariable(image),
        );
        Ok(self.image_expr(
            out,
            ImageRequest::Load {
                image: handle,
                coordinate,
                array_index: None,
                sample,
                level,
            },
            naga::Expression::ImageLoad {
                image: handle,
                coordinate,
                array_index: None,
                sample,
                level,
            },
        ))
    }

    /// `texture_width_2d` / `texture_height_2d(group, binding)`.
    pub(crate) fn texture_size(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        axis: u32,
        kind: TextureKind,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != 2 {
            return Err(TranslateError::unsupported());
        }
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let image = self.builders.texture(group, binding, kind)?;
        let handle = self.image_expr(
            out,
            ImageRequest::Global(image),
            naga::Expression::GlobalVariable(image),
        );
        let size = self.image_expr(
            out,
            ImageRequest::Size {
                image: handle,
                level: None,
            },
            naga::Expression::ImageQuery {
                image: handle,
                query: naga::ImageQuery::Size { level: None },
            },
        );
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: size,
                index: axis,
            },
        ))
    }

    pub(crate) fn cast(
        &mut self,
        out: &mut naga::Block,
        value: &Operand,
        kind: naga::ScalarKind,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let expr = self.operand(out, value)?;
        Ok(self.expr(
            out,
            naga::Expression::As {
                expr,
                kind,
                convert: Some(SCALAR_WIDTH),
            },
        ))
    }

    /// `sitofp`: reinterpret the stored `u32` as `i32`, then convert to
    /// `kind`; a plain `As` from `u32` would convert the unsigned value.
    pub(crate) fn cast_signed(
        &mut self,
        out: &mut naga::Block,
        value: &Operand,
        kind: naga::ScalarKind,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let expr = self.operand(out, value)?;
        let expr = self.bitcast_sint(out, expr);
        Ok(self.expr(
            out,
            naga::Expression::As {
                expr,
                kind,
                convert: Some(SCALAR_WIDTH),
            },
        ))
    }

    pub(crate) fn texel_coord(
        &mut self,
        out: &mut naga::Block,
        x: &Operand,
        y: &Operand,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let x = self.operand(out, x)?;
        let y = self.operand(out, y)?;
        let x = self.expr(
            out,
            naga::Expression::As {
                expr: x,
                kind: naga::ScalarKind::Sint,
                convert: None,
            },
        );
        let y = self.expr(
            out,
            naga::Expression::As {
                expr: y,
                kind: naga::ScalarKind::Sint,
                convert: None,
            },
        );
        let ty = self.builders.ivec2();
        Ok(self.expr(
            out,
            naga::Expression::Compose {
                ty,
                components: vec![x, y],
            },
        ))
    }
}
