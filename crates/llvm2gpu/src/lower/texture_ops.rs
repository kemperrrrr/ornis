//! Texture and image sampling lowered to Naga sample and load ops.

use super::VEC4_LANES;
use super::emit::const_u32;
use super::flow::vec_f32_type;
use super::helpers::UNIFORM_PREFIX;
use super::image_cache::{ImageRequest, SampleLevelKey};
use super::lowerer::Lowerer;
use super::texture::{
    COMPARE_VALUE_ARGS, OWNED_SAMPLE_ARGS, SampleMip, SampleShape, SampleTarget, TextureKind,
};
use crate::error::TranslateError;
use crate::parse::Operand;

impl<'a, 'b> Lowerer<'a, 'b> {
    /// `texture_load_2d_f32(group, binding, x, y)` at mip 0. Depth textures
    /// return the scalar WGSL `textureLoad` gives them.
    pub(crate) fn texture_load_f32(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        self.image_load_args(out, args, TextureKind::Depth)
    }

    /// `texture_load_2d_u32(group, binding, x, y)`: the first lane of a
    /// `texture_2d<u32>` texel.
    pub(crate) fn texture_load_u32(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        self.texture_load_lane_of(out, args, TextureKind::Unsigned, 0)
    }

    /// `texture_sample_level_2d_f32x4(group, binding, sampler_group,
    /// sampler_binding, x, y, lane)`: one lane of `textureSampleLevel(…, 0.0)`.
    /// The parts both sample paths share: the composed coordinate and the
    /// image and sampler globals.
    ///
    /// The first [`OWNED_SAMPLE_ARGS`] operands are always the two binding
    /// pairs, and the coordinate follows them; only what comes after the
    /// coordinate differs between a plain sample (a level) and a depth compare
    /// (a layer and a reference value).
    pub(crate) fn sample_target(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        kind: TextureKind,
        shape: SampleShape,
        arrayed: bool,
        comparison: bool,
    ) -> Result<SampleTarget, TranslateError> {
        let coordinate_lanes = shape.coordinate_lanes();
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let sampler_group = const_u32(&args[2])?;
        let sampler_binding = const_u32(&args[3])?;
        let mut components = Vec::with_capacity(coordinate_lanes);
        for index in 0..coordinate_lanes {
            components.push(self.operand(out, &args[OWNED_SAMPLE_ARGS + index])?);
        }
        let coord_ty = vec_f32_type(&mut self.builders.module, coordinate_lanes as u32)
            .ok_or_else(TranslateError::unsupported)?;
        let coordinate = self.image_expr(
            out,
            ImageRequest::Compose {
                ty: coord_ty,
                components: components.clone(),
            },
            naga::Expression::Compose {
                ty: coord_ty,
                components,
            },
        );
        let image = self
            .builders
            .texture_of(group, binding, kind, shape.dimension(), arrayed)?;
        let image = self.image_expr(
            out,
            ImageRequest::Global(image),
            naga::Expression::GlobalVariable(image),
        );
        let sampler = self
            .builders
            .sampler_of(sampler_group, sampler_binding, comparison);
        let sampler = self.image_expr(
            out,
            ImageRequest::Global(sampler),
            naga::Expression::GlobalVariable(sampler),
        );
        Ok(SampleTarget {
            coordinate,
            image,
            sampler,
        })
    }

    /// One lane of `textureSample`/`textureSampleLevel` on the image shape
    /// named by the import.
    ///
    /// The operand order is texture group, texture binding, sampler group,
    /// sampler binding, the coordinate lanes, then the mip level when the
    /// import takes one, then the lane. `Auto` leaves the level to the
    /// hardware, which needs a fragment stage: the translator emits it and
    /// naga rejects it anywhere else.
    pub(crate) fn image_sample_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        shape: SampleShape,
        level: SampleMip,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let coordinate_lanes = shape.coordinate_lanes();
        let level_args = usize::from(matches!(level, SampleMip::Operand));
        // Bindings, coordinate lanes, the level when the import takes one, and
        // the lane.
        if args.len() != OWNED_SAMPLE_ARGS + coordinate_lanes + level_args + 1 {
            return Err(TranslateError::unsupported());
        }
        let lane = const_u32(&args[args.len() - 1])?;
        if lane >= VEC4_LANES {
            return Err(TranslateError::unsupported());
        }
        let target = self.sample_target(out, args, TextureKind::Sampled, shape, false, false)?;
        let (level, level_key) = match level {
            SampleMip::Auto => (naga::SampleLevel::Auto, SampleLevelKey::Auto),
            SampleMip::Zero => {
                let zero = self.expr(out, naga::Expression::Literal(naga::Literal::F32(0.0)));
                (naga::SampleLevel::Exact(zero), SampleLevelKey::Zero)
            }
            SampleMip::Operand => {
                let lod = self.operand(out, &args[args.len() - 2])?;
                (naga::SampleLevel::Exact(lod), SampleLevelKey::Exact(lod))
            }
        };
        // `textureSample` without a level derives its mip from the
        // derivatives at the call, so it is shared only within the block
        // that emits it — which is where the per-lane imports sit.
        let sample = naga::Expression::ImageSample {
            image: target.image,
            sampler: target.sampler,
            gather: None,
            coordinate: target.coordinate,
            array_index: None,
            offset: None,
            level,
            depth_ref: None,
            clamp_to_edge: false,
        };
        let texel = self.image_expr(
            out,
            ImageRequest::Sample {
                image: target.image,
                sampler: target.sampler,
                coordinate: target.coordinate,
                array_index: None,
                depth_ref: None,
                level: level_key,
                clamp_to_edge: false,
            },
            sample,
        );
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: texel,
                index: lane,
            },
        ))
    }

    /// `textureSampleCompare` of an arrayed depth image, one `f32` result.
    ///
    /// The operands are texture group, texture binding, sampler group,
    /// sampler binding, the coordinate lanes, the array layer, then the
    /// reference value. A depth compare goes through a comparison sampler;
    /// the hardware filters the 2×2 taps into one visibility factor, which is
    /// what a soft shadow edge is made of.
    pub(crate) fn image_sample_compare(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        shape: SampleShape,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let coordinate_lanes = shape.coordinate_lanes();
        if args.len() != OWNED_SAMPLE_ARGS + coordinate_lanes + COMPARE_VALUE_ARGS {
            return Err(TranslateError::unsupported());
        }
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let sampler_group = const_u32(&args[2])?;
        let sampler_binding = const_u32(&args[3])?;
        let mut components = Vec::with_capacity(coordinate_lanes);
        for index in 0..coordinate_lanes {
            components.push(self.operand(out, &args[OWNED_SAMPLE_ARGS + index])?);
        }
        let coord_ty = vec_f32_type(&mut self.builders.module, coordinate_lanes as u32)
            .ok_or_else(TranslateError::unsupported)?;
        let coordinate = self.image_expr(
            out,
            ImageRequest::Compose {
                ty: coord_ty,
                components: components.clone(),
            },
            naga::Expression::Compose {
                ty: coord_ty,
                components,
            },
        );
        let layer = self.operand(out, &args[args.len() - 2])?;
        let reference = self.operand(out, &args[args.len() - 1])?;
        let image = self.builders.texture_of(
            group,
            binding,
            TextureKind::Depth,
            shape.dimension(),
            true,
        )?;
        let image = self.image_expr(
            out,
            ImageRequest::Global(image),
            naga::Expression::GlobalVariable(image),
        );
        let sampler = self
            .builders
            .sampler_of(sampler_group, sampler_binding, true);
        let sampler = self.image_expr(
            out,
            ImageRequest::Global(sampler),
            naga::Expression::GlobalVariable(sampler),
        );
        Ok(self.image_expr(
            out,
            ImageRequest::Sample {
                image,
                sampler,
                coordinate,
                array_index: Some(layer),
                depth_ref: Some(reference),
                level: SampleLevelKey::Zero,
                clamp_to_edge: false,
            },
            naga::Expression::ImageSample {
                image,
                sampler,
                gather: None,
                coordinate,
                array_index: Some(layer),
                offset: None,
                level: naga::SampleLevel::Zero,
                depth_ref: Some(reference),
                clamp_to_edge: false,
            },
        ))
    }

    /// `texture_load_2d_f32x4_sample(group, binding, x, y, lane, sample)`:
    /// lane `lane` of one sample of a multisampled `texture_2d<f32>`.
    pub(crate) fn texture_load_lane_sample(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != UNIFORM_PREFIX + 2 {
            return Err(TranslateError::unsupported());
        }
        let lane = const_u32(&args[UNIFORM_PREFIX])?;
        if lane >= VEC4_LANES {
            return Err(TranslateError::unsupported());
        }
        self.texture_load_sample_of(
            out,
            &args[..UNIFORM_PREFIX + 2],
            TextureKind::Sampled,
            Some(lane),
        )
    }

    /// A load of one sample of a multisampled image, optionally a lane of it.
    ///
    /// The sample index is the last argument and stays an operand: the resolve
    /// walks the samples in a loop. With a lane, the arguments are
    /// `(group, binding, x, y, lane, sample)`; the lane is the literal the texel
    /// access indexes by and the sample is what `textureLoad` reads.
    pub(crate) fn texture_load_sample_of(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        kind: TextureKind,
        lane: Option<u32>,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let expected = if lane.is_some() {
            UNIFORM_PREFIX + 2
        } else {
            UNIFORM_PREFIX + 1
        };
        if args.len() != expected {
            return Err(TranslateError::unsupported());
        }
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let sample = &args[expected - 1];
        let image = self.builders.texture(group, binding, kind)?;
        if !self.builders.sampling.multisampled_at(group, binding) {
            // The sample operand only exists on a multisampled image; a plain
            // one has mip levels instead, so the build is missing the flag.
            return Err(TranslateError::unsupported());
        }
        let texel = self.image_load(out, image, true, &args[2], &args[3], Some(sample))?;
        // An integer target is read word-wise: the SDK returns the first lane,
        // the texel load returns the whole `vec4<u32>`.
        let lane = match (kind, lane) {
            (TextureKind::Unsigned, None) => Some(0),
            _ => lane,
        };
        let Some(lane) = lane else {
            return Ok(texel);
        };
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: texel,
                index: lane,
            },
        ))
    }
}
