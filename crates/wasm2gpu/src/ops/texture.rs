//! Texture import lowering.

use crate::cx::Val;
use crate::env::Emit;
use crate::error::TranslateError;
use crate::globals;
use naga::Expression;

const TEXTURE_LOAD_MIP_LEVEL_ZERO: i32 = 0;

pub(super) fn texture_load(em: &mut Emit<'_, '_>) -> Result<(), TranslateError> {
    let y_value = em.pop()?;
    let y = em.as_uint(y_value)?;
    let x_value = em.pop()?;
    let x = em.as_uint(x_value)?;
    let binding = em.pop_imm()?;
    let group = em.pop_imm()?;
    let x = em.ex(as_signed(x));
    let y = em.ex(as_signed(y));
    let coordinate_ty = em.cx.ivec2;
    let coordinate = em.ex(Expression::Compose {
        ty: coordinate_ty,
        components: vec![x, y],
    });
    let texture = globals::texture(em.cx, group, binding);
    let image = em.ex(Expression::GlobalVariable(texture));
    let level = em.ex(Expression::Literal(naga::Literal::I32(
        TEXTURE_LOAD_MIP_LEVEL_ZERO,
    )));
    let texel = em.ex(Expression::ImageLoad {
        image,
        coordinate,
        array_index: None,
        sample: None,
        level: Some(level),
    });
    em.stack.push(Val::Vec(texel));
    Ok(())
}

fn as_signed(expr: naga::Handle<Expression>) -> Expression {
    Expression::As {
        expr,
        kind: naga::ScalarKind::Sint,
        convert: None,
    }
}

pub(super) fn dimension(em: &mut Emit<'_, '_>, axis: u32) -> Result<(), TranslateError> {
    let binding = em.pop_imm()?;
    let group = em.pop_imm()?;
    let texture = globals::texture(em.cx, group, binding);
    let image = em.ex(Expression::GlobalVariable(texture));
    let size = em.ex(Expression::ImageQuery {
        image,
        query: naga::ImageQuery::Size { level: None },
    });
    let lane = em.ex(Expression::AccessIndex {
        base: size,
        index: axis,
    });
    em.stack.push(Val::Index(lane));
    Ok(())
}

pub(super) fn sample(em: &mut Emit<'_, '_>, explicit_level: bool) -> Result<(), TranslateError> {
    let level = if explicit_level {
        let value = em.pop()?;
        Some(crate::env::float_of(value)?)
    } else {
        None
    };
    let v_value = em.pop()?;
    let u_value = em.pop()?;
    let v = crate::env::float_of(v_value)?;
    let u = crate::env::float_of(u_value)?;
    let samp_binding = em.pop_imm()?;
    let samp_group = em.pop_imm()?;
    let tex_binding = em.pop_imm()?;
    let tex_group = em.pop_imm()?;
    let texture = globals::texture(em.cx, tex_group, tex_binding);
    let sampler = globals::sampler(em.cx, samp_group, samp_binding);
    let image = em.ex(Expression::GlobalVariable(texture));
    let sampler = em.ex(Expression::GlobalVariable(sampler));
    let coordinate = em.ex(Expression::Compose {
        ty: em.cx.vec2,
        components: vec![u, v],
    });
    let sampled = em.ex(Expression::ImageSample {
        image,
        sampler,
        gather: None,
        coordinate,
        array_index: None,
        offset: None,
        level: level.map_or(naga::SampleLevel::Auto, naga::SampleLevel::Exact),
        depth_ref: None,
        clamp_to_edge: false,
    });
    em.stack.push(Val::Vec(sampled));
    Ok(())
}
