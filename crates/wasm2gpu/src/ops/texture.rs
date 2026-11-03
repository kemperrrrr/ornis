//! Texture-load import lowering.

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
