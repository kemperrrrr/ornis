//! Detection and literal parsing for graph evaluation.
//!
//! Parses literal constant values and evaluates data-shaping nodes
//! (convert, combine, constant).

use super::{CodegenError, OutputValue};
use std::collections::HashMap;

/// Components in a MaterialX `color3` / `vector3`.
const COLOR3_COMPONENTS: usize = 3;
/// Components in a MaterialX `color4` / `vector4`.
const COLOR4_COMPONENTS: usize = 4;
/// Components in a MaterialX `vector2`.
const VECTOR2_COMPONENTS: usize = 2;

pub(crate) fn parse_constant(value: &str, ty: &str) -> Result<OutputValue, CodegenError> {
    match ty {
        "float" => parse_float_constant(value),
        "color3" => parse_color_constant(value, COLOR3_COMPONENTS),
        "color4" => parse_color_constant(value, COLOR4_COMPONENTS),
        "vector2" => parse_vector_constant(value, VECTOR2_COMPONENTS),
        "vector3" => parse_vector_constant(value, COLOR3_COMPONENTS),
        "vector4" => parse_vector_constant(value, COLOR4_COMPONENTS),
        "boolean" => parse_boolean_constant(value),
        "string" => Ok(OutputValue::String(value.to_string())),
        _ => Err(CodegenError::TypeConversion(format!(
            "unsupported type: {}",
            ty
        ))),
    }
}

fn parse_float_constant(value: &str) -> Result<OutputValue, CodegenError> {
    let v: f32 = value
        .trim()
        .parse()
        .map_err(|_| CodegenError::TypeConversion(format!("float: {}", value)))?;
    Ok(OutputValue::Float(v))
}

/// Parse `n` comma-separated floats labeled `r/g/b[/a]`.
fn parse_color_constant(value: &str, n: usize) -> Result<OutputValue, CodegenError> {
    let labels: &[&str] = if n == COLOR3_COMPONENTS {
        &["r", "g", "b"]
    } else {
        &["r", "g", "b", "a"]
    };
    let c = parse_components(
        value,
        n,
        if n == COLOR3_COMPONENTS {
            "color3"
        } else {
            "color4"
        },
        labels,
    )?;
    let mut v = [0.0_f32; COLOR4_COMPONENTS];
    v[..n].copy_from_slice(&c);
    if n == COLOR3_COMPONENTS {
        Ok(OutputValue::Color3([v[0], v[1], v[2]]))
    } else {
        Ok(OutputValue::Color4([v[0], v[1], v[2], v[3]]))
    }
}

/// Parse `n` comma-separated floats labeled `x/y/z[/w]`.
fn parse_vector_constant(value: &str, n: usize) -> Result<OutputValue, CodegenError> {
    let (ty, labels): (&str, &[&str]) = match n {
        VECTOR2_COMPONENTS => ("vector2", &["x", "y"]),
        COLOR3_COMPONENTS => ("vector3", &["x", "y", "z"]),
        _ => ("vector4", &["x", "y", "z", "w"]),
    };
    let c = parse_components(value, n, ty, labels)?;
    let mut v = [0.0_f32; COLOR4_COMPONENTS];
    v[..n].copy_from_slice(&c);
    match n {
        VECTOR2_COMPONENTS => Ok(OutputValue::Vector2([v[0], v[1]])),
        COLOR3_COMPONENTS => Ok(OutputValue::Vector3([v[0], v[1], v[2]])),
        _ => Ok(OutputValue::Vector4([v[0], v[1], v[2], v[3]])),
    }
}

fn parse_boolean_constant(value: &str) -> Result<OutputValue, CodegenError> {
    let b = match value.trim().to_lowercase().as_str() {
        "true" | "1" | "yes" => true,
        "false" | "0" | "no" => false,
        _ => return Err(CodegenError::TypeConversion(format!("boolean: {}", value))),
    };
    Ok(OutputValue::Boolean(b))
}

/// Split `value` into exactly `n` floats; component errors are reported as
/// `"<ty> <label>: <raw value>"` to match the original per-component messages.
fn parse_components(
    value: &str,
    n: usize,
    ty: &str,
    labels: &[&str],
) -> Result<Vec<f32>, CodegenError> {
    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != n {
        return Err(CodegenError::TypeConversion(format!("{}: {}", ty, value)));
    }
    parts
        .iter()
        .zip(labels)
        .map(|(part, label)| parse_component(part, &format!("{} {}", ty, label), value))
        .collect()
}

fn parse_component(part: &str, what: &str, raw: &str) -> Result<f32, CodegenError> {
    part.trim()
        .parse()
        .map_err(|_| CodegenError::TypeConversion(format!("{}: {}", what, raw)))
}

pub(crate) fn eval_convert(
    inputs: &HashMap<String, OutputValue>,
) -> Result<OutputValue, CodegenError> {
    let input = inputs
        .get("in")
        .ok_or(CodegenError::InputNotFound("in".into()))?;
    Ok(input.clone())
}

pub(crate) fn eval_combine2(
    inputs: &HashMap<String, OutputValue>,
) -> Result<OutputValue, CodegenError> {
    let in1 = inputs
        .get("in1")
        .ok_or(CodegenError::InputNotFound("in1".into()))?;
    let in2 = inputs
        .get("in2")
        .ok_or(CodegenError::InputNotFound("in2".into()))?;
    match (in1, in2) {
        (OutputValue::Float(a), OutputValue::Float(b)) => Ok(OutputValue::Vector2([*a, *b])),
        _ => Err(CodegenError::TypeConversion(
            "combine2 expects float".into(),
        )),
    }
}

pub(crate) fn eval_combine3(
    inputs: &HashMap<String, OutputValue>,
) -> Result<OutputValue, CodegenError> {
    let in1 = inputs
        .get("in1")
        .ok_or(CodegenError::InputNotFound("in1".into()))?;
    let in2 = inputs
        .get("in2")
        .ok_or(CodegenError::InputNotFound("in2".into()))?;
    let in3 = inputs
        .get("in3")
        .ok_or(CodegenError::InputNotFound("in3".into()))?;
    match (in1, in2, in3) {
        (OutputValue::Float(a), OutputValue::Float(b), OutputValue::Float(c)) => {
            Ok(OutputValue::Vector3([*a, *b, *c]))
        }
        _ => Err(CodegenError::TypeConversion(
            "combine3 expects float".into(),
        )),
    }
}

pub(crate) fn eval_combine4(
    inputs: &HashMap<String, OutputValue>,
) -> Result<OutputValue, CodegenError> {
    let in1 = inputs
        .get("in1")
        .ok_or(CodegenError::InputNotFound("in1".into()))?;
    let in2 = inputs
        .get("in2")
        .ok_or(CodegenError::InputNotFound("in2".into()))?;
    let in3 = inputs
        .get("in3")
        .ok_or(CodegenError::InputNotFound("in3".into()))?;
    let in4 = inputs
        .get("in4")
        .ok_or(CodegenError::InputNotFound("in4".into()))?;
    match (in1, in2, in3, in4) {
        (
            OutputValue::Float(a),
            OutputValue::Float(b),
            OutputValue::Float(c),
            OutputValue::Float(d),
        ) => Ok(OutputValue::Vector4([*a, *b, *c, *d])),
        _ => Err(CodegenError::TypeConversion(
            "combine4 expects float".into(),
        )),
    }
}

pub(crate) fn eval_constant(
    inputs: &HashMap<String, OutputValue>,
) -> Result<OutputValue, CodegenError> {
    if let Some(v) = inputs.get("value") {
        return Ok(v.clone());
    }
    if let Some(v) = inputs.get("in") {
        return Ok(v.clone());
    }
    Err(CodegenError::InputNotFound("constant value".into()))
}

#[cfg(test)]
mod tests {
    use crate::graph::test_helpers::{assert_close, color3, eval, eval_err, float};
    use crate::graph::{CodegenError, OutputValue};

    #[test]
    fn test_combine_and_mix() {
        let outputs = eval(
            r#"<constant name="x" type="float"><input name="value" type="float" value="0.1" /></constant>
    <constant name="y" type="float"><input name="value" type="float" value="0.2" /></constant>
    <constant name="z" type="float"><input name="value" type="float" value="0.3" /></constant>
    <constant name="w" type="float"><input name="value" type="float" value="0.4" /></constant>
    <combine2 name="c2" type="vector2"><input name="in1" type="float" nodename="x" /><input name="in2" type="float" nodename="y" /></combine2>
    <combine3 name="c3" type="vector3"><input name="in1" type="float" nodename="x" /><input name="in2" type="float" nodename="y" /><input name="in3" type="float" nodename="z" /></combine3>
    <combine4 name="c4" type="vector4"><input name="in1" type="float" nodename="x" /><input name="in2" type="float" nodename="y" /><input name="in3" type="float" nodename="z" /><input name="in4" type="float" nodename="w" /></combine4>
    <mix name="mf" type="float"><input name="fg" type="float" value="1.0" /><input name="bg" type="float" value="0.0" /><input name="mix" type="float" value="0.25" /></mix>
    <mix name="mc" type="color3"><input name="fg" type="color3" value="1.0, 1.0, 1.0" /><input name="bg" type="color3" value="0.0, 0.0, 0.0" /><input name="mix" type="float" value="0.5" /></mix>
    <layer name="ly" type="float"><input name="fg" type="float" value="1.0" /><input name="bg" type="float" value="0.0" /><input name="mix" type="float" value="0.75" /></layer>
    <open_pbr_anisotropy name="an" type="vector2"><input name="roughness" type="float" value="0.2" /><input name="anisotropy" type="float" value="0.8" /></open_pbr_anisotropy>
    <output name="out" type="float">
      <input name="c2" type="vector2" nodename="c2" />
      <input name="c3" type="vector3" nodename="c3" />
      <input name="c4" type="vector4" nodename="c4" />
      <input name="mixf" type="float" nodename="mf" />
      <input name="mixc" type="color3" nodename="mc" />
      <input name="layer" type="float" nodename="ly" />
      <input name="aniso" type="vector2" nodename="an" />
    </output>"#,
        );

        match outputs.outputs.get("c2") {
            Some(OutputValue::Vector2(v)) => assert_eq!(*v, [0.1, 0.2]),
            other => panic!("expected vector2, got {other:?}"),
        }
        match outputs.outputs.get("c3") {
            Some(OutputValue::Vector3(v)) => assert_eq!(*v, [0.1, 0.2, 0.3]),
            other => panic!("expected vector3, got {other:?}"),
        }
        match outputs.outputs.get("c4") {
            Some(OutputValue::Vector4(v)) => assert_eq!(*v, [0.1, 0.2, 0.3, 0.4]),
            other => panic!("expected vector4, got {other:?}"),
        }
        assert_close(float(&outputs, "mixf"), 0.25);
        assert_eq!(color3(&outputs, "mixc"), [0.5, 0.5, 0.5]);
        assert_close(float(&outputs, "layer"), 0.75);
        match outputs.outputs.get("aniso") {
            Some(OutputValue::Vector2(v)) => assert_eq!(*v, [0.2, 0.8]),
            other => panic!("expected vector2, got {other:?}"),
        }
    }

    #[test]
    fn test_constant_types() {
        let outputs = eval(
            r#"<constant name="f" type="float"><parameter name="value" type="float" value="3.5" /></constant>
    <constant name="c4" type="color4"><input name="value" type="color4" value="0.1, 0.2, 0.3, 0.4" /></constant>
    <constant name="v2" type="vector2"><input name="value" type="vector2" value="1.0, 2.0" /></constant>
    <constant name="v4" type="vector4"><input name="value" type="vector4" value="1.0, 2.0, 3.0, 4.0" /></constant>
    <constant name="b" type="boolean"><input name="value" type="boolean" value="true" /></constant>
    <constant name="s" type="string"><input name="value" type="string" value="R" /></constant>
    <output name="out" type="float">
      <input name="f" type="float" nodename="f" />
      <input name="c4" type="color4" nodename="c4" />
      <input name="v2" type="vector2" nodename="v2" />
      <input name="v4" type="vector4" nodename="v4" />
      <input name="b" type="boolean" nodename="b" />
      <input name="s" type="string" nodename="s" />
    </output>"#,
        );

        assert_close(float(&outputs, "f"), 3.5);
        match outputs.outputs.get("c4") {
            Some(OutputValue::Color4(v)) => assert_eq!(*v, [0.1, 0.2, 0.3, 0.4]),
            other => panic!("expected color4, got {other:?}"),
        }
        match outputs.outputs.get("v2") {
            Some(OutputValue::Vector2(v)) => assert_eq!(*v, [1.0, 2.0]),
            other => panic!("expected vector2, got {other:?}"),
        }
        match outputs.outputs.get("v4") {
            Some(OutputValue::Vector4(v)) => assert_eq!(*v, [1.0, 2.0, 3.0, 4.0]),
            other => panic!("expected vector4, got {other:?}"),
        }
        match outputs.outputs.get("b") {
            Some(OutputValue::Boolean(v)) => assert!(v),
            other => panic!("expected boolean, got {other:?}"),
        }
        match outputs.outputs.get("s") {
            Some(OutputValue::String(v)) => assert_eq!(v, "R"),
            other => panic!("expected string, got {other:?}"),
        }
    }

    #[test]
    fn test_bad_constant_value_is_error() {
        let err = eval_err(
            r#"<constant name="c" type="float"><input name="value" type="float" value="abc" /></constant>
    <output name="out" type="float"><input name="x" type="float" nodename="c" /></output>"#,
        );
        assert!(matches!(err, CodegenError::TypeConversion(_)), "{err:?}");
    }
}
