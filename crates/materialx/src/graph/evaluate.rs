//! Evaluation of node graphs to output values.
//!
//! Drives connected nodes to values, dispatching arithmetic, data-shaping
//! and shading nodes to their handlers.

use super::parse::{eval_combine2, eval_combine3, eval_combine4, eval_constant, eval_convert};
use super::{CodegenError, EvaluatedGraph, MaterialXConverter, OutputValue};
use crate::nodes::{MtlxNodeKind, Node, NodeGraph};
use std::collections::HashMap;

pub(crate) struct GraphEvaluator<'a> {
    pub(crate) converter: &'a MaterialXConverter,
    pub(crate) graph: &'a NodeGraph,
    pub(crate) node_values: HashMap<String, OutputValue>,
    pub(crate) visited: HashMap<String, VisitState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VisitState {
    Visiting,
    Visited,
}

impl<'a> GraphEvaluator<'a> {
    pub(crate) fn new(converter: &'a MaterialXConverter, graph: &'a NodeGraph) -> Self {
        Self {
            converter,
            graph,
            node_values: HashMap::new(),
            visited: HashMap::new(),
        }
    }

    pub(crate) fn evaluate(&mut self) -> Result<EvaluatedGraph, CodegenError> {
        let output_nodes: Vec<&Node> = self
            .graph
            .nodes
            .iter()
            .filter(|n| n.node_type == MtlxNodeKind::Output)
            .collect();

        for output_node in output_nodes {
            self.evaluate_node(output_node)?;
        }

        for node in &self.graph.nodes {
            if matches!(node.node_type, MtlxNodeKind::Surface | MtlxNodeKind::Edf) {
                self.evaluate_node(node)?;
            }
        }

        let mut outputs = HashMap::new();
        for node in &self.graph.nodes {
            if node.node_type == MtlxNodeKind::Output {
                // `<output ... nodename="..."/>` references its source node
                // via the attribute instead of a child `<input>`.
                if !node.nodename.is_empty()
                    && let Some(value) = self.node_values.get(&node.nodename)
                {
                    outputs.insert(node.name.clone(), value.clone());
                }
                for input in &node.inputs {
                    if let Some(value) = self.node_values.get(&input.nodename) {
                        outputs.insert(input.name.clone(), value.clone());
                    }
                }
            }
        }

        Ok(EvaluatedGraph { outputs })
    }

    pub(crate) fn evaluate_node(&mut self, node: &Node) -> Result<OutputValue, CodegenError> {
        self.check_visit_state(node)?;
        self.visited.insert(node.name.clone(), VisitState::Visiting);

        let node_def = self
            .converter
            .node_defs
            .get(node.node_type.as_str())
            .ok_or_else(|| CodegenError::NodeDefNotFound {
                node_type: node.node_type.as_str().to_owned(),
                node: node.name.clone(),
            })?;

        let input_values = self.collect_input_values(node, node_def)?;
        let result = self.dispatch_node(node, &input_values)?;

        self.node_values.insert(node.name.clone(), result.clone());
        self.visited.insert(node.name.clone(), VisitState::Visited);
        Ok(result)
    }

    /// Route a fully-evaluated node to its handler by category.
    fn dispatch_node(
        &mut self,
        node: &Node,
        inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        let ty = node.node_type.as_str();
        if is_arithmetic_node(ty) {
            return eval_arithmetic(ty, inputs);
        }
        if is_data_node(ty) {
            return eval_data(ty, inputs);
        }
        self.dispatch_shading(node, inputs)
    }

    /// Surface/BSDF/EDF/VDF nodes plus graph `output` pass-throughs.
    fn dispatch_shading(
        &mut self,
        node: &Node,
        inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        match &node.node_type {
            MtlxNodeKind::OpenPbrSurface => Ok(OutputValue::String("surface".to_string())),
            MtlxNodeKind::Surface => Ok(OutputValue::BSDF(node.name.clone())),
            MtlxNodeKind::OrenNayarDiffuseBsdf => Ok(OutputValue::BSDF("oren_nayar".to_string())),
            MtlxNodeKind::DielectricBsdf => Ok(OutputValue::BSDF("dielectric".to_string())),
            MtlxNodeKind::GeneralizedSchlickBsdf => Ok(OutputValue::BSDF("schlick".to_string())),
            MtlxNodeKind::SheenBsdf => Ok(OutputValue::BSDF("sheen".to_string())),
            MtlxNodeKind::ThinFilmBsdf => Ok(OutputValue::BSDF("thin_film".to_string())),
            MtlxNodeKind::TranslucentBsdf => Ok(OutputValue::BSDF("translucent".to_string())),
            MtlxNodeKind::SubsurfaceBsdf => Ok(OutputValue::BSDF("subsurface".to_string())),
            MtlxNodeKind::AnisotropicVdf => Ok(OutputValue::VDF("anisotropic".to_string())),
            MtlxNodeKind::UniformEdf => Ok(OutputValue::EDF("uniform".to_string())),
            MtlxNodeKind::GeneralizedSchlickEdf => Ok(OutputValue::EDF("schlick".to_string())),
            MtlxNodeKind::Output => self.eval_output_node(node, inputs),
            other => Err(CodegenError::UnsupportedNode(other.as_str().to_owned())),
        }
    }

    /// `<output>` passes through its BSDF/EDF input or the node named by
    /// the `nodename` attribute.
    fn eval_output_node(
        &mut self,
        node: &Node,
        inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        if let Some(bsdf_input) = inputs.get("bsdf") {
            return Ok(bsdf_input.clone());
        }
        if let Some(edf_input) = inputs.get("edf") {
            return Ok(edf_input.clone());
        }
        if !node.nodename.is_empty() {
            return self.evaluate_connected(&node.nodename);
        }
        Ok(OutputValue::String("output".to_string()))
    }
}

fn is_arithmetic_node(ty: &str) -> bool {
    matches!(
        ty,
        "multiply"
            | "add"
            | "divide"
            | "subtract"
            | "invert"
            | "clamp"
            | "min"
            | "max"
            | "power"
            | "sqrt"
            | "ifgreater"
    )
}

fn is_data_node(ty: &str) -> bool {
    matches!(
        ty,
        "mix"
            | "layer"
            | "open_pbr_anisotropy"
            | "convert"
            | "combine2"
            | "combine3"
            | "combine4"
            | "constant"
    )
}

/// Arithmetic nodes with float/color3/vector3 element-wise semantics.
fn eval_arithmetic(
    ty: &str,
    inputs: &HashMap<String, OutputValue>,
) -> Result<OutputValue, CodegenError> {
    match ty {
        "multiply" | "add" | "divide" | "subtract" => eval_math(ty, inputs),
        "invert" => eval_invert(inputs),
        "clamp" => eval_clamp(inputs),
        "min" | "max" => eval_minmax(ty, inputs),
        "power" => eval_power(inputs),
        "sqrt" => eval_sqrt(inputs),
        "ifgreater" => eval_ifgreater(inputs),
        _ => Err(CodegenError::UnsupportedNode(ty.to_string())),
    }
}

/// Data-shaping nodes: conversion, combining, mixing, constants.
fn eval_data(ty: &str, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
    match ty {
        "convert" => eval_convert(inputs),
        "combine2" => eval_combine2(inputs),
        "combine3" => eval_combine3(inputs),
        "combine4" => eval_combine4(inputs),
        "constant" => eval_constant(inputs),
        "mix" | "layer" => eval_mix(inputs),
        "open_pbr_anisotropy" => eval_anisotropy(inputs),
        _ => Err(CodegenError::UnsupportedNode(ty.to_string())),
    }
}

fn eval_math(op: &str, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
    let in1 = inputs
        .get("in1")
        .or_else(|| inputs.get("in"))
        .ok_or(CodegenError::InputNotFound("in1".into()))?;
    let in2 = inputs
        .get("in2")
        .ok_or(CodegenError::InputNotFound("in2".into()))?;

    match (in1, in2) {
        (OutputValue::Float(a), OutputValue::Float(b)) => {
            let result = match op {
                "multiply" => a * b,
                "add" => a + b,
                "divide" => a / b,
                "subtract" => a - b,
                _ => return Err(CodegenError::UnsupportedNode(op.into())),
            };
            Ok(OutputValue::Float(result))
        }
        (OutputValue::Color3(a), OutputValue::Color3(b)) => {
            let result = match op {
                "multiply" => [a[0] * b[0], a[1] * b[1], a[2] * b[2]],
                "add" => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                "divide" => [a[0] / b[0], a[1] / b[1], a[2] / b[2]],
                "subtract" => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
                _ => return Err(CodegenError::UnsupportedNode(op.into())),
            };
            Ok(OutputValue::Color3(result))
        }
        (OutputValue::Vector3(a), OutputValue::Vector3(b)) => {
            let result = match op {
                "multiply" => [a[0] * b[0], a[1] * b[1], a[2] * b[2]],
                "add" => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                "divide" => [a[0] / b[0], a[1] / b[1], a[2] / b[2]],
                "subtract" => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
                _ => return Err(CodegenError::UnsupportedNode(op.into())),
            };
            Ok(OutputValue::Vector3(result))
        }
        _ => Err(CodegenError::TypeConversion(
            "mismatched types for math op".into(),
        )),
    }
}

fn eval_invert(inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
    let input = inputs
        .get("in")
        .ok_or(CodegenError::InputNotFound("in".into()))?;
    match input {
        OutputValue::Float(v) => Ok(OutputValue::Float(1.0 - v)),
        OutputValue::Color3(v) => Ok(OutputValue::Color3([1.0 - v[0], 1.0 - v[1], 1.0 - v[2]])),
        _ => Err(CodegenError::TypeConversion(
            "invert expects float or color3".into(),
        )),
    }
}

fn eval_clamp(inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
    let input = inputs
        .get("in")
        .ok_or(CodegenError::InputNotFound("in".into()))?;
    let low = inputs
        .get("low")
        .ok_or(CodegenError::InputNotFound("low".into()))?;
    let high = inputs
        .get("high")
        .ok_or(CodegenError::InputNotFound("high".into()))?;

    match (input, low, high) {
        (OutputValue::Float(v), OutputValue::Float(l), OutputValue::Float(h)) => {
            Ok(OutputValue::Float(v.clamp(*l, *h)))
        }
        (OutputValue::Color3(v), OutputValue::Float(l), OutputValue::Float(h)) => {
            Ok(OutputValue::Color3([
                v[0].clamp(*l, *h),
                v[1].clamp(*l, *h),
                v[2].clamp(*l, *h),
            ]))
        }
        _ => Err(CodegenError::TypeConversion("clamp type mismatch".into())),
    }
}

fn eval_minmax(
    op: &str,
    inputs: &HashMap<String, OutputValue>,
) -> Result<OutputValue, CodegenError> {
    let in1 = inputs
        .get("in1")
        .ok_or(CodegenError::InputNotFound("in1".into()))?;
    let in2 = inputs
        .get("in2")
        .ok_or(CodegenError::InputNotFound("in2".into()))?;

    match (in1, in2) {
        (OutputValue::Float(a), OutputValue::Float(b)) => {
            let result = if op == "max" { a.max(*b) } else { a.min(*b) };
            Ok(OutputValue::Float(result))
        }
        (OutputValue::Color3(a), OutputValue::Color3(b)) => {
            let result = if op == "max" {
                [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]
            } else {
                [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])]
            };
            Ok(OutputValue::Color3(result))
        }
        _ => Err(CodegenError::TypeConversion("minmax type mismatch".into())),
    }
}

fn eval_power(inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
    let input = inputs
        .get("in")
        .ok_or(CodegenError::InputNotFound("in".into()))?;
    let exp = inputs
        .get("exponent")
        .ok_or(CodegenError::InputNotFound("exponent".into()))?;

    match (input, exp) {
        (OutputValue::Float(b), OutputValue::Float(e)) => Ok(OutputValue::Float(b.powf(*e))),
        _ => Err(CodegenError::TypeConversion("power expects float".into())),
    }
}

fn eval_sqrt(inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
    let input = inputs
        .get("in")
        .ok_or(CodegenError::InputNotFound("in".into()))?;
    match input {
        OutputValue::Float(v) => Ok(OutputValue::Float(v.sqrt())),
        _ => Err(CodegenError::TypeConversion("sqrt expects float".into())),
    }
}

fn eval_ifgreater(inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
    let v1 = inputs
        .get("value1")
        .ok_or(CodegenError::InputNotFound("value1".into()))?;
    let v2 = inputs
        .get("value2")
        .ok_or(CodegenError::InputNotFound("value2".into()))?;
    let in1 = inputs
        .get("in1")
        .ok_or(CodegenError::InputNotFound("in1".into()))?;
    let in2 = inputs
        .get("in2")
        .ok_or(CodegenError::InputNotFound("in2".into()))?;

    match (v1, v2) {
        (OutputValue::Float(a), OutputValue::Float(b)) => {
            if a > b {
                Ok(in1.clone())
            } else {
                Ok(in2.clone())
            }
        }
        _ => Err(CodegenError::TypeConversion(
            "ifgreater expects float".into(),
        )),
    }
}

fn eval_anisotropy(inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
    let roughness = get_float(inputs, "roughness")?;
    let anisotropy = get_float(inputs, "anisotropy")?;
    Ok(OutputValue::Vector2([roughness, anisotropy]))
}

fn eval_mix(inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
    let fg = inputs
        .get("fg")
        .ok_or(CodegenError::InputNotFound("fg".into()))?;
    let bg = inputs
        .get("bg")
        .ok_or(CodegenError::InputNotFound("bg".into()))?;
    let mix = get_float(inputs, "mix")?;

    match (fg, bg) {
        (OutputValue::Float(f), OutputValue::Float(b)) => {
            Ok(OutputValue::Float(b * (1.0 - mix) + f * mix))
        }
        (OutputValue::Color3(f), OutputValue::Color3(b)) => Ok(OutputValue::Color3([
            b[0] * (1.0 - mix) + f[0] * mix,
            b[1] * (1.0 - mix) + f[1] * mix,
            b[2] * (1.0 - mix) + f[2] * mix,
        ])),
        _ => Err(CodegenError::TypeConversion("mix type mismatch".into())),
    }
}

fn get_float(inputs: &HashMap<String, OutputValue>, name: &str) -> Result<f32, CodegenError> {
    inputs
        .get(name)
        .and_then(|v| match v {
            OutputValue::Float(f) => Some(*f),
            _ => None,
        })
        .ok_or_else(|| CodegenError::InputNotFound(name.into()))
}

#[cfg(test)]
mod tests {
    use crate::graph::OutputValue;
    use crate::graph::test_helpers::{assert_close, color3, eval, float};

    #[test]
    fn test_math_ops_float() {
        let outputs = eval(
            r#"<constant name="two" type="float"><input name="value" type="float" value="2.0" /></constant>
    <constant name="six" type="float"><input name="value" type="float" value="6.0" /></constant>
    <add name="a" type="float"><input name="in1" type="float" nodename="two" /><input name="in2" type="float" nodename="six" /></add>
    <subtract name="s" type="float"><input name="in1" type="float" nodename="two" /><input name="in2" type="float" nodename="six" /></subtract>
    <multiply name="m" type="float"><input name="in1" type="float" nodename="two" /><input name="in2" type="float" nodename="six" /></multiply>
    <divide name="d" type="float"><input name="in1" type="float" nodename="six" /><input name="in2" type="float" nodename="two" /></divide>
    <output name="out" type="float">
      <input name="add" type="float" nodename="a" />
      <input name="sub" type="float" nodename="s" />
      <input name="mul" type="float" nodename="m" />
      <input name="div" type="float" nodename="d" />
    </output>"#,
        );

        assert_close(float(&outputs, "add"), 8.0);
        assert_close(float(&outputs, "sub"), -4.0);
        assert_close(float(&outputs, "mul"), 12.0);
        assert_close(float(&outputs, "div"), 3.0);
    }

    #[test]
    fn test_math_ops_color3() {
        let outputs = eval(
            r#"<constant name="c1" type="color3"><input name="value" type="color3" value="1.0, 2.0, 3.0" /></constant>
    <constant name="c2" type="color3"><input name="value" type="color3" value="0.5, 0.5, 2.0" /></constant>
    <multiply name="m" type="color3"><input name="in1" type="color3" nodename="c1" /><input name="in2" type="color3" nodename="c2" /></multiply>
    <subtract name="s" type="color3"><input name="in1" type="color3" nodename="c1" /><input name="in2" type="color3" nodename="c2" /></subtract>
    <output name="out" type="color3">
      <input name="mul" type="color3" nodename="m" />
      <input name="sub" type="color3" nodename="s" />
    </output>"#,
        );

        assert_eq!(color3(&outputs, "mul"), [0.5, 1.0, 6.0]);
        assert_eq!(color3(&outputs, "sub"), [0.5, 1.5, 1.0]);
    }

    #[test]
    fn test_unary_and_comparison_ops() {
        let outputs = eval(
            r#"<invert name="inv" type="float"><input name="in" type="float" value="0.25" /></invert>
    <invert name="invc" type="color3"><input name="in" type="color3" value="0.2, 0.4, 0.6" /></invert>
    <clamp name="cl" type="float"><input name="in" type="float" value="1.5" /><input name="low" type="float" value="0.0" /><input name="high" type="float" value="1.0" /></clamp>
    <max name="mx" type="float"><input name="in1" type="float" value="0.3" /><input name="in2" type="float" value="0.6" /></max>
    <min name="mn" type="float"><input name="in1" type="float" value="0.3" /><input name="in2" type="float" value="0.6" /></min>
    <power name="pw" type="float"><input name="in" type="float" value="0.5" /><input name="exponent" type="float" value="2.0" /></power>
    <sqrt name="sq" type="float"><input name="in" type="float" value="0.64" /></sqrt>
    <ifgreater name="ifg" type="float"><input name="value1" type="float" value="2.0" /><input name="value2" type="float" value="1.0" /><input name="in1" type="float" value="0.9" /><input name="in2" type="float" value="0.1" /></ifgreater>
    <convert name="cv" type="float"><input name="in" type="float" value="0.42" /></convert>
    <output name="out" type="float">
      <input name="inv" type="float" nodename="inv" />
      <input name="invc" type="color3" nodename="invc" />
      <input name="clamp" type="float" nodename="cl" />
      <input name="max" type="float" nodename="mx" />
      <input name="min" type="float" nodename="mn" />
      <input name="power" type="float" nodename="pw" />
      <input name="sqrt" type="float" nodename="sq" />
      <input name="ifgreater" type="float" nodename="ifg" />
      <input name="convert" type="float" nodename="cv" />
    </output>"#,
        );

        assert_close(float(&outputs, "inv"), 0.75);
        let invc = color3(&outputs, "invc");
        for (got, want) in invc.iter().zip([0.8, 0.6, 0.4]) {
            assert_close(*got, want);
        }
        assert_close(float(&outputs, "clamp"), 1.0);
        assert_close(float(&outputs, "max"), 0.6);
        assert_close(float(&outputs, "min"), 0.3);
        assert_close(float(&outputs, "power"), 0.25);
        assert_close(float(&outputs, "sqrt"), 0.8);
        assert_close(float(&outputs, "ifgreater"), 0.9);
        assert_close(float(&outputs, "convert"), 0.42);
    }

    /// attribute (no child `<input>`) must resolve to the referenced node,
    /// exposed under the output's own name.
    #[test]
    fn test_output_nodename_connection() {
        let outputs = eval(
            r#"<constant name="two" type="float"><input name="value" type="float" value="2.0" /></constant>
    <constant name="six" type="float"><input name="value" type="float" value="6.0" /></constant>
    <multiply name="m" type="float"><input name="in1" type="float" nodename="two" /><input name="in2" type="float" nodename="six" /></multiply>
    <output name="result" type="float" nodename="m" />"#,
        );

        assert_close(float(&outputs, "result"), 12.0);
    }

    /// The `nodename` connection is followed transitively: the referenced
    /// node's own subtree is evaluated as well.
    #[test]
    fn test_output_nodename_evaluates_upstream_graph() {
        let outputs = eval(
            r#"<constant name="c" type="color3"><input name="value" type="color3" value="0.8, 0.4, 0.2" /></constant>
    <invert name="inv" type="color3"><input name="in" type="color3" nodename="c" /></invert>
    <output name="base_color" type="color3" nodename="inv" />"#,
        );

        let color = color3(&outputs, "base_color");
        for (got, want) in color.iter().zip([0.2, 0.6, 0.8]) {
            assert_close(*got, want);
        }
    }

    #[test]
    fn test_bsdf_edf_surface_nodes() {
        let outputs = eval(
            r#"<dielectric_bsdf name="d" type="BSDF"><input name="weight" type="float" value="1.0" /></dielectric_bsdf>
    <oren_nayar_diffuse_bsdf name="o" type="BSDF"><input name="weight" type="float" value="1.0" /></oren_nayar_diffuse_bsdf>
    <uniform_edf name="e" type="EDF"><input name="color" type="color3" value="1.0, 1.0, 1.0" /></uniform_edf>
    <generalized_schlick_edf name="se" type="EDF"><input name="color" type="color3" value="1.0, 1.0, 1.0" /></generalized_schlick_edf>
    <anisotropic_vdf name="v" type="VDF"><input name="color" type="color3" value="1.0, 1.0, 1.0" /></anisotropic_vdf>
    <surface name="surf" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="d" /></surface>
    <output name="out" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="d" /></output>"#,
        );

        match outputs.outputs.get("bsdf") {
            Some(OutputValue::BSDF(v)) => assert_eq!(v, "dielectric"),
            other => panic!("expected BSDF, got {other:?}"),
        }
    }
}
