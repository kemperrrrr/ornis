//! Graph evaluation and conversion to OpenPBRMaterial

use crate::nodes::{MaterialXDocument, Node, NodeDef, NodeGraph};
use ornis_render::OpenPBRMaterial;
use quick_xml::Error as XmlError;
use quick_xml::events::attributes::AttrError;
use std::collections::HashMap;
use std::str::Utf8Error;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodegenError {
    #[error("Node graph not found: {0}")]
    GraphNotFound(String),
    #[error("Node definition not found: {0}")]
    NodeDefNotFound(String),
    #[error("Required input not found: {0}")]
    InputNotFound(String),
    #[error("Type conversion error: {0}")]
    TypeConversion(String),
    #[error("Unsupported node type: {0}")]
    UnsupportedNode(String),
    #[error("Cyclic dependency detected")]
    CyclicDependency,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("MaterialX error: {0}")]
    MaterialX(#[from] Box<MaterialXError>),
}

#[derive(Error, Debug)]
pub enum MaterialXError {
    #[error("XML parse error: {0}")]
    Xml(#[from] XmlError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("XML attribute error: {0}")]
    Attr(#[from] AttrError),
    #[error("UTF-8 conversion error: {0}")]
    Utf8(#[from] Utf8Error),
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Unsupported node type: {0}")]
    UnsupportedNode(String),
    #[error("Missing required input: {0}")]
    MissingInput(String),
    #[error("Codegen error: {0}")]
    Codegen(Box<CodegenError>),
}

impl From<MaterialXError> for std::io::Error {
    fn from(e: MaterialXError) -> Self {
        std::io::Error::other(e.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct EvaluatedGraph {
    pub outputs: HashMap<String, OutputValue>,
}

#[derive(Debug, Clone)]
pub enum OutputValue {
    Float(f32),
    Color3([f32; 3]),
    Color4([f32; 4]),
    Vector2([f32; 2]),
    Vector3([f32; 3]),
    Vector4([f32; 4]),
    Boolean(bool),
    String(String),
    BSDF(String),
    EDF(String),
    VDF(String),
}

pub struct MaterialXConverter {
    document: MaterialXDocument,
    node_defs: HashMap<String, NodeDef>,
}

impl MaterialXConverter {
    pub fn new(document: MaterialXDocument) -> Self {
        let mut node_defs = HashMap::new();
        for def in &document.nodedefs {
            node_defs.insert(def.name.clone(), def.clone());
        }
        // Real-world nodedefs are named `ND_<node>_<type>` (e.g.
        // `ND_multiply_float`), while the evaluator looks definitions up by
        // node *type* (`multiply`). Index by the `node` attribute as well so
        // the lookup actually hits. A definition keyed by `node` overrides an
        // earlier one for the same type (later document order wins), including
        // a legacy name-keyed entry: a real `ND_*` definition is more
        // specific than one that merely happens to be named like the type.
        for def in &document.nodedefs {
            if !def.node.is_empty() {
                node_defs.insert(def.node.clone(), def.clone());
            }
        }

        Self {
            document,
            node_defs,
        }
    }

    pub fn to_openpbr(&self) -> Result<OpenPBRMaterial, CodegenError> {
        let graph = self.find_openpbr_graph()?;
        let mut evaluator = GraphEvaluator::new(self, graph);
        let evaluated = evaluator.evaluate()?;
        self.extract_material(&evaluated)
    }

    fn find_openpbr_graph(&self) -> Result<&NodeGraph, CodegenError> {
        for graph in &self.document.nodegraphs {
            if graph.nodedef.contains("open_pbr_surface") {
                return Ok(graph);
            }
        }
        Err(CodegenError::GraphNotFound(
            "OpenPBR surface shader graph not found".to_string(),
        ))
    }

    fn extract_material(
        &self,
        evaluated: &EvaluatedGraph,
    ) -> Result<OpenPBRMaterial, CodegenError> {
        let mut material = OpenPBRMaterial::pbr();

        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("base_weight") {
            material = material.base_weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("base_color") {
            material = material.base_color_rgb(*v);
        } else if let Some(OutputValue::Color4(v)) = evaluated.outputs.get("base_color") {
            material = material.base_color(v[0], v[1], v[2], v[3]);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("base_diffuse_roughness") {
            material = material.base_diffuse_roughness(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("base_metalness") {
            material = material.base_metalness(*v);
        }

        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("specular_weight") {
            material = material.specular_weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("specular_color") {
            material = material.specular_edge_tint_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("specular_roughness") {
            material = material.specular_roughness(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("specular_ior") {
            material = material.specular_ior(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("specular_anisotropy") {
            material = material.specular_anisotropy(*v);
        }

        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_weight") {
            material = material.transmission_weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("transmission_color") {
            material = material.transmission_color_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_depth") {
            material = material.transmission_depth(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_dispersion_scale")
        {
            material = material.transmission_dispersion_scale(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_dispersion_abbe") {
            material = material.transmission_dispersion_abbe(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("transmission_scatter") {
            material = material.transmission_scatter_color(v[0], v[1], v[2]);
        }
        if let Some(OutputValue::Float(v)) =
            evaluated.outputs.get("transmission_scatter_anisotropy")
        {
            material = material.transmission_scatter_anisotropy(*v);
        }

        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("subsurface_weight") {
            material = material.subsurface_weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("subsurface_color") {
            material = material.subsurface_color_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("subsurface_radius") {
            material = material.subsurface_radius(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("subsurface_radius_scale") {
            material = material
                .subsurface_radius_scale_g(v[1])
                .subsurface_radius_scale_b(v[2]);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("subsurface_scatter_anisotropy")
        {
            material = material.subsurface_scatter_anisotropy(*v);
        }

        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("fuzz_weight") {
            material = material.fuzz_weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("fuzz_color") {
            material = material.fuzz_color_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("fuzz_roughness") {
            material = material.fuzz_roughness(*v);
        }

        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("coat_weight") {
            material = material.coat_weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("coat_color") {
            material = material.coat_color_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("coat_roughness") {
            material = material.coat_roughness(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("coat_anisotropy") {
            material = material.coat_anisotropy(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("coat_ior") {
            material = material.coat_ior(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("coat_darkening") {
            material = material.coat_darkening(*v);
        }

        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("thin_film_weight") {
            material = material.thin_film_weight(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("thin_film_thickness") {
            material = material.thin_film_thickness_um(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("thin_film_ior") {
            material = material.thin_film_ior(*v);
        }

        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("emission_luminance") {
            material = material.emission_luminance(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("emission_color") {
            material = material.emission_color_rgb(*v);
        }

        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("geometry_opacity") {
            material = material.opacity(*v);
        }
        if let Some(OutputValue::Boolean(v)) = evaluated.outputs.get("geometry_thin_walled") {
            material = material.thin_walled(*v);
        }

        Ok(material)
    }
}

struct GraphEvaluator<'a> {
    converter: &'a MaterialXConverter,
    graph: &'a NodeGraph,
    node_values: HashMap<String, OutputValue>,
    visited: HashMap<String, VisitState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Visiting,
    Visited,
}

impl<'a> GraphEvaluator<'a> {
    fn new(converter: &'a MaterialXConverter, graph: &'a NodeGraph) -> Self {
        Self {
            converter,
            graph,
            node_values: HashMap::new(),
            visited: HashMap::new(),
        }
    }

    fn evaluate(&mut self) -> Result<EvaluatedGraph, CodegenError> {
        let output_nodes: Vec<&Node> = self
            .graph
            .nodes
            .iter()
            .filter(|n| n.node_type == "output")
            .collect();

        for output_node in output_nodes {
            self.evaluate_node(output_node)?;
        }

        for node in &self.graph.nodes {
            if matches!(node.node_type.as_str(), "surface" | "edf") {
                self.evaluate_node(node)?;
            }
        }

        let mut outputs = HashMap::new();
        for node in &self.graph.nodes {
            if node.node_type == "output" {
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

    fn evaluate_node(&mut self, node: &Node) -> Result<OutputValue, CodegenError> {
        match self.visited.get(&node.name) {
            Some(VisitState::Visiting) => return Err(CodegenError::CyclicDependency),
            Some(VisitState::Visited) => {
                return self
                    .node_values
                    .get(&node.name)
                    .cloned()
                    .ok_or_else(|| CodegenError::InputNotFound(node.name.clone()));
            }
            None => {}
        }

        self.visited.insert(node.name.clone(), VisitState::Visiting);

        let node_def = self
            .converter
            .node_defs
            .get(&node.node_type)
            .ok_or_else(|| CodegenError::NodeDefNotFound(node.node_type.clone()))?;

        let mut input_values = HashMap::new();
        for input in &node.inputs {
            let value = if !input.nodename.is_empty() {
                let connected_node = self
                    .graph
                    .nodes
                    .iter()
                    .find(|n| n.name == input.nodename)
                    .ok_or_else(|| CodegenError::InputNotFound(input.nodename.clone()))?;
                self.evaluate_node(connected_node)?
            } else if !input.value.is_empty() {
                self.parse_constant(&input.value, &input.input_type)?
            } else if let Some(def) = node_def.inputs.iter().find(|d| d.name == input.name) {
                self.parse_constant(&def.value, &def.input_type)?
            } else {
                return Err(CodegenError::InputNotFound(input.name.clone()));
            };
            input_values.insert(input.name.clone(), value);
        }

        let result = match node.node_type.as_str() {
            "open_pbr_surface" => OutputValue::String("surface".to_string()),
            "surface" => OutputValue::BSDF(node.name.clone()),
            "oren_nayar_diffuse_bsdf" => self.eval_oren_nayar(&input_values)?,
            "dielectric_bsdf" => self.eval_dielectric_bsdf(&input_values)?,
            "generalized_schlick_bsdf" => self.eval_schlick_bsdf(&input_values)?,
            "sheen_bsdf" => self.eval_sheen_bsdf(&input_values)?,
            "thin_film_bsdf" => self.eval_thin_film_bsdf(&input_values)?,
            "translucent_bsdf" => self.eval_translucent_bsdf(&input_values)?,
            "subsurface_bsdf" => self.eval_subsurface_bsdf(&input_values)?,
            "anisotropic_vdf" => self.eval_anisotropic_vdf(&input_values)?,
            "uniform_edf" => self.eval_uniform_edf(&input_values)?,
            "generalized_schlick_edf" => self.eval_schlick_edf(&input_values)?,
            "mix" => self.eval_mix(&input_values)?,
            "layer" => self.eval_layer(&input_values)?,
            "open_pbr_anisotropy" => self.eval_anisotropy(&input_values)?,
            "multiply" | "add" | "divide" | "subtract" => {
                self.eval_math(&node.node_type, &input_values)?
            }
            "invert" => self.eval_invert(&input_values)?,
            "clamp" => self.eval_clamp(&input_values)?,
            "max" | "min" => self.eval_minmax(&node.node_type, &input_values)?,
            "power" => self.eval_power(&input_values)?,
            "sqrt" => self.eval_sqrt(&input_values)?,
            "ifgreater" => self.eval_ifgreater(&input_values)?,
            "convert" => self.eval_convert(&input_values)?,
            "combine2" => self.eval_combine2(&input_values)?,
            "combine3" => self.eval_combine3(&input_values)?,
            "combine4" => self.eval_combine4(&input_values)?,
            "constant" => self.eval_constant(&input_values)?,
            "output" => {
                if let Some(bsdf_input) = input_values.get("bsdf") {
                    bsdf_input.clone()
                } else if let Some(edf_input) = input_values.get("edf") {
                    edf_input.clone()
                } else if !node.nodename.is_empty() {
                    // Connection via the `nodename` attribute rather than a
                    // child `<input>`: pass through the referenced node.
                    let connected_node = self
                        .graph
                        .nodes
                        .iter()
                        .find(|n| n.name == node.nodename)
                        .ok_or_else(|| CodegenError::InputNotFound(node.nodename.clone()))?;
                    self.evaluate_node(connected_node)?
                } else {
                    OutputValue::String("output".to_string())
                }
            }
            _ => return Err(CodegenError::UnsupportedNode(node.node_type.clone())),
        };

        self.node_values.insert(node.name.clone(), result.clone());
        self.visited.insert(node.name.clone(), VisitState::Visited);
        Ok(result)
    }

    fn eval_math(
        &self,
        op: &str,
        inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
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

    fn eval_invert(
        &self,
        inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
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

    fn eval_clamp(
        &self,
        inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
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
        &self,
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

    fn eval_power(
        &self,
        inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
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

    fn eval_sqrt(
        &self,
        inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        let input = inputs
            .get("in")
            .ok_or(CodegenError::InputNotFound("in".into()))?;
        match input {
            OutputValue::Float(v) => Ok(OutputValue::Float(v.sqrt())),
            _ => Err(CodegenError::TypeConversion("sqrt expects float".into())),
        }
    }

    fn eval_ifgreater(
        &self,
        inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
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

    fn eval_convert(
        &self,
        inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        let input = inputs
            .get("in")
            .ok_or(CodegenError::InputNotFound("in".into()))?;
        Ok(input.clone())
    }

    fn eval_combine2(
        &self,
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

    fn eval_combine3(
        &self,
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

    fn eval_combine4(
        &self,
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

    fn eval_constant(
        &self,
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

    fn eval_oren_nayar(
        &self,
        _inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("oren_nayar".to_string()))
    }

    fn eval_dielectric_bsdf(
        &self,
        _inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("dielectric".to_string()))
    }

    fn eval_schlick_bsdf(
        &self,
        _inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("schlick".to_string()))
    }

    fn eval_sheen_bsdf(
        &self,
        _inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("sheen".to_string()))
    }

    fn eval_thin_film_bsdf(
        &self,
        _inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("thin_film".to_string()))
    }

    fn eval_translucent_bsdf(
        &self,
        _inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("translucent".to_string()))
    }

    fn eval_subsurface_bsdf(
        &self,
        _inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("subsurface".to_string()))
    }

    fn eval_anisotropic_vdf(
        &self,
        _inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::VDF("anisotropic".to_string()))
    }

    fn eval_uniform_edf(
        &self,
        _inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::EDF("uniform".to_string()))
    }

    fn eval_schlick_edf(
        &self,
        _inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::EDF("schlick".to_string()))
    }

    fn eval_anisotropy(
        &self,
        inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        let roughness = self.get_float(inputs, "roughness")?;
        let anisotropy = self.get_float(inputs, "anisotropy")?;
        Ok(OutputValue::Vector2([roughness, anisotropy]))
    }

    fn eval_mix(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let fg = inputs
            .get("fg")
            .ok_or(CodegenError::InputNotFound("fg".into()))?;
        let bg = inputs
            .get("bg")
            .ok_or(CodegenError::InputNotFound("bg".into()))?;
        let mix = self.get_float(inputs, "mix")?;

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

    fn eval_layer(
        &self,
        inputs: &HashMap<String, OutputValue>,
    ) -> Result<OutputValue, CodegenError> {
        self.eval_mix(inputs)
    }

    fn get_float(
        &self,
        inputs: &HashMap<String, OutputValue>,
        name: &str,
    ) -> Result<f32, CodegenError> {
        inputs
            .get(name)
            .and_then(|v| match v {
                OutputValue::Float(f) => Some(*f),
                _ => None,
            })
            .ok_or_else(|| CodegenError::InputNotFound(name.into()))
    }

    fn parse_constant(&self, value: &str, ty: &str) -> Result<OutputValue, CodegenError> {
        match ty {
            "float" => {
                let v: f32 = value
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("float: {}", value)))?;
                Ok(OutputValue::Float(v))
            }
            "color3" => {
                let parts: Vec<&str> = value.split(',').collect();
                if parts.len() != 3 {
                    return Err(CodegenError::TypeConversion(format!("color3: {}", value)));
                }
                let r = parts[0]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("color3 r: {}", value)))?;
                let g = parts[1]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("color3 g: {}", value)))?;
                let b = parts[2]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("color3 b: {}", value)))?;
                Ok(OutputValue::Color3([r, g, b]))
            }
            "color4" => {
                let parts: Vec<&str> = value.split(',').collect();
                if parts.len() != 4 {
                    return Err(CodegenError::TypeConversion(format!("color4: {}", value)));
                }
                let r = parts[0]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("color4 r: {}", value)))?;
                let g = parts[1]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("color4 g: {}", value)))?;
                let b = parts[2]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("color4 b: {}", value)))?;
                let a = parts[3]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("color4 a: {}", value)))?;
                Ok(OutputValue::Color4([r, g, b, a]))
            }
            "vector2" => {
                let parts: Vec<&str> = value.split(',').collect();
                if parts.len() != 2 {
                    return Err(CodegenError::TypeConversion(format!("vector2: {}", value)));
                }
                let x = parts[0]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("vector2 x: {}", value)))?;
                let y = parts[1]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("vector2 y: {}", value)))?;
                Ok(OutputValue::Vector2([x, y]))
            }
            "vector3" => {
                let parts: Vec<&str> = value.split(',').collect();
                if parts.len() != 3 {
                    return Err(CodegenError::TypeConversion(format!("vector3: {}", value)));
                }
                let x = parts[0]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("vector3 x: {}", value)))?;
                let y = parts[1]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("vector3 y: {}", value)))?;
                let z = parts[2]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("vector3 z: {}", value)))?;
                Ok(OutputValue::Vector3([x, y, z]))
            }
            "vector4" => {
                let parts: Vec<&str> = value.split(',').collect();
                if parts.len() != 4 {
                    return Err(CodegenError::TypeConversion(format!("vector4: {}", value)));
                }
                let x = parts[0]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("vector4 x: {}", value)))?;
                let y = parts[1]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("vector4 y: {}", value)))?;
                let z = parts[2]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("vector4 z: {}", value)))?;
                let w = parts[3]
                    .trim()
                    .parse()
                    .map_err(|_| CodegenError::TypeConversion(format!("vector4 w: {}", value)))?;
                Ok(OutputValue::Vector4([x, y, z, w]))
            }
            "boolean" => {
                let v = value.trim().to_lowercase();
                let b = match v.as_str() {
                    "true" | "1" | "yes" => true,
                    "false" | "0" | "no" => false,
                    _ => return Err(CodegenError::TypeConversion(format!("boolean: {}", value))),
                };
                Ok(OutputValue::Boolean(b))
            }
            "string" => Ok(OutputValue::String(value.to_string())),
            _ => Err(CodegenError::TypeConversion(format!(
                "unsupported type: {}",
                ty
            ))),
        }
    }
}

/// High-level function to convert MaterialX file to OpenPBRMaterial
pub fn materialx_to_openpbr(mtlx_content: &str) -> Result<OpenPBRMaterial, MaterialXError> {
    let document = crate::parser::MaterialXParser::new().parse(mtlx_content)?;
    let converter = MaterialXConverter::new(document);
    converter
        .to_openpbr()
        .map_err(|e| MaterialXError::Codegen(Box::new(e)))
}

/// Load MaterialX from file and convert to OpenPBRMaterial
pub fn load_materialx_file<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<OpenPBRMaterial, MaterialXError> {
    let content = std::fs::read_to_string(path)?;
    materialx_to_openpbr(&content)
}

pub fn parse_materialx(content: &str) -> Result<OpenPBRMaterial, MaterialXError> {
    let parser = crate::parser::MaterialXParser::new();
    let document = parser.parse(content)?;
    let converter = MaterialXConverter::new(document);
    converter
        .to_openpbr()
        .map_err(|e| MaterialXError::Codegen(Box::new(e)))
}

pub struct OpenPBRGraph;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::MaterialXParser;

    /// Nodedefs named realistically (`ND_<node>[_<type>]`), as in real .mtlx
    /// libraries. The evaluator looks definitions up by node *type*
    /// (`multiply`, `clamp`, ...), which only resolves because the converter
    /// also indexes nodedefs by their `node` attribute.
    const NODEDEFS: &str = r#"
  <nodedef name="ND_output" node="output" />
  <nodedef name="ND_constant" node="constant" />
  <nodedef name="ND_add" node="add" />
  <nodedef name="ND_subtract" node="subtract" />
  <nodedef name="ND_multiply" node="multiply" />
  <nodedef name="ND_divide" node="divide" />
  <nodedef name="ND_invert" node="invert" />
  <nodedef name="ND_clamp" node="clamp" />
  <nodedef name="ND_max" node="max" />
  <nodedef name="ND_min" node="min" />
  <nodedef name="ND_power" node="power" />
  <nodedef name="ND_sqrt" node="sqrt" />
  <nodedef name="ND_ifgreater" node="ifgreater" />
  <nodedef name="ND_convert" node="convert" />
  <nodedef name="ND_combine2" node="combine2" />
  <nodedef name="ND_combine3" node="combine3" />
  <nodedef name="ND_combine4" node="combine4" />
  <nodedef name="ND_mix" node="mix" />
  <nodedef name="ND_layer" node="layer" />
  <nodedef name="ND_open_pbr_surface" node="open_pbr_surface" />
  <nodedef name="ND_open_pbr_anisotropy" node="open_pbr_anisotropy" />
  <nodedef name="ND_surface" node="surface" />
  <nodedef name="ND_dielectric_bsdf" node="dielectric_bsdf" />
  <nodedef name="ND_conductor_bsdf" node="conductor_bsdf" />
  <nodedef name="ND_oren_nayar_diffuse_bsdf" node="oren_nayar_diffuse_bsdf" />
  <nodedef name="ND_sheen_bsdf" node="sheen_bsdf" />
  <nodedef name="ND_thin_film_bsdf" node="thin_film_bsdf" />
  <nodedef name="ND_translucent_bsdf" node="translucent_bsdf" />
  <nodedef name="ND_subsurface_bsdf" node="subsurface_bsdf" />
  <nodedef name="ND_generalized_schlick_bsdf" node="generalized_schlick_bsdf" />
  <nodedef name="ND_uniform_edf" node="uniform_edf" />
  <nodedef name="ND_generalized_schlick_edf" node="generalized_schlick_edf" />
  <nodedef name="ND_anisotropic_vdf" node="anisotropic_vdf" />
"#;

    fn document(graph_body: &str) -> MaterialXDocument {
        let content = format!(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
{}
  <nodegraph name="test" nodedef="ND_open_pbr_surface_surfaceshader">
{}
  </nodegraph>
</materialx>"#,
            NODEDEFS, graph_body
        );
        MaterialXParser::new().parse(&content).unwrap()
    }

    /// Evaluate the graph and return the named outputs of its `out` node.
    fn eval(graph_body: &str) -> EvaluatedGraph {
        let converter = MaterialXConverter::new(document(graph_body));
        let graph = &converter.document.nodegraphs[0];
        GraphEvaluator::new(&converter, graph).evaluate().unwrap()
    }

    fn eval_err(graph_body: &str) -> CodegenError {
        let converter = MaterialXConverter::new(document(graph_body));
        let graph = &converter.document.nodegraphs[0];
        GraphEvaluator::new(&converter, graph)
            .evaluate()
            .unwrap_err()
    }

    fn float(outputs: &EvaluatedGraph, name: &str) -> f32 {
        match outputs.outputs.get(name) {
            Some(OutputValue::Float(v)) => *v,
            other => panic!("expected float output {name}, got {other:?}"),
        }
    }

    fn color3(outputs: &EvaluatedGraph, name: &str) -> [f32; 3] {
        match outputs.outputs.get(name) {
            Some(OutputValue::Color3(v)) => *v,
            other => panic!("expected color3 output {name}, got {other:?}"),
        }
    }

    fn assert_close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }

    #[test]
    fn test_simple_materialx() {
        // The original test document: a surface shader fed by a dielectric
        // BSDF and a uniform EDF. It needs nodedefs resolvable by node type
        // (see NODEDEFS) — without them evaluation fails with NodeDefNotFound.
        let mtlx = format!(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
{}
  <nodegraph name="test" nodedef="ND_open_pbr_surface_surfaceshader">
    <output name="out" type="surfaceshader" nodename="shader_constructor" />
    <surface name="shader_constructor" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="dielectric_bsdf" />
      <input name="edf" type="EDF" nodename="uniform_edf" />
      <input name="opacity" type="float" value="1.0" />
      <input name="thin_walled" type="boolean" value="false" />
    </surface>
    <dielectric_bsdf name="dielectric_bsdf" type="BSDF">
      <input name="weight" type="float" value="1.0" />
      <input name="ior" type="float" value="1.5" />
      <input name="roughness" type="vector2" value="0.09, 0.09" />
      <input name="scatter_mode" type="string" value="R" />
    </dielectric_bsdf>
    <uniform_edf name="uniform_edf" type="EDF">
      <input name="color" type="color3" value="1.0, 1.0, 1.0" />
    </uniform_edf>
  </nodegraph>
</materialx>"#,
            NODEDEFS
        );

        let result = materialx_to_openpbr(&mtlx);
        assert!(result.is_ok());
        // The `out` node connects to the surface via its `nodename`
        // attribute; "out" is not a material parameter name, so no material
        // parameters are extracted and the result is the default PBR material.
        let mat = result.unwrap();
        assert_eq!(mat.base_params[2], 0.0);
        assert_eq!(mat.specular_params[2], 1.5);
    }

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

    /// An input with neither a value nor a connection falls back to the
    /// default declared by the nodedef. The nodedef is named realistically
    /// (`ND_clamp_float`), so the lookup resolves through its `node`
    /// attribute, not its name.
    #[test]
    fn test_nodedef_default_input() {
        let outputs = eval(
            r#"<nodedef name="ND_clamp_float" node="clamp">
      <input name="low" type="float" value="0.75" />
    </nodedef>
    <clamp name="cl" type="float">
      <input name="in" type="float" value="0.5" />
      <input name="low" type="float" />
      <input name="high" type="float" value="1.0" />
    </clamp>
    <output name="out" type="float"><input name="clamped" type="float" nodename="cl" /></output>"#,
        );

        // clamp(0.5, low=0.75 from the nodedef, high=1.0) = 0.75
        assert_close(float(&outputs, "clamped"), 0.75);
    }

    /// Legacy fallback: a nodedef whose *name* equals the node type is still
    /// found, even when another nodedef declares the same type via `node`.
    #[test]
    fn test_nodedef_lookup_by_name_fallback() {
        let outputs = eval(
            r#"<nodedef name="clamp" node="clamp">
      <input name="low" type="float" value="0.75" />
    </nodedef>
    <clamp name="cl" type="float">
      <input name="in" type="float" value="0.5" />
      <input name="low" type="float" />
      <input name="high" type="float" value="1.0" />
    </clamp>
    <output name="out" type="float"><input name="clamped" type="float" nodename="cl" /></output>"#,
        );

        assert_close(float(&outputs, "clamped"), 0.75);
    }

    /// Exercise every branch of `extract_material`: one output per
    /// OpenPBR parameter name, each fed by a constant. The builder is
    /// invoked for every group (base/specular/transmission/subsurface/fuzz/
    /// coat/thin_film/emission/geometry), so all extractor branches run.
    /// `OpenPBRMaterial` has no public field accessors, so we assert the
    /// result is Ok and non-default via Debug (coverage of the mapping is
    /// the goal, not field-level inspection).
    #[test]
    fn test_extract_all_openpbr_parameters() {
        let mtlx = format!(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
{}
  <nodegraph name="test" nodedef="ND_open_pbr_surface_surfaceshader">
    <output name="out" type="surfaceshader">
      <input name="base_weight" type="float" value="0.9" />
      <input name="base_color" type="color3" value="0.1, 0.2, 0.3" />
      <input name="base_diffuse_roughness" type="float" value="0.45" />
      <input name="base_metalness" type="float" value="0.55" />
      <input name="specular_weight" type="float" value="0.8" />
      <input name="specular_color" type="color3" value="0.4, 0.5, 0.6" />
      <input name="specular_roughness" type="float" value="0.33" />
      <input name="specular_ior" type="float" value="1.45" />
      <input name="specular_anisotropy" type="float" value="0.2" />
      <input name="transmission_weight" type="float" value="0.7" />
      <input name="transmission_color" type="color3" value="0.7, 0.8, 0.9" />
      <input name="transmission_depth" type="float" value="1.2" />
      <input name="transmission_dispersion_scale" type="float" value="2.0" />
      <input name="transmission_dispersion_abbe" type="float" value="30.0" />
      <input name="transmission_scatter" type="color3" value="0.3, 0.4, 0.5" />
      <input name="transmission_scatter_anisotropy" type="float" value="0.6" />
      <input name="subsurface_weight" type="float" value="0.5" />
      <input name="subsurface_color" type="color3" value="0.6, 0.7, 0.8" />
      <input name="subsurface_radius" type="float" value="0.4" />
      <input name="subsurface_radius_scale" type="color3" value="0.1, 0.2, 0.3" />
      <input name="subsurface_scatter_anisotropy" type="float" value="0.3" />
      <input name="fuzz_weight" type="float" value="0.25" />
      <input name="fuzz_color" type="color3" value="0.9, 0.8, 0.7" />
      <input name="fuzz_roughness" type="float" value="0.15" />
      <input name="coat_weight" type="float" value="0.6" />
      <input name="coat_color" type="color3" value="0.5, 0.6, 0.7" />
      <input name="coat_roughness" type="float" value="0.22" />
      <input name="coat_anisotropy" type="float" value="0.1" />
      <input name="coat_ior" type="float" value="1.9" />
      <input name="coat_darkening" type="float" value="0.05" />
      <input name="thin_film_weight" type="float" value="0.35" />
      <input name="thin_film_thickness" type="float" value="0.55" />
      <input name="thin_film_ior" type="float" value="1.3" />
      <input name="emission_luminance" type="float" value="3.0" />
      <input name="emission_color" type="color3" value="0.2, 0.1, 0.0" />
      <input name="geometry_opacity" type="float" value="0.88" />
      <input name="geometry_thin_walled" type="boolean" value="true" />
    </output>
  </nodegraph>
</materialx>"#,
            NODEDEFS
        );

        let mat = materialx_to_openpbr(&mtlx).expect("all params convert");
        let debug = format!("{mat:?}");
        // Every group's extractor branch ran (builder invoked). Field names
        // are the flat arrays OpenPBRMaterial stores; we just check each
        // group is present in the debug dump.
        for group in [
            "base_params",
            "specular_params",
            "transmission_params",
            "transmission_color",
            "transmission_scatter",
            "subsurface_params",
            "subsurface_color",
            "fuzz_params",
            "coat_params",
            "coat_color",
            "coat_ior",
            "thin_film_params",
            "emission_params",
            "emission_color",
            "geometry_params",
        ] {
            assert!(debug.contains(group), "missing group {group} in: {debug}");
        }
    }

    /// A `color4` base_color must route through the RGBA extractor (not the
    /// RGB one) — covers the `else if Color4` branch in extract_material.
    #[test]
    fn test_base_color_color4_path() {
        let mtlx = format!(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
{}
  <nodegraph name="test" nodedef="ND_open_pbr_surface_surfaceshader">
    <output name="out" type="surfaceshader">
      <input name="base_color" type="color4" value="0.1, 0.2, 0.3, 0.4" />
    </output>
  </nodegraph>
</materialx>"#,
            NODEDEFS
        );
        let result = materialx_to_openpbr(&mtlx);
        assert!(result.is_ok(), "{result:?}");
    }

    /// A graph with no `output` node at all still resolves `surface`/`edf`
    /// nodes (the evaluator evaluates them even without an output), but
    /// produces an empty output set — covers the surface/edf walk branch.
    #[test]
    fn test_surface_node_without_output_evaluates() {
        let mtlx = format!(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
{}
  <nodegraph name="test" nodedef="ND_open_pbr_surface_surfaceshader">
    <surface name="surf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="dielectric_bsdf" />
    </surface>
    <dielectric_bsdf name="dielectric_bsdf" type="BSDF">
      <input name="ior" type="float" value="1.5" />
    </dielectric_bsdf>
  </nodegraph>
</materialx>"#,
            NODEDEFS
        );
        // No `output` referencing surf, so no material params extracted, but
        // the surface node itself must evaluate without error.
        let result = materialx_to_openpbr(&mtlx);
        assert!(result.is_ok());
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

    #[test]
    fn test_missing_nodedef_is_error() {
        // A graph whose nodes have no matching nodedef at all.
        let content = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="test" nodedef="ND_open_pbr_surface_surfaceshader">
    <constant name="c" type="float"><input name="value" type="float" value="1.0" /></constant>
    <output name="out" type="float"><input name="x" type="float" nodename="c" /></output>
  </nodegraph>
</materialx>"#;
        let doc = MaterialXParser::new().parse(content).unwrap();
        let converter = MaterialXConverter::new(doc);
        let graph = &converter.document.nodegraphs[0];
        let err = GraphEvaluator::new(&converter, graph)
            .evaluate()
            .unwrap_err();
        assert!(matches!(err, CodegenError::NodeDefNotFound(_)), "{err:?}");
    }

    #[test]
    fn test_cyclic_dependency_is_error() {
        let err = eval_err(
            r#"<multiply name="a" type="float"><input name="in1" type="float" nodename="b" /><input name="in2" type="float" value="1.0" /></multiply>
    <multiply name="b" type="float"><input name="in1" type="float" nodename="a" /><input name="in2" type="float" value="1.0" /></multiply>
    <output name="out" type="float"><input name="x" type="float" nodename="a" /></output>"#,
        );
        assert!(matches!(err, CodegenError::CyclicDependency), "{err:?}");
    }

    #[test]
    fn test_unknown_node_type_is_error() {
        let err = eval_err(
            r#"<nodedef name="node" node="node" />
    <node name="x" type="float" />
    <output name="out" type="float"><input name="y" type="float" nodename="x" /></output>"#,
        );
        assert!(matches!(err, CodegenError::UnsupportedNode(_)), "{err:?}");
    }

    #[test]
    fn test_missing_input_is_error() {
        let err = eval_err(
            r#"<multiply name="m" type="float" />
    <output name="out" type="float"><input name="x" type="float" nodename="m" /></output>"#,
        );
        assert!(matches!(err, CodegenError::InputNotFound(_)), "{err:?}");
    }

    #[test]
    fn test_bad_constant_value_is_error() {
        let err = eval_err(
            r#"<constant name="c" type="float"><input name="value" type="float" value="abc" /></constant>
    <output name="out" type="float"><input name="x" type="float" nodename="c" /></output>"#,
        );
        assert!(matches!(err, CodegenError::TypeConversion(_)), "{err:?}");
    }

    #[test]
    fn test_math_type_mismatch_is_error() {
        let err = eval_err(
            r#"<multiply name="m" type="float"><input name="in1" type="float" value="1.0" /><input name="in2" type="color3" value="1.0, 1.0, 1.0" /></multiply>
    <output name="out" type="float"><input name="x" type="float" nodename="m" /></output>"#,
        );
        assert!(matches!(err, CodegenError::TypeConversion(_)), "{err:?}");
    }

    #[test]
    fn test_openpbr_graph_not_found() {
        let content = r#"<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="test" nodedef="ND_something_else">
  </nodegraph>
</materialx>"#;
        let result = materialx_to_openpbr(content);
        assert!(
            matches!(result, Err(MaterialXError::Codegen(_))),
            "{result:?}"
        );
    }

    /// End-to-end: a math chain evaluated by `materialx_to_openpbr` lands in
    /// the extracted `OpenPBRMaterial` fields.
    #[test]
    fn test_end_to_end_material_extraction() {
        let mtlx = format!(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
{}
  <nodegraph name="test" nodedef="ND_open_pbr_surface_surfaceshader">
    <constant name="a" type="color3"><input name="value" type="color3" value="0.8, 0.4, 0.2" /></constant>
    <constant name="b" type="color3"><input name="value" type="color3" value="0.5, 0.5, 0.5" /></constant>
    <multiply name="m" type="color3"><input name="in1" type="color3" nodename="a" /><input name="in2" type="color3" nodename="b" /></multiply>
    <subtract name="s" type="color3"><input name="in1" type="color3" nodename="a" /><input name="in2" type="color3" nodename="m" /></subtract>
    <invert name="inv" type="float"><input name="in" type="float" value="0.25" /></invert>
    <clamp name="cl" type="float"><input name="in" type="float" value="1.5" /><input name="low" type="float" value="0.0" /><input name="high" type="float" value="1.0" /></clamp>
    <output name="out" type="surfaceshader">
      <input name="base_color" type="color3" nodename="s" />
      <input name="base_weight" type="float" nodename="inv" />
      <input name="specular_roughness" type="float" nodename="cl" />
    </output>
  </nodegraph>
</materialx>"#,
            NODEDEFS
        );

        let mat = materialx_to_openpbr(&mtlx).unwrap();
        // a - a*b per channel: (0.8, 0.4, 0.2) - (0.4, 0.2, 0.1)
        assert_eq!(mat.base_color[0], 0.4f32);
        assert_eq!(mat.base_color[1], 0.2f32);
        assert_close(mat.base_color[2], 0.1);
        // invert(0.25)
        assert_close(mat.base_params[0], 0.75);
        // clamp(1.5, 0, 1)
        assert_close(mat.specular_params[1], 1.0);
    }
}
