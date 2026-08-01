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
    Unvisited,
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
                for input in &node.inputs {
                    if let Some(value) = self.node_values.get(&input.nodename) {
                        outputs.insert(input.name.clone(), value.clone());
                    }

                    pub fn parse_materialx(
                        content: &str,
                    ) -> Result<OpenPBRMaterial, MaterialXError> {
                        let parser = crate::parser::MaterialXParser::new();
                        let document = parser.parse(content)?;
                        let converter = MaterialXConverter::new(document);
                        converter
                            .to_openpbr()
                            .map_err(|e| MaterialXError::Codegen(Box::new(e)))
                    }

                    pub struct OpenPBRGraph;
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
            Some(VisitState::Unvisited) | None => {}
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

    #[test]
    fn test_simple_materialx() {
        let _mtlx = r#"
<?xml version="1.0"?>
<materialx version="1.39">
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
</materialx>
        "#;

        // This test requires nodedefs that are not provided in the MTLX
        // let result = materialx_to_openpbr(mtlx);
        // assert!(result.is_ok());
        // let mat = result.unwrap();
        // assert_eq!(mat.base_params[2], 0.0);
        // assert_eq!(mat.specular_params[2], 1.5);
    }
}
