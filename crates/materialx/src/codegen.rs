//! MaterialX to OpenPBRMaterial code generation

use crate::parser::{MaterialXDocument, NodeGraph, Node, Input, NodeDef};
use ornis_render::OpenPBRMaterial;
use std::collections::HashMap;

/// Errors during code generation
#[derive(Debug, thiserror::Error)]
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
}

/// Result of evaluating a node graph
#[derive(Debug, Clone)]
pub struct EvaluatedGraph {
    pub outputs: HashMap<String, OutputValue>,
}

/// Possible output values from node evaluation
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
    BSDF(String), // Reference to a BSDF node
    EDF(String),  // Reference to an EDF node
    VDF(String),  // Reference to a VDF node
}

/// MaterialX to OpenPBRMaterial converter
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
    
    /// Convert a MaterialX document to OpenPBRMaterial
    /// Looks for a nodegraph with nodedef="ND_open_pbr_surface_surfaceshader"
    pub fn to_openpbr(&self) -> Result<OpenPBRMaterial, CodegenError> {
        // Find the OpenPBR surface shader graph
        let graph = self.find_openpbr_graph()?;
        
        // Evaluate the graph
        let evaluator = GraphEvaluator::new(self, &graph);
        let evaluated = evaluator.evaluate()?;
        
        // Extract outputs and map to OpenPBRMaterial
        self.extract_material(&evaluated)
    }
    
    fn find_openpbr_graph(&self) -> Result<&NodeGraph, CodegenError> {
        for graph in &self.document.nodegraphs {
            if graph.nodedef.contains("open_pbr_surface") {
                return Ok(graph);
            }
        }
        Err(CodegenError::GraphNotFound("OpenPBR surface shader graph not found".to_string()))
    }
    
    fn extract_material(&self, evaluated: &EvaluatedGraph) -> Result<OpenPBRMaterial, CodegenError> {
        let mut material = OpenPBRMaterial::pbr();
        
        // Base parameters
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
        
        // Specular
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
        
        // Transmission
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_weight") {
            material = material.transmission_weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("transmission_color") {
            material = material.transmission_color_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_depth") {
            material = material.transmission_depth(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_dispersion_scale") {
            material = material.transmission_dispersion_scale(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_dispersion_abbe") {
            material = material.transmission_dispersion_abbe(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("transmission_scatter") {
            material = material.transmission_scatter_color_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_scatter_anisotropy") {
            material = material.transmission_scatter_anisotropy(*v);
        }
        
        // Subsurface
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
            material = material.subsurface_radius_scale_g(v[1]).subsurface_radius_scale_b(v[2]);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("subsurface_scatter_anisotropy") {
            material = material.subsurface_scatter_anisotropy(*v);
        }
        
        // Fuzz
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("fuzz_weight") {
            material = material.fuzz_weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("fuzz_color") {
            material = material.fuzz_color_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("fuzz_roughness") {
            material = material.fuzz_roughness(*v);
        }
        
        // Coat
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
        
        // Thin Film
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("thin_film_weight") {
            material = material.thin_film_weight(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("thin_film_thickness") {
            material = material.thin_film_thickness_um(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("thin_film_ior") {
            material = material.thin_film_ior(*v);
        }
        
        // Emission
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("emission_luminance") {
            material = material.emission_luminance(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("emission_color") {
            material = material.emission_color_rgb(*v);
        }
        
        // Geometry
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("geometry_opacity") {
            material = material.opacity(*v);
        }
        if let Some(OutputValue::Boolean(v)) = evaluated.outputs.get("geometry_thin_walled") {
            material = material.thin_walled(*v);
        }
        
        Ok(material)
    }
}

/// Evaluates a node graph by topologically sorting nodes
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
        // First, find all output nodes
        let output_nodes: Vec<&Node> = self.graph.nodes.iter()
            .filter(|n| n.node_type == "output")
            .collect();
        
        // Evaluate each output
        for output_node in output_nodes {
            self.evaluate_node(output_node)?;
        }
        
        // Also evaluate any surface/edf nodes that might not have outputs
        for node in &self.graph.nodes {
            if matches!(node.node_type.as_str(), "surface" | "edf") {
                self.evaluate_node(node)?;
            }
        }
        
        // Collect outputs
        let mut outputs = HashMap::new();
        for node in &self.graph.nodes {
            if node.node_type == "output" {
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
        // Check for cycles
        match self.visited.get(&node.name) {
            Some(VisitState::Visiting) => return Err(CodegenError::CyclicDependency),
            Some(VisitState::Visited) => {
                return self.node_values.get(&node.name)
                    .cloned()
                    .ok_or_else(|| CodegenError::InputNotFound(node.name.clone()));
            }
            None => {}
        }
        
        self.visited.insert(node.name.clone(), VisitState::Visiting);
        
        // Get node definition for type info
        let node_def = self.converter.node_defs.get(&node.node_type)
            .ok_or_else(|| CodegenError::NodeDefNotFound(node.node_type.clone()))?;
        
        // Evaluate all inputs
        let mut input_values = HashMap::new();
        for input in &node.inputs {
            let value = if !input.nodename.is_empty() {
                // Connected to another node
                let connected_node = self.graph.nodes.iter()
                    .find(|n| n.name == input.nodename)
                    .ok_or_else(|| CodegenError::InputNotFound(input.nodename.clone()))?;
                self.evaluate_node(connected_node)?
            } else if !input.value.is_empty() {
                // Constant value
                self.parse_constant(&input.value, &input.input_type)?
            } else if let Some(def) = node_def.inputs.iter().find(|d| d.name == input.name) {
                // Default value from node definition
                self.parse_constant(&def.value, &def.input_type)?
            } else {
                return Err(CodegenError::InputNotFound(input.name.clone()));
            };
            input_values.insert(input.name.clone(), value);
        }
        
        // Evaluate based on node type
        let result = match node.node_type.as_str() {
            // OpenPBR surface shader - this is the main output
            "open_pbr_surface" => {
                // This node just passes through to outputs
                OutputValue::String("surface".to_string())
            }
            // Surface shader constructor
            "surface" => {
                OutputValue::BSDF(node.name.clone())
            }
            // BSDF nodes
            "oren_nayar_diffuse_bsdf" => self.eval_oren_nayar(&input_values)?,
            "dielectric_bsdf" => self.eval_dielectric_bsdf(&input_values)?,
            "generalized_schlick_bsdf" => self.eval_schlick_bsdf(&input_values)?,
            "sheen_bsdf" => self.eval_sheen_bsdf(&input_values)?,
            "thin_film_bsdf" => self.eval_thin_film_bsdf(&input_values)?,
            "translucent_bsdf" => self.eval_translucent_bsdf(&input_values)?,
            "subsurface_bsdf" => self.eval_subsurface_bsdf(&input_values)?,
            // VDF
            "anisotropic_vdf" => self.eval_anisotropic_vdf(&input_values)?,
            // EDF
            "uniform_edf" => self.eval_uniform_edf(&input_values)?,
            "generalized_schlick_edf" => self.eval_schlick_edf(&input_values)?,
            // Mix/Layer
            "mix" => self.eval_mix(&input_values)?,
            "layer" => self.eval_layer(&input_values)?,
            // Utility
            "open_pbr_anisotropy" => self.eval_anisotropy(&input_values)?,
            "multiply" | "add" | "divide" | "subtract" => self.eval_math(&node.node_type, &input_values)?,
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
            // Constants
            "constant" => self.eval_constant(&input_values)?,
            // Output
            "output" => {
                // Output just passes through the connected value
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
    
    // Math operations
    fn eval_math(&self, op: &str, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let in1 = inputs.get("in1").or_else(|| inputs.get("in")).ok_or(CodegenError::InputNotFound("in1".into()))?;
        let in2 = inputs.get("in2").ok_or(CodegenError::InputNotFound("in2".into()))?;
        
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
                    "multiply" => [a[0]*b[0], a[1]*b[1], a[2]*b[2]],
                    "add" => [a[0]+b[0], a[1]+b[1], a[2]+b[2]],
                    "divide" => [a[0]/b[0], a[1]/b[1], a[2]/b[2]],
                    "subtract" => [a[0]-b[0], a[1]-b[1], a[2]-b[2]],
                    _ => return Err(CodegenError::UnsupportedNode(op.into())),
                };
                Ok(OutputValue::Color3(result))
            }
            (OutputValue::Vector3(a), OutputValue::Vector3(b)) => {
                let result = match op {
                    "multiply" => [a[0]*b[0], a[1]*b[1], a[2]*b[2]],
                    "add" => [a[0]+b[0], a[1]+b[1], a[2]+b[2]],
                    "divide" => [a[0]/b[0], a[1]/b[1], a[2]/b[2]],
                    "subtract" => [a[0]-b[0], a[1]-b[1], a[2]-b[2]],
                    _ => return Err(CodegenError::UnsupportedNode(op.into())),
                };
                Ok(OutputValue::Vector3(result))
            }
            _ => Err(CodegenError::TypeConversion("mismatched types for math op".into())),
        }
    }
    
    fn eval_invert(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let input = inputs.get("in").ok_or(CodegenError::InputNotFound("in".into()))?;
        match input {
            OutputValue::Float(v) => Ok(OutputValue::Float(1.0 - v)),
            OutputValue::Color3(v) => Ok(OutputValue::Color3([1.0-v[0], 1.0-v[1], 1.0-v[2]])),
            _ => Err(CodegenError::TypeConversion("invert expects float or color3".into())),
        }
    }
    
    fn eval_clamp(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let input = inputs.get("in").ok_or(CodegenError::InputNotFound("in".into()))?;
        let low = inputs.get("low").ok_or(CodegenError::InputNotFound("low".into()))?;
        let high = inputs.get("high").ok_or(CodegenError::InputNotFound("high".into()))?;
        
        match (input, low, high) {
            (OutputValue::Float(v), OutputValue::Float(l), OutputValue::Float(h)) => {
                Ok(OutputValue::Float(v.clamp(*l, *h)))
            }
            (OutputValue::Color3(v), OutputValue::Float(l), OutputValue::Float(h)) => {
                Ok(OutputValue::Color3([v[0].clamp(*l, *h), v[1].clamp(*l, *h), v[2].clamp(*l, *h)]))
            }
            _ => Err(CodegenError::TypeConversion("clamp type mismatch".into())),
        }
    }
    
    fn eval_minmax(&self, op: &str, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let in1 = inputs.get("in1").ok_or(CodegenError::InputNotFound("in1".into()))?;
        let in2 = inputs.get("in2").ok_or(CodegenError::InputNotFound("in2".into()))?;
        
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
    
    fn eval_power(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let input = inputs.get("in").ok_or(CodegenError::InputNotFound("in".into()))?;
        let exp = inputs.get("exponent").ok_or(CodegenError::InputNotFound("exponent".into()))?;
        
        match (input, exp) {
            (OutputValue::Float(b), OutputValue::Float(e)) => Ok(OutputValue::Float(b.powf(*e))),
            _ => Err(CodegenError::TypeConversion("power expects float".into())),
        }
    }
    
    fn eval_sqrt(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let input = inputs.get("in").ok_or(CodegenError::InputNotFound("in".into()))?;
        match input {
            OutputValue::Float(v) => Ok(OutputValue::Float(v.sqrt())),
            _ => Err(CodegenError::TypeConversion("sqrt expects float".into())),
        }
    }
    
    fn eval_ifgreater(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let v1 = inputs.get("value1").ok_or(CodegenError::InputNotFound("value1".into()))?;
        let v2 = inputs.get("value2").ok_or(CodegenError::InputNotFound("value2".into()))?;
        let in1 = inputs.get("in1").ok_or(CodegenError::InputNotFound("in1".into()))?;
        let in2 = inputs.get("in2").ok_or(CodegenError::InputNotFound("in2".into()))?;
        
        match (v1, v2) {
            (OutputValue::Float(a), OutputValue::Float(b)) => {
                if a > b { Ok(in1.clone()) } else { Ok(in2.clone()) }
            }
            _ => Err(CodegenError::TypeConversion("ifgreater expects float".into())),
        }
    }
    
    fn eval_convert(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let input = inputs.get("in").ok_or(CodegenError::InputNotFound("in".into()))?;
        // Just pass through for now - type conversion is handled by the type system
        Ok(input.clone())
    }
    
    fn eval_combine2(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let in1 = inputs.get("in1").ok_or(CodegenError::InputNotFound("in1".into()))?;
        let in2 = inputs.get("in2").ok_or(CodegenError::InputNotFound("in2".into()))?;
        
        match (in1, in2) {
            (OutputValue::Float(a), OutputValue::Float(b)) => Ok(OutputValue::Vector2([*a, *b])),
            _ => Err(CodegenError::TypeConversion("combine2 expects float".into())),
        }
    }
    
    fn eval_combine3(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let in1 = inputs.get("in1").ok_or(CodegenError::InputNotFound("in1".into()))?;
        let in2 = inputs.get("in2").ok_or(CodegenError::InputNotFound("in2".into()))?;
        let in3 = inputs.get("in3").ok_or(CodegenError::InputNotFound("in3".into()))?;
        
        match (in1, in2, in3) {
            (OutputValue::Float(a), OutputValue::Float(b), OutputValue::Float(c)) => Ok(OutputValue::Vector3([*a, *b, *c])),
            _ => Err(CodegenError::TypeConversion("combine3 expects float".into())),
        }
    }
    
    fn eval_combine4(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let in1 = inputs.get("in1").ok_or(CodegenError::InputNotFound("in1".into()))?;
        let in2 = inputs.get("in2").ok_or(CodegenError::InputNotFound("in2".into()))?;
        let in3 = inputs.get("in3").ok_or(CodegenError::InputNotFound("in3".into()))?;
        let in4 = inputs.get("in4").ok_or(CodegenError::InputNotFound("in4".into()))?;
        
        match (in1, in2, in3, in4) {
            (OutputValue::Float(a), OutputValue::Float(b), OutputValue::Float(c), OutputValue::Float(d)) => Ok(OutputValue::Vector4([*a, *b, *c, *d])),
            _ => Err(CodegenError::TypeConversion("combine4 expects float".into())),
        }
    }
    
    fn eval_constant(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        // Constants are handled by constant values in inputs
        // If there's a "value" input, use it
        if let Some(v) = inputs.get("value") {
            return Ok(v.clone());
        }
        // Otherwise check for type-specific inputs
        if let Some(v) = inputs.get("in") {
            return Ok(v.clone());
        }
        Err(CodegenError::InputNotFound("constant value".into()))
    }
    
    // BSDF evaluations
    fn eval_oren_nayar(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        // Returns a BSDF reference
        Ok(OutputValue::BSDF("oren_nayar".to_string()))
    }
    
    fn eval_dielectric_bsdf(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("dielectric".to_string()))
    }
    
    fn eval_schlick_bsdf(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("schlick".to_string()))
    }
    
    fn eval_sheen_bsdf(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("sheen".to_string()))
    }
    
    fn eval_thin_film_bsdf(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("thin_film".to_string()))
    }
    
    fn eval_translucent_bsdf(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("translucent".to_string()))
    }
    
    fn eval_subsurface_bsdf(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::BSDF("subsurface".to_string()))
    }
    
    fn eval_anisotropic_vdf(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::VDF("anisotropic".to_string()))
    }
    
    fn eval_uniform_edf(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::EDF("uniform".to_string()))
    }
    
    fn eval_schlick_edf(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        Ok(OutputValue::EDF("schlick".to_string()))
    }
    
    fn eval_anisotropy(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let roughness = self.get_float(inputs, "roughness")?;
        let anisotropy = self.get_float(inputs, "anisotropy")?;
        Ok(OutputValue::Vector2([roughness, anisotropy]))
    }
    
    fn eval_mix(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let fg = inputs.get("fg").ok_or(CodegenError::InputNotFound("fg".into()))?;
        let bg = inputs.get("bg").ok_or(CodegenError::InputNotFound("bg".into()))?;
        let mix = self.get_float(inputs, "mix")?;
        
        match (fg, bg) {
            (OutputValue::Float(f), OutputValue::Float(b)) => Ok(OutputValue::Float(b * (1.0 - mix) + f * mix)),
            (OutputValue::Color3(f), OutputValue::Color3(b)) => {
                Ok(OutputValue::Color3([
                    b[0] * (1.0 - mix) + f[0] * mix,
                    b[1] * (1.0 - mix) + f[1] * mix,
                    b[2] * (1.0 - mix) + f[2] * mix,
                ]))
            }
            (OutputValue::BSDF(f), OutputValue::BSDF(b)) => {
                // For BSDF mix, just return fg for now
                Ok(OutputValue::BSDF(f.clone()))
            }
            _ => Err(CodegenError::TypeConversion("mix type mismatch".into())),
        }
    }
    
    fn eval_layer(&self, inputs: &HashMap<String, OutputValue>) -> Result<OutputValue, CodegenError> {
        let top = inputs.get("top").ok_or(CodegenError::InputNotFound("top".into()))?;
        let base = inputs.get("base").ok_or(CodegenError::InputNotFound("base".into()))?;
        
        match (top, base) {
            (OutputValue::BSDF(t), OutputValue::BSDF(b)) => {
                // Layering - return top for now
                Ok(OutputValue::BSDF(t.clone()))
            }
            _ => Err(CodegenError::TypeConversion("layer expects BSDF".into())),
        }
    }
    
    // Helper methods
    fn get_float(&self, inputs: &HashMap<String, OutputValue>, key: &str) -> Result<f32, CodegenError> {
        match inputs.get(key) {
            Some(OutputValue::Float(v)) => Ok(*v),
            Some(OutputValue::Color3(v)) => Ok(v[0]),
            Some(OutputValue::Vector2(v)) => Ok(v[0]),
            Some(OutputValue::Vector3(v)) => Ok(v[0]),
            Some(OutputValue::Vector4(v)) => Ok(v[0]),
            Some(OutputValue::Boolean(v)) => Ok(if *v { 1.0 } else { 0.0 }),
            Some(OutputValue::String(s)) => s.parse().map_err(|_| CodegenError::TypeConversion("string to float".into())),
            _ => Err(CodegenError::InputNotFound(key.into())),
        }
    }
    
    fn parse_constant(&self, value: &str, _type: &str) -> Result<OutputValue, CodegenError> {
        // Parse color3: "0.8, 0.8, 0.8"
        if value.contains(',') {
            let parts: Vec<&str> = value.split(',').map(|s| s.trim()).collect();
            if parts.len() == 3 {
                let r = parts[0].parse::<f32>().map_err(|_| CodegenError::TypeConversion("parse float".into()))?;
                let g = parts[1].parse::<f32>().map_err(|_| CodegenError::TypeConversion("parse float".into()))?;
                let b = parts[2].parse::<f32>().map_err(|_| CodegenError::TypeConversion("parse float".into()))?;
                return Ok(OutputValue::Color3([r, g, b]));
            }
            if parts.len() == 4 {
                let r = parts[0].parse::<f32>().map_err(|_| CodegenError::TypeConversion("parse float".into()))?;
                let g = parts[1].parse::<f32>().map_err(|_| CodegenError::TypeConversion("parse float".into()))?;
                let b = parts[2].parse::<f32>().map_err(|_| CodegenError::TypeConversion("parse float".into()))?;
                let a = parts[3].parse::<f32>().map_err(|_| CodegenError::TypeConversion("parse float".into()))?;
                return Ok(OutputValue::Color4([r, g, b, a]));
            }
            if parts.len() == 2 {
                let x = parts[0].parse::<f32>().map_err(|_| CodegenError::TypeConversion("parse float".into()))?;
                let y = parts[1].parse::<f32>().map_err(|_| CodegenError::TypeConversion("parse float".into()))?;
                return Ok(OutputValue::Vector2([x, y]));
            }
        }
        
        // Parse float
        if let Ok(v) = value.parse::<f32>() {
            return Ok(OutputValue::Float(v));
        }
        
        // Parse boolean
        if value == "true" || value == "false" {
            return Ok(OutputValue::Boolean(value == "true"));
        }
        
        // String
        Ok(OutputValue::String(value.to_string()))
    }
}

/// High-level function to convert MaterialX file to OpenPBRMaterial
pub fn materialx_to_openpbr(mtlx_content: &str) -> Result<OpenPBRMaterial, CodegenError> {
    let document = crate::parser::parse_materialx(mtlx_content)?;
    let converter = MaterialXConverter::new(document);
    converter.to_openpbr()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_materialx() {
        let mtlx = r#"
<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_open_pbr_surface_surfaceshader" nodedef="ND_open_pbr_surface_surfaceshader">
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
        
        let result = materialx_to_openpbr(mtlx);
        assert!(result.is_ok());
        let mat = result.unwrap();
        assert_eq!(mat.base_params[2], 0.0); // metalness = 0
        assert_eq!(mat.specular_params[2], 1.5); // ior
    }
}