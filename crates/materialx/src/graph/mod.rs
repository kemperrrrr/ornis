//! Graph evaluation and conversion to OpenPBRMaterial.
//!
//! Phase root: shared error/value types, converter indexing and the
//! parse-to-extract entry points. Per-phase logic lives in the child
//! modules parse, validate, evaluate and extract.

mod evaluate;
mod extract;
mod parse;
mod validate;

use crate::nodes::{MaterialXDocument, NodeDef};
use ornis_render::OpenPBRMaterial;
use quick_xml::Error as XmlError;
use quick_xml::events::attributes::AttrError;
use std::collections::HashMap;
use std::str::Utf8Error;
use thiserror::Error;

/// Failures of graph evaluation and OpenPBR extraction.
#[derive(Error, Debug)]
pub enum CodegenError {
    /// No nodegraph implementing `open_pbr_surface` exists in the document.
    #[error("Node graph not found: {0}")]
    GraphNotFound(String),
    /// A node's type has no matching nodedef in the document.
    #[error("Node definition not found for node '{node}' (type '{node_type}')")]
    NodeDefNotFound {
        /// Node type that was looked up (`Node.node_type`).
        node_type: String,
        /// Node instance name that needed the definition.
        node: String,
    },
    /// An input with no connection, literal value or nodedef default.
    #[error("Required input not found: {0}")]
    InputNotFound(String),
    /// A literal value could not be parsed into the declared type.
    #[error("Type conversion error: {0}")]
    TypeConversion(String),
    /// The evaluator has no handler for this node type.
    #[error("Unsupported node type: {0}")]
    UnsupportedNode(String),
    /// The nodegraph contains a dependency cycle.
    #[error("Cyclic dependency detected")]
    CyclicDependency,
    #[error("IO error: {0}")]
    /// Filesystem error raised while loading a document.
    Io(#[from] std::io::Error),
    /// Wraps a parse-stage failure raised while converting.
    #[error("MaterialX error: {0}")]
    MaterialX(#[from] Box<MaterialXError>),
}

/// Unified failure type for the whole parse→convert pipeline.
#[derive(Error, Debug)]
pub enum MaterialXError {
    /// quick_xml failed on the document.
    #[error("XML parse error: {0}")]
    Xml(#[from] XmlError),
    /// Filesystem access failed while reading the document.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// An XML attribute was malformed.
    #[error("XML attribute error: {0}")]
    Attr(#[from] AttrError),
    #[error("UTF-8 conversion error: {0}")]
    /// Non-UTF-8 text inside the XML document.
    Utf8(#[from] Utf8Error),
    /// A referenced node does not exist in the graph.
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    #[error("Invalid parameter: {0}")]
    /// A parameter value is malformed or out of contract.
    InvalidParameter(String),
    #[error("Unsupported node type: {0}")]
    /// The parser/evaluator has no support for this node type.
    UnsupportedNode(String),
    #[error("Missing required input: {0}")]
    /// A node references an input that is nowhere defined.
    MissingInput(String),
    /// Graph evaluation / material extraction failed after a successful parse.
    #[error("Codegen error: {0}")]
    Codegen(Box<CodegenError>),
}

/// Adapter so MaterialX failures can flow through IO-typed call sites.
impl From<MaterialXError> for std::io::Error {
    fn from(e: MaterialXError) -> Self {
        std::io::Error::other(e.to_string())
    }
}

/// Result of evaluating one nodegraph: named output values keyed by the
/// `<output>` element name (e.g. `"base_color"`).
#[derive(Debug, Clone)]
pub struct EvaluatedGraph {
    /// Fully evaluated outputs of the graph.
    pub outputs: HashMap<String, OutputValue>,
}

/// Dynamically typed value produced by an evaluated MaterialX node.
#[derive(Debug, Clone)]
pub enum OutputValue {
    /// Scalar (`float`).
    Float(f32),
    /// Linear RGB color.
    Color3([f32; 3]),
    /// RGBA color.
    Color4([f32; 4]),
    /// 2-component vector.
    Vector2([f32; 2]),
    /// 3-component vector.
    Vector3([f32; 3]),
    /// 4-component vector.
    Vector4([f32; 4]),
    /// Truth value.
    Boolean(bool),
    /// Free-form string (also used for `open_pbr_surface` markers).
    String(String),
    /// Reference to a bidirectional scattering distribution.
    BSDF(String),
    /// Reference to an emission distribution.
    EDF(String),
    /// Reference to a volume distribution.
    VDF(String),
}

/// Evaluates a parsed [`MaterialXDocument`] and extracts an
/// [`OpenPBRMaterial`] from its `open_pbr_surface` graph.
pub struct MaterialXConverter {
    pub(crate) document: MaterialXDocument,
    pub(crate) node_defs: HashMap<String, NodeDef>,
}

impl MaterialXConverter {
    /// Index all nodedefs of `document` for lookup. Definitions are indexed
    /// both by their `ND_*` name and their `node` attribute (the evaluator
    /// looks them up by node *type*); later document order wins conflicts.
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
}

/// One-shot conversion of MaterialX XML text to an [`OpenPBRMaterial`]:
/// parse → evaluate → extract. Evaluation failures surface wrapped in
/// [`MaterialXError::Codegen`].
///
/// # Errors
///
/// [`MaterialXError`] from the parse stage, or `Codegen` wrapping the
/// evaluation failure (missing graph, unresolved nodedef, bad input).
pub fn materialx_to_openpbr(mtlx_content: &str) -> Result<OpenPBRMaterial, MaterialXError> {
    let document = crate::parser::MaterialXParser::new().parse(mtlx_content)?;
    let converter = MaterialXConverter::new(document);
    converter
        .to_openpbr()
        .map_err(|e| MaterialXError::Codegen(Box::new(e)))
}

/// File-based variant of [`materialx_to_openpbr`].
///
/// # Errors
/// IO errors from reading `path` plus everything [`materialx_to_openpbr`] can raise.
pub fn load_materialx_file<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<OpenPBRMaterial, MaterialXError> {
    let content = std::fs::read_to_string(path)?;
    materialx_to_openpbr(&content)
}

/// Alias of [`materialx_to_openpbr`] kept for API stability; identical
/// parse→evaluate→extract pipeline.
///
/// # Errors
/// Same contract as [`materialx_to_openpbr`].
pub fn parse_materialx(content: &str) -> Result<OpenPBRMaterial, MaterialXError> {
    let parser = crate::parser::MaterialXParser::new();
    let document = parser.parse(content)?;
    let converter = MaterialXConverter::new(document);
    converter
        .to_openpbr()
        .map_err(|e| MaterialXError::Codegen(Box::new(e)))
}

/// Marker namespace for OpenPBR graph helpers (kept for API compatibility;
/// currently carries no behavior).
pub struct OpenPBRGraph;

#[cfg(test)]
pub(crate) mod test_helpers {
    //! Shared graph test fixtures: realistic nodedefs plus eval helpers.

    use super::evaluate::GraphEvaluator;
    use super::{CodegenError, EvaluatedGraph, MaterialXConverter, OutputValue};
    use crate::nodes::MaterialXDocument;
    use crate::parser::MaterialXParser;

    /// Nodedefs named realistically (`ND_<node>[_<type>]`), as in real .mtlx
    /// libraries. The evaluator looks definitions up by node *type*
    /// (`multiply`, `clamp`, ...), which only resolves because the converter
    /// also indexes nodedefs by their `node` attribute.
    pub(crate) const NODEDEFS: &str = r#"
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

    pub(crate) fn document(graph_body: &str) -> MaterialXDocument {
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
    pub(crate) fn eval(graph_body: &str) -> EvaluatedGraph {
        let converter = MaterialXConverter::new(document(graph_body));
        let graph = &converter.document.nodegraphs[0];
        GraphEvaluator::new(&converter, graph).evaluate().unwrap()
    }

    pub(crate) fn eval_err(graph_body: &str) -> CodegenError {
        let converter = MaterialXConverter::new(document(graph_body));
        let graph = &converter.document.nodegraphs[0];
        GraphEvaluator::new(&converter, graph)
            .evaluate()
            .unwrap_err()
    }

    pub(crate) fn float(outputs: &EvaluatedGraph, name: &str) -> f32 {
        match outputs.outputs.get(name) {
            Some(OutputValue::Float(v)) => *v,
            other => panic!("expected float output {name}, got {other:?}"),
        }
    }

    pub(crate) fn color3(outputs: &EvaluatedGraph, name: &str) -> [f32; 3] {
        match outputs.outputs.get(name) {
            Some(OutputValue::Color3(v)) => *v,
            other => panic!("expected color3 output {name}, got {other:?}"),
        }
    }

    pub(crate) fn assert_close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }
}

#[cfg(test)]
mod tests {
    use super::materialx_to_openpbr;
    use super::test_helpers::NODEDEFS;

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
        assert_eq!(mat.base.params[2], 0.0);
        assert_eq!(mat.specular.params[2], 1.5);
    }
}
