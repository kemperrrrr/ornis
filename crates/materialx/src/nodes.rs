//! MaterialX AST node definitions

use serde::{Deserialize, Serialize};

/// Root AST of a parsed `.mtlx` file: reusable node definitions plus
/// concrete shader graphs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaterialXDocument {
    /// Library-level `<nodedef>` declarations.
    pub nodedefs: Vec<NodeDef>,
    /// Document-level `<nodegraph>` implementations.
    pub nodegraphs: Vec<NodeGraph>,
}

/// A `<nodedef>`: the signature of a reusable node type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDef {
    /// Unique definition name (`ND_<node>_<type>` convention).
    pub name: String,
    /// Node type this definition implements (the evaluator's lookup key).
    pub node: String,
    /// Grouping label (e.g. `math`), informational.
    pub nodegroup: String,
    /// Version string of the definition, may be empty.
    pub version: String,
    /// Whether this is the default version of the node type.
    pub isdefaultversion: bool,
    /// Documentation string from the library.
    pub doc: String,
    /// UI display name, informational.
    pub uiname: String,
    /// Declared inputs with defaults and UI hints.
    pub inputs: Vec<NodeDefInput>,
    /// Declared outputs.
    pub outputs: Vec<NodeDefOutput>,
}

/// Declared input of a [`NodeDef`]: type, default value and editor hints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDefInput {
    /// Input name as referenced by nodes using this definition.
    pub name: String,
    /// MaterialX type string (e.g. `float`, `color3`).
    pub input_type: String,
    /// Default value as written in the XML (empty when none).
    pub value: String,
    /// UI minimum hint.
    pub uimin: String,
    /// UI maximum hint.
    pub uimax: String,
    /// UI soft-minimum hint.
    pub uisoftmin: String,
    /// UI soft-maximum hint.
    pub uisoftmax: String,
    /// UI display name.
    pub uiname: String,
    /// UI folder grouping.
    pub uifolder: String,
    /// Whether the input is advanced in UIs.
    pub uiadvanced: String,
    /// Documentation string.
    pub doc: String,
    /// Extra hint string.
    pub hint: String,
    /// Uniformity flag as written in the XML.
    pub uniform: String,
    /// Default geometric property binding, if any.
    pub defaultgeomprop: String,
    /// Interface parameter name when the def exposes it upstream.
    pub interfacename: String,
    /// Raw string-typed value for `string` inputs.
    pub string: String,
}

/// Declared output of a [`NodeDef`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDefOutput {
    /// Output name (typically `out`).
    pub name: String,
    /// MaterialX type string of the produced value.
    pub output_type: String,
}

/// A `<nodegraph>`: the concrete wiring of nodes behind a material.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeGraph {
    /// Graph name referenced by materials.
    pub name: String,
    /// Nodedef this graph implements (matched against `open_pbr_surface`
    /// during conversion).
    pub nodedef: String,
    /// Shading nodes in document order.
    pub nodes: Vec<Node>,
    /// Top-level graph outputs.
    pub outputs: Vec<Output>,
}

/// One shading node inside a [`NodeGraph`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Node {
    /// Node type (`constant`, `multiply`, `dielectric_bsdf`, ...); doubles as
    /// the nodedef lookup key.
    pub node_type: String,
    /// Instance name unique within the graph.
    pub name: String,
    /// Requested definition version, may be empty.
    pub version: String,
    /// Upstream node referenced by the `nodename` attribute. Only meaningful
    /// for `<output>` elements, which commonly reference their source node
    /// via the attribute instead of a child `<input>`.
    pub nodename: String,
    /// Child `<input>`/`<parameter>` elements.
    pub inputs: Vec<Input>,
}

/// An `<input>`/`<parameter>` child of a [`Node`]: either a literal value or
/// a connection to an upstream node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Input {
    /// Input name per the node's definition.
    pub name: String,
    /// MaterialX type string.
    pub input_type: String,
    /// Literal value as written (empty when connected instead).
    pub value: String,
    /// Upstream node instance name when this input is a connection.
    pub nodename: String,
    /// Which output of the upstream node feeds this input.
    pub output: String,
    /// UI minimum hint.
    pub uimin: String,
    /// UI maximum hint.
    pub uimax: String,
    /// UI display name.
    pub uiname: String,
    /// UI folder grouping.
    pub uifolder: String,
    /// Whether the input is advanced in UIs.
    pub uiadvanced: String,
    /// Documentation string.
    pub doc: String,
    /// Extra hint string.
    pub hint: String,
    /// Uniformity flag as written in the XML.
    pub uniform: String,
    /// Default geometric property binding, if any.
    pub defaultgeomprop: String,
    /// Interface parameter name when exposed upstream.
    pub interfacename: String,
    /// Raw string-typed value for `string` inputs.
    pub string: String,
}

/// A top-level `<output>` of a [`NodeGraph`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Output {
    /// Output name exposed by the graph.
    pub name: String,
    /// MaterialX type string.
    pub output_type: String,
    /// Source node instance feeding this output.
    pub nodename: String,
}
