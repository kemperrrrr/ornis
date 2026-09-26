//! MaterialX AST node definitions

use std::convert::Infallible;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Typed shading-node kind: the element name inside a `<nodegraph>`.
///
/// Closed over every node type [`crate::graph::evaluate::GraphEvaluator`]
/// can execute (plus the generic `<node>`/`<output>`/`<surface>`/`<edf>`
/// containers); anything else parses to [`MtlxNodeKind::Custom`] instead of
/// being dropped, so unknown elements stay visible to diagnostics rather
/// than vanishing in the parser. Serializes as the plain element string,
/// so stored JSON documents are unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MtlxNodeKind {
    /// Generic `<node>` element.
    Node,
    /// Graph `<output>` (also the pass-through marker in evaluation).
    Output,
    /// `<open_pbr_surface>` material root.
    OpenPbrSurface,
    /// `<open_pbr_anisotropy>` data node.
    OpenPbrAnisotropy,
    /// `<mix>` blend node.
    Mix,
    /// `<layer>` blend node.
    Layer,
    /// `<add>` arithmetic node.
    Add,
    /// `<multiply>` arithmetic node.
    Multiply,
    /// `<divide>` arithmetic node.
    Divide,
    /// `<subtract>` arithmetic node.
    Subtract,
    /// `<invert>` arithmetic node.
    Invert,
    /// `<clamp>` arithmetic node.
    Clamp,
    /// `<max>` arithmetic node.
    Max,
    /// `<min>` arithmetic node.
    Min,
    /// `<power>` arithmetic node.
    Power,
    /// `<sqrt>` arithmetic node.
    Sqrt,
    /// `<ifgreater>` arithmetic node.
    IfGreater,
    /// `<convert>` data node.
    Convert,
    /// `<combine2>` data node.
    Combine2,
    /// `<combine3>` data node.
    Combine3,
    /// `<combine4>` data node.
    Combine4,
    /// `<constant>` literal node.
    Constant,
    /// `<subsurface_bsdf>` shading node.
    SubsurfaceBsdf,
    /// `<dielectric_bsdf>` shading node.
    DielectricBsdf,
    /// `<conductor_bsdf>` shading node.
    ConductorBsdf,
    /// `<oren_nayar_diffuse_bsdf>` shading node.
    OrenNayarDiffuseBsdf,
    /// `<sheen_bsdf>` shading node.
    SheenBsdf,
    /// `<thin_film_bsdf>` shading node.
    ThinFilmBsdf,
    /// `<translucent_bsdf>` shading node.
    TranslucentBsdf,
    /// `<generalized_schlick_bsdf>` shading node.
    GeneralizedSchlickBsdf,
    /// `<uniform_edf>` emission node.
    UniformEdf,
    /// `<generalized_schlick_edf>` emission node.
    GeneralizedSchlickEdf,
    /// `<anisotropic_vdf>` volume node.
    AnisotropicVdf,
    /// `<surface>` shader container.
    Surface,
    /// `<edf>` emission container.
    Edf,
    /// Element outside the known vocabulary (e.g. `<frob>`): preserved
    /// verbatim, never executed.
    Custom(Box<str>),
}

impl MtlxNodeKind {
    /// Element name as written in the XML (`"multiply"`, …).
    pub fn as_str(&self) -> &str {
        match self {
            MtlxNodeKind::Node => "node",
            MtlxNodeKind::Output => "output",
            MtlxNodeKind::OpenPbrSurface => "open_pbr_surface",
            MtlxNodeKind::OpenPbrAnisotropy => "open_pbr_anisotropy",
            MtlxNodeKind::Mix => "mix",
            MtlxNodeKind::Layer => "layer",
            MtlxNodeKind::Add => "add",
            MtlxNodeKind::Multiply => "multiply",
            MtlxNodeKind::Divide => "divide",
            MtlxNodeKind::Subtract => "subtract",
            MtlxNodeKind::Invert => "invert",
            MtlxNodeKind::Clamp => "clamp",
            MtlxNodeKind::Max => "max",
            MtlxNodeKind::Min => "min",
            MtlxNodeKind::Power => "power",
            MtlxNodeKind::Sqrt => "sqrt",
            MtlxNodeKind::IfGreater => "ifgreater",
            MtlxNodeKind::Convert => "convert",
            MtlxNodeKind::Combine2 => "combine2",
            MtlxNodeKind::Combine3 => "combine3",
            MtlxNodeKind::Combine4 => "combine4",
            MtlxNodeKind::Constant => "constant",
            MtlxNodeKind::SubsurfaceBsdf => "subsurface_bsdf",
            MtlxNodeKind::DielectricBsdf => "dielectric_bsdf",
            MtlxNodeKind::ConductorBsdf => "conductor_bsdf",
            MtlxNodeKind::OrenNayarDiffuseBsdf => "oren_nayar_diffuse_bsdf",
            MtlxNodeKind::SheenBsdf => "sheen_bsdf",
            MtlxNodeKind::ThinFilmBsdf => "thin_film_bsdf",
            MtlxNodeKind::TranslucentBsdf => "translucent_bsdf",
            MtlxNodeKind::GeneralizedSchlickBsdf => "generalized_schlick_bsdf",
            MtlxNodeKind::UniformEdf => "uniform_edf",
            MtlxNodeKind::GeneralizedSchlickEdf => "generalized_schlick_edf",
            MtlxNodeKind::AnisotropicVdf => "anisotropic_vdf",
            MtlxNodeKind::Surface => "surface",
            MtlxNodeKind::Edf => "edf",
            MtlxNodeKind::Custom(name) => name,
        }
    }

    /// Whether this is an executable/structural element (anything but
    /// [`MtlxNodeKind::Custom`]).
    pub fn is_element(&self) -> bool {
        !matches!(self, MtlxNodeKind::Custom(_))
    }

    /// Whether an element name belongs to the node vocabulary.
    pub fn is_element_name(name: &str) -> bool {
        !matches!(
            Self::from_str(name).expect("FromStr is infallible"),
            MtlxNodeKind::Custom(_)
        )
    }
}

impl std::fmt::Display for MtlxNodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MtlxNodeKind {
    type Err = Infallible;

    /// Maps known element names to variants; anything else becomes
    /// [`MtlxNodeKind::Custom`] (never fails, so the parser keeps unknown
    /// elements visible instead of rejecting the document).
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Ok(match name {
            "node" => MtlxNodeKind::Node,
            "output" => MtlxNodeKind::Output,
            "open_pbr_surface" => MtlxNodeKind::OpenPbrSurface,
            "open_pbr_anisotropy" => MtlxNodeKind::OpenPbrAnisotropy,
            "mix" => MtlxNodeKind::Mix,
            "layer" => MtlxNodeKind::Layer,
            "add" => MtlxNodeKind::Add,
            "multiply" => MtlxNodeKind::Multiply,
            "divide" => MtlxNodeKind::Divide,
            "subtract" => MtlxNodeKind::Subtract,
            "invert" => MtlxNodeKind::Invert,
            "clamp" => MtlxNodeKind::Clamp,
            "max" => MtlxNodeKind::Max,
            "min" => MtlxNodeKind::Min,
            "power" => MtlxNodeKind::Power,
            "sqrt" => MtlxNodeKind::Sqrt,
            "ifgreater" => MtlxNodeKind::IfGreater,
            "convert" => MtlxNodeKind::Convert,
            "combine2" => MtlxNodeKind::Combine2,
            "combine3" => MtlxNodeKind::Combine3,
            "combine4" => MtlxNodeKind::Combine4,
            "constant" => MtlxNodeKind::Constant,
            "subsurface_bsdf" => MtlxNodeKind::SubsurfaceBsdf,
            "dielectric_bsdf" => MtlxNodeKind::DielectricBsdf,
            "conductor_bsdf" => MtlxNodeKind::ConductorBsdf,
            "oren_nayar_diffuse_bsdf" => MtlxNodeKind::OrenNayarDiffuseBsdf,
            "sheen_bsdf" => MtlxNodeKind::SheenBsdf,
            "thin_film_bsdf" => MtlxNodeKind::ThinFilmBsdf,
            "translucent_bsdf" => MtlxNodeKind::TranslucentBsdf,
            "generalized_schlick_bsdf" => MtlxNodeKind::GeneralizedSchlickBsdf,
            "uniform_edf" => MtlxNodeKind::UniformEdf,
            "generalized_schlick_edf" => MtlxNodeKind::GeneralizedSchlickEdf,
            "anisotropic_vdf" => MtlxNodeKind::AnisotropicVdf,
            "surface" => MtlxNodeKind::Surface,
            "edf" => MtlxNodeKind::Edf,
            other => MtlxNodeKind::Custom(other.into()),
        })
    }
}

impl From<&str> for MtlxNodeKind {
    fn from(name: &str) -> Self {
        Self::from_str(name).expect("FromStr is infallible")
    }
}

impl From<String> for MtlxNodeKind {
    fn from(name: String) -> Self {
        Self::from_str(&name).expect("FromStr is infallible")
    }
}

impl Default for MtlxNodeKind {
    /// Empty element name (mirrors the former `String::new()` default).
    fn default() -> Self {
        MtlxNodeKind::Custom("".into())
    }
}

impl Serialize for MtlxNodeKind {
    /// Encodes as the plain element string (stored documents unchanged).
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for MtlxNodeKind {
    /// Decodes from the plain element string (unknown names → `Custom`).
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::from)
    }
}

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
    /// Node kind (`constant`, `multiply`, `dielectric_bsdf`, ...); doubles as
    /// the nodedef lookup key. Serde-compatible with the former plain string.
    pub node_type: MtlxNodeKind,
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
