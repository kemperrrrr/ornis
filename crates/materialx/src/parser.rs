//! MaterialX XML parser

use crate::nodes::NodeDefInput;
use crate::{Input, MaterialXDocument, MaterialXError, Node, NodeDef, NodeGraph};
use quick_xml::Reader;
use quick_xml::events::{BytesEnd, BytesStart, Event};

/// Every element that denotes a shading node inside a `<nodegraph>`.
/// Kept in sync with the node types `GraphEvaluator` can execute; the same
/// set is used for Start and End events so the two can never diverge
/// (a node opened but never closed here is silently dropped).
fn is_node_element(name: &str) -> bool {
    matches!(
        name,
        "node"
            | "output"
            | "open_pbr_surface"
            | "open_pbr_anisotropy"
            | "mix"
            | "layer"
            | "add"
            | "multiply"
            | "divide"
            | "subtract"
            | "invert"
            | "clamp"
            | "max"
            | "min"
            | "power"
            | "sqrt"
            | "ifgreater"
            | "convert"
            | "combine2"
            | "combine3"
            | "combine4"
            | "constant"
            | "subsurface_bsdf"
            | "dielectric_bsdf"
            | "conductor_bsdf"
            | "oren_nayar_diffuse_bsdf"
            | "sheen_bsdf"
            | "thin_film_bsdf"
            | "translucent_bsdf"
            | "generalized_schlick_bsdf"
            | "uniform_edf"
            | "generalized_schlick_edf"
            | "anisotropic_vdf"
            | "surface"
    )
}

/// Streaming quick-xml parser producing a [`crate::nodes::MaterialXDocument`]
/// from `.mtlx` text. Stateless: create with [`MaterialXParser::new`] and reuse freely.
pub struct MaterialXParser;

impl Default for MaterialXParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming parse state threaded through the Start/End event handlers.
struct ParseState {
    document: MaterialXDocument,
    current_nodedef: Option<NodeDef>,
    current_nodegraph: Option<NodeGraph>,
    current_node: Option<Node>,
    in_nodegraph: bool,
}

impl ParseState {
    fn new() -> Self {
        Self {
            document: MaterialXDocument::default(),
            current_nodedef: None,
            current_nodegraph: None,
            current_node: None,
            in_nodegraph: false,
        }
    }

    fn handle_start(&mut self, e: &BytesStart) -> Result<(), MaterialXError> {
        match e.name().into_inner() {
            "nodedef" => {
                self.current_nodedef = Some(NodeDef::from_bytes_start(e)?);
            }
            "nodegraph" => {
                self.in_nodegraph = true;
                self.current_nodegraph = Some(NodeGraph::from_bytes_start(e)?);
            }
            "input" | "parameter" => {
                if let Some(node) = &mut self.current_node {
                    node.inputs.push(Input::from_bytes_start(e)?);
                } else if let Some(nodedef) = &mut self.current_nodedef {
                    nodedef.inputs.push(NodeDefInput::from_bytes_start(e)?);
                }
            }
            name if is_node_element(name) && self.in_nodegraph => {
                self.current_node = Some(Node::from_bytes_start(e)?);
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_end(&mut self, e: &BytesEnd) {
        match e.name().into_inner() {
            "nodedef" => {
                if let Some(nd) = self.current_nodedef.take() {
                    self.document.nodedefs.push(nd);
                }
            }
            "nodegraph" => {
                self.in_nodegraph = false;
                if let Some(ng) = self.current_nodegraph.take() {
                    self.document.nodegraphs.push(ng);
                }
            }
            name if is_node_element(name) => {
                if self.in_nodegraph
                    && let Some(node) = self.current_node.take()
                    && let Some(graph) = &mut self.current_nodegraph
                {
                    graph.nodes.push(node);
                }
            }
            _ => {}
        }
    }
}

impl MaterialXParser {
    /// Create a stateless parser instance.
    pub fn new() -> Self {
        Self
    }

    /// Parse `.mtlx` XML into a structured document of nodedefs and nodegraphs.
    ///
    /// Self-closing tags are expanded so `<input .../>` children are never
    /// dropped. Unknown elements are ignored rather than rejected.
    ///
    /// # Errors
    /// [`crate::MaterialXError::Xml`]/[`crate::MaterialXError::Attr`]/
    /// [`crate::MaterialXError::Utf8`] from the underlying reader.
    pub fn parse(&self, content: &str) -> Result<MaterialXDocument, MaterialXError> {
        let mut reader = Reader::from_str(content);
        reader.config_mut().trim_text(true);
        // Self-closing tags (`<input ... />`) arrive as Start+End pairs;
        // without this they surface as `Event::Empty` and are dropped.
        reader.config_mut().expand_empty_elements = true;

        let mut state = ParseState::new();
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => state.handle_start(e)?,
                Ok(Event::End(ref e)) => state.handle_end(e),
                Ok(Event::Eof) => break,
                Ok(_) => {}
                Err(e) => return Err(MaterialXError::Xml(e)),
            }
            buf.clear();
        }

        Ok(state.document)
    }
}

impl NodeDef {
    fn from_bytes_start(e: &BytesStart) -> Result<Self, MaterialXError> {
        let mut nodedef = NodeDef {
            name: String::new(),
            node: String::new(),
            nodegroup: String::new(),
            version: String::new(),
            isdefaultversion: false,
            doc: String::new(),
            uiname: String::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };

        for attr in e.attributes() {
            let attr = attr?;
            let key = attr.key.into_inner();
            let value: &str = &attr.value;

            match key {
                "name" => nodedef.name = value.to_string(),
                "node" => nodedef.node = value.to_string(),
                "nodegroup" => nodedef.nodegroup = value.to_string(),
                "version" => nodedef.version = value.to_string(),
                "isdefaultversion" => nodedef.isdefaultversion = value == "true",
                "doc" => nodedef.doc = value.to_string(),
                "uiname" => nodedef.uiname = value.to_string(),
                _ => {}
            }
        }

        Ok(nodedef)
    }
}

impl NodeGraph {
    fn from_bytes_start(e: &BytesStart) -> Result<Self, MaterialXError> {
        let mut graph = NodeGraph {
            name: String::new(),
            nodedef: String::new(),
            nodes: Vec::new(),
            outputs: Vec::new(),
        };

        for attr in e.attributes() {
            let attr = attr?;
            let key = attr.key.into_inner();
            let value: &str = &attr.value;

            match key {
                "name" => graph.name = value.to_string(),
                "nodedef" => graph.nodedef = value.to_string(),
                _ => {}
            }
        }

        Ok(graph)
    }
}

impl Node {
    fn from_bytes_start(e: &BytesStart) -> Result<Self, MaterialXError> {
        let mut node = Node {
            node_type: String::new(),
            name: String::new(),
            version: String::new(),
            nodename: String::new(),
            inputs: Vec::new(),
        };

        let name_binding = e.name();
        let name = name_binding.into_inner();
        node.node_type = name.to_string();

        for attr in e.attributes() {
            let attr = attr?;
            let key = attr.key.into_inner();
            let value: &str = &attr.value;

            match key {
                "name" => node.name = value.to_string(),
                "version" => node.version = value.to_string(),
                "nodename" => node.nodename = value.to_string(),
                _ => {}
            }
        }

        Ok(node)
    }
}

impl NodeDefInput {
    fn from_bytes_start(e: &BytesStart) -> Result<Self, MaterialXError> {
        let mut input = NodeDefInput {
            name: String::new(),
            input_type: String::new(),
            value: String::new(),
            uimin: String::new(),
            uimax: String::new(),
            uisoftmin: String::new(),
            uisoftmax: String::new(),
            uiname: String::new(),
            uifolder: String::new(),
            uiadvanced: String::new(),
            doc: String::new(),
            hint: String::new(),
            uniform: String::new(),
            defaultgeomprop: String::new(),
            interfacename: String::new(),
            string: String::new(),
        };

        for attr in e.attributes() {
            let attr = attr?;
            let key = attr.key.into_inner();
            let value: &str = &attr.value;

            match key {
                "name" => input.name = value.to_string(),
                "type" => input.input_type = value.to_string(),
                "value" => input.value = value.to_string(),
                "uisoftmin" => input.uisoftmin = value.to_string(),
                "uisoftmax" => input.uisoftmax = value.to_string(),
                _ => input.set_meta_attr(key, value),
            }
        }

        Ok(input)
    }

    /// Apply the metadata/UI attributes common to `<input>` and `<parameter>`.
    fn set_meta_attr(&mut self, key: &str, value: &str) {
        match key {
            "uimin" => self.uimin = value.to_string(),
            "uimax" => self.uimax = value.to_string(),
            "uisoftmin" => self.uisoftmin = value.to_string(),
            "uisoftmax" => self.uisoftmax = value.to_string(),
            "uiname" => self.uiname = value.to_string(),
            "uifolder" => self.uifolder = value.to_string(),
            "uiadvanced" => self.uiadvanced = value.to_string(),
            "doc" => self.doc = value.to_string(),
            "hint" => self.hint = value.to_string(),
            "uniform" => self.uniform = value.to_string(),
            "defaultgeomprop" => self.defaultgeomprop = value.to_string(),
            "interfacename" => self.interfacename = value.to_string(),
            "string" => self.string = value.to_string(),
            _ => {}
        }
    }
}

impl Input {
    fn from_bytes_start(e: &BytesStart) -> Result<Self, MaterialXError> {
        let mut input = Input {
            name: String::new(),
            input_type: String::new(),
            value: String::new(),
            nodename: String::new(),
            output: String::new(),
            uimin: String::new(),
            uimax: String::new(),
            uiname: String::new(),
            uifolder: String::new(),
            uiadvanced: String::new(),
            doc: String::new(),
            hint: String::new(),
            uniform: String::new(),
            defaultgeomprop: String::new(),
            interfacename: String::new(),
            string: String::new(),
        };

        for attr in e.attributes() {
            let attr = attr?;
            let key = attr.key.into_inner();
            let value: &str = &attr.value;

            match key {
                "name" => input.name = value.to_string(),
                "type" => input.input_type = value.to_string(),
                "value" => input.value = value.to_string(),
                "nodename" => input.nodename = value.to_string(),
                "output" => input.output = value.to_string(),
                _ => input.set_meta_attr(key, value),
            }
        }

        Ok(input)
    }

    /// Apply the metadata/UI attributes common to `<input>` and `<parameter>`.
    fn set_meta_attr(&mut self, key: &str, value: &str) {
        match key {
            "uimin" => self.uimin = value.to_string(),
            "uimax" => self.uimax = value.to_string(),
            "uiname" => self.uiname = value.to_string(),
            "uifolder" => self.uifolder = value.to_string(),
            "uiadvanced" => self.uiadvanced = value.to_string(),
            "doc" => self.doc = value.to_string(),
            "hint" => self.hint = value.to_string(),
            "uniform" => self.uniform = value.to_string(),
            "defaultgeomprop" => self.defaultgeomprop = value.to_string(),
            "interfacename" => self.interfacename = value.to_string(),
            "string" => self.string = value.to_string(),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> Result<MaterialXDocument, MaterialXError> {
        let content = format!(
            r#"<?xml version="1.0"?><materialx version="1.39">{}</materialx>"#,
            body
        );
        MaterialXParser::new().parse(&content)
    }

    fn node_names(doc: &MaterialXDocument) -> Vec<&str> {
        doc.nodegraphs[0]
            .nodes
            .iter()
            .map(|n| n.name.as_str())
            .collect()
    }

    /// Regression: the parser whitelist silently dropped every node type the
    /// evaluator supports but that was missing from the Start/End matches
    /// (constant, divide, subtract, invert, clamp, ...). All of them must be
    /// captured now.
    #[test]
    fn parses_all_evaluator_node_types() {
        let doc = parse(
            r#"<nodegraph name="g">
                <constant name="n_constant" type="float" />
                <add name="n_add" type="float" />
                <subtract name="n_subtract" type="float" />
                <multiply name="n_multiply" type="float" />
                <divide name="n_divide" type="float" />
                <invert name="n_invert" type="float" />
                <clamp name="n_clamp" type="float" />
                <max name="n_max" type="float" />
                <min name="n_min" type="float" />
                <power name="n_power" type="float" />
                <sqrt name="n_sqrt" type="float" />
                <ifgreater name="n_ifgreater" type="float" />
                <convert name="n_convert" type="float" />
                <combine2 name="n_combine2" type="vector2" />
                <combine3 name="n_combine3" type="vector3" />
                <combine4 name="n_combine4" type="vector4" />
                <mix name="n_mix" type="float" />
                <layer name="n_layer" type="float" />
                <open_pbr_surface name="n_open_pbr" type="surfaceshader" />
                <open_pbr_anisotropy name="n_anisotropy" type="vector2" />
                <node name="n_generic" type="float" />
                <surface name="n_surface" type="surfaceshader" />
                <subsurface_bsdf name="n_subsurface" type="BSDF" />
                <dielectric_bsdf name="n_dielectric" type="BSDF" />
                <conductor_bsdf name="n_conductor" type="BSDF" />
                <oren_nayar_diffuse_bsdf name="n_oren" type="BSDF" />
                <sheen_bsdf name="n_sheen" type="BSDF" />
                <thin_film_bsdf name="n_thinfilm" type="BSDF" />
                <translucent_bsdf name="n_translucent" type="BSDF" />
                <generalized_schlick_bsdf name="n_schlick" type="BSDF" />
                <uniform_edf name="n_uniform_edf" type="EDF" />
                <generalized_schlick_edf name="n_schlick_edf" type="EDF" />
                <anisotropic_vdf name="n_vdf" type="VDF" />
                <output name="n_output" type="float" />
            </nodegraph>"#,
        )
        .unwrap();

        assert_eq!(doc.nodegraphs.len(), 1);
        assert_eq!(doc.nodegraphs[0].nodes.len(), 34);
        for expected in [
            "n_constant",
            "n_divide",
            "n_subtract",
            "n_invert",
            "n_clamp",
            "n_output",
        ] {
            assert!(
                node_names(&doc).contains(&expected),
                "node {} was dropped",
                expected
            );
        }
    }

    #[test]
    fn parses_node_type_and_attributes() {
        let doc = parse(
            r#"<nodegraph name="g" nodedef="ND_test">
                <multiply name="m" version="1.0">
                  <input name="in1" type="float" value="2.0" />
                  <input name="in2" type="float" nodename="other" output="out" />
                </multiply>
            </nodegraph>"#,
        )
        .unwrap();

        let graph = &doc.nodegraphs[0];
        assert_eq!(graph.name, "g");
        assert_eq!(graph.nodedef, "ND_test");

        let node = &graph.nodes[0];
        assert_eq!(node.node_type, "multiply");
        assert_eq!(node.name, "m");
        assert_eq!(node.version, "1.0");
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.inputs[0].name, "in1");
        assert_eq!(node.inputs[0].input_type, "float");
        assert_eq!(node.inputs[0].value, "2.0");
        assert_eq!(node.inputs[1].nodename, "other");
        assert_eq!(node.inputs[1].output, "out");
    }

    /// Regression: self-closing tags arrive as `Event::Empty` unless
    /// `expand_empty_elements` is set — they used to be dropped entirely,
    /// which emptied every node's input list in typical MaterialX files.
    #[test]
    fn parses_self_closing_nodes_and_inputs() {
        let doc = parse(
            r#"<nodegraph name="g">
                <constant name="c" type="color3">
                  <parameter name="value" type="color3" value="0.8, 0.2, 0.2" />
                </constant>
                <constant name="self_closing" type="float" />
            </nodegraph>"#,
        )
        .unwrap();

        let graph = &doc.nodegraphs[0];
        assert_eq!(node_names(&doc), ["c", "self_closing"]);
        assert_eq!(graph.nodes[0].inputs.len(), 1);
        assert_eq!(graph.nodes[0].inputs[0].value, "0.8, 0.2, 0.2");
    }

    #[test]
    fn parses_nodedef_with_inputs() {
        let doc = parse(
            r#"<nodedef name="ND_clamp_float" node="clamp" nodegroup="math" version="1.0" isdefaultversion="true" uiname="Clamp">
                <input name="in" type="float" value="0.5" uimin="0.0" uimax="1.0" uisoftmin="-1.0" uisoftmax="2.0" />
                <input name="low" type="float" value="0.0" />
            </nodedef>"#,
        )
        .unwrap();

        assert_eq!(doc.nodedefs.len(), 1);
        let nd = &doc.nodedefs[0];
        assert_eq!(nd.name, "ND_clamp_float");
        assert_eq!(nd.node, "clamp");
        assert_eq!(nd.nodegroup, "math");
        assert_eq!(nd.version, "1.0");
        assert!(nd.isdefaultversion);
        assert_eq!(nd.uiname, "Clamp");
        assert_eq!(nd.inputs.len(), 2);
        assert_eq!(nd.inputs[0].name, "in");
        assert_eq!(nd.inputs[0].value, "0.5");
        assert_eq!(nd.inputs[0].uimin, "0.0");
        assert_eq!(nd.inputs[0].uisoftmax, "2.0");
        assert_eq!(nd.inputs[1].name, "low");
    }

    /// Regression: the `nodename` attribute on `<output>` (and node) elements
    /// was silently dropped, severing output→node connections expressed via
    /// the attribute instead of a child `<input>`.
    #[test]
    fn parses_output_nodename_attribute() {
        let doc = parse(
            r#"<nodegraph name="g">
                <constant name="c" type="float" />
                <output name="out" type="float" nodename="c" />
            </nodegraph>"#,
        )
        .unwrap();

        let graph = &doc.nodegraphs[0];
        let output = &graph.nodes[1];
        assert_eq!(output.node_type, "output");
        assert_eq!(output.name, "out");
        assert_eq!(output.nodename, "c");
        // Nodes without the attribute default to an empty nodename.
        assert_eq!(graph.nodes[0].nodename, "");
    }

    #[test]
    fn ignores_unknown_and_out_of_graph_elements() {
        let doc = parse(
            r#"<constant name="outside" type="float" />
            <nodegraph name="g">
                <constant name="inside" type="float" />
                <frob name="unknown" type="float" />
            </nodegraph>"#,
        )
        .unwrap();

        assert_eq!(node_names(&doc), ["inside"]);
    }

    #[test]
    fn malformed_xml_returns_error() {
        let result = parse(r#"<nodegraph name="g""#);
        assert!(result.is_err());
    }

    #[test]
    fn empty_document_parses() {
        let doc = parse("").unwrap();
        assert!(doc.nodedefs.is_empty());
        assert!(doc.nodegraphs.is_empty());
    }
}
