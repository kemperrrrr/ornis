//! MaterialX XML parser

use crate::{Input, MaterialXDocument, MaterialXError, Node, NodeDef, NodeGraph, Output};
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};

pub struct MaterialXParser;

impl Default for MaterialXParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialXParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse(&self, content: &str) -> Result<MaterialXDocument, MaterialXError> {
        let mut reader = Reader::from_str(content);
        reader.config_mut().trim_text(true);

        let mut document = MaterialXDocument::default();
        let mut current_nodedef: Option<NodeDef> = None;
        let mut current_nodegraph: Option<NodeGraph> = None;
        let mut current_node: Option<Node> = None;
        let mut in_nodegraph = false;

        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => match e.name().as_ref() {
                    b"nodedef" => {
                        current_nodedef = Some(NodeDef::from_bytes_start(e)?);
                    }
                    b"nodegraph" => {
                        in_nodegraph = true;
                        current_nodegraph = Some(NodeGraph::from_bytes_start(e)?);
                    }
                    b"node"
                    | b"open_pbr_surface"
                    | b"mix"
                    | b"layer"
                    | b"add"
                    | b"multiply"
                    | b"subsurface_bsdf"
                    | b"dielectric_bsdf"
                    | b"conductor_bsdf"
                    | b"oren_nayar_diffuse_bsdf"
                    | b"sheen_bsdf"
                    | b"thin_film_bsdf"
                    | b"translucent_bsdf"
                    | b"generalized_schlick_bsdf"
                    | b"uniform_edf"
                    | b"generalized_schlick_edf"
                    | b"anisotropic_vdf"
                    | b"surface"
                    | b"output" => {
                        if in_nodegraph {
                            current_node = Some(Node::from_bytes_start(e)?);
                        }
                    }
                    b"input" => {
                        if let Some(node) = &mut current_node {
                            node.inputs.push(Input::from_bytes_start(e)?);
                        }
                    }
                    b"output" => {
                        if let Some(graph) = &mut current_nodegraph {
                            graph.outputs.push(Output::from_bytes_start(e)?);
                        }
                    }
                    _ => {}
                },
                Ok(Event::End(ref e)) => match e.name().as_ref() {
                    b"nodedef" => {
                        if let Some(nd) = current_nodedef.take() {
                            document.nodedefs.push(nd);
                        }
                    }
                    b"nodegraph" => {
                        in_nodegraph = false;
                        if let Some(ng) = current_nodegraph.take() {
                            document.nodegraphs.push(ng);
                        }
                    }
                    b"node"
                    | b"open_pbr_surface"
                    | b"mix"
                    | b"layer"
                    | b"add"
                    | b"multiply"
                    | b"subsurface_bsdf"
                    | b"dielectric_bsdf"
                    | b"conductor_bsdf"
                    | b"oren_nayar_diffuse_bsdf"
                    | b"sheen_bsdf"
                    | b"thin_film_bsdf"
                    | b"translucent_bsdf"
                    | b"generalized_schlick_bsdf"
                    | b"uniform_edf"
                    | b"generalized_schlick_edf"
                    | b"anisotropic_vdf"
                    | b"surface" => {
                        if in_nodegraph
                            && let Some(node) = current_node.take()
                            && let Some(graph) = &mut current_nodegraph
                        {
                            graph.nodes.push(node);
                        }
                    }
                    _ => {}
                },
                Ok(Event::Eof) => break,
                Ok(_) => {}
                Err(e) => return Err(MaterialXError::Xml(e)),
            }
            buf.clear();
        }

        Ok(document)
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
            let key = std::str::from_utf8(attr.key.as_ref())?;
            let value = std::str::from_utf8(&attr.value)?;

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
            let key = std::str::from_utf8(attr.key.as_ref())?;
            let value = std::str::from_utf8(&attr.value)?;

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
            inputs: Vec::new(),
        };

        let name_binding = e.name();
        let name_bytes = name_binding.as_ref();
        let name = std::str::from_utf8(name_bytes)?;
        node.node_type = name.to_string();

        for attr in e.attributes() {
            let attr = attr?;
            let key = std::str::from_utf8(attr.key.as_ref())?;
            let value = std::str::from_utf8(&attr.value)?;

            match key {
                "name" => node.name = value.to_string(),
                "version" => node.version = value.to_string(),
                _ => {}
            }
        }

        Ok(node)
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
            let key = std::str::from_utf8(attr.key.as_ref())?;
            let value = std::str::from_utf8(&attr.value)?;

            match key {
                "name" => input.name = value.to_string(),
                "type" => input.input_type = value.to_string(),
                "value" => input.value = value.to_string(),
                "nodename" => input.nodename = value.to_string(),
                "output" => input.output = value.to_string(),
                "uimin" => input.uimin = value.to_string(),
                "uimax" => input.uimax = value.to_string(),
                "uiname" => input.uiname = value.to_string(),
                "uifolder" => input.uifolder = value.to_string(),
                "uiadvanced" => input.uiadvanced = value.to_string(),
                "doc" => input.doc = value.to_string(),
                "hint" => input.hint = value.to_string(),
                "uniform" => input.uniform = value.to_string(),
                "defaultgeomprop" => input.defaultgeomprop = value.to_string(),
                "interfacename" => input.interfacename = value.to_string(),
                "string" => input.string = value.to_string(),
                _ => {}
            }
        }

        Ok(input)
    }
}

impl Output {
    fn from_bytes_start(e: &BytesStart) -> Result<Self, MaterialXError> {
        let mut output = Output {
            name: String::new(),
            output_type: String::new(),
            nodename: String::new(),
        };

        for attr in e.attributes() {
            let attr = attr?;
            let key = std::str::from_utf8(attr.key.as_ref())?;
            let value = std::str::from_utf8(&attr.value)?;

            match key {
                "name" => output.name = value.to_string(),
                "type" => output.output_type = value.to_string(),
                "nodename" => output.nodename = value.to_string(),
                _ => {}
            }
        }

        Ok(output)
    }
}
