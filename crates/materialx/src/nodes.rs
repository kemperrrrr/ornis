//! MaterialX AST node definitions

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialXDocument {
    pub nodedefs: Vec<NodeDef>,
    pub nodegraphs: Vec<NodeGraph>,
}

impl Default for MaterialXDocument {
    fn default() -> Self {
        Self {
            nodedefs: Vec::new(),
            nodegraphs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDef {
    pub name: String,
    pub node: String,
    pub nodegroup: String,
    pub version: String,
    pub isdefaultversion: bool,
    pub doc: String,
    pub uiname: String,
    pub inputs: Vec<NodeDefInput>,
    pub outputs: Vec<NodeDefOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDefInput {
    pub name: String,
    pub input_type: String,
    pub value: String,
    pub uimin: String,
    pub uimax: String,
    pub uisoftmin: String,
    pub uisoftmax: String,
    pub uiname: String,
    pub uifolder: String,
    pub uiadvanced: String,
    pub doc: String,
    pub hint: String,
    pub uniform: String,
    pub defaultgeomprop: String,
    pub interfacename: String,
    pub string: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDefOutput {
    pub name: String,
    pub output_type: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeGraph {
    pub name: String,
    pub nodedef: String,
    pub nodes: Vec<Node>,
    pub outputs: Vec<Output>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Node {
    pub node_type: String,
    pub name: String,
    pub version: String,
    pub inputs: Vec<Input>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Input {
    pub name: String,
    pub input_type: String,
    pub value: String,
    pub nodename: String,
    pub output: String,
    pub uimin: String,
    pub uimax: String,
    pub uiname: String,
    pub uifolder: String,
    pub uiadvanced: String,
    pub doc: String,
    pub hint: String,
    pub uniform: String,
    pub defaultgeomprop: String,
    pub interfacename: String,
    pub string: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Output {
    pub name: String,
    pub output_type: String,
    pub nodename: String,
}