//! Validation of graph wiring before evaluation.
//!
//! Finds the OpenPBR graph and resolves every declared input to a
//! connected node, a literal value or a nodedef default, rejecting
//! cycles and missing data.

use super::evaluate::{GraphEvaluator, VisitState};
use super::parse::parse_constant;
use super::{CodegenError, MaterialXConverter, OutputValue};
use crate::nodes::{Node, NodeDef, NodeGraph};
use std::collections::HashMap;

impl MaterialXConverter {
    pub(crate) fn find_openpbr_graph(&self) -> Result<&NodeGraph, CodegenError> {
        for graph in &self.document.nodegraphs {
            if graph.nodedef.contains("open_pbr_surface") {
                return Ok(graph);
            }
        }
        Err(CodegenError::GraphNotFound(
            "OpenPBR surface shader graph not found".to_string(),
        ))
    }
}

impl<'a> GraphEvaluator<'a> {
    /// Reject re-entrant visits (cycles) and serve memoized results.
    pub(crate) fn check_visit_state(&self, node: &Node) -> Result<(), CodegenError> {
        match self.visited.get(&node.name) {
            Some(VisitState::Visiting) => Err(CodegenError::CyclicDependency),
            Some(VisitState::Visited) => self
                .node_values
                .get(&node.name)
                .cloned()
                .map(|_| ())
                .ok_or_else(|| CodegenError::InputNotFound(node.name.clone())),
            None => Ok(()),
        }
    }

    /// Resolve every declared input: connected node, literal value, or the
    /// node definition default.
    pub(crate) fn collect_input_values(
        &mut self,
        node: &Node,
        node_def: &NodeDef,
    ) -> Result<HashMap<String, OutputValue>, CodegenError> {
        let mut input_values = HashMap::new();
        for input in &node.inputs {
            let value = if !input.nodename.is_empty() {
                self.evaluate_connected(&input.nodename)?
            } else if !input.value.is_empty() {
                parse_constant(&input.value, &input.input_type)?
            } else if let Some(def) = node_def.inputs.iter().find(|d| d.name == input.name) {
                parse_constant(&def.value, &def.input_type)?
            } else {
                return Err(CodegenError::InputNotFound(input.name.clone()));
            };
            input_values.insert(input.name.clone(), value);
        }
        Ok(input_values)
    }

    /// Evaluate a node referenced through a `nodename` attribute.
    pub(crate) fn evaluate_connected(
        &mut self,
        nodename: &str,
    ) -> Result<OutputValue, CodegenError> {
        let connected_node = self
            .graph
            .nodes
            .iter()
            .find(|n| n.name == nodename)
            .ok_or_else(|| CodegenError::InputNotFound(nodename.to_string()))?;
        self.evaluate_node(connected_node)
    }
}

#[cfg(test)]
mod tests {
    use super::super::evaluate::GraphEvaluator;
    use crate::graph::test_helpers::{assert_close, eval, eval_err, float};
    use crate::graph::{CodegenError, MaterialXConverter, MaterialXError, materialx_to_openpbr};
    use crate::parser::MaterialXParser;

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
}
