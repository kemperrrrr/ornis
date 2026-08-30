#![warn(missing_docs)]
//! MaterialX parser and OpenPBR material converter for Ornis Engine.
//!
//! [`parser`] streams `.mtlx` XML into the serde-friendly [`nodes`] AST;
//! [`graph`] evaluates node graphs and converts them into the engine's
//! GPU-ready [`OpenPBRMaterial`]. [`load_materialx`] is the one-shot
//! convenience entry point.

/// Graph evaluation and conversion of parsed documents to [`OpenPBRMaterial`].
pub mod graph;
/// serde-friendly AST mirroring the MaterialX XML structure.
pub mod nodes;
/// Streaming XML reader turning `.mtlx` text into [`crate::nodes::MaterialXDocument`]s.
pub mod parser;

pub use graph::{
    CodegenError, EvaluatedGraph, MaterialXConverter, MaterialXError, OpenPBRGraph, OutputValue,
    load_materialx_file, materialx_to_openpbr, parse_materialx,
};
pub use nodes::{Input, MaterialXDocument, Node, NodeDef, NodeGraph, Output};
pub use parser::MaterialXParser;

use ornis_render::OpenPBRMaterial;

/// Crate-root convenience wrapper around [`graph::load_materialx_file`].
///
/// # Errors
/// Same contract as [`graph::load_materialx_file`].
pub fn load_materialx<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<OpenPBRMaterial, MaterialXError> {
    let content = std::fs::read_to_string(path)?;
    materialx_to_openpbr(&content)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_MTLX: &str = r#"
<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="test">
    <output name="out" type="color3" nodename="color" />
    <constant name="color" type="color3">
      <parameter name="value" type="color3" value="0.8, 0.2, 0.2" />
    </constant>
  </nodegraph>
</materialx>
"#;

    #[test]
    fn test_parse_simple_materialx() {
        let parser = MaterialXParser::new();
        let document = parser.parse(SIMPLE_MTLX);
        assert!(document.is_ok());
    }

    #[test]
    fn test_load_from_string() {
        let parser = MaterialXParser::new();
        let document = parser.parse(SIMPLE_MTLX);
        assert!(document.is_ok());
    }
}
