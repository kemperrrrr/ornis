use std::collections::HashMap;
use std::borrow::Cow;

use html5ever::interface::tree_builder::{ElementFlags, NodeOrText, QuirksMode, TreeSink};
use html5ever::tendril::{StrTendril, TendrilSink};
use html5ever::{parse_document, Attribute, ExpandedName, QualName};

use crate::dom::{Document, Element, Node};

type Handle = usize;

struct HtmlSink {
    nodes: Vec<DomNode>,
    document: Handle,
    next_id: usize,
}

struct DomNode {
    parent: Option<Handle>,
    children: Vec<Handle>,
    data: DomNodeData,
}

enum DomNodeData {
    Document,
    Element {
        name: QualName,
        attrs: HashMap<String, String>,
    },
    Text(String),
}

impl HtmlSink {
    fn new_node(&mut self, data: DomNodeData) -> Handle {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(DomNode {
            parent: None,
            children: Vec::new(),
            data,
        });
        id
    }
}

impl TreeSink for HtmlSink {
    type Handle = Handle;
    type Output = Document;

    fn finish(self) -> Document {
        fn build_node(nodes: &[DomNode], handle: Handle) -> Node {
            match &nodes[handle].data {
                DomNodeData::Document => {
                    let children: Vec<Node> = nodes[handle]
                        .children
                        .iter()
                        .filter_map(|&child| {
                            let n = build_node(nodes, child);
                            match &n {
                                Node::Text(s) if s.trim().is_empty() => None,
                                _ => Some(n),
                            }
                        })
                        .collect();
                    if children.len() == 1 {
                        children.into_iter().next().unwrap()
                    } else {
                        Node::Element(Element {
                            tag: "#document".into(),
                            attrs: HashMap::new(),
                            children,
                        })
                    }
                }
                DomNodeData::Element { name, attrs } => {
                    let children = nodes[handle]
                        .children
                        .iter()
                        .map(|&child| build_node(nodes, child))
                        .collect();
                    Node::Element(Element {
                        tag: name.local.to_string(),
                        attrs: attrs.clone(),
                        children,
                    })
                }
                DomNodeData::Text(text) => Node::Text(text.clone()),
            }
        }

        let root = build_node(&self.nodes, self.document);
        Document { root }
    }

    fn parse_error(&mut self, _msg: Cow<'static, str>) {}

    fn get_document(&mut self) -> Handle {
        self.document
    }

    fn get_template_contents(&mut self, _target: &Handle) -> Handle {
        panic!("templates not supported");
    }

    fn set_quirks_mode(&mut self, _mode: QuirksMode) {}

    fn same_node(&self, x: &Handle, y: &Handle) -> bool {
        x == y
    }

    fn elem_name<'a>(&'a self, target: &'a Handle) -> ExpandedName<'a> {
        match &self.nodes[*target].data {
            DomNodeData::Element { name, .. } => name.expanded(),
            _ => panic!("not an element"),
        }
    }

    fn create_element(
        &mut self,
        name: QualName,
        attrs: Vec<Attribute>,
        _flags: ElementFlags,
    ) -> Handle {
        let mut our_attrs = HashMap::new();
        for attr in attrs {
            our_attrs.insert(attr.name.local.to_string(), attr.value.to_string());
        }
        self.new_node(DomNodeData::Element {
            name,
            attrs: our_attrs,
        })
    }

    fn create_comment(&mut self, _text: StrTendril) -> Handle {
        self.new_node(DomNodeData::Text(String::new()))
    }

    fn create_pi(&mut self, _target: StrTendril, _data: StrTendril) -> Handle {
        self.new_node(DomNodeData::Text(String::new()))
    }

    fn append(&mut self, parent: &Handle, child: NodeOrText<Handle>) {
        let child_handle = match child {
            NodeOrText::AppendNode(node) => node,
            NodeOrText::AppendText(text) => {
                // Try to merge with previous text sibling
                if let Some(&last) = self.nodes[*parent].children.last() {
                    if let DomNodeData::Text(ref mut existing) = self.nodes[last].data {
                        existing.push_str(&text);
                        return;
                    }
                }
                self.new_node(DomNodeData::Text(text.to_string()))
            }
        };
        self.nodes[child_handle].parent = Some(*parent);
        self.nodes[*parent].children.push(child_handle);
    }

    fn append_before_sibling(&mut self, _sibling: &Handle, _child: NodeOrText<Handle>) {
        // Not needed for simple parsing
    }

    fn append_based_on_parent_node(
        &mut self,
        element: &Handle,
        prev_element: &Handle,
        child: NodeOrText<Handle>,
    ) {
        if self.nodes[*element].parent.is_some() {
            self.append_before_sibling(element, child);
        } else {
            self.append(prev_element, child);
        }
    }

    fn append_doctype_to_document(
        &mut self,
        _name: StrTendril,
        _public_id: StrTendril,
        _system_id: StrTendril,
    ) {
    }

    fn add_attrs_if_missing(&mut self, target: &Handle, attrs: Vec<Attribute>) {
        if let DomNodeData::Element { attrs: our, .. } = &mut self.nodes[*target].data {
            for attr in attrs {
                let key = attr.name.local.to_string();
                our.entry(key).or_insert_with(|| attr.value.to_string());
            }
        }
    }

    fn remove_from_parent(&mut self, _target: &Handle) {}

    fn reparent_children(&mut self, _node: &Handle, _new_parent: &Handle) {}

    fn mark_script_already_started(&mut self, _node: &Handle) {}
}

pub fn parse_html(html: &str) -> Document {
    let sink = HtmlSink {
        nodes: vec![DomNode {
            parent: None,
            children: Vec::new(),
            data: DomNodeData::Document,
        }],
        document: 0,
        next_id: 1,
    };

    parse_document(sink, Default::default())
        .from_utf8()
        .one(html.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dom::find_by_id;

    #[test]
    fn test_parse_simple_html() {
        let doc = parse_html("<html><body><h1 id=\"title\">Hello</h1></body></html>");
        let h1 = find_by_id(&doc.root, "title");
        assert!(h1.is_some());
        assert_eq!(h1.unwrap().tag, "h1");
    }
}
