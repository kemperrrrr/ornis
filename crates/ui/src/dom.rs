use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Document {
    pub root: Node,
}

#[derive(Debug, Clone)]
pub enum Node {
    Element(Element),
    Text(String),
}

#[derive(Debug, Clone)]
pub struct Element {
    pub tag: String,
    pub attrs: HashMap<String, String>,
    pub children: Vec<Node>,
}

impl Document {
    pub fn new() -> Self {
        Document {
            root: Node::Element(Element::new("html")),
        }
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

impl Element {
    pub fn new(tag: &str) -> Self {
        Element {
            tag: tag.to_string(),
            attrs: HashMap::new(),
            children: Vec::new(),
        }
    }

    pub fn id(&self) -> Option<&str> {
        self.attrs.get("id").map(|s| s.as_str())
    }

    pub fn classes(&self) -> Vec<&str> {
        self.attrs
            .get("class")
            .map(|s| s.split_whitespace().collect())
            .unwrap_or_default()
    }

    pub fn get_attr(&self, name: &str) -> Option<&str> {
        self.attrs.get(name).map(|s| s.as_str())
    }
}

pub fn find_by_id<'a>(node: &'a Node, id: &str) -> Option<&'a Element> {
    match node {
        Node::Element(el) => {
            if el.id() == Some(id) {
                return Some(el);
            }
            for child in &el.children {
                if let Some(found) = find_by_id(child, id) {
                    return Some(found);
                }
            }
            None
        }
        Node::Text(_) => None,
    }
}

pub fn find_by_tag<'a>(node: &'a Node, tag: &str) -> Vec<&'a Element> {
    let mut result = Vec::new();
    find_by_tag_inner(node, tag, &mut result);
    result
}

fn find_by_tag_inner<'a>(node: &'a Node, tag: &str, result: &mut Vec<&'a Element>) {
    match node {
        Node::Element(el) => {
            if el.tag == tag {
                result.push(el);
            }
            for child in &el.children {
                find_by_tag_inner(child, tag, result);
            }
        }
        Node::Text(_) => {}
    }
}
