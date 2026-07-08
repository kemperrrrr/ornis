use boa_engine::{
    Context, JsValue, Source,
    js_string,
    native_function::NativeFunction,
    object::builtins::JsArray,
    property::PropertyKey,
};

use crate::dom::{Element, Node};

const INIT_SCRIPT: &str = r##"
(function() {
    function setupElement(el) {
        el.appendChild = function(child) {
            this.children.push(child);
            return child;
        };
        el.setAttribute = function(name, value) {
            this.attributes[name] = value;
        };
        el.getAttribute = function(name) {
            return this.attributes[name] !== undefined ? this.attributes[name] : null;
        };
    }

    var body = { tagName: "body", children: [], attributes: {}, style: {}, textContent: null };
    setupElement(body);

    document = {
        body: body,
        createElement: function(tag) {
            var el = { tagName: tag, children: [], attributes: {}, style: {}, textContent: null };
            setupElement(el);
            return el;
        },
        createTextNode: function(text) {
            return { tagName: "#text", children: [], attributes: {}, style: {}, textContent: String(text) };
        }
    };
})();
"##;

pub struct JsRuntime {
    ctx: Context,
}

impl JsRuntime {
    pub fn new() -> Self {
        let mut ctx = Context::default();

        ctx.register_global_callable(
            js_string!("console_log"),
            1,
            NativeFunction::from_fn_ptr(|_, args, _| {
                if let Some(val) = args.first() {
                    println!("[JS] {}", val.display());
                }
                Ok(JsValue::undefined())
            }),
        ).expect("register console_log");

        let _ = ctx.eval(Source::from_bytes(INIT_SCRIPT));

        JsRuntime { ctx }
    }

    pub fn eval(&mut self, code: &str) -> Result<(), String> {
        self.ctx.eval(Source::from_bytes(code))
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }

    pub fn document_node(&mut self) -> Node {
        let global = self.ctx.global_object();
        let doc_val = global.get(js_string!("document"), &mut self.ctx)
            .unwrap_or_else(|_| JsValue::undefined());

        if !doc_val.is_object() {
            return Node::Element(Element::new("body"));
        }

        let body_val = doc_val.as_object().unwrap()
            .get(js_string!("body"), &mut self.ctx)
            .unwrap_or(JsValue::undefined());

        self.js_to_node(&body_val)
    }

    fn js_to_node(&mut self, val: &JsValue) -> Node {
        let Some(obj) = val.as_object() else {
            return Node::Text(String::new());
        };

        let tag_val = obj.get(js_string!("tagName"), &mut self.ctx)
            .unwrap_or(JsValue::undefined());
        let tag = if tag_val.is_string() {
            tag_val.to_string(&mut self.ctx).unwrap().to_std_string_escaped()
        } else {
            return Node::Text(String::new());
        };

        if tag == "#text" {
            let text_val = obj.get(js_string!("textContent"), &mut self.ctx)
                .unwrap_or(JsValue::null());
            let text = if text_val.is_null() || text_val.is_undefined() {
                String::new()
            } else {
                text_val.to_string(&mut self.ctx).unwrap().to_std_string_escaped()
            };
            return Node::Text(text);
        }

        let mut el = Element::new(&tag);

        if let Ok(attrs_obj) = obj.get(js_string!("attributes"), &mut self.ctx) {
            if let Some(attrs_obj) = attrs_obj.as_object() {
                if let Ok(keys) = attrs_obj.own_property_keys(&mut self.ctx) {
                    for key in keys {
                        let name = match &key {
                            PropertyKey::String(s) => s.to_std_string_escaped(),
                            PropertyKey::Index(i) => i.get().to_string(),
                            PropertyKey::Symbol(_) => continue,
                        };
                        if let Ok(val) = attrs_obj.get(key, &mut self.ctx) {
                            if let Ok(v) = val.to_string(&mut self.ctx) {
                                el.attrs.insert(name, v.to_std_string_escaped());
                            }
                        }
                    }
                }
            }
        }

        if let Ok(children_val) = obj.get(js_string!("children"), &mut self.ctx) {
            if let Some(arr_obj) = children_val.as_object() {
                if let Ok(arr) = JsArray::from_object(arr_obj.clone()) {
                    if let Ok(len) = arr.length(&mut self.ctx) {
                        for i in 0..len {
                            if let Ok(child) = arr.get(i, &mut self.ctx) {
                                el.children.push(self.js_to_node(&child));
                            }
                        }
                    }
                }
            }
        }

        Node::Element(el)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dom::find_by_tag;

    #[test]
    fn test_js_document_create_element() {
        let mut js = JsRuntime::new();
        js.eval(r##"
            var el = document.createElement("div");
            el.setAttribute("class", "my-class");
            el.textContent = "Hello World";
            document.body.appendChild(el);
        "##).unwrap();

        let node = js.document_node();
        let elements = find_by_tag(&node, "div");
        assert!(!elements.is_empty(), "should have found a div");
        assert_eq!(elements[0].get_attr("class"), Some("my-class"));
    }

    #[test]
    fn test_js_document_empty() {
        let mut js = JsRuntime::new();
        let node = js.document_node();
        let body = find_by_tag(&node, "body");
        assert_eq!(body.len(), 1);
        assert!(body[0].children.is_empty());
    }

    #[test]
    fn test_js_create_text_node() {
        let mut js = JsRuntime::new();
        js.eval(r##"
            var text = document.createTextNode("hello");
            document.body.appendChild(text);
        "##).unwrap();

        let node = js.document_node();
        // body should have one text child
        if let crate::dom::Node::Element(body) = &node {
            assert_eq!(body.children.len(), 1);
            if let crate::dom::Node::Text(t) = &body.children[0] {
                assert_eq!(t, "hello");
            } else {
                panic!("expected text node");
            }
        } else {
            panic!("expected body element");
        }
    }

    #[test]
    fn test_js_nested_elements() {
        let mut js = JsRuntime::new();
        js.eval(r##"
            var parent = document.createElement("ul");
            var child = document.createElement("li");
            child.textContent = "item 1";
            parent.appendChild(child);
            document.body.appendChild(parent);
        "##).unwrap();

        let node = js.document_node();
        let uls = find_by_tag(&node, "ul");
        assert_eq!(uls.len(), 1);
        let lis = find_by_tag(&node, "li");
        assert_eq!(lis.len(), 1);
        assert_eq!(lis[0].get_attr("textContent"), None); // textContent is not attr
    }
}
