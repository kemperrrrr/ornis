use boa_engine::{
    Context, JsValue, Source,
    js_string,
    native_function::NativeFunction,
    object::builtins::JsArray,
    property::PropertyKey,
};

use crate::components::EcsBridge;
use crate::dom::{Element, Node};

const INIT_SCRIPT: &str = r##"
(function() {
var ElementPrototype = {
    appendChild: function(child) {
        if (child.parentNode) {
            var idx = child.parentNode.children.indexOf(child);
            if (idx >= 0) child.parentNode.children.splice(idx, 1);
        }
        child.parentNode = this;
        this.children.push(child);
        return child;
    },
    removeChild: function(child) {
        var idx = this.children.indexOf(child);
        if (idx >= 0) {
            this.children.splice(idx, 1);
            child.parentNode = null;
            return child;
        }
        throw new Error("NotFoundError");
    },
    insertBefore: function(newChild, refChild) {
        if (refChild === null || refChild === undefined) {
            return this.appendChild(newChild);
        }
        var idx = this.children.indexOf(refChild);
        if (idx < 0) throw new Error("NotFoundError");
        if (newChild.parentNode) {
            var oldIdx = newChild.parentNode.children.indexOf(newChild);
            if (oldIdx >= 0) newChild.parentNode.children.splice(oldIdx, 1);
        }
        newChild.parentNode = this;
        this.children.splice(idx, 0, newChild);
        return newChild;
    },
    setAttribute: function(name, value) {
        this.attributes[name] = String(value);
        if (name === "class") { this.className = String(value); }
        if (name === "id") { this.id = String(value); }
    },
    getAttribute: function(name) {
        return this.attributes[name] !== undefined ? this.attributes[name] : null;
    },
    hasAttribute: function(name) {
        return this.attributes[name] !== undefined;
    },
    removeAttribute: function(name) {
        delete this.attributes[name];
    },
    addEventListener: function(type, listener) {
        if (!this.listeners[type]) { this.listeners[type] = []; }
        this.listeners[type].push(listener);
    },
    removeEventListener: function(type, listener) {
        var arr = this.listeners[type];
        if (arr) {
            var idx = arr.indexOf(listener);
            if (idx >= 0) arr.splice(idx, 1);
        }
    },
    dispatchEvent: function(event) {
        event.target = this;
        var arr = this.listeners[event.type];
        if (arr) {
            for (var i = 0; i < arr.length; i++) {
                arr[i].call(this, event);
            }
        }
    },
    getElementsByTagName: function(tag) {
        var result = [];
        function walk(node) {
            if (node.tagName === tag) result.push(node);
            for (var i = 0; i < node.children.length; i++) {
                if (node.children[i].tagName) walk(node.children[i]);
            }
        }
        walk(this);
        return result;
    },
    querySelector: function(sel) {
        return this.querySelectorAll(sel)[0] || null;
    },
    querySelectorAll: function(sel) {
        var results = [];
        function walk(node) {
            if (matchSimple(node, sel)) results.push(node);
            for (var i = 0; i < node.children.length; i++) {
                if (node.children[i].tagName) walk(node.children[i]);
            }
        }
        function matchSimple(el, s) {
            s = s.trim();
            if (s === "*") return true;
            if (s.charAt(0) === ".") return (" " + el.className + " ").indexOf(" " + s.slice(1) + " ") >= 0;
            if (s.charAt(0) === "#") return el.id === s.slice(1);
            return el.tagName === s;
        }
        walk(this);
        return results;
    }
};

function makeClassList(el) {
    return {
        add: function(name) {
            var parts = (el.className || "").split(/\s+/);
            if (parts.indexOf(name) < 0) {
                parts.push(name);
                el.className = parts.join(" ").trim();
                el.setAttribute("class", el.className);
            }
        },
        remove: function(name) {
            var parts = (el.className || "").split(/\s+/);
            var idx = parts.indexOf(name);
            if (idx >= 0) {
                parts.splice(idx, 1);
                el.className = parts.join(" ").trim();
                el.setAttribute("class", el.className);
            }
        },
        contains: function(name) {
            return (el.className || "").split(/\s+/).indexOf(name) >= 0;
        },
        toggle: function(name) {
            if (this.contains(name)) { this.remove(name); return false; }
            else { this.add(name); return true; }
        }
    };
}

function makeStyle(el) {
    var style = {
        setProperty: function(name, value) {
            style[name] = String(value);
            syncStyle();
        },
        removeProperty: function(name) {
            delete style[name];
            syncStyle();
        },
        getPropertyValue: function(name) {
            return style[name] !== undefined ? style[name] : "";
        }
    };
    function syncStyle() {
        var parts = [];
        for (var k in style) {
            if (style.hasOwnProperty(k) && k !== "setProperty" && k !== "removeProperty" && k !== "getPropertyValue") {
                var cssName = k.replace(/([A-Z])/g, "-$1").toLowerCase();
                parts.push(cssName + ": " + style[k]);
            }
        }
        el.setAttribute("style", parts.join("; "));
    }
    return style;
}

function createElement(tag) {
    var el = Object.create(ElementPrototype);
    el.tagName = tag.toLowerCase();
    el.children = [];
    el.attributes = {};
    el.textContent = null;
    el.id = "";
    el.className = "";
    el.parentNode = null;
    el.listeners = {};
    el.classList = makeClassList(el);
    el.style = makeStyle(el);
    return el;
}

function createTextNode(text) {
    var el = Object.create(ElementPrototype);
    el.tagName = "#text";
    el.children = [];
    el.attributes = {};
    el.textContent = String(text);
    el.id = "";
    el.className = "";
    el.parentNode = null;
    el.listeners = {};
    el.classList = makeClassList(el);
    return el;
}

var _doc = {
    body: null,
    head: null,
    documentElement: null,
    createElement: createElement,
    createTextNode: createTextNode,
    getElementById: function(id) {
        function walk(node) {
            if (node.id === id) return node;
            for (var i = 0; i < node.children.length; i++) {
                var found = walk(node.children[i]);
                if (found) return found;
            }
            return null;
        }
        return walk(this.body);
    },
    createEvent: function(type) {
        return { type: type, target: null };
    }
};

var html = _doc.createElement("html");
var head = _doc.createElement("head");
var body = _doc.createElement("body");
html.children.push(head);
html.children.push(body);
head.parentNode = html;
body.parentNode = html;
_doc.documentElement = html;
_doc.head = head;
_doc.body = body;
document = _doc;

var _console = {
    log: function() {
        var args = Array.prototype.join.call(arguments, " ");
        console_log(args);
    },
    error: function() {
        var args = Array.prototype.join.call(arguments, " ");
        console_log("[ERROR] " + args);
    },
    warn: function() {
        var args = Array.prototype.join.call(arguments, " ");
        console_log("[WARN] " + args);
    }
};
console = _console;
})();
"##;

pub struct JsRuntime {
    ctx: Context,
    pub bridge: EcsBridge,
}

impl JsRuntime {
    pub fn new(bridge: EcsBridge) -> Self {
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

        bridge.register_js_functions(&mut ctx);

        let _ = ctx.eval(Source::from_bytes(INIT_SCRIPT));

        JsRuntime { ctx, bridge }
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

        // Extract id from JS object property
        if let Ok(id_val) = obj.get(js_string!("id"), &mut self.ctx) {
            if let Ok(id_str) = id_val.to_string(&mut self.ctx) {
                let s = id_str.to_std_string_escaped();
                if !s.is_empty() {
                    el.attrs.insert("id".into(), s);
                }
            }
        }
        // Extract className and sync as "class" attribute
        if let Ok(cls_val) = obj.get(js_string!("className"), &mut self.ctx) {
            if let Ok(cls_str) = cls_val.to_string(&mut self.ctx) {
                let s = cls_str.to_std_string_escaped();
                if !s.is_empty() && !el.attrs.contains_key("class") {
                    el.attrs.insert("class".into(), s);
                }
            }
        }

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

    fn test_bridge() -> EcsBridge {
        EcsBridge::new()
    }

    #[test]
    fn test_js_document_create_element() {
        let mut js = JsRuntime::new(test_bridge());
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
        let mut js = JsRuntime::new(test_bridge());
        let node = js.document_node();
        let body = find_by_tag(&node, "body");
        assert_eq!(body.len(), 1);
        assert!(body[0].children.is_empty());
    }

    #[test]
    fn test_js_create_text_node() {
        let mut js = JsRuntime::new(test_bridge());
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
        let mut js = JsRuntime::new(test_bridge());
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
        assert_eq!(lis[0].get_attr("textContent"), None);
    }

    #[test]
    fn test_js_classlist() {
        let mut js = JsRuntime::new(test_bridge());
        js.eval(r##"
            var el = document.createElement("div");
            el.classList.add("foo");
            el.classList.add("bar");
            document.body.appendChild(el);
        "##).unwrap();

        let node = js.document_node();
        if let crate::dom::Node::Element(body) = &node {
            assert_eq!(body.children.len(), 1);
            if let crate::dom::Node::Element(div) = &body.children[0] {
                let cls = div.get_attr("class").unwrap_or("");
                assert!(cls.contains("foo"));
                assert!(cls.contains("bar"));
            } else { panic!("expected element"); }
        } else { panic!("expected body"); }
    }

    #[test]
    fn test_js_remove_child() {
        let mut js = JsRuntime::new(test_bridge());
        js.eval(r##"
            var parent = document.createElement("div");
            var child = document.createElement("span");
            parent.appendChild(child);
            parent.removeChild(child);
            document.body.appendChild(parent);
        "##).unwrap();

        let node = js.document_node();
        let divs = find_by_tag(&node, "div");
        assert_eq!(divs.len(), 1);
        assert!(divs[0].children.is_empty());
    }

    #[test]
    fn test_js_insert_before() {
        let mut js = JsRuntime::new(test_bridge());
        js.eval(r##"
            var ul = document.createElement("ul");
            var li1 = document.createElement("li");
            li1.textContent = "first";
            var li2 = document.createElement("li");
            li2.textContent = "second";
            ul.appendChild(li2);
            ul.insertBefore(li1, li2);
            document.body.appendChild(ul);
        "##).unwrap();

        let node = js.document_node();
        let uls = find_by_tag(&node, "ul");
        assert_eq!(uls.len(), 1);
        assert_eq!(uls[0].children.len(), 2);
    }

    #[test]
    fn test_js_get_element_by_id() {
        let mut js = JsRuntime::new(test_bridge());
        js.eval(r##"
            var el = document.createElement("div");
            el.id = "my-id";
            el.setAttribute("class", "test");
            document.body.appendChild(el);
            var found = document.getElementById("my-id");
        "##).unwrap();

        let node = js.document_node();
        let divs = find_by_tag(&node, "div");
        assert_eq!(divs.len(), 1);
        assert_eq!(divs[0].get_attr("id"), Some("my-id"));
    }

    #[test]
    fn test_js_style_set_property() {
        let mut js = JsRuntime::new(test_bridge());
        js.eval(r##"
            var el = document.createElement("div");
            el.style.setProperty("color", "red");
            el.style.setProperty("font-size", "16px");
            document.body.appendChild(el);
        "##).unwrap();

        let node = js.document_node();
        let divs = find_by_tag(&node, "div");
        assert_eq!(divs.len(), 1);
        let style = divs[0].get_attr("style").unwrap_or("");
        assert!(style.contains("color: red"), "style should contain color: red, got: {style}");
        assert!(style.contains("font-size: 16px"), "style should contain font-size: 16px, got: {style}");
    }

    #[test]
    fn test_js_document_structure() {
        let mut js = JsRuntime::new(test_bridge());
        js.eval(r##""##).unwrap();

        let node = js.document_node();
        let bodies = find_by_tag(&node, "body");
        assert_eq!(bodies.len(), 1);
        assert!(bodies[0].children.is_empty());
    }

    #[test]
    fn test_js_console() {
        let mut js = JsRuntime::new(test_bridge());
        js.eval(r##"console.log("hello from js");"##).unwrap();
        // just check it doesn't crash
    }
}
