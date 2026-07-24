use std::collections::HashMap;

use lightningcss::rules::CssRule;
use lightningcss::selector::{
    Combinator as LcCombinator, Component, Direction, PseudoClass as LcPseudoClass,
    PseudoElement as LcPseudoElement, Selector as LcSelector, SelectorList as LcSelectorList,
};
use lightningcss::stylesheet::{ParserOptions, StyleSheet};
use lightningcss::traits::ToCss;
use parcel_selectors::attr::{AttrSelectorOperator, ParsedAttrSelectorOperation};
use parcel_selectors::parser::NthType;
use vello::peniko::Color;

use crate::dom::Element;

#[derive(Debug, Clone)]
pub struct Stylesheet {
    pub rules: Vec<Rule>,
    pub custom_properties: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub selectors: Vec<Selector>,
    pub declarations: HashMap<String, String>,
    pub important: HashMap<String, String>,
    pub specificity: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Combinator {
    Descendant,
    Child,
    NextSibling,
    SubsequentSibling,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttrOp {
    Exists,
    Equals,
    Includes,
    DashMatch,
    Prefix,
    Suffix,
    Substring,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimpleSelector {
    Universal,
    Type(String),
    Class(String),
    Id(String),
    Attr {
        name: String,
        value: Option<String>,
        op: AttrOp,
    },
    Root,
    Empty,
    NthChild {
        a: i32,
        b: i32,
    },
    NthLastChild {
        a: i32,
        b: i32,
    },
    NthOfType {
        a: i32,
        b: i32,
    },
    NthLastOfType {
        a: i32,
        b: i32,
    },
    FirstChild,
    LastChild,
    FirstOfType,
    LastOfType,
    OnlyChild,
    OnlyOfType,
    PseudoClass {
        name: String,
        args: Option<String>,
    },
    PseudoElement {
        name: String,
    },
    Not(Vec<Selector>),
    Where(Vec<Selector>),
    Is(Vec<Selector>),
    Has(Vec<Selector>),
    Unsupported,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompoundPart {
    pub comb: Combinator,
    pub parts: Vec<SimpleSelector>,
}

pub type Selector = Vec<CompoundPart>;

fn selector_specificity(sel: &Selector) -> u32 {
    sel.iter()
        .flat_map(|cp| cp.parts.iter())
        .map(specificity_of)
        .fold(0u32, |a, b| a.saturating_add(b))
}

fn specificity_of(sel: &SimpleSelector) -> u32 {
    match sel {
        SimpleSelector::Id(_) => 0x01_00_00,
        SimpleSelector::Class(_)
        | SimpleSelector::Attr { .. }
        | SimpleSelector::PseudoClass { .. }
        | SimpleSelector::Root
        | SimpleSelector::Empty
        | SimpleSelector::NthChild { .. }
        | SimpleSelector::NthLastChild { .. }
        | SimpleSelector::NthOfType { .. }
        | SimpleSelector::NthLastOfType { .. }
        | SimpleSelector::FirstChild
        | SimpleSelector::LastChild
        | SimpleSelector::FirstOfType
        | SimpleSelector::LastOfType
        | SimpleSelector::OnlyChild
        | SimpleSelector::OnlyOfType => 0x00_01_00,
        SimpleSelector::Type(_) | SimpleSelector::PseudoElement { .. } => 0x00_00_01,
        SimpleSelector::Universal | SimpleSelector::Unsupported => 0,
        SimpleSelector::Not(selectors)
        | SimpleSelector::Is(selectors)
        | SimpleSelector::Has(selectors) => selectors
            .iter()
            .map(|s| selector_specificity(s))
            .max()
            .unwrap_or(0),
        SimpleSelector::Where(_) => 0,
    }
}

fn convert_attr_op(op: AttrSelectorOperator) -> AttrOp {
    match op {
        AttrSelectorOperator::Equal => AttrOp::Equals,
        AttrSelectorOperator::Includes => AttrOp::Includes,
        AttrSelectorOperator::DashMatch => AttrOp::DashMatch,
        AttrSelectorOperator::Prefix => AttrOp::Prefix,
        AttrSelectorOperator::Substring => AttrOp::Substring,
        AttrSelectorOperator::Suffix => AttrOp::Suffix,
    }
}

fn convert_combinator(c: LcCombinator) -> Option<Combinator> {
    match c {
        LcCombinator::Descendant => Some(Combinator::Descendant),
        LcCombinator::Child => Some(Combinator::Child),
        LcCombinator::NextSibling => Some(Combinator::NextSibling),
        LcCombinator::LaterSibling => Some(Combinator::SubsequentSibling),
        _ => None,
    }
}

fn convert_component<'i>(c: &Component<'i>) -> Option<SimpleSelector> {
    match c {
        Component::ExplicitUniversalType | Component::ExplicitAnyNamespace => {
            Some(SimpleSelector::Universal)
        }
        Component::LocalName(local) => Some(SimpleSelector::Type(local.name.0.to_string())),
        Component::Class(ident) => Some(SimpleSelector::Class(ident.0.to_string())),
        Component::ID(ident) => Some(SimpleSelector::Id(ident.0.to_string())),
        Component::AttributeInNoNamespaceExists { local_name, .. } => Some(SimpleSelector::Attr {
            name: local_name.0.to_string(),
            value: None,
            op: AttrOp::Exists,
        }),
        Component::AttributeInNoNamespace {
            local_name,
            operator,
            value,
            ..
        } => Some(SimpleSelector::Attr {
            name: local_name.0.to_string(),
            value: Some(value.0.to_string()),
            op: convert_attr_op(*operator),
        }),
        Component::AttributeOther(attr) => {
            let (op, value) = match &attr.operation {
                ParsedAttrSelectorOperation::Exists => (AttrOp::Exists, None),
                ParsedAttrSelectorOperation::WithValue {
                    operator, expected_value, ..
                } => (convert_attr_op(*operator), Some(expected_value.0.to_string())),
            };
            Some(SimpleSelector::Attr {
                name: attr.local_name.0.to_string(),
                value,
                op,
            })
        }
        Component::Root => Some(SimpleSelector::Root),
        Component::Empty => Some(SimpleSelector::Empty),
        Component::Nth(data) => {
            match data.ty {
                NthType::Child => {
                    if data.a == 0 && data.b == 1 {
                        Some(SimpleSelector::FirstChild)
                    } else {
                        Some(SimpleSelector::NthChild {
                            a: data.a,
                            b: data.b,
                        })
                    }
                }
                NthType::LastChild => {
                    if data.a == 0 && data.b == 1 {
                        Some(SimpleSelector::LastChild)
                    } else {
                        Some(SimpleSelector::NthLastChild {
                            a: data.a,
                            b: data.b,
                        })
                    }
                }
                NthType::OnlyChild => Some(SimpleSelector::OnlyChild),
                NthType::OfType => {
                    if data.a == 0 && data.b == 1 {
                        Some(SimpleSelector::FirstOfType)
                    } else {
                        Some(SimpleSelector::NthOfType {
                            a: data.a,
                            b: data.b,
                        })
                    }
                }
                NthType::LastOfType => {
                    if data.a == 0 && data.b == 1 {
                        Some(SimpleSelector::LastOfType)
                    } else {
                        Some(SimpleSelector::NthLastOfType {
                            a: data.a,
                            b: data.b,
                        })
                    }
                }
                NthType::OnlyOfType => Some(SimpleSelector::OnlyOfType),
                _ => None,
            }
        }
        Component::NonTSPseudoClass(pc) => convert_pseudo_class(pc),
        Component::PseudoElement(pe) => convert_pseudo_element(pe),
        Component::Negation(selectors) => {
            Some(SimpleSelector::Not(convert_selector_slice(selectors)))
        }
        Component::Is(selectors) => Some(SimpleSelector::Is(convert_selector_slice(selectors))),
        Component::Where(selectors) => {
            Some(SimpleSelector::Where(convert_selector_slice(selectors)))
        }
        Component::Has(selectors) => {
            Some(SimpleSelector::Has(convert_selector_slice(selectors)))
        }
        _ => None,
    }
}

fn pseudo_class_name_args<'i>(pc: &LcPseudoClass<'i>) -> (Option<String>, Option<String>) {
    match pc {
        LcPseudoClass::Hover => (Some("hover".into()), None),
        LcPseudoClass::Active => (Some("active".into()), None),
        LcPseudoClass::Focus => (Some("focus".into()), None),
        LcPseudoClass::FocusVisible => (Some("focus-visible".into()), None),
        LcPseudoClass::FocusWithin => (Some("focus-within".into()), None),
        LcPseudoClass::AnyLink(_) => (Some("any-link".into()), None),
        LcPseudoClass::Link => (Some("link".into()), None),
        LcPseudoClass::Visited => (Some("visited".into()), None),
        LcPseudoClass::Target => (Some("target".into()), None),
        LcPseudoClass::Enabled => (Some("enabled".into()), None),
        LcPseudoClass::Disabled => (Some("disabled".into()), None),
        LcPseudoClass::Checked => (Some("checked".into()), None),
        LcPseudoClass::Indeterminate => (Some("indeterminate".into()), None),
        LcPseudoClass::Default => (Some("default".into()), None),
        LcPseudoClass::Valid => (Some("valid".into()), None),
        LcPseudoClass::Invalid => (Some("invalid".into()), None),
        LcPseudoClass::Required => (Some("required".into()), None),
        LcPseudoClass::Optional => (Some("optional".into()), None),
        LcPseudoClass::ReadOnly(_) => (Some("read-only".into()), None),
        LcPseudoClass::ReadWrite(_) => (Some("read-write".into()), None),
        LcPseudoClass::PlaceholderShown(_) => (Some("placeholder-shown".into()), None),
        LcPseudoClass::InRange => (Some("in-range".into()), None),
        LcPseudoClass::OutOfRange => (Some("out-of-range".into()), None),
        LcPseudoClass::Open => (Some("open".into()), None),
        LcPseudoClass::Closed => (Some("closed".into()), None),
        LcPseudoClass::Modal => (Some("modal".into()), None),
        LcPseudoClass::Fullscreen(_) => (Some("fullscreen".into()), None),
        LcPseudoClass::PictureInPicture => (Some("picture-in-picture".into()), None),
        LcPseudoClass::PopoverOpen => (Some("popover-open".into()), None),
        LcPseudoClass::Defined => (Some("defined".into()), None),
        LcPseudoClass::Playing => (Some("playing".into()), None),
        LcPseudoClass::Paused => (Some("paused".into()), None),
        LcPseudoClass::Seeking => (Some("seeking".into()), None),
        LcPseudoClass::Buffering => (Some("buffering".into()), None),
        LcPseudoClass::Stalled => (Some("stalled".into()), None),
        LcPseudoClass::Muted => (Some("muted".into()), None),
        LcPseudoClass::VolumeLocked => (Some("volume-locked".into()), None),
        LcPseudoClass::Lang { languages } => {
            (Some("lang".into()), Some(languages.join(",")))
        }
        LcPseudoClass::Dir { direction } => {
            let d = match direction {
                Direction::Ltr => "ltr",
                Direction::Rtl => "rtl",
            };
            (Some("dir".into()), Some(d.into()))
        }
        LcPseudoClass::Local { selector } => {
            let s = selector
                .to_css_string(lightningcss::stylesheet::PrinterOptions::default())
                .unwrap_or_default();
            (Some("local".into()), Some(s))
        }
        LcPseudoClass::Global { selector } => {
            let s = selector
                .to_css_string(lightningcss::stylesheet::PrinterOptions::default())
                .unwrap_or_default();
            (Some("global".into()), Some(s))
        }
        LcPseudoClass::Custom { name } => (Some(name.to_string()), None),
        LcPseudoClass::CustomFunction { name, .. } => (Some(name.to_string()), None),
        _ => (None, None),
    }
}

fn convert_pseudo_class<'i>(pc: &LcPseudoClass<'i>) -> Option<SimpleSelector> {
    let (name, args) = pseudo_class_name_args(pc);
    let name = name?;
    Some(SimpleSelector::PseudoClass { name, args })
}

fn convert_pseudo_element<'i>(pe: &LcPseudoElement<'i>) -> Option<SimpleSelector> {
    let name = match pe {
        LcPseudoElement::After => "after",
        LcPseudoElement::Before => "before",
        LcPseudoElement::FirstLine => "first-line",
        LcPseudoElement::FirstLetter => "first-letter",
        LcPseudoElement::Selection(_) => "selection",
        LcPseudoElement::Placeholder(_) => "placeholder",
        LcPseudoElement::Marker => "marker",
        LcPseudoElement::Backdrop(_) => "backdrop",
        LcPseudoElement::FileSelectorButton(_) => "file-selector-button",
        LcPseudoElement::DetailsContent => "details-content",
        LcPseudoElement::TargetText => "target-text",
        LcPseudoElement::Cue => "cue",
        LcPseudoElement::CueRegion => "cue-region",
        LcPseudoElement::PickerIcon => "picker-icon",
        LcPseudoElement::Checkmark => "checkmark",
        LcPseudoElement::GrammarError => "grammar-error",
        LcPseudoElement::SpellingError => "spelling-error",
        LcPseudoElement::ViewTransition => "view-transition",
        LcPseudoElement::Custom { name } => {
            return Some(SimpleSelector::PseudoElement {
                name: name.to_string(),
            })
        }
        _ => return None,
    };
    Some(SimpleSelector::PseudoElement {
        name: name.into(),
    })
}

fn convert_selector_slice<'i>(selectors: &[LcSelector<'i>]) -> Vec<Selector> {
    selectors.iter().map(convert_one_selector).collect()
}

fn convert_selector_list<'i>(list: &LcSelectorList<'i>) -> Vec<Selector> {
    list.0.iter().map(convert_one_selector).collect()
}

fn convert_one_selector<'i>(sel: &LcSelector<'i>) -> Selector {
    let mut compounds: Vec<Vec<SimpleSelector>> = Vec::new();
    let mut combinators: Vec<Combinator> = Vec::new();
    let mut current: Vec<SimpleSelector> = Vec::new();

    for comp in sel.iter_raw_match_order().rev() {
        if comp.is_combinator() {
            if let Some(c) = comp.as_combinator() {
                if let Some(our_c) = convert_combinator(c) {
                    if !current.is_empty() {
                        compounds.push(std::mem::take(&mut current));
                    }
                    combinators.push(our_c);
                }
            }
        } else if let Some(simple) = convert_component(comp) {
            current.push(simple);
        }
    }
    if !current.is_empty() {
        compounds.push(std::mem::take(&mut current));
    }

    let mut result = Vec::new();
    for i in 0..compounds.len() {
        let comb = if i > 0 {
            combinators[i - 1].clone()
        } else {
            Combinator::Descendant
        };
        result.push(CompoundPart {
            comb,
            parts: std::mem::take(&mut compounds[i]),
        });
    }
    result
}

fn nth_matches(one_indexed: usize, a: i32, b: i32) -> bool {
    if a == 0 {
        return one_indexed == b as usize;
    }
    if (one_indexed as i32) < b {
        return false;
    }
    let n = (one_indexed as i32 - b) / a;
    n >= 0 && (one_indexed as i32 - b) % a == 0
}

fn match_attr(element: &Element, name: &str, expected: Option<&str>, op: &AttrOp) -> bool {
    let actual = match element.attrs.get(name) {
        Some(v) => v.as_str(),
        None => return false,
    };
    match op {
        AttrOp::Exists => true,
        AttrOp::Equals => match expected {
            Some(e) => actual == e,
            None => false,
        },
        AttrOp::Includes => match expected {
            Some(e) => actual.split_whitespace().any(|part| part == e),
            None => false,
        },
        AttrOp::DashMatch => match expected {
            Some(e) => actual == e || actual.starts_with(&format!("{e}-")),
            None => false,
        },
        AttrOp::Prefix => match expected {
            Some(e) => actual.starts_with(e),
            None => false,
        },
        AttrOp::Suffix => match expected {
            Some(e) => actual.ends_with(e),
            None => false,
        },
        AttrOp::Substring => match expected {
            Some(e) => actual.contains(e),
            None => false,
        },
    }
}

fn element_sibling_index(element: &Element, parent: &Element) -> Option<usize> {
    parent.children.iter().position(|child| match child {
        crate::dom::Node::Element(el) => {
            std::ptr::eq(el as *const Element, element as *const Element)
        }
        _ => false,
    })
}

fn element_sibling_count(parent: &Element) -> usize {
    parent
        .children
        .iter()
        .filter(|c| matches!(c, crate::dom::Node::Element(_)))
        .count()
}

fn prev_sibling<'a>(element: &Element, parent: &'a Element) -> Option<&'a Element> {
    let idx = element_sibling_index(element, parent)?;
    for i in (0..idx).rev() {
        if let crate::dom::Node::Element(el) = &parent.children[i] {
            return Some(el);
        }
    }
    None
}

fn match_simple(sel: &SimpleSelector, element: &Element, ancestors: &[&Element]) -> bool {
    match sel {
        SimpleSelector::Universal => true,
        SimpleSelector::Type(t) => element.tag == *t,
        SimpleSelector::Class(c) => element.classes().iter().any(|cl| *cl == c.as_str()),
        SimpleSelector::Id(id) => element.id() == Some(id.as_str()),
        SimpleSelector::Attr { name, value, op } => match_attr(element, name, value.as_deref(), op),
        SimpleSelector::Root => element.tag == "html",
        SimpleSelector::Empty => element.children.is_empty(),

        SimpleSelector::FirstChild => ancestors
            .last()
            .map_or(false, |parent| element_sibling_index(element, parent) == Some(0)),

        SimpleSelector::LastChild => ancestors.last().map_or(false, |parent| {
            let count = element_sibling_count(parent);
            element_sibling_index(element, parent) == Some(count - 1)
        }),

        SimpleSelector::OnlyChild => ancestors
            .last()
            .map_or(false, |parent| element_sibling_count(parent) == 1),

        SimpleSelector::FirstOfType => ancestors.last().map_or(false, |parent| {
            let idx = match element_sibling_index(element, parent) {
                Some(i) => i,
                None => return false,
            };
            parent
                .children
                .iter()
                .take(idx)
                .filter(|c| matches!(c, crate::dom::Node::Element(el) if el.tag == element.tag))
                .count()
                == 0
        }),

        SimpleSelector::LastOfType => ancestors.last().map_or(false, |parent| {
            let idx = match element_sibling_index(element, parent) {
                Some(i) => i,
                None => return false,
            };
            let total = parent
                .children
                .iter()
                .filter(|c| matches!(c, crate::dom::Node::Element(el) if el.tag == element.tag))
                .count();
            let before = parent
                .children
                .iter()
                .take(idx)
                .filter(|c| matches!(c, crate::dom::Node::Element(el) if el.tag == element.tag))
                .count();
            before == total - 1
        }),

        SimpleSelector::OnlyOfType => ancestors.last().map_or(false, |parent| {
            parent
                .children
                .iter()
                .filter(|c| matches!(c, crate::dom::Node::Element(el) if el.tag == element.tag))
                .count()
                == 1
        }),

        SimpleSelector::NthChild { a, b } => ancestors.last().map_or(false, |parent| {
            let idx = match element_sibling_index(element, parent) {
                Some(i) => i + 1,
                None => return false,
            };
            nth_matches(idx, *a, *b)
        }),

        SimpleSelector::NthLastChild { a, b } => ancestors.last().map_or(false, |parent| {
            let count = element_sibling_count(parent);
            let idx = match element_sibling_index(element, parent) {
                Some(i) => i,
                None => return false,
            };
            nth_matches(count - idx, *a, *b)
        }),

        SimpleSelector::NthOfType { a, b } => ancestors.last().map_or(false, |parent| {
            let idx = match element_sibling_index(element, parent) {
                Some(i) => i,
                None => return false,
            };
            let pos = parent
                .children
                .iter()
                .take(idx)
                .filter(|c| matches!(c, crate::dom::Node::Element(el) if el.tag == element.tag))
                .count();
            nth_matches(pos + 1, *a, *b)
        }),

        SimpleSelector::NthLastOfType { a, b } => ancestors.last().map_or(false, |parent| {
            let idx = match element_sibling_index(element, parent) {
                Some(i) => i,
                None => return false,
            };
            let total = parent
                .children
                .iter()
                .filter(|c| matches!(c, crate::dom::Node::Element(el) if el.tag == element.tag))
                .count();
            let before = parent
                .children
                .iter()
                .take(idx)
                .filter(|c| matches!(c, crate::dom::Node::Element(el) if el.tag == element.tag))
                .count();
            nth_matches(total - before, *a, *b)
        }),

        SimpleSelector::PseudoClass { name, .. } => match name.as_str() {
            "link" => element.tag == "a",
            "enabled" => !element.attrs.contains_key("disabled"),
            "disabled" => element.attrs.contains_key("disabled"),
            "checked" => element.attrs.contains_key("checked"),
            "any-link" => element.tag == "a",
            "defined" => true,
            _ => false,
        },

        SimpleSelector::PseudoElement { .. } => false,
        SimpleSelector::Not(selectors) => {
            !selectors
                .iter()
                .any(|sel| matches_selector(sel, element, ancestors))
        }
        SimpleSelector::Is(selectors) | SimpleSelector::Where(selectors) => selectors
            .iter()
            .any(|sel| matches_selector(sel, element, ancestors)),
        SimpleSelector::Has(_) => false,
        SimpleSelector::Unsupported => false,
    }
}

fn match_compound(parts: &[SimpleSelector], element: &Element, ancestors: &[&Element]) -> bool {
    parts
        .iter()
        .all(|s| match_simple(s, element, ancestors))
}

fn matches_selector(sel: &Selector, element: &Element, ancestors: &[&Element]) -> bool {
    if sel.is_empty() {
        return false;
    }
    let target = &sel[sel.len() - 1];
    if !match_compound(&target.parts, element, ancestors) {
        return false;
    }
    let mut cursor = ancestors.len();
    for k in (0..sel.len() - 1).rev() {
        let comb = &sel[k + 1].comb;
        let parts = &sel[k].parts;
        match comb {
            Combinator::Child => {
                if cursor > 0 {
                    let parent = ancestors[cursor - 1];
                    if match_compound(parts, parent, ancestors) {
                        cursor -= 1;
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            Combinator::Descendant => {
                let mut found = false;
                for j in (0..cursor).rev() {
                    if match_compound(parts, ancestors[j], ancestors) {
                        cursor = j;
                        found = true;
                        break;
                    }
                }
                if !found {
                    return false;
                }
            }
            Combinator::NextSibling => {
                let parent = match ancestors.last() {
                    Some(p) => p,
                    None => return false,
                };
                let prev = match prev_sibling(element, parent) {
                    Some(el) => el,
                    None => return false,
                };
                if !match_compound(parts, prev, ancestors) {
                    return false;
                }
            }
            Combinator::SubsequentSibling => {
                let parent = match ancestors.last() {
                    Some(p) => p,
                    None => return false,
                };
                let idx = match element_sibling_index(element, parent) {
                    Some(i) => i,
                    None => return false,
                };
                let mut found = false;
                for i in (0..idx).rev() {
                    if let crate::dom::Node::Element(el) = &parent.children[i] {
                        if match_compound(parts, el, ancestors) {
                            found = true;
                            break;
                        }
                    }
                }
                if !found {
                    return false;
                }
            }
        }
    }
    true
}

impl Stylesheet {
    pub fn parse(css: &str) -> Result<Self, String> {
        let sheet = StyleSheet::parse(css, ParserOptions::default())
            .map_err(|e| format!("CSS parse error: {e}"))?;

        let mut rules = Vec::new();
        let mut custom_properties = HashMap::new();

        for rule in sheet.rules.0.iter() {
            match rule {
                CssRule::Style(style_rule) => {
                    let selectors = convert_selector_list(&style_rule.selectors);

                    let mut declarations = HashMap::new();
                    let mut important = HashMap::new();
                    for (prop, is_important) in style_rule.declarations.iter() {
                        let name = prop.property_id().name().to_string();
                        let mut value = String::new();
                        let mut printer = lightningcss::printer::Printer::new(
                            &mut value,
                            lightningcss::stylesheet::PrinterOptions::default(),
                        );
                        if prop.to_css(&mut printer, false).is_ok() {
                            let v = value
                                .strip_prefix(&format!("{name}:"))
                                .map(|s| s.strip_prefix(' ').unwrap_or(s).to_string())
                                .unwrap_or_else(|| value.clone());
                            if is_important {
                                important.insert(name.clone(), v.clone());
                            } else {
                                declarations.insert(name.clone(), v.clone());
                            }
                            if selectors.iter().any(|sel| {
                                sel.iter().any(|cp| {
                                    cp.parts.iter().any(|s| matches!(s, SimpleSelector::Root))
                                })
                            }) && name.starts_with("--")
                            {
                                custom_properties.insert(name, v);
                            }
                        }
                    }

                    let specificity = selectors
                        .iter()
                        .map(|s| selector_specificity(s))
                        .max()
                        .unwrap_or(0);
                    rules.push(Rule {
                        selectors,
                        declarations,
                        important,
                        specificity,
                    });
                }
                _ => {}
            }
        }

        Ok(Stylesheet {
            rules,
            custom_properties,
        })
    }

    pub fn match_element(
        &self,
        element: &Element,
        ancestors: &[&Element],
    ) -> HashMap<String, String> {
        let mut props = HashMap::new();
        let mut sorted: Vec<&Rule> = self.rules.iter().collect();
        sorted.sort_by_key(|r| r.specificity);
        for rule in &sorted {
            if rule
                .selectors
                .iter()
                .any(|sel| matches_selector(sel, element, ancestors))
            {
                for (k, v) in &rule.declarations {
                    props.insert(k.clone(), v.clone());
                }
            }
        }
        for rule in &sorted {
            if rule
                .selectors
                .iter()
                .any(|sel| matches_selector(sel, element, ancestors))
            {
                for (k, v) in &rule.important {
                    props.insert(k.clone(), v.clone());
                }
            }
        }
        props
    }
}

const MASTER_CSS: &str = r##"
html { display: block }
body { display: block; margin: 8px; overflow: hidden }
head, meta, title, link, style, script { display: none }

div { display: block }
span { display: inline }
a { display: inline; text-decoration: underline; color: -webkit-link; cursor: pointer }

p { display: block; margin-top: 1em; margin-bottom: 1em }

b, strong { display: inline; font-weight: bold }
i, em, cite { display: inline; font-style: italic }
s, strike, del { text-decoration: line-through }
u, ins { text-decoration: underline }
small { font-size: smaller }
sub { font-size: smaller; vertical-align: sub }
sup { font-size: smaller; vertical-align: super }

h1 { display: block; font-weight: bold; font-size: 2em; margin: 0.67em 0 }
h2 { display: block; font-weight: bold; font-size: 1.5em; margin: 0.83em 0 }
h3 { display: block; font-weight: bold; font-size: 1.17em; margin: 1em 0 }
h4 { display: block; font-weight: bold; margin: 1.33em 0 }
h5 { display: block; font-weight: bold; font-size: 0.83em; margin: 1.67em 0 }
h6 { display: block; font-weight: bold; font-size: 0.67em; margin: 2.33em 0 }

ul, menu, dir { display: block; list-style-type: disc; margin: 1em 0; padding-left: 40px }
ol { display: block; list-style-type: decimal; margin: 1em 0; padding-left: 40px }
li { display: list-item }

table { display: table; border-collapse: separate; border-spacing: 2px }
thead, tbody, tfoot { display: table-row-group; vertical-align: middle }
tr { display: table-row }
td, th { display: table-cell; padding: 1px }
th { font-weight: bold }

img { display: inline-block; object-fit: fill }
svg { display: inline-block }
br { display: inline-block }

input, textarea, select, button {
    display: inline-block; box-sizing: border-box;
    margin: 0; color: initial; line-height: normal;
    text-transform: none; text-indent: 0
}
input, textarea, select { padding: 1px 2px; border: 1px inset rgb(118,118,118); background-color: white }
button { padding: 1px 6px; border: 2px outset rgb(118,118,118); background-color: rgb(239,239,239) }
input[type="hidden"] { display: none }

pre, xmp, plaintext, listing { display: block; font-family: monospace; white-space: pre; margin: 1em 0 }
code, kbd, samp, tt { font-family: monospace }

blockquote { display: block; margin: 1em 40px }
figure { display: block; margin: 1em 40px }
figcaption { display: block }
hr { display: block; margin: 0.5em auto; border-style: inset; border-width: 1px }

article, aside, footer, header, hgroup, nav, section { display: block }
details { display: block }
summary { display: block }
canvas { display: inline-block }
iframe { display: inline-block }
"##;

use std::sync::OnceLock;
static MASTER: OnceLock<Stylesheet> = OnceLock::new();

fn master_stylesheet() -> &'static Stylesheet {
    MASTER.get_or_init(|| Stylesheet::parse(MASTER_CSS).expect("master CSS is valid"))
}

pub fn compute_style(
    element: &Element,
    stylesheets: &[Stylesheet],
    ancestors: &[&Element],
) -> HashMap<String, String> {
    let mut props = HashMap::new();
    for ss in std::iter::once(master_stylesheet()).chain(stylesheets.iter()) {
        let matched = ss.match_element(element, ancestors);
        props.extend(matched);
    }
    let mut vars = HashMap::new();
    for ss in stylesheets {
        vars.extend(ss.custom_properties.clone());
    }
    if let Some(style_attr) = element.attrs.get("style") {
        for decl in style_attr.split(';') {
            if let Some((k, v)) = decl.split_once(':') {
                let k = k.trim().to_ascii_lowercase();
                let v = v.trim();
                if !k.is_empty() && !v.is_empty() {
                    props.insert(k, v.to_string());
                }
            }
        }
    }
    for v in props.values_mut() {
        *v = resolve_var(v, &vars);
    }
    props
}

fn resolve_var(value: &str, vars: &HashMap<String, String>) -> String {
    let mut result = value.to_string();
    loop {
        let start = match result.find("var(") {
            Some(s) => s,
            None => break,
        };
        let mut depth = 0usize;
        let mut end = None;
        for (i, ch) in result[start..].char_indices() {
            if ch == '(' {
                depth += 1;
            } else if ch == ')' {
                depth -= 1;
                if depth == 0 {
                    end = Some(start + i);
                    break;
                }
            }
        }
        let end = match end {
            Some(e) => e,
            None => break,
        };
        let inner = &result[start + 4..end];
        let parts: Vec<&str> = inner.splitn(2, ',').collect();
        let name = parts[0].trim();
        let fallback = parts
            .get(1)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        let replacement = vars.get(name).cloned().unwrap_or(fallback);
        result.replace_range(start..=end, &replacement);
    }
    result
}

pub fn parse_css_color(value: &str) -> Option<Color> {
    let value = value.trim();
    if let Some(hex) = value.strip_prefix('#') {
        parse_hex_color(hex)
    } else if value == "transparent" {
        Some(Color::new([0.0, 0.0, 0.0, 0.0]))
    } else if value == "white" {
        Some(Color::new([1.0, 1.0, 1.0, 1.0]))
    } else if value == "black" {
        Some(Color::new([0.0, 0.0, 0.0, 1.0]))
    } else if value == "red" {
        Some(Color::new([1.0, 0.0, 0.0, 1.0]))
    } else if value == "green" {
        Some(Color::new([0.0, 0.5, 0.0, 1.0]))
    } else if value == "blue" {
        Some(Color::new([0.0, 0.0, 1.0, 1.0]))
    } else if value == "gray" || value == "grey" {
        Some(Color::new([0.5, 0.5, 0.5, 1.0]))
    } else if value.starts_with("rgb(") || value.starts_with("rgba(") {
        parse_rgb_function(value)
    } else {
        None
    }
}

fn parse_hex_color(hex: &str) -> Option<Color> {
    let hex = hex.trim();
    match hex.len() {
        3 => {
            let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
            Some(Color::new([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]))
        }
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            Some(Color::new([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]))
        }
        8 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
            Some(Color::new([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, a as f32 / 255.0]))
        }
        _ => None,
    }
}

fn parse_rgb_function(value: &str) -> Option<Color> {
    let inner = value
        .trim()
        .strip_prefix("rgb(")
        .or_else(|| value.trim().strip_prefix("rgba("))
        .and_then(|s| s.strip_suffix(')'))?;
    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
    let r: f32 = parts.get(0)?.parse().ok()?;
    let g: f32 = parts.get(1)?.parse().ok()?;
    let b: f32 = parts.get(2)?.parse().ok()?;
    let raw_a: f32 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let a = if raw_a > 1.0 { raw_a / 255.0 } else { raw_a };
    Some(Color::new([r / 255.0, g / 255.0, b / 255.0, a]))
}

pub fn parse_css_length(value: &str) -> Option<f64> {
    let value = value.trim();
    if let Some(px) = value.strip_suffix("px") {
        px.trim().parse::<f64>().ok()
    } else if let Some(rem) = value.strip_suffix("rem") {
        rem.trim().parse::<f64>().ok().map(|v| v * 16.0)
    } else {
        value.parse::<f64>().ok()
    }
}

pub fn parse_css_border_radius(value: &str) -> Vec<f64> {
    value
        .split_whitespace()
        .filter_map(parse_css_length)
        .collect()
}

pub fn parse_css_border_width(value: &str) -> Option<f32> {
    let value = value.trim();
    if let Some(px) = value.strip_suffix("px") {
        px.trim().parse::<f32>().ok()
    } else if value == "thin" {
        Some(1.0)
    } else if value == "medium" {
        Some(3.0)
    } else if value == "thick" {
        Some(5.0)
    } else {
        value.parse::<f32>().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dom::Element;

    #[test]
    fn test_parse_simple_css() {
        let css = "h1 { color: red; font-size: 16px; } .foo { color: blue; }";
        let ss = Stylesheet::parse(css).unwrap();
        assert_eq!(ss.rules.len(), 2);
    }

    #[test]
    fn test_match_tag() {
        let css = "h1 { color: red; }";
        let ss = Stylesheet::parse(css).unwrap();
        let el = Element::new("h1");
        let props = ss.match_element(&el, &[]);
        assert!(props.contains_key("color") || !props.is_empty());
    }

    #[test]
    fn test_match_descendant_tag() {
        let css = "button svg { width: 1.8rem; }";
        let ss = Stylesheet::parse(css).unwrap();
        let button = Element::new("button");
        let mut svg = Element::new("svg");
        let ancestors = [&button];
        let props = ss.match_element(&svg, &ancestors);
        assert_eq!(props.get("width").map(|s| s.as_str()), Some("1.8rem"));
    }

    #[test]
    fn test_match_descendant_class() {
        let css = ".tab .icon { width: 13px; }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut tab = Element::new("div");
        tab.attrs.insert("class".into(), "tab".into());
        let mut icon = Element::new("div");
        icon.attrs.insert("class".into(), "icon".into());
        let ancestors = [&tab];
        let props = ss.match_element(&icon, &ancestors);
        assert_eq!(props.get("width").map(|s| s.as_str()), Some("13px"));
    }

    #[test]
    fn test_match_child_class() {
        let css = ".header > span { opacity: 0.5; }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut header = Element::new("div");
        header.attrs.insert("class".into(), "header".into());
        let mut span = Element::new("span");
        let ancestors = [&header];
        let props = ss.match_element(&span, &ancestors);
        assert_eq!(props.get("opacity").map(|s| s.as_str()), Some(".5"));
    }

    #[test]
    fn test_match_child_not_descendant() {
        let css = ".header > span { opacity: 0.5; }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut header = Element::new("div");
        header.attrs.insert("class".into(), "header".into());
        let mut logs = Element::new("div");
        let mut span = Element::new("span");
        let ancestors = [&header, &logs];
        let props = ss.match_element(&span, &ancestors);
        assert!(props.is_empty());
    }

    #[test]
    fn test_match_descendant_still_works() {
        let css = ".header span { opacity: 0.5; }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut header = Element::new("div");
        header.attrs.insert("class".into(), "header".into());
        let mut logs = Element::new("div");
        let mut span = Element::new("span");
        let ancestors = [&header, &logs];
        let props = ss.match_element(&span, &ancestors);
        assert_eq!(props.get("opacity").map(|s| s.as_str()), Some(".5"));
    }

    #[test]
    fn test_parse_master_css() {
        let ss = Stylesheet::parse(MASTER_CSS).unwrap();
        assert!(!ss.rules.is_empty());
    }

    #[test]
    fn test_match_attribute_selector() {
        let css = "input[type=\"hidden\"] { display: none }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut hidden = Element::new("input");
        hidden.attrs.insert("type".into(), "hidden".into());
        let mut visible = Element::new("input");
        visible.attrs.insert("type".into(), "text".into());
        let props_hidden = ss.match_element(&hidden, &[]);
        let props_visible = ss.match_element(&visible, &[]);
        assert_eq!(props_hidden.get("display").map(|s| s.as_str()), Some("none"));
        assert!(!props_visible.contains_key("display"));
    }

    #[test]
    fn test_match_first_child() {
        let css = ".item:first-child { font-weight: bold }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut parent = Element::new("div");
        let mut children = vec![
            crate::dom::Node::Element(Element::new("span")),
            crate::dom::Node::Element(Element::new("span")),
        ];
        if let crate::dom::Node::Element(ref mut c1) = children[0] {
            c1.attrs.insert("class".into(), "item".into());
        }
        if let crate::dom::Node::Element(ref mut c2) = children[1] {
            c2.attrs.insert("class".into(), "item".into());
        }
        parent.children = children;
        if let crate::dom::Node::Element(child1) = &parent.children[0] {
            if let crate::dom::Node::Element(child2) = &parent.children[1] {
                let ancestors = [&parent];
                let props = ss.match_element(child1, &ancestors);
                assert_eq!(props.get("font-weight").map(|s| s.as_str()), Some("bold"));
                let props2 = ss.match_element(child2, &ancestors);
                assert!(!props2.contains_key("font-weight"));
            }
        }
    }

    #[test]
    fn test_match_pseudo_class_hover() {
        let css = "a:hover { color: red }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut el = Element::new("a");
        let props = ss.match_element(&el, &[]);
        assert!(props.is_empty());
    }

    #[test]
    fn test_match_only_child() {
        let css = "li:only-child { list-style: none }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut ul = Element::new("ul");
        let li = Element::new("li");
        ul.children.push(crate::dom::Node::Element(li));
        if let crate::dom::Node::Element(li_ref) = &ul.children[0] {
            let ancestors = [&ul];
            let props = ss.match_element(li_ref, &ancestors);
            assert_eq!(props.get("list-style").map(|s| s.as_str()), Some("none"));
        }
    }

    #[test]
    fn test_match_nth_child() {
        let css = "tr:nth-child(2n+1) { background: gray }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut table = Element::new("table");
        let mut rows: Vec<crate::dom::Node> = (0..4)
            .map(|_| crate::dom::Node::Element(Element::new("tr")))
            .collect();
        table.children = rows;
        let ancestors = vec![&table];
        if let crate::dom::Node::Element(r0) = &table.children[0] {
            let props1 = ss.match_element(r0, &ancestors);
            assert_eq!(props1.get("background").map(|s| s.as_str()), Some("gray"));
        }
        if let crate::dom::Node::Element(r1) = &table.children[1] {
            let props2 = ss.match_element(r1, &ancestors);
            assert!(!props2.contains_key("background"));
        }
        if let crate::dom::Node::Element(r2) = &table.children[2] {
            let props3 = ss.match_element(r2, &ancestors);
            assert_eq!(props3.get("background").map(|s| s.as_str()), Some("gray"));
        }
    }

    #[test]
    fn test_match_not_pseudo() {
        let css = "input:not([type=\"hidden\"]) { display: inline-block }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut text_input = Element::new("input");
        text_input.attrs.insert("type".into(), "text".into());
        let mut hidden = Element::new("input");
        hidden.attrs.insert("type".into(), "hidden".into());
        let props_text = ss.match_element(&text_input, &[]);
        assert_eq!(
            props_text.get("display").map(|s| s.as_str()),
            Some("inline-block")
        );
        let props_hidden = ss.match_element(&hidden, &[]);
        assert!(!props_hidden.contains_key("display"));
    }

    #[test]
    fn test_empty_pseudo() {
        let css = "div:empty { display: none }";
        let ss = Stylesheet::parse(css).unwrap();
        let empty = Element::new("div");
        let mut non_empty = Element::new("div");
        non_empty
            .children
            .push(crate::dom::Node::Element(Element::new("span")));
        assert_eq!(
            ss.match_element(&empty, &[])
                .get("display")
                .map(|s| s.as_str()),
            Some("none")
        );
        assert!(!ss
            .match_element(&non_empty, &[])
            .contains_key("display"));
    }

    #[test]
    fn test_parse_complex_selectors() {
        let css = r#"
            div.container > ul.list li.active { color: red }
            a[href^="https"]:hover { text-decoration: underline }
            p:first-of-type { font-size: 1.2em }
            .foo ~ .bar { margin-top: 1em }
            .baz + .qux { margin-left: 0 }
        "#;
        let ss = Stylesheet::parse(css).unwrap();
        assert_eq!(ss.rules.len(), 5);
    }

    #[test]
    fn test_specificity_overrides() {
        let css = ".foo { color: blue }\ndiv { color: red }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut el = Element::new("div");
        el.attrs.insert("class".into(), "foo".into());
        let props = ss.match_element(&el, &[]);
        assert_eq!(props.get("color").map(|s| s.as_str()), Some("#00f"));
    }

    #[test]
    fn test_specificity_id_wins() {
        let css = ".foo.bar { color: red }\n#x { color: green }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut el = Element::new("div");
        el.attrs.insert("class".into(), "foo bar".into());
        el.attrs.insert("id".into(), "x".into());
        let props = ss.match_element(&el, &[]);
        assert_eq!(props.get("color").map(|s| s.as_str()), Some("green"));
    }

    #[test]
    fn test_specificity_source_order_tie() {
        let css = ".a { color: red }\n.b { color: blue }";
        let ss = Stylesheet::parse(css).unwrap();
        let mut el = Element::new("div");
        el.attrs.insert("class".into(), "a b".into());
        let props = ss.match_element(&el, &[]);
        assert_eq!(props.get("color").map(|s| s.as_str()), Some("#00f"));
    }
}
