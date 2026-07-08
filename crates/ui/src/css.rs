use std::collections::HashMap;

use lightningcss::stylesheet::{ParserOptions, StyleSheet};
use lightningcss::traits::ToCss;
use vello::peniko::Color;

use crate::dom::Element;

#[derive(Debug, Clone)]
pub struct Stylesheet {
    pub rules: Vec<Rule>,
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub selectors: Vec<SimpleSelector>,
    pub declarations: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimpleSelector {
    Universal,
    Tag(String),
    Class(String),
    Id(String),
}

fn extract_selectors(selector_str: &str) -> Vec<SimpleSelector> {
    let mut sels = Vec::new();
    for part in selector_str.split(|c: char| c == ' ' || c == '>' || c == '+' || c == '~' || c == ',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if part == "*" {
            sels.push(SimpleSelector::Universal);
            continue;
        }
        let mut tag = String::new();
        let mut in_tag = true;
        for ch in part.chars() {
            match ch {
                '.' => {
                    in_tag = false;
                }
                '#' => {
                    if !tag.is_empty() {
                        sels.push(SimpleSelector::Tag(tag.clone()));
                        tag.clear();
                    }
                    in_tag = false;
                }
                _ => {
                    if in_tag {
                        tag.push(ch);
                    }
                }
            }
        }
            let mut current = String::new();
        let mut mode = 't'; // t=tag, c=class, i=id
        for ch in part.chars() {
            match ch {
                '.' => {
                    if mode == 't' && !current.is_empty() {
                        sels.push(SimpleSelector::Tag(current.clone()));
                    } else if mode == 'c' && !current.is_empty() {
                        sels.push(SimpleSelector::Class(current.clone()));
                    } else if mode == 'i' && !current.is_empty() {
                        sels.push(SimpleSelector::Id(current.clone()));
                    }
                    current.clear();
                    mode = 'c';
                }
                '#' => {
                    if mode == 't' && !current.is_empty() {
                        sels.push(SimpleSelector::Tag(current.clone()));
                    } else if mode == 'c' && !current.is_empty() {
                        sels.push(SimpleSelector::Class(current.clone()));
                    } else if mode == 'i' && !current.is_empty() {
                        sels.push(SimpleSelector::Id(current.clone()));
                    }
                    current.clear();
                    mode = 'i';
                }
                _ => {
                    current.push(ch);
                }
            }
        }
        // Push the last segment
        if mode == 't' && !current.is_empty() {
            sels.push(SimpleSelector::Tag(current));
        } else if mode == 'c' && !current.is_empty() {
            sels.push(SimpleSelector::Class(current));
        } else if mode == 'i' && !current.is_empty() {
            sels.push(SimpleSelector::Id(current));
        }
    }
    sels
}

impl Stylesheet {
    pub fn parse(css: &str) -> Result<Self, String> {
        let sheet = StyleSheet::parse(css, ParserOptions::default())
            .map_err(|e| format!("CSS parse error: {e}"))?;

        let mut rules = Vec::new();

        for rule in sheet.rules.0.iter() {
            use lightningcss::rules::CssRule;
            match rule {
                CssRule::Style(style_rule) => {
                    // Render selectors to string
                    let mut selector_str = String::new();
                    let mut printer = lightningcss::printer::Printer::new(
                        &mut selector_str,
                        lightningcss::stylesheet::PrinterOptions::default(),
                    );
                    let _ = style_rule.selectors.to_css(&mut printer);

                    let selectors = extract_selectors(&selector_str);

                    let mut declarations = HashMap::new();
                    for (prop, _important) in style_rule.declarations.iter() {
                        let mut value = String::new();
                        let mut printer = lightningcss::printer::Printer::new(
                            &mut value,
                            lightningcss::stylesheet::PrinterOptions::default(),
                        );
                        if prop.to_css(&mut printer, false).is_ok() {
                            declarations.insert(prop.property_id().name().to_string(), value);
                        }
                    }

                    rules.push(Rule {
                        selectors,
                        declarations,
                    });
                }
                _ => {}
            }
        }

        Ok(Stylesheet { rules })
    }

    pub fn match_element(&self, element: &Element) -> HashMap<String, String> {
        let mut props = HashMap::new();
        for rule in &self.rules {
            if rule.selectors.iter().any(|sel| matches_selector(sel, element)) {
                for (k, v) in &rule.declarations {
                    props.insert(k.clone(), v.clone());
                }
            }
        }
        props
    }
}

fn matches_selector(sel: &SimpleSelector, element: &Element) -> bool {
    match sel {
        SimpleSelector::Universal => true,
        SimpleSelector::Tag(t) => element.tag == *t,
        SimpleSelector::Class(c) => element.classes().contains(&c.as_str()),
        SimpleSelector::Id(id) => element.id() == Some(id.as_str()),
    }
}

pub fn compute_style(element: &Element, stylesheets: &[Stylesheet]) -> HashMap<String, String> {
    let mut props = HashMap::new();
    for ss in stylesheets {
        let matched = ss.match_element(element);
        props.extend(matched);
    }
    props
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
            let b = u8::from_str_radix(&hex[2..3], 16).ok()?;
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
    let inner = value.trim().strip_prefix("rgb(")
        .or_else(|| value.trim().strip_prefix("rgba("))
        .and_then(|s| s.strip_suffix(')'))?;
    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
    let r: f32 = parts.get(0)?.parse().ok()?;
    let g: f32 = parts.get(1)?.parse().ok()?;
    let b: f32 = parts.get(2)?.parse().ok()?;
    let a: f32 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(255.0);
    Some(Color::new([r / 255.0, g / 255.0, b / 255.0, a / 255.0]))
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
    value.split_whitespace()
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
        let props = ss.match_element(&el);
        assert!(props.contains_key("color") || !props.is_empty());
    }
}
