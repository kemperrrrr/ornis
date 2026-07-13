use std::collections::HashMap;

use lightningcss::stylesheet::{ParserOptions, StyleSheet};
use lightningcss::traits::ToCss;
use vello::peniko::Color;

use crate::dom::Element;

#[derive(Debug, Clone)]
pub struct Stylesheet {
    pub rules: Vec<Rule>,
    pub custom_properties: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Rule {
    /// Each comma-separated selector is a list of (combinator, simple) parts,
    /// ordered from the outermost ancestor (`[0]`) to the target (last).
    pub selectors: Vec<Selector>,
    pub declarations: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimpleSelector {
    Universal,
    /// The `:root` pseudo-class — matches the document's root element
    /// (`<html>`). Needed so `:root { font-size: 12px }` actually applies and
    /// drives `rem` sizing.
    Root,
    Tag(String),
    Class(String),
    Id(String),
    /// Pseudo-classes, attribute selectors and functional selectors that we
    /// don't model (e.g. `:hover`, `:not([open])`, `[open]`, `::before`).
    /// Matching one of these always fails so the whole compound doesn't apply.
    Unsupported,
}

/// Combinator that precedes a simple selector in a compound selector.
#[derive(Debug, Clone, PartialEq)]
pub enum Combinator {
    /// ` ` descendant combinator.
    Descendant,
    /// `>` child combinator.
    Child,
}

/// A single (comma-separated) selector: ordered ancestor-first, target last.
/// Each `CompoundPart` is a compound selector (a sequence of simple selectors
/// with no combinators between them, e.g. `.icon.close` or `button.add svg`),
/// plus the combinator that links it to the previous part.
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundPart {
    pub comb: Combinator,
    pub parts: Vec<SimpleSelector>,
}

pub type Selector = Vec<CompoundPart>;

/// Split a single compound (no combinator chars) into its simple selectors.
/// E.g. `.icon.close` -> [Class("icon"), Class("close")],
/// `div.panel.left-upper` -> [Tag("div"), Class("panel"), Class("left-upper")].
fn parse_compound(s: &str) -> Vec<SimpleSelector> {
    let mut out = Vec::new();
    let cs: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut buf = String::new();
    let flush = |buf: &str, out: &mut Vec<SimpleSelector>| {
        let t = buf.trim();
        if !t.is_empty() {
            out.push(parse_simple(t));
        }
    };
    while i < cs.len() {
        let c = cs[i];
        // A new simple selector begins at `.`, `#`, `:` and runs until the
        // next such boundary char (or whitespace). The boundary char itself
        // belongs to the simple selector it opens (e.g. the `.` in `.icon`),
        // so it must be consumed too — otherwise `i` never advances and the
        // loop spins forever on `.icon.close`.
        if c == '.' || c == '#' || c == ':' {
            flush(&buf, &mut out);
            buf.clear();
            let mut j = i;
            while j < cs.len() {
                let d = cs[j];
                // Stop at the *next* boundary / whitespace, keeping the one
                // that opened this simple selector (j == i) included.
                if j > i && (d == '.' || d == '#' || d == ':' || d.is_whitespace()) {
                    break;
                }
                buf.push(d);
                j += 1;
            }
            flush(&buf, &mut out);
            buf.clear();
            i = j;
        } else {
            buf.push(c);
            i += 1;
        }
    }
    flush(&buf, &mut out);
    out
}

fn parse_simple(s: &str) -> SimpleSelector {
    let s = s.trim();
    if s.is_empty() {
        return SimpleSelector::Unsupported;
    }
    if s == "*" {
        return SimpleSelector::Universal;
    }
    if s == ":root" {
        return SimpleSelector::Root;
    }
    // Pseudo-classes, pseudo-elements and attribute/functional selectors are
    // not modelled.
    if s.starts_with(':') || s.starts_with('[') || s.contains('(') {
        return SimpleSelector::Unsupported;
    }
    if let Some(c) = s.strip_prefix('#') {
        return SimpleSelector::Id(c.to_string());
    }
    if let Some(c) = s.strip_prefix('.') {
        return SimpleSelector::Class(c.to_string());
    }
    SimpleSelector::Tag(s.to_string())
}

fn parse_one_selector(group: &str) -> Selector {
    let mut sel: Selector = Vec::new();
    let mut comb = Combinator::Descendant;
    let mut buf = String::new();
    let cs: Vec<char> = group.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c == '>' {
            if !buf.trim().is_empty() {
                sel.push(CompoundPart {
                    comb,
                    parts: parse_compound(&buf),
                });
                buf.clear();
            }
            comb = Combinator::Child;
            i += 1;
        } else if c.is_whitespace() {
            if !buf.trim().is_empty() {
                sel.push(CompoundPart {
                    comb,
                    parts: parse_compound(&buf),
                });
                buf.clear();
            }
            comb = Combinator::Descendant;
            i += 1;
        } else {
            buf.push(c);
            i += 1;
        }
    }
    if !buf.trim().is_empty() {
        sel.push(CompoundPart {
            comb,
            parts: parse_compound(&buf),
        });
    }
    sel
}

fn extract_selectors(selector_str: &str) -> Vec<Selector> {
    let v: Vec<Selector> = selector_str
        .split(',')
        .map(|g| parse_one_selector(g))
        .filter(|s| !s.is_empty())
        .collect();
    v
}

impl Stylesheet {
    pub fn parse(css: &str) -> Result<Self, String> {
        let sheet = StyleSheet::parse(css, ParserOptions::default())
            .map_err(|e| format!("CSS parse error: {e}"))?;

        let mut rules = Vec::new();
        let mut custom_properties = HashMap::new();

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
                        let name = prop.property_id().name().to_string();
                        let mut value = String::new();
                        let mut printer = lightningcss::printer::Printer::new(
                            &mut value,
                            lightningcss::stylesheet::PrinterOptions::default(),
                        );
                        if prop.to_css(&mut printer, false).is_ok() {
                            // lightningcss serializes the whole `name: value`
                            // declaration, so strip the `name:` prefix to get
                            // just the value.
                            let v = value
                                .strip_prefix(&format!("{name}:"))
                                .map(|s| s.strip_prefix(' ').unwrap_or(s).to_string())
                                .unwrap_or_else(|| value.clone());
                            declarations.insert(name.clone(), v.clone());
                            // `:root` custom properties are document-global.
                            if selector_str.contains(":root") && name.starts_with("--") {
                                custom_properties.insert(name, v);
                            }
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

        Ok(Stylesheet { rules, custom_properties })
    }

    pub fn match_element(&self, element: &Element, ancestors: &[&Element]) -> HashMap<String, String> {
        let mut props = HashMap::new();
        for rule in &self.rules {
            if rule.selectors.iter().any(|sel| matches_selector(sel, element, ancestors)) {
                for (k, v) in &rule.declarations {
                    props.insert(k.clone(), v.clone());
                }
            }
        }
        props
    }
}

fn match_simple(sel: &SimpleSelector, element: &Element) -> bool {
    match sel {
        SimpleSelector::Universal => true,
        SimpleSelector::Root => element.tag == "html",
        SimpleSelector::Tag(t) => element.tag == *t,
        SimpleSelector::Class(c) => element.classes().contains(&c.as_str()),
        SimpleSelector::Id(id) => element.id() == Some(id.as_str()),
        SimpleSelector::Unsupported => false,
    }
}

/// Matches a compound selector. `ancestors` is ordered root-first, with the
/// last element being the immediate parent of `element`.
fn match_compound(parts: &[SimpleSelector], element: &Element) -> bool {
    parts.iter().all(|s| match_simple(s, element))
}

fn matches_selector(sel: &Selector, element: &Element, ancestors: &[&Element]) -> bool {
    if sel.is_empty() {
        return false;
    }
    let target = &sel[sel.len() - 1];
    if !match_compound(&target.parts, element) {
        return false;
    }
    // Walk the remaining parts from the target's immediate predecessor up to
    // the outermost ancestor.
    let mut cursor = ancestors.len();
    for k in (0..sel.len() - 1).rev() {
        let (comb, parts) = (&sel[k].comb, &sel[k].parts);
        let found = if matches!(comb, Combinator::Child) {
            // Only the direct parent qualifies.
            if cursor > 0 {
                let parent = ancestors[cursor - 1];
                if match_compound(parts, parent) {
                    Some(cursor - 1)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            let mut hit = None;
            for j in (0..cursor).rev() {
                if match_compound(parts, ancestors[j]) {
                    hit = Some(j);
                    break;
                }
            }
            hit
        };
        match found {
            Some(j) => cursor = j,
            None => return false,
        }
    }
    true
}

pub fn compute_style(
    element: &Element,
    stylesheets: &[Stylesheet],
    ancestors: &[&Element],
) -> HashMap<String, String> {
    let mut props = HashMap::new();
    let mut vars = HashMap::new();
    for ss in stylesheets {
        vars.extend(ss.custom_properties.clone());
        let matched = ss.match_element(element, ancestors);
        props.extend(matched);
    }
    // The inline `style="..."` attribute takes precedence over stylesheet
    // rules (it's how the shared editor sets per-icon colors, e.g.
    // `<div class="icon" style="fill:#5796e8">`). Without parsing it, every
    // SVG icon loses its color and falls back to white.
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

/// Resolves `var(--name)` (and `var(--name, fallback)`) references in a CSS
/// declaration value using the supplied custom-property map.
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
        let fallback = parts.get(1).map(|s| s.trim().to_string()).unwrap_or_default();
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
    let inner = value.trim().strip_prefix("rgb(")
        .or_else(|| value.trim().strip_prefix("rgba("))
        .and_then(|s| s.strip_suffix(')'))?;
    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
    let r: f32 = parts.get(0)?.parse().ok()?;
    let g: f32 = parts.get(1)?.parse().ok()?;
    let b: f32 = parts.get(2)?.parse().ok()?;
    // CSS `rgba(r, g, b, a)` uses an alpha in the 0..1 range; only treat it as
    // 0..255 if the value is clearly out of that range (non-standard input).
    // The previous code divided `a` by 255 unconditionally, which made every
    // `rgba(...)` color (e.g. themed panels) virtually transparent.
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
        let props = ss.match_element(&el, &[]);
        assert!(props.contains_key("color") || !props.is_empty());
    }

    #[test]
    fn test_match_descendant_tag() {
        // `button svg` must match an <svg> whose ancestor is <button>.
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
        // `.tab .icon` must match a `.icon` inside a `.tab`.
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
}
