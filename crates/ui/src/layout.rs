use std::collections::HashMap;
use std::sync::Arc;

use taffy::MaybeResolve;
use taffy::geometry::Line;
use taffy::style::{
    Dimension, Display, GridPlacement, LengthPercentage, LengthPercentageAuto, Position, Style,
    TrackSizingFunction,
};
use taffy::style_helpers::{
    FromLength, FromPercent, TaffyAuto, TaffyGridLine, TaffyGridSpan, flex,
};
use taffy::{AlignItems, AvailableSpace, JustifyContent, Size, TaffyResult, TaffyTree};
use vello::peniko::FontData;

use crate::css::{SimpleSelector, Stylesheet};
use crate::dom::{Document, Node};
use crate::image_loader::{DecodedImage, ImageCache};

pub type LayoutNodeId = usize;

#[derive(Debug, Clone, Default)]
pub struct LayoutTree {
    pub arena: Vec<LayoutNode>,
    pub root: LayoutNodeId,
    /// Shared cache of decoded `<img>` assets so re-layouts (resize / editor
    /// toggle) reuse decoded bytes instead of hitting the filesystem.
    pub image_cache: Arc<ImageCache>,
    /// The viewport (CSS px) the tree was laid out against. Kept so `to_json`
    /// can report the same container size a browser probe would use.
    pub viewport: (f32, f32),
}

#[derive(Debug, Clone)]
pub struct LayoutNode {
    pub id: LayoutNodeId,
    pub tag: String,
    pub dom_id: Option<String>,
    pub dom_class: Option<String>,
    pub rect: Rect,
    pub styles: HashMap<String, String>,
    pub children: Vec<LayoutNodeId>,
    pub parent: Option<LayoutNodeId>,
    pub text: Option<String>,
    /// For `<svg>` nodes: the viewBox size (width, height) used to scale paths.
    pub svg_view_box: Option<(f32, f32, f32, f32)>,
    /// For `<svg>`/`<path>` nodes: the icon path `d` and its explicit `fill`
    /// (if any). When `None`, the fill is inherited from ancestors (CSS `fill`
    /// is an inherited property) and resolved at paint time.
    pub svg_path: Option<(String, Option<String>)>,
    /// For `<img>` nodes: the decoded image to paint, plus its intrinsic
    /// (natural) pixel size. `None` for images that failed to load or for
    /// non-image elements.
    pub image: Option<crate::image_loader::DecodedImage>,
    /// For `<img>` nodes: intrinsic pixel size (width, height) of the source.
    /// Used to derive an aspect ratio when only one CSS dimension is given
    /// (e.g. `.file-icon { height: 4.5rem }` with no width, mirroring the
    /// browser's replaced-element sizing).
    pub image_intrinsic_size: Option<(u32, u32)>,
    /// For elements with a CSS `background-image: url(...)`: the decoded image
    /// to paint as the element's background. `None` when no background image is
    /// declared or it failed to load.
    pub background_image: Option<crate::image_loader::DecodedImage>,
    /// For elements with a CSS `background-image: url(...)`: the layout sizing
    /// keyword (`cover`, `contain`, or `None`/repeat). Mirrors the browser's
    /// `background-size` shorthand (`cover`/`contain`); anything else paints at
    /// intrinsic size.
    pub background_size: Option<String>,
    /// Chain of ancestors from root down to (but not including) this node,
    /// each rendered as `tag.class1 class2`. Mirrors the browser probe's
    /// `parent_chain` so a layout/style diff can identify which DOM path a
    /// node took. Populated at build time from `ancestors`.
    pub parent_chain: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Elements whose content must never be rendered as visible text/layout
/// (`<head>` subtree, scripts, templates, etc.).
const SKIP_TAGS: &[&str] = &[
    "head", "style", "script", "title", "meta", "link", "noscript", "template",
];

impl LayoutTree {
    pub fn build(doc: &Document, stylesheets: &[Stylesheet], font: &FontData) -> TaffyResult<Self> {
        Self::build_with_viewport(doc, stylesheets, 1024.0, 768.0, font)
    }

    pub fn build_with_viewport(
        doc: &Document,
        stylesheets: &[Stylesheet],
        viewport_width: f32,
        viewport_height: f32,
        font: &FontData,
    ) -> TaffyResult<Self> {
        let mut taffy = TaffyTree::new();
        let mut arena = Vec::<LayoutNode>::new();
        let mut taffy_to_arena: HashMap<u64, LayoutNodeId> = HashMap::new();
        let inherited = HashMap::new();
        // `rem` units resolve against the root element's computed font-size
        // (defaults to 16px, but the editor's `:root { font-size: 12px }`
        // overrides it). Pass this down so length parsing is correct.
        let rem_base = root_font_size(doc, stylesheets);
        let image_cache = Arc::new(ImageCache::new());
        let (root_tid, root_aid) = Self::build_node(
            &mut taffy,
            &doc.root,
            stylesheets,
            &mut arena,
            &mut taffy_to_arena,
            &inherited,
            &[],
            viewport_width,
            viewport_height,
            true,
            false,
            false,
            rem_base,
            &image_cache,
        )?;

        taffy.compute_layout_with_measure(
            root_tid,
            Size {
                width: AvailableSpace::Definite(viewport_width),
                height: AvailableSpace::Definite(viewport_height),
            },
            |_known, _available, node_id, _ctx, _style| {
                let tkey: u64 = node_id.into();
                let aid = match taffy_to_arena.get(&tkey) {
                    Some(&a) => a,
                    None => {
                        return taffy::geometry::Size {
                            width: 0.0,
                            height: 0.0,
                        };
                    }
                };
                let node = &arena[aid];
                if let Some(text) = &node.text {
                    let fs = node
                        .styles
                        .get("font-size")
                        .and_then(|v| parse_length(v, 16.0))
                        .unwrap_or(14.0);
                    let (w, h) = crate::text::measure_text(font, text, fs as f32);
                    taffy::geometry::Size {
                        width: w,
                        height: h,
                    }
                } else {
                    taffy::geometry::Size {
                        width: 0.0,
                        height: 0.0,
                    }
                }
            },
        )?;

        Self::apply_layout(&taffy, &mut arena, root_tid, &taffy_to_arena);

        Ok(LayoutTree {
            arena,
            root: root_aid,
            image_cache,
            viewport: (viewport_width, viewport_height),
        })
    }

    /// Serializes the laid-out tree to a JSON value that mirrors what a browser
    /// probe (`getComputedStyle` + `getBoundingClientRect`) would report for the
    /// same HTML/CSS. Intended for `diff.py` so layout/style bugs can be
    /// verified against a real browser instead of by eye.
    ///
    /// Gated behind the `serialize` feature (pulls in `serde_json`). Only
    /// called from the `serialize_layout` example / tests.
    #[cfg(feature = "serialize")]
    pub fn to_json(&self) -> serde_json::Value {
        let nodes: Vec<serde_json::Value> = self
            .arena
            .iter()
            .map(|n| {
                let label = n
                    .dom_id
                    .clone()
                    .map(|i| format!("#{}", i))
                    .or_else(|| n.dom_class.clone().map(|c| format!(".{}", c)))
                    .unwrap_or_else(|| n.tag.clone());
                let mut styles: std::collections::HashMap<String, String> = n.styles.clone();
                if !styles.contains_key("width") {
                    styles.insert("width".into(), format!("{}px", n.rect.width));
                }
                if !styles.contains_key("height") {
                    styles.insert("height".into(), format!("{}px", n.rect.height));
                }
                let svg = if n.svg_path.is_some() {
                    serde_json::json!({
                        "has_path": true,
                        "view_box": n.svg_view_box,
                        "intrinsic_w": n.svg_view_box.map(|v| v.2),
                        "intrinsic_h": n.svg_view_box.map(|v| v.3),
                    })
                } else {
                    serde_json::json!({ "has_path": false })
                };
                serde_json::json!({
                    "label": label,
                    "tag": n.tag,
                    "dom_id": n.dom_id,
                    "dom_class": n.dom_class,
                    "rect": {
                        "x": n.rect.x,
                        "y": n.rect.y,
                        "width": n.rect.width,
                        "height": n.rect.height,
                    },
                    "styles": styles,
                    "svg": svg,
                    "parent_chain": n.parent_chain,
                    "image_intrinsic": n.image_intrinsic_size,
                    "has_background_image": n.background_image.is_some(),
                })
            })
            .collect();
        serde_json::json!({
            "viewport": { "width": self.viewport.0, "height": self.viewport.1 },
            "node_count": nodes.len(),
            "nodes": nodes,
        })
    }

    /// Extracts SVG icon geometry from an element so it can be painted as a
    /// `<path>`'s `d`; the fill color comes from the computed CSS `fill`.
    /// Returns the full viewBox as `(min_x, min_y, width, height)`. The origin
    /// (min-x/min-y) matters: Material-style icons use e.g. `0 -960 960 960`,
    /// and the painter must offset the path by `min-y` or the icon lands
    /// outside its box.
    /// True for elements that must be excluded from the layout tree entirely.
    /// Only `display: none` qualifies — absolute/fixed positioned elements are
    /// now laid out by taffy (resolved against their nearest `position:
    /// relative` containing block via `inset`), matching browser behaviour.
    fn is_out_of_flow(styles: &HashMap<String, String>) -> bool {
        // Only `display: none` removes an element from the rendered tree.
        // `position: absolute | fixed | sticky` is handled natively by taffy
        // (see `apply_styles`: it sets `Position::Absolute`), so those elements
        // are taken out of normal flow automatically without us having to skip
        // them. Skipping them here was wrong: it dropped legitimate overlay
        // content such as the per-panel `.tab-list` (tabs like Viewport /
        // Hierarchy / Resources / Inspector) and the command palette entirely.
        if let Some(d) = styles.get("display") {
            if d.trim() == "none" {
                return true;
            }
        }
        false
    }

    fn parse_view_box(attrs: &HashMap<String, String>) -> Option<(f32, f32, f32, f32)> {
        attrs.get("viewBox").and_then(|vb| {
            let parts: Vec<f32> = vb
                .split_whitespace()
                .filter_map(|p| p.parse::<f32>().ok())
                .collect();
            if parts.len() == 4 && parts[2] > 0.0 && parts[3] > 0.0 {
                Some((parts[0], parts[1], parts[2], parts[3]))
            } else {
                None
            }
        })
    }

    /// Intrinsic size for an `<svg>` that has no CSS width/height. Browsers use
    /// the viewBox dimensions; we do too, but clamp oversized viewBoxes (e.g.
    /// the 960x960 Material-style icons) to a sensible icon size so they don't
    /// blow up the layout.
    fn intrinsic_svg_size(vbw: f32, vbh: f32) -> (f32, f32) {
        const MAX: f32 = 64.0;
        let max_dim = vbw.max(vbh);
        if max_dim <= MAX {
            (vbw, vbh)
        } else {
            let scale = 24.0 / max_dim;
            (vbw * scale, vbh * scale)
        }
    }

    fn extract_svg(
        el: &crate::dom::Element,
        styles: &HashMap<String, String>,
    ) -> (
        Option<(f32, f32, f32, f32)>,
        Option<(String, Option<String>)>,
    ) {
        let view_box = if el.tag == "svg" {
            Self::parse_view_box(&el.attrs)
        } else {
            None
        };

        // Only `<svg>` carries the drawable path (extracted from its child
        // `<path>`). A bare `<path>` element is also built as its own layout
        // node, but it must NOT also carry an svg_path — otherwise the same
        // vector is painted twice (once on the <svg>, once on the <path>),
        // which showed up as duplicated icons.
        let path_d = if el.tag == "svg" {
            el.children.iter().find_map(|c| match c {
                Node::Element(ce) if ce.tag == "path" => ce.attrs.get("d").cloned(),
                _ => None,
            })
        } else {
            None
        };

        // The `fill` may live on the <svg> (e.g. fill="currentColor") or on the
        // <path> itself. Prefer an explicit value over the inherited default.
        let path_fill = if el.tag == "svg" {
            el.children.iter().find_map(|c| match c {
                Node::Element(ce) if ce.tag == "path" => ce.attrs.get("fill").cloned(),
                _ => None,
            })
        } else {
            None
        };

        let svg_path = path_d.map(|d| {
            let fill = styles
                .get("fill")
                .cloned()
                .or_else(|| el.attrs.get("fill").cloned())
                .or_else(|| path_fill.clone());
            (d, fill)
        });

        (view_box, svg_path)
    }

    /// Extracts the first `url(...)` reference from a CSS property value, used
    /// for `background-image` / `background`. Strips optional quotes and returns
    /// the bare path (e.g. `url('foo.png')` -> `foo.png`). Returns `None` when
    /// there is no `url()` (e.g. `none`, a CSS gradient, or a solid color).
    fn extract_url(value: &str) -> Option<String> {
        let start = value.find("url(")?;
        let rest = &value[start + 4..];
        let end = rest.find(')')?;
        let inner = &rest[..end];
        let inner = inner.trim();
        let inner = inner
            .strip_prefix('\'')
            .or_else(|| inner.strip_prefix('"'))
            .unwrap_or(inner);
        let inner = inner
            .strip_suffix('\'')
            .or_else(|| inner.strip_suffix('"'))
            .unwrap_or(inner);
        let inner = inner.trim();
        if inner.is_empty() || inner.eq_ignore_ascii_case("none") {
            None
        } else {
            Some(inner.to_string())
        }
    }

    fn build_node(
        taffy: &mut TaffyTree,
        node: &Node,
        stylesheets: &[Stylesheet],
        arena: &mut Vec<LayoutNode>,
        taffy_to_arena: &mut HashMap<u64, LayoutNodeId>,
        inherited: &HashMap<String, String>,
        ancestors: &[&crate::dom::Element],
        vw: f32,
        vh: f32,
        is_root: bool,
        parent_is_column: bool,
        parent_is_row: bool,
        rem_base: f32,
        image_cache: &Arc<ImageCache>,
    ) -> TaffyResult<(taffy::NodeId, LayoutNodeId)> {
        match node {
            Node::Element(el) => {
                let mut styles = crate::css::compute_style(el, stylesheets, ancestors);
                if el.tag == "svg" {
                    let parent_tag = ancestors.last().map(|p| p.tag.as_str()).unwrap_or("NONE");
                    if parent_tag == "button" {
                        // (debug removed)
                    }
                }
                for (k, v) in inherited {
                    // `fill` is an inherited CSS property (browsers inherit it),
                    // so SVG icons pick up an inline `style="fill:#5796e8"` from
                    // their parent `.icon` container. Without this, every icon
                    // falls back to white.
                    if ["color", "font-size", "font-family", "fill", "visibility"]
                        .contains(&k.as_str())
                    {
                        styles.entry(k.clone()).or_insert_with(|| v.clone());
                    }
                }
                // For `<svg>`, the CSS box (incl. `width:100%` inherited from a
                // sized `.icon`/`.file` container) must win over the legacy
                // `width`/`height` *attributes* — exactly like browsers. The
                // editor's icons carry `width="24" height="24"` attributes but
                // are laid out from CSS, so an attribute-derived size would
                // override the real CSS size and blow the icon up to 24x24.
                // Drop any size key whose resolved px value equals the
                // attribute, leaving the CSS/parent path to size the svg.
                if el.tag == "svg" {
                    for (attr, key) in [("width", "width"), ("height", "height")] {
                        if let Some(av) = el.attrs.get(attr) {
                            if let Ok(af) = av.parse::<f32>() {
                                if let Some(sv) = styles.get(key) {
                                    let sv_trim = sv.trim();
                                    let is_attr_px = sv_trim == format!("{}px", af as i32)
                                        || sv_trim == format!("{}.0px", af as i32)
                                        || sv_trim == format!("{}px", af)
                                        || sv_trim == format!("{}.0px", af);
                                    if is_attr_px {
                                        styles.remove(key);
                                    }
                                }
                            }
                        }
                    }
                }
                let mut taffy_style = css_to_taffy_style(&styles, vw, vh, rem_base);
                // SVG elements with no explicit CSS width/height:
                //  - if their parent has a definite size, fill it (`width/height:
                //    100%`, matching browser behaviour for replaced elements) so
                //    the icon doesn't overflow its container and overlap siblings;
                //  - otherwise fall back to the viewBox intrinsic size (clamped),
                //    so icons without a sized container still paint instead of
                //    collapsing to 0x0.
                if el.tag == "svg" {
                    let has_size = styles.contains_key("width") || styles.contains_key("height");
                    if !has_size {
                        // A `<svg>` with no explicit CSS size inherits the box
                        // of its nearest sized ancestor (the `.icon`/`.file`
                        // container, which carries `width: Nrem`). Browsers make
                        // the svg fill that box (the icon is `width:100%` of its
                        // container), so we size the svg to the parent's definite
                        // dimension — width OR height is enough (the other axis
                        // follows from the viewBox aspect ratio in paint).
                        let parent_has_size = ancestors.last().map(|p| {
                            let ps = crate::css::compute_style(
                                p,
                                stylesheets,
                                &ancestors[..ancestors.len().saturating_sub(1)],
                            );
                            ps.contains_key("width") || ps.contains_key("height")
                        });
                        if parent_has_size == Some(true) {
                            // Size the svg to its parent's explicit box (a
                            // definite length in the editor assets) instead of
                            // using `percent(100)` — taffy resolves child
                            // percentages against the wrong containing block for
                            // inline-block parents, blowing the icon up to the
                            // full viewport.
                            let pw = ancestors
                                .last()
                                .and_then(|p| {
                                    let ps = crate::css::compute_style(
                                        p,
                                        stylesheets,
                                        &ancestors[..ancestors.len().saturating_sub(1)],
                                    );
                                    ps.get("width")
                                        .and_then(|w| parse_dimension(w, vw, vh, rem_base))
                                })
                                .unwrap_or(Dimension::percent(100.0));
                            let ph = ancestors
                                .last()
                                .and_then(|p| {
                                    let ps = crate::css::compute_style(
                                        p,
                                        stylesheets,
                                        &ancestors[..ancestors.len().saturating_sub(1)],
                                    );
                                    ps.get("height")
                                        .and_then(|h| parse_dimension(h, vw, vh, rem_base))
                                })
                                .unwrap_or(Dimension::percent(100.0));
                            taffy_style.size.width = pw;
                            taffy_style.size.height = ph;
                        } else if let Some((_mx, _my, vbw, vbh)) = Self::parse_view_box(&el.attrs) {
                            let (w, h) = Self::intrinsic_svg_size(vbw, vbh);
                            taffy_style.size.width = Dimension::length(w);
                            taffy_style.size.height = Dimension::length(h);
                        }
                    }
                }
                // `<img>` is a replaced element: its box uses explicit CSS
                // width/height when given, otherwise the image's intrinsic size,
                // or (browser-like) an aspect ratio derived from the intrinsic
                // size when only one dimension is set. The decoded image itself is
                // stashed on the node for the paint stage.
                let mut img_decoded: Option<DecodedImage> = None;
                let mut img_intrinsic: Option<(u32, u32)> = None;
                if let Some(src) = el.attrs.get("src") {
                    if let Some(decoded) = image_cache.get(src) {
                        let (iw, ih) = decoded.intrinsic_size();
                        img_intrinsic = Some((iw, ih));
                        img_decoded = Some((*decoded).clone());
                        let has_w = styles.contains_key("width");
                        let has_h = styles.contains_key("height");
                        if !has_w && !has_h {
                            // No CSS size: use intrinsic pixel size directly.
                            taffy_style.size.width = Dimension::length(iw as f32);
                            taffy_style.size.height = Dimension::length(ih as f32);
                        }
                        // Always advertize the intrinsic aspect ratio so taffy
                        // can derive the missing dimension from a CSS `width` or
                        // `height` (including percentage sizes, which must resolve
                        // against the *parent* box — not the viewport). Using
                        // `aspect_ratio` (instead of manually resolving a
                        // percentage against `vw`/`vh`) is what keeps a
                        // `height:100%` logo sized to its header rather than to
                        // the whole viewport.
                        taffy_style.aspect_ratio = Some(iw as f32 / ih as f32);
                    }
                }
                // `background-image: url(...)` / shorthand `background: ... url(...)`.
                // The decoded image is cached on the node and painted (clipped to
                // the box, sized per `background-size`) in the paint stage.
                let mut bg_decoded: Option<crate::image_loader::DecodedImage> = None;
                let mut bg_size: Option<String> = None;
                if let Some(bg_src) = styles
                    .get("background-image")
                    .and_then(|v| Self::extract_url(v))
                    .or_else(|| styles.get("background").and_then(|v| Self::extract_url(v)))
                {
                    if let Some(decoded) = image_cache.get(&bg_src) {
                        bg_decoded = Some((*decoded).clone());
                        bg_size = styles
                            .get("background-size")
                            .map(|v| v.trim().to_ascii_lowercase());
                    }
                }
                // that percentage sizes and block width-stretch resolve correctly.
                if is_root {
                    taffy_style.size.width = Dimension::length(vw);
                    taffy_style.size.height = Dimension::length(vh);
                }
                // `height/width: 100%` on a child of a flex container means "fill
                // the parent" -> express it as flex-grow (taffy doesn't resolve
                // percentage main-size against a flex parent).
                let parent_is_column = taffy_style.display == Display::Flex
                    && taffy_style.flex_direction == taffy::style::FlexDirection::Column;
                let parent_is_row = taffy_style.display == Display::Flex
                    && taffy_style.flex_direction == taffy::style::FlexDirection::Row;
                if parent_is_column
                    && styles
                        .get("height")
                        .map(|v| v.trim() == "100%")
                        .unwrap_or(false)
                {
                    taffy_style.flex_grow = 1.0;
                }
                if parent_is_row
                    && styles
                        .get("width")
                        .map(|v| v.trim() == "100%")
                        .unwrap_or(false)
                {
                    taffy_style.flex_grow = 1.0;
                }
                let mut child_taffy_ids = Vec::new();
                let mut child_arena_ids = Vec::new();
                for child in &el.children {
                    // `<head>` and its children (`<title>`, `<style>`, `<meta>`,
                    // `<link>`) are not rendered. `<script>`/`<noscript>`/`<template>`
                    // content must not appear as visible text either.
                    if let Node::Element(ce) = child {
                        if SKIP_TAGS.contains(&ce.tag.as_str()) {
                            continue;
                        }
                        // `<path>` inside `<svg>` is NOT a separate layout node:
                        // `extract_svg` (called for the parent `<svg>`) already
                        // lifts the path's `d`/`fill`/`viewBox` onto the `<svg>`
                        // node so paint draws the icon once. Creating a node for
                        // `<path>` too would double-draw the icon (bug b1).
                        if ce.tag == "path" {
                            continue;
                        }
                        let mut child_ancestors = ancestors.to_vec();
                        child_ancestors.push(el);
                        // Elements taken out of normal flow (`position:
                        // absolute`/`fixed`) or explicitly hidden (`display:
                        // none`) are not part of the flow layout. Skipping them
                        // keeps overlays (e.g. the command-palette) from blowing
                        // up the document box at large viewport sizes.
                        let cs = crate::css::compute_style(ce, stylesheets, &child_ancestors);
                        if Self::is_out_of_flow(&cs) {
                            continue;
                        }
                        let (ct, ca) = Self::build_node(
                            taffy,
                            child,
                            stylesheets,
                            arena,
                            taffy_to_arena,
                            &styles,
                            &child_ancestors,
                            vw,
                            vh,
                            false,
                            parent_is_column,
                            parent_is_row,
                            rem_base,
                            image_cache,
                        )?;
                        child_taffy_ids.push(ct);
                        child_arena_ids.push(ca);
                    } else {
                        // Whitespace-only text nodes are not rendered and must not
                        // become (anonymous) grid/flex items that disrupt placement.
                        if let Node::Text(t) = child {
                            if t.trim().is_empty() {
                                continue;
                            }
                        }
                        let (ct, ca) = Self::build_node(
                            taffy,
                            child,
                            stylesheets,
                            arena,
                            taffy_to_arena,
                            &styles,
                            ancestors,
                            vw,
                            vh,
                            false,
                            parent_is_column,
                            parent_is_row,
                            rem_base,
                            image_cache,
                        )?;
                        child_taffy_ids.push(ct);
                        child_arena_ids.push(ca);
                    }
                }

                let taffy_id = if child_taffy_ids.is_empty() {
                    taffy.new_leaf(taffy_style)?
                } else {
                    taffy.new_with_children(taffy_style, &child_taffy_ids)?
                };

                // Capture SVG icon geometry so `paint` can draw the path.
                let (svg_view_box, svg_path) = Self::extract_svg(el, &styles);

                let id = arena.len();
                let parent_chain: Vec<String> = ancestors
                    .iter()
                    .rev() // ancestors are parent-most-last; chain is root-first
                    .map(|a| {
                        format!(
                            "{}{}",
                            a.tag,
                            a.classes().into_iter().fold(String::new(), |mut s, c| {
                                s.push_str(&format!(".{}", c));
                                s
                            })
                        )
                    })
                    .collect();
                arena.push(LayoutNode {
                    id,
                    tag: el.tag.clone(),
                    dom_id: el.id().map(|s| s.to_string()),
                    dom_class: el.classes().into_iter().next().map(|s| s.to_string()),
                    rect: Rect::default(),
                    styles,
                    children: child_arena_ids.clone(),
                    parent: None,
                    text: None,
                    svg_view_box,
                    svg_path,
                    image: img_decoded,
                    image_intrinsic_size: img_intrinsic,
                    background_image: bg_decoded,
                    background_size: bg_size,
                    parent_chain,
                });
                for &ca in &child_arena_ids {
                    arena[ca].parent = Some(id);
                }
                taffy_to_arena.insert(taffy_id.into(), id);

                Ok((taffy_id, id))
            }
            Node::Text(text) => {
                let mut text_style = Style::default();
                text_style.display = Display::Block;
                // Width/height are resolved by taffy via the text-measure
                // function supplied to `compute_layout_with_measure`.
                let taffy_id = taffy.new_leaf(text_style)?;
                let id = arena.len();
                arena.push(LayoutNode {
                    id,
                    tag: "#text".into(),
                    dom_id: None,
                    dom_class: None,
                    rect: Rect::default(),
                    styles: inherited.clone(),
                    children: Vec::new(),
                    parent: None,
                    text: Some(text.clone()),
                    svg_view_box: None,
                    svg_path: None,
                    image: None,
                    image_intrinsic_size: None,
                    background_image: None,
                    background_size: None,
                    parent_chain: Vec::new(),
                });
                taffy_to_arena.insert(taffy_id.into(), id);
                Ok((taffy_id, id))
            }
        }
    }

    fn apply_layout(
        taffy: &TaffyTree,
        arena: &mut [LayoutNode],
        taffy_root: taffy::NodeId,
        taffy_to_arena: &HashMap<u64, LayoutNodeId>,
    ) {
        fn walk(
            taffy: &TaffyTree,
            arena: &mut [LayoutNode],
            taffy_id: taffy::NodeId,
            taffy_to_arena: &HashMap<u64, LayoutNodeId>,
            off_x: f32,
            off_y: f32,
        ) {
            let tkey: u64 = taffy_id.into();
            if let Some(&aid) = taffy_to_arena.get(&tkey) {
                if let Ok(layout) = taffy.layout(taffy_id) {
                    let abs_x = off_x + layout.location.x;
                    let abs_y = off_y + layout.location.y;
                    arena[aid].rect = Rect {
                        x: abs_x,
                        y: abs_y,
                        width: layout.size.width,
                        height: layout.size.height,
                    };
                    if let Ok(children) = taffy.children(taffy_id) {
                        for child_tid in children {
                            walk(taffy, arena, child_tid, taffy_to_arena, abs_x, abs_y);
                        }
                    }
                }
            }
        }

        walk(taffy, arena, taffy_root, taffy_to_arena, 0.0, 0.0);
    }
}

/// Determine the root element's computed `font-size` so that `rem` units
/// resolve correctly. Defaults to 16px when unspecified.
fn root_font_size(doc: &Document, stylesheets: &[Stylesheet]) -> f32 {
    // Prefer an explicit `:root` / `html` font-size rule — it drives every
    // `rem` size. Our DOM's document root may not be the literal `<html>`
    // element (the parser can drop <html>/<head>), so we look the rule up
    // directly in the stylesheets instead of relying on `compute_style` to
    // match `:root` against whatever element ended up at the top.
    for ss in stylesheets {
        for rule in &ss.rules {
            let targets_root = rule.selectors.iter().any(|sel| {
                sel.iter().any(|cp| {
                    cp.parts.iter().any(|p| {
                        matches!(p, SimpleSelector::Root)
                            || matches!(p, SimpleSelector::Tag(t) if t == "html")
                    })
                })
            });
            if targets_root {
                if let Some(fs) = rule.declarations.get("font-size") {
                    if let Some(px) = parse_length(fs, 16.0) {
                        return px;
                    }
                }
            }
        }
    }
    // Fallback: compute the style on whatever element is at the document root.
    fn root_element<'a>(node: &'a Node) -> Option<&'a crate::dom::Element> {
        match node {
            Node::Element(el) => {
                if el.tag == "html" {
                    return Some(el);
                }
                // The document root may be a <body> or a wrapper; descend to
                // the first element child to find <html>.
                for child in &el.children {
                    if let Some(f) = root_element(child) {
                        return Some(f);
                    }
                }
                Some(el)
            }
            Node::Text(_) => None,
        }
    }
    let root = root_element(&doc.root);
    root.map(|el| crate::css::compute_style(el, stylesheets, &[]))
        .and_then(|s| s.get("font-size").and_then(|v| parse_length(v, 16.0)))
        .unwrap_or(16.0)
}

fn css_to_taffy_style(styles: &HashMap<String, String>, vw: f32, vh: f32, rem_base: f32) -> Style {
    // taffy defaults `display` to `Flex` (row); for HTML we want the default to
    // be block flow.
    let mut s = Style::default();
    s.display = Display::Block;

    if let Some(w) = styles
        .get("width")
        .and_then(|v| parse_dimension(v, vw, vh, rem_base))
    {
        s.size.width = w;
    }
    if let Some(h) = styles
        .get("height")
        .and_then(|v| parse_dimension(v, vw, vh, rem_base))
    {
        s.size.height = h;
    }
    if let Some(m) = styles
        .get("min-height")
        .and_then(|v| parse_dimension(v, vw, vh, rem_base))
    {
        s.min_size = Size {
            width: Dimension::auto(),
            height: m,
        };
    }
    if let Some(m) = styles
        .get("min-width")
        .and_then(|v| parse_dimension(v, vw, vh, rem_base))
    {
        s.min_size.width = m;
    }
    if let Some(m) = styles
        .get("max-width")
        .and_then(|v| parse_dimension(v, vw, vh, rem_base))
    {
        s.max_size.width = m;
    }
    if let Some(m) = styles
        .get("max-height")
        .and_then(|v| parse_dimension(v, vw, vh, rem_base))
    {
        s.max_size.height = m;
    }
    let parse_margin_len = |v: &str| -> Option<LengthPercentageAuto> {
        let v = v.trim();
        if v == "auto" {
            Some(LengthPercentageAuto::auto())
        } else {
            parse_length(v, rem_base).map(LengthPercentageAuto::length)
        }
    };
    if let Some(m) = styles.get("margin").and_then(|v| parse_margin_len(v)) {
        s.margin = taffy::geometry::Rect {
            left: m,
            right: m,
            top: m,
            bottom: m,
        };
    }
    let ml = styles.get("margin-left").and_then(|v| parse_margin_len(v));
    let mr = styles.get("margin-right").and_then(|v| parse_margin_len(v));
    let mt = styles
        .get("margin-top")
        .and_then(|v| parse_length(v, rem_base))
        .map(LengthPercentageAuto::length);
    let mb = styles
        .get("margin-bottom")
        .and_then(|v| parse_length(v, rem_base))
        .map(LengthPercentageAuto::length);
    if ml.is_some() || mr.is_some() || mt.is_some() || mb.is_some() {
        s.margin = taffy::geometry::Rect {
            left: ml.unwrap_or(s.margin.left),
            right: mr.unwrap_or(s.margin.right),
            top: mt.unwrap_or(s.margin.top),
            bottom: mb.unwrap_or(s.margin.bottom),
        };
    }
    // `padding` shorthand (1-4 values) plus per-side overrides. Missing sides
    // default to 0. Previously only a single-value `padding` was honored, so
    // `padding: 0.1rem 0.5rem` (and directional props) were silently dropped —
    // that lost every panel's inner spacing and collapsed content upward.
    {
        let mut top = 0.0_f32;
        let mut right = 0.0_f32;
        let mut bottom = 0.0_f32;
        let mut left = 0.0_f32;
        let mut has_padding = false;
        if let Some((t, r, b, l)) = styles
            .get("padding")
            .and_then(|v| parse_rect_shorthand(v, rem_base))
        {
            top = t;
            right = r;
            bottom = b;
            left = l;
            has_padding = true;
        }
        if let Some(v) = styles
            .get("padding-top")
            .and_then(|v| parse_length(v, rem_base))
        {
            top = v;
            has_padding = true;
        }
        if let Some(v) = styles
            .get("padding-right")
            .and_then(|v| parse_length(v, rem_base))
        {
            right = v;
            has_padding = true;
        }
        if let Some(v) = styles
            .get("padding-bottom")
            .and_then(|v| parse_length(v, rem_base))
        {
            bottom = v;
            has_padding = true;
        }
        if let Some(v) = styles
            .get("padding-left")
            .and_then(|v| parse_length(v, rem_base))
        {
            left = v;
            has_padding = true;
        }
        if has_padding {
            s.padding = taffy::geometry::Rect {
                left: LengthPercentage::length(left),
                right: LengthPercentage::length(right),
                top: LengthPercentage::length(top),
                bottom: LengthPercentage::length(bottom),
            };
        }
    }
    if let Some(d) = styles.get("display") {
        if d == "none" {
            s.display = Display::None;
        } else if d == "flex" {
            s.display = Display::Flex;
        } else if d == "grid" {
            s.display = Display::Grid;
        }
    }

    // CSS Grid.
    if let Some(v) = styles.get("grid-template-columns") {
        s.grid_template_columns = parse_grid_tracks(v, rem_base);
    }
    if let Some(v) = styles.get("grid-template-rows") {
        s.grid_template_rows = parse_grid_tracks(v, rem_base);
    }
    if let Some(v) = styles.get("grid-column") {
        s.grid_column = parse_grid_line(v);
    }
    if let Some(v) = styles.get("grid-row") {
        s.grid_row = parse_grid_line(v);
    }

    // Absolute positioning.
    if let Some(p) = styles.get("position") {
        if p == "absolute" {
            s.position = Position::Absolute;
        } else if p == "relative" {
            s.position = Position::Relative;
        }
    }
    let mut inset = taffy::geometry::Rect {
        top: LengthPercentageAuto::length(0.0),
        bottom: LengthPercentageAuto::length(0.0),
        left: LengthPercentageAuto::length(0.0),
        right: LengthPercentageAuto::length(0.0),
    };
    let mut has_inset = false;
    for (prop, field) in [("top", 0), ("bottom", 1), ("left", 2), ("right", 3)] {
        if let Some(v) = styles
            .get(prop)
            .and_then(|v| parse_length_or_percent(v, rem_base))
        {
            let lp = match v {
                InsetVal::Length(l) => LengthPercentageAuto::length(l),
                InsetVal::Percent(p) => LengthPercentageAuto::percent(p),
            };
            match field {
                0 => inset.top = lp,
                1 => inset.bottom = lp,
                2 => inset.left = lp,
                _ => inset.right = lp,
            }
            has_inset = true;
        }
    }
    if has_inset {
        s.inset = inset;
    }

    if let Some(v) = styles.get("align-items") {
        s.align_items = parse_align_items(v);
    }

    if let Some(v) = styles.get("justify-content") {
        s.justify_content = parse_justify_content(v);
    }

    if let Some(v) = styles.get("gap").and_then(|v| parse_length(v, rem_base)) {
        s.gap = Size {
            width: LengthPercentage::length(v),
            height: LengthPercentage::length(v),
        };
    }

    if let Some(d) = styles.get("flex-direction") {
        if d == "column" {
            s.flex_direction = taffy::style::FlexDirection::Column;
        } else if d == "row-reverse" {
            s.flex_direction = taffy::style::FlexDirection::RowReverse;
        } else if d == "column-reverse" {
            s.flex_direction = taffy::style::FlexDirection::ColumnReverse;
        }
    }

    if let Some(g) = styles
        .get("flex-grow")
        .and_then(|v| v.trim().parse::<f32>().ok())
        .or_else(|| parse_flex_grow(styles.get("flex").map(|s| s.as_str())))
    {
        s.flex_grow = g;
    }

    s
}

fn parse_align_items(v: &str) -> Option<AlignItems> {
    match v.trim() {
        "center" => Some(AlignItems::Center),
        "flex-start" | "start" => Some(AlignItems::FlexStart),
        "flex-end" | "end" => Some(AlignItems::FlexEnd),
        "stretch" => Some(AlignItems::Stretch),
        "baseline" => Some(AlignItems::Baseline),
        _ => None,
    }
}

fn parse_justify_content(v: &str) -> Option<JustifyContent> {
    match v.trim() {
        "center" => Some(JustifyContent::Center),
        "flex-start" | "start" => Some(JustifyContent::FlexStart),
        "flex-end" | "end" => Some(JustifyContent::FlexEnd),
        "space-between" => Some(JustifyContent::SpaceBetween),
        "space-around" => Some(JustifyContent::SpaceAround),
        "space-evenly" => Some(JustifyContent::SpaceEvenly),
        _ => None,
    }
}

/// Parse a CSS length/percentage/viewport unit into a taffy `Dimension`.
fn parse_dimension(s: &str, vw: f32, vh: f32, rem_base: f32) -> Option<Dimension> {
    let s = s.trim();
    if s == "auto" {
        return None;
    }
    if let Some(pct) = s.strip_suffix('%') {
        return pct
            .trim()
            .parse::<f32>()
            .ok()
            .map(|v| Dimension::percent(v / 100.0));
    }
    if let Some(px) = s.strip_suffix("px") {
        return px.trim().parse::<f32>().ok().map(Dimension::length);
    }
    if let Some(v) = s.strip_suffix("vh") {
        return v
            .trim()
            .parse::<f32>()
            .ok()
            .map(|v| Dimension::length(v / 100.0 * vh));
    }
    if let Some(v) = s.strip_suffix("vw") {
        return v
            .trim()
            .parse::<f32>()
            .ok()
            .map(|v| Dimension::length(v / 100.0 * vw));
    }
    if let Some(rem) = s.strip_suffix("rem") {
        return rem
            .trim()
            .parse::<f32>()
            .ok()
            .map(|v| Dimension::length(v * rem_base));
    }
    s.parse::<f32>().ok().map(Dimension::length)
}

/// Extract `flex-grow` from a `flex` shorthand, e.g. `flex: 1` or `flex: 1 1 0%`.
fn parse_flex_grow(v: Option<&str>) -> Option<f32> {
    let v = v?.trim();
    if v == "none" || v.is_empty() {
        return None;
    }
    let first = v.split_whitespace().next()?;
    first.parse::<f32>().ok()
}

fn parse_length(s: &str, rem_base: f32) -> Option<f32> {
    let s = s.trim();
    if let Some(px) = s.strip_suffix("px") {
        px.trim().parse::<f32>().ok()
    } else if let Some(rem) = s.strip_suffix("rem") {
        rem.trim().parse::<f32>().ok().map(|v| v * rem_base)
    } else {
        s.parse::<f32>().ok()
    }
}

/// Parse a CSS box shorthand (`padding`/`margin`) of 1-4 space-separated
/// lengths into `(top, right, bottom, left)` per the CSS spec:
///   1 value  -> all four sides
///   2 values -> vertical | horizontal
///   3 values -> top | horizontal | bottom
///   4 values -> top | right | bottom | left
/// Returns `None` if the string is empty or any token fails to parse.
fn parse_rect_shorthand(s: &str, rem_base: f32) -> Option<(f32, f32, f32, f32)> {
    let parts: Vec<f32> = s
        .split_whitespace()
        .map(|tok| parse_length(tok, rem_base))
        .collect::<Option<Vec<f32>>>()?;
    match parts.as_slice() {
        [a] => Some((*a, *a, *a, *a)),
        [v, h] => Some((*v, *h, *v, *h)),
        [t, h, b] => Some((*t, *h, *b, *h)),
        [t, r, b, l] => Some((*t, *r, *b, *l)),
        _ => None,
    }
}

/// A parsed `top`/`left`/`right`/`bottom` inset value.
enum InsetVal {
    Length(f32),
    Percent(f32),
}

/// Parse an inset value (`12px`, `1rem`, `50%`) into a concrete value.
fn parse_length_or_percent(s: &str, rem_base: f32) -> Option<InsetVal> {
    let s = s.trim();
    if let Some(pct) = s.strip_suffix('%') {
        return pct
            .trim()
            .parse::<f32>()
            .ok()
            .map(|v| InsetVal::Percent(v / 100.0));
    }
    if let Some(px) = s.strip_suffix("px") {
        return px.trim().parse::<f32>().ok().map(InsetVal::Length);
    }
    if let Some(rem) = s.strip_suffix("rem") {
        return rem
            .trim()
            .parse::<f32>()
            .ok()
            .map(|v| InsetVal::Length(v * rem_base));
    }
    s.parse::<f32>().ok().map(InsetVal::Length)
}

/// Parse a `grid-template-columns` / `grid-template-rows` track list.
fn parse_grid_tracks(s: &str, rem_base: f32) -> Vec<TrackSizingFunction> {
    let mut tracks = Vec::new();
    for token in s.split_whitespace() {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        if let Some(fr) = token.strip_suffix("fr") {
            if let Ok(f) = fr.trim().parse::<f32>() {
                // `1fr` in CSS is `minmax(auto, 1fr)`. With our imperfect
                // content min-size measurement (e.g. `width: 100%` grid items)
                // the `auto` minimum can balloon and overflow the container.
                // `flex(f)` == `minmax(0, Nfr)` so a flexible track never grows
                // past the available space due to its items' min-content.
                tracks.push(flex(f));
                continue;
            }
        }
        if token == "auto" {
            tracks.push(TrackSizingFunction::AUTO);
            continue;
        }
        if let Some(pct) = token.strip_suffix('%') {
            if let Ok(p) = pct.trim().parse::<f32>() {
                tracks.push(TrackSizingFunction::from_percent(p / 100.0));
                continue;
            }
        }
        if let Some(px) = token.strip_suffix("px") {
            if let Ok(p) = px.trim().parse::<f32>() {
                tracks.push(TrackSizingFunction::from_length(p));
                continue;
            }
        }
        if let Some(rem) = token.strip_suffix("rem") {
            if let Ok(r) = rem.trim().parse::<f32>() {
                tracks.push(TrackSizingFunction::from_length(r * rem_base));
                continue;
            }
        }
        if let Ok(n) = token.parse::<f32>() {
            tracks.push(TrackSizingFunction::from_length(n));
        }
    }
    tracks
}

/// Parse a single grid line placement (`auto`, `span N`, or a line index).
fn parse_single_placement(s: &str) -> GridPlacement {
    let s = s.trim();
    if s == "auto" {
        return GridPlacement::Auto;
    }
    if let Some(span) = s.strip_prefix("span") {
        if let Ok(n) = span.trim().parse::<u16>() {
            return GridPlacement::Span(n);
        }
    }
    if let Ok(n) = s.parse::<i16>() {
        return GridPlacement::from_line_index(n);
    }
    GridPlacement::Auto
}

/// Parse a `grid-column` / `grid-row` value, e.g. `span 3` or `1 / 3`.
fn parse_grid_line(s: &str) -> Line<GridPlacement> {
    let s = s.trim();
    if let Some((start, end)) = s.split_once('/') {
        return Line {
            start: parse_single_placement(start),
            end: parse_single_placement(end),
        };
    }
    if s.starts_with("span") {
        if let Some(span) = s.strip_prefix("span") {
            if let Ok(n) = span.trim().parse::<u16>() {
                return Line::from_span(n);
            }
        }
    }
    if let Ok(n) = s.parse::<i16>() {
        return Line::from_line_index(n);
    }
    if s == "auto" {
        return Line {
            start: GridPlacement::Auto,
            end: GridPlacement::Auto,
        };
    }
    Line {
        start: GridPlacement::Auto,
        end: GridPlacement::Auto,
    }
}

#[cfg(test)]
mod layout_probe {
    use super::*;
    use crate::editor_template::EditorTemplate;
    use crate::html::parse_html;
    use crate::unified_editor::UnifiedEditorConfig;
    use std::io::Read;

    fn load_font() -> FontData {
        // Use the bundled Inter font (falls back to system fonts if missing).
        crate::text::load_inter_font()
    }

    #[test]
    fn probe_real_editor_layout() {
        let config = UnifiedEditorConfig::default();
        let html = EditorTemplate::generate_html_with_theme(&config);
        let doc = parse_html(&html);
        let css = EditorTemplate::generate_css_with_theme(&config);
        let sheet = crate::css::Stylesheet::parse(&css).expect("css parses");
        let font = load_font();

        let tree = LayoutTree::build_with_viewport(&doc, &[sheet], 1280.0, 800.0, &font)
            .expect("layout builds");

        println!("\n=== ALL LAID-OUT NODES (1280x800) ===");
        for n in &tree.arena {
            let label = if let Some(id) = &n.dom_id {
                format!("#{}", id)
            } else if let Some(cls) = &n.dom_class {
                format!(".{}", cls)
            } else {
                format!("<{}>", n.tag)
            };
            let pos = n.styles.get("position").cloned().unwrap_or_default();
            let disp = n.styles.get("display").cloned().unwrap_or_default();
            let w = n.styles.get("width").cloned().unwrap_or_default();
            let h = n.styles.get("height").cloned().unwrap_or_default();
            println!(
                "{:24} x={:7.1} y={:7.1} w={:7.1} h={:7.1}  pos={:9} disp={:10} w={:6} h={:6} svg={}",
                label,
                n.rect.x,
                n.rect.y,
                n.rect.width,
                n.rect.height,
                pos,
                disp,
                w,
                h,
                n.svg_path.is_some()
            );
        }

        // Soft checks (do not panic on missing wrapper nodes like .app which
        // may be display:contents).
        let by_class: std::collections::HashMap<_, _> = tree
            .arena
            .iter()
            .map(|n| {
                let k = n
                    .dom_class
                    .clone()
                    .or_else(|| n.dom_id.clone().map(|i| format!("#{}", i)))
                    .unwrap_or_else(|| n.tag.clone());
                (k, n.rect)
            })
            .collect();

        if let Some(vp) = by_class.get("viewport") {
            println!(
                "\nCHECK viewport: x={:.1} w={:.1} (center grid col ~342/596)",
                vp.x, vp.width
            );
            assert!(
                vp.width > 100.0 && vp.height > 100.0,
                "viewport has real size"
            );
        }
        if let Some(right) = by_class.get("right") {
            println!(
                "CHECK right: x={:.1} y={:.1} (absolute, near 1242/780)",
                right.x, right.y
            );
        }
        println!("\nProbe complete.");
    }
}
