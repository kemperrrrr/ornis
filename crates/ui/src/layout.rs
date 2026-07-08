use std::collections::HashMap;

use taffy::{AlignItems, AvailableSpace, JustifyContent, Size, TaffyResult, TaffyTree};
use taffy::style::{Dimension, Display, LengthPercentage, LengthPercentageAuto, Style};

use crate::css::Stylesheet;
use crate::dom::{Document, Node};

pub type LayoutNodeId = usize;

#[derive(Debug, Clone)]
pub struct LayoutTree {
    pub arena: Vec<LayoutNode>,
    pub root: LayoutNodeId,
}

#[derive(Debug, Clone)]
pub struct LayoutNode {
    pub id: LayoutNodeId,
    pub tag: String,
    pub rect: Rect,
    pub styles: HashMap<String, String>,
    pub children: Vec<LayoutNodeId>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl LayoutTree {
    pub fn build(doc: &Document, stylesheets: &[Stylesheet]) -> TaffyResult<Self> {
        Self::build_with_viewport(doc, stylesheets, 1024.0, 768.0)
    }

    pub fn build_with_viewport(
        doc: &Document,
        stylesheets: &[Stylesheet],
        viewport_width: f32,
        viewport_height: f32,
    ) -> TaffyResult<Self> {
        let mut taffy = TaffyTree::new();
        let mut arena = Vec::<LayoutNode>::new();
        let mut taffy_to_arena: HashMap<u64, LayoutNodeId> = HashMap::new();
        let (root_tid, _root_aid) = Self::build_node(&mut taffy, &doc.root, stylesheets, &mut arena, &mut taffy_to_arena)?;
        taffy.compute_layout(
            root_tid,
            Size {
                width: AvailableSpace::Definite(viewport_width),
                height: AvailableSpace::Definite(viewport_height),
            },
        )?;

        Self::apply_layout(&taffy, &mut arena, root_tid, &taffy_to_arena);

        Ok(LayoutTree { arena, root: 0 })
    }

    fn build_node(
        taffy: &mut TaffyTree,
        node: &Node,
        stylesheets: &[Stylesheet],
        arena: &mut Vec<LayoutNode>,
        taffy_to_arena: &mut HashMap<u64, LayoutNodeId>,
    ) -> TaffyResult<(taffy::NodeId, LayoutNodeId)> {
        match node {
            Node::Element(el) => {
                let styles = crate::css::compute_style(el, stylesheets);
                let taffy_style = css_to_taffy_style(&styles);
                let mut child_taffy_ids = Vec::new();
                let mut child_arena_ids = Vec::new();
                for child in &el.children {
                    let (ct, ca) = Self::build_node(taffy, child, stylesheets, arena, taffy_to_arena)?;
                    child_taffy_ids.push(ct);
                    child_arena_ids.push(ca);
                }

                let taffy_id = if child_taffy_ids.is_empty() {
                    taffy.new_leaf(taffy_style)?
                } else {
                    taffy.new_with_children(taffy_style, &child_taffy_ids)?
                };

                let id = arena.len();
                arena.push(LayoutNode {
                    id,
                    tag: el.tag.clone(),
                    rect: Rect::default(),
                    styles,
                    children: child_arena_ids,
                });
                taffy_to_arena.insert(taffy_id.into(), id);

                Ok((taffy_id, id))
            }
            Node::Text(_) => {
                let taffy_id = taffy.new_leaf(Style::default())?;
                let id = arena.len();
                arena.push(LayoutNode {
                    id,
                    tag: "#text".into(),
                    rect: Rect::default(),
                    styles: HashMap::new(),
                    children: Vec::new(),
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
        ) {
            let tkey: u64 = taffy_id.into();
            if let Some(&aid) = taffy_to_arena.get(&tkey) {
                if let Ok(layout) = taffy.layout(taffy_id) {
                    arena[aid].rect = Rect {
                        x: layout.location.x,
                        y: layout.location.y,
                        width: layout.size.width,
                        height: layout.size.height,
                    };
                }
                if let Ok(children) = taffy.children(taffy_id) {
                    for child_tid in children {
                        walk(taffy, arena, child_tid, taffy_to_arena);
                    }
                }
            }
        }

        walk(taffy, arena, taffy_root, taffy_to_arena);
    }
}

fn css_to_taffy_style(styles: &HashMap<String, String>) -> Style {
    let mut s = Style::default();

    if let Some(w) = styles.get("width").and_then(|v| parse_length(v).map(Dimension::Length)) {
        s.size.width = w;
    }
    if let Some(h) = styles.get("height").and_then(|v| parse_length(v).map(Dimension::Length)) {
        s.size.height = h;
    }
    if let Some(m) = styles.get("margin").and_then(|v| parse_length(v)) {
        s.margin = taffy::geometry::Rect {
            left: LengthPercentageAuto::Length(m),
            right: LengthPercentageAuto::Length(m),
            top: LengthPercentageAuto::Length(m),
            bottom: LengthPercentageAuto::Length(m),
        };
    }
    if let Some(p) = styles.get("padding").and_then(|v| parse_length(v)) {
        s.padding = taffy::geometry::Rect {
            left: LengthPercentage::Length(p),
            right: LengthPercentage::Length(p),
            top: LengthPercentage::Length(p),
            bottom: LengthPercentage::Length(p),
        };
    }
    if let Some(d) = styles.get("display") {
        if d == "none" {
            s.display = Display::None;
        } else if d == "flex" {
            s.display = Display::Flex;
        }
    }

    if let Some(v) = styles.get("align-items") {
        s.align_items = parse_align_items(v);
    }

    if let Some(v) = styles.get("justify-content") {
        s.justify_content = parse_justify_content(v);
    }

    if let Some(v) = styles.get("gap").and_then(|v| parse_length(v)) {
        s.gap = Size { width: LengthPercentage::Length(v), height: LengthPercentage::Length(v) };
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

fn parse_length(s: &str) -> Option<f32> {
    let s = s.trim();
    if let Some(px) = s.strip_suffix("px") {
        px.trim().parse::<f32>().ok()
    } else if let Some(rem) = s.strip_suffix("rem") {
        rem.trim().parse::<f32>().ok().map(|v| v * 16.0)
    } else {
        s.parse::<f32>().ok()
    }
}
