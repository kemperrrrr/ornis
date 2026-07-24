use std::collections::HashMap;
use std::time::Instant;

use crate::layout::{LayoutNodeId, LayoutTree};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// Mutable text state for an `<input>` or `<textarea>` node.
#[derive(Debug, Clone, Default)]
pub struct TextInputState {
    pub value: String,
    pub caret: usize,
}

impl TextInputState {
    pub fn with_value(value: String) -> Self {
        let caret = value.len();
        Self { value, caret }
    }

    pub fn insert(&mut self, text: &str) {
        self.value.insert_str(self.caret, text);
        self.caret += text.len();
    }

    pub fn backspace(&mut self) {
        if self.caret == 0 {
            return;
        }
        let prev = self.value[..self.caret]
            .char_indices()
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.value.replace_range(prev..self.caret, "");
        self.caret = prev;
    }

    pub fn delete(&mut self) {
        if self.caret >= self.value.len() {
            return;
        }
        let next = self.value[self.caret..]
            .char_indices()
            .nth(1)
            .map(|(i, _)| self.caret + i)
            .unwrap_or(self.value.len());
        self.value.replace_range(self.caret..next, "");
    }

    pub fn move_caret(&mut self, delta: isize) {
        if delta < 0 {
            let n = delta.abs() as usize;
            for _ in 0..n {
                if self.caret == 0 {
                    break;
                }
                self.caret = self.value[..self.caret]
                    .char_indices()
                    .last()
                    .map(|(i, _)| i)
                    .unwrap_or(0);
            }
        } else {
            let n = delta as usize;
            for _ in 0..n {
                if self.caret >= self.value.len() {
                    break;
                }
                let c = self.value[self.caret..].chars().next().unwrap();
                self.caret += c.len_utf8();
            }
        }
    }

    pub fn move_caret_home(&mut self) {
        self.caret = 0;
    }

    pub fn move_caret_end(&mut self) {
        self.caret = self.value.len();
    }
}

/// Tracks all interactive state: hover, focus, clicks, scroll, text input.
#[derive(Debug, Clone, Default)]
pub struct InteractionState {
    pub hovered: Option<LayoutNodeId>,
    pub focused: Option<LayoutNodeId>,
    pub mouse_down: bool,
    pub mouse_down_on: Option<LayoutNodeId>,
    pub last_mouse_pos: (f32, f32),
    /// Scroll offsets per scrollable node: (scroll_x, scroll_y) in CSS px.
    pub scroll_offsets: HashMap<LayoutNodeId, (f32, f32)>,
    /// `<details>` open state per sequential index (stable across layout rebuilds).
    pub details_open: HashMap<usize, bool>,
    /// Text input state per node.
    pub text_input: HashMap<LayoutNodeId, TextInputState>,
    /// Last click instant (for future double-click detection).
    pub last_click_time: Option<Instant>,
}

impl InteractionState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Handle mouse movement. Returns `true` if the hover target changed.
    pub fn handle_mouse_move(&mut self, tree: &LayoutTree, x: f32, y: f32) -> bool {
        self.last_mouse_pos = (x, y);
        let new_hover = tree.hit_test(x, y);
        let changed = self.hovered != new_hover;
        self.hovered = new_hover;
        changed
    }

    pub fn handle_mouse_down(
        &mut self,
        tree: &LayoutTree,
        x: f32,
        y: f32,
        _button: MouseButton,
    ) {
        self.last_mouse_pos = (x, y);
        self.mouse_down = true;
        if let Some(node_id) = tree.hit_test(x, y) {
            self.mouse_down_on = Some(node_id);
            self.focused = Some(node_id);
        } else {
            self.mouse_down_on = None;
            self.focused = None;
        }
    }

    /// Handle mouse up. Returns `Some(node_id)` when a click occurred
    /// (mouse-down and mouse-up on the same element).
    pub fn handle_mouse_up(
        &mut self,
        tree: &LayoutTree,
        x: f32,
        y: f32,
        _button: MouseButton,
    ) -> Option<LayoutNodeId> {
        self.last_mouse_pos = (x, y);
        self.mouse_down = false;
        let was_down_on = self.mouse_down_on.take();
        let now_over = tree.hit_test(x, y);
        if was_down_on == now_over {
            was_down_on
        } else {
            None
        }
    }

    pub fn toggle_details(&mut self, details_index: usize) {
        let open = self.details_open.entry(details_index).or_insert(true);
        *open = !*open;
    }

    pub fn is_details_open(&self, details_index: usize) -> bool {
        self.details_open.get(&details_index).copied().unwrap_or(true)
    }

    /// Returns the list of `<details>` indices that are currently closed.
    pub fn closed_details_indices(&self) -> Vec<usize> {
        self.details_open
            .iter()
            .filter(|&(_, open)| !open)
            .map(|(&idx, _)| idx)
            .collect()
    }

    pub fn scroll_node(&mut self, node_id: LayoutNodeId, delta_x: f32, delta_y: f32) {
        let entry = self.scroll_offsets.entry(node_id).or_insert((0.0, 0.0));
        entry.0 = (entry.0 + delta_x).max(0.0);
        entry.1 = (entry.1 + delta_y).max(0.0);
    }

    pub fn get_scroll_offset(&self, node_id: LayoutNodeId) -> (f32, f32) {
        self.scroll_offsets.get(&node_id).copied().unwrap_or((0.0, 0.0))
    }

    pub fn text_input_mut(&mut self, node_id: LayoutNodeId) -> &mut TextInputState {
        self.text_input.entry(node_id).or_default()
    }

    pub fn text_input_ref(&self, node_id: LayoutNodeId) -> Option<&TextInputState> {
        self.text_input.get(&node_id)
    }

    pub fn set_input_value(&mut self, node_id: LayoutNodeId, value: String) {
        self.text_input.insert(node_id, TextInputState::with_value(value));
    }
}
