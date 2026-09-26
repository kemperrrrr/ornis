//! Action mapping: named gameplay actions over raw input codes.
//!
//! Producers keep writing raw codes into [`crate::InputState`]; an
//! [`InputMap`] binds actions (for example `"move_forward"`) to sets of
//! keys and mouse buttons, and consumers query
//! [`InputMap::action_down`] on the read edge. Remapping never touches the
//! wire format.

use std::collections::HashMap;

use crate::{InputState, KeyCode, MouseButton};

/// Typed action identifier: the [`InputMap`] key.
///
/// Owned (`Box<str>`) so static gameplay names and dynamic tool/test
/// actions share one type. Accepts plain strings (`impl Into<ActionId>` on
/// the write edge) and serves `&str` lookups on the read edge through the
/// [`std::borrow::Borrow`] impl, so existing `bind("jump", …)` /
/// `action_down(&input, "jump")` call sites keep compiling while new code
/// can pass [`GameAction`] or `ActionId` instead of a bare `String`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ActionId(Box<str>);

impl ActionId {
    /// Wraps a `'static` action name without extra bookkeeping.
    pub fn from_static(name: &'static str) -> Self {
        Self(name.into())
    }

    /// Action name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::borrow::Borrow<str> for ActionId {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for ActionId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for ActionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for ActionId {
    fn from(name: &str) -> Self {
        Self(name.into())
    }
}

impl From<String> for ActionId {
    fn from(name: String) -> Self {
        Self(name.into_boxed_str())
    }
}

impl From<&String> for ActionId {
    fn from(name: &String) -> Self {
        Self(name.as_str().into())
    }
}

/// Closed gameplay-action vocabulary over [`ActionId`].
///
/// The four movement directions are first-class variants (matching
/// [`InputMap::default_gameplay`]); tools and tests keep their own words
/// through [`GameAction::Custom`]. Converts into [`ActionId`] for the map,
/// so typed and stringly call sites interoperate.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GameAction {
    /// Forward intent (`W` / Up by default).
    MoveForward,
    /// Back intent (`S` / Down by default).
    MoveBack,
    /// Strafe-left intent (`A` / Left by default).
    MoveLeft,
    /// Strafe-right intent (`D` / Right by default).
    MoveRight,
    /// Tool/test-defined action.
    Custom(ActionId),
}

impl GameAction {
    /// Canonical [`ActionId`] of this action.
    pub fn id(&self) -> ActionId {
        ActionId::from(self.as_str())
    }

    /// Canonical action name (`"move_forward"`, …, or the custom name).
    pub fn as_str(&self) -> &str {
        match self {
            GameAction::MoveForward => InputMap::MOVE_FORWARD,
            GameAction::MoveBack => InputMap::MOVE_BACK,
            GameAction::MoveLeft => InputMap::MOVE_LEFT,
            GameAction::MoveRight => InputMap::MOVE_RIGHT,
            GameAction::Custom(id) => id.as_str(),
        }
    }
}

impl From<GameAction> for ActionId {
    fn from(action: GameAction) -> Self {
        action.id()
    }
}

impl From<&GameAction> for ActionId {
    fn from(action: &GameAction) -> Self {
        action.id()
    }
}

/// Raw-code set bound to one action: any held entry fires the action.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputBinding {
    keys: Vec<u32>,
    mouse_buttons: Vec<u8>,
}

impl InputBinding {
    /// Creates an empty binding that matches nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds named keys, expanding each into its raw aliases.
    pub fn with_keys(keys: impl IntoIterator<Item = KeyCode>) -> Self {
        let mut binding = Self::new();
        binding.add_keys(keys);
        binding
    }

    /// Binds raw key codes (for wire-edge mappings without named keys).
    pub fn with_key_codes(codes: impl IntoIterator<Item = u32>) -> Self {
        let mut binding = Self::new();
        binding.add_key_codes(codes);
        binding
    }

    /// Binds named mouse buttons.
    pub fn with_mouse(buttons: impl IntoIterator<Item = MouseButton>) -> Self {
        let mut binding = Self::new();
        binding.add_mouse(buttons);
        binding
    }

    /// Adds named keys (raw-alias expansion) to this binding.
    pub fn add_keys(&mut self, keys: impl IntoIterator<Item = KeyCode>) -> &mut Self {
        for key in keys {
            self.add_key_codes(key.codes().iter().copied());
        }
        self
    }

    /// Adds raw key codes to this binding.
    pub fn add_key_codes(&mut self, codes: impl IntoIterator<Item = u32>) -> &mut Self {
        self.keys.extend(codes);
        self.keys.sort_unstable();
        self.keys.dedup();
        self
    }

    /// Adds named mouse buttons to this binding.
    pub fn add_mouse(&mut self, buttons: impl IntoIterator<Item = MouseButton>) -> &mut Self {
        for button in buttons {
            let code = button.code();
            if !self.mouse_buttons.contains(&code) {
                self.mouse_buttons.push(code);
            }
        }
        self.mouse_buttons.sort_unstable();
        self
    }

    /// Raw key codes in this binding, sorted ascending and deduplicated.
    pub fn keys(&self) -> &[u32] {
        &self.keys
    }

    /// Raw mouse-button codes in this binding, sorted ascending.
    pub fn mouse_buttons(&self) -> &[u8] {
        &self.mouse_buttons
    }

    /// Whether any bound key or button is currently held in `input`.
    pub fn is_down(&self, input: &InputState) -> bool {
        self.keys.iter().any(|code| input.key_down(*code))
            || self
                .mouse_buttons
                .iter()
                .any(|code| input.mouse_button_down(*code))
    }
}

/// Named-action table over raw input codes.
///
/// Actions are keyed by [`ActionId`] (constructible from plain strings,
/// so gameplay, tools and tests keep introducing their own vocabulary);
/// the canonical movement names live on the associated constants,
/// [`GameAction`] and [`InputMap::default_gameplay`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputMap {
    bindings: HashMap<ActionId, InputBinding>,
}

impl InputMap {
    /// Forward intent (`W` / Up by default).
    pub const MOVE_FORWARD: &'static str = "move_forward";
    /// Back intent (`S` / Down by default).
    pub const MOVE_BACK: &'static str = "move_back";
    /// Strafe-left intent (`A` / Left by default).
    pub const MOVE_LEFT: &'static str = "move_left";
    /// Strafe-right intent (`D` / Right by default).
    pub const MOVE_RIGHT: &'static str = "move_right";

    /// Creates an empty map where every action query returns `false`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Default gameplay bindings: WASD plus arrows on the four movement
    /// actions. Matches the legacy `player_input` key sets exactly.
    pub fn default_gameplay() -> Self {
        let mut map = Self::new();
        map.bind_keys(Self::MOVE_FORWARD, [KeyCode::KeyW, KeyCode::ArrowUp]);
        map.bind_keys(Self::MOVE_BACK, [KeyCode::KeyS, KeyCode::ArrowDown]);
        map.bind_keys(Self::MOVE_LEFT, [KeyCode::KeyA, KeyCode::ArrowLeft]);
        map.bind_keys(Self::MOVE_RIGHT, [KeyCode::KeyD, KeyCode::ArrowRight]);
        map
    }

    /// Binds `action` to `binding`, replacing any previous binding.
    ///
    /// Accepts [`ActionId`], [`GameAction`] or plain strings
    /// (`impl Into<ActionId>`), so existing `bind("jump", …)` call sites
    /// keep compiling.
    pub fn bind(&mut self, action: impl Into<ActionId>, binding: InputBinding) -> &mut Self {
        self.bindings.insert(action.into(), binding);
        self
    }

    /// Binds `action` to named keys (raw-alias expansion), replacing any
    /// previous binding.
    pub fn bind_keys(
        &mut self,
        action: impl Into<ActionId>,
        keys: impl IntoIterator<Item = KeyCode>,
    ) -> &mut Self {
        self.bind(action, InputBinding::with_keys(keys))
    }

    /// Binds `action` to raw key codes, replacing any previous binding.
    pub fn bind_key_codes(
        &mut self,
        action: impl Into<ActionId>,
        codes: impl IntoIterator<Item = u32>,
    ) -> &mut Self {
        self.bind(action, InputBinding::with_key_codes(codes))
    }

    /// Binds `action` to named mouse buttons, replacing any previous
    /// binding.
    pub fn bind_mouse(
        &mut self,
        action: impl Into<ActionId>,
        buttons: impl IntoIterator<Item = MouseButton>,
    ) -> &mut Self {
        self.bind(action, InputBinding::with_mouse(buttons))
    }

    /// Removes the binding for `action`; returns `false` when absent.
    pub fn unbind(&mut self, action: &str) -> bool {
        self.bindings.remove(action).is_some()
    }

    /// Removes the binding for a typed action; returns `false` when absent.
    pub fn unbind_id(&mut self, action: &ActionId) -> bool {
        self.bindings.remove(action.as_str()).is_some()
    }

    /// Returns the binding for `action`, if any.
    pub fn binding(&self, action: &str) -> Option<&InputBinding> {
        self.bindings.get(action)
    }

    /// Returns the binding for a typed action, if any.
    pub fn binding_id(&self, action: &ActionId) -> Option<&InputBinding> {
        self.bindings.get(action.as_str())
    }

    /// Whether the action's binding has any key/button currently held.
    ///
    /// Unknown actions return `false` so a partially configured map stays
    /// total on the read edge.
    pub fn action_down(&self, input: &InputState, action: &str) -> bool {
        self.bindings
            .get(action)
            .is_some_and(|binding| binding.is_down(input))
    }

    /// Typed [`action_down`](Self::action_down): accepts [`GameAction`] or
    /// [`ActionId`] instead of a bare string.
    pub fn action_down_id(&self, input: &InputState, action: &ActionId) -> bool {
        self.action_down(input, action.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pressed(codes: &[u32]) -> InputState {
        let mut input = InputState::new();
        for &code in codes {
            input.set_key(code, true);
        }
        input
    }

    #[test]
    fn default_gameplay_covers_wasd_arrows_and_ascii() {
        let map = InputMap::default_gameplay();
        // Legacy physical aliases.
        assert!(map.action_down(&pressed(&[17]), InputMap::MOVE_FORWARD));
        assert!(map.action_down(&pressed(&[31]), InputMap::MOVE_BACK));
        assert!(map.action_down(&pressed(&[30]), InputMap::MOVE_LEFT));
        assert!(map.action_down(&pressed(&[32]), InputMap::MOVE_RIGHT));
        // ASCII aliases.
        assert!(map.action_down(&pressed(&[87]), InputMap::MOVE_FORWARD));
        assert!(map.action_down(&pressed(&[83]), InputMap::MOVE_BACK));
        assert!(map.action_down(&pressed(&[65]), InputMap::MOVE_LEFT));
        assert!(map.action_down(&pressed(&[68]), InputMap::MOVE_RIGHT));
        // Arrow aliases.
        assert!(map.action_down(&pressed(&[38]), InputMap::MOVE_FORWARD));
        assert!(map.action_down(&pressed(&[40]), InputMap::MOVE_BACK));
        assert!(map.action_down(&pressed(&[37]), InputMap::MOVE_LEFT));
        assert!(map.action_down(&pressed(&[39]), InputMap::MOVE_RIGHT));
        // No cross-talk between directions.
        assert!(!map.action_down(&pressed(&[17]), InputMap::MOVE_BACK));
        assert!(!map.action_down(&pressed(&[65]), InputMap::MOVE_RIGHT));
        assert!(!map.action_down(&InputState::new(), InputMap::MOVE_FORWARD));
    }

    #[test]
    fn custom_bindings_replace_defaults_and_support_mouse() {
        let mut map = InputMap::default_gameplay();
        map.bind_keys("jump", [KeyCode::Space]);
        let mut space = InputState::new();
        space.set_key(62, true);
        assert!(map.action_down(&space, "jump"));

        let mut click = InputState::new();
        click.set_mouse_button(0, true);
        assert!(!map.action_down(&click, "jump"));
        map.bind_mouse("jump", [MouseButton::Left]);
        assert!(map.action_down(&click, "jump"));

        // Rebinding replaces: forward no longer fires on W.
        map.bind_key_codes(InputMap::MOVE_FORWARD, [62]);
        assert!(!map.action_down(&pressed(&[17, 87]), InputMap::MOVE_FORWARD));
        assert!(map.action_down(&pressed(&[62]), InputMap::MOVE_FORWARD));

        assert!(map.unbind(InputMap::MOVE_FORWARD));
        assert!(!map.unbind(InputMap::MOVE_FORWARD));
        assert!(!map.action_down(&pressed(&[62]), InputMap::MOVE_FORWARD));
    }

    #[test]
    fn unknown_actions_and_empty_bindings_stay_quiet() {
        let map = InputMap::new();
        assert!(!map.action_down(&pressed(&[17]), "move_forward"));
        let mut map = map;
        map.bind("empty", InputBinding::new());
        assert!(!map.action_down(&pressed(&[17]), "empty"));
        assert_eq!(map.binding("empty"), Some(&InputBinding::new()));
        assert_eq!(map.binding("missing"), None);
    }

    #[test]
    fn bindings_deduplicate_raw_codes() {
        let binding = InputBinding::with_keys([KeyCode::KeyW, KeyCode::KeyW]);
        assert_eq!(binding.keys(), &[17, 87]);
        let mut binding = InputBinding::with_key_codes([87, 17, 87]);
        binding.add_mouse([MouseButton::Left, MouseButton::Left]);
        assert_eq!(binding.keys(), &[17, 87]);
        assert_eq!(binding.mouse_buttons(), &[0]);
        assert!(binding.is_down(&pressed(&[87])));
        assert!(!binding.is_down(&InputState::new()));
    }

    #[test]
    fn typed_action_ids_interoperate_with_strings() {
        // `GameAction` covers the canonical vocabulary; `ActionId` carries
        // tool-defined names — both land on the same map keys as strings.
        assert_eq!(GameAction::MoveForward.as_str(), InputMap::MOVE_FORWARD);
        assert_eq!(
            GameAction::Custom(ActionId::from("jump")).id(),
            ActionId::from("jump")
        );
        let mut map = InputMap::default_gameplay();
        map.bind(
            GameAction::MoveForward,
            InputBinding::with_keys([KeyCode::KeyW]),
        );
        // Rebinding through the typed id replaces the string-keyed default.
        assert!(map.action_down(&pressed(&[17]), InputMap::MOVE_FORWARD));
        assert!(map.action_down_id(&pressed(&[87]), &GameAction::MoveForward.id()));
        assert_eq!(
            map.binding_id(&ActionId::from("move_forward")),
            map.binding(InputMap::MOVE_FORWARD)
        );
        assert!(map.unbind_id(&GameAction::MoveBack.id()));
        assert!(!map.action_down(&pressed(&[31]), InputMap::MOVE_BACK));
    }
}
