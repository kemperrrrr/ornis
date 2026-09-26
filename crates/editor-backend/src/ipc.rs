//! Editor ↔ engine IPC protocol.
//!
//! Command and event types exchanged between the browser editor and the
//! engine over crossbeam-channels (see `remote.rs`: `POST /api/command`
//! → `UiCommand`, engine events → `GET /api/events` ← `GameEvent`).
//!
//! The variant set is the protocol surface for the roadmap (engine↔editor
//! command handler, `GET /api/scene`): `Custom`/`CustomEvent` carry
//! entity-level commands (create/destroy/list), `SetComponent` is produced
//! by `remote.rs` for `{"type":"set_component"}` posts and executed
//! generically through the component registry (F0, audit §10 D2), and
//! `ComponentUpdated` reports successful edits back. The HTTP transport adds
//! request acknowledgements and snapshot sequence metadata in `remote.rs`,
//! while wrapped commands receive correlated completion events. The remaining
//! typed variants
//! are reserved and marked `#[allow(dead_code)]`.

use crossbeam_channel::{Receiver, Sender, unbounded};

use ornis_input::{KeyCode, MouseButton};

/// Browser input snapshot forwarded over WebSocket / `POST /api/input`.
///
/// This is the server-side mirror of [`ornis_core::InputState`]: the browser
/// sends pressed keys/buttons and pointer/wheel deltas, the engine replaces
/// its authoritative [`ornis_core::InputState`] resource with the snapshot.
/// Transient deltas are consumed once per frame and cleared by the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct BrowserInput {
    /// Pressed key codes.
    pub pressed_keys: Vec<u32>,
    /// Pressed mouse button codes.
    pub pressed_mouse_buttons: Vec<u8>,
    /// Absolute pointer position `[x, y]`.
    pub pointer_position: [f32; 2],
    /// Pointer movement since the last snapshot.
    pub pointer_delta: [f32; 2],
    /// Wheel delta since the last snapshot.
    pub wheel_delta: f32,
}

impl Default for BrowserInput {
    fn default() -> Self {
        Self {
            pressed_keys: Vec::new(),
            pressed_mouse_buttons: Vec::new(),
            pointer_position: [0.0, 0.0],
            pointer_delta: [0.0, 0.0],
            wheel_delta: 0.0,
        }
    }
}

impl BrowserInput {
    /// Pressed keys mapped to named [`KeyCode`]s on the wire boundary.
    ///
    /// Raw codes no named key covers (native `Other` keys) are dropped here;
    /// the raw [`BrowserInput::pressed_keys`] snapshot stays intact for
    /// forward-compat consumers.
    pub fn keys(&self) -> Vec<KeyCode> {
        self.pressed_keys
            .iter()
            .filter_map(|&code| KeyCode::from_code(code))
            .collect()
    }

    /// Pressed mouse buttons mapped to named [`MouseButton`]s.
    ///
    /// Raw codes above `Forward` stay raw-only (see [`MouseButton`]) and are
    /// dropped here; [`BrowserInput::pressed_mouse_buttons`] is unchanged.
    pub fn mouse_buttons(&self) -> Vec<MouseButton> {
        self.pressed_mouse_buttons
            .iter()
            .filter_map(|&code| MouseButton::from_code(code))
            .collect()
    }
}

/// Transport request id: correlates an HTTP `/api/command` acknowledgement
/// with the engine's [`GameEvent::CommandCompleted`].
///
/// Transparent over `u64`: serializes as a plain JSON number, so ack and
/// event payloads are byte-identical to the former bare-`u64` form.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(u64);

/// Transport event sequence: the replay cursor of `/api/events`.
/// Transparent over `u64` (plain JSON number, as before).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventSeq(u64);

/// Shared implementation of the transparent `u64` transport ids above.
macro_rules! impl_transport_id {
    ($ty:ident) => {
        impl $ty {
            /// Wraps a raw id.
            pub fn new(id: u64) -> Self {
                Self(id)
            }

            /// Returns the raw id.
            pub fn get(self) -> u64 {
                self.0
            }

            /// Returns the next id, saturating instead of wrapping and
            /// skipping the reserved `0` (client cursors start there).
            #[must_use]
            pub fn next(self) -> Self {
                Self(self.0.saturating_add(1).max(1))
            }
        }

        impl From<u64> for $ty {
            /// Wraps a raw id.
            fn from(id: u64) -> Self {
                Self(id)
            }
        }

        impl From<$ty> for u64 {
            /// Unwraps back to the raw id.
            fn from(id: $ty) -> Self {
                id.0
            }
        }

        impl PartialEq<u64> for $ty {
            /// Compares against a raw id (ack/event shorthand in tests).
            fn eq(&self, other: &u64) -> bool {
                self.0 == *other
            }
        }

        impl PartialEq<$ty> for u64 {
            /// Compares a raw id against a typed id.
            fn eq(&self, other: &$ty) -> bool {
                *self == other.0
            }
        }

        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl serde::Serialize for $ty {
            /// Encodes as the plain underlying number.
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_u64(self.0)
            }
        }

        impl<'de> serde::Deserialize<'de> for $ty {
            /// Decodes from the plain underlying number.
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                u64::deserialize(deserializer).map(Self)
            }
        }
    };
}

impl_transport_id!(RequestId);
impl_transport_id!(EventSeq);

/// Wire-level component name: the registry key of a `set_component` payload.
///
/// Owned (`Box<str>`) mirror of `ornis_core::ComponentName` without pulling
/// the whole ECS into this crate; lookups accept `&str`, so existing
/// `type_name: "Health".into()` call sites keep compiling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComponentName(Box<str>);

/// Arbitrary command/event tag for forward-compat `Custom` payloads
/// (`"ping"`, `"scene_saved"`, …).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CommandName(Box<str>);

/// Shared implementation of the `Box<str>` wire names above.
macro_rules! impl_wire_name {
    ($ty:ident) => {
        impl $ty {
            /// Wraps a `'static` wire name without extra bookkeeping.
            pub fn from_static(name: &'static str) -> Self {
                Self(name.into())
            }

            /// Wire name as a string slice.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl std::borrow::Borrow<str> for $ty {
            fn borrow(&self) -> &str {
                self.as_str()
            }
        }

        impl AsRef<str> for $ty {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl From<&str> for $ty {
            fn from(name: &str) -> Self {
                Self(name.into())
            }
        }

        impl From<String> for $ty {
            fn from(name: String) -> Self {
                Self(name.into_boxed_str())
            }
        }

        impl From<&String> for $ty {
            fn from(name: &String) -> Self {
                Self(name.as_str().into())
            }
        }

        impl PartialEq<&str> for $ty {
            /// Compares against a plain tag (wire shorthand in handlers).
            fn eq(&self, other: &&str) -> bool {
                self.as_str() == *other
            }
        }

        impl serde::Serialize for $ty {
            /// Encodes as the plain wire string.
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> serde::Deserialize<'de> for $ty {
            /// Decodes from the plain wire string.
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                String::deserialize(deserializer).map(Self::from)
            }
        }
    };
}

impl_wire_name!(ComponentName);
impl_wire_name!(CommandName);

/// Typed `type`-tag vocabulary of the `/api/command` envelope
/// (`{"type": …, "data": …}`).
///
/// This is the `tag = "type"` side of the protocol: known commands are
/// first-class variants, anything else arrives as [`EditorCommand::Custom`]
/// so old and future JSON payloads keep parsing instead of rejecting the
/// whole post. Serializes as the plain tag string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EditorCommand {
    /// Spawn a new entity with default components.
    CreateEntity,
    /// Despawn the entity with this id (any generation).
    DestroyEntity,
    /// List entities (`entity_list` event).
    ListEntities,
    /// Generic component upsert by registry name.
    SetComponent,
    /// Persist the scene to disk.
    SaveScene,
    /// Load the scene from disk.
    LoadScene,
    /// Browser input snapshot (WS / `POST /api/input`).
    Input,
    /// Connectivity probe.
    Ping,
    /// Forward-compat command/event tag.
    Custom(CommandName),
}

impl EditorCommand {
    /// Canonical tag string (`"create_entity"`, …, or the custom tag).
    pub fn as_str(&self) -> &str {
        match self {
            EditorCommand::CreateEntity => "create_entity",
            EditorCommand::DestroyEntity => "destroy_entity",
            EditorCommand::ListEntities => "list_entities",
            EditorCommand::SetComponent => "set_component",
            EditorCommand::SaveScene => "save_scene",
            EditorCommand::LoadScene => "load_scene",
            EditorCommand::Input => "input",
            EditorCommand::Ping => "ping",
            EditorCommand::Custom(name) => name.as_str(),
        }
    }
}

impl std::fmt::Display for EditorCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for EditorCommand {
    /// Maps known tags to variants; anything else becomes `Custom`
    /// (never fails, so unknown wire tags keep parsing).
    fn from(tag: &str) -> Self {
        match tag {
            "create_entity" => EditorCommand::CreateEntity,
            "destroy_entity" => EditorCommand::DestroyEntity,
            "list_entities" => EditorCommand::ListEntities,
            "set_component" => EditorCommand::SetComponent,
            "save_scene" => EditorCommand::SaveScene,
            "load_scene" => EditorCommand::LoadScene,
            "input" => EditorCommand::Input,
            "ping" => EditorCommand::Ping,
            other => EditorCommand::Custom(CommandName::from(other)),
        }
    }
}

impl From<String> for EditorCommand {
    fn from(tag: String) -> Self {
        Self::from(tag.as_str())
    }
}

impl From<&String> for EditorCommand {
    fn from(tag: &String) -> Self {
        Self::from(tag.as_str())
    }
}

impl PartialEq<&str> for EditorCommand {
    /// Compares against a plain tag (handler shorthand).
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl serde::Serialize for EditorCommand {
    /// Encodes as the plain tag string.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for EditorCommand {
    /// Decodes from the plain tag string (unknown tags → `Custom`).
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::from)
    }
}

/// Typed `set_component` wire payload
/// (`{"id": u32, "generation"?: u32, "component": name, "value": {…}}`).
///
/// Deserializes `id`/`generation` directly as `u32` (serde rejects
/// out-of-range numbers instead of truncating them) and converts into
/// [`UiCommand::SetComponent`] through [`TryFrom`]. Field aliases accept
/// the in-process names (`entity_id`/`type_name`/`json_data`); old
/// `{"id","component","value"}` posts parse unchanged.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct SetComponentPayload {
    /// Id of the entity to edit.
    #[serde(alias = "entity_id")]
    pub id: u32,
    /// `None` matches any alive generation.
    #[serde(default)]
    pub generation: Option<u32>,
    /// Registry name of the component.
    #[serde(alias = "type_name")]
    pub component: ComponentName,
    /// serde-canonical JSON of the whole component (full replace).
    #[serde(alias = "json_data")]
    pub value: serde_json::Value,
}

/// Rejection of a [`SetComponentPayload`] → [`UiCommand`] conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetComponentError {
    /// The component name is empty (no registry entry can match it).
    EmptyComponent,
}

impl std::fmt::Display for SetComponentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SetComponentError::EmptyComponent => f.write_str("component name is empty"),
        }
    }
}

impl std::error::Error for SetComponentError {}

/// Commands sent from UI (JS) to the game thread
#[derive(Debug, Clone)]
#[allow(dead_code)] // protocol surface for editor↔engine (roadmap)
pub enum UiCommand {
    /// Spawn a new entity with default components.
    CreateEntity,
    /// Despawn the entity with this id (any generation).
    DestroyEntity {
        /// Entity id to despawn.
        entity_id: u32,
    },
    /// Generic component upsert by registry name: `json_data` is the
    /// serde-canonical JSON of the whole component (full replace).
    /// `generation: None` matches any alive entity with this id.
    /// Prefer building it from [`SetComponentPayload`] (`TryFrom`), which
    /// validates the wire shape without truncating integer fields.
    SetComponent {
        /// Id of the entity to edit.
        entity_id: u32,
        /// `None` matches any alive generation.
        generation: Option<u32>,
        /// Registry name of the component ("Transform", "Mesh", ...).
        type_name: ComponentName,
        /// serde-canonical JSON of the whole component (full replace).
        json_data: String,
    },
    /// Generic command with a type tag and JSON payload.
    Custom {
        /// Command tag, e.g. "create_entity"/"destroy_entity"/"list_entities".
        cmd_type: EditorCommand,
        /// JSON object payload for the command.
        json_data: String,
    },
    /// Browser input snapshot: replaces the engine's `InputState` resource.
    ///
    /// Sent via WebSocket (`/api/events` bidirectionally) or `POST /api/input`.
    /// No polling / `scene.ron` fallback is required.
    Input {
        /// Input snapshot from the browser.
        input: BrowserInput,
    },
    /// Transport wrapper carrying the request id through the engine queue.
    ///
    /// The HTTP layer sends this variant after returning its queue-level ACK;
    /// the engine emits a matching [`GameEvent::CommandCompleted`] when the
    /// wrapped command finishes. Existing in-process callers may continue to
    /// use the unwrapped variants.
    WithRequestId {
        /// Request id assigned by the HTTP transport.
        request_id: RequestId,
        /// Command to execute on the engine thread.
        command: Box<UiCommand>,
    },
}

impl TryFrom<SetComponentPayload> for UiCommand {
    type Error = SetComponentError;

    /// Builds the typed upsert without truncating integer fields (serde
    /// already rejected out-of-range `id`/`generation` at parse time).
    fn try_from(payload: SetComponentPayload) -> Result<Self, Self::Error> {
        if payload.component.as_str().is_empty() {
            return Err(SetComponentError::EmptyComponent);
        }
        Ok(UiCommand::SetComponent {
            entity_id: payload.id,
            generation: payload.generation,
            type_name: payload.component,
            json_data: payload.value.to_string(),
        })
    }
}

/// Events pushed from the game thread back to the UI thread
#[derive(Debug, Clone)]
#[allow(dead_code)] // protocol surface for editor↔engine (roadmap)
pub enum GameEvent {
    /// Emitted after a successful `SetComponent`: `json_data` echoes the
    /// applied payload (serde-canonical component JSON).
    ComponentUpdated {
        /// Id of the edited entity.
        entity_id: u32,
        /// Registry name of the component.
        type_name: ComponentName,
        /// Applied component JSON.
        json_data: String,
    },
    /// A new entity was spawned.
    EntityCreated {
        /// Id of the created entity.
        entity_id: u32,
    },
    /// An entity was destroyed.
    EntityDestroyed {
        /// Id of the destroyed entity.
        entity_id: u32,
    },
    /// Generic event for remote editor / extensibility.
    CustomEvent {
        /// Event tag mirroring the originating command type.
        cmd_type: EditorCommand,
        /// JSON payload of the event.
        json_data: String,
    },
    /// Completion result correlated with a transport request id.
    CommandCompleted {
        /// Request id from [`UiCommand::WithRequestId`].
        request_id: RequestId,
        /// Normalized command name (`create_entity`, `set_component`, ...).
        command: EditorCommand,
        /// Whether the engine completed the command successfully.
        success: bool,
        /// Human-readable failure reason, if `success` is false.
        error: Option<String>,
    },
    /// Transport marker indicating that a bounded event history no longer
    /// contains everything after the client's cursor.
    EventGap {
        /// Cursor supplied by the client before the gap was detected.
        after: EventSeq,
        /// Earliest event sequence still retained by the server.
        oldest: EventSeq,
    },
}

/// UI-side handle for two-way IPC with the game thread.
/// Clone it freely — all clones share the same channel endpoints.
///
// reserved: two-way channel for the future editor↔engine protocol;
// remote.rs currently works with the raw channels directly.
#[derive(Clone)]
#[allow(dead_code)]
pub struct IpcChannel {
    ui_to_game: Sender<UiCommand>,
    game_to_ui: Receiver<GameEvent>,
}

#[allow(dead_code)] // reserved: see comment on the struct
impl IpcChannel {
    /// Create a new IPC pair. Returns the UI handle and the game connection.
    #[allow(missing_docs)] // reserved struct, see comment above
    pub fn pair() -> (Self, GameConnection) {
        let (ui_tx, game_rx) = unbounded();
        let (game_tx, ui_rx) = unbounded();
        (
            Self {
                ui_to_game: ui_tx,
                game_to_ui: ui_rx,
            },
            GameConnection {
                game_to_ui: game_tx,
                ui_to_game: game_rx,
            },
        )
    }

    /// Send a command to the game thread.
    pub fn send(&self, cmd: UiCommand) {
        let _ = self.ui_to_game.send(cmd);
    }

    /// Try to receive an event from the game thread (non-blocking).
    pub fn poll(&self) -> Option<GameEvent> {
        self.game_to_ui.try_recv().ok()
    }
}

/// Game-side handle for two-way IPC with the UI thread.
// reserved: see IpcChannel — protocol surface (roadmap).
#[allow(dead_code)]
pub struct GameConnection {
    game_to_ui: Sender<GameEvent>,
    ui_to_game: Receiver<UiCommand>,
}

#[allow(dead_code)] // reserved: see comment on the struct
impl GameConnection {
    /// Try to receive a command from the UI thread (non-blocking).
    pub fn poll(&self) -> Option<UiCommand> {
        self.ui_to_game.try_recv().ok()
    }

    /// Send an event back to the UI thread.
    pub fn send(&self, event: GameEvent) {
        let _ = self.game_to_ui.send(event);
    }

    /// Block until a command arrives from the UI thread.
    pub fn recv(&self) -> Result<UiCommand, crossbeam_channel::RecvError> {
        self.ui_to_game.recv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_send_command() {
        let (ui, game) = IpcChannel::pair();

        ui.send(UiCommand::CreateEntity);
        ui.send(UiCommand::DestroyEntity { entity_id: 42 });
        ui.send(UiCommand::SetComponent {
            entity_id: 0,
            generation: Some(0),
            type_name: "UIStyle".into(),
            json_data: r#"{"color":[1,0,0,1]}"#.into(),
        });

        let cmd1 = game.poll().expect("should receive CreateEntity");
        assert!(matches!(cmd1, UiCommand::CreateEntity));

        let cmd2 = game.poll().expect("should receive DestroyEntity");
        assert!(matches!(cmd2, UiCommand::DestroyEntity { entity_id: 42 }));

        let cmd3 = game.poll().expect("should receive SetComponent");
        match cmd3 {
            UiCommand::SetComponent {
                entity_id,
                generation,
                type_name,
                json_data,
            } => {
                assert_eq!(entity_id, 0);
                assert_eq!(generation, Some(0));
                assert_eq!(type_name, "UIStyle");
                assert_eq!(json_data, r#"{"color":[1,0,0,1]}"#);
            }
            _ => panic!("expected SetComponent"),
        }

        assert!(game.poll().is_none(), "no more commands");
    }

    #[test]
    fn test_ipc_send_event() {
        let (ui, game) = IpcChannel::pair();

        game.send(GameEvent::EntityCreated { entity_id: 7 });
        game.send(GameEvent::ComponentUpdated {
            entity_id: 7,
            type_name: "UIStyle".into(),
            json_data: r#"{"font_size":24}"#.into(),
        });

        let ev1 = ui.poll().expect("should receive EntityCreated");
        assert!(matches!(ev1, GameEvent::EntityCreated { entity_id: 7 }));

        let ev2 = ui.poll().expect("should receive ComponentUpdated");
        match ev2 {
            GameEvent::ComponentUpdated {
                entity_id,
                type_name,
                json_data,
            } => {
                assert_eq!(entity_id, 7);
                assert_eq!(type_name, "UIStyle");
                assert_eq!(json_data, r#"{"font_size":24}"#);
            }
            _ => panic!("expected ComponentUpdated"),
        }

        assert!(ui.poll().is_none(), "no more events");
    }

    #[test]
    fn test_ipc_bidirectional() {
        let (ui, game) = IpcChannel::pair();

        // UI → Game
        ui.send(UiCommand::SetComponent {
            entity_id: 1,
            generation: None,
            type_name: "Health".into(),
            json_data: r#"{"hp":100}"#.into(),
        });

        // Game processes it, sends a response
        if let Some(cmd) = game.poll() {
            match cmd {
                UiCommand::SetComponent { entity_id, .. } => {
                    game.send(GameEvent::ComponentUpdated {
                        entity_id,
                        type_name: "Health".into(),
                        json_data: r#"{"hp":100}"#.into(),
                    });
                }
                _ => panic!("unexpected command"),
            }
        }

        // UI receives the response
        let ev = ui.poll().expect("should receive response");
        match ev {
            GameEvent::ComponentUpdated {
                entity_id,
                json_data,
                ..
            } => {
                assert_eq!(entity_id, 1);
                assert!(json_data.contains("\"hp\":100"));
            }
            _ => panic!("expected ComponentUpdated"),
        }
    }

    #[test]
    fn editor_command_tags_round_trip_and_keep_unknown_tags() {
        // Known tags map to variants and serialize back byte-identically.
        for (tag, command) in [
            ("create_entity", EditorCommand::CreateEntity),
            ("set_component", EditorCommand::SetComponent),
            ("ping", EditorCommand::Ping),
        ] {
            assert_eq!(EditorCommand::from(tag), command);
            assert_eq!(command.as_str(), tag);
            assert_eq!(serde_json::to_string(&command).unwrap(), format!("{tag:?}"));
        }
        // Unknown tags stay visible as `Custom`, never rejected (old and
        // future JSON keep parsing).
        let custom = EditorCommand::from("frobnicate");
        assert!(matches!(custom, EditorCommand::Custom(_)));
        assert_eq!(custom.as_str(), "frobnicate");
        assert_eq!(
            EditorCommand::from("status".to_string()),
            EditorCommand::Custom(CommandName::from("status"))
        );
    }

    #[test]
    fn set_component_payload_converts_without_truncation() {
        // Canonical wire shape converts losslessly.
        let payload: SetComponentPayload = serde_json::from_value(serde_json::json!({
            "id": 7u64,
            "generation": 3u64,
            "component": "Transform",
            "value": {"x": 1.0}
        }))
        .expect("valid payload parses");
        let command = UiCommand::try_from(payload).expect("valid payload converts");
        assert!(matches!(
            command,
            UiCommand::SetComponent {
                entity_id: 7,
                generation: Some(3),
                ..
            }
        ));
        // Out-of-range integers are rejected at parse time, never wrapped.
        let overflow = serde_json::from_value::<SetComponentPayload>(serde_json::json!({
            "id": 4_294_967_296u64,
            "component": "T",
            "value": {}
        }));
        assert!(overflow.is_err(), "u32 overflow must not truncate");
        // Empty component names are rejected at conversion time.
        let empty = SetComponentPayload {
            id: 1,
            generation: None,
            component: ComponentName::from(""),
            value: serde_json::json!({}),
        };
        assert!(UiCommand::try_from(empty).is_err());
    }

    #[test]
    fn transport_ids_serialize_as_plain_numbers() {
        assert_eq!(serde_json::to_string(&RequestId::new(99)).unwrap(), "99");
        assert_eq!(serde_json::to_string(&EventSeq::new(12)).unwrap(), "12");
        assert_eq!(RequestId::new(41).next(), RequestId::new(42));
        assert_eq!(EventSeq::new(0).next(), EventSeq::new(1));
    }

    #[test]
    fn browser_input_maps_raw_codes_to_named_keys() {
        let input = BrowserInput {
            pressed_keys: vec![17, 87, 999],
            pressed_mouse_buttons: vec![0, 9],
            ..BrowserInput::default()
        };
        // Raw snapshots stay intact; the typed view drops unknown codes.
        assert_eq!(input.pressed_keys, vec![17, 87, 999]);
        assert_eq!(input.keys(), vec![KeyCode::KeyW, KeyCode::KeyW]);
        assert_eq!(input.mouse_buttons(), vec![MouseButton::Left]);
    }
}
