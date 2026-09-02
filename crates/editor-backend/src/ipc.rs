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
    SetComponent {
        /// Id of the entity to edit.
        entity_id: u32,
        /// `None` matches any alive generation.
        generation: Option<u32>,
        /// Registry name of the component ("Transform", "Mesh", ...).
        type_name: String,
        /// serde-canonical JSON of the whole component (full replace).
        json_data: String,
    },
    /// Generic command with a type tag and JSON payload.
    Custom {
        /// Command tag, e.g. "create_entity"/"destroy_entity"/"list_entities".
        cmd_type: String,
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
        request_id: u64,
        /// Command to execute on the engine thread.
        command: Box<UiCommand>,
    },
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
        type_name: String,
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
        cmd_type: String,
        /// JSON payload of the event.
        json_data: String,
    },
    /// Completion result correlated with a transport request id.
    CommandCompleted {
        /// Request id from [`UiCommand::WithRequestId`].
        request_id: u64,
        /// Normalized command name (`create_entity`, `set_component`, ...).
        command: String,
        /// Whether the engine completed the command successfully.
        success: bool,
        /// Human-readable failure reason, if `success` is false.
        error: Option<String>,
    },
    /// Transport marker indicating that a bounded event history no longer
    /// contains everything after the client's cursor.
    EventGap {
        /// Cursor supplied by the client before the gap was detected.
        after: u64,
        /// Earliest event sequence still retained by the server.
        oldest: u64,
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
}
