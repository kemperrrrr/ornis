use crossbeam_channel::{Receiver, Sender, unbounded};

/// Commands sent from UI (JS) to the game thread
#[derive(Debug, Clone)]
pub enum UiCommand {
    CreateEntity,
    DestroyEntity {
        entity_id: u32,
    },
    SetComponent {
        entity_id: u32,
        type_name: String,
        json_data: String,
    },
    /// Generic command with a type tag and JSON payload.
    Custom {
        cmd_type: String,
        json_data: String,
    },
}

/// Events pushed from the game thread back to the UI thread
#[derive(Debug, Clone)]
pub enum GameEvent {
    ComponentUpdated {
        entity_id: u32,
        type_name: String,
        json_data: String,
    },
    EntityCreated {
        entity_id: u32,
    },
    EntityDestroyed {
        entity_id: u32,
    },
    /// Generic event for remote editor / extensibility.
    CustomEvent {
        cmd_type: String,
        json_data: String,
    },
}

/// UI-side handle for two-way IPC with the game thread.
/// Clone it freely — all clones share the same channel endpoints.
#[derive(Clone)]
pub struct IpcChannel {
    ui_to_game: Sender<UiCommand>,
    game_to_ui: Receiver<GameEvent>,
}

impl IpcChannel {
    /// Create a new IPC pair. Returns the UI handle and the game connection.
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
pub struct GameConnection {
    game_to_ui: Sender<GameEvent>,
    ui_to_game: Receiver<UiCommand>,
}

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
                type_name,
                json_data,
            } => {
                assert_eq!(entity_id, 0);
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
