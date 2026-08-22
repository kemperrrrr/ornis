//! Server-side ECS world for `editor-only` mode.
//!
//! In `editor-only` there is no native winit loop to consume `UiCommand`s,
//! so [`run`] spawns an `editor-world` thread that owns an [`EditorWorld`]
//! (ornis-core `EntityAllocator` + `ComponentStore`s), executes commands
//! from `POST /api/command` and publishes `GameEvent`s back to the HTTP
//! server (`status`/`scene` snapshots are cached by `remote.rs` for
//! `GET /api/status` and `GET /api/scene`; the rest reach `GET /api/events`).

use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, Sender};

use ornis_core::{ComponentStore, Entity, EntityAllocator};

use crate::ipc::{GameEvent, UiCommand};

/// Editor-side name component attached to every spawned entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Name(pub String);

/// Minimal live scene: alive entities plus their components.
#[derive(Default)]
pub struct EditorWorld {
    allocator: EntityAllocator,
    alive: Vec<Entity>,
    names: ComponentStore<Name>,
}

impl EditorWorld {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entity_count(&self) -> usize {
        self.alive.len()
    }

    pub fn spawn(&mut self, name: Option<String>) -> Entity {
        let entity = self.allocator.allocate();
        self.alive.push(entity);
        let name = name.unwrap_or_else(|| format!("Entity {}", entity.id()));
        self.names.insert(entity, Name(name));
        entity
    }

    /// Despawn by id/generation. Returns the entity if it was alive.
    pub fn despawn(&mut self, id: u32, generation: u32) -> Option<Entity> {
        let entity = Entity::new_with_gen(id, generation);
        if !self.allocator.is_alive(entity) {
            return None;
        }
        self.alive.retain(|e| *e != entity);
        self.names.remove(entity);
        self.allocator.deallocate(entity);
        Some(entity)
    }

    pub fn name_of(&self, entity: Entity) -> Option<&str> {
        self.names.get(entity).map(|n| n.0.as_str())
    }

    /// JSON snapshot for `GET /api/scene`.
    pub fn scene_json(&self) -> String {
        let mut entities: Vec<serde_json::Value> = Vec::with_capacity(self.alive.len());
        for &entity in &self.alive {
            let mut components = Vec::new();
            let mut name = serde_json::Value::Null;
            if let Some(n) = self.name_of(entity) {
                components.push("Name");
                name = serde_json::Value::String(n.to_string());
            }
            entities.push(serde_json::json!({
                "id": entity.id(),
                "generation": entity.generation(),
                "name": name,
                "components": components,
            }));
        }
        serde_json::json!({
            "entity_count": self.entity_count(),
            "entities": entities,
        })
        .to_string()
    }

    /// JSON payload for `GET /api/status` (cached by the HTTP server).
    pub fn status_json(&self) -> String {
        serde_json::json!({
            "entity_count": self.entity_count(),
            "name": "Ornis Engine",
        })
        .to_string()
    }

    /// Publish `status` + `scene` snapshots so the HTTP server's caches
    /// (`GET /api/status`, `GET /api/scene`) reflect the current world.
    fn publish_state(&self, ev_tx: &Sender<GameEvent>) {
        ev_tx
            .send(GameEvent::CustomEvent {
                cmd_type: "status".into(),
                json_data: self.status_json(),
            })
            .ok();
        ev_tx
            .send(GameEvent::CustomEvent {
                cmd_type: "scene".into(),
                json_data: self.scene_json(),
            })
            .ok();
    }

    /// Execute one command, emitting the corresponding events.
    ///
    /// Command set mirrors the native mode handler (`process_remote_commands`
    /// in `main.rs`) and extends it with `destroy_entity`.
    pub fn handle_command(&mut self, cmd: &UiCommand, ev_tx: &Sender<GameEvent>) {
        match cmd {
            UiCommand::CreateEntity => {
                self.spawn(None);
                self.publish_state(ev_tx);
            }
            UiCommand::DestroyEntity { entity_id } => {
                // The typed variant carries no generation; match any alive
                // entity with this id.
                let target = self.alive.iter().find(|e| e.id() == *entity_id).copied();
                if let Some(entity) = target {
                    self.despawn(entity.id(), entity.generation());
                    self.publish_state(ev_tx);
                }
            }
            UiCommand::Custom {
                cmd_type,
                json_data,
            } => match cmd_type.as_str() {
                "create_entity" => {
                    let name = serde_json::from_str::<serde_json::Value>(json_data)
                        .ok()
                        .and_then(|v| v.get("name").and_then(|n| n.as_str()).map(String::from));
                    let entity = self.spawn(name);
                    ev_tx
                        .send(GameEvent::CustomEvent {
                            cmd_type: "entity_created".into(),
                            json_data: serde_json::json!({
                                "entity_id": entity.id(),
                                "generation": entity.generation(),
                                "name": self.name_of(entity),
                            })
                            .to_string(),
                        })
                        .ok();
                    self.publish_state(ev_tx);
                }
                "destroy_entity" => {
                    let data =
                        serde_json::from_str::<serde_json::Value>(json_data).unwrap_or_default();
                    let id = data.get("id").and_then(|v| v.as_u64()).map(|v| v as u32);
                    let generation = data
                        .get("generation")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32)
                        .unwrap_or(0);
                    if let Some(id) = id
                        && let Some(entity) = self.despawn(id, generation)
                    {
                        ev_tx
                            .send(GameEvent::EntityDestroyed {
                                entity_id: entity.id(),
                            })
                            .ok();
                        self.publish_state(ev_tx);
                    }
                }
                "list_entities" => {
                    let ids: Vec<u32> = self.alive.iter().map(|e| e.id()).collect();
                    ev_tx
                        .send(GameEvent::CustomEvent {
                            cmd_type: "entity_list".into(),
                            json_data: serde_json::json!({
                                "count": self.alive.len(),
                                "entities": ids,
                            })
                            .to_string(),
                        })
                        .ok();
                }
                _ => {}
            },
            // SetComponent has no editable components yet on the server side.
            UiCommand::SetComponent { .. } => {}
        }
    }
}

/// Spawn the `editor-world` thread: owns the world, blocks on `cmd_rx`,
/// executes commands until the HTTP server side drops its sender.
pub fn run(cmd_rx: Receiver<UiCommand>, ev_tx: Sender<GameEvent>) -> JoinHandle<()> {
    thread::Builder::new()
        .name("editor-world".into())
        .spawn(move || {
            let mut world = EditorWorld::new();
            // Publish the initial (empty) state so the HTTP caches are live
            // before the first command arrives.
            world.publish_state(&ev_tx);
            while let Ok(cmd) = cmd_rx.recv() {
                world.handle_command(&cmd, &ev_tx);
            }
        })
        .expect("spawn editor-world thread")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;

    fn world_and_events() -> (EditorWorld, Sender<GameEvent>, Receiver<GameEvent>) {
        let (ev_tx, ev_rx) = unbounded();
        (EditorWorld::new(), ev_tx, ev_rx)
    }

    fn custom(cmd_type: &str, json_data: &str) -> UiCommand {
        UiCommand::Custom {
            cmd_type: cmd_type.into(),
            json_data: json_data.into(),
        }
    }

    /// Drain all pending events from the channel.
    fn drain_all(rx: &Receiver<GameEvent>) -> Vec<GameEvent> {
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    /// Extract `json_data` payloads of `CustomEvent`s with the given type.
    fn custom_events(events: &[GameEvent], cmd_type: &str) -> Vec<String> {
        events
            .iter()
            .filter_map(|ev| match ev {
                GameEvent::CustomEvent {
                    cmd_type: t,
                    json_data,
                } if t == cmd_type => Some(json_data.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn spawn_assigns_names_and_counts() {
        let mut world = EditorWorld::new();
        let a = world.spawn(None);
        let b = world.spawn(Some("Hero".into()));
        assert_eq!(world.entity_count(), 2);
        assert_eq!(world.name_of(a), Some("Entity 0"));
        assert_eq!(world.name_of(b), Some("Hero"));
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn despawn_recycles_ids_with_new_generation() {
        let mut world = EditorWorld::new();
        let a = world.spawn(None);
        let b = world.spawn(None);
        // Stale generation must not despawn.
        assert!(world.despawn(a.id(), a.generation() + 1).is_none());
        assert_eq!(world.entity_count(), 2);
        assert_eq!(world.despawn(a.id(), a.generation()), Some(a));
        assert_eq!(world.entity_count(), 1);
        let c = world.spawn(None);
        assert_eq!(c.id(), a.id());
        assert_ne!(c.generation(), a.generation());
        assert_eq!(world.name_of(b), Some("Entity 1"));
    }

    #[test]
    fn scene_json_lists_entities_with_names() {
        let mut world = EditorWorld::new();
        world.spawn(None);
        world.spawn(Some("Hero".into()));
        let scene: serde_json::Value = serde_json::from_str(&world.scene_json()).unwrap();
        assert_eq!(scene["entity_count"], 2);
        let entities = scene["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0]["id"], 0);
        assert_eq!(entities[0]["generation"], 0);
        assert_eq!(entities[0]["name"], "Entity 0");
        assert_eq!(entities[0]["components"], serde_json::json!(["Name"]));
        assert_eq!(entities[1]["name"], "Hero");
    }

    #[test]
    fn scene_json_empty_world() {
        let world = EditorWorld::new();
        let scene: serde_json::Value = serde_json::from_str(&world.scene_json()).unwrap();
        assert_eq!(scene["entity_count"], 0);
        assert_eq!(scene["entities"], serde_json::json!([]));
    }

    #[test]
    fn create_entity_command_emits_events_and_updates_state() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", r#"{"name":"Hero"}"#), &ev_tx);
        assert_eq!(world.entity_count(), 1);

        let events = drain_all(&ev_rx);

        let created = custom_events(&events, "entity_created");
        assert_eq!(created.len(), 1);
        let created: serde_json::Value = serde_json::from_str(&created[0]).unwrap();
        assert_eq!(created["entity_id"], 0);
        assert_eq!(created["name"], "Hero");

        let statuses = custom_events(&events, "status");
        assert_eq!(statuses.len(), 1);
        let status: serde_json::Value = serde_json::from_str(&statuses[0]).unwrap();
        assert_eq!(status["entity_count"], 1);

        let scenes = custom_events(&events, "scene");
        assert_eq!(scenes.len(), 1);
        let scene: serde_json::Value = serde_json::from_str(&scenes[0]).unwrap();
        assert_eq!(scene["entity_count"], 1);
        assert_eq!(scene["entities"][0]["name"], "Hero");

        assert!(ev_rx.try_recv().is_err(), "no leftover events");
    }

    #[test]
    fn create_entity_without_name_uses_default() {
        let (mut world, ev_tx, _ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        let scene: serde_json::Value = serde_json::from_str(&world.scene_json()).unwrap();
        assert_eq!(scene["entities"][0]["name"], "Entity 0");
    }

    #[test]
    fn destroy_entity_command_removes_entity() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        while ev_rx.try_recv().is_ok() {}

        world.handle_command(
            &custom("destroy_entity", r#"{"id":0,"generation":0}"#),
            &ev_tx,
        );
        assert_eq!(world.entity_count(), 1);
        let destroyed: Vec<u32> = std::iter::from_fn(|| ev_rx.try_recv().ok())
            .filter_map(|ev| match ev {
                GameEvent::EntityDestroyed { entity_id } => Some(entity_id),
                _ => None,
            })
            .collect();
        assert_eq!(destroyed, vec![0]);

        // Wrong generation: nothing happens, no event.
        while ev_rx.try_recv().is_ok() {}
        world.handle_command(
            &custom("destroy_entity", r#"{"id":1,"generation":9}"#),
            &ev_tx,
        );
        assert_eq!(world.entity_count(), 1);
        let destroyed: Vec<_> = std::iter::from_fn(|| ev_rx.try_recv().ok())
            .filter(|ev| matches!(ev, GameEvent::EntityDestroyed { .. }))
            .collect();
        assert!(destroyed.is_empty());
    }

    #[test]
    fn list_entities_command_reports_ids() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        while ev_rx.try_recv().is_ok() {}

        world.handle_command(&custom("list_entities", ""), &ev_tx);
        let lists = custom_events(&drain_all(&ev_rx), "entity_list");
        assert_eq!(lists.len(), 1);
        let list: serde_json::Value = serde_json::from_str(&lists[0]).unwrap();
        assert_eq!(list["count"], 2);
        assert_eq!(list["entities"], serde_json::json!([0, 1]));
    }

    #[test]
    fn typed_create_and_destroy_variants() {
        let (mut world, ev_tx, _ev_rx) = world_and_events();
        world.handle_command(&UiCommand::CreateEntity, &ev_tx);
        assert_eq!(world.entity_count(), 1);
        world.handle_command(&UiCommand::DestroyEntity { entity_id: 0 }, &ev_tx);
        assert_eq!(world.entity_count(), 0);
        // Destroying an unknown entity is a no-op.
        world.handle_command(&UiCommand::DestroyEntity { entity_id: 42 }, &ev_tx);
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn run_thread_processes_commands_until_disconnect() {
        let (cmd_tx, cmd_rx) = unbounded();
        let (ev_tx, ev_rx) = unbounded();
        let handle = run(cmd_rx, ev_tx);

        cmd_tx.send(custom("create_entity", r#"{"name":"Hero"}"#)).unwrap();
        // Wait for the scene snapshot reflecting the new entity.
        let mut seen_scene = None;
        for _ in 0..100 {
            while let Ok(ev) = ev_rx.try_recv() {
                if let GameEvent::CustomEvent { cmd_type, json_data } = ev
                    && cmd_type == "scene"
                {
                    seen_scene = Some(json_data);
                }
            }
            if let Some(scene) = &seen_scene
                && scene.contains("Hero")
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let scene = seen_scene.expect("scene snapshot expected");
        assert!(scene.contains("Hero"));

        drop(cmd_tx);
        handle.join().expect("editor-world thread must finish");
    }
}
