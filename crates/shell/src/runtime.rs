//! Process wiring: the engine thread and the event pump.
//!
//! Two long-lived pieces sit behind the window: the engine session (owns the
//! world, executes commands, publishes events) and the pump that forwards those
//! events into the winit loop, because every webview call must stay on the main
//! thread. They live here rather than in the binary so the binary stays a thin
//! argument parser + event handler, and so the channel plumbing is testable
//! without a window.

use crossbeam_channel::{Receiver, Sender};
use editor_backend::ipc::{GameEvent, UiCommand};

/// Starts the live ECS world on its own thread.
///
/// Same entry point as `editor-only`: the engine executes queued commands and
/// publishes `scene`/`status`/`CommandCompleted` events, so the shell never owns
/// a world of its own. Returns the command sender and the event receiver the
/// pump drains.
///
/// The engine's join handle is deliberately dropped: the thread runs until the
/// process exits (or the command channel closes), which is what the shell
/// wants — there is nothing to join at shutdown, and joining here would either
/// block on a live world or panic on a poisoned one.
pub fn start_engine() -> (Sender<UiCommand>, Receiver<GameEvent>) {
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<UiCommand>();
    let (ev_tx, ev_rx) = crossbeam_channel::unbounded::<GameEvent>();
    drop(ornis_app::session::run(cmd_rx, ev_tx));
    (cmd_tx, ev_rx)
}

/// Forwards every engine event to `forward`, until the channel closes or
/// `forward` rejects one.
///
/// The pump is generic over the handoff so this loop — the part with the
/// termination contract — is testable: the binary passes a winit proxy, a test
/// passes a channel. `forward` returning `false` means the other end is gone
/// (window closed), which ends the pump cleanly rather than spinning.
pub fn pump_events<F>(events: Receiver<GameEvent>, mut forward: F)
where
    F: FnMut(GameEvent) -> bool,
{
    while let Ok(event) = events.recv() {
        if !forward(event) {
            break;
        }
    }
}

/// Spawns [`pump_events`] on a named thread.
///
/// # Errors
///
/// The OS refusing a thread (out of resources). The caller decides: without a
/// pump no event ever reaches the page, so the binary treats it as fatal.
pub fn spawn_event_pump<F>(events: Receiver<GameEvent>, forward: F) -> std::io::Result<()>
where
    F: FnMut(GameEvent) -> bool + Send + 'static,
{
    std::thread::Builder::new()
        .name("shell-event-pump".into())
        .spawn(move || pump_events(events, forward))
        .map(|_handle| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use editor_backend::ipc::RequestId;

    fn event(id: u32) -> GameEvent {
        GameEvent::EntityCreated { entity_id: id }
    }

    fn created_id(event: &GameEvent) -> Option<u32> {
        match event {
            GameEvent::EntityCreated { entity_id } => Some(*entity_id),
            _ => None,
        }
    }

    #[test]
    fn the_pump_forwards_every_event_in_order() {
        let (tx, rx) = crossbeam_channel::unbounded::<GameEvent>();
        for id in 1..=3 {
            tx.send(event(id)).expect("channel open");
        }
        drop(tx);
        // `GameEvent` carries no `PartialEq` (string payloads), so compare the
        // ids the forwarded events carry.
        let mut seen = Vec::new();
        pump_events(rx, |event| {
            seen.push(created_id(&event));
            true
        });
        assert_eq!(
            seen,
            vec![Some(1), Some(2), Some(3)],
            "every event reaches the page, in publication order"
        );
    }

    #[test]
    fn the_pump_stops_when_the_forwarder_says_the_page_is_gone() {
        let (tx, rx) = crossbeam_channel::unbounded::<GameEvent>();
        for id in 1..=3 {
            tx.send(event(id)).expect("channel open");
        }
        let remaining = rx.clone();
        let mut seen = Vec::new();
        pump_events(rx, |event| {
            seen.push(created_id(&event));
            // The window closed after the first event.
            false
        });
        assert_eq!(seen, vec![Some(1)], "a closed page ends the pump");
        // The remaining events stay queued: the pump must not drain and drop
        // them, or a restarted pump would lose the tail of the stream.
        assert_eq!(remaining.len(), 2);
    }

    #[test]
    fn the_pump_ends_when_the_engine_goes_away() {
        let (tx, rx) = crossbeam_channel::unbounded::<GameEvent>();
        tx.send(event(1)).expect("channel open");
        drop(tx);
        let mut count = 0;
        pump_events(rx, |_| {
            count += 1;
            true
        });
        assert_eq!(count, 1, "a disconnected engine ends the loop, not a spin");
    }

    #[test]
    fn the_pump_tolerates_a_full_event_stream() {
        // A dense burst is the normal case (every mutation publishes a scene);
        // the pump must hand each one over rather than coalesce them.
        let (tx, rx) = crossbeam_channel::unbounded::<GameEvent>();
        for id in 0..64u64 {
            tx.send(GameEvent::CommandCompleted {
                request_id: RequestId::new(id + 1),
                command: "ping".into(),
                success: true,
                error: None,
            })
            .expect("channel open");
        }
        drop(tx);
        let mut count = 0;
        pump_events(rx, |_| {
            count += 1;
            true
        });
        assert_eq!(count, 64);
    }

    #[test]
    fn starting_the_engine_yields_a_live_command_channel() {
        // The real session thread: a `ping` must be accepted, which proves the
        // receiver end is alive and the sender the shell holds is the right one.
        let (cmd_tx, _ev_rx) = start_engine();
        assert!(
            cmd_tx.send(UiCommand::CreateEntity).is_ok(),
            "the engine must accept a command right after start"
        );
    }

    #[test]
    fn the_pump_spawns_on_its_named_thread() {
        let (tx, rx) = crossbeam_channel::unbounded::<GameEvent>();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        spawn_event_pump(rx, move |_| {
            let _ = done_tx.send(());
            false
        })
        .expect("a pump thread must start");
        tx.send(event(1)).expect("channel open");
        done_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the pump forwards on its thread");
    }
}
