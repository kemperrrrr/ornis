//! Native launch for a ready [`GameWorld`].
//!
//! [`run`] is the call user code makes. It opens the window, reads
//! `--frames` and `--remote-editor` from the process arguments, and seeds
//! the remote-editor entity counter from the world. The showcase binary
//! and the examples go through it.

#![warn(missing_docs)]

use std::error::Error;

use ornis_app::GameWorld;
use ornis_runner::{NativeOptions, run_native};

/// Opens a native window and runs `world` until the window closes.
///
/// The window title is [`GameWorld::title`](ornis_app::GameWorld::title)
/// (default `Ornis Engine`, replaced by
/// [`GameWorld::set_title`](ornis_app::GameWorld::set_title)). `--frames`
/// and `--remote-editor` are applied inside the runner, and the
/// remote-editor counter is the world's scene entity count. The returned
/// error is the runner's event-loop or GPU init failure, forwarded as
/// `Box<dyn Error>` so callers can use `?` without a second error type.
///
/// # Errors
///
/// When the event loop or GPU init fails. A frame that fails at runtime
/// prints to stderr and does not return here.
pub fn run(world: GameWorld) -> Result<(), Box<dyn Error>> {
    let title = world.title().to_owned();
    run_native(world, NativeOptions::from_env(title))
}
