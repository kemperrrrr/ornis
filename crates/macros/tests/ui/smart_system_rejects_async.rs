//! UI test: `#[smart_system]` rejects `async fn` — `System::run` is sync.

use ornis_macros::smart_system;

#[smart_system]
async fn async_system(resources: &ornis_core::Resources) {
    let _ = resources.len();
}

fn main() {}
