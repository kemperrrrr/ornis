//! UI test: `#[smart_system]` rejects methods — only free functions become systems.

use ornis_macros::smart_system;

struct Host;

impl Host {
    #[smart_system]
    fn method(&self, resources: &ornis_core::Resources) {
        let _ = resources.len();
    }
}

fn main() {}
