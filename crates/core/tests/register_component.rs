//! F0 derive integration: `RegisterComponent` generates canonical names.

use ornis_core::{ComponentRegistry, RegisterComponent, SmartStore};
use ornis_macros::RegisterComponent;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, RegisterComponent)]
struct AutoNamed {
    v: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, RegisterComponent)]
#[component(name = "custom_health")]
struct CustomNamed {
    hp: u32,
}

#[test]
fn register_component_derive_uses_canonical_name() {
    assert_eq!(AutoNamed::COMPONENT_NAME, "AutoNamed");
    assert_eq!(CustomNamed::COMPONENT_NAME, "custom_health");

    let mut registry = ComponentRegistry::new();
    registry.register_component::<AutoNamed>();
    registry.register_component::<CustomNamed>();

    assert!(registry.by_name("AutoNamed").is_some());
    assert!(registry.by_name("custom_health").is_some());
    assert_eq!(
        registry.by_name("AutoNamed").unwrap().type_id(),
        std::any::TypeId::of::<AutoNamed>()
    );

    // Registry ops work through the derived name.
    let mut store = SmartStore::new();
    let e = store.create_entity();
    registry
        .by_name("custom_health")
        .unwrap()
        .set_json(&mut store, e, &serde_json::json!({"hp": 42}))
        .unwrap();
    assert_eq!(
        registry.by_name("custom_health").unwrap().lane_len(&store),
        1
    );
}
