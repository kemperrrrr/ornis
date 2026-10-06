//! Shared test helpers for extraction unit tests.

use ornis_assets::scene::MaterialDesc;
use ornis_assets::scene::MeshDesc;
use ornis_assets::scene::TransformDesc;
use ornis_core::Engine;
use ornis_core::units::Clamped01;
use ornis_core::units::PositiveF32;

/// Test helper: one entity with a transform plus optional mesh and
/// material lanes (missing lanes = incomplete entity).
pub(super) fn push_test_entity(
    engine: &mut Engine,
    mesh: Option<MeshDesc>,
    material: Option<MaterialDesc>,
) {
    let store = engine.world_mut().store_mut().expect("store");
    let handle = store.create_entity();
    store.insert(handle, TransformDesc::IDENTITY);
    if let Some(mesh) = mesh {
        store.insert(handle, mesh);
    }
    if let Some(material) = material {
        store.insert(handle, material);
    }
}

pub(super) fn test_material() -> MaterialDesc {
    MaterialDesc::Dielectric {
        base_color: [0.8, 0.2, 0.2],
        roughness: Clamped01::new(0.4),
        emission: [0.0, 0.0, 0.0],
        metallic: ornis_core::Metallic::new(0.0),
    }
}

pub(super) fn test_sphere() -> MeshDesc {
    MeshDesc::Sphere {
        radius: PositiveF32::expect_valid(1.0),
        segments: 16,
        rings: 12,
    }
}

pub(super) fn test_quad() -> MeshDesc {
    MeshDesc::Custom {
        positions: vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}
