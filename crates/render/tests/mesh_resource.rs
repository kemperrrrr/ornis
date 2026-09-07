//! X2 (S5e) gate: the GPU mesh as its own resource — re-creation follows
//! the `MeshDesc` lane (`max_mesh_params`, the same canon as the oracle
//! snapshot: complete entities only, (32, 24) floor), probed across
//! scenes with different tessellation. Runs headless — on CI via
//! lavapipe, locally on any adapter; skipped when no adapter is found.
//! The scene and harness live in `common` (shared with the pixel gates).

mod common;

use std::sync::Mutex;

use ornis_render::gpu_resources::{GpuDevice, GpuMesh, install_render_mesh};
use ornis_render::mesh::create_sphere;
use ornis_render::scene::{
    CameraDesc, EntityDesc, MaterialDesc, MeshDesc, Scene, TransformDesc,
};
use ornis_render::RenderWorld;

/// A probe scene with one dielectric sphere per requested tessellation.
fn probe_scene(tessellations: &[(u32, u32)]) -> Scene {
    Scene {
        name: "mesh probe".into(),
        entities: tessellations
            .iter()
            .map(|&(segments, rings)| EntityDesc {
                name: "sphere".into(),
                transform: TransformDesc {
                    translation: [0.0, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                mesh: MeshDesc::Sphere {
                    radius: 1.0,
                    segments,
                    rings,
                },
                material: MaterialDesc::Dielectric {
                    base_color: [0.8, 0.2, 0.2],
                    roughness: 0.4,
                },
            })
            .collect(),
        lights: Vec::new(),
        camera: CameraDesc {
            position: [0.0, 2.5, 9.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 60.0,
            near: 0.1,
            far: 100.0,
        },
        ambient: [0.1, 0.1, 0.1],
    }
}

/// The mesh resource state: `(segments, rings, vertex_count, num_indices)`.
fn mesh_state(world: &RenderWorld) -> (u32, u32, u32, u32) {
    let mesh = world
        .engine()
        .world()
        .resources()
        .get::<Mutex<GpuMesh>>()
        .expect("gpu mesh resource")
        .lock()
        .expect("gpu mesh lock");
    (mesh.params.0, mesh.params.1, mesh.mesh.vertex_count, mesh.mesh.num_indices)
}

#[test]
fn mesh_resource_follows_tessellation_lane() {
    common::with_headless_scene(|scene| {
        let device = &scene.device;
        let mut world = RenderWorld::from_scene(&probe_scene(&[(48, 32)]));
        let _ = world.engine_mut().world_mut().insert(GpuDevice(device.clone()));
        // Start at the floor tessellation — the probe scenes must move it.
        let initial = GpuMesh {
            mesh: create_sphere(device, 1.0, 32, 24),
            params: (32, 24),
        };
        install_render_mesh(world.engine_mut(), initial);
        world.run_frame(0.0);

        let (segments, rings, vertices, indices) = mesh_state(&world);
        assert_eq!((segments, rings), (48, 32), "X2: params follow the lane");
        let reference = create_sphere(device, 1.0, 48, 32);
        assert_eq!(vertices, reference.vertex_count);
        assert_eq!(indices, reference.num_indices);

        // A coarser scene joins — the max (and the mesh) stays.
        world.replace_scene(&probe_scene(&[(48, 32), (16, 12)]));
        world.run_frame(0.0);
        assert_eq!(mesh_state(&world), (48, 32, vertices, indices));

        // A finer scene — the mesh is re-created.
        world.replace_scene(&probe_scene(&[(48, 32), (16, 12), (64, 40)]));
        world.run_frame(0.0);
        let (segments, rings, vertices, indices) = mesh_state(&world);
        assert_eq!((segments, rings), (64, 40));
        let reference = create_sphere(device, 1.0, 64, 40);
        assert_eq!(vertices, reference.vertex_count);
        assert_eq!(indices, reference.num_indices);
    });
}
