//! Packing checks and GPU parity of the guest kernel against the CPU skin.

use glam::Mat4;
use glam::Vec3;
use ornis_animation::CPU_GPU_TOLERANCE;
use ornis_animation::SkinningMode;

use super::SkinGpu;
use super::SkinInputs;
use super::SyntheticSkin;
use super::lower_guest;
use super::synthetic_skin;
use crate::antialias::MotionValidity;
use crate::extraction::CustomMeshEntry;
use crate::extraction::MeshPose;
use crate::extraction::SkinInfluences;
use crate::mesh::Vertex;
use crate::renderer::InstanceData;
use crate::renderer::MaterialIdx;
use crate::skinning::SkinJoint;

/// Parity scene size: deliberately not a multiple of the 64-lane
/// workgroup, so the kernel's tail guard is exercised.
const PARITY_VERTICES: usize = 130;
/// Parity palette size.
const PARITY_JOINTS: usize = 3;
/// Out-of-range joint index planted in the parity scene.
const OUT_OF_RANGE_JOINT: u16 = 999;

fn bind_entry(vertices: Vec<Vertex>, influences: SkinInfluences) -> CustomMeshEntry {
    CustomMeshEntry {
        entity: ornis_core::Entity::new(7),
        vertices,
        indices: vec![0, 1, 2],
        instance: InstanceData {
            model_matrix: Mat4::IDENTITY,
            normal_matrix: Mat4::IDENTITY,
            material_index: MaterialIdx::from_raw(0),
        },
        motion_valid: MotionValidity::Valid,
        skinning: SkinningMode::Gpu,
        joint_palette: Some(vec![
            Mat4::IDENTITY.to_cols_array_2d(),
            Mat4::from_translation(Vec3::X).to_cols_array_2d(),
        ]),
        pose: MeshPose::Bind(influences),
    }
}

fn vertex(position: [f32; 3]) -> Vertex {
    Vertex {
        position,
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        tangent: [1.0, 0.0, 0.0],
    }
}

#[test]
fn pack_entry_stages_production_palette_bytes() {
    let entry = bind_entry(
        vec![vertex([1.0, 0.0, 0.0]), vertex([0.0, 1.0, 0.0])],
        SkinInfluences {
            joints: vec![[1, 0, 0, 0], [0, 0, 0, 0]],
            weights: vec![[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        },
    );
    let packed = SkinInputs::from_entry(&entry).expect("bind entry packs");
    assert_eq!((packed.vertex_count, packed.joint_count), (2, 2));
    assert_eq!(packed.bind[..6], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(packed.joints[..4], [1, 0, 0, 0]);
    // The guest palette is the production SkinJoint staging, lane for lane.
    let identity = SkinJoint::from_matrix(Mat4::IDENTITY.to_cols_array_2d());
    let expected: Vec<f32> = identity
        .matrix
        .into_iter()
        .flatten()
        .chain(identity.normal_matrix.into_iter().flatten())
        .collect();
    assert_eq!(packed.palette[..32], expected[..]);
}

#[test]
fn pack_rejects_cpu_entries_and_lane_defects() {
    let mut cpu_entry = bind_entry(
        vec![vertex([0.0; 3])],
        SkinInfluences {
            joints: vec![[0, 0, 0, 0]],
            weights: vec![[1.0, 0.0, 0.0, 0.0]],
        },
    );
    cpu_entry.pose = MeshPose::Skinned;
    assert!(SkinInputs::from_entry(&cpu_entry).is_none());
    // Length defects never stage a partial guest input.
    assert!(
        SkinInputs::pack(
            &[Mat4::IDENTITY],
            &[[0.0; 3]],
            &[],
            &[[0, 0, 0, 0]],
            &[[1.0, 0.0, 0.0, 0.0]],
        )
        .is_none()
    );
    assert!(SkinInputs::pack(&[], &[[0.0; 3]], &[[0.0; 3]], &[[0; 4]], &[[0.0; 4]]).is_none());
}

/// One parity scene with the CPU-path edge cases planted in: an
/// out-of-range joint (identity fallback) and collapsed weights
/// (slot-0 fallback) on the first two vertices.
fn parity_scene() -> SyntheticSkin {
    let mut scene = synthetic_skin(PARITY_VERTICES, PARITY_JOINTS);
    scene.joints[0] = [OUT_OF_RANGE_JOINT, 0, 0, 0];
    scene.weights[0] = [1.0, 0.0, 0.0, 0.0];
    scene.weights[1] = [0.0, 0.0, 0.0, 0.0];
    scene
}

/// The second `anim` instance: same mesh, palette posed differently.
fn second_instance(mut scene: SyntheticSkin) -> SyntheticSkin {
    let shift = Mat4::from_rotation_translation(
        glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
        Vec3::new(2.0, 0.0, -1.0),
    );
    for joint in &mut scene.palette {
        *joint = shift * *joint;
    }
    scene
}

fn assert_skin_parity(
    label: &str,
    guest: (Vec<[f32; 3]>, Vec<[f32; 3]>),
    cpu: (Vec<[f32; 3]>, Vec<[f32; 3]>),
) {
    assert_eq!(guest.0.len(), cpu.0.len(), "{label}: vertex count");
    for index in 0..cpu.0.len() {
        let position_drift =
            (Vec3::from_array(guest.0[index]) - Vec3::from_array(cpu.0[index])).length();
        let normal_drift =
            (Vec3::from_array(guest.1[index]) - Vec3::from_array(cpu.1[index])).length();
        assert!(
            position_drift < CPU_GPU_TOLERANCE,
            "{label}: vertex {index} position drifts {position_drift}"
        );
        assert!(
            normal_drift < CPU_GPU_TOLERANCE,
            "{label}: vertex {index} normal drifts {normal_drift}"
        );
    }
}

#[test]
fn guest_skin_matches_cpu_skin_for_two_instances() {
    let lowered = lower_guest();
    let Some((device, queue)) = crate::try_headless_device() else {
        assert!(
            !crate::gpu_required(),
            "ORNIS_REQUIRE_GPU is set but skin parity has no wgpu device"
        );
        return;
    };
    let gpu = SkinGpu::new(
        &device,
        &lowered.wgsl,
        PARITY_VERTICES as u32,
        PARITY_JOINTS as u32,
    );
    let first = parity_scene();
    let second = second_instance(parity_scene());
    for (label, scene) in [("first instance", &first), ("second instance", &second)] {
        let inputs = scene.guest_inputs().expect("scene packs");
        let guest = gpu
            .run(&device, &queue, &inputs)
            .expect("guest readback matches its allocation");
        assert_skin_parity(label, guest, scene.cpu_skin());
    }
    // The two instances really are posed differently (no shared-state fluke).
    assert_ne!(first.cpu_skin().0, second.cpu_skin().0);
}
