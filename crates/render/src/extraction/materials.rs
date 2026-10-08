//! Material conversion from scene descriptions to GPU-ready OpenPBR rows.

use super::ExtractionStats;
use super::FrameUpload;
use ornis_assets::scene::MaterialDesc;
use ornis_core::OpenPBRMaterial;
use ornis_core::material::ShadingMode;

/// Returns the [`FrameUpload::materials`] index for `material`,
/// pushing its GPU conversion only on first sight (exact `PartialEq`
/// dedup — identical [`MaterialDesc`] values with the same
/// [`ornis_core::Surface`] share one entry). Every reuse of an existing
/// entry bumps [`ExtractionStats::materials_deduped`].
///
/// `surface` is applied after [`material_to_gpu`], so a glTF metalness
/// value (the assets track writes it as a number) is the input and
/// the override replaces only the three surface slots.
pub(super) fn deduped_material_index(
    extracted: &mut FrameUpload,
    seen: &mut Vec<(MaterialDesc, Option<ornis_core::Surface>)>,
    material: &MaterialDesc,
    surface: Option<ornis_core::Surface>,
    stats: &mut ExtractionStats,
) -> crate::renderer::MaterialIdx {
    if let Some(index) = seen
        .iter()
        .position(|(known, known_surface)| known == material && *known_surface == surface)
    {
        stats.materials_deduped += 1;
        return crate::renderer::MaterialIdx::from(index as u32);
    }
    seen.push((material.clone(), surface));
    let mut gpu = material_to_gpu(material);
    if let Some(surface) = surface {
        apply_surface_override(&mut gpu, &surface);
    }
    extracted.materials.push(gpu);
    crate::renderer::MaterialIdx::from_raw(extracted.materials.len() as u32 - 1)
}
/// `specular.params[0]`: specular lobe weight.
const SPECULAR_WEIGHT_SLOT: usize = 0;
/// `specular.params[1]`: specular roughness.
const SPECULAR_ROUGHNESS_SLOT: usize = 1;
/// `base.params[2]`: metalness. The assets track should write the glTF
/// factor here (`OpenPBRMaterial::base.metalness`) before this override.
const BASE_METALNESS_SLOT: usize = 2;
/// Rewrites roughness, metalness, and specular weight on a GPU material.
///
/// Called after [`material_to_gpu`]. Color, IOR (`specular.params[2]`),
/// and emission are left as the preset wrote them.
fn apply_surface_override(material: &mut OpenPBRMaterial, surface: &ornis_core::Surface) {
    material.specular.params[SPECULAR_ROUGHNESS_SLOT] = surface.roughness.get();
    material.base.params[BASE_METALNESS_SLOT] = surface.metallic.get();
    material.specular.params[SPECULAR_WEIGHT_SLOT] = surface.specular.get();
}
fn material_to_gpu(material: &MaterialDesc) -> OpenPBRMaterial {
    match material {
        // `dielectric()` and `metal()` differ only in metalness once color
        // and roughness are overwritten. Both presets start from the
        // dielectric recipe and write the continuous factor.
        MaterialDesc::Dielectric {
            base_color,
            roughness,
            emission,
            ..
        }
        | MaterialDesc::Metal {
            base_color,
            roughness,
            emission,
            ..
        } => {
            let mut output = OpenPBRMaterial::dielectric();
            output.base.color_rgb(*base_color);
            output.specular.roughness(roughness.get());
            output.base.metalness(material.metallic_units().get());
            apply_emission(&mut output, *emission);
            output
        }
        MaterialDesc::Coat {
            base_color,
            coat_weight,
            coat_roughness,
            emission,
            ..
        } => {
            let mut output = OpenPBRMaterial::coat();
            output.base.color_rgb(*base_color);
            output.coat.weight(coat_weight.get());
            output.coat.roughness(coat_roughness.get());
            output.base.metalness(material.metallic_units().get());
            apply_emission(&mut output, *emission);
            output
        }
        MaterialDesc::Matte {
            base_color,
            roughness,
            ..
        } => {
            let mut output = OpenPBRMaterial::dielectric();
            output.base.color_rgb(*base_color);
            output.base.diffuse_roughness(roughness.get());
            output.base.metalness(material.metallic_units().get());
            // Matte is diffuse-only: no specular lobe.
            output.specular.weight(0.0);
            output
        }
        MaterialDesc::Glass {
            base_color,
            roughness,
            ior,
            ..
        } => {
            let mut output = OpenPBRMaterial::glass();
            output.transmission.color_rgb(*base_color);
            output.specular.roughness(roughness.get());
            output.specular.ior(ior.get());
            output.base.metalness(material.metallic_units().get());
            output
        }
        MaterialDesc::Unlit { color } => {
            // Unlit sprite: no BRDF lobe (zero weights, black base), the
            // sprite color travels as emission, and the shading-mode flag
            // tells both evaluators (forward + deferred) to output the
            // emission alone — no light, shadow or IBL term. Tonemap
            // applies downstream exactly as for emission.
            let mut output = OpenPBRMaterial::dielectric();
            output.base.color_rgb([0.0, 0.0, 0.0]);
            output.base.weight(0.0);
            output.specular.weight(0.0);
            output.base.metalness(0.0);
            apply_emission(&mut output, *color);
            output.geometry.set_shading(ShadingMode::Unlit);
            output
        }
    }
}
/// Maps an emissive RGB triple onto the OpenPBR emission group.
///
/// The shader evaluates `emission_color * emission_luminance / PI`, so the
/// peak channel becomes the luminance (nits) and the color carries the
/// normalized chromaticity — `luminance * color` reproduces the input
/// exactly. Black input leaves the preset default (luminance 0 = off).
fn apply_emission(output: &mut OpenPBRMaterial, emission: [f32; 3]) {
    let peak = emission[0].max(emission[1]).max(emission[2]).max(0.0);
    if peak > 0.0 {
        output.emission.luminance(peak);
        output
            .emission
            .color_rgb([emission[0] / peak, emission[1] / peak, emission[2] / peak]);
    }
}
#[cfg(test)]
mod tests {
    use super::super::RenderLights;
    use super::super::extract_render_data;
    use super::super::extract_render_data_with_stats;
    use super::*;
    use ornis_assets::scene::MaterialDesc;
    use ornis_assets::scene::MeshDesc;
    use ornis_assets::scene::TransformDesc;
    use ornis_core::Engine;
    use ornis_core::Metallic;
    use ornis_core::OpenPBRMaterial;
    use ornis_core::Surface;
    use ornis_core::units::Clamped01;
    use ornis_core::units::EnvironmentWeight;
    use ornis_core::units::Ior;
    use ornis_core::units::PositiveF32;
    use ornis_core::units::Roughness;
    use ornis_core::units::Specular;

    /// `specular.params[2]` is IOR and must survive a surface override.
    const SPECULAR_IOR_SLOT: usize = 2;

    #[test]
    fn surface_override_rewrites_three_slots_and_dedups_with_it() {
        let dielectric = MaterialDesc::Dielectric {
            base_color: [0.2, 0.4, 0.6],
            roughness: Clamped01::new(0.7),
            emission: [0.3, 0.0, 0.0],
            metallic: ornis_core::Metallic::new(0.0),
        };
        let surface = Surface {
            roughness: Roughness::new(0.15),
            metallic: Metallic::new(0.3),
            specular: Specular::new(0.4),
        };
        let mut store = ornis_core::SmartStore::new();
        let plain = store.create_entity();
        let overridden = store.create_entity();
        let again = store.create_entity();
        for entity in [plain, overridden, again] {
            store.insert(entity, TransformDesc::from_translation(glam::Vec3::ZERO));
            store.insert(
                entity,
                MeshDesc::Sphere {
                    radius: PositiveF32::expect_valid(1.0),
                    segments: 8,
                    rings: 6,
                },
            );
            store.insert(entity, dielectric.clone());
        }
        store.insert(overridden, surface);
        store.insert(again, surface);

        let (upload, stats) = extract_render_data_with_stats(&store);
        assert_eq!(
            upload.materials.len(),
            2,
            "override must not share the glTF slot"
        );
        assert_eq!(
            stats.materials_deduped, 1,
            "identical overrides share one slot"
        );
        let plain_gpu = upload.materials[upload.instances[0].material_index.index()];
        let over_gpu = upload.materials[upload.instances[1].material_index.index()];
        assert_eq!(
            upload.instances[2].material_index,
            upload.instances[1].material_index
        );
        assert_eq!(plain_gpu.specular.params[SPECULAR_ROUGHNESS_SLOT], 0.7);
        assert_eq!(plain_gpu.base.params[BASE_METALNESS_SLOT], 0.0);
        assert_eq!(plain_gpu.specular.params[SPECULAR_WEIGHT_SLOT], 1.0);
        assert_eq!(over_gpu.specular.params[SPECULAR_ROUGHNESS_SLOT], 0.15);
        assert_eq!(over_gpu.base.params[BASE_METALNESS_SLOT], 0.3);
        assert_eq!(over_gpu.specular.params[SPECULAR_WEIGHT_SLOT], 0.4);
        assert_eq!(over_gpu.base.color, plain_gpu.base.color);
        assert_eq!(
            over_gpu.specular.params[SPECULAR_IOR_SLOT],
            plain_gpu.specular.params[SPECULAR_IOR_SLOT]
        );
        assert_eq!(over_gpu.emission.params, plain_gpu.emission.params);
        assert_eq!(over_gpu.emission.color, plain_gpu.emission.color);
    }

    #[test]
    fn missing_environment_weight_stays_automatic() {
        let full = ron::ser::to_string(&RenderLights::default()).expect("serialize");
        let key = "environment_weight:";
        let start = full.find(key).expect("field is serialized");
        let tail = &full[start + key.len()..];
        let end = tail.find([',', ')']).expect("field terminator");
        let mut stripped = String::new();
        stripped.push_str(full[..start].trim_end_matches(|c: char| c == ',' || c.is_whitespace()));
        stripped.push_str(&tail[end..]);
        let back: RenderLights = ron::de::from_str(&stripped).expect("legacy payload");
        assert!(back.environment_weight.is_none());
        let explicit = RenderLights {
            environment_weight: Some(EnvironmentWeight::new(1.4)),
            ..RenderLights::default()
        };
        let round: RenderLights =
            ron::de::from_str(&ron::ser::to_string(&explicit).expect("ser")).expect("de");
        assert_eq!(
            round.environment_weight.map(EnvironmentWeight::get),
            Some(1.0)
        );
    }

    #[test]
    fn identical_materials_deduplicate_to_one_entry() {
        // Three entities with the same `MaterialDesc` share one
        // `FrameUpload::materials` entry; a distinct material adds a
        // second, and every instance index points at its own entry.
        let shared = MaterialDesc::Metal {
            base_color: [0.9, 0.7, 0.1],
            roughness: Clamped01::new(0.2),
            emission: [0.0, 0.0, 0.0],
            metallic: ornis_core::Metallic::new(1.0),
        };
        let other = MaterialDesc::Matte {
            base_color: [0.2, 0.2, 0.2],
            roughness: Clamped01::new(0.8),
            metallic: ornis_core::Metallic::new(0.0),
        };
        let mut engine = Engine::new();
        for (i, material) in [shared.clone(), shared.clone(), shared.clone(), other]
            .into_iter()
            .enumerate()
        {
            let store = engine.world_mut().store_mut().expect("store");
            let handle = store.create_entity();
            store.insert(
                handle,
                TransformDesc::from_translation(glam::Vec3::new(i as f32, 0.0, 0.0)),
            );
            store.insert(
                handle,
                MeshDesc::Sphere {
                    radius: PositiveF32::expect_valid(1.0),
                    segments: 16,
                    rings: 12,
                },
            );
            store.insert(handle, material);
        }

        let extracted = extract_render_data(engine.world().store().expect("store"));
        assert_eq!(extracted.instances.len(), 4);
        assert_eq!(extracted.materials.len(), 2, "3 identical + 1 distinct");
        assert_eq!(
            extracted
                .instances
                .iter()
                .map(|instance| instance.material_index)
                .collect::<Vec<_>>(),
            vec![
                crate::renderer::MaterialIdx::from_raw(0),
                crate::renderer::MaterialIdx::from_raw(0),
                crate::renderer::MaterialIdx::from_raw(0),
                crate::renderer::MaterialIdx::from_raw(1)
            ],
            "shared entries reuse the first index"
        );
    }

    #[test]
    fn partial_metallic_reaches_gpu_metalness_and_edges_match_presets() {
        // 0.3 is written as metalness, with albedo, roughness and emission
        // still applied. Extraction uses the same conversion, so a stored
        // description reaches the frame upload unchanged when no
        // `Surface` override is present.
        let partial = MaterialDesc::Dielectric {
            base_color: [0.2, 0.4, 0.6],
            roughness: Clamped01::new(0.4),
            emission: [0.2, 0.0, 0.1],
            metallic: Metallic::new(0.3),
        };
        let gpu = material_to_gpu(&partial);
        assert!((gpu.base.params[BASE_METALNESS_SLOT] - 0.3).abs() < 1e-6);
        assert_eq!(&gpu.base.color[..3], &[0.2, 0.4, 0.6]);
        assert!((gpu.specular.params[SPECULAR_ROUGHNESS_SLOT] - 0.4).abs() < 1e-6);
        assert_eq!(gpu.emission.params[0], 0.2);

        let mut store = ornis_core::SmartStore::new();
        let entity = store.create_entity();
        store.insert(entity, TransformDesc::from_translation(glam::Vec3::ZERO));
        store.insert(
            entity,
            MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(1.0),
                segments: 8,
                rings: 6,
            },
        );
        store.insert(entity, partial);
        let upload = extract_render_data(&store);
        let extracted = upload.materials[upload.instances[0].material_index.index()];
        assert!((extracted.base.params[BASE_METALNESS_SLOT] - 0.3).abs() < 1e-6);

        let dielectric = material_to_gpu(&MaterialDesc::Dielectric {
            base_color: [0.2, 0.4, 0.6],
            roughness: Clamped01::new(0.4),
            emission: [0.0, 0.0, 0.0],
            metallic: Metallic::new(0.0),
        });
        let mut expected = OpenPBRMaterial::dielectric();
        expected.base.color_rgb([0.2, 0.4, 0.6]);
        expected.specular.roughness(0.4);
        assert_eq!(
            bytemuck::bytes_of(&dielectric),
            bytemuck::bytes_of(&expected)
        );

        let metal = material_to_gpu(&MaterialDesc::Metal {
            base_color: [0.9, 0.7, 0.1],
            roughness: Clamped01::new(0.2),
            emission: [0.0, 0.0, 0.0],
            metallic: Metallic::new(1.0),
        });
        let mut expected_metal = OpenPBRMaterial::metal();
        expected_metal.base.color_rgb([0.9, 0.7, 0.1]);
        expected_metal.specular.roughness(0.2);
        assert_eq!(
            bytemuck::bytes_of(&metal),
            bytemuck::bytes_of(&expected_metal)
        );
    }

    #[test]
    fn material_to_gpu_maps_emission_and_matte() {
        // Emission `[2, 1, 0.5]`: peak 2 nits, normalized chromaticity.
        let gpu = material_to_gpu(&MaterialDesc::Dielectric {
            base_color: [0.5, 0.5, 0.5],
            roughness: Clamped01::new(0.9),
            emission: [2.0, 1.0, 0.5],
            metallic: ornis_core::Metallic::new(0.0),
        });
        assert_eq!(gpu.emission.params[0], 2.0);
        assert_eq!(gpu.emission.color[0], 1.0);
        assert_eq!(gpu.emission.color[1], 0.5);
        assert_eq!(gpu.emission.color[2], 0.25);
        // Black emission leaves the preset default (luminance 0 = off).
        let off = material_to_gpu(&MaterialDesc::Metal {
            base_color: [0.9, 0.7, 0.1],
            roughness: Clamped01::new(0.2),
            emission: [0.0, 0.0, 0.0],
            metallic: ornis_core::Metallic::new(1.0),
        });
        assert_eq!(off.emission.params[0], 0.0);
        // Matte: diffuse albedo + roughness, no specular lobe.
        let matte = material_to_gpu(&MaterialDesc::Matte {
            base_color: [0.2, 0.4, 0.6],
            roughness: Clamped01::new(0.7),
            metallic: ornis_core::Metallic::new(0.0),
        });
        assert_eq!(matte.base.color[0], 0.2);
        assert_eq!(matte.base.color[1], 0.4);
        assert_eq!(matte.base.color[2], 0.6);
        assert_eq!(matte.base.params[2], 0.0, "metalness 0");
        assert!((matte.base.params[1] - 0.7).abs() < f32::EPSILON);
        assert_eq!(matte.specular.params[0], 0.0, "no specular lobe");
    }

    #[test]
    fn material_to_gpu_maps_glass() {
        // Glass tint → transmission color, roughness/ior → specular lobe;
        // the preset keeps full transmission over thin walls.
        let gpu = material_to_gpu(&MaterialDesc::Glass {
            base_color: [0.9, 0.95, 1.0],
            roughness: Clamped01::new(0.05),
            ior: Ior::new(1.33),
            metallic: ornis_core::Metallic::new(0.0),
        });
        assert_eq!(gpu.transmission.color[0], 0.9);
        assert_eq!(gpu.transmission.color[1], 0.95);
        assert_eq!(gpu.transmission.color[2], 1.0);
        assert_eq!(gpu.transmission.params[0], 1.0, "full transmission");
        assert!((gpu.specular.params[1] - 0.05).abs() < f32::EPSILON);
        assert!((gpu.specular.params[2] - 1.33).abs() < f32::EPSILON);
    }

    #[test]
    fn material_to_gpu_maps_unlit_to_emission_with_flag() {
        // Unlit sprite: black base with no BRDF lobe, the sprite color as
        // emission (peak-luminance mapping, like the other presets), and
        // the unlit shading-mode flag so both evaluators output the
        // emission alone.
        use ornis_core::material::ShadingMode;
        use ornis_core::units::LinearRgb;

        let gpu = material_to_gpu(&MaterialDesc::unlit_units(LinearRgb::new([0.2, 0.4, 0.8])));
        assert_eq!(&gpu.base.color[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(gpu.base.color[3], 1.0, "opaque alpha survives");
        assert_eq!(gpu.base.params[0], 0.0, "no diffuse lobe");
        assert_eq!(gpu.specular.params[0], 0.0, "no specular lobe");
        assert_eq!(gpu.base.params[2], 0.0, "no metalness");
        assert_eq!(gpu.emission.params[0], 0.8, "peak channel is the luminance");
        assert_eq!(gpu.emission.color[0], 0.25);
        assert_eq!(gpu.emission.color[1], 0.5);
        assert_eq!(gpu.emission.color[2], 1.0);
        assert_eq!(gpu.geometry.shading(), ShadingMode::Unlit);
        assert_eq!(gpu.geometry.params[2], 1.0);
        // Black unlit stays black (emission off) but keeps the flag.
        let black = material_to_gpu(&MaterialDesc::unlit_units(LinearRgb::new([0.0, 0.0, 0.0])));
        assert_eq!(black.emission.params[0], 0.0);
        assert_eq!(black.geometry.shading(), ShadingMode::Unlit);
        // Lit presets never set the flag (golden frames stay bit-identical).
        let lit = material_to_gpu(&MaterialDesc::Dielectric {
            base_color: [0.5, 0.5, 0.5],
            roughness: Clamped01::new(0.9),
            emission: [0.0, 0.0, 0.0],
            metallic: ornis_core::Metallic::new(0.0),
        });
        assert_eq!(lit.geometry.shading(), ShadingMode::Lit);
    }

    #[test]
    fn three_identical_glass_materials_share_one_entry() {
        // Spec case: 3 entities with the same `MaterialDesc` → one
        // `FrameUpload::materials` entry, every instance pointing at it.
        let shared = MaterialDesc::Glass {
            base_color: [0.9, 0.95, 1.0],
            roughness: Clamped01::new(0.05),
            ior: Ior::new(1.5),
            metallic: ornis_core::Metallic::new(0.0),
        };
        let mut engine = Engine::new();
        for i in 0..3 {
            let store = engine.world_mut().store_mut().expect("store");
            let handle = store.create_entity();
            store.insert(
                handle,
                TransformDesc::from_translation(glam::Vec3::new(i as f32, 0.0, 0.0)),
            );
            store.insert(
                handle,
                MeshDesc::Sphere {
                    radius: PositiveF32::expect_valid(1.0),
                    segments: 16,
                    rings: 12,
                },
            );
            store.insert(handle, shared.clone());
        }

        let extracted = extract_render_data(engine.world().store().expect("store"));
        assert_eq!(extracted.instances.len(), 3);
        assert_eq!(extracted.materials.len(), 1, "3 identical → 1 entry");
        assert!(
            extracted.instances.iter().all(
                |instance| instance.material_index == crate::renderer::MaterialIdx::from_raw(0)
            )
        );
    }
}
