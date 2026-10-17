//! Material-slot bake pass over one imported model.

use ornis_gltf::LoadedTextures;
use ornis_texture::{CpuImage, TextureImage, TextureRole};

use super::{BakeError, BakeOrigin, BakeSettings, BakeStats, Baked, bake_texture};

/// Why [`BakeOrigin::Downgraded`] was chosen.
///
/// Recorded in [`BakeStats::last_failure`] so a `PATH` miss is visible. The
/// slot is still a valid RGBA8 container; this is not [`BakeStats::fallback`].
pub(super) const DOWNGRADE_REASON: &str =
    "no basisu on PATH; wrote RGBA8 KTX2, which does not meet the compressed-size goal";

/// Material slots [`bake_model`] bakes, in upload order.
pub(super) const BAKED_ROLES: [TextureRole; 4] = [
    TextureRole::BaseColor,
    TextureRole::MetallicRoughness,
    TextureRole::Emissive,
    TextureRole::Normal,
];

/// Bakes every decoded material image of an imported model in place.
///
/// Slots already holding KTX2 keep the chain the file authored, and a slot
/// whose bake fails keeps its decoded PNG/JPEG pixels, so the import result
/// stays uploadable either way; [`BakeStats`] says what happened. Content
/// reached by two primitives in the same role is encoded once: the second slot
/// is served from the cache.
pub fn bake_model(settings: &BakeSettings, model: &mut crate::Model) -> BakeStats {
    let mut stats = BakeStats::default();
    for primitive in &mut model.primitives {
        let textures = &mut primitive.material.textures;
        BAKED_ROLES
            .iter()
            .for_each(|role| bake_slot(settings, slot_mut(textures, *role), *role, &mut stats));
    }
    stats
}

/// Bakes one material slot when it still holds decoded pixels.
fn bake_slot(
    settings: &BakeSettings,
    slot: &mut Option<TextureImage>,
    role: TextureRole,
    stats: &mut BakeStats,
) {
    let Some(source) = slot.as_ref().and_then(decoded) else {
        return;
    };
    apply_baked(slot, bake_texture(settings, source, role), stats);
}

/// Mutable slot of one material role.
fn slot_mut(textures: &mut LoadedTextures, role: TextureRole) -> &mut Option<TextureImage> {
    match role {
        TextureRole::BaseColor => &mut textures.base_color_texture,
        TextureRole::MetallicRoughness => &mut textures.metallic_roughness_texture,
        TextureRole::Emissive => &mut textures.emissive_texture,
        TextureRole::Normal => &mut textures.normal_texture,
    }
}

/// Decoded pixels of a slot, if it still needs baking.
fn decoded(image: &TextureImage) -> Option<&CpuImage> {
    match image {
        TextureImage::Rgba8(pixels) => Some(pixels),
        TextureImage::Ktx2(_) => None,
    }
}

/// Stores a produced container, or counts the pixels left as fallback.
fn apply_baked(
    slot: &mut Option<TextureImage>,
    result: Result<Baked, BakeError>,
    stats: &mut BakeStats,
) {
    match result {
        Ok(baked) => {
            count_origin(stats, baked.origin);
            *slot = Some(TextureImage::Ktx2(baked.image));
        }
        Err(error) => {
            stats.fallback += 1;
            stats.last_failure = Some(error.to_string());
        }
    }
}

/// Adds one produced container to the counters.
fn count_origin(stats: &mut BakeStats, origin: BakeOrigin) {
    match origin {
        BakeOrigin::Cached => stats.cached += 1,
        BakeOrigin::Encoded => stats.encoded += 1,
        BakeOrigin::Authored => stats.authored += 1,
        BakeOrigin::Downgraded => count_downgrade(stats),
    }
}

/// Counts one RGBA8 fallback and records why, unless a harder failure is set.
fn count_downgrade(stats: &mut BakeStats) {
    stats.downgraded += 1;
    if stats.last_failure.is_none() {
        stats.last_failure = Some(DOWNGRADE_REASON.to_string());
    }
}
