//! The material record read into the fields the layers take.
//!
//! The host uploads one `OpenPBRMaterial` per entity as twenty `vec4` slots. The
//! pass reads it once per fragment and the light walk then reuses the scalars,
//! which is why they are gathered before the loop rather than at the point of
//! use.

use ornis_gpu_openpbr::layers::specular_aa_roughness;
use ornis_gpu_openpbr::math::v3;
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Vec3;

use crate::layout::BASE_COLOR_WORD;
use crate::layout::BASE_PARAMS_WORD;
use crate::layout::COAT_COLOR_WORD;
use crate::layout::COAT_IOR_WORD;
use crate::layout::COAT_PARAMS_WORD;
use crate::layout::EMISSION_COLOR_WORD;
use crate::layout::EMISSION_PARAMS_WORD;
use crate::layout::FUZZ_COLOR_WORD;
use crate::layout::FUZZ_PARAMS_WORD;
use crate::layout::GEOMETRY_PARAMS_WORD;
use crate::layout::GROUP;
use crate::layout::LANE_W;
use crate::layout::LANE_Y;
use crate::layout::LANE_Z;
use crate::layout::MATERIAL_STRIDE;
use crate::layout::MATERIALS_BINDING;
use crate::layout::RADIUS_SCALE_WORD;
use crate::layout::SPECULAR_COLOR_WORD;
use crate::layout::SPECULAR_PARAMS_WORD;
use crate::layout::SUBSURFACE_COLOR_WORD;
use crate::layout::SUBSURFACE_PARAMS_WORD;
use crate::layout::THIN_FILM_PARAMS_WORD;
use crate::layout::TRANSMISSION_COLOR_WORD;
use crate::layout::TRANSMISSION_PARAMS_WORD;
use crate::layout::TRANSMISSION_SCATTER_WORD;

/// `f32` word `word` of material record `base`.
///
/// The pass reads the record through this shape so the word offsets above stay
/// the only place the record layout appears.
#[inline(always)]
fn mat(materials: &Storage<u32, GROUP, MATERIALS_BINDING, ReadOnly>, base: u32, word: u32) -> f32 {
    crate::layout::material_word(materials, base, word)
}

/// Lanes `x`, `y`, `z` of the `vec4` field at `word` of the material record.
#[inline(always)]
fn mat_rgb(
    materials: &Storage<u32, GROUP, MATERIALS_BINDING, ReadOnly>,
    base: u32,
    word: u32,
) -> Vec3 {
    v3(
        mat(materials, base, word),
        mat(materials, base, word + 1),
        mat(materials, base, word + 2),
    )
}

/// A material record read into the fields the layers take.
///
/// The host uploads one `OpenPBRMaterial` per entity; the pass reads it once
/// per fragment and the light walk then reuses the scalars, which is why they
/// are gathered before the loop rather than at the point of use.
pub struct Material {
    /// Weight of the base layer.
    pub(crate) base_weight: f32,
    /// Base colour.
    pub(crate) base_color: Vec3,
    /// Metalness.
    pub(crate) metalness: f32,
    /// Diffuse roughness before the specular floor.
    pub(crate) diffuse_roughness: f32,
    /// Weight of the specular lobe.
    pub(crate) specular_weight: f32,
    /// Roughness after the geometric specular antialiasing.
    pub(crate) specular_roughness: f32,
    /// Index of refraction of the specular lobe.
    pub(crate) specular_ior: f32,
    /// Anisotropy of the specular lobe.
    pub(crate) specular_anisotropy: f32,
    /// Edge tint of the specular lobe.
    pub(crate) specular_edge_tint: Vec3,
    /// Weight of the transmission lobe.
    pub(crate) transmission_weight: f32,
    /// Depth the transmission colour is integrated over.
    pub(crate) transmission_depth: f32,
    /// Dispersion scale of the transmission lobe.
    pub(crate) transmission_dispersion_scale: f32,
    /// Abbe number of the dispersion.
    pub(crate) transmission_dispersion_abbe: f32,
    /// Transmission tint.
    pub(crate) transmission_color: Vec3,
    /// Scattering tint inside the medium.
    pub(crate) transmission_scatter: Vec3,
    /// Anisotropy of that scattering.
    pub(crate) transmission_scatter_anisotropy: f32,
    /// Weight of the subsurface lobe.
    pub(crate) subsurface_weight: f32,
    /// Base scattering radius.
    pub(crate) subsurface_radius: f32,
    /// Red scale of the radius.
    pub(crate) subsurface_radius_scale_r: f32,
    /// Green scale of the radius.
    pub(crate) subsurface_radius_scale_g: f32,
    /// Blue scale of the radius.
    pub(crate) subsurface_radius_scale_b: f32,
    /// Anisotropy of the subsurface phase.
    pub(crate) subsurface_scatter_anisotropy: f32,
    /// Subsurface colour.
    pub(crate) subsurface_color: Vec3,
    /// Weight of the fuzz lobe.
    pub(crate) fuzz_weight: f32,
    /// Roughness of the fuzz lobe.
    pub(crate) fuzz_roughness: f32,
    /// Fuzz colour.
    pub(crate) fuzz_color: Vec3,
    /// Weight of the coat layer.
    pub(crate) coat_weight: f32,
    /// Roughness of the coat layer.
    pub(crate) coat_roughness: f32,
    /// Anisotropy of the coat layer.
    pub(crate) coat_anisotropy: f32,
    /// How far the coat darkens what is under it.
    pub(crate) coat_darkening: f32,
    /// Coat tint.
    pub(crate) coat_color: Vec3,
    /// Index of refraction of the coat.
    pub(crate) coat_ior: f32,
    /// Weight of the thin-film modulation.
    pub(crate) thin_film_weight: f32,
    /// Thin-film thickness in nanometers.
    pub(crate) thin_film_thickness_um: f32,
    /// Index of refraction of the thin film.
    pub(crate) thin_film_ior: f32,
    /// Emissive luminance.
    pub(crate) emission_luminance: f32,
    /// Emissive colour.
    pub(crate) emission_color: Vec3,
    /// Opacity, written straight to alpha.
    pub(crate) opacity: f32,
    /// `geometry_params.y`: `1` for a wall thinner than the ray path.
    pub(crate) thin_walled: f32,
    /// `geometry_params.z`: at or above `UNLIT_THRESHOLD` the surface is unlit.
    pub(crate) unlit: f32,
}

impl Material {
    /// Read the record of `material_index` out of the material table.
    ///
    /// `specular_roughness` is filtered by the screen-space slope of the
    /// normal: a surface whose normal changes within the pixel cannot carry a
    /// mirror highlight, and the filter is what removes the shimmer.
    ///
    /// The thin-film thickness arrives in micrometers and the modulation wants
    /// nanometers, so the scale is applied here rather than in the layer.
    pub(crate) fn load(
        materials: &Storage<u32, GROUP, MATERIALS_BINDING, ReadOnly>,
        material_index: u32,
        n: Vec3,
    ) -> Self {
        let base = material_index * MATERIAL_STRIDE;
        Self {
            base_weight: mat(materials, base, BASE_PARAMS_WORD),
            diffuse_roughness: mat(materials, base, BASE_PARAMS_WORD + LANE_Y),
            metalness: mat(materials, base, BASE_PARAMS_WORD + LANE_Z),
            base_color: mat_rgb(materials, base, BASE_COLOR_WORD),
            specular_weight: mat(materials, base, SPECULAR_PARAMS_WORD),
            specular_roughness: specular_aa_roughness(
                mat(materials, base, SPECULAR_PARAMS_WORD + LANE_Y),
                n,
            ),
            specular_ior: mat(materials, base, SPECULAR_PARAMS_WORD + LANE_Z),
            specular_anisotropy: mat(materials, base, SPECULAR_PARAMS_WORD + LANE_W),
            specular_edge_tint: mat_rgb(materials, base, SPECULAR_COLOR_WORD),
            transmission_weight: mat(materials, base, TRANSMISSION_PARAMS_WORD),
            transmission_depth: mat(materials, base, TRANSMISSION_PARAMS_WORD + LANE_Y),
            transmission_dispersion_scale: mat(materials, base, TRANSMISSION_PARAMS_WORD + LANE_Z),
            transmission_dispersion_abbe: mat(materials, base, TRANSMISSION_PARAMS_WORD + LANE_W),
            transmission_color: mat_rgb(materials, base, TRANSMISSION_COLOR_WORD),
            transmission_scatter: mat_rgb(materials, base, TRANSMISSION_SCATTER_WORD),
            transmission_scatter_anisotropy: mat(
                materials,
                base,
                TRANSMISSION_SCATTER_WORD + LANE_W,
            ),
            subsurface_weight: mat(materials, base, SUBSURFACE_PARAMS_WORD),
            subsurface_radius: mat(materials, base, SUBSURFACE_PARAMS_WORD + LANE_Y),
            subsurface_radius_scale_r: mat(materials, base, SUBSURFACE_PARAMS_WORD + LANE_Z),
            subsurface_scatter_anisotropy: mat(materials, base, SUBSURFACE_PARAMS_WORD + LANE_W),
            subsurface_color: mat_rgb(materials, base, SUBSURFACE_COLOR_WORD),
            subsurface_radius_scale_g: mat(materials, base, RADIUS_SCALE_WORD),
            subsurface_radius_scale_b: mat(materials, base, RADIUS_SCALE_WORD + LANE_Y),
            fuzz_weight: mat(materials, base, FUZZ_PARAMS_WORD),
            fuzz_roughness: mat(materials, base, FUZZ_PARAMS_WORD + LANE_Y),
            fuzz_color: mat_rgb(materials, base, FUZZ_COLOR_WORD),
            coat_weight: mat(materials, base, COAT_PARAMS_WORD),
            coat_roughness: mat(materials, base, COAT_PARAMS_WORD + LANE_Y),
            coat_anisotropy: mat(materials, base, COAT_PARAMS_WORD + LANE_Z),
            coat_darkening: mat(materials, base, COAT_PARAMS_WORD + LANE_W),
            coat_color: mat_rgb(materials, base, COAT_COLOR_WORD),
            coat_ior: mat(materials, base, COAT_IOR_WORD),
            thin_film_weight: mat(materials, base, THIN_FILM_PARAMS_WORD),
            thin_film_thickness_um: mat(materials, base, THIN_FILM_PARAMS_WORD + LANE_Y),
            thin_film_ior: mat(materials, base, THIN_FILM_PARAMS_WORD + LANE_Z),
            emission_luminance: mat(materials, base, EMISSION_PARAMS_WORD),
            emission_color: mat_rgb(materials, base, EMISSION_COLOR_WORD),
            opacity: mat(materials, base, GEOMETRY_PARAMS_WORD),
            thin_walled: mat(materials, base, GEOMETRY_PARAMS_WORD + LANE_Y),
            unlit: mat(materials, base, GEOMETRY_PARAMS_WORD + LANE_Z),
        }
    }
}
