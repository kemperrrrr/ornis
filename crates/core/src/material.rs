//! OpenPBR material definition.
//!
//! The material is a flat, GPU-friendly blob of 20 `vec4` slots, but its
//! setter surface is split across parameter-group structs (`BaseGroup`,
//! `SpecularGroup`, …) so each concern owns a small, focused set of
//! builders instead of one type carrying dozens of methods. The nested
//! `#[repr(C)]` groups preserve the exact flat memory layout expected by
//! the WGSL uniform/storage buffers.

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Base (diffuse) parameters — GPU slot 0.
pub struct BaseGroup {
    /// Scalar params: `[weight, diffuse_roughness, metalness, reserved]`.
    pub params: [f32; 4],
    /// Linear base color (RGBA).
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Specular reflection parameters — GPU slot 1.
pub struct SpecularGroup {
    /// Scalar params: `[weight, roughness, ior, anisotropy]`.
    pub params: [f32; 4],
    /// Metallic edge tint (RGB; alpha unused).
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Transmission (refraction) parameters — GPU slot 2.
pub struct TransmissionGroup {
    /// Scalar params: `[weight, depth, dispersion_scale, dispersion_abbe]`.
    pub params: [f32; 4],
    /// Transmitted tint color (RGB) with `anisotropy` in alpha.
    pub color: [f32; 4],
    /// Volume scattering color (RGB) with `anisotropy` in alpha.
    pub scatter: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Subsurface scattering parameters — GPU slot 3.
pub struct SubsurfaceGroup {
    /// Scalar params: `[weight, radius, radius_scale_r, scatter_anisotropy]`.
    pub params: [f32; 4],
    /// Subsurface scattering color (RGBA).
    pub color: [f32; 4],
    /// Per-channel radius scales `[g, b, _, _]`; red scale lives in `params[2]`.
    pub radius_scale_gb: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Fuzz (sheen) parameters — GPU slot 4.
pub struct FuzzGroup {
    /// Scalar params: `[weight, roughness, reserved]`.
    pub params: [f32; 4],
    /// Fuzz tint color (RGB; alpha unused).
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Clearcoat parameters — GPU slot 5.
pub struct CoatGroup {
    /// Scalar params: `[weight, roughness, anisotropy, darkening]`.
    pub params: [f32; 4],
    /// Coat tint color (RGB; alpha unused).
    pub color: [f32; 4],
    /// Coat index of refraction in `x` (rest unused).
    pub ior: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Thin-film interference parameters — GPU slot 6.
pub struct ThinFilmGroup {
    /// Scalar params: `[weight, thickness_um, ior, reserved]`.
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Emission parameters — GPU slot 7.
pub struct EmissionGroup {
    /// Emission luminance in nits at `x` (rest reserved).
    pub params: [f32; 4],
    /// Emission color (RGBA).
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Geometry (opacity/thin-walled) parameters — GPU slot 8.
pub struct GeometryGroup {
    /// First param vec4: `[opacity, thin_walled_flag, reserved]`.
    pub params: [f32; 4],
    /// Second reserved param vec4 kept for 16-byte alignment.
    pub params2: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Full OpenPBR material as a flat `#[repr(C)]` blob of 20 `vec4` slots (320 bytes),
/// matching the WGSL uniform layout expected by the renderer.
pub struct OpenPBRMaterial {
    /// Diffuse base parameters.
    pub base: BaseGroup,
    /// Specular reflection parameters.
    pub specular: SpecularGroup,

    /// Refraction/transmission parameters.
    pub transmission: TransmissionGroup,

    /// Subsurface scattering parameters.
    pub subsurface: SubsurfaceGroup,

    /// Fuzz (sheen) parameters.
    pub fuzz: FuzzGroup,

    /// Clearcoat parameters.
    pub coat: CoatGroup,

    /// Thin-film parameters.
    pub thin_film: ThinFilmGroup,

    /// Emission parameters.
    pub emission: EmissionGroup,

    /// Geometry (opacity) parameters.
    pub geometry: GeometryGroup,
}

impl BaseGroup {
    /// Sets diffuse weight (clamped to \[0, 1\]).
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    /// Sets diffuse roughness (clamped to \[0, 1\]).
    pub fn diffuse_roughness(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.clamp(0.0, 1.0);
        self
    }
    /// Sets metalness (clamped to \[0, 1\]).
    pub fn metalness(&mut self, v: f32) -> &mut Self {
        self.params[2] = v.clamp(0.0, 1.0);
        self
    }

    /// Sets base color from explicit RGBA components.
    pub fn color(&mut self, r: f32, g: f32, b: f32, a: f32) -> &mut Self {
        self.color = [r, g, b, a];
        self
    }
    /// Sets base color from an RGB array (alpha unchanged).
    pub fn color_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }
}

impl SpecularGroup {
    /// Sets specular weight (clamped to \[0, 1\]).
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    /// Sets specular roughness (clamped to \[0, 1\]).
    pub fn roughness(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.clamp(0.0, 1.0);
        self
    }
    /// Sets index of refraction (at least 1.0).
    pub fn ior(&mut self, v: f32) -> &mut Self {
        self.params[2] = v.max(1.0);
        self
    }
    /// Sets specular anisotropy (clamped to \[0, 1\]).
    pub fn anisotropy(&mut self, v: f32) -> &mut Self {
        self.params[3] = v.clamp(0.0, 1.0);
        self
    }

    /// Sets metallic edge tint from explicit RGB components.
    pub fn edge_tint(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color = [r, g, b, 0.0];
        self
    }
    /// Sets metallic edge tint from an RGB array.
    pub fn edge_tint_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }
}

impl TransmissionGroup {
    /// Sets transmission weight (clamped to \[0, 1\]).
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    /// Sets absorption depth (non-negative).
    pub fn depth(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.max(0.0);
        self
    }
    /// Sets chromatic dispersion strength (non-negative).
    pub fn dispersion_scale(&mut self, v: f32) -> &mut Self {
        self.params[2] = v.max(0.0);
        self
    }
    /// Sets Abbe number controlling dispersion spread (non-negative).
    pub fn dispersion_abbe(&mut self, v: f32) -> &mut Self {
        self.params[3] = v.max(0.0);
        self
    }

    /// Sets transmission tint from explicit RGB components.
    pub fn color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color = [r, g, b, 0.0];
        self
    }
    /// Sets transmission tint from an RGB array.
    pub fn color_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }

    /// Sets volume scattering color from explicit RGB components.
    pub fn scatter_color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.scatter = [r, g, b, 0.0];
        self
    }
    /// Sets volume scattering anisotropy (clamped to \[-1, 1\]).
    pub fn scatter_anisotropy(&mut self, v: f32) -> &mut Self {
        self.scatter[3] = v.clamp(-1.0, 1.0);
        self
    }
}

impl SubsurfaceGroup {
    /// Sets subsurface weight (clamped to \[0, 1\]).
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    /// Sets mean free path radius (non-negative).
    pub fn radius(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.max(0.0);
        self
    }
    /// Sets red-channel radius scale (non-negative).
    pub fn radius_scale_r(&mut self, v: f32) -> &mut Self {
        self.params[2] = v.max(0.0);
        self
    }
    /// Sets scattering anisotropy (clamped to \[-1, 1\]).
    pub fn scatter_anisotropy(&mut self, v: f32) -> &mut Self {
        self.params[3] = v.clamp(-1.0, 1.0);
        self
    }

    /// Sets subsurface color from explicit RGB components.
    pub fn color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color = [r, g, b, 1.0];
        self
    }
    /// Sets subsurface color from an RGB array.
    pub fn color_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }

    /// Sets green-channel radius scale (non-negative).
    pub fn radius_scale_g(&mut self, v: f32) -> &mut Self {
        self.radius_scale_gb[0] = v.max(0.0);
        self
    }
    /// Sets blue-channel radius scale (non-negative).
    pub fn radius_scale_b(&mut self, v: f32) -> &mut Self {
        self.radius_scale_gb[1] = v.max(0.0);
        self
    }
}

impl FuzzGroup {
    /// Sets fuzz weight (clamped to \[0, 1\]).
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    /// Sets fuzz roughness (clamped to \[0, 1\]).
    pub fn roughness(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.clamp(0.0, 1.0);
        self
    }

    /// Sets fuzz tint from explicit RGB components.
    pub fn color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color = [r, g, b, 0.0];
        self
    }
    /// Sets fuzz tint from an RGB array.
    pub fn color_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }
}

impl CoatGroup {
    /// Sets coat weight (clamped to \[0, 1\]).
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    /// Sets coat roughness (clamped to \[0, 1\]).
    pub fn roughness(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.clamp(0.0, 1.0);
        self
    }
    /// Sets coat anisotropy (clamped to \[0, 1\]).
    pub fn anisotropy(&mut self, v: f32) -> &mut Self {
        self.params[2] = v.clamp(0.0, 1.0);
        self
    }
    /// Sets coat darkening (clamped to \[0, 1\]).
    pub fn darkening(&mut self, v: f32) -> &mut Self {
        self.params[3] = v.clamp(0.0, 1.0);
        self
    }

    /// Sets coat tint from explicit RGB components.
    pub fn color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color = [r, g, b, 0.0];
        self
    }
    /// Sets coat tint from an RGB array.
    pub fn color_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }

    /// Sets coat index of refraction (at least 1.0).
    pub fn ior(&mut self, v: f32) -> &mut Self {
        self.ior[0] = v.max(1.0);
        self
    }
}

impl ThinFilmGroup {
    /// Sets thin-film weight (clamped to \[0, 1\]).
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    /// Sets film thickness in micrometers (non-negative).
    pub fn thickness_um(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.max(0.0);
        self
    }
    /// Sets film index of refraction (at least 1.0).
    pub fn ior(&mut self, v: f32) -> &mut Self {
        self.params[2] = v.max(1.0);
        self
    }
}

impl EmissionGroup {
    /// Sets emitted luminance in nits (non-negative).
    pub fn luminance(&mut self, nits: f32) -> &mut Self {
        self.params[0] = nits.max(0.0);
        self
    }
    /// Sets emission color from explicit RGB components.
    pub fn color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color = [r, g, b, 1.0];
        self
    }
    /// Sets emission color from an RGB array.
    pub fn color_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }
}

impl GeometryGroup {
    /// Sets surface opacity (clamped to \[0, 1\]).
    pub fn opacity(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    /// Toggles thin-walled geometry mode.
    pub fn thin_walled(&mut self, v: bool) -> &mut Self {
        self.params[1] = if v { 1.0 } else { 0.0 };
        self
    }
}

impl OpenPBRMaterial {
    /// Number of `vec4` slots the flat material occupies on the GPU.
    pub const VEC4_COUNT: usize = 20;
    /// Size of the flat material blob in bytes.
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Default physically-plausible PBR material (gray dielectric).
    pub fn pbr() -> Self {
        Self {
            base: BaseGroup {
                params: [1.0, 0.0, 0.0, 0.0],
                color: [0.8, 0.8, 0.8, 1.0],
            },
            specular: SpecularGroup {
                params: [1.0, 0.3, 1.5, 0.0],
                color: [1.0, 1.0, 1.0, 0.0],
            },
            transmission: TransmissionGroup {
                params: [0.0, 0.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 0.0],
                scatter: [0.0, 0.0, 0.0, 0.0],
            },
            subsurface: SubsurfaceGroup {
                params: [0.0, 0.0, 1.0, 0.0],
                color: [0.8, 0.8, 0.8, 1.0],
                radius_scale_gb: [1.0, 1.0, 0.0, 0.0],
            },
            fuzz: FuzzGroup {
                params: [0.0, 0.5, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 0.0],
            },
            coat: CoatGroup {
                params: [0.0, 0.0, 0.0, 1.0],
                color: [1.0, 1.0, 1.0, 0.0],
                ior: [1.5, 0.0, 0.0, 0.0],
            },
            thin_film: ThinFilmGroup {
                params: [0.0, 0.0, 1.33, 0.0],
            },
            emission: EmissionGroup {
                params: [0.0, 0.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            geometry: GeometryGroup {
                params: [1.0, 0.0, 0.0, 0.0],
                params2: [0.0, 0.0, 0.0, 0.0],
            },
        }
    }

    /// Polished gray metal preset.
    pub fn metal() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(1.0);
        m.base.color_rgb([0.9, 0.9, 0.9]);
        m.specular.weight(1.0);
        m.specular.roughness(0.1);
        m
    }

    /// Rough dielectric preset.
    pub fn dielectric() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.8, 0.8, 0.8]);
        m.specular.weight(1.0);
        m.specular.roughness(0.3);
        m.specular.ior(1.5);
        m
    }

    /// Transparent glass preset with full transmission and thin walls.
    pub fn glass() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.0, 0.0, 0.0]);
        m.specular.weight(1.0);
        m.specular.roughness(0.0);
        m.specular.ior(1.5);
        m.transmission.weight(1.0);
        m.transmission.depth(1.0);
        m.transmission.color_rgb([1.0, 1.0, 1.0]);
        m.geometry.thin_walled(true);
        m
    }

    /// Reddish subsurface scattering preset.
    pub fn subsurface() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.8, 0.8, 0.8]);
        m.specular.weight(1.0);
        m.specular.roughness(0.3);
        m.specular.ior(1.3);
        m.subsurface.weight(1.0);
        m.subsurface.radius(0.1);
        m.subsurface.color_rgb([0.8, 0.2, 0.1]);
        m
    }

    /// Dielectric with a glossy clearcoat layer.
    pub fn coat() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.8, 0.8, 0.8]);
        m.specular.weight(1.0);
        m.specular.roughness(0.3);
        m.specular.ior(1.5);
        m.coat.weight(1.0);
        m.coat.roughness(0.1);
        m.coat.ior(1.5);
        m
    }

    /// Fabric-like preset with a fuzz (sheen) layer.
    pub fn fuzz() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.8, 0.8, 0.8]);
        m.specular.weight(1.0);
        m.specular.roughness(0.3);
        m.specular.ior(1.5);
        m.fuzz.weight(1.0);
        m.fuzz.roughness(0.3);
        m.fuzz.color_rgb([0.8, 0.8, 0.8]);
        m
    }

    /// Iridescent soap-bubble preset driven by thin-film interference.
    pub fn thin_film() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.0, 0.0, 0.0]);
        m.specular.weight(1.0);
        m.specular.roughness(0.0);
        m.specular.ior(1.5);
        m.thin_film.weight(1.0);
        m.thin_film.thickness_um(300.0);
        m.thin_film.ior(1.33);
        m
    }

    /// Warm emissive light-source preset (1000 nits).
    pub fn emission() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.0, 0.0, 0.0]);
        m.specular.weight(0.0);
        m.emission.luminance(1000.0);
        m.emission.color_rgb([1.0, 0.8, 0.6]);
        m
    }

    /// Thick refractive transmissive medium preset.
    pub fn transmission() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.0, 0.0, 0.0]);
        m.specular.weight(1.0);
        m.specular.roughness(0.0);
        m.specular.ior(1.5);
        m.transmission.weight(1.0);
        m.transmission.depth(2.0);
        m.transmission.color_rgb([1.0, 1.0, 1.0]);
        m
    }
}

impl Default for OpenPBRMaterial {
    fn default() -> Self {
        Self::pbr()
    }
}

/// Flat-slot count of [`OpenPBRMaterial`] (20 `vec4`s).
pub const OPENPBR_MATERIAL_VEC4_COUNT: usize = 20;
/// Byte size of [`OpenPBRMaterial`] (320 bytes).
pub const OPENPBR_MATERIAL_SIZE: usize = std::mem::size_of::<OpenPBRMaterial>();

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn test_openpbr_material_size() {
        assert_eq!(OPENPBR_MATERIAL_SIZE, 320);
        assert_eq!(OPENPBR_MATERIAL_VEC4_COUNT, 20);
    }

    #[test]
    fn test_openpbr_material_alignment() {
        let mat = OpenPBRMaterial::default();
        let bytes = bytemuck::bytes_of(&mat);
        assert_eq!(bytes.len(), OPENPBR_MATERIAL_SIZE);
        assert_eq!(bytes.len() % 16, 0);
    }

    #[test]
    fn test_default_material_is_neutral_dielectric() {
        let mat = OpenPBRMaterial::default();
        assert_eq!(mat.base.params[0], 1.0);
        assert_eq!(mat.base.color[0], 0.8);
        assert_eq!(mat.base.color[1], 0.8);
        assert_eq!(mat.base.color[2], 0.8);
        assert_eq!(mat.specular.params[1], 0.3);
        assert_eq!(mat.specular.params[2], 1.5);
    }

    #[test]
    fn test_metal_builder() {
        let mat = OpenPBRMaterial::metal();
        assert_eq!(mat.base.params[2], 1.0);
        assert_eq!(mat.specular.params[0], 1.0);
        assert_eq!(mat.specular.params[1], 0.1);
    }

    #[test]
    fn test_dielectric_builder() {
        let mat = OpenPBRMaterial::dielectric();
        assert_eq!(mat.base.params[2], 0.0);
        assert_eq!(mat.specular.params[0], 1.0);
        assert_eq!(mat.specular.params[1], 0.3);
        assert_eq!(mat.specular.params[2], 1.5);
    }

    #[test]
    fn test_glass_builder() {
        let mat = OpenPBRMaterial::glass();
        assert_eq!(mat.base.params[2], 0.0);
        assert_eq!(mat.transmission.params[0], 1.0);
        assert_eq!(mat.geometry.params[1], 1.0);
    }

    #[test]
    fn test_subsurface_builder() {
        let mat = OpenPBRMaterial::subsurface();
        assert_eq!(mat.subsurface.params[0], 1.0);
        assert_eq!(mat.subsurface.params[1], 0.1);
        assert_eq!(mat.subsurface.color[0], 0.8);
        assert_eq!(mat.subsurface.color[1], 0.2);
        assert_eq!(mat.subsurface.color[2], 0.1);
    }

    #[test]
    fn test_coat_builder() {
        let mat = OpenPBRMaterial::coat();
        assert_eq!(mat.coat.params[0], 1.0);
        assert_eq!(mat.coat.params[1], 0.1);
        assert_eq!(mat.coat.ior[0], 1.5);
    }

    #[test]
    fn test_fuzz_builder() {
        let mat = OpenPBRMaterial::fuzz();
        assert_eq!(mat.fuzz.params[0], 1.0);
        assert_eq!(mat.fuzz.params[1], 0.3);
    }

    #[test]
    fn test_thin_film_builder() {
        let mat = OpenPBRMaterial::thin_film();
        assert_eq!(mat.thin_film.params[0], 1.0);
        assert_eq!(mat.thin_film.params[1], 300.0);
        assert_eq!(mat.thin_film.params[2], 1.33);
    }

    #[test]
    fn test_emission_builder() {
        let mat = OpenPBRMaterial::emission();
        assert_eq!(mat.emission.params[0], 1000.0);
        assert_eq!(mat.emission.color[0], 1.0);
        assert_eq!(mat.emission.color[1], 0.8);
        assert_eq!(mat.emission.color[2], 0.6);
    }

    #[test]
    fn test_transmission_builder() {
        let mat = OpenPBRMaterial::transmission();
        assert_eq!(mat.transmission.params[0], 1.0);
        assert_eq!(mat.transmission.params[1], 2.0);
        assert_eq!(mat.transmission.color[0], 1.0);
    }

    #[test]
    fn test_parameter_clamping() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.base.weight(1.5);
        mat.base.metalness(-0.5);
        mat.specular.roughness(2.0);
        mat.specular.ior(0.5);
        assert_eq!(mat.base.params[0], 1.0);
        assert_eq!(mat.base.params[2], 0.0);
        assert_eq!(mat.specular.params[1], 1.0);
        assert_eq!(mat.specular.params[2], 1.0);
    }

    #[test]
    fn test_opacity_and_thin_walled() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.geometry.opacity(0.5);
        mat.geometry.thin_walled(true);
        assert_eq!(mat.geometry.params[0], 0.5);
        assert_eq!(mat.geometry.params[1], 1.0);
    }

    #[test]
    fn test_coat_darkening_parameter() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.coat.weight(1.0);
        mat.coat.roughness(0.1);
        mat.coat.anisotropy(0.3);
        mat.coat.darkening(0.5);
        assert_eq!(mat.coat.params[0], 1.0);
        assert_eq!(mat.coat.params[1], 0.1);
        assert_eq!(mat.coat.params[2], 0.3);
        assert_eq!(mat.coat.params[3], 0.5);
    }

    #[test]
    fn test_anisotropy_parameters() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.specular.anisotropy(0.5);
        mat.coat.anisotropy(0.8);
        assert_eq!(mat.specular.params[3], 0.5);
        assert_eq!(mat.coat.params[2], 0.8);
    }

    #[test]
    fn test_transmission_dispersion() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.transmission.weight(1.0);
        mat.transmission.dispersion_scale(0.5);
        mat.transmission.dispersion_abbe(50.0);
        assert_eq!(mat.transmission.params[0], 1.0);
        assert_eq!(mat.transmission.params[2], 0.5);
        assert_eq!(mat.transmission.params[3], 50.0);
    }

    #[test]
    fn test_subsurface_radius_scales() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.subsurface.weight(1.0);
        mat.subsurface.radius(0.5);
        mat.subsurface.radius_scale_r(1.0);
        mat.subsurface.radius_scale_g(0.5);
        mat.subsurface.radius_scale_b(0.25);
        assert_eq!(mat.subsurface.params[0], 1.0);
        assert_eq!(mat.subsurface.params[1], 0.5);
        assert_eq!(mat.subsurface.params[2], 1.0);
        assert_eq!(mat.subsurface.radius_scale_gb[0], 0.5);
        assert_eq!(mat.subsurface.radius_scale_gb[1], 0.25);
    }

    #[test]
    fn test_fuzz_parameters() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.fuzz.weight(0.8);
        mat.fuzz.roughness(0.4);
        mat.fuzz.color_rgb([0.9, 0.9, 0.9]);
        assert_eq!(mat.fuzz.params[0], 0.8);
        assert_eq!(mat.fuzz.params[1], 0.4);
        assert_eq!(mat.fuzz.color[0], 0.9);
    }

    #[test]
    fn test_pod_zeroable() {
        let _mat = OpenPBRMaterial::default();
        let _zeroed = OpenPBRMaterial::zeroed();
        let _pod = unsafe {
            std::mem::transmute::<[u8; OPENPBR_MATERIAL_SIZE], OpenPBRMaterial>(
                [0u8; OPENPBR_MATERIAL_SIZE],
            )
        };
    }

    #[test]
    fn scalar_builders_write_distinct_values() {
        // Each scalar builder must write its argument into the right slot.
        // Values are chosen to differ from the pbr() defaults so a mutant
        // that returns Default::default() instead of `self` fails.
        let mut mat = OpenPBRMaterial::pbr();
        mat.base.weight(0.5);
        mat.base.diffuse_roughness(0.4);
        mat.transmission.scatter_anisotropy(-0.5);
        mat.subsurface.scatter_anisotropy(-0.25);
        assert_eq!(mat.base.params[0], 0.5);
        assert_eq!(mat.base.params[1], 0.4);
        assert_eq!(mat.transmission.scatter[3], -0.5);
        assert_eq!(mat.subsurface.params[3], -0.25);
    }

    #[test]
    fn color_builders_write_distinct_values() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.base.color(0.1, 0.2, 0.3, 0.9);
        mat.specular.edge_tint(0.4, 0.5, 0.6);
        mat.specular.edge_tint_rgb([0.7, 0.8, 0.9]);
        mat.transmission.color(0.11, 0.12, 0.13);
        mat.transmission.scatter_color(0.21, 0.22, 0.23);
        mat.subsurface.color(0.31, 0.32, 0.33);
        mat.fuzz.color(0.41, 0.42, 0.43);
        mat.coat.color(0.51, 0.52, 0.53);
        mat.coat.color_rgb([0.61, 0.62, 0.63]);
        mat.emission.color(0.71, 0.72, 0.73);
        assert_eq!(mat.base.color, [0.1, 0.2, 0.3, 0.9]);
        assert_eq!(mat.specular.color, [0.7, 0.8, 0.9, 0.0]);
        assert_eq!(mat.transmission.color, [0.11, 0.12, 0.13, 0.0]);
        assert_eq!(&mat.transmission.scatter[..3], &[0.21, 0.22, 0.23]);
        assert_eq!(mat.subsurface.color, [0.31, 0.32, 0.33, 1.0]);
        assert_eq!(mat.fuzz.color, [0.41, 0.42, 0.43, 0.0]);
        assert_eq!(mat.coat.color, [0.61, 0.62, 0.63, 0.0]);
        assert_eq!(mat.emission.color, [0.71, 0.72, 0.73, 1.0]);
    }

    #[test]
    fn builders_do_not_disturb_other_fields() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.base.weight(0.5);
        mat.specular.edge_tint(0.1, 0.2, 0.3);
        // Fields untouched by the chain keep their pbr() defaults.
        assert_eq!(mat.base.params[1], 0.0);
        assert_eq!(mat.base.color, [0.8, 0.8, 0.8, 1.0]);
        assert_eq!(mat.transmission.color, [1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn dielectric_preset_writes_expected_fields() {
        // `dielectric()` is a convenience preset; a mutant that collapses
        // it to Default::default() must be caught by checking the fields
        // it is supposed to set.
        let mat = OpenPBRMaterial::dielectric();
        assert_eq!(mat.base.params[2], 0.0); // base metalness
        assert_eq!(&mat.base.color[..3], &[0.8, 0.8, 0.8]);
        assert_eq!(mat.specular.params[0], 1.0); // specular weight
        assert_eq!(mat.specular.params[1], 0.3); // specular roughness
    }
}
