//! Declarative scene descriptions (de)serialized as RON.
//!
//! The `*Desc` types are the serde-canonical contract shared by the demo
//! asset (`assets/scene.ron`), the editor protocol and the WASM viewport:
//! component payloads travel over the wire in exactly this shape.
//!
//! Placement, light and camera fields are typed (`Vec3`, [`UnitVec3`],
//! [`UnitQuat`], [`Meters`], [`Degrees`]) but serialize through
//! `serde(with = ...)` adapters (`crate::wire`) as the legacy `[f32; 3]` /
//! `[x, y, z, w]` / `f32` shapes, so RON files and the editor/WASM JSON
//! protocol are byte-compatible. On read, non-unit directions and
//! quaternions are normalized; zero-length or non-finite ones are a parse
//! error (see `crate::wire`).

use glam::{Quat, Vec3};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use ornis_core::units::{
    Clamped01, Degrees, Ior, LinearRgb, Meters, PositiveF32, UnitQuat, UnitVec3,
};

#[cfg(not(feature = "gltf"))]
pub use crate::tri::{TriIndex, Triangle};
#[cfg(feature = "gltf")]
pub use ornis_gltf::{TriIndex, Triangle};

/// Indices per triangle (flat soup alignment).
const TRIANGLE_VERTS: usize = 3;
/// Minimum sphere sector / cylinder radial segments.
const MIN_RADIAL_SEGMENTS: u32 = 3;
/// Minimum sphere stack rings.
const MIN_SPHERE_RINGS: u32 = 2;
/// Squared length below which a direction/up is treated as degenerate.
const DEGENERATE_LEN2: f32 = 1e-12;
/// Open upper bound for camera FOV (degrees).
const FOV_OPEN_MAX_DEG: f32 = 180.0;
/// Absolute view·up cosine above which the camera basis is parallel.
const CAMERA_UP_PARALLEL_DOT: f32 = 0.999;

/// Full scene description in RON format (see `assets/scene.ron`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    /// Human-readable scene label.
    pub name: String,
    /// Renderable entities.
    pub entities: Vec<EntityDesc>,
    /// Scene lights.
    pub lights: Vec<LightDesc>,
    /// The single viewing camera.
    pub camera: CameraDesc,
    /// Ambient light RGB multiplier.
    pub ambient: [f32; 3],
}

/// One renderable object: identity plus its three components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDesc {
    /// Display name; the editor uses it as the default `Name` component.
    pub name: String,
    /// Placement in world space.
    pub transform: TransformDesc,
    /// Geometry.
    pub mesh: MeshDesc,
    /// OpenPBR surface description.
    pub material: MaterialDesc,
}

/// Placement of an entity in world space.
///
/// Wire form (RON/JSON) is unchanged: `translation`/`scale` as `[f32; 3]`,
/// `rotation` as `[x, y, z, w]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformDesc {
    /// Translation in world units.
    #[serde(with = "crate::wire::vec3")]
    pub translation: Vec3,
    /// Orientation (unit quaternion; serialized in `(x, y, z, w)` order,
    /// normalized on load).
    #[serde(with = "crate::wire::unit_quat")]
    pub rotation: UnitQuat,
    /// Non-uniform scale per axis.
    #[serde(with = "crate::wire::vec3")]
    pub scale: Vec3,
}

impl Default for TransformDesc {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl TransformDesc {
    /// Origin, no rotation, unit scale.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: UnitQuat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Placement at `translation` with no rotation and unit scale.
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            rotation: UnitQuat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    /// Builds from raw wire arrays (`rotation` in `(x, y, z, w)` order).
    /// The quaternion is normalized; a zero/non-finite one falls back to
    /// identity (same policy as glTF imports of degenerate nodes).
    pub fn from_arrays(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Self {
        Self {
            translation: Vec3::from_array(translation),
            rotation: crate::wire::stable_unit_quat(Quat::from_array(rotation))
                .unwrap_or(UnitQuat::IDENTITY),
            scale: Vec3::from_array(scale),
        }
    }

    /// Rotation as a raw `[x, y, z, w]` array (wire/GPU order).
    pub fn rotation_array(&self) -> [f32; 4] {
        self.rotation.get().to_array()
    }
}

/// Geometry description (procedurally generated at load time).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshDesc {
    /// UV sphere centered at the transform origin.
    Sphere {
        /// Radius in world units (positive; checked at construction).
        radius: PositiveF32,
        /// Longitude divisions (minimum 3 at generation time).
        segments: u32,
        /// Latitude divisions (minimum 2 at generation time).
        rings: u32,
    },
    /// Axis-aligned box centered at the transform origin.
    Box {
        /// Full extents per axis in world units (all positive).
        size: [PositiveF32; 3],
    },
    /// Flat quad in the local XZ plane (`+Y` face normal), centered at
    /// the transform origin.
    Plane {
        /// Full extents in world units: `[width_x, depth_z]` (both positive).
        size: [PositiveF32; 2],
    },
    /// Right circular cylinder around local `+Y`, centered at the
    /// transform origin.
    Cylinder {
        /// Radius in world units (positive; checked at construction).
        radius: PositiveF32,
        /// Height along `+Y` in world units (positive; checked at construction).
        height: PositiveF32,
        /// Radial divisions (minimum 3 at generation time).
        radial_segments: u32,
    },
    /// Inline vertex soup: positions plus a triangle index list.
    /// Shading normals are NOT stored — recompute them at load (see
    /// `MeshData::from_positions` + `with_computed_normals` in the mesh
    /// editor); shape checks also live there, not in the transport.
    Custom {
        /// Vertex positions in engine units.
        positions: Vec<[f32; 3]>,
        /// Triangle index list (`u32`, triples, CCW from outside).
        indices: Vec<u32>,
    },
}

impl MeshDesc {
    /// Borrows the inline soup of a [`MeshDesc::Custom`].
    ///
    /// Returns `None` for procedural variants (`Sphere`, `Box`, `Plane`,
    /// `Cylinder`). The render extraction routes `Some` into the per-entity
    /// upload path (`mesh_upload::custom_vertices`); the transport itself
    /// never validates shapes — see `MeshData::validate` in the mesh editor.
    /// The flat list stays triple-aligned by construction (see
    /// [`MeshDesc::try_custom`], built through [`Triangle::from_raw`]).
    pub fn as_custom(&self) -> Option<(&[[f32; 3]], &[u32])> {
        match self {
            Self::Custom { positions, indices } => Some((positions, indices)),
            Self::Sphere { .. } | Self::Box { .. } | Self::Plane { .. } | Self::Cylinder { .. } => {
                None
            }
        }
    }

    /// Checked inline soup: `None` unless `indices.len() % 3 == 0` and every
    /// index is in range for `positions` (checked through
    /// [`Triangle::from_raw`] / [`TriIndex::index`).
    pub fn try_custom(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Option<Self> {
        if !indices.len().is_multiple_of(TRIANGLE_VERTS) {
            return None;
        }
        let triangles: Vec<Triangle> = indices
            .chunks_exact(TRIANGLE_VERTS)
            .map(|c| Triangle::from_raw([c[0], c[1], c[2]]))
            .collect();
        if triangles
            .iter()
            .flat_map(|t| t.as_u32())
            .any(|index| TriIndex::from_raw(index).index() >= positions.len())
        {
            return None;
        }
        Some(Self::Custom { positions, indices })
    }

    /// Typed triangle view over a [`MeshDesc::Custom`] soup.
    ///
    /// Returns `None` for procedural variants or malformed soups
    /// (non-triple length or out-of-range indices) — the physics projection
    /// (`body_for`) reports those as typed errors instead.
    pub fn as_triangles(&self) -> Option<Vec<Triangle>> {
        let (positions, indices) = self.as_custom()?;
        if !indices.len().is_multiple_of(TRIANGLE_VERTS) {
            return None;
        }
        let triangles: Vec<Triangle> = indices
            .chunks_exact(TRIANGLE_VERTS)
            .map(|c| Triangle::from_raw([c[0], c[1], c[2]]))
            .collect();
        if triangles
            .iter()
            .flat_map(|t| t.as_u32())
            .any(|index| TriIndex::from_raw(index).index() >= positions.len())
        {
            return None;
        }
        Some(triangles)
    }

    /// Checked sphere: `None` unless `radius` is finite and `> 0`,
    /// `segments >= 3` and `rings >= 2` (the documented generation minima).
    /// This is the canonical sphere constructor: the `Sphere` fields are
    /// typed, so direct literals need [`PositiveF32`] values.
    pub fn try_sphere_units(radius: Meters, segments: u32, rings: u32) -> Option<Self> {
        let radius = PositiveF32::try_new(radius.get())?;
        if segments >= MIN_RADIAL_SEGMENTS && rings >= MIN_SPHERE_RINGS {
            Some(Self::Sphere {
                radius,
                segments,
                rings,
            })
        } else {
            None
        }
    }

    /// Checked box: `None` unless every full extent is finite and `> 0`.
    /// Canonical constructor for the typed `Box` fields.
    pub fn try_box_units(size: [Meters; 3]) -> Option<Self> {
        let size = [
            PositiveF32::try_new(size[0].get())?,
            PositiveF32::try_new(size[1].get())?,
            PositiveF32::try_new(size[2].get())?,
        ];
        Some(Self::Box { size })
    }

    /// Checked plane: `None` unless both extents are finite and `> 0`.
    /// Canonical constructor for the typed `Plane` fields.
    pub fn try_plane_units(size: [Meters; 2]) -> Option<Self> {
        let size = [
            PositiveF32::try_new(size[0].get())?,
            PositiveF32::try_new(size[1].get())?,
        ];
        Some(Self::Plane { size })
    }

    /// Checked cylinder: `None` unless radius/height are finite and `> 0`
    /// and `radial_segments >= 3`. Canonical constructor for the typed
    /// `Cylinder` fields.
    pub fn try_cylinder_units(
        radius: Meters,
        height: Meters,
        radial_segments: u32,
    ) -> Option<Self> {
        let radius = PositiveF32::try_new(radius.get())?;
        let height = PositiveF32::try_new(height.get())?;
        if radial_segments >= MIN_RADIAL_SEGMENTS {
            Some(Self::Cylinder {
                radius,
                height,
                radial_segments,
            })
        } else {
            None
        }
    }

    /// Sphere radius as [`PositiveF32`], or `None` for other variants.
    pub fn sphere_radius_units(&self) -> Option<PositiveF32> {
        match self {
            Self::Sphere { radius, .. } => Some(*radius),
            _ => None,
        }
    }

    /// Full box extents in meters, or `None` for other variants.
    pub fn box_size_units(&self) -> Option<[Meters; 3]> {
        match self {
            Self::Box { size } => Some(size.map(|v| Meters::new(v.get()))),
            _ => None,
        }
    }
}

/// Material preset mapped onto the engine's OpenPBR surface model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MaterialDesc {
    /// Non-metal with specular reflection.
    Dielectric {
        /// Albedo color in linear space.
        base_color: [f32; 3],
        /// Microfacet roughness in [0, 1].
        roughness: Clamped01,
        /// Emissive RGB in linear space (`[0, 0, 0]` = no emission).
        /// Absent in older files — defaults to off.
        #[serde(default)]
        emission: [f32; 3],
    },
    /// Conductor with tinted specular reflection.
    Metal {
        /// Reflectance color in linear space.
        base_color: [f32; 3],
        /// Microfacet roughness in [0, 1].
        roughness: Clamped01,
        /// Emissive RGB in linear space (`[0, 0, 0]` = no emission).
        /// Absent in older files — defaults to off.
        #[serde(default)]
        emission: [f32; 3],
    },
    /// Base layer with a clearcoat on top.
    Coat {
        /// Albedo color of the base layer.
        base_color: [f32; 3],
        /// Clearcoat strength in [0, 1].
        coat_weight: Clamped01,
        /// Clearcoat roughness in [0, 1].
        coat_roughness: Clamped01,
        /// Emissive RGB in linear space (`[0, 0, 0]` = no emission).
        /// Absent in older files — defaults to off.
        #[serde(default)]
        emission: [f32; 3],
    },
    /// Rough diffuse-only surface (no specular lobe).
    Matte {
        /// Albedo color in linear space.
        base_color: [f32; 3],
        /// Diffuse roughness in [0, 1].
        roughness: Clamped01,
    },
    /// Transparent refractive surface (thin-walled glass).
    Glass {
        /// Transmitted tint in linear space.
        base_color: [f32; 3],
        /// Microfacet roughness in [0, 1].
        roughness: Clamped01,
        /// Index of refraction (>= 1.0). Absent in older files —
        /// defaults to 1.5.
        #[serde(default = "default_glass_ior")]
        ior: Ior,
    },
}

/// Default [`MaterialDesc::Glass`] index of refraction (crown glass).
fn default_glass_ior() -> Ior {
    Ior::default()
}

impl MaterialDesc {
    /// Typed dielectric: albedo as [`LinearRgb`], roughness as [`Clamped01`].
    pub fn dielectric_units(base_color: LinearRgb, roughness: Clamped01) -> Self {
        Self::Dielectric {
            base_color: base_color.as_array(),
            roughness,
            emission: [0.0, 0.0, 0.0],
        }
    }

    /// Typed metal: reflectance as [`LinearRgb`], roughness as [`Clamped01`].
    pub fn metal_units(base_color: LinearRgb, roughness: Clamped01) -> Self {
        Self::Metal {
            base_color: base_color.as_array(),
            roughness,
            emission: [0.0, 0.0, 0.0],
        }
    }

    /// Typed matte: albedo as [`LinearRgb`], roughness as [`Clamped01`].
    pub fn matte_units(base_color: LinearRgb, roughness: Clamped01) -> Self {
        Self::Matte {
            base_color: base_color.as_array(),
            roughness,
        }
    }

    /// Typed coat: albedo as [`LinearRgb`], weights as [`Clamped01`].
    pub fn coat_units(
        base_color: LinearRgb,
        coat_weight: Clamped01,
        coat_roughness: Clamped01,
    ) -> Self {
        Self::Coat {
            base_color: base_color.as_array(),
            coat_weight,
            coat_roughness,
            emission: [0.0, 0.0, 0.0],
        }
    }

    /// Typed glass: tint as [`LinearRgb`], roughness as [`Clamped01`],
    /// index of refraction as [`Ior`] (values below `1.0` clamp up).
    pub fn glass_units(base_color: LinearRgb, roughness: Clamped01, ior: Ior) -> Self {
        Self::Glass {
            base_color: base_color.as_array(),
            roughness,
            ior,
        }
    }

    /// Albedo/base color as [`LinearRgb`] (all variants).
    pub fn base_color_units(&self) -> LinearRgb {
        LinearRgb::new(match self {
            Self::Dielectric { base_color, .. }
            | Self::Metal { base_color, .. }
            | Self::Coat { base_color, .. }
            | Self::Matte { base_color, .. }
            | Self::Glass { base_color, .. } => *base_color,
        })
    }

    /// Roughness as [`Clamped01`] (stored typed; legacy out-of-range
    /// values were clamped on load through the `serde` impl).
    pub fn roughness_units(&self) -> Clamped01 {
        match self {
            Self::Dielectric { roughness, .. }
            | Self::Metal { roughness, .. }
            | Self::Matte { roughness, .. }
            | Self::Glass { roughness, .. } => *roughness,
            Self::Coat { coat_roughness, .. } => *coat_roughness,
        }
    }

    /// Index of refraction as [`Ior`] (`None` for non-glass variants).
    pub fn ior_units(&self) -> Option<Ior> {
        match self {
            Self::Glass { ior, .. } => Some(*ior),
            _ => None,
        }
    }
}

/// Whether a light casts a shadow map (typed storage for the `shadow`
/// field on [`LightDesc`] variants).
///
/// The serialized form stays `bool` (older files default to off), so the
/// wire schema is unchanged: construction and storage take [`ShadowCast`],
/// `serde` reads/writes the legacy `bool` polarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ShadowCast {
    /// No shadow map (default; absent in older files).
    #[default]
    Disabled,
    /// Depth pre-pass + PCF sampling in the evaluators.
    Enabled,
}

impl Serialize for ShadowCast {
    /// Wire form is the legacy `shadow: bool` (transport schemas stay unchanged).
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bool(self.is_enabled())
    }
}

impl<'de> Deserialize<'de> for ShadowCast {
    /// Reads the legacy `shadow: bool` wire form.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::from(bool::deserialize(deserializer)?))
    }
}

impl ShadowCast {
    /// `true` for [`ShadowCast::Enabled`].
    pub fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

impl From<bool> for ShadowCast {
    /// Legacy `shadow: bool` polarity.
    fn from(shadow: bool) -> Self {
        if shadow {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }
}

impl From<ShadowCast> for bool {
    /// Legacy `shadow: bool` polarity.
    fn from(cast: ShadowCast) -> bool {
        cast.is_enabled()
    }
}

/// Light source description.
///
/// Convention (shared by the shader evaluators): `direction` fields point
/// *toward* the light, except spot axes, which point from the light into
/// the scene (spotlight aim).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LightDesc {
    /// Infinitely distant light shining from a fixed direction.
    Directional {
        /// Direction toward the light (from the scene); unit length.
        #[serde(with = "crate::wire::unit_vec3")]
        direction: UnitVec3,
        /// Radiometric strength multiplier.
        // TODO(core): switch to `ornis_core::Lux` once the Core track lands it.
        intensity: f32,
        /// Emission color in linear space.
        // TODO(core): switch to `ornis_core::Color` once the Core track lands it.
        color: [f32; 3],
        /// Cast a shadow map (depth pre-pass + PCF in the evaluators).
        /// Absent in older files — defaults to off. Wire form is the
        /// legacy `bool`.
        #[serde(default)]
        shadow: ShadowCast,
    },
    /// Local light with inverse-square falloff and a finite range.
    Point {
        /// World-space position.
        #[serde(with = "crate::wire::vec3")]
        position: Vec3,
        /// Radiometric strength multiplier.
        // TODO(core): switch to `ornis_core::Lux` once the Core track lands it.
        intensity: f32,
        /// Emission color in linear space.
        // TODO(core): switch to `ornis_core::Color` once the Core track lands it.
        color: [f32; 3],
        /// Cutoff distance (must be > 0).
        #[serde(with = "crate::wire::meters")]
        range: Meters,
        /// Cast a shadow cube (6 depth faces + analytic major-axis
        /// sample in the evaluators).
        /// Absent in older files — defaults to off. Wire form is the
        /// legacy `bool`.
        #[serde(default)]
        shadow: ShadowCast,
    },
    /// Local light inside a cone aimed into the scene.
    Spot {
        /// World-space position.
        #[serde(with = "crate::wire::vec3")]
        position: Vec3,
        /// Spotlight axis, from the light into the scene; unit length.
        #[serde(with = "crate::wire::unit_vec3")]
        direction: UnitVec3,
        /// Radiometric strength multiplier.
        // TODO(core): switch to `ornis_core::Lux` once the Core track lands it.
        intensity: f32,
        /// Emission color in linear space.
        // TODO(core): switch to `ornis_core::Color` once the Core track lands it.
        color: [f32; 3],
        /// Cutoff distance (must be > 0).
        #[serde(with = "crate::wire::meters")]
        range: Meters,
        /// Inner cone angle (full brightness inside).
        #[serde(with = "crate::wire::degrees")]
        inner_angle: Degrees,
        /// Outer cone angle (zero outside, soft edge between).
        #[serde(with = "crate::wire::degrees")]
        outer_angle: Degrees,
        /// Cast a shadow map (depth pre-pass + PCF in the evaluators).
        /// Absent in older files — defaults to off. Wire form is the
        /// legacy `bool`.
        #[serde(default)]
        shadow: ShadowCast,
    },
}

impl LightDesc {
    /// Normalizes a raw direction; `None` for zero/non-finite input.
    fn checked_dir(direction: [f32; 3]) -> Option<UnitVec3> {
        crate::wire::stable_unit_vec3(Vec3::from_array(direction))
    }

    /// Checked intensity: `Some` only for finite values `>= 0`.
    fn checked_intensity(intensity: f32) -> Option<f32> {
        if intensity.is_finite() && intensity >= 0.0 {
            Some(intensity)
        } else {
            None
        }
    }

    /// Typed directional light (`None` for degenerate direction or
    /// negative/non-finite intensity).
    pub fn directional_units(
        direction: [f32; 3],
        intensity: f32,
        color: LinearRgb,
        shadow: ShadowCast,
    ) -> Option<Self> {
        Some(Self::Directional {
            direction: Self::checked_dir(direction)?,
            intensity: Self::checked_intensity(intensity)?,
            color: color.as_array(),
            shadow,
        })
    }

    /// Typed point light (`None` unless the position is finite, the
    /// intensity is `>= 0` and the range is finite and `> 0`).
    pub fn point_units(
        position: [Meters; 3],
        intensity: f32,
        color: LinearRgb,
        range: Meters,
        shadow: ShadowCast,
    ) -> Option<Self> {
        let position = Vec3::new(position[0].get(), position[1].get(), position[2].get());
        if !position.is_finite() {
            return None;
        }
        if !range.is_finite() || range.get() <= 0.0 {
            return None;
        }
        Some(Self::Point {
            position,
            intensity: Self::checked_intensity(intensity)?,
            color: color.as_array(),
            range,
            shadow,
        })
    }

    /// Typed spot light (`None` unless position/range/angles/intensity
    /// satisfy the documented invariants: finite position, `intensity >= 0`,
    /// `range > 0`, `0 <= inner <= outer`).
    #[allow(clippy::too_many_arguments)]
    pub fn spot_units(
        position: [Meters; 3],
        direction: [f32; 3],
        intensity: f32,
        color: LinearRgb,
        range: Meters,
        inner_angle: Degrees,
        outer_angle: Degrees,
        shadow: ShadowCast,
    ) -> Option<Self> {
        let position = Vec3::new(position[0].get(), position[1].get(), position[2].get());
        if !position.is_finite() {
            return None;
        }
        if !range.is_finite() || range.get() <= 0.0 {
            return None;
        }
        let (inner, outer) = (inner_angle.get(), outer_angle.get());
        if !inner.is_finite() || !outer.is_finite() || inner < 0.0 || inner > outer {
            return None;
        }
        Some(Self::Spot {
            position,
            direction: Self::checked_dir(direction)?,
            intensity: Self::checked_intensity(intensity)?,
            color: color.as_array(),
            range,
            inner_angle,
            outer_angle,
            shadow,
        })
    }

    /// Whether this light casts a shadow map (stored typed; the wire
    /// form is the legacy `shadow: bool`).
    pub fn shadow_cast(&self) -> ShadowCast {
        match self {
            Self::Directional { shadow, .. }
            | Self::Point { shadow, .. }
            | Self::Spot { shadow, .. } => *shadow,
        }
    }

    /// Sets shadow casting from a [`ShadowCast`].
    pub fn set_shadow_cast(&mut self, cast: ShadowCast) {
        let slot = match self {
            Self::Directional { shadow, .. }
            | Self::Point { shadow, .. }
            | Self::Spot { shadow, .. } => shadow,
        };
        *slot = cast;
    }

    /// Emission color as [`LinearRgb`] (all variants).
    pub fn color_units(&self) -> LinearRgb {
        LinearRgb::new(match self {
            Self::Directional { color, .. }
            | Self::Point { color, .. }
            | Self::Spot { color, .. } => *color,
        })
    }

    /// Cutoff range in meters (`None` for directionals).
    pub fn range_units(&self) -> Option<Meters> {
        match self {
            Self::Directional { .. } => None,
            Self::Point { range, .. } | Self::Spot { range, .. } => Some(*range),
        }
    }
}

/// Viewing camera described look-at style.
///
/// Wire form unchanged: vectors as `[f32; 3]`, `fov`/`near`/`far` as `f32`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraDesc {
    /// Eye position in world units.
    #[serde(with = "crate::wire::vec3")]
    pub position: Vec3,
    /// Point the camera looks at.
    #[serde(with = "crate::wire::vec3")]
    pub target: Vec3,
    /// Up direction, unit length (should not be parallel to the view
    /// direction; normalized on load).
    #[serde(with = "crate::wire::unit_vec3")]
    pub up: UnitVec3,
    /// Vertical field of view.
    #[serde(with = "crate::wire::degrees")]
    pub fov: Degrees,
    /// Near clip distance.
    #[serde(with = "crate::wire::meters")]
    pub near: Meters,
    /// Far clip distance.
    #[serde(with = "crate::wire::meters")]
    pub far: Meters,
}

impl CameraDesc {
    /// Checked camera: `None` unless the fov is strictly inside
    /// `(0, 180)` degrees, `near` is finite and `> 0`, `far > near`, all
    /// positions are finite, and `up` is finite, non-zero and not parallel
    /// to the view direction.
    pub fn try_new_units(
        position: [Meters; 3],
        target: [Meters; 3],
        up: [f32; 3],
        fov: Degrees,
        near: Meters,
        far: Meters,
    ) -> Option<Self> {
        let position = [position[0].get(), position[1].get(), position[2].get()];
        let target = [target[0].get(), target[1].get(), target[2].get()];
        if !position.iter().chain(target.iter()).all(|v| v.is_finite()) {
            return None;
        }
        if !fov.is_finite() || fov.get() <= 0.0 || fov.get() >= FOV_OPEN_MAX_DEG {
            return None;
        }
        if !near.is_finite() || near.get() <= 0.0 || !far.is_finite() || far.get() <= near.get() {
            return None;
        }
        if !up.iter().all(|v| v.is_finite()) {
            return None;
        }
        let view = [
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
        ];
        let view_len2 = view[0] * view[0] + view[1] * view[1] + view[2] * view[2];
        let up_len2 = up[0] * up[0] + up[1] * up[1] + up[2] * up[2];
        if !view_len2.is_finite()
            || view_len2 < DEGENERATE_LEN2
            || !up_len2.is_finite()
            || up_len2 < DEGENERATE_LEN2
        {
            return None;
        }
        let dot = (view[0] * up[0] + view[1] * up[1] + view[2] * up[2]).abs()
            / (view_len2.sqrt() * up_len2.sqrt());
        if !dot.is_finite() || dot > CAMERA_UP_PARALLEL_DOT {
            return None;
        }
        Some(Self {
            position: Vec3::from_array(position),
            target: Vec3::from_array(target),
            up: crate::wire::stable_unit_vec3(Vec3::from_array(up))?,
            fov,
            near,
            far,
        })
    }

    /// Vertical field of view in degrees.
    pub fn fov_units(&self) -> Degrees {
        self.fov
    }

    /// Near clip distance in meters.
    pub fn near_units(&self) -> Meters {
        self.near
    }

    /// Far clip distance in meters.
    pub fn far_units(&self) -> Meters {
        self.far
    }
}

impl Scene {
    /// Load a scene from a RON file.
    pub fn from_ron(ron_str: &str) -> Result<Self, ron::error::SpannedError> {
        ron::de::from_str(ron_str)
    }

    /// Save scene to RON string.
    pub fn to_ron(&self) -> Result<String, ron::error::Error> {
        ron::ser::to_string_pretty(self, Default::default())
    }

    /// Ambient multiplier as [`LinearRgb`].
    pub fn ambient_units(&self) -> LinearRgb {
        LinearRgb::new(self.ambient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One entity per material variant, one directional light.
    const FULL_SCENE_RON: &str = r#"
Scene(
    name: "test",
    entities: [
        (
            name: "dielectric",
            transform: (
                translation: (1.0, 2.0, 3.0),
                rotation: (0.0, 0.0, 0.0, 1.0),
                scale: (1.0, 1.0, 1.0),
            ),
            mesh: Sphere(radius: 2.0, segments: 16, rings: 8),
            material: Dielectric(base_color: (0.5, 0.5, 0.5), roughness: 0.9),
        ),
        (
            name: "metal",
            transform: (
                translation: (0.0, 0.0, 0.0),
                rotation: (0.0, 0.7071, 0.0, 0.7071),
                scale: (2.0, 2.0, 2.0),
            ),
            mesh: Sphere(radius: 1.0, segments: 32, rings: 24),
            material: Metal(base_color: (0.9, 0.7, 0.1), roughness: 0.2),
        ),
        (
            name: "coat",
            transform: (
                translation: (-1.0, 0.0, 0.0),
                rotation: (0.0, 0.0, 0.0, 1.0),
                scale: (1.0, 1.0, 1.0),
            ),
            mesh: Sphere(radius: 0.5, segments: 8, rings: 4),
            material: Coat(base_color: (1.0, 1.0, 1.0), coat_weight: 1.0, coat_roughness: 0.1),
        ),
    ],
    lights: [
        Directional(direction: (1.0, 1.0, 1.0), intensity: 0.6, color: (1.0, 1.0, 1.0)),
    ],
    camera: (
        position: (0.0, 2.5, 9.0),
        target: (0.0, 0.0, 0.0),
        up: (0.0, 1.0, 0.0),
        fov: 60.0,
        near: 0.1,
        far: 100.0,
    ),
    ambient: (0.1, 0.1, 0.15),
)
"#;

    #[test]
    fn parses_all_material_variants() {
        let scene = Scene::from_ron(FULL_SCENE_RON).expect("valid scene");
        assert_eq!(scene.name, "test");
        assert_eq!(scene.entities.len(), 3);
        assert_eq!(scene.lights.len(), 1);

        match &scene.entities[0].material {
            MaterialDesc::Dielectric {
                base_color,
                roughness,
                emission,
            } => {
                assert_eq!(*base_color, [0.5, 0.5, 0.5]);
                assert_eq!(roughness.get(), 0.9);
                // Old RON without `emission` defaults to no emission.
                assert_eq!(*emission, [0.0, 0.0, 0.0]);
            }
            other => panic!("expected Dielectric, got {other:?}"),
        }
        assert!(matches!(
            scene.entities[1].material,
            MaterialDesc::Metal { .. }
        ));
        match &scene.entities[2].material {
            MaterialDesc::Coat {
                coat_weight,
                coat_roughness,
                ..
            } => {
                assert_eq!(coat_weight.get(), 1.0);
                assert_eq!(coat_roughness.get(), 0.1);
            }
            other => panic!("expected Coat, got {other:?}"),
        }

        match &scene.entities[0].mesh {
            MeshDesc::Sphere {
                radius,
                segments,
                rings,
            } => {
                assert_eq!(radius.get(), 2.0);
                assert_eq!(*segments, 16);
                assert_eq!(*rings, 8);
            }
            // New variants must extend this assertion, not break it: old
            // files stay Sphere-only.
            other => panic!("expected Sphere, got {other:?}"),
        }
        assert_eq!(
            scene.entities[0].transform.translation.to_array(),
            [1.0, 2.0, 3.0]
        );
        assert_eq!(
            scene.entities[0].transform.rotation_array(),
            [0.0, 0.0, 0.0, 1.0]
        );

        match &scene.lights[0] {
            LightDesc::Directional {
                direction,
                intensity,
                color,
                ..
            } => {
                // `(1, 1, 1)` on the wire is normalized on load.
                let n = 1.0 / 3.0_f32.sqrt();
                assert!(direction.get().abs_diff_eq(Vec3::splat(n), 1e-6));
                assert_eq!(*intensity, 0.6);
                assert_eq!(*color, [1.0, 1.0, 1.0]);
            }
            _ => panic!("expected the directional test light"),
        }
        assert_eq!(scene.camera.fov.get(), 60.0);
        assert_eq!(scene.camera.near.get(), 0.1);
        assert_eq!(scene.camera.far.get(), 100.0);
        assert_eq!(scene.ambient, [0.1, 0.1, 0.15]);
    }

    #[test]
    fn ron_round_trip_is_stable() {
        let scene = Scene::from_ron(FULL_SCENE_RON).expect("valid scene");
        let serialized = scene.to_ron().expect("serialize");
        let reparsed = Scene::from_ron(&serialized).expect("re-parse");
        let reserialized = reparsed.to_ron().expect("re-serialize");
        assert_eq!(serialized, reserialized);
    }

    #[test]
    fn rejects_malformed_ron() {
        assert!(Scene::from_ron("Scene(name: 42)").is_err());
        assert!(Scene::from_ron("not a scene at all").is_err());
        // Unknown material variant.
        assert!(
            Scene::from_ron(&FULL_SCENE_RON.replace("Dielectric(base_color", "Crystal(base_color"))
                .is_err()
        );
    }

    #[test]
    fn rejects_missing_fields() {
        // `ambient` is missing.
        let broken = FULL_SCENE_RON.replace("    ambient: (0.1, 0.1, 0.15),\n", "");
        assert!(Scene::from_ron(&broken).is_err());
    }

    #[test]
    fn emission_defaults_to_off_for_old_ron() {
        // `FULL_SCENE_RON` predates `emission`: all three variants must
        // deserialize with zero emission (additive schema change).
        let scene = Scene::from_ron(FULL_SCENE_RON).expect("valid scene");
        for entity in &scene.entities {
            let emission = match &entity.material {
                MaterialDesc::Dielectric { emission, .. }
                | MaterialDesc::Metal { emission, .. }
                | MaterialDesc::Coat { emission, .. } => *emission,
                MaterialDesc::Matte { .. } | MaterialDesc::Glass { .. } => [0.0, 0.0, 0.0],
            };
            assert_eq!(emission, [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn emission_and_matte_round_trip() {
        let scene = Scene::from_ron(FULL_SCENE_RON).expect("valid scene");
        let with_new = scene.to_ron().expect("serialize").replacen(
            "emission: (0.0, 0.0, 0.0)",
            "emission: (2.0, 1.0, 0.5)",
            1,
        );
        let parsed = Scene::from_ron(&with_new).expect("emission parses");
        assert!(matches!(
            parsed.entities[0].material,
            MaterialDesc::Dielectric {
                emission: [2.0, 1.0, 0.5],
                ..
            }
        ));

        let matte_ron = FULL_SCENE_RON.replace(
            "Dielectric(base_color: (0.5, 0.5, 0.5), roughness: 0.9)",
            "Matte(base_color: (0.2, 0.4, 0.6), roughness: 0.7)",
        );
        let matte = Scene::from_ron(&matte_ron).expect("matte parses");
        assert!(matches!(
            matte.entities[0].material,
            MaterialDesc::Matte {
                base_color: [0.2, 0.4, 0.6],
                roughness,
            } if (roughness.get() - 0.7).abs() < f32::EPSILON
        ));
        let reserialized = matte.to_ron().expect("serialize");
        let reparsed = Scene::from_ron(&reserialized).expect("re-parse");
        assert_eq!(reserialized, reparsed.to_ron().expect("re-serialize"));
    }

    #[test]
    fn glass_defaults_ior_and_round_trips() {
        // `ior` absent → 1.5 (additive schema change, same as `emission`).
        let without_ior = FULL_SCENE_RON.replace(
            "Dielectric(base_color: (0.5, 0.5, 0.5), roughness: 0.9)",
            "Glass(base_color: (0.9, 0.95, 1.0), roughness: 0.05)",
        );
        let scene = Scene::from_ron(&without_ior).expect("glass parses");
        assert!(matches!(
            scene.entities[0].material,
            MaterialDesc::Glass {
                base_color: [0.9, 0.95, 1.0],
                roughness,
                ior,
            } if (roughness.get() - 0.05).abs() < f32::EPSILON
                && (ior.get() - 1.5).abs() < f32::EPSILON
        ));

        let with_ior = FULL_SCENE_RON.replace(
            "Dielectric(base_color: (0.5, 0.5, 0.5), roughness: 0.9)",
            "Glass(base_color: (0.9, 0.95, 1.0), roughness: 0.05, ior: 1.33)",
        );
        let explicit = Scene::from_ron(&with_ior).expect("glass ior parses");
        assert!(matches!(
            explicit.entities[0].material,
            MaterialDesc::Glass { ior, .. } if (ior.get() - 1.33).abs() < f32::EPSILON
        ));
        let serialized = explicit.to_ron().expect("serialize");
        let reparsed = Scene::from_ron(&serialized).expect("re-parse");
        assert_eq!(serialized, reparsed.to_ron().expect("re-serialize"));
    }

    #[test]
    fn demo_asset_parses() {
        // Keeps the shipped demo scene in sync with the schema.
        let scene = Scene::from_ron(include_str!("../../../assets/scene.ron")).expect("demo scene");
        assert_eq!(scene.name, "demo");
        assert_eq!(scene.entities.len(), 5);
        assert_eq!(scene.lights.len(), 2);
        assert_eq!(scene.camera.fov.get(), 60.0);
    }

    /// Shipped scene files (the wire contract under test).
    const SHIPPED_RON: [(&str, &str); 2] = [
        ("scene.ron", include_str!("../../../assets/scene.ron")),
        (
            "demo_scene.ron",
            include_str!("../../../assets/demo_scene.ron"),
        ),
    ];

    /// Legacy, untyped mirror of the pre-typing descriptors: the exact
    /// serde shape the RON files and the editor/WASM JSON protocol used.
    mod legacy {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub struct Transform {
            pub translation: [f32; 3],
            pub rotation: [f32; 4],
            pub scale: [f32; 3],
        }

        /// Only the typed parts of a scene; other fields are ignored.
        #[derive(Debug, Clone, Deserialize)]
        pub struct Scene {
            pub lights: Vec<Light>,
            #[allow(dead_code)]
            pub camera: Camera,
        }

        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub struct Camera {
            pub position: [f32; 3],
            pub target: [f32; 3],
            pub up: [f32; 3],
            pub fov: f32,
            pub near: f32,
            pub far: f32,
        }

        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub enum Light {
            Directional {
                direction: [f32; 3],
                intensity: f32,
                color: [f32; 3],
                #[serde(default)]
                shadow: bool,
            },
            Point {
                position: [f32; 3],
                intensity: f32,
                color: [f32; 3],
                range: f32,
                #[serde(default)]
                shadow: bool,
            },
            Spot {
                position: [f32; 3],
                direction: [f32; 3],
                intensity: f32,
                color: [f32; 3],
                range: f32,
                inner_angle: f32,
                outer_angle: f32,
                #[serde(default)]
                shadow: bool,
            },
        }
    }

    fn unit(v: [f32; 3]) -> [f32; 3] {
        Vec3::from_array(v).normalize().to_array()
    }

    #[test]
    fn shipped_ron_round_trips_equivalently_and_stably() {
        for (name, text) in SHIPPED_RON {
            let scene = Scene::from_ron(text).unwrap_or_else(|e| panic!("{name}: {e}"));
            let first = scene.to_ron().expect("serialize");
            let reparsed = Scene::from_ron(&first).expect("re-parse");
            // Idempotent after the first (normalizing) load: byte-identical.
            assert_eq!(first, reparsed.to_ron().expect("re-serialize"), "{name}");
            // Equivalent to the source: same entities/transforms/camera;
            // directions equal up to normalization.
            assert_eq!(reparsed.entities.len(), scene.entities.len());
            for (a, b) in scene.entities.iter().zip(&reparsed.entities) {
                assert_eq!(a.transform, b.transform, "{name}: {}", a.name);
            }
            assert_eq!(scene.camera, reparsed.camera, "{name}");
            // The typed re-serialization reads back through the legacy
            // untyped mirror (shape unchanged) with matching values.
            let legacy: Vec<legacy::Light> = reparsed
                .lights
                .iter()
                .map(|light| {
                    ron::de::from_str(&ron::ser::to_string(light).expect("ser")).expect("legacy")
                })
                .collect();
            let source: Vec<legacy::Light> = scene_lights_legacy(text);
            assert_eq!(legacy.len(), source.len());
            for (typed, raw) in legacy.iter().zip(&source) {
                match (typed, raw) {
                    (
                        legacy::Light::Directional { direction: a, .. },
                        legacy::Light::Directional { direction: b, .. },
                    ) => {
                        let (a, b) = (Vec3::from_array(*a), Vec3::from_array(unit(*b)));
                        assert!(a.abs_diff_eq(b, 1e-6), "{name}: {a} vs {b}");
                    }
                    other => panic!("{name}: unexpected light pair {other:?}"),
                }
            }
        }
    }

    /// Lights of a shipped file parsed through the legacy mirror.
    fn scene_lights_legacy(text: &str) -> Vec<legacy::Light> {
        let scene: legacy::Scene = ron::de::from_str(text).expect("legacy mirror parses");
        scene.lights
    }

    #[test]
    fn typed_wire_shape_matches_legacy_in_ron_and_json() {
        let transform = TransformDesc {
            translation: Vec3::new(1.0, -2.5, 3.0),
            rotation: UnitQuat::normalize(Quat::from_xyzw(0.0, 0.6, 0.0, 0.8)).expect("unit"),
            scale: Vec3::new(2.0, 2.0, 0.5),
        };
        let legacy_transform = legacy::Transform {
            translation: [1.0, -2.5, 3.0],
            rotation: transform.rotation_array(),
            scale: [2.0, 2.0, 0.5],
        };
        assert_eq!(
            ron::ser::to_string(&transform).expect("ron"),
            ron::ser::to_string(&legacy_transform).expect("ron")
        );
        assert_eq!(
            serde_json::to_string(&transform).expect("json"),
            serde_json::to_string(&legacy_transform).expect("json")
        );
        // Rotation stays (x, y, z, w) on the wire.
        assert!(
            serde_json::to_string(&transform)
                .expect("json")
                .contains("\"rotation\":[0.0,0.6,0.0,0.8]")
        );

        let camera = CameraDesc {
            position: Vec3::new(0.0, 2.5, 9.0),
            target: Vec3::ZERO,
            up: UnitVec3::Y,
            fov: Degrees::new(60.0),
            near: Meters::new(0.1),
            far: Meters::new(100.0),
        };
        let legacy_camera = legacy::Camera {
            position: [0.0, 2.5, 9.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 60.0,
            near: 0.1,
            far: 100.0,
        };
        assert_eq!(
            serde_json::to_string(&camera).expect("json"),
            serde_json::to_string(&legacy_camera).expect("json")
        );
        assert_eq!(
            ron::ser::to_string(&camera).expect("ron"),
            ron::ser::to_string(&legacy_camera).expect("ron")
        );

        let spot = LightDesc::spot_units(
            [Meters::new(1.0), Meters::new(4.0), Meters::new(0.0)],
            [0.0, -1.0, 0.0],
            8.0,
            LinearRgb::WHITE,
            Meters::new(12.0),
            Degrees::new(15.0),
            Degrees::new(30.0),
            ShadowCast::Enabled,
        )
        .expect("valid spot");
        let legacy_spot = legacy::Light::Spot {
            position: [1.0, 4.0, 0.0],
            direction: [0.0, -1.0, 0.0],
            intensity: 8.0,
            color: [1.0, 1.0, 1.0],
            range: 12.0,
            inner_angle: 15.0,
            outer_angle: 30.0,
            shadow: true,
        };
        assert_eq!(
            serde_json::to_string(&spot).expect("json"),
            serde_json::to_string(&legacy_spot).expect("json")
        );
        let point = LightDesc::point_units(
            [Meters::new(0.0), Meters::new(3.0), Meters::new(0.0)],
            5.0,
            LinearRgb::WHITE,
            Meters::new(10.0),
            ShadowCast::Disabled,
        )
        .expect("valid point");
        let legacy_point = legacy::Light::Point {
            position: [0.0, 3.0, 0.0],
            intensity: 5.0,
            color: [1.0, 1.0, 1.0],
            range: 10.0,
            shadow: false,
        };
        assert_eq!(
            ron::ser::to_string(&point).expect("ron"),
            ron::ser::to_string(&legacy_point).expect("ron")
        );
        // Legacy JSON reads back into the typed descriptor.
        let parsed: LightDesc =
            serde_json::from_str(&serde_json::to_string(&legacy_spot).expect("json"))
                .expect("legacy JSON parses");
        assert_eq!(
            serde_json::to_string(&parsed).expect("json"),
            serde_json::to_string(&legacy_spot).expect("json")
        );
    }

    #[test]
    fn non_unit_rotation_and_direction_are_normalized_on_load() {
        let ron = FULL_SCENE_RON
            .replace(
                "rotation: (0.0, 0.0, 0.0, 1.0),\n                scale: (1.0, 1.0, 1.0),\n            ),\n            mesh: Sphere(radius: 2.0",
                "rotation: (0.0, 0.0, 0.0, 2.0),\n                scale: (1.0, 1.0, 1.0),\n            ),\n            mesh: Sphere(radius: 2.0",
            )
            .replace("up: (0.0, 1.0, 0.0)", "up: (0.0, 3.0, 0.0)");
        assert_ne!(ron, FULL_SCENE_RON, "fixture replacement applied");
        let scene = Scene::from_ron(&ron).expect("non-unit values normalize");
        assert_eq!(
            scene.entities[0].transform.rotation_array(),
            [0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(scene.camera.up, UnitVec3::Y);
        // The 0.7071 hand-typed quaternion is accepted and normalized.
        let q = scene.entities[1].transform.rotation.get();
        assert!((q.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn degenerate_rotation_and_direction_are_parse_errors() {
        let zero_quat = FULL_SCENE_RON.replacen(
            "rotation: (0.0, 0.0, 0.0, 1.0)",
            "rotation: (0.0, 0.0, 0.0, 0.0)",
            1,
        );
        let error = Scene::from_ron(&zero_quat).expect_err("zero quaternion rejected");
        assert!(error.to_string().contains("rotation"), "{error}");

        let zero_dir =
            FULL_SCENE_RON.replace("direction: (1.0, 1.0, 1.0)", "direction: (0.0, 0.0, 0.0)");
        let error = Scene::from_ron(&zero_dir).expect_err("zero direction rejected");
        assert!(error.to_string().contains("direction"), "{error}");

        let zero_up = FULL_SCENE_RON.replace("up: (0.0, 1.0, 0.0)", "up: (0.0, 0.0, 0.0)");
        assert!(Scene::from_ron(&zero_up).is_err(), "zero up rejected");

        // Same policy over JSON (editor/WASM protocol).
        let json = r#"{"translation":[0,0,0],"rotation":[0,0,0,0],"scale":[1,1,1]}"#;
        assert!(serde_json::from_str::<TransformDesc>(json).is_err());
        let json = r#"{"translation":[0,0,0],"rotation":[0,0,0,3],"scale":[1,1,1]}"#;
        let t: TransformDesc = serde_json::from_str(json).expect("normalized");
        assert_eq!(t.rotation, UnitQuat::IDENTITY);
    }

    #[test]
    fn transform_from_arrays_normalizes_or_falls_back_to_identity() {
        let t = TransformDesc::from_arrays([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0], [1.0; 3]);
        assert_eq!(t.rotation, UnitQuat::IDENTITY);
        assert_eq!(t.translation, Vec3::new(1.0, 2.0, 3.0));
        let t = TransformDesc::from_arrays([0.0; 3], [0.0, 0.0, 2.0, 0.0], [1.0; 3]);
        assert_eq!(t.rotation_array(), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(TransformDesc::default(), TransformDesc::IDENTITY);
    }

    #[test]
    fn procedural_variants_parse_and_round_trip() {
        let ron = FULL_SCENE_RON
            .replace(
                "mesh: Sphere(radius: 2.0, segments: 16, rings: 8),",
                "mesh: Box(size: (2.0, 4.0, 6.0)),",
            )
            .replace(
                "mesh: Sphere(radius: 1.0, segments: 32, rings: 24),",
                "mesh: Plane(size: (3.0, 5.0)),",
            )
            .replace(
                "mesh: Sphere(radius: 0.5, segments: 8, rings: 4),",
                "mesh: Cylinder(radius: 1.5, height: 7.0, radial_segments: 12),",
            );
        let scene = Scene::from_ron(&ron).expect("procedural variants parse");
        assert!(matches!(
            scene.entities[0].mesh,
            MeshDesc::Box { size } if size.map(PositiveF32::get) == [2.0, 4.0, 6.0]
        ));
        assert!(matches!(
            scene.entities[1].mesh,
            MeshDesc::Plane { size } if size.map(PositiveF32::get) == [3.0, 5.0]
        ));
        assert!(matches!(
            scene.entities[2].mesh,
            MeshDesc::Cylinder {
                radius,
                height,
                radial_segments,
            } if radius.get() == 1.5 && height.get() == 7.0 && radial_segments == 12
        ));
        let serialized = scene.to_ron().expect("serialize");
        let reparsed = Scene::from_ron(&serialized).expect("re-parse");
        let reserialized = reparsed.to_ron().expect("re-serialize");
        assert_eq!(serialized, reserialized);
    }

    #[test]
    fn as_custom_returns_none_for_procedurals() {
        let variants = [
            MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(1.0),
                segments: 16,
                rings: 8,
            },
            MeshDesc::Box {
                size: [
                    PositiveF32::expect_valid(1.0),
                    PositiveF32::expect_valid(2.0),
                    PositiveF32::expect_valid(3.0),
                ],
            },
            MeshDesc::Plane {
                size: [
                    PositiveF32::expect_valid(1.0),
                    PositiveF32::expect_valid(2.0),
                ],
            },
            MeshDesc::Cylinder {
                radius: PositiveF32::expect_valid(1.0),
                height: PositiveF32::expect_valid(2.0),
                radial_segments: 8,
            },
        ];
        for mesh in &variants {
            assert!(mesh.as_custom().is_none(), "procedural: {mesh:?}");
        }
        let custom = MeshDesc::Custom {
            positions: vec![[0.0, 0.0, 0.0]],
            indices: vec![0, 0, 0],
        };
        let (positions, indices) = custom.as_custom().expect("custom borrows soup");
        assert_eq!(positions.len(), 1);
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn try_custom_validates_triples_and_range() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let valid = MeshDesc::try_custom(positions.clone(), vec![0, 1, 2]).expect("triple builds");
        let tris = valid.as_triangles().expect("typed view");
        assert_eq!(tris, vec![Triangle::from_raw([0, 1, 2])]);
        assert_eq!(tris[0].as_u32(), [0, 1, 2]);
        assert_eq!(tris[0].index(1), TriIndex::from_raw(1));
        assert!(MeshDesc::try_custom(positions.clone(), vec![0, 1]).is_none());
        assert!(MeshDesc::try_custom(positions, vec![0, 1, 9]).is_none());
        assert!(
            MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(1.0),
                segments: 16,
                rings: 8,
            }
            .as_triangles()
            .is_none()
        );
    }

    #[test]
    fn checked_dir_normalizes_or_rejects() {
        assert_eq!(LightDesc::checked_dir([2.0, 0.0, 0.0]), Some(UnitVec3::X));
        assert!(LightDesc::checked_dir([0.0, 0.0, 0.0]).is_none());
        assert!(LightDesc::checked_dir([f32::NAN, 0.0, 0.0]).is_none());
        assert!(LightDesc::checked_dir([f32::INFINITY, 0.0, 0.0]).is_none());
    }

    #[test]
    fn checked_intensity_accepts_finite_non_negative() {
        assert_eq!(LightDesc::checked_intensity(2.5), Some(2.5));
        assert_eq!(LightDesc::checked_intensity(0.0), Some(0.0));
        assert!(LightDesc::checked_intensity(-1.0).is_none());
        assert!(LightDesc::checked_intensity(f32::NAN).is_none());
        assert!(LightDesc::checked_intensity(f32::INFINITY).is_none());
    }

    #[test]
    fn glass_ior_defaults_to_crown() {
        assert_eq!(default_glass_ior().get(), 1.5);
    }
}
