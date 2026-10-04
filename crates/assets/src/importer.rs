//! Pluggable importers and the extension registry.
//!
//! An [`Importer`] owns one file format: it declares the extensions it
//! claims and the [`AssetKind`] it produces, and turns a file into an
//! [`ImportedAsset`]. The [`ImporterRegistry`] maps lowercased extensions
//! to importers; [`AssetServer::load`](crate::AssetServer::load) dispatches
//! through it. Built-in formats are gated by cargo features of this crate:
//! `gltf` (default: `.gltf`/`.glb`), `fbx` (placeholder: `.fbx` →
//! [`AssetError::UnsupportedFormat`]); RON scenes (`.ron`) are always on.

use std::collections::HashMap;
use std::path::Path;

use crate::error::AssetError;
use crate::scene::Scene;
use crate::server::{AssetKind, parse_scene_ron};

/// Format label of the RON scene importer.
pub const RON_FORMAT: &str = "ron";
/// Format label of the glTF importer.
pub const GLTF_FORMAT: &str = "gltf";
/// Format label of the FBX placeholder importer.
pub const FBX_FORMAT: &str = "fbx";

/// Result of one importer run, ready to be stored by the server.
#[non_exhaustive]
pub enum ImportedAsset {
    /// A scene ([`AssetKind::Scene`]): `.ron`.
    Scene(SceneImport),
    /// A glTF model ([`AssetKind::Model`]): `.gltf`/`.glb`.
    #[cfg(feature = "gltf")]
    Model(ornis_gltf::Model),
}

impl ImportedAsset {
    /// Kind of the imported payload.
    pub fn kind(&self) -> AssetKind {
        match self {
            Self::Scene(_) => AssetKind::Scene,
            #[cfg(feature = "gltf")]
            Self::Model(_) => AssetKind::Model,
        }
    }
}

/// Scene payload plus the format-specific sidecars the server retains.
pub struct SceneImport {
    /// Converted scene description.
    pub scene: Scene,
    /// Original text for text formats (RON round-trip/hot-reload plan).
    pub source_text: Option<String>,
}

impl SceneImport {
    /// Plain scene without sidecars.
    pub fn new(scene: Scene) -> Self {
        Self {
            scene,
            source_text: None,
        }
    }
}

/// One file format: declared extensions, produced kind, import routine.
///
/// Imports are path-based so multi-file formats (glTF with sibling
/// `.bin`/images) resolve their neighbours themselves.
pub trait Importer: Send + Sync + 'static {
    /// Stable format label (`"gltf"`, `"ron"`, ...) used to tag
    /// [`AssetError::Import`].
    fn format(&self) -> &'static str;

    /// Extensions this importer claims: lowercase, without the dot.
    fn extensions(&self) -> &'static [&'static str];

    /// Asset kind this importer produces.
    fn kind(&self) -> AssetKind;

    /// Imports the file at `path`.
    ///
    /// # Errors
    ///
    /// [`AssetError`] (IO, not found, format-tagged import failure,
    /// unsupported format); nothing is stored on error.
    fn import_path(&self, path: &Path) -> Result<ImportedAsset, AssetError>;
}

/// Extension → importer map.
///
/// Later registrations win for an extension they share with an earlier
/// importer (hosts can override a built-in format).
pub struct ImporterRegistry {
    importers: Vec<Box<dyn Importer>>,
    by_extension: HashMap<String, usize>,
}

impl Default for ImporterRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

impl ImporterRegistry {
    /// Registry with no importers.
    pub fn empty() -> Self {
        Self {
            importers: Vec::new(),
            by_extension: HashMap::new(),
        }
    }

    /// Registry with every built-in importer enabled by cargo features
    /// (`.ron` always; `.gltf`/`.glb` with `gltf`; `.fbx` with `fbx`).
    pub fn with_builtins() -> Self {
        let mut registry = Self::empty();
        registry.register(RonSceneImporter);
        #[cfg(feature = "gltf")]
        registry.register(GltfImporter);
        #[cfg(feature = "fbx")]
        registry.register(FbxImporter);
        registry
    }

    /// Adds an importer and claims its extensions (overriding earlier
    /// claims for the same extension).
    pub fn register(&mut self, importer: impl Importer) {
        let index = self.importers.len();
        for extension in importer.extensions() {
            self.by_extension
                .insert(extension.to_ascii_lowercase(), index);
        }
        self.importers.push(Box::new(importer));
    }

    /// Importer claiming `extension` (case-insensitive, without the dot).
    pub fn find(&self, extension: &str) -> Option<&dyn Importer> {
        let index = *self.by_extension.get(&extension.to_ascii_lowercase())?;
        self.importers.get(index).map(Box::as_ref)
    }

    /// Importer for `path` by its extension.
    ///
    /// # Errors
    ///
    /// [`AssetError::UnsupportedExtension`] when no importer claims the
    /// extension (or the path has none).
    pub fn for_path(&self, path: &Path) -> Result<&dyn Importer, AssetError> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        self.find(&extension)
            .ok_or_else(|| AssetError::UnsupportedExtension {
                path: path.to_path_buf(),
                extension,
            })
    }

    /// Every claimed extension, sorted.
    pub fn extensions(&self) -> Vec<&str> {
        let mut out: Vec<&str> = self.by_extension.keys().map(String::as_str).collect();
        out.sort_unstable();
        out
    }
}

/// Reads a whole file, mapping IO failures to [`AssetError`].
fn read_file(path: &Path) -> Result<Vec<u8>, AssetError> {
    std::fs::read(path).map_err(|error| AssetError::io(path, error))
}

/// Scene `.ron` importer (always enabled).
#[derive(Debug, Clone, Copy, Default)]
pub struct RonSceneImporter;

impl Importer for RonSceneImporter {
    fn format(&self) -> &'static str {
        RON_FORMAT
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["ron"]
    }

    fn kind(&self) -> AssetKind {
        AssetKind::Scene
    }

    fn import_path(&self, path: &Path) -> Result<ImportedAsset, AssetError> {
        let bytes = read_file(path)?;
        let text = String::from_utf8(bytes)
            .map_err(|error| AssetError::import(RON_FORMAT, Some(path), error))?;
        let scene =
            parse_scene_ron(&text).map_err(|error| AssetError::from(error).with_path(path))?;
        let mut import = SceneImport::new(scene);
        import.source_text = Some(text);
        Ok(ImportedAsset::Scene(import))
    }
}

/// glTF 2.0 importer (`.gltf` + sibling buffers, `.glb`); feature `gltf`.
#[cfg(feature = "gltf")]
#[derive(Debug, Clone, Copy, Default)]
pub struct GltfImporter;

#[cfg(feature = "gltf")]
impl GltfImporter {
    /// Imports in-memory glTF bytes (external URIs rejected, see
    /// [`ornis_gltf::load_slice`]).
    ///
    /// # Errors
    ///
    /// [`AssetError::Import`] tagged `"gltf"`.
    pub fn import_slice(&self, bytes: &[u8]) -> Result<ImportedAsset, AssetError> {
        let model = ornis_gltf::load_slice(bytes)?;
        Ok(ImportedAsset::Model(model))
    }
}

#[cfg(feature = "gltf")]
impl Importer for GltfImporter {
    fn format(&self) -> &'static str {
        GLTF_FORMAT
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["gltf", "glb"]
    }

    fn kind(&self) -> AssetKind {
        AssetKind::Model
    }

    fn import_path(&self, path: &Path) -> Result<ImportedAsset, AssetError> {
        let model = ornis_gltf::load_path(path).map_err(|error| match error {
            ornis_gltf::ImportError::Io(source) => AssetError::io(path, source),
            other => AssetError::from(other).with_path(path),
        })?;
        Ok(ImportedAsset::Model(model))
    }
}

/// FBX placeholder (feature `fbx`): claims `.fbx` so the failure is an
/// explicit [`AssetError::UnsupportedFormat`] instead of an unknown
/// extension. No parser yet.
#[cfg(feature = "fbx")]
#[derive(Debug, Clone, Copy, Default)]
pub struct FbxImporter;

#[cfg(feature = "fbx")]
impl Importer for FbxImporter {
    fn format(&self) -> &'static str {
        FBX_FORMAT
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["fbx"]
    }

    fn kind(&self) -> AssetKind {
        AssetKind::Scene
    }

    fn import_path(&self, path: &Path) -> Result<ImportedAsset, AssetError> {
        Err(AssetError::UnsupportedFormat {
            format: FBX_FORMAT,
            path: Some(path.to_path_buf()),
            reason: "the FBX importer is a placeholder (feature `fbx`); convert to glTF",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_claim_feature_gated_extensions() {
        let registry = ImporterRegistry::with_builtins();
        assert_eq!(registry.find("ron").map(Importer::format), Some(RON_FORMAT));
        assert_eq!(
            registry.find("GLB").map(Importer::format),
            cfg!(feature = "gltf").then_some(GLTF_FORMAT)
        );
        assert_eq!(
            registry.find("gltf").map(Importer::format),
            cfg!(feature = "gltf").then_some(GLTF_FORMAT)
        );
        assert_eq!(
            registry.find("fbx").map(Importer::format),
            cfg!(feature = "fbx").then_some(FBX_FORMAT)
        );
        assert!(registry.find("obj").is_none());
    }

    #[test]
    fn unknown_or_missing_extension_is_unsupported() {
        let registry = ImporterRegistry::with_builtins();
        for path in ["model.obj", "no_extension"] {
            assert!(matches!(
                registry.for_path(Path::new(path)),
                Err(AssetError::UnsupportedExtension { .. })
            ));
        }
    }

    struct Custom;

    impl Importer for Custom {
        fn format(&self) -> &'static str {
            "custom"
        }
        fn extensions(&self) -> &'static [&'static str] {
            &["ron", "CUST"]
        }
        fn kind(&self) -> AssetKind {
            AssetKind::Scene
        }
        fn import_path(&self, _path: &Path) -> Result<ImportedAsset, AssetError> {
            Err(AssetError::UnsupportedFormat {
                format: "custom",
                path: None,
                reason: "test",
            })
        }
    }

    #[test]
    fn later_registration_overrides_extension() {
        let mut registry = ImporterRegistry::with_builtins();
        registry.register(Custom);
        assert_eq!(registry.find("ron").map(Importer::format), Some("custom"));
        assert_eq!(registry.find("cust").map(Importer::format), Some("custom"));
        assert!(registry.extensions().contains(&"cust"));
    }

    #[cfg(feature = "fbx")]
    #[test]
    fn fbx_placeholder_reports_unsupported_format() {
        let error = FbxImporter
            .import_path(Path::new("model.fbx"))
            .err()
            .expect("placeholder never imports");
        assert!(matches!(
            error,
            AssetError::UnsupportedFormat { format: "fbx", .. }
        ));
    }
}
