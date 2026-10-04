//! Format-neutral asset loading error.
//!
//! [`AssetError`] is the one error surface of the [`AssetServer`](crate::AssetServer)
//! API: every importer (glTF, RON scenes, future FBX) maps its own typed
//! error into it, tagged by format, so callers never depend on a
//! particular loader crate. The value is cheap to clone (sources sit
//! behind [`Arc`]) because the same failure is both returned to the caller
//! and broadcast as [`AssetEvent::Failed`](crate::AssetEvent::Failed).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::server::{AssetKind, SceneLoadError};

/// Why an asset could not be loaded. The registry is untouched on error.
///
/// Equality compares the variant and the rendered message (sources are
/// opaque `dyn Error`s), which is what event assertions need.
/// [`std::error::Error::source`] yields the format's own typed error
/// (downcastable, e.g. to `ornis_gltf::ImportError`), not the `Arc`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AssetError {
    /// The requested file does not exist.
    NotFound {
        /// Requested path.
        path: PathBuf,
    },
    /// Filesystem failure while reading the asset or one of its siblings.
    Io {
        /// Requested asset path (`None` for in-memory loads).
        path: Option<PathBuf>,
        /// Underlying IO error.
        source: Arc<std::io::Error>,
    },
    /// No enabled importer claims this extension (or the path has none).
    /// Formats behind a disabled cargo feature land here too.
    UnsupportedExtension {
        /// Requested path.
        path: PathBuf,
        /// Lowercased extension without the dot (empty when absent).
        extension: String,
    },
    /// The format is recognised but cannot be imported yet (e.g. the
    /// `fbx` placeholder importer).
    UnsupportedFormat {
        /// Stable format label (`"fbx"`).
        format: &'static str,
        /// Requested path, when the load came from a file.
        path: Option<PathBuf>,
        /// Human-readable reason.
        reason: &'static str,
    },
    /// The importer rejected the content (parse/validation failure).
    Import {
        /// Stable format label (`"gltf"`, `"ron"`, ...).
        format: &'static str,
        /// Requested path (`None` for in-memory loads).
        path: Option<PathBuf>,
        /// Format-specific typed error (e.g. `ornis_gltf::ImportError`,
        /// [`SceneLoadError`]); downcast via [`std::error::Error::source`].
        source: Arc<dyn std::error::Error + Send + Sync>,
    },
    /// The path is registered (or its importer produces) another asset
    /// kind than the one requested from [`AssetServer::load`](crate::AssetServer::load).
    WrongKind {
        /// Requested path.
        path: Option<PathBuf>,
        /// Kind requested by the caller.
        expected: AssetKind,
        /// Kind the path/importer actually yields.
        found: AssetKind,
    },
    /// The handle/id does not address a loaded asset (never loaded or
    /// already unloaded).
    UnknownHandle {
        /// Raw [`AssetId::index`](crate::AssetId::index).
        index: u64,
    },
}

impl std::fmt::Display for AssetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound { path } => write!(f, "asset not found: {}", path.display()),
            Self::Io { path, source } => {
                write!(
                    f,
                    "IO error loading {}: {source}",
                    display_opt(path.as_deref())
                )
            }
            Self::UnsupportedExtension { path, extension } => {
                write!(f, "no importer for '.{extension}' ({})", path.display())
            }
            Self::UnsupportedFormat { format, reason, .. } => {
                write!(f, "{format} import is not supported yet: {reason}")
            }
            Self::Import {
                format,
                path,
                source,
            } => write!(
                f,
                "{format} import failed for {}: {source}",
                display_opt(path.as_deref())
            ),
            Self::WrongKind {
                path,
                expected,
                found,
            } => write!(
                f,
                "{} is a {found} asset, requested {expected}",
                display_opt(path.as_deref())
            ),
            Self::UnknownHandle { index } => write!(f, "unknown asset #{index}"),
        }
    }
}

impl std::error::Error for AssetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source.as_ref()),
            Self::Import { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl PartialEq for AssetError {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
            && self.to_string() == other.to_string()
    }
}

impl Eq for AssetError {}

impl AssetError {
    /// Stable format label for format-tagged variants
    /// ([`AssetError::Import`], [`AssetError::UnsupportedFormat`]).
    pub fn format(&self) -> Option<&'static str> {
        match self {
            Self::Import { format, .. } | Self::UnsupportedFormat { format, .. } => Some(format),
            _ => None,
        }
    }

    /// Requested path, when the failing load came from a file.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::NotFound { path } | Self::UnsupportedExtension { path, .. } => Some(path),
            Self::Io { path, .. }
            | Self::UnsupportedFormat { path, .. }
            | Self::Import { path, .. }
            | Self::WrongKind { path, .. } => path.as_deref(),
            Self::UnknownHandle { .. } => None,
        }
    }

    /// Attaches `path` to variants that carry an optional path and have
    /// none yet (importers that only see bytes, `From` conversions).
    pub fn with_path(mut self, new_path: &Path) -> Self {
        match &mut self {
            Self::Io { path, .. }
            | Self::UnsupportedFormat { path, .. }
            | Self::Import { path, .. }
            | Self::WrongKind { path, .. } => {
                if path.is_none() {
                    *path = Some(new_path.to_path_buf());
                }
            }
            Self::NotFound { .. }
            | Self::UnsupportedExtension { .. }
            | Self::UnknownHandle { .. } => {}
        }
        self
    }

    /// IO error for a load of `path` (`NotFound` collapses to
    /// [`AssetError::NotFound`] only for the requested file itself — a
    /// missing sibling stays [`AssetError::Io`]).
    pub(crate) fn io(path: &Path, source: std::io::Error) -> Self {
        if source.kind() == std::io::ErrorKind::NotFound && !path.exists() {
            Self::NotFound {
                path: path.to_path_buf(),
            }
        } else {
            Self::Io {
                path: Some(path.to_path_buf()),
                source: Arc::new(source),
            }
        }
    }

    /// Format-tagged import failure.
    pub fn import(
        format: &'static str,
        path: Option<&Path>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Import {
            format,
            path: path.map(Path::to_path_buf),
            source: Arc::new(source),
        }
    }
}

/// `"<memory>"` for in-memory loads, the path otherwise.
fn display_opt(path: Option<&Path>) -> String {
    path.map_or_else(|| "<memory>".to_owned(), |p| p.display().to_string())
}

impl From<SceneLoadError> for AssetError {
    /// RON scene parse failure → `Import { format: "ron" }`.
    fn from(error: SceneLoadError) -> Self {
        Self::import(crate::importer::RON_FORMAT, None, error)
    }
}

#[cfg(feature = "gltf")]
impl From<ornis_gltf::ImportError> for AssetError {
    /// glTF failure → `Io` for filesystem errors, `Import { format: "gltf" }`
    /// otherwise (the typed `ImportError` stays reachable as the source).
    fn from(error: ornis_gltf::ImportError) -> Self {
        match error {
            ornis_gltf::ImportError::Io(source) => Self::Io {
                path: None,
                source: Arc::new(source),
            },
            other => Self::import(crate::importer::GLTF_FORMAT, None, other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_format_and_path() {
        let error = AssetError::from(SceneLoadError::new("bad")).with_path(Path::new("a.ron"));
        assert_eq!(error.format(), Some("ron"));
        assert_eq!(error.path(), Some(Path::new("a.ron")));
        assert_eq!(error.to_string(), "ron import failed for a.ron: bad");
        let source = std::error::Error::source(&error).expect("typed source");
        assert!(source.downcast_ref::<SceneLoadError>().is_some());
    }

    #[test]
    fn equality_is_variant_plus_message() {
        let a = AssetError::UnknownHandle { index: 1 };
        assert_eq!(a.clone(), a);
        assert_ne!(a, AssetError::UnknownHandle { index: 2 });
    }

    #[test]
    fn io_not_found_collapses_only_for_missing_requested_file() {
        let missing = Path::new("definitely/not/here.glb");
        let error = AssetError::io(missing, std::io::Error::from(std::io::ErrorKind::NotFound));
        assert!(matches!(error, AssetError::NotFound { .. }), "{error:?}");
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn gltf_errors_keep_their_typed_source() {
        let error = AssetError::from(ornis_gltf::ImportError::NoScene);
        assert_eq!(error.format(), Some("gltf"));
        let source = std::error::Error::source(&error).expect("typed source");
        assert!(matches!(
            source.downcast_ref::<ornis_gltf::ImportError>(),
            Some(ornis_gltf::ImportError::NoScene)
        ));
    }
}
