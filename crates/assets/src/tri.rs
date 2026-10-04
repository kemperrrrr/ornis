//! Fallback `TriIndex`/`Triangle` when the `gltf` feature is off.
//!
//! With `gltf` enabled these names re-export `ornis_gltf`'s definitions
//! (one type across loader and scene); without it the scene contract still
//! needs the typed triangle view, so this module carries an identical copy
//! (same layout and API). Keep in sync with `crates/gltf/src/lib.rs`.

/// Vertex index into a mesh vertex list.
///
/// Newtype over raw `u32` soup so vertex indices never mix with document
/// node ids at the type level. Layout is `repr(transparent)` over `u32`
/// (12 bytes per [`Triangle`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct TriIndex(pub u32);

impl TriIndex {
    /// Wraps a raw vertex index without validation.
    pub const fn from_raw(index: u32) -> Self {
        Self(index)
    }

    /// Raw `u32` vertex index (for upload transports).
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Vertex position in a slice (`as usize`).
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for TriIndex {
    fn from(index: u32) -> Self {
        Self::from_raw(index)
    }
}

/// One triangle as three vertex indices (CCW from outside).
///
/// Stored as three [`TriIndex`] (12 bytes, `repr(C)`); use
/// [`Triangle::from_raw`]/[`Triangle::as_u32`] at transport boundaries and
/// [`Triangle::index`] for corner access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Triangle(pub TriIndex, pub TriIndex, pub TriIndex);

impl Triangle {
    /// Wraps three raw vertex indices without validation.
    pub const fn from_raw(indices: [u32; 3]) -> Self {
        Self(
            TriIndex(indices[0]),
            TriIndex(indices[1]),
            TriIndex(indices[2]),
        )
    }

    /// Raw `[u32; 3]` triple (for upload transports).
    pub const fn as_u32(self) -> [u32; 3] {
        [self.0.0, self.1.0, self.2.0]
    }

    /// `i`-th corner (`0..3`) as a vertex index. Out-of-range indices
    /// clamp to the last corner (same bit pattern as a saturated read).
    pub const fn index(self, i: usize) -> TriIndex {
        match i {
            0 => self.0,
            1 => self.1,
            _ => self.2,
        }
    }
}

impl From<[u32; 3]> for Triangle {
    fn from(indices: [u32; 3]) -> Self {
        Self::from_raw(indices)
    }
}
