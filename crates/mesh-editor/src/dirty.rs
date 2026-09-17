//! Dirty-region tracking for the preview/exact split.
//!
//! [`MeshDirty`] records which attribute classes changed since the last GPU
//! upload and which faces are affected, so the renderer uploads only the
//! dirty region instead of the whole mesh every frame.

/// Dirty-region flags for one pending preview edit.
#[derive(Debug, Clone, Default)]
pub struct MeshDirty {
    /// Bitmask of [`MeshDirty`] flag constants.
    bits: u8,
    /// Faces (indices into `indices` triples) touched by the edit.
    pub affected: Vec<u32>,
}

impl MeshDirty {
    /// Vertex positions changed.
    pub const VERTS: u8 = 0x01;
    /// Index buffer / topology changed.
    pub const TOPO: u8 = 0x02;
    /// Shading normals changed.
    pub const NORMALS: u8 = 0x04;
    /// Texture coordinates changed.
    pub const UV: u8 = 0x08;
    /// GPU buffer needs re-upload.
    pub const GPU_UPLOAD: u8 = 0x10;

    /// Clean flags with no affected faces.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark flag bits dirty and record affected faces.
    pub fn set(&mut self, flags: u8, faces: &[u32]) {
        self.bits |= flags;
        self.affected.extend_from_slice(faces);
    }

    /// True when the given flag bits are dirty.
    pub fn contains(&self, flags: u8) -> bool {
        self.bits & flags != 0
    }

    /// True when no flag bits are set.
    pub fn is_clean(&self) -> bool {
        self.bits == 0
    }

    /// Clear all flags and the affected-face list.
    pub fn clear(&mut self) {
        self.bits = 0;
        self.affected.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_clear_roundtrip() {
        let mut dirty = MeshDirty::new();
        assert!(dirty.is_clean());
        dirty.set(MeshDirty::VERTS | MeshDirty::GPU_UPLOAD, &[3, 7]);
        assert!(!dirty.is_clean());
        assert!(dirty.contains(MeshDirty::VERTS));
        assert!(!dirty.contains(MeshDirty::TOPO));
        assert_eq!(dirty.affected, vec![3, 7]);
        dirty.clear();
        assert!(dirty.is_clean());
        assert!(dirty.affected.is_empty());
    }
}
