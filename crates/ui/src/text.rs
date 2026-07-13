use skrifa::{FontRef, MetadataProvider, prelude::Size, instance::LocationRef};
use vello::peniko::{Blob, FontData};
use vello::Glyph;

/// Loads the bundled **Inter** font (shipped in the crate's assets) so the
/// editor renders with its real typeface instead of a system fallback.
///
/// Resolution order:
/// 1. `assets/editor/fonts/Inter-Regular.ttf` next to the crate manifest
///    (works on every platform, no external dependency).
/// 2. A writable cache dir (e.g. when the binary is run from a different cwd).
/// 3. System fallbacks (Arial / DejaVu / Segoe) if the bundle is missing.
pub fn load_inter_font() -> FontData {
    let candidates: Vec<std::path::PathBuf> = {
        let mut v = Vec::new();
        if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
            v.push(std::path::Path::new(&manifest)
                .join("assets/editor/fonts/Inter-Regular.ttf"));
        }
        if let Ok(home) = std::env::var("HOME") {
            v.push(std::path::Path::new(&home)
                .join(".cache/ornis/fonts/Inter-Regular.ttf"));
        }
        v.push(std::path::PathBuf::from(
            "crates/ui/assets/editor/fonts/Inter-Regular.ttf",
        ));
        // system fallbacks
        v.extend([
            std::path::PathBuf::from("/System/Library/Fonts/Supplemental/Arial.ttf"),
            std::path::PathBuf::from("/Library/Fonts/Arial.ttf"),
            std::path::PathBuf::from("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            std::path::PathBuf::from("/usr/share/fonts/TTF/DejaVuSans.ttf"),
            std::path::PathBuf::from("C:/Windows/Fonts/arial.ttf"),
        ]);
        v
    };
    for path in &candidates {
        if let Ok(data) = std::fs::read(path) {
            return load_font_from_bytes(&data);
        }
    }
    eprintln!("warning: no font found (Inter or system fallback), text will be invisible");
    FontData::new(Blob::from(Vec::new()), 0)
}

pub fn load_font_from_bytes(bytes: &[u8]) -> FontData {
    FontData::new(Blob::from(bytes.to_vec()), 0)
}

pub fn layout_text(font: &FontData, text: &str, font_size: f32) -> Vec<Glyph> {
    let data: &[u8] = font.data.as_ref();
    let Ok(font_ref) = FontRef::from_index(data, font.index) else {
        return Vec::new();
    };
    let charmap = font_ref.charmap();
    let metrics = font_ref.glyph_metrics(Size::new(font_size), LocationRef::default());

    let mut x = 0.0f32;
    let mut glyphs = Vec::with_capacity(text.len());
    for ch in text.chars() {
        if let Some(gid) = charmap.map(ch) {
            if let Some(advance) = metrics.advance_width(gid) {
                glyphs.push(Glyph { id: gid.to_u32(), x, y: 0.0 });
                x += advance;
            }
        }
    }

    glyphs
}

/// Measures the ink extent of `text` at `font_size` (single line).
/// Returns `(width, height)` where height is the typical line height.
pub fn measure_text(font: &FontData, text: &str, font_size: f32) -> (f32, f32) {
    let data: &[u8] = font.data.as_ref();
    let Ok(font_ref) = FontRef::from_index(data, font.index) else {
        return (0.0, font_size * 1.2);
    };
    let charmap = font_ref.charmap();
    let metrics = font_ref.glyph_metrics(Size::new(font_size), LocationRef::default());
    let mut width = 0.0f32;
    for ch in text.chars() {
        if let Some(gid) = charmap.map(ch) {
            if let Some(advance) = metrics.advance_width(gid) {
                width += advance;
            }
        }
    }
    (width, font_size * 1.2)
}
