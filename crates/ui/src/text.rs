use skrifa::{FontRef, MetadataProvider, prelude::Size, instance::LocationRef};
use vello::peniko::{Blob, FontData};
use vello::Glyph;

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
