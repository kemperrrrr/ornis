use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping};
use std::sync::OnceLock;
use vello::peniko::{Blob, FontData};
use vello::Glyph;

fn font_system() -> &'static std::sync::Mutex<FontSystem> {
    static FS: OnceLock<std::sync::Mutex<FontSystem>> = OnceLock::new();
    FS.get_or_init(|| std::sync::Mutex::new(FontSystem::new()))
}

pub fn load_inter_font() -> FontData {
    let candidates: Vec<std::path::PathBuf> = {
        let mut v = Vec::new();
        if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
            v.push(std::path::Path::new(&manifest).join("assets/editor/fonts/Inter-Regular.ttf"));
        }
        if let Ok(home) = std::env::var("HOME") {
            v.push(std::path::Path::new(&home).join(".cache/ornis/fonts/Inter-Regular.ttf"));
        }
        v.push(std::path::PathBuf::from(
            "crates/ui/assets/editor/fonts/Inter-Regular.ttf",
        ));
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

/// Layout text using cosmic-text. Returns vello-compatible Glyphs.
pub fn layout_text(font: &FontData, text: &str, font_size: f32) -> Vec<Glyph> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut fs = font_system().lock().unwrap();
    let data: &[u8] = font.data.as_ref();
    fs.db_mut().load_font_data(data.to_vec());
    let metrics = Metrics::new(font_size, font_size * 1.2);
    let mut buffer = Buffer::new(&mut fs, metrics);
    buffer.set_text(
        &mut fs,
        text,
        Attrs::new().family(cosmic_text::Family::Name("Inter")),
        Shaping::Advanced,
    );
    let mut glyphs = Vec::new();
    for line in &buffer.lines {
        let Some(layout) = line.layout_opt() else { continue };
        for run in layout {
            for g in &run.glyphs {
                glyphs.push(Glyph { id: g.glyph_id as u32, x: g.x, y: g.y });
            }
        }
    }
    glyphs
}

/// Measure text extent: (width, height)
pub fn measure_text(font: &FontData, text: &str, font_size: f32) -> (f32, f32) {
    if text.is_empty() {
        return (0.0, font_size * 1.2);
    }
    let mut fs = font_system().lock().unwrap();
    let data: &[u8] = font.data.as_ref();
    fs.db_mut().load_font_data(data.to_vec());
    let metrics = Metrics::new(font_size, font_size * 1.2);
    let mut buffer = Buffer::new(&mut fs, metrics);
    buffer.set_text(
        &mut fs,
        text,
        Attrs::new().family(cosmic_text::Family::Name("Inter")),
        Shaping::Advanced,
    );
    let w = buffer.lines.iter()
        .filter_map(|l| l.layout_opt())
        .flat_map(|runs| runs.iter())
        .map(|l| l.w)
        .fold(0.0f32, f32::max);
    (w, font_size * 1.2)
}
