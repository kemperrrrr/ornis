//! Decoded image loading for `<img>` elements in the HTML/CSS editor.
//!
//! Browsers render `<img>` as a "replaced element": its layout size comes from
//! (a) explicit CSS `width`/`height`, (b) the image's intrinsic size, or (c) an
//! aspect ratio derived from the intrinsic size when only one dimension is set.
//! We replicate that here: [`load_image`] decodes an asset into a [`DecodedImage`]
//! (SVG path geometry, or a raster `peniko::ImageData`), and the layout stage
//! uses [`DecodedImage::intrinsic_size`] to give `<img>` nodes a real box.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use image::imageops::FilterType;
use vello::peniko::{Blob, ImageData};

/// Resizes a raster image using Lanczos3 filter for high-quality downscaling.
/// Returns None if resize fails.
pub fn resize_lanczos(data: &ImageData, new_w: u32, new_h: u32) -> Option<ImageData> {
    let img = image::RgbaImage::from_raw(
        data.width,
        data.height,
        data.data.as_ref().as_ref().to_vec(),
    )?;
    let resized = image::imageops::resize(&img, new_w, new_h, FilterType::Lanczos3);
    let raw = resized.into_raw();
    let blob = Blob::new(std::sync::Arc::new(raw) as std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>);
    Some(ImageData {
        data: blob,
        format: vello::peniko::ImageFormat::Rgba8,
        alpha_type: vello::peniko::ImageAlphaType::Alpha,
        width: new_w,
        height: new_h,
    })
}

/// A decoded `<img>` source, ready to paint.
#[derive(Debug, Clone)]
pub enum DecodedImage {
    /// Vector icon: the SVG path `d`, the viewBox (x, y, w, h) for scaling, the
    /// intrinsic pixel size (used for aspect ratio), and the path `fill` if the
    /// SVG declares one explicitly.
    Svg {
        path_d: String,
        view_box: (f32, f32, f32, f32),
        intrinsic_size: (u32, u32),
        fill: Option<String>,
    },
    /// Raster image bytes in `peniko::ImageData` (RGBA8, straight alpha). The
    /// width/height double as the intrinsic size.
    Raster(ImageData),
}

impl DecodedImage {
    /// Intrinsic (natural) pixel size of the image, used to derive an aspect
    /// ratio when only one CSS dimension is provided.
    pub fn intrinsic_size(&self) -> (u32, u32) {
        match self {
            DecodedImage::Svg { intrinsic_size, .. } => *intrinsic_size,
            DecodedImage::Raster(img) => (img.width, img.height),
        }
    }

    /// Build a `vello::peniko::ImageData` for painting. For SVG we return `None`
    /// (the paint stage draws the path directly via `BezPath`); for raster we
    /// return the decoded bytes.
    pub fn raster(&self) -> Option<ImageData> {
        match self {
            DecodedImage::Svg { .. } => None,
            DecodedImage::Raster(img) => Some(img.clone()),
        }
    }
}

/// Resolves a (possibly relative) `<img src>` against the editor assets dir.
fn resolve_src(src: &str) -> Option<PathBuf> {
    if src.is_empty() || src.starts_with("http://") || src.starts_with("https://") {
        return None;
    }
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/editor");
    // `src` may be relative ("favicon.png", "./folder.svg", "../favicon.png").
    // Join handles the common cases. A leading "../" can walk out of
    // assets/editor (e.g. the template's "../favicon.png" which is actually
    // stored under assets/editor/), so if the joined path doesn't exist we
    // fall back to looking the bare filename up directly in the assets dir.
    let joined = base.join(src);
    if joined.exists() {
        Some(joined)
    } else if let Some(name) = Path::new(src).file_name() {
        let direct = base.join(name);
        if direct.exists() { Some(direct) } else { None }
    } else {
        None
    }
}

/// Loads and decodes an `<img src>` into a [`DecodedImage`].
///
/// - `.svg` files are parsed for their first `<path d="...">` and `viewBox`.
/// - everything else (`.png`, `.jpg`, `.gif`, …) goes through the `image` crate
///   and is converted to straight-alpha RGBA8 `peniko::ImageData`.
///
/// Returns `None` if the source is a remote URL, missing, or undecodable — the
/// caller should treat that as a missing image (the box may still have a CSS
/// size from the stylesheet).
pub fn load_image(src: &str) -> Option<DecodedImage> {
    let path = resolve_src(src)?;
    let lower = path.to_string_lossy().to_lowercase();
    if lower.ends_with(".svg") {
        load_svg(&path)
    } else {
        load_raster(&path)
    }
}

fn load_svg(path: &Path) -> Option<DecodedImage> {
    let content = std::fs::read_to_string(path).ok()?;
    let view_box = find_attr(&content, "svg", "viewBox")
        .and_then(|vb| parse_view_box(&vb))
        .or_else(|| Some((0.0, 0.0, 24.0, 24.0)))?;
    // Collect every `<path>` opening tag, then pick the first *visible* one
    // (i.e. whose resolved fill is not `none`). SVGs exported from editors
    // (Inkscape, etc.) often start with a transparent bounding-box path
    // (`<path d="M0 0h24v24H0z" fill="none"/>`) that must be skipped, otherwise
    // the icon paints nothing. `fill` may live in a `fill="..."` attribute or
    // inside a `style="fill:...;..."` attribute.
    let mut best_d: Option<String> = None;
    let mut best_fill: Option<String> = None;
    for path_tag in find_all_tags(&content, "path") {
        let d = find_attr_in_tag(&path_tag, "d")?;
        let fill = extract_fill(&path_tag);
        let is_visible = match fill.as_deref() {
            Some("none") | Some("transparent") => false,
            _ => true,
        };
        if is_visible {
            best_d = Some(d);
            best_fill = fill;
            break;
        }
        // Fallback: keep the first path even if it looked invisible.
        if best_d.is_none() {
            best_d = Some(d);
            best_fill = fill;
        }
    }
    let path_d = best_d?;

    let (vbw, vbh) = (view_box.2.max(1.0), view_box.3.max(1.0));
    // Clamp oversized viewBoxes (960x960 Material icons) to a sane icon size so
    // the intrinsic aspect ratio is preserved without blowing up the layout.
    let (iw, ih) = if vbw.max(vbh) <= 64.0 {
        (vbw as u32, vbh as u32)
    } else {
        let scale = 24.0 / vbw.max(vbh);
        ((vbw * scale) as u32, (vbh * scale) as u32)
    };
    Some(DecodedImage::Svg {
        path_d,
        view_box,
        intrinsic_size: (iw, ih),
        fill: best_fill,
    })
}

/// Returns the text of every opening tag named `tag` (e.g. every `<path ...>`).
fn find_all_tags(html: &str, tag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let open = format!("<{tag}");
    let mut rest = html;
    while let Some(start) = rest.find(&open) {
        let after = match rest[start..].find('>') {
            Some(i) => start + i + 1,
            None => break,
        };
        let tag_text = &rest[start..after];
        out.push(tag_text.to_string());
        rest = &rest[after..];
    }
    out
}

/// Extracts the value of `attr` from a single opening-tag string (already
/// sliced to the `<tag ...>` portion).
fn find_attr_in_tag(tag_text: &str, attr: &str) -> Option<String> {
    let pat = format!("{attr}=");
    let idx = tag_text.find(&pat)? + pat.len();
    let rest = &tag_text[idx..];
    let quote = rest.chars().next()?;
    let q = if quote == '"' || quote == '\'' {
        quote
    } else {
        return None;
    };
    let close = rest[1..].find(q)?;
    Some(rest[1..1 + close].to_string())
}

/// Resolves a CSS `fill` color from a tag's attributes, checking both the
/// `fill="..."` attribute and a `style="...; fill:...; ..."` attribute.
fn extract_fill(tag_text: &str) -> Option<String> {
    if let Some(f) = find_attr_in_tag(tag_text, "fill") {
        // A bare `fill="..."` wins.
        return Some(f);
    }
    // Fall back to parsing `fill:` out of the `style` attribute.
    let style = find_attr_in_tag(tag_text, "style")?;
    let idx = style.find("fill:")? + "fill:".len();
    let rest = style[idx..].trim_start();
    let end = rest
        .find(|c: char| c == ';' || c == '"' || c.is_whitespace())
        .unwrap_or(rest.len());
    let val = rest[..end].trim().to_string();
    if val.is_empty() { None } else { Some(val) }
}

fn load_raster(path: &Path) -> Option<DecodedImage> {
    let bytes = std::fs::read(path).ok()?;
    let img = image::load_from_memory(&bytes).ok()?;
    let img = img.to_rgba8();
    let (w, h) = (img.width(), img.height());
    // `image` gives straight (unpremultiplied) alpha; peniko expects the same.
    let raw: Vec<u8> = img.into_raw();
    let data = Blob::new(std::sync::Arc::new(raw) as std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>);
    Some(DecodedImage::Raster(ImageData {
        data,
        format: vello::peniko::ImageFormat::Rgba8,
        alpha_type: vello::peniko::ImageAlphaType::Alpha,
        width: w,
        height: h,
    }))
}

/// Extracts an attribute value from the first tag of the given name.
fn find_attr(html: &str, tag: &str, attr: &str) -> Option<String> {
    let open = format!("<{tag}");
    let start = html.find(&open)? + open.len();
    let tail = &html[start..];
    let end = tail.find('>')?;
    let tag_text = &tail[..end];
    let pat = format!("{attr}=");
    let idx = tag_text.find(&pat)? + pat.len();
    let rest = &tag_text[idx..];
    let quote = rest.chars().next()?;
    let q = if quote == '"' || quote == '\'' {
        quote
    } else {
        return None;
    };
    let close = rest[1..].find(q)?;
    Some(rest[1..1 + close].to_string())
}

fn parse_view_box(vb: &str) -> Option<(f32, f32, f32, f32)> {
    let parts: Vec<f32> = vb
        .split_whitespace()
        .filter_map(|p| p.parse::<f32>().ok())
        .collect();
    if parts.len() == 4 && parts[2] > 0.0 && parts[3] > 0.0 {
        Some((parts[0], parts[1], parts[2], parts[3]))
    } else {
        None
    }
}

/// Process-wide cache so re-building the layout (on resize / editor toggle)
/// does not re-read and re-decode image assets from disk every time.
#[derive(Default)]
pub struct ImageCache {
    map: Mutex<HashMap<String, Option<Arc<DecodedImage>>>>,
}

impl std::fmt::Debug for ImageCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageCache").finish()
    }
}

impl ImageCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a cached decoded image for `src`, decoding on a miss.
    pub fn get(&self, src: &str) -> Option<Arc<DecodedImage>> {
        if let Some(cached) = self.map.lock().ok().and_then(|m| m.get(src).cloned()) {
            return cached;
        }
        let decoded = load_image(src).map(Arc::new);
        if let Ok(mut m) = self.map.lock() {
            m.insert(src.to_string(), decoded.clone());
        }
        decoded
    }
}
