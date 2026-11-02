//! CPU-rasterized stats panel: an RGBA8 [`Canvas`] with rectangle and
//! bitmap-text primitives, and [`draw_panel`] laying out the readouts and
//! the frame-time history graph from a [`StatsCollector`]. Kept free of
//! GPU types so the layout is unit-testable; the window only uploads the
//! finished pixels.

use std::time::Duration;

use super::StatsCollector;
use super::font::{self, ADVANCE, HEIGHT, WIDTH};

/// RGBA8 color.
pub type Rgba = [u8; BYTES_PER_PIXEL];
/// Bytes per RGBA8 pixel.
pub const BYTES_PER_PIXEL: usize = 4;
/// Pixel stride as `u32` (row pitch arithmetic).
pub const PIXEL_STRIDE: u32 = BYTES_PER_PIXEL as u32;

const BACKGROUND: Rgba = [18, 18, 24, 255];
const TEXT: Rgba = [220, 220, 220, 255];
const DIM: Rgba = [130, 130, 140, 255];
const GOOD: Rgba = [90, 200, 110, 255];
const WARN: Rgba = [230, 190, 60, 255];
const BAD: Rgba = [235, 80, 70, 255];
const GRID: Rgba = [60, 60, 72, 255];
const GRAPH_BACKGROUND: Rgba = [0, 0, 0, 255];
/// Logical panel margin and line pitch (font pixels).
const MARGIN: u32 = 6;
const LINE: u32 = HEIGHT + 4;
/// History graph height in logical pixels.
const GRAPH_HEIGHT: u32 = 80;
/// Graph full scale: 50 ms (bars above it are clipped and drawn red).
const GRAPH_SCALE: Duration = Duration::from_millis(50);
/// Frame-time budgets coloring bars: 60 FPS and 30 FPS.
const BUDGET_60: Duration = Duration::from_micros(16_667);
const BUDGET_30: Duration = Duration::from_micros(33_333);

/// Software framebuffer in physical pixels with an integer UI `scale`
/// (logical → physical, HiDPI).
#[derive(Debug, Clone)]
pub struct Canvas {
    /// Width in physical pixels.
    pub width: u32,
    /// Height in physical pixels.
    pub height: u32,
    scale: u32,
    /// Row-major RGBA8 pixels.
    pub pixels: Vec<u8>,
}

impl Canvas {
    /// Blank canvas of `width × height` physical pixels.
    pub fn new(width: u32, height: u32, scale: u32) -> Self {
        Self {
            width,
            height,
            scale: scale.max(1),
            pixels: vec![0; (width as usize) * (height as usize) * BYTES_PER_PIXEL],
        }
    }

    /// Fills the logical rectangle (clipped to the canvas).
    pub fn fill_rect(&mut self, x: u32, y: u32, w: u32, h: u32, color: Rgba) {
        let s = self.scale;
        let (x0, y0) = ((x * s).min(self.width), (y * s).min(self.height));
        let (x1, y1) = (
            ((x + w) * s).min(self.width),
            ((y + h) * s).min(self.height),
        );
        for row in y0..y1 {
            let start = ((row * self.width + x0) * PIXEL_STRIDE) as usize;
            let end = ((row * self.width + x1) * PIXEL_STRIDE) as usize;
            for px in self.pixels[start..end].chunks_exact_mut(BYTES_PER_PIXEL) {
                px.copy_from_slice(&color);
            }
        }
    }

    /// Draws `text` with its top-left corner at logical `(x, y)`.
    pub fn text(&mut self, x: u32, y: u32, text: &str, color: Rgba) {
        for (i, ch) in text.chars().enumerate() {
            let Some(rows) = font::glyph(ch) else {
                continue;
            };
            let gx = x + (i as u32) * ADVANCE;
            for (dy, bits) in rows.iter().enumerate() {
                for dx in 0..WIDTH {
                    if u32::from(*bits) >> (WIDTH - 1 - dx) & 1 != 0 {
                        self.fill_rect(gx + dx, y + dy as u32, 1, 1, color);
                    }
                }
            }
        }
    }

    /// Logical width (physical / scale).
    fn logical_width(&self) -> u32 {
        self.width / self.scale
    }
}

/// Milliseconds with two decimals.
pub fn format_ms(d: Duration) -> String {
    format!("{:.2}", d.as_secs_f64() * 1000.0)
}

/// Bar color by frame budget.
pub fn budget_color(frame: Duration) -> Rgba {
    if frame <= BUDGET_60 {
        GOOD
    } else if frame <= BUDGET_30 {
        WARN
    } else {
        BAD
    }
}

/// Redraws the whole panel from `stats`.
pub fn draw_panel(canvas: &mut Canvas, stats: &StatsCollector) {
    let width = canvas.logical_width();
    canvas.fill_rect(0, 0, width, canvas.height, BACKGROUND);
    let avg = stats.frames.average();
    let worst = stats.worst_frame();
    let mut y = MARGIN;
    let header = [
        (format!("FPS {:.1}", stats.frames.fps()), TEXT),
        (
            format!("FRAME AVG {} MS", format_ms(avg)),
            budget_color(avg),
        ),
        (
            format!("FRAME WORST 2S {} MS", format_ms(worst)),
            budget_color(worst),
        ),
        (
            format!("CPU WORK AVG {} MS", format_ms(stats.work.average())),
            TEXT,
        ),
    ];
    for (line, color) in header {
        canvas.text(MARGIN, y, &line, color);
        y += LINE;
    }
    y += MARGIN;
    draw_graph(canvas, stats, y, width.saturating_sub(2 * MARGIN));
    y += GRAPH_HEIGHT + MARGIN + 2;
    canvas.text(MARGIN, y, "PHASE / SYSTEM        AVG MS  MAX MS", DIM);
    y += LINE;
    for row in &stats.rows {
        let indent = if row.label.contains('/') { 2 } else { 0 };
        let line = format!(
            "{:<20}{:>8}{:>8}",
            row.label.rsplit('/').next().unwrap_or(&row.label),
            format_ms(row.avg),
            format_ms(row.max)
        );
        canvas.text(MARGIN + indent * ADVANCE, y, &line, TEXT);
        y += LINE;
    }
}

/// Frame-time history as one bar per frame (newest on the right) with
/// 60/30 FPS budget lines.
fn draw_graph(canvas: &mut Canvas, stats: &StatsCollector, top: u32, width: u32) {
    canvas.fill_rect(MARGIN, top, width, GRAPH_HEIGHT, GRAPH_BACKGROUND);
    let scale_ns = GRAPH_SCALE.as_nanos().max(1);
    let to_px =
        |d: Duration| (d.as_nanos().min(scale_ns) * u128::from(GRAPH_HEIGHT) / scale_ns) as u32;
    for budget in [BUDGET_60, BUDGET_30] {
        canvas.fill_rect(MARGIN, top + GRAPH_HEIGHT - to_px(budget), width, 1, GRID);
    }
    let samples: Vec<Duration> = stats.frames.iter_chronological().collect();
    let skip = samples.len().saturating_sub(width as usize);
    let left = MARGIN + width.saturating_sub((samples.len() - skip) as u32);
    for (i, frame) in samples[skip..].iter().enumerate() {
        let h = to_px(*frame).max(1);
        canvas.fill_rect(
            left + i as u32,
            top + GRAPH_HEIGHT - h,
            1,
            h,
            budget_color(*frame),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(c: &Canvas, x: u32, y: u32) -> Rgba {
        let i = ((y * c.width + x) * 4) as usize;
        [
            c.pixels[i],
            c.pixels[i + 1],
            c.pixels[i + 2],
            c.pixels[i + 3],
        ]
    }

    #[test]
    fn fill_rect_scales_and_clips() {
        let mut c = Canvas::new(8, 8, 2);
        c.fill_rect(1, 1, 10, 1, GOOD);
        assert_eq!(pixel(&c, 2, 2), GOOD);
        assert_eq!(pixel(&c, 7, 3), GOOD);
        assert_eq!(pixel(&c, 1, 2), [0; 4]);
        assert_eq!(pixel(&c, 2, 4), [0; 4]);
    }

    #[test]
    fn text_sets_glyph_pixels() {
        let mut c = Canvas::new(16, 8, 1);
        c.text(0, 0, "1", TEXT);
        assert_eq!(pixel(&c, 2, 0), TEXT, "top of the 1 stem");
        assert_eq!(pixel(&c, 0, 0), [0; 4]);
    }

    #[test]
    fn budget_colors_and_ms_format() {
        assert_eq!(budget_color(Duration::from_millis(10)), GOOD);
        assert_eq!(budget_color(Duration::from_millis(20)), WARN);
        assert_eq!(budget_color(Duration::from_millis(40)), BAD);
        assert_eq!(format_ms(Duration::from_micros(16_667)), "16.67");
    }

    #[test]
    fn panel_draws_graph_bars_for_frames() {
        let mut stats = StatsCollector::default();
        for _ in 0..3 {
            stats.record(Duration::from_millis(40), Duration::ZERO, &[]);
        }
        let mut c = Canvas::new(300, 400, 1);
        draw_panel(&mut c, &stats);
        let graph_bottom = MARGIN + 4 * LINE + MARGIN + GRAPH_HEIGHT - 1;
        let right = MARGIN + 300 - 2 * MARGIN - 1;
        assert_eq!(pixel(&c, right, graph_bottom), BAD);
        assert_eq!(pixel(&c, 0, 0), BACKGROUND);
    }
}
