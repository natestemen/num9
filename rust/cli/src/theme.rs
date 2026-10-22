//! Colours. Tiles are coloured by value, like the physical game, and shaded
//! by level: the higher a tile sits, the brighter it is.

use ratatui::style::Color;

const VALUE: [(u8, u8, u8); 10] = [
    (154, 160, 166), // 0 grey
    (160, 102, 60),  // 1 brown
    (242, 140, 40),  // 2 orange
    (242, 197, 61),  // 3 yellow
    (76, 175, 80),   // 4 green
    (79, 195, 247),  // 5 light blue
    (63, 81, 181),   // 6 dark blue
    (156, 39, 176),  // 7 purple
    (240, 98, 146),  // 8 pink
    (229, 57, 53),   // 9 red
];

const LEVEL: [(u8, u8, u8); 10] = [
    (47, 59, 82),
    (43, 108, 163),
    (42, 157, 143),
    (233, 196, 106),
    (244, 162, 97),
    (231, 111, 81),
    (214, 69, 120),
    (170, 70, 190),
    (120, 90, 230),
    (235, 235, 245),
];

pub const EMPTY_DOT: Color = Color::Rgb(70, 74, 86);
pub const ILLEGAL_BG: Color = Color::Rgb(110, 28, 28);
pub const ILLEGAL_FG: Color = Color::Rgb(255, 130, 130);
pub const HINT: Color = Color::Rgb(80, 230, 240);
pub const GOOD: Color = Color::Rgb(120, 220, 120);
pub const BAD: Color = Color::Rgb(255, 120, 120);
pub const MUTED: Color = Color::Rgb(140, 145, 160);
pub const ACCENT: Color = Color::Rgb(255, 200, 80);

fn rgb((r, g, b): (u8, u8, u8)) -> Color {
    Color::Rgb(r, g, b)
}

fn scale((r, g, b): (u8, u8, u8), f: f32) -> (u8, u8, u8) {
    let s = |x: u8| (x as f32 * f).round().min(255.0) as u8;
    (s(r), s(g), s(b))
}

/// Brightness for a tile on `layer`: lower tiles are darker.
fn shade(layer: usize) -> f32 {
    [0.55, 0.72, 0.86, 1.0][layer.min(3)]
}

/// A tile of `value` as it looks on `layer`.
pub fn tile(value: u8, layer: usize) -> Color {
    rgb(scale(VALUE[value as usize], shade(layer)))
}

/// A tile of `value` at full brightness (the tile in hand).
pub fn value(value: u8) -> Color {
    rgb(VALUE[value as usize])
}

/// Height-map colour for `layer`.
pub fn level(layer: usize) -> Color {
    rgb(LEVEL[layer.min(LEVEL.len() - 1)])
}

/// Black or white, whichever reads better on `bg`.
pub fn text_on(bg: Color) -> Color {
    match bg {
        Color::Rgb(r, g, b) if (r as u32 * 299 + g as u32 * 587 + b as u32 * 114) / 1000 > 140 => {
            Color::Black
        }
        _ => Color::White,
    }
}

/// Small raised digit for a level number.
pub fn superscript(n: usize) -> char {
    ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'][n.min(9)]
}
