// Colours, as in the terminal game: tiles coloured by value like the
// physical game and shaded by level (the higher, the brighter).

const VALUE = [
  [154, 160, 166], // 0 grey
  [160, 102, 60], // 1 brown
  [242, 140, 40], // 2 orange
  [242, 197, 61], // 3 yellow
  [76, 175, 80], // 4 green
  [79, 195, 247], // 5 light blue
  [63, 81, 181], // 6 dark blue
  [156, 39, 176], // 7 purple
  [240, 98, 146], // 8 pink
  [229, 57, 53], // 9 red
];

const LEVEL = [
  [47, 59, 82],
  [43, 108, 163],
  [42, 157, 143],
  [233, 196, 106],
  [244, 162, 97],
  [231, 111, 81],
  [214, 69, 120],
  [170, 70, 190],
  [120, 90, 230],
  [235, 235, 245],
];

export const EMPTY_DOT = "rgb(70,74,86)";
export const ILLEGAL_BG = "rgb(110,28,28)";
export const ILLEGAL_FG = "rgb(255,130,130)";
export const HINT = "rgb(80,230,240)";
export const GOOD = "rgb(120,220,120)";
export const BAD = "rgb(255,120,120)";
export const MUTED = "rgb(140,145,160)";
export const ACCENT = "rgb(255,200,80)";

const css = ([r, g, b]) => `rgb(${r},${g},${b})`;
const shade = (layer) => [0.55, 0.72, 0.86, 1][Math.min(layer, 3)];
const scale = (c, f) => c.map((x) => Math.min(255, Math.round(x * f)));

/// [r, g, b] of a tile of `value` on `layer` (or at full brightness).
export const tileRgb = (value, layer = 3) => scale(VALUE[value], shade(layer));
export const levelRgb = (layer) => LEVEL[Math.min(layer, LEVEL.length - 1)];
export const valueCss = (value) => css(VALUE[value]);
export const levelCss = (layer) => css(levelRgb(layer));
export { css };

/// Black or white, whichever reads better on `rgb`.
export function textOn([r, g, b]) {
  return (r * 299 + g * 587 + b * 114) / 1000 > 140 ? "#000" : "#fff";
}

export const superscript = (n) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Math.min(n, 9)];
