// Drawing the board on a canvas, as the terminal game draws it: tiles
// coloured by value (or by level, as a height map) with their value and
// level written on them, the engine's suggestion hatched in cyan, and the
// tile in hand on top.

import * as theme from "./theme.js";
import { cells, geometry, labelCell, size } from "./game.js";

const FONT = '"JetBrains Mono", ui-monospace, monospace';

export class BoardView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.cell = 26;
    this.vp = null;
  }

  /// Fit the canvas to its box; returns whether the grid size changed.
  resize() {
    const box = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(box.width));
    const h = Math.max(1, Math.floor(box.height));
    if (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr) {
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
      this.canvas.style.width = `${w}px`;
      this.canvas.style.height = `${h}px`;
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w;
    this.h = h;
    const rows = Math.max(1, Math.floor(h / this.cell));
    const cols = Math.max(1, Math.floor(w / this.cell));
    const changed = !this.vp || this.vp.rows !== rows || this.vp.cols !== cols;
    if (changed && this.vp) {
      // Keep the same middle.
      const ci = this.vp.top + this.vp.rows / 2;
      const cj = this.vp.left + this.vp.cols / 2;
      this.vp = this.clampVp({ top: Math.round(ci - rows / 2), left: Math.round(cj - cols / 2), rows, cols });
    } else if (!this.vp) {
      this.vp = { top: 0, left: 0, rows, cols };
      this.fresh = true;
    }
    return changed;
  }

  zoom(delta) {
    this.setCell(this.cell + delta);
  }

  setCell(px) {
    this.cell = Math.round(Math.max(10, Math.min(56, px)));
    this.resize();
  }

  clampVp(v) {
    v.top = Math.max(0, Math.min(v.top, geometry.height - v.rows));
    v.left = Math.max(0, Math.min(v.left, geometry.width - v.cols));
    return v;
  }

  /// Centre on cells (min_i, max_i, min_j, max_j).
  centre([a, b, c, d]) {
    if (!this.vp) this.resize();
    const v = this.vp;
    v.top = Math.round((a + b) / 2 - v.rows / 2);
    v.left = Math.round((c + d) / 2 - v.cols / 2);
    this.clampVp(v);
    this.fresh = false;
  }

  /// Keep the tile in hand on screen (centring on everything the first time).
  follow(cursor, value, rot, pieces) {
    const v = this.vp;
    const { h, w } = size(value ?? 0, rot);
    if (this.fresh) {
      this.centre(extent(pieces, [cursor.i, cursor.i + h, cursor.j, cursor.j + w]));
    }
    const mi = Math.min(2, Math.floor(Math.max(0, v.rows - h) / 2));
    const mj = Math.min(3, Math.floor(Math.max(0, v.cols - w) / 2));
    if (cursor.i < v.top + mi) v.top = cursor.i - mi;
    if (cursor.i + h + mi > v.top + v.rows) v.top = cursor.i + h + mi - v.rows;
    if (cursor.j < v.left + mj) v.left = cursor.j - mj;
    if (cursor.j + w + mj > v.left + v.cols) v.left = cursor.j + w + mj - v.cols;
    this.clampVp(v);
  }

  /// Board cell under a point on the page, if any.
  cellAt(x, y) {
    const r = this.canvas.getBoundingClientRect();
    const { ox, oy } = this.offset();
    const c = Math.floor((x - r.left - ox) / this.cell);
    const row = Math.floor((y - r.top - oy) / this.cell);
    if (c < 0 || row < 0 || c >= this.vp.cols || row >= this.vp.rows) return null;
    return [this.vp.top + row, this.vp.left + c];
  }

  /// Centre the grid in the canvas.
  offset() {
    return {
      ox: Math.floor((this.w - this.vp.cols * this.cell) / 2),
      oy: Math.floor((this.h - this.vp.rows * this.cell) / 2),
    };
  }

  /// Draw `pieces` (with `top`, their TopView). Options: `heightMap`,
  /// `ghost` ({value, move, problem}), `hint` ({value, move}), `last`
  /// (index of the piece to underline).
  draw(pieces, top, { heightMap = false, ghost = null, hint = null, last = pieces.length - 1 } = {}) {
    const { ctx, cell: s, vp } = this;
    const { ox, oy } = this.offset();
    ctx.clearRect(0, 0, this.w, this.h);
    const x = (j) => ox + (j - vp.left) * s;
    const y = (i) => oy + (i - vp.top) * s;
    const inView = (i, j) => i >= vp.top && j >= vp.left && i < vp.top + vp.rows && j < vp.left + vp.cols;
    const labels = new Map(pieces.map((p, k) => [labelCell(p).join(","), k]));

    for (let i = vp.top; i < vp.top + vp.rows; i++) {
      for (let j = vp.left; j < vp.left + vp.cols; j++) {
        const t = top.get(i, j);
        if (!t) {
          ctx.fillStyle = theme.EMPTY_DOT;
          ctx.fillRect(x(j) + s / 2 - 1, y(i) + s / 2 - 1, 2, 2);
          continue;
        }
        const rgb = heightMap ? theme.levelRgb(t.layer) : theme.tileRgb(t.value, t.layer);
        ctx.fillStyle = theme.css(rgb);
        ctx.fillRect(x(j), y(i), s, s);
        if (labels.get(`${i},${j}`) === t.piece) {
          this.label(x(j), y(i), `${t.value}${theme.superscript(t.layer)}`, theme.textOn(rgb), t.piece === last);
        }
      }
    }
    // Outline each piece where it meets another or the floor.
    ctx.strokeStyle = "rgba(10,12,16,0.75)";
    ctx.lineWidth = Math.max(1, s / 14);
    ctx.beginPath();
    for (let i = vp.top; i < vp.top + vp.rows; i++) {
      for (let j = vp.left; j < vp.left + vp.cols; j++) {
        const t = top.get(i, j);
        if (!t) continue;
        const other = (a, b) => top.get(a, b)?.piece !== t.piece;
        if (other(i - 1, j)) edge(ctx, x(j), y(i), x(j) + s, y(i));
        if (other(i + 1, j)) edge(ctx, x(j), y(i) + s, x(j) + s, y(i) + s);
        if (other(i, j - 1)) edge(ctx, x(j), y(i), x(j), y(i) + s);
        if (other(i, j + 1)) edge(ctx, x(j) + s, y(i), x(j) + s, y(i) + s);
      }
    }
    ctx.stroke();

    if (hint) {
      const cs = cells(hint.value, hint.move).filter(([i, j]) => inView(i, j));
      for (const [i, j] of cs) hatch(ctx, x(j), y(i), s, theme.HINT, 0.9);
      outline(ctx, cs, x, y, s, theme.HINT, Math.max(2, s / 9));
    }

    if (ghost) {
      const cs = cells(ghost.value, ghost.move).filter(([i, j]) => inView(i, j));
      const legal = !ghost.problem;
      const rgb = theme.tileRgb(ghost.value);
      cs.forEach(([i, j], k) => {
        ctx.fillStyle = legal ? theme.css(rgb) : theme.ILLEGAL_BG;
        ctx.globalAlpha = legal ? 0.93 : 0.9;
        ctx.fillRect(x(j), y(i), s, s);
        ctx.globalAlpha = 1;
        if (k > 0) dots(ctx, x(j), y(i), s, legal ? "rgba(255,255,255,0.35)" : theme.ILLEGAL_FG);
      });
      if (cs.length) {
        const [i, j] = cs[0];
        const text = legal ? `${ghost.value}${theme.superscript(ghost.move.layer)}` : "✗";
        this.label(x(j), y(i), text, legal ? theme.textOn(rgb) : theme.ILLEGAL_FG, false, true);
      }
      outline(ctx, cs, x, y, s, legal ? "rgba(255,255,255,0.9)" : theme.ILLEGAL_FG, Math.max(1.5, s / 13));
    }
  }

  label(px, py, text, color, underline, bold = false) {
    const { ctx, cell: s } = this;
    ctx.fillStyle = color;
    ctx.font = `${bold || underline ? 700 : 500} ${Math.round(s * 0.56)}px ${FONT}`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillText(text, px + s / 2, py + s / 2 + 1);
    if (underline) {
      ctx.fillRect(px + s * 0.2, py + s * 0.82, s * 0.6, Math.max(1, s / 14));
    }
  }
}

/// Ground-level extent of `pieces` and a box (min_i, max_i, min_j, max_j).
export function extent(pieces, [a, b, c, d] = [Infinity, -Infinity, Infinity, -Infinity]) {
  for (const p of pieces) {
    for (const [i, j] of cells(p.value, p)) {
      a = Math.min(a, i);
      b = Math.max(b, i + 1);
      c = Math.min(c, j);
      d = Math.max(d, j + 1);
    }
  }
  return [a, b, c, d];
}

function edge(ctx, x0, y0, x1, y1) {
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
}

function hatch(ctx, px, py, s, color, alpha) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(px, py, s, s);
  ctx.clip();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1, s / 10);
  ctx.beginPath();
  for (let k = -s; k < s; k += s / 3) {
    ctx.moveTo(px + k, py + s);
    ctx.lineTo(px + k + s, py);
  }
  ctx.stroke();
  ctx.restore();
}

function dots(ctx, px, py, s, color) {
  ctx.fillStyle = color;
  const d = Math.max(1, s / 12);
  for (const fx of [0.3, 0.7]) for (const fy of [0.3, 0.7]) ctx.fillRect(px + s * fx - d / 2, py + s * fy - d / 2, d, d);
}

/// Stroke the outer boundary of a set of cells.
function outline(ctx, cs, x, y, s, color, width) {
  const set = new Set(cs.map(([i, j]) => `${i},${j}`));
  const has = (i, j) => set.has(`${i},${j}`);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  for (const [i, j] of cs) {
    if (!has(i - 1, j)) edge(ctx, x(j), y(i), x(j) + s, y(i));
    if (!has(i + 1, j)) edge(ctx, x(j), y(i) + s, x(j) + s, y(i) + s);
    if (!has(i, j - 1)) edge(ctx, x(j), y(i), x(j), y(i) + s);
    if (!has(i, j + 1)) edge(ctx, x(j) + s, y(i), x(j) + s, y(i) + s);
  }
  ctx.stroke();
}
