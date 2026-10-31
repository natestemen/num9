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

  /// Keep the tile in hand on screen (centring on everything the first time,
  /// and moving with the tile only if `move`).
  follow(cursor, value, rot, pieces, move = true) {
    const v = this.vp;
    const { h, w } = size(value ?? 0, rot);
    if (this.fresh) {
      this.centre(extent(pieces, [cursor.i, cursor.i + h, cursor.j, cursor.j + w]));
    }
    if (!move) return;
    const mi = Math.min(2, Math.floor(Math.max(0, v.rows - h) / 2));
    const mj = Math.min(3, Math.floor(Math.max(0, v.cols - w) / 2));
    if (cursor.i < v.top + mi) v.top = cursor.i - mi;
    if (cursor.i + h + mi > v.top + v.rows) v.top = cursor.i + h + mi - v.rows;
    if (cursor.j < v.left + mj) v.left = cursor.j - mj;
    if (cursor.j + w + mj > v.left + v.cols) v.left = cursor.j + w + mj - v.cols;
    this.clampVp(v);
  }

  /// Put the view's top-left corner at (`top`, `left`).
  panTo(top, left) {
    this.vp.top = top;
    this.vp.left = left;
    this.clampVp(this.vp);
    this.fresh = false;
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
  /// (index of the piece to underline), `mark` (a {value, move} to outline),
  /// `spotlight` (a level: tiles on others are dimmed), `outlines` (more
  /// {value, move, color} to outline).
  draw(
    pieces,
    top,
    { heightMap = false, ghost = null, hint = null, last = pieces.length - 1, mark = null, spotlight = null, outlines = [] } = {},
  ) {
    const { ctx, cell: s, vp } = this;
    const { ox, oy } = this.offset();
    ctx.clearRect(0, 0, this.w, this.h);
    const x = (j) => ox + (j - vp.left) * s;
    const y = (i) => oy + (i - vp.top) * s;
    const inView = (i, j) => i >= vp.top && j >= vp.left && i < vp.top + vp.rows && j < vp.left + vp.cols;
    const labels = new Map(pieces.map((p, k) => [labelCell(p).join(","), k]));

    const rows = [vp.top, vp.top + vp.rows];
    const colsRange = [vp.left, vp.left + vp.cols];
    const each = (f) => {
      for (let i = rows[0]; i < rows[1]; i++) for (let j = colsRange[0]; j < colsRange[1]; j++) f(i, j, top.get(i, j));
    };

    // The tiles (and dots on the empty floor).
    each((i, j, t) => {
      if (!t) {
        ctx.fillStyle = theme.EMPTY_DOT;
        ctx.fillRect(x(j) + s / 2 - 1, y(i) + s / 2 - 1, 2, 2);
        return;
      }
      ctx.fillStyle = theme.css(heightMap ? theme.levelRgb(t.layer) : theme.tileRgb(t.value, t.layer));
      ctx.fillRect(x(j), y(i), s, s);
    });

    // Shadows: with the light from the top left, a taller neighbour above or
    // to the left casts a shadow on a cell, longer the bigger the drop.
    each((i, j, t) => {
      const h = t ? t.layer + 1 : 0;
      const up = top.height(i - 1, j) - h;
      const left = top.height(i, j - 1) - h;
      const diag = top.height(i - 1, j - 1) - h;
      const depth = (d) => s * Math.min(0.5, 0.14 + 0.1 * d);
      if (up > 0) shade(ctx, x(j), y(i), s, depth(up), "down");
      if (left > 0) shade(ctx, x(j), y(i), s, depth(left), "right");
      if (diag > 0 && up <= 0 && left <= 0) {
        const d = depth(diag);
        const g = ctx.createRadialGradient(x(j), y(i), 0, x(j), y(i), d);
        g.addColorStop(0, "rgba(0,0,0,0.5)");
        g.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = g;
        ctx.fillRect(x(j), y(i), d, d);
      }
    });

    // Outlines: thin seams between tiles on the same level, heavier lines
    // where the level changes.
    const seams = new Path2D();
    const steps = new Path2D();
    each((i, j, t) => {
      if (!t) return;
      const line = (a, b, x0, y0, x1, y1) => {
        const o = top.get(a, b);
        if (o?.piece === t.piece) return;
        (o?.layer === t.layer ? seams : steps).moveTo(x0, y0);
        (o?.layer === t.layer ? seams : steps).lineTo(x1, y1);
      };
      line(i - 1, j, x(j), y(i), x(j) + s, y(i));
      line(i + 1, j, x(j), y(i) + s, x(j) + s, y(i) + s);
      line(i, j - 1, x(j), y(i), x(j), y(i) + s);
      line(i, j + 1, x(j) + s, y(i), x(j) + s, y(i) + s);
    });
    ctx.strokeStyle = "rgba(10,12,16,0.55)";
    ctx.lineWidth = Math.max(1, s / 16);
    ctx.stroke(seams);
    ctx.strokeStyle = "rgba(6,7,10,0.95)";
    ctx.lineWidth = Math.max(1.5, s / 10);
    ctx.stroke(steps);

    // A rim in the level's colour (as in the Levels box) inside every tile
    // above the ground.
    if (!heightMap) {
      const w = Math.max(1.5, s / 13);
      const inset = w / 2 + Math.max(1, s / 22);
      each((i, j, t) => {
        if (!t || t.layer === 0) return;
        const edgeOf = (a, b) => top.get(a, b)?.piece !== t.piece;
        const [n, so, we, ea] = [edgeOf(i - 1, j), edgeOf(i + 1, j), edgeOf(i, j - 1), edgeOf(i, j + 1)];
        const x0 = x(j);
        const y0 = y(i);
        ctx.strokeStyle = theme.css(theme.rimRgb(t.layer));
        ctx.lineWidth = w;
        ctx.beginPath();
        const a0 = we ? x0 + inset : x0;
        const a1 = ea ? x0 + s - inset : x0 + s;
        const b0 = n ? y0 + inset : y0;
        const b1 = so ? y0 + s - inset : y0 + s;
        if (n) edge(ctx, a0, y0 + inset, a1, y0 + inset);
        if (so) edge(ctx, a0, y0 + s - inset, a1, y0 + s - inset);
        if (we) edge(ctx, x0 + inset, b0, x0 + inset, b1);
        if (ea) edge(ctx, x0 + s - inset, b0, x0 + s - inset, b1);
        ctx.stroke();
      });
    }

    if (spotlight != null) {
      ctx.fillStyle = "rgba(15,17,22,0.78)";
      each((i, j, t) => {
        if (t && t.layer !== spotlight) ctx.fillRect(x(j), y(i), s, s);
      });
    }

    // Labels last, over shadows and rims.
    each((i, j, t) => {
      if (spotlight != null && t?.layer !== spotlight) return;
      if (t && labels.get(`${i},${j}`) === t.piece) {
        const rgb = heightMap ? theme.levelRgb(t.layer) : theme.tileRgb(t.value, t.layer);
        this.label(x(j), y(i), `${t.value}${theme.superscript(t.layer)}`, theme.textOn(rgb), t.piece === last);
      }
    });

    if (mark) outlines = [...outlines, { ...mark, color: theme.HINT }];
    for (const o of outlines) {
      const cs = cells(o.value, o.move).filter(([i, j]) => inView(i, j));
      outline(ctx, cs, x, y, s, o.color, Math.max(2, s / 8));
    }

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

/// A shadow falling from the top edge of a cell ("down") or its left edge
/// ("right"), `depth` pixels long.
function shade(ctx, px, py, s, depth, direction) {
  const g =
    direction === "down"
      ? ctx.createLinearGradient(px, py, px, py + depth)
      : ctx.createLinearGradient(px, py, px + depth, py);
  g.addColorStop(0, "rgba(0,0,0,0.55)");
  g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  if (direction === "down") ctx.fillRect(px, py, s, depth);
  else ctx.fillRect(px, py, depth, s);
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
