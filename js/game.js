// Game state for interactive play, as in the terminal game (rust/cli
// game.rs): the deal, the tiles placed, and the "ghost" tile the player
// moves around. Like a physical tile, the ghost comes to rest on whatever is
// underneath, which is always the only level it could legally go, so the
// level never needs choosing. The engine checks legality.

export const DEPTH = 10;

/// Set once the engine is loaded: shapes[value][rot] = [[row, col], ...].
export const geometry = { shapes: null, height: 160, width: 120 };

export function cells(value, m) {
  return geometry.shapes[value][m.rot].map(([r, c]) => [m.i + r, m.j + c]);
}

export function size(value, rot) {
  const cs = geometry.shapes[value][rot];
  return {
    h: 1 + Math.max(...cs.map((c) => c[0])),
    w: 1 + Math.max(...cs.map((c) => c[1])),
  };
}

const moveKey = (m) => `${m.layer},${m.i},${m.j},${m.rot}`;
export const sameMove = (a, b) =>
  !!a && !!b && a.layer === b.layer && a.i === b.i && a.j === b.j && a.rot === b.rot;

/// The same placement (pieces 0 and 8 look the same after a half turn).
export function samePlacement(value, a, b) {
  if (!a || !b || a.layer !== b.layer || a.i !== b.i || a.j !== b.j) return false;
  const s = geometry.shapes[value];
  return JSON.stringify(s[a.rot]) === JSON.stringify(s[b.rot]);
}

/// The board seen from above: the topmost piece in each cell.
export class TopView {
  constructor(pieces) {
    this.cells = new Map();
    pieces.forEach((p, piece) => {
      for (const [i, j] of cells(p.value, p)) {
        const k = i * geometry.width + j;
        const t = this.cells.get(k);
        if (!t || t.layer < p.layer) this.cells.set(k, { value: p.value, layer: p.layer, piece });
      }
    });
  }
  get(i, j) {
    return this.cells.get(i * geometry.width + j);
  }
  height(i, j) {
    const t = this.get(i, j);
    return t ? t.layer + 1 : 0;
  }
}

export const PROBLEMS = {
  overhang: () => "Would hang over a gap",
  onePiece: () => "Must rest on at least two tiles",
  notTouching: (layer) => `Must touch a tile on level ${layer}`,
  tooHigh: () => "The stack can't go any higher",
};

/// Cell where a piece's label is drawn: its first cell in reading order.
export function labelCell(p) {
  const [r, c] = geometry.shapes[p.value][p.rot][0];
  return [p.i + r, p.j + c];
}

/// Per level: [level, tiles, points] for levels in use.
export function levels(pieces) {
  const out = Array.from({ length: DEPTH }, (_, l) => [l, 0, 0]);
  for (const p of pieces) {
    out[p.layer][1] += 1;
    out[p.layer][2] += p.value * p.layer;
  }
  return out.filter((l) => l[1] > 0);
}

export const score = (pieces) => pieces.reduce((s, p) => s + p.value * p.layer, 0);

export class Game {
  /// Start a game with `tiles` in drawing order; the first goes in the
  /// middle straight away (every spot is the same on an empty board).
  constructor(engine, tiles) {
    this.engine = engine;
    this.tiles = tiles;
    const first = engine.centerMove();
    this.pieces = [{ value: tiles[0], ...first }];
    this.cursor = { i: first.i, j: first.j + size(tiles[0], 0).w };
    this.rot = 0;
    this.refresh();
    this.snapToLegal();
  }

  /// A finished game from a recorded play ([{value, layer, i, j, rot}]),
  /// or null if a move breaks the rules.
  static fromPlay(engine, steps) {
    const center = engine.centerMove();
    if (!steps.length || !sameMove(steps[0], center)) return null;
    if (!engine.load(steps)) return null;
    const g = new Game(engine, steps.map((s) => s.value));
    g.pieces = steps.map((s) => ({ ...s }));
    g.refresh();
    return g;
  }

  /// Bring the engine's board (shared by every game on the page) up to
  /// date, with the current tile's legal moves.
  refresh() {
    this.engine.load(this.pieces);
    this.top = new TopView(this.pieces);
    const v = this.current();
    this.legalList = v == null ? [] : this.engine.validMoves(v);
    this.legal = new Set(this.legalList.map(moveKey));
  }

  placed() {
    return this.pieces.length;
  }
  isOver() {
    return this.placed() === this.tiles.length;
  }
  current() {
    return this.tiles[this.placed()];
  }
  score() {
    return score(this.pieces);
  }
  levels() {
    return levels(this.pieces);
  }

  /// How many of each value are still to come after the current tile.
  stillToCome() {
    const counts = Array(10).fill(0);
    for (const v of this.tiles.slice(this.placed() + 1)) counts[v]++;
    return counts;
  }

  /// Where the current tile would land at the cursor, and whether it may:
  /// {value, move, problem} (problem null when legal).
  preview() {
    const v = this.current();
    if (v == null) return null;
    return this.previewAt(v, this.cursor.i, this.cursor.j, this.rot);
  }

  previewAt(value, i, j, rot) {
    const cs = cells(value, { i, j, rot });
    const layer = Math.max(...cs.map(([a, b]) => this.top.height(a, b)));
    const move = { layer: Math.min(layer, DEPTH - 1), i, j, rot };
    let problem = null;
    if (layer >= DEPTH) problem = PROBLEMS.tooHigh();
    else if (!this.legal.has(moveKey(move))) problem = this.problem(cs, layer);
    return { value, move, problem };
  }

  problem(cs, layer) {
    if (cs.some(([a, b]) => this.top.height(a, b) !== layer)) return PROBLEMS.overhang();
    if (layer > 0) {
      const first = this.top.get(...cs[0])?.piece;
      if (cs.every(([a, b]) => this.top.get(a, b)?.piece === first)) return PROBLEMS.onePiece();
    }
    return PROBLEMS.notTouching(layer);
  }

  ghostSize() {
    return size(this.current() ?? 0, this.rot);
  }

  clamp() {
    const { h, w } = this.ghostSize();
    this.cursor.i = Math.max(0, Math.min(this.cursor.i, geometry.height - h));
    this.cursor.j = Math.max(0, Math.min(this.cursor.j, geometry.width - w));
  }

  shift(di, dj) {
    this.cursor.i += di;
    this.cursor.j += dj;
    this.clamp();
  }

  moveTo(i, j) {
    this.cursor = { i, j };
    this.clamp();
  }

  /// Rotate by `turns` clockwise quarter turns, about the tile's centre.
  rotate(turns) {
    const a = this.ghostSize();
    this.rot = (((this.rot + turns) % 4) + 4) % 4;
    const b = this.ghostSize();
    this.cursor.i += Math.trunc((a.h - b.h) / 2);
    this.cursor.j += Math.trunc((a.w - b.w) / 2);
    this.clamp();
  }

  jumpTo(m) {
    this.rot = m.rot;
    this.cursor = { i: m.i, j: m.j };
    this.clamp();
  }

  /// Move the ghost to the legal spot nearest where it is, keeping its
  /// rotation if possible.
  snapToLegal() {
    const { i, j } = this.cursor;
    let best = null;
    let bestKey = null;
    for (const m of this.legalList) {
      const key = [m.rot !== this.rot ? 1 : 0, Math.abs(m.i - i) + Math.abs(m.j - j), -m.layer];
      if (!best || compare(key, bestKey) < 0) {
        best = m;
        bestKey = key;
      }
    }
    if (best) this.jumpTo(best);
  }

  /// Place the current tile where the ghost is: the move, or {problem}.
  place() {
    const p = this.preview();
    if (!p) return { problem: PROBLEMS.tooHigh() };
    if (p.problem) return { problem: p.problem };
    this.pieces.push({ value: p.value, ...p.move });
    this.refresh();
    this.snapToLegal();
    return { move: p.move, value: p.value };
  }

  /// Take back the last tile (the first stays); the ghost goes back to it.
  undo() {
    if (this.placed() <= 1) return false;
    const p = this.pieces.pop();
    this.refresh();
    this.jumpTo(p);
    return true;
  }
}

function compare(a, b) {
  for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return a[k] - b[k];
  return 0;
}

/// Read a recorded play: every `value@level:row:col:rotation` token.
export function parsePlay(text) {
  const steps = [];
  for (const token of text.split(/\s+/)) {
    const m = /^(\d)@(\d+):(\d+):(\d+):([0-3])$/.exec(token);
    if (m) {
      const [value, layer, i, j, rot] = m.slice(1).map(Number);
      steps.push({ value, layer, i, j, rot });
    }
  }
  return steps;
}
