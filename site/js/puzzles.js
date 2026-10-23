// Puzzles: the game with a small deck, solved exactly (num9_core::exact), so
// every move can be judged against perfect play. Perfect play is what earns
// the highest final score on average, not knowing the order of the tiles to
// come; each move is scored by how much of that average it gives away.

import { Game, samePlacement, sameMove } from "./game.js";
import { PlayApp, describe, placementLine, SUGGESTIONS } from "./play.js";
import { Background } from "./workers.js";
import * as theme from "./theme.js";
import * as ui from "./ui.js";

export const DECKS = [
  { id: "two-0-1", name: "Two 0s and two 1s", counts: [2, 2], table: null },
  { id: "one-0-3", name: "One each of 0 to 3", counts: [1, 1, 1, 1], table: null },
  { id: "one-0-4", name: "One each of 0 to 4", counts: [1, 1, 1, 1, 1], table: "tables/one-0-4.bin" },
  { id: "two-0-2", name: "Two each of 0, 1 and 2", counts: [2, 2, 2], table: "tables/two-0-2.bin" },
];

function shuffled(counts) {
  const tiles = counts.flatMap((n, v) => Array(n).fill(v));
  for (let k = tiles.length - 1; k > 0; k--) {
    const r = Math.floor(Math.random() * (k + 1));
    [tiles[k], tiles[r]] = [tiles[r], tiles[k]];
  }
  return tiles;
}

/// Perfect play's verdict on each placement of the tile in hand, in the
/// shape the game screen's hint box expects.
class ExactHint {
  constructor(exact, yours) {
    this.placed = exact.placed;
    this.value = exact.value;
    this.askedAt = yours;
    this.exact = true;
    this.digits = 2;
    this.label = "Exact expected final score";
    this.ranked = exact.placements;
  }
  stop() {}
  finished() {
    return true;
  }
  progress() {
    return 1;
  }
  forced() {
    return this.ranked.length <= 1;
  }
  ranking() {
    return this.ranked;
  }
  yoursRated() {
    const y = this.ranked.find((r) => samePlacement(this.value, r.move, this.askedAt));
    if (!y) return null;
    const rank = 1 + this.ranked.filter((r) => r.expected > y.expected + 1e-9).length;
    return { rank, ...y };
  }
}

export class PuzzleApp extends PlayApp {
  constructor(section, engine, pool, opts = {}) {
    super(section, engine, pool, { ...opts, puzzle: true });
  }

  /// A new deal of the chosen deck (or `tiles` again).
  async newGame(_seed, tiles = null) {
    this.deck ??= DECKS[0];
    this.solver ??= new Background();
    this.stopWork();
    this.game = new Game(this.engine, tiles ?? shuffled(this.deck.counts));
    this.replayName = null;
    this.reset();
    this.history = [];
    this.exact = null;
    this.render();
    await this.loadDeck();
    this.evaluate();
  }

  async chooseDeck(deck) {
    this.deck = deck;
    this.opts.onDeck?.(deck);
    await this.newGame();
  }

  /// Give the solver the deck and its positions solved in advance.
  async loadDeck() {
    if (this.loaded === this.deck.id) return;
    const deck = this.deck;
    let table = new ArrayBuffer(0);
    if (deck.table) table = await (await fetch(deck.table)).arrayBuffer();
    const counts = Array.from({ length: 10 }, (_, v) => deck.counts[v] ?? 0);
    await this.solver.run({ type: "exactInit", deck: counts, table });
    const root = await this.solver.run({ type: "exact", pieces: [], value: null });
    this.loaded = deck.id;
    this.deckValue = root.before;
  }

  /// Ask the solver about the position on the board.
  async evaluate() {
    const g = this.game;
    const placed = g.placed();
    const value = g.current();
    if (value == null) return;
    const reply = await this.solver.run({ type: "exact", pieces: g.pieces, value });
    if (g !== this.game || g.placed() !== placed) return; // moved on
    reply.placements.sort((a, b) => b.expected - a.expected);
    this.exact = { placed, value, before: reply.before, placements: reply.placements };
    if (placed === 1) this.dealValue = reply.before;
    this.render();
  }

  stopWork() {
    this.hint = null;
  }

  currentHint() {
    return this.hint && this.hint.placed === this.game.placed() ? this.hint : null;
  }

  askHint() {
    if (this.game.isOver()) return;
    if (!this.exact || this.exact.placed !== this.game.placed()) {
      this.say("Still working out perfect play…");
      return;
    }
    this.hint = new ExactHint(this.exact, this.ghostMove());
    this.suggestion = 0;
    this.showHint = true;
  }

  afterPlace(r) {
    const e = this.exact;
    if (e && e.placed === this.game.placed() - 1) {
      const yours = e.placements.find((p) => samePlacement(r.value, p.move, r.move));
      const best = e.placements[0];
      this.history.push({ value: r.value, move: r.move, yours: yours?.expected, best: best.expected, bestMove: best.move });
    }
    this.exact = null;
    this.evaluate();
  }

  gameOver() {}

  undo() {
    if (this.game.undo()) {
      this.history = this.history.slice(0, this.game.placed() - 1);
      this.showHint = false;
      this.exact = null;
      this.evaluate();
      this.say("Took the last tile back");
    } else this.say("Nothing to undo");
  }

  onOverKey(key) {
    if (key === "b") return true; // no best play to review
    if (key === "d") {
      this.cycleDeck();
      return true;
    }
    return super.onOverKey(key);
  }

  onPlayKey(e, pending) {
    if (e.key === "d") {
      this.cycleDeck();
      return true;
    }
    return super.onPlayKey(e, pending);
  }

  cycleDeck() {
    const k = DECKS.indexOf(this.deck);
    this.chooseDeck(DECKS[(k + 1) % DECKS.length]);
  }

  lost() {
    return this.history.reduce((s, h) => s + (h.best - (h.yours ?? h.best)), 0);
  }

  panel() {
    const g = this.game;
    const n = g.tiles.length;
    const turn = Math.min(n, g.placed() + (g.isOver() ? 0 : 1));
    const fmt = (x) => (x == null ? "…" : x.toFixed(2));
    let head = `<div>${ui.bold(this.deck.name, theme.ACCENT)}</div>`;
    head += `<div>${ui.muted(`tile ${turn} of ${n}`)}   Score ${ui.bold(g.score())}</div>`;
    head += `<div>${ui.muted("Perfect play averages")} ${ui.bold(fmt(this.deckValue))}</div>`;
    head += `<div>${ui.muted("Points given away")} ${ui.bold(fmt(this.lost()), this.lost() > 1e-9 ? theme.BAD : theme.GOOD)}</div>`;
    let hand;
    const v = g.current();
    if (v != null) {
      hand =
        `<div class="hand">${ui.tileShape(v, g.rot)}<div><div>${ui.bold(v, theme.valueCss(v))}</div>` +
        `<div>${ui.muted(`↻ ${g.rot * 90}°`)}</div></div></div>${placementLine(g.preview())}`;
    } else hand = ui.muted("All tiles placed");
    const last = this.history.at(-1);
    let verdict = "";
    if (last) {
      const gap = last.best - (last.yours ?? last.best);
      verdict =
        gap < 1e-9
          ? `<div>${ui.muted("Last move:")} ${ui.bold("perfect", theme.GOOD)}</div>`
          : `<div>${ui.muted("Last move:")} ${ui.span(`−${gap.toFixed(2)}`, `color:${theme.BAD};font-weight:700`)} ${ui.muted(`(best: ${describe(last.value, last.bestMove).trim()})`)}</div>`;
    }
    const counts = Array(10).fill(0);
    for (const t of g.tiles.slice(g.placed() + 1)) counts[t]++;
    const bag = this.deck.counts
      .map((c, v) => ui.span(`${v}${Array.from({ length: c }, (_, k) => (k < counts[v] ? "●" : "○")).join("")}`, `color:${counts[v] ? theme.valueCss(v) : theme.MUTED};font-weight:700`))
      .join("  ");
    return (
      ui.box("Puzzle", head) +
      ui.box("Tile in hand", hand + verdict) +
      ui.box("Still to come", `<div>${bag}</div>`) +
      ui.box("Levels", ui.levelLines(g.levels()) || "&nbsp;", "grow") +
      ui.box("Perfect play", this.exact || g.isOver() ? this.hintBody() : ui.muted("working it out…")) +
      this.messageLine()
    );
  }

  hintBody() {
    if (!this.showHint || !this.currentHint()) {
      if (this.game.isOver()) return ui.muted("Puzzle over");
      return `${ui.bold("h", theme.ACCENT)} ${ui.muted("show the best moves")}`;
    }
    return super.hintBody().replace(/1-3/, `1-${SUGGESTIONS}`);
  }

  keys() {
    if (this.game.isOver()) {
      return [
        ["n", "new deal", "n"],
        ["r", "same tiles again", "r"],
        ["d", "next deck", "d"],
        ["esc", "show/hide summary", "Escape"],
        ["u", "undo", "u"],
      ];
    }
    return [
      ["←↑↓→", "move", null],
      ["space", "rotate", " "],
      ["enter", "place", "Enter"],
      ["h", "best moves", "h"],
      ["g", "next", "g"],
      ["u", "undo", "u"],
      ["d", "next deck", "d"],
      ["n", "new deal", "n"],
      ["?", "help", "?"],
    ];
  }

  summary() {
    const g = this.game;
    const s = g.score();
    const lost = this.lost();
    let out = `<div class="box-title">Puzzle over</div>`;
    out += `<div>Final score  ${ui.bold(s, theme.ACCENT)}</div><div class="gap"></div>`;
    out += `<div>Perfect play averages ${ui.bold(this.deckValue?.toFixed(2) ?? "…")} on this deck,</div>`;
    out += `<div>and ${ui.bold(this.dealValue?.toFixed(2) ?? "…")} once the first tile is down.</div>`;
    out +=
      lost < 1e-9
        ? `<div>${ui.bold("Every move was perfect!", theme.GOOD)}</div>`
        : `<div>Your moves gave away ${ui.bold(lost.toFixed(2), theme.BAD)} points on average:</div>`;
    out += `<div class="moves">`;
    this.history.forEach((h, k) => {
      const gap = h.best - (h.yours ?? h.best);
      const mark = gap < 1e-9 ? ui.span("✓", `color:${theme.GOOD}`) : ui.span(`−${gap.toFixed(2)}`, `color:${theme.BAD}`);
      out += `<div>${ui.muted(`${k + 2}.`)} ${ui.bold(h.value, theme.valueCss(h.value))} ${ui.esc(describe(h.value, h.move))}${mark}</div>`;
    });
    out += `</div><div class="gap"></div>`;
    out += `<div>${ui.muted("Your score also depends on the order the tiles came in; the loss above doesn't.")}</div><div class="gap"></div>`;
    out += `<div class="actions">${[
      ["n", "new deal"],
      ["r", "same tiles again"],
      ["d", "next deck"],
      ["esc", "view board", "Escape"],
    ]
      .map(([k, w, send]) => `<button class="key" data-send="${send ?? k}"><b>${k}</b> ${w}</button>`)
      .join("")}</div>`;
    return out;
  }
}
