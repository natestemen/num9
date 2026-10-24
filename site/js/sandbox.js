// The challenges on a computer: every tile of the deck lies on the board from
// the start, to be moved about freely (dragged, or picked up with a click and
// dropped with another). A tile lands on whatever is under it, as a real one
// would, and must lie flat. When the stack looks right, submit it: the engine
// checks that it can be built and scores it against the target.
//
// A finished stack can be built exactly when every tile above the ground
// rests flat on at least two tiles and each level is one connected group:
// lay the ground first, then each level, each tile touching one already
// down. That order is what gets checked, move by move, by the engine.

import { TopView, cells, geometry, levels, parsePlay, size } from "./game.js";
import { BoardView, extent } from "./board.js";
import { loadBests, saveBest } from "./challenges.js";
import * as theme from "./theme.js";
import * as ui from "./ui.js";
import { link, said, share, squares } from "./share.js";

/// Where the shelf the tiles start on begins: to the right of the building
/// space, which is around the middle of the board.
const SHELF = { i: 70, j: 72 };
/// Each tile on the shelf gets a slot this big (the largest tile is 4 × 5).
const SLOT = { h: 6, w: 5 };
const PER_ROW = 5;
/// Where the first tile of a play goes (see Board::center_move).
const CENTRE = { i: 80, j: 60 };

const key = (i, j) => `${i},${j}`;
const copy = (tiles) => tiles.map((t) => ({ ...t }));

/// The deck's tiles laid out on the shelf, a few to a row.
function shelf(counts) {
  const tiles = [];
  for (let v = 0; v < 10; v++) {
    for (let c = 0; c < (counts[v] ?? 0); c++) {
      const k = tiles.length;
      const i = SHELF.i + Math.floor(k / PER_ROW) * SLOT.h;
      const j = SHELF.j + (k % PER_ROW) * SLOT.w;
      tiles.push({ value: v, i, j, rot: 0, layer: 0 });
    }
  }
  return tiles;
}

/// The building space and the shelf, for the view to show at first.
function workspace(tiles) {
  const [, b, , d] = extent(tiles);
  return [SHELF.i - 1, Math.max(b, SHELF.i + 20), CENTRE.j - 12, d];
}

/// Turn a whole stack a quarter turn clockwise (the board's rotations are
/// clockwise quarter turns, so each tile's rotation goes up by one).
function turn(tiles) {
  return tiles.map((t) => {
    const cs = cells(t.value, t).map(([i, j]) => [j, -i]);
    return {
      ...t,
      i: Math.min(...cs.map((c) => c[0])),
      j: Math.min(...cs.map((c) => c[1])),
      rot: (t.rot + 1) % 4,
    };
  });
}

export class SandboxApp {
  constructor(section, engine, pool, opts) {
    this.engine = engine;
    this.opts = opts;
    this.s = ui.stage(section);
    this.view = new BoardView(this.s.canvas);
    this.view.cell = 22;
    this.heightMap = false;
    this.wire();
    this.choose(opts.challenges[0]);
  }

  choose(deck) {
    this.deck = deck;
    this.opts.onDeck?.(deck);
    this.reset();
  }

  /// Follow a link to a challenge.
  route(arg) {
    const deck = this.opts.challenges.find((d) => d.id === arg);
    if (deck && deck !== this.deck) this.choose(deck);
  }

  /// Send someone how you did, with a link to the same challenge.
  async shareResult(score, pieces) {
    const d = this.deck;
    const of = d.proven ? `the most possible is ${d.target}` : `the best known is ${d.target}`;
    const text = [`NMBR 9 challenge · ${d.name}`, `I scored ${score} (${of})`, squares(pieces), "Can you get there?"].join("\n");
    const outcome = await share(text, link(`challenges/${d.id}`));
    this.say(...said(outcome, "your score"));
    this.render();
  }

  next() {
    const list = this.opts.challenges;
    this.choose(list[(list.indexOf(this.deck) + 1) % list.length]);
  }

  /// Every tile back on the shelf.
  reset() {
    this.tiles = shelf(this.deck.counts);
    this.history = [];
    this.held = null;
    this.last = -1;
    this.result = null;
    this.answer = null;
    this.message = null;
    this.hideSummary = false;
    this.fit(workspace(this.tiles));
    this.render();
  }

  /// Zoom so that cells (min_i, max_i, min_j, max_j) fit, and centre them.
  fit(box) {
    const [a, b, c, d] = box;
    this.view.resize();
    const cell = Math.floor(Math.min(this.view.w / (d - c + 4), this.view.h / (b - a + 4)));
    this.view.setCell(Math.min(26, cell));
    this.view.centre(box);
  }

  say(text, tone = "info") {
    this.message = { text, tone };
    clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.message = null;
      this.render();
    }, 4000);
  }

  // --------------------------------------------------------------------
  // Moving tiles

  top() {
    return new TopView(this.tiles);
  }

  /// Where a tile of `value` at (i, j, rot) would rest: its level, and
  /// whether it lies flat (every cell on the same height).
  resting(value, i, j, rot) {
    const top = this.top();
    const hs = cells(value, { i, j, rot }).map(([a, b]) => top.height(a, b));
    return { layer: Math.max(...hs), flat: hs.every((h) => h === hs[0]) };
  }

  /// Whether another tile lies on part of tile `k`.
  covered(k) {
    const t = this.tiles[k];
    const mine = new Set(cells(t.value, t).map(([i, j]) => key(i, j)));
    return this.tiles.some((o) => o.layer === t.layer + 1 && cells(o.value, o).some(([i, j]) => mine.has(key(i, j))));
  }

  /// Pick up tile `k`, held by the cell `at`.
  lift(k, at) {
    if (this.covered(k)) {
      this.say("Something is resting on that tile", "bad");
      return false;
    }
    this.history.push(copy(this.tiles));
    const [t] = this.tiles.splice(k, 1);
    this.held = { value: t.value, rot: t.rot, di: at[0] - t.i, dj: at[1] - t.j, from: t, sticky: false };
    this.cursor = at;
    this.result = null;
    return true;
  }

  /// The held tile where it would go now.
  heldMove() {
    const h = this.held;
    const { h: rows, w } = size(h.value, h.rot);
    const i = Math.max(0, Math.min(geometry.height - rows, this.cursor[0] - h.di));
    const j = Math.max(0, Math.min(geometry.width - w, this.cursor[1] - h.dj));
    const { layer, flat } = this.resting(h.value, i, j, h.rot);
    return { move: { layer, i, j, rot: h.rot }, flat };
  }

  /// Put the held tile down where it is, if it lies flat there.
  drop() {
    const { move, flat } = this.heldMove();
    if (!flat) {
      this.say("It would hang over a gap: tiles must lie flat", "bad");
      return false;
    }
    this.tiles.push({ value: this.held.value, ...move });
    this.last = this.tiles.length - 1;
    this.held = null;
    return true;
  }

  /// Return the held tile to where it came from.
  putBack() {
    if (!this.held) return;
    this.tiles = this.history.pop();
    this.held = null;
  }

  rotate(turns) {
    const h = this.held;
    if (!h) return;
    h.rot = (((h.rot + turns) % 4) + 4) % 4;
    const { h: rows, w } = size(h.value, h.rot);
    h.di = Math.min(h.di, rows - 1);
    h.dj = Math.min(h.dj, w - 1);
  }

  undo() {
    if (this.held) this.putBack();
    else if (this.history.length) {
      this.tiles = this.history.pop();
      this.last = -1;
      this.result = null;
    } else this.say("Nothing to undo");
  }

  // --------------------------------------------------------------------
  // Checking the stack

  /// Which tile is on each cell of each level.
  owners() {
    const by = new Map();
    this.tiles.forEach((t, k) => {
      if (!by.has(t.layer)) by.set(t.layer, new Map());
      for (const [i, j] of cells(t.value, t)) by.get(t.layer).set(key(i, j), k);
    });
    return by;
  }

  /// The tiles on `layer`, in groups that touch.
  groups(layer, owners = this.owners()) {
    const at = owners.get(layer) ?? new Map();
    const ks = this.tiles.map((t, k) => k).filter((k) => this.tiles[k].layer === layer);
    const seen = new Set();
    const out = [];
    for (const start of ks) {
      if (seen.has(start)) continue;
      const group = [];
      const queue = [start];
      seen.add(start);
      while (queue.length) {
        const k = queue.shift();
        group.push(k);
        const t = this.tiles[k];
        for (const [i, j] of cells(t.value, t)) {
          for (const [a, b] of [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]) {
            const o = at.get(key(a, b));
            if (o != null && !seen.has(o)) {
              seen.add(o);
              queue.push(o);
            }
          }
        }
      }
      out.push(group);
    }
    return out;
  }

  /// What stops the stack from being built: [{text, tiles}].
  problems() {
    const owners = this.owners();
    const out = [];
    this.tiles.forEach((t, k) => {
      if (t.layer === 0) return;
      const under = cells(t.value, t).map(([i, j]) => owners.get(t.layer - 1)?.get(key(i, j)));
      if (under.some((o) => o == null)) out.push({ text: `The ${t.value} on level ${t.layer} hangs over a gap`, tiles: [k] });
      else if (new Set(under).size < 2) out.push({ text: `The ${t.value} on level ${t.layer} rests on a single tile`, tiles: [k] });
    });
    const top = Math.max(0, ...this.tiles.map((t) => t.layer));
    for (let l = 0; l <= top; l++) {
      const gs = this.groups(l, owners);
      if (gs.length > 1) {
        gs.sort((a, b) => b.length - a.length);
        const loose = gs.slice(1).flat();
        if (l === 0) {
          // Tiles still on the shelf, or set apart: not an error, just unfinished.
          const n = loose.length;
          out.push({ text: `${n} tile${n === 1 ? " isn't" : "s aren't"} part of the stack yet: every tile must join it`, tiles: [], waiting: true });
        } else out.push({ text: `Level ${l} is in ${gs.length} separate groups: each level must be one`, tiles: loose });
      }
    }
    return out;
  }

  /// The stack as a play the engine can check: level by level, each tile
  /// touching one already placed, turned and moved so that the first tile is
  /// unrotated in the middle (where every recorded play starts).
  play() {
    const owners = this.owners();
    const top = Math.max(0, ...this.tiles.map((t) => t.layer));
    let order = [];
    for (let l = 0; l <= top; l++) {
      const gs = this.groups(l, owners);
      if (gs.length !== 1) return null;
      order = order.concat(gs[0].map((k) => this.tiles[k]));
    }
    let tiles = copy(order);
    for (let k = 0; k < (4 - tiles[0].rot) % 4; k++) tiles = turn(tiles);
    const di = CENTRE.i - tiles[0].i;
    const dj = CENTRE.j - tiles[0].j;
    return tiles.map((t) => ({ ...t, i: t.i + di, j: t.j + dj }));
  }

  submit() {
    if (this.held) {
      this.say("Put the tile down first");
      return;
    }
    const problems = this.problems();
    if (problems.length) {
      this.say(problems[0].text, "bad");
      return;
    }
    const steps = this.play();
    if (!steps || !this.engine.load(steps)) {
      this.say("The engine couldn't build this stack", "bad");
      return;
    }
    const score = steps.reduce((s, t) => s + t.value * t.layer, 0);
    this.result = { score, steps };
    this.hideSummary = false;
    saveBest(this.deck.id, score);
    this.opts.onDeck?.(this.deck);
  }

  score() {
    return this.tiles.reduce((s, t) => s + t.value * t.layer, 0);
  }

  playText() {
    const moves = this.result.steps.map((t) => `${t.value}@${t.layer}:${t.i}:${t.j}:${t.rot}`).join(" ");
    return `score ${this.result.score}\nmoves (value@layer:row:col:rotation) ${moves}\n`;
  }

  // --------------------------------------------------------------------
  // Input

  onKey(e) {
    const k = e.key;
    const pending = this.confirm;
    this.confirm = null;
    let used = true;
    if (this.answer) {
      if (["Escape", "s", "Enter"].includes(k)) {
        this.answer = null;
        this.view.centre(extent(this.tiles));
      } else used = false;
    } else
      switch (k) {
        case " ":
        case "r":
          this.rotate(1);
          break;
        case "R":
          this.rotate(-1);
          break;
        case "Escape":
          if (this.held) this.putBack();
          else if (this.result) this.hideSummary = !this.hideSummary;
          else used = false;
          break;
        case "Enter":
          this.submit();
          break;
        case "u":
        case "Backspace":
          this.undo();
          break;
        case "n":
          if (pending === "n" || this.history.length === 0) this.reset();
          else {
            this.confirm = "n";
            this.say("Press again to put every tile back on the shelf");
          }
          break;
        case "s":
          if (this.result || pending === "s") {
            this.putBack();
            this.answer = parsePlay(this.deck.play);
            this.view.centre(extent(this.answer));
          } else {
            this.confirm = "s";
            this.say("Press again to see a best play (a spoiler!)");
          }
          break;
        case "d":
          this.next();
          break;
        case "l":
          if (this.result) this.shareResult(this.result.score, this.result.steps);
          break;
        case "c":
          this.view.centre(extent(this.tiles));
          break;
        case "Tab":
        case "v":
          this.heightMap = !this.heightMap;
          break;
        case "?":
          this.opts.onHelp?.();
          break;
        case "copy":
          navigator.clipboard?.writeText(this.playText()).then(
            () => this.say("Copied your play", "good"),
            () => this.say("Couldn't copy", "bad"),
          );
          break;
        default:
          used = false;
      }
    if (used) this.render();
    return used;
  }

  wire() {
    const c = this.s.canvas;
    let press = null;
    c.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      c.setPointerCapture(e.pointerId);
      const at = this.view.cellAt(e.clientX, e.clientY);
      press = { x: e.clientX, y: e.clientY, top: this.view.vp.top, left: this.view.vp.left, moved: false, mode: "pan" };
      if (this.answer || !at) return;
      if (this.held) press.mode = "drop";
      else {
        const t = this.top().get(...at);
        if (t && this.lift(t.piece, at)) press.mode = "carry";
      }
      this.render();
    });
    c.addEventListener("pointermove", (e) => {
      const at = this.view.cellAt(e.clientX, e.clientY);
      if (press && Math.hypot(e.clientX - press.x, e.clientY - press.y) > 4) press.moved = true;
      if (press?.mode === "pan" && press.moved) {
        const s = this.view.cell;
        this.view.panTo(press.top - Math.round((e.clientY - press.y) / s), press.left - Math.round((e.clientX - press.x) / s));
        this.render();
        return;
      }
      if (this.held && at && (at[0] !== this.cursor[0] || at[1] !== this.cursor[1])) {
        this.cursor = at;
        this.render();
      } else if (!this.held && !press) {
        const t = at && !this.answer ? this.top().get(...at) : null;
        c.style.cursor = t ? (this.covered(t.piece) ? "not-allowed" : "grab") : "default";
      }
    });
    c.addEventListener("pointerup", () => {
      const p = press;
      press = null;
      if (!p) return;
      if (p.mode === "carry") {
        // A click picks the tile up to carry; a drag puts it down here.
        if (!p.moved) this.held.sticky = true;
        else if (!this.drop()) this.putBack();
      } else if (p.mode === "drop") this.drop();
      this.render();
    });
    c.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      this.rotate(1);
      this.render();
    });
    c.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        if (this.held) this.rotate(e.deltaY > 0 ? 1 : -1);
        else this.view.zoom(e.deltaY > 0 ? -2 : 2);
        this.render();
      },
      { passive: false },
    );
    this.s.zoom.addEventListener("click", (e) => {
      const z = e.target.closest("button[data-zoom]");
      if (z) this.view.zoom(Number(z.dataset.zoom));
      if (e.target.closest("button[data-centre]")) this.view.centre(extent(this.answer ?? this.tiles));
      if (e.target.closest("button[data-map]")) this.heightMap = !this.heightMap;
      this.render();
    });
    for (const el of [this.s.panel, this.s.popup]) {
      el.addEventListener("click", (e) => {
        const b = e.target.closest("[data-send]");
        if (b) this.onKey({ key: b.dataset.send, preventDefault() {} });
      });
    }
    new ResizeObserver(() => this.render()).observe(this.s.canvas.parentElement);
  }

  // --------------------------------------------------------------------
  // Drawing

  render() {
    if (!this.tiles) return;
    this.view.resize();
    const problems = this.answer ? [] : this.problems();
    const pieces = this.answer ?? this.tiles;
    let ghost = null;
    if (this.held) {
      const { move, flat } = this.heldMove();
      ghost = { value: this.held.value, move, problem: flat ? null : "flat" };
    }
    const bad = new Set(problems.flatMap((p) => p.tiles));
    this.s.title.textContent = this.answer
      ? `A ${this.deck.target}-point play${this.heightMap ? " · height map" : ""}`
      : `Board${this.heightMap ? " · height map" : ""}`;
    this.view.draw(pieces, new TopView(pieces), {
      heightMap: this.heightMap,
      ghost,
      last: this.answer ? -1 : this.last,
      outlines: [...bad].map((k) => ({ value: this.tiles[k].value, move: this.tiles[k], color: theme.BAD })),
    });
    this.s.panel.innerHTML = this.panel(problems);
    this.s.keys.innerHTML = "";
    ui.keyBar(this.s.keys, this.keys(), (e) => this.onKey(e));
    const show = this.result && !this.hideSummary && !this.answer;
    this.s.popup.hidden = !show;
    if (show) this.s.popup.innerHTML = this.summary();
  }

  targetLine() {
    const d = this.deck;
    return d.proven ? `${ui.muted("the most possible:")} ${ui.bold(d.target)}` : `${ui.muted("best known:")} ${ui.bold(d.target)}`;
  }

  panel(problems) {
    const d = this.deck;
    let head = `<div>${ui.bold(d.name, theme.ACCENT)}</div><div>${this.targetLine()}</div>`;
    const best = loadBests()[d.id];
    if (best != null) head += `<div>${ui.muted("your best here:")} ${ui.bold(best, best >= d.target ? theme.GOOD : "")}</div>`;
    if (this.answer) {
      return (
        ui.box("Challenge", head) +
        ui.box("A best play", `<div class="wrap">${ui.esc(`One way to score ${d.target}.`)}</div>${ui.levelLines(levels(this.answer))}`, "grow") +
        `<button class="submit" data-send="Escape">Back to your stack</button>`
      );
    }
    let stack = `<div>Score ${ui.bold(this.score())}</div>${ui.levelLines(levels(this.tiles))}<div class="gap"></div>`;
    if (this.held) stack += `<div class="wrap">${ui.muted("Holding a ")}${ui.bold(this.held.value, theme.valueCss(this.held.value))}${ui.muted(": click to put it down, scroll or space to turn it, esc to put it back.")}</div>`;
    else if (problems.length) {
      stack += problems
        .slice(0, 3)
        .map((p) => `<div class="wrap">${p.waiting ? ui.muted(`… ${p.text}`) : ui.span(`✗ ${p.text}`, `color:${theme.BAD}`)}</div>`)
        .join("");
    } else stack += `<div class="wrap">${ui.span("✓ A legal stack: submit it to score it", `color:${theme.GOOD}`)}</div>`;
    const how =
      "Drag a tile to move it, or click it to pick it up and click again to put it down. It lands on whatever is under it and must lie flat. Scroll or space turns the tile you hold; drag the empty board to look around.";
    const msg = this.message
      ? `<div class="message" style="color:${{ info: "inherit", good: theme.GOOD, bad: theme.BAD }[this.message.tone]}">${ui.esc(this.message.text)}</div>`
      : "";
    return (
      ui.box("Challenge", head) +
      ui.box("Your stack", stack, "grow") +
      `<button class="submit" data-send="Enter" ${problems.length || this.held ? "disabled" : ""}>Submit</button>` +
      ui.box("How to play", `<div class="wrap">${ui.muted(how)}</div>`) +
      msg
    );
  }

  keys() {
    if (this.answer) return [["esc", "back to your stack", "Escape"]];
    return [
      ["drag", "move a tile", null],
      ["scroll / space", "turn it", " "],
      ["esc", "put it back", "Escape"],
      ["enter", "submit", "Enter"],
      ["u", "undo", "u"],
      ["n", "start over", "n"],
      ["s", "a best play", "s"],
      ["d", "next challenge", "d"],
      ["?", "help", "?"],
    ];
  }

  summary() {
    const d = this.deck;
    const s = this.result.score;
    const t = d.target;
    let verdict;
    if (s > t) verdict = ui.bold(`${s}: you beat the best known!`, theme.GOOD);
    else if (s === t) verdict = ui.bold(d.proven ? "Perfect: the most possible!" : "You matched the best known!", theme.GOOD);
    else verdict = `${ui.muted(`${Math.round((100 * s) / t)}% of the ${d.proven ? "maximum" : "best known"}`)}`;
    let out = `<div class="box-title">${ui.esc(d.name)}</div>`;
    out += `<div>Your score  ${ui.bold(s, theme.ACCENT)}  ${ui.muted(`(${d.proven ? "the most possible" : "best known"} is ${t})`)}</div><div class="gap"></div>`;
    out += ui.levelLines(levels(this.result.steps)) + `<div class="gap"></div><div>${verdict}</div>`;
    if (s > t) out += `<div>${ui.muted("That would be a new record: copy your play and share it.")}</div>`;
    out += `<div class="gap"></div><div class="actions">${[
      ["l", "share your score"],
      ...(s > t ? [["copy", "copy your play"]] : []),
      ["s", "see a best play"],
      ["esc", "keep building", "Escape"],
      ["n", "start over"],
      ["d", "next challenge"],
    ]
      .map(([k, w, send]) => `<button class="key" data-send="${send ?? k}"><b>${k}</b> ${w}</button>`)
      .join("")}</div>`;
    return out;
  }
}
