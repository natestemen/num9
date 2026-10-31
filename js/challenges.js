// Challenges: every tile of a deck is yours from the start, to place in any
// order you like, and the goal is the highest score the deck allows. That
// maximum is proven: an exhaustive search (num9_core::freeorder) tried
// every stack, and gives a play reaching it to step through once you've had
// a go.

import { Game, TopView, parsePlay } from "./game.js";
import { PlayApp, placementLine } from "./play.js";
import { extent } from "./board.js";
import * as theme from "./theme.js";
import * as ui from "./ui.js";
import { link, said, share, squares } from "./share.js";

const STORE = "num9-challenges";

/// Each challenge's best score so far, kept in this browser only.
export function loadBests() {
  try {
    return JSON.parse(localStorage.getItem(STORE)) ?? {};
  } catch {
    return {};
  }
}

export function saveBest(id, s) {
  try {
    const bests = loadBests();
    if (!(bests[id] >= s)) {
      bests[id] = s;
      localStorage.setItem(STORE, JSON.stringify(bests));
    }
  } catch {
    // Storage unavailable (private windows): nothing to remember.
  }
}

export class ChallengeApp extends PlayApp {
  /// `opts.challenges`: [{id, name, counts, target, play}].
  constructor(section, engine, pool, opts) {
    super(section, engine, pool, opts);
  }

  get challenges() {
    return this.opts.challenges;
  }

  /// A fresh attempt at the current challenge.
  newGame() {
    this.deck ??= this.challenges[0];
    this.stopWork();
    this.game = null;
    this.holding = null;
    this.solution = null;
    this.reset();
    this.render();
  }

  choose(deck) {
    this.deck = deck;
    this.opts.onDeck?.(deck);
    this.newGame();
  }

  /// Follow a link to a challenge.
  route(arg) {
    const deck = this.challenges.find((d) => d.id === arg);
    if (deck && deck !== this.deck) this.choose(deck);
  }

  /// Send someone how you did, with a link to the same challenge.
  async shareResult(score, pieces) {
    const d = this.deck;
    const text = [`NMBR 9 challenge · ${d.name}`, `I scored ${score} (the most possible is ${d.target})`, squares(pieces), "Can you get there?"].join("\n");
    const outcome = await share(text, link(`challenges/${d.id}`));
    this.say(...said(outcome, "your score"));
    this.render();
  }

  next() {
    const k = this.challenges.indexOf(this.deck);
    this.choose(this.challenges[(k + 1) % this.challenges.length]);
  }

  total() {
    return this.deck.counts.reduce((a, b) => a + b, 0);
  }

  placed() {
    return this.game?.placed() ?? 0;
  }

  done() {
    return this.placed() === this.total();
  }

  /// How many of each value are still to place (the one in hand included).
  left() {
    const counts = Array.from({ length: 10 }, (_, v) => this.deck.counts[v] ?? 0);
    for (const p of this.game?.pieces ?? []) counts[p.value]--;
    return counts;
  }

  /// Take a tile of `v` in hand (the first one goes straight in the middle).
  pick(v) {
    if (this.done()) return;
    if (!(this.left()[v] > 0)) {
      this.say(this.deck.counts[v] ? `No ${v}s left` : `This deck has no ${v}s`, "bad");
      return;
    }
    if (!this.game) {
      this.game = new Game(this.engine, [v]);
      this.view.fresh = true;
      this.afterPlace();
      return;
    }
    const g = this.game;
    if (this.holding != null) g.tiles[g.placed()] = v;
    else g.tiles.push(v);
    this.holding = v;
    g.refresh();
    g.snapToLegal();
  }

  afterPlace() {
    this.holding = null;
    if (this.done()) {
      saveBest(this.deck.id, this.game.score());
      this.opts.onDeck?.(this.deck);
      this.view.centre(extent(this.game.pieces));
    }
  }

  gameOver() {}

  /// Take back the last tile placed: into your hand if it's empty (where it
  /// was), otherwise back to the tray.
  undo() {
    const g = this.game;
    if (!g) {
      this.say("Nothing to undo");
      return;
    }
    const p = g.pieces.pop();
    if (!g.pieces.length) {
      this.game = null;
      this.holding = null;
      this.say(`Took back the ${p.value}`);
      return;
    }
    if (this.holding == null) this.holding = p.value;
    g.tiles = [...g.pieces.map((q) => q.value), this.holding];
    g.refresh();
    if (this.holding === p.value) g.jumpTo(p);
    else g.snapToLegal();
    this.say(`Took back the ${p.value}`);
  }

  /// Step through one of the best plays known for this deck.
  showSolution() {
    const steps = parsePlay(this.deck.play);
    const g = Game.fromPlay(this.engine, steps);
    if (!g) return;
    this.solution = { game: this.game, holding: this.holding };
    this.game = g;
    this.replayName = `A ${this.deck.target}-point play`;
    this.showReview("yours", 1);
  }

  hideSolution() {
    const s = this.solution;
    this.solution = null;
    this.review = null;
    this.replayName = null;
    this.game = s.game;
    this.holding = s.holding;
    this.game?.refresh();
    if (this.game) this.view.centre(extent(this.game.pieces));
  }

  onKey(e) {
    const key = e.key;
    const pending = this.confirm;
    this.confirm = null;
    let used = true;
    if (key === "info") this.infoOpen = !this.infoOpen;
    else if (this.solution && (key === "Escape" || key === "s")) this.hideSolution();
    else if (this.review) used = this.onReviewKey(key);
    else if (/^[0-9]$/.test(key)) this.pick(Number(key));
    else {
      switch (key) {
        case "u":
        case "Backspace":
          this.undo();
          break;
        case "s":
          if (this.done() || pending === "s") this.showSolution();
          else {
            this.confirm = "s";
            this.say("Press again to see a best play (a spoiler!)");
          }
          break;
        case "d":
          this.next();
          break;
        case "l":
          if (this.done()) this.shareResult(this.game.score(), this.game.pieces);
          break;
        case "n":
          if (this.placed() <= 1 || this.done() || pending === "n") this.newGame();
          else {
            this.confirm = "n";
            this.say("Press again to start over");
          }
          break;
        case "Escape":
          if (this.done()) this.hideSummary = !this.hideSummary;
          else used = false;
          break;
        case "Tab":
        case "v":
          this.heightMap = !this.heightMap;
          break;
        case "c":
          this.recentre();
          break;
        case "?":
          this.opts.onHelp?.();
          break;
        case "h":
        case "g":
          this.say("No hints here: that's the challenge!");
          break;
        default:
          used = this.holding != null && this.onPlayKey(e, null);
      }
    }
    if (used) this.render();
    return used;
  }

  recentre() {
    if (this.review) this.view.centre(this.review.extent);
    else if (this.game) this.view.fresh = true;
    this.mouseAim = false;
  }

  // --------------------------------------------------------------------
  // Drawing

  render() {
    this.view.resize();
    this.drawTray();
    this.drawBoard();
    this.drawUnspot();
    const body = this.review ? this.reviewPanel() : this.panel();
    this.s.panel.innerHTML = `<button class="sheet-close" data-send="info">Close ✕</button>` + body;
    this.s.panel.classList.toggle("open", !!this.infoOpen);
    this.s.status.innerHTML = this.statusHtml();
    ui.touchBar(this.s.touchbar, this.touchButtons(), (e) => this.onKey(e));
    this.drawKeys();
    this.drawPopup();
  }

  drawBoard() {
    if (this.review || this.game) return super.drawBoard();
    this.s.title.textContent = "Board";
    this.view.draw([], new TopView([]));
  }

  /// The tiles still to place, to pick from.
  drawTray() {
    if (!this.tray) {
      this.tray = document.createElement("div");
      this.tray.className = "cards tray";
      const stage = this.s.panel.closest(".stage");
      stage.parentElement.insertBefore(this.tray, stage);
      this.tray.addEventListener("click", (e) => {
        const b = e.target.closest("button[data-card]");
        if (b) this.onKey({ key: b.dataset.card, preventDefault() {} });
      });
    }
    const left = this.left();
    const sig = `${this.deck.id}|${left.join()}|${this.holding}|${!!this.solution}`;
    if (this.tray.dataset.sig === sig) return;
    this.tray.dataset.sig = sig;
    this.tray.innerHTML =
      `<span class="cards-label">Your tiles</span>` +
      this.deck.counts
        .map((c, v) => {
          if (!c) return "";
          const n = left[v];
          const rgb = theme.tileRgb(v);
          const held = this.holding === v ? " held" : "";
          const dots = "●".repeat(Math.max(0, n)) + "○".repeat(Math.max(0, c - n));
          // Phones show the number and dots; wider screens the shapes, one
          // per copy (those already placed drawn faint).
          const shapes = Array.from({ length: c }, (_, k) => ui.tileShape(v, 0, k < n ? "" : "used")).join("");
          return `<button class="${held}" data-card="${v}" ${n > 0 && !this.solution ? "" : "disabled"} style="--c:${theme.css(rgb)};--t:${theme.textOn(rgb)}"><b>${v}</b><small>${dots}</small><span class="shapes">${shapes}</span></button>`;
        })
        .join("");
  }

  targetLine() {
    const d = this.deck;
    return `${ui.muted("the most possible:")} ${ui.bold(d.target)}`;
  }

  panel() {
    const d = this.deck;
    const s = this.game?.score() ?? 0;
    let head = `<div>${ui.bold(d.name, theme.ACCENT)}</div>`;
    head += `<div>Score ${ui.bold(s)}   ${this.targetLine()}</div>`;
    head += `<div>${ui.muted(`tiles placed ${this.placed()} of ${this.total()}`)}</div>`;
    const best = loadBests()[d.id];
    if (best != null) head += `<div>${ui.muted("your best here:")} ${ui.bold(best, best >= d.target ? theme.GOOD : "")}</div>`;
    let hand;
    if (this.holding != null) {
      const g = this.game;
      const v = this.holding;
      hand =
        `<div class="hand">${ui.tileShape(v, g.rot)}<div><div>${ui.bold(v, theme.valueCss(v))}</div>` +
        `<div>${ui.muted(`↻ ${g.rot * 90}°`)}</div></div></div>${placementLine(g.preview())}`;
    } else if (this.done()) hand = ui.muted("All tiles placed");
    else hand = ui.muted(this.game ? "Pick the next tile above (or press its number)" : "Pick a first tile: it goes in the middle");
    const levels = ui.levelLines(this.game?.levels() ?? []) || "&nbsp;";
    return (
      ui.box("Challenge", head) +
      ui.box("Tile in hand", hand) +
      ui.box("Levels", levels, "grow") +
      this.messageLine()
    );
  }

  statusHtml() {
    if (this.review) return super.statusHtml();
    const s = this.game?.score() ?? 0;
    let line1;
    if (this.holding != null) {
      const p = this.game.preview();
      const v = this.holding;
      const where = p.problem
        ? ui.span(`✗ ${p.problem}`, `color:${theme.BAD}`)
        : ui.span(`✓ level ${p.move.layer}${v * p.move.layer ? `  +${v * p.move.layer}` : ""}`, `color:${theme.GOOD}`);
      line1 = `<span class="in-hand">${ui.tileShape(v, this.game.rot, "mini")}${ui.bold(v, theme.valueCss(v))}</span> ${where}`;
    } else if (this.done()) line1 = this.verdict();
    else line1 = ui.muted(this.game ? "Pick the next tile above" : "Pick a first tile above");
    const facts = `Score ${ui.bold(s)}  ${this.targetLine()}  ${ui.muted(`${this.placed()}/${this.total()}`)}`;
    const msg = this.message ? ui.span(this.message.text, `color:${{ info: "inherit", good: theme.GOOD, bad: theme.BAD }[this.message.tone]}`) : "&nbsp;";
    return `<div>${line1}</div><div>${facts}</div><div>${msg}</div>`;
  }

  verdict() {
    const s = this.game.score();
    const t = this.deck.target;
    if (s >= t) return ui.bold(`${s}: perfect, the most possible!`, theme.GOOD);
    return `${ui.bold(s)} ${ui.muted(`of ${t}: ${Math.round((100 * s) / t)}% of the maximum`)}`;
  }

  touchButtons() {
    const info = ["Info", "info"];
    if (this.review) return [["◀", "ArrowLeft"], ["▶", "ArrowRight", "primary"], ["⏮", "Home"], ["⏭", "End"], ["Back", "Escape"], info];
    if (this.done()) return [["Share", "l", "primary"], ["Best play", "s"], ["Again", "n"], ["Next", "d"], ["Board", "Escape"], info];
    if (this.holding != null) return [["↺", "R"], ["↻", " "], ["Place", "Enter", "primary"], ["Undo", "u"], ["Restart", "n"], info];
    return [["Undo", "u"], ["Restart", "n"], ["Answer", "s"], ["Next", "d"], info];
  }

  keys() {
    if (this.review) return [["←→", "step", "ArrowRight"], ["home/end", "first/last", "Home"], ["esc", "back", "Escape"]];
    if (this.done()) {
      return [
        ["l", "share your score", "l"],
        ["s", "see a best play", "s"],
        ["n", "try again", "n"],
        ["d", "next challenge", "d"],
        ["esc", "show/hide summary", "Escape"],
        ["u", "undo", "u"],
      ];
    }
    return [
      ["0-9", "pick a tile", null],
      ["←↑↓→", "move", null],
      ["space", "rotate", " "],
      ["enter", "place", "Enter"],
      ["u", "undo", "u"],
      ["n", "start over", "n"],
      ["s", "answer", "s"],
      ["d", "next challenge", "d"],
      ["?", "help", "?"],
    ];
  }

  drawPopup() {
    const show = this.done() && !this.hideSummary && !this.review;
    this.s.popup.hidden = !show;
    if (show) this.s.popup.innerHTML = this.summary();
  }

  summary() {
    const d = this.deck;
    const s = this.game.score();
    let out = `<div class="box-title">${ui.esc(d.name)}</div>`;
    out += `<div>Your score  ${ui.bold(s, theme.ACCENT)}  ${ui.muted(`(the most possible is ${d.target})`)}</div><div class="gap"></div>`;
    out += ui.levelLines(this.game.levels()) + `<div class="gap"></div>`;
    out += `<div>${this.verdict()}</div>`;
    out += `<div class="gap"></div><div class="actions">${[
      ["l", "share your score"],
      ["s", "see a best play"],
      ["n", "try again"],
      ["d", "next challenge"],
      ["esc", "view board", "Escape"],
    ]
      .map(([k, w, send]) => `<button class="key" data-send="${send ?? k}"><b>${k}</b> ${w}</button>`)
      .join("")}</div>`;
    return out;
  }
}

