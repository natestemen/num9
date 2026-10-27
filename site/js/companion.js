// Companion: play NMBR 9 at the table against the engine. Turn over a card,
// tell the page its number, and the engine places that tile on its own board
// here, so you both play the same deal. It decides as the hints do (Monte
// Carlo rollouts over the cards still in the deck), never knowing what comes
// next. At the end, enter your score to see who won, and step through the
// engine's play next to the best play the search can find for those cards.

import { Game, TopView } from "./game.js";
import { PlayApp, describe } from "./play.js";
import { extent } from "./board.js";
import { Hint } from "./hint.js";
import * as theme from "./theme.js";
import * as ui from "./ui.js";
import { dealCode, link, said, share, squares } from "./share.js";

const TILES = 20;
const ROLLOUTS = 256;

export class CompanionApp extends PlayApp {
  /// A new game: no cards turned over yet.
  newGame() {
    this.stopWork();
    this.thinking?.stop();
    this.thinking = null;
    this.cards = [];
    this.game = null;
    this.reset();
    this.yourScore = null;
    this.lastMove = null;
    this.hidden ??= false;
    this.render();
  }

  placed() {
    return this.game?.placed() ?? 0;
  }

  done() {
    return this.placed() === TILES;
  }

  /// How many of each value are still in the deck (not yet turned over).
  inDeck() {
    const counts = Array(10).fill(2);
    for (const v of this.cards) counts[v]--;
    return counts;
  }

  /// A card was turned over.
  enter(v) {
    if (this.cards.length === TILES) return;
    if (this.inDeck()[v] === 0) {
      this.say(`Both ${v}s have already been turned over`, "bad");
      return;
    }
    this.cards.push(v);
    this.pump();
  }

  /// Have the engine place the next card turned over, if it isn't busy.
  pump() {
    if (this.thinking || this.cards.length === this.placed()) return;
    const v = this.cards[this.placed()];
    if (!this.game) {
      // The first tile goes in the middle.
      this.game = new Game(this.engine, [v]);
      this.lastMove = { value: v, move: this.game.pieces[0], expected: null };
      this.view.fresh = true;
      this.pump();
      return;
    }
    const g = this.game;
    g.tiles.push(v);
    g.refresh();
    const hint = new Hint(this.pool, g, null, ROLLOUTS, () => {
      if (this.thinking === hint && hint.finished()) this.commit();
      else this.render();
    });
    this.thinking = hint;
    if (hint.finished()) this.commit();
    this.render();
  }

  /// Place the card the engine has been thinking about where it rates best.
  commit() {
    const h = this.thinking;
    const best = h.ranking()[0];
    this.thinking = null;
    this.game.jumpTo(best.move);
    this.game.place();
    this.lastMove = { value: h.value, move: best.move, expected: h.forced() ? null : best.expected };
    this.view.centre(extent(this.game.pieces));
    if (this.done()) this.searchBestPlay();
    this.render();
    this.pump();
  }

  /// Take back the last card entered (a mistyped one, say).
  undo() {
    if (!this.cards.length) {
      this.say("No cards to take back");
      return;
    }
    const v = this.cards.pop();
    if (this.cards.length < this.placed()) {
      if (this.placed() === 1) this.game = null;
      else {
        this.game.undo();
        this.game.tiles.pop();
        this.game.refresh();
        this.view.centre(extent(this.game.pieces));
      }
      this.background.stop();
      this.analysis = null;
      this.review = null;
      this.yourScore = null;
      this.lastMove = null;
    } else if (this.thinking && this.cards.length === this.placed()) {
      // The card the engine was thinking about.
      this.thinking.stop();
      this.thinking = null;
      this.game.tiles.pop();
      this.game.refresh();
    }
    this.say(`Took back the ${v}`);
    this.pump();
  }

  yoursLabel() {
    return "Engine";
  }

  yoursTitle() {
    return "Engine's play";
  }

  onKey(e) {
    const key = e.key;
    const pending = this.confirm;
    this.confirm = null;
    let used = true;
    if (key === "info") this.infoOpen = !this.infoOpen;
    else if (this.review) used = this.onReviewKey(key);
    else if (/^[0-9]$/.test(key)) this.enter(Number(key));
    else {
      switch (key) {
        case "u":
        case "Backspace":
          this.undo();
          break;
        case "h":
          this.hidden = !this.hidden;
          break;
        case "b":
          if (this.done()) this.showReview(this.movesOf("best") ? "best" : "yours", TILES);
          break;
        case "l":
          if (this.done()) this.shareResult();
          break;
        case "Tab":
        case "v":
          this.heightMap = !this.heightMap;
          break;
        case "c":
          this.recentre();
          break;
        case "Escape":
          this.hideSummary = !this.hideSummary;
          break;
        case "?":
          this.opts.onHelp?.();
          break;
        case "n":
          if (!this.cards.length || this.done() || pending === "n") this.newGame();
          else {
            this.confirm = "n";
            this.say("Press again to start a new game");
          }
          break;
        default:
          used = false;
      }
    }
    if (used) this.render();
    return used;
  }

  /// Send someone how the game at the table went, with a link to play the
  /// same cards online.
  async shareResult() {
    const ai = this.game.score();
    const you = this.yourScore;
    const lines = ["NMBR 9 at the table, against the engine"];
    lines.push(you == null ? `The engine scored ${ai}` : `I scored ${you} · the engine ${ai}${you > ai ? " · I won!" : ""}`);
    if (this.analysis?.best) lines.push(`Best found for these cards: ${this.analysis.best.score}`);
    lines.push(squares(this.game.pieces), "Play the same cards:");
    const outcome = await share(lines.join("\n"), link(`play/t/${dealCode(this.cards)}`));
    this.say(...said(outcome, "the result"));
    this.render();
  }

  recentre() {
    if (this.review) this.view.centre(this.review.extent);
    else if (this.game) this.view.centre(extent(this.game.pieces));
  }

  // --------------------------------------------------------------------
  // Drawing

  render() {
    this.view.resize();
    this.drawPicker();
    this.drawBoard();
    this.drawUnspot();
    this.s.panel.innerHTML = `<button class="sheet-close" data-send="info">Close ✕</button>` + (this.review ? this.reviewPanel() : this.panel());
    this.s.panel.classList.toggle("open", !!this.infoOpen);
    this.s.status.innerHTML = this.statusHtml();
    ui.touchBar(this.s.touchbar, this.touchButtons(), (e) => this.onKey(e));
    this.drawKeys();
    this.drawPopup();
  }

  secret() {
    return this.hidden && !this.done();
  }

  drawBoard() {
    if (this.review) return super.drawBoard();
    const pieces = this.game?.pieces ?? [];
    const top = this.game?.top ?? new TopView([]);
    if (this.secret()) {
      this.s.title.textContent = "Engine's board · hidden until the end";
      this.view.draw([], new TopView([]));
      return;
    }
    this.s.title.textContent = `Engine's board${this.heightMap ? " · height map" : ""}`;
    if (this.view.fresh && pieces.length) this.view.centre(extent(pieces));
    const m = this.lastMove;
    this.view.draw(pieces, top, {
      heightMap: this.heightMap,
      spotlight: this.spotlight(),
      mark: m && !this.done() ? { value: m.value, move: m.move } : null,
    });
  }

  /// The card buttons (or, after the game, where to enter your score).
  drawPicker() {
    if (!this.picker) {
      this.picker = document.createElement("div");
      this.picker.className = "cards";
      const stage = this.s.panel.closest(".stage");
      stage.parentElement.insertBefore(this.picker, stage);
      this.picker.addEventListener("click", (e) => {
        const b = e.target.closest("button[data-card]");
        if (b) this.onKey({ key: b.dataset.card, preventDefault() {} });
      });
      // Enter (or Done on a phone keyboard) leaves the score box, so the
      // keys work again.
      this.picker.addEventListener("keydown", (e) => {
        if (e.key === "Enter") e.target.blur();
      });
      this.picker.addEventListener("input", (e) => {
        const n = Number.parseInt(e.target.value, 10);
        this.yourScore = Number.isFinite(n) ? n : null;
        this.render();
      });
    }
    const left = this.inDeck();
    const sig = this.cards.length === TILES ? "score" : left.join();
    if (this.picker.dataset.sig === sig) return;
    this.picker.dataset.sig = sig;
    if (sig === "score") {
      this.picker.innerHTML = `<label class="your-score">Your score at the table <input type="number" inputmode="numeric" enterkeyhint="done" min="0" max="400" placeholder="?" value="${this.yourScore ?? ""}"></label>`;
      // Ready for typing, where that doesn't pop up a keyboard.
      if (matchMedia("(pointer: fine)").matches) this.picker.querySelector("input").focus();
      return;
    }
    this.picker.innerHTML =
      `<span class="cards-label">Card turned over</span>` +
      left
        .map((n, v) => {
          const rgb = theme.tileRgb(v);
          const dots = "●".repeat(n) + "○".repeat(2 - n);
          return `<button data-card="${v}" ${n ? "" : "disabled"} style="background:${theme.css(rgb)};color:${theme.textOn(rgb)}"><b>${v}</b><small>${dots}</small></button>`;
        })
        .join("");
  }

  /// Where the engine put the latest card, for the panel and status line.
  lastLine() {
    if (this.thinking) {
      const h = this.thinking;
      return ui.muted(`Placing the ${h.value}… ${Math.round(h.progress() * 100)}%`);
    }
    const m = this.lastMove;
    if (!m) return ui.muted(this.cards.length ? "" : "Turn over the first card and enter it");
    if (this.secret()) return `${ui.muted("Placed the")} ${ui.bold(m.value, theme.valueCss(m.value))}`;
    return `${ui.bold(m.value, theme.valueCss(m.value))} ${ui.muted("→")} ${ui.esc(describe(m.value, m.move).trim())}`;
  }

  facts() {
    const n = this.cards.length;
    const waiting = n - this.placed() - (this.thinking ? 1 : 0);
    let out = `${ui.muted("cards")} ${ui.bold(`${n}/${TILES}`)}`;
    if (waiting > 0) out += ui.muted(` (${waiting} waiting)`);
    if (this.secret()) return out + `  ${ui.muted("engine's score hidden")}`;
    out += `  ${ui.muted("engine")} ${ui.bold(this.game?.score() ?? 0)}`;
    const e = this.lastMove?.expected;
    if (e != null && !this.done()) out += `  ${ui.muted(`expects ≈ ${e.toFixed(0)}`)}`;
    return out;
  }

  panel() {
    const intro = ui.muted("You play at the table; the engine plays the same cards here.");
    const levels = this.secret() ? ui.muted("hidden until the end") : ui.levelLines(this.game?.levels() ?? []) || "&nbsp;";
    return (
      ui.box("Companion", `<div class="wrap">${intro}</div><div class="gap"></div><div>${this.facts()}</div>`) +
      ui.box("Latest card", `<div>${this.lastLine() || "&nbsp;"}</div>`) +
      ui.box("Still in the deck", ui.bagLines(this.inDeck()) + ui.bagShapes(this.inDeck())) +
      ui.box("Engine's levels", levels, "grow") +
      this.messageLine()
    );
  }

  statusHtml() {
    if (this.review) return super.statusHtml();
    let line2 = this.lastLine();
    if (this.message) {
      const color = { info: "inherit", good: theme.GOOD, bad: theme.BAD }[this.message.tone];
      line2 = ui.span(this.message.text, `color:${color}`);
    }
    if (this.done()) line2 = this.verdict();
    const bag = this.done() ? "" : ui.bagLine(this.inDeck());
    return `<div>${this.facts()}</div>${bag}<div>${line2 || "&nbsp;"}</div>`;
  }

  verdict() {
    const ai = this.game.score();
    const you = this.yourScore;
    if (you == null) return ui.muted(`The engine scored ${ai}. Enter your score above.`);
    if (you > ai) return ui.bold(`You won by ${you - ai}!`, theme.GOOD);
    if (you === ai) return ui.bold("A tie!", theme.ACCENT);
    return ui.bold(`The engine won by ${ai - you}`, theme.BAD);
  }

  touchButtons() {
    const info = ["Info", "info"];
    if (this.review) return super.touchButtons();
    if (this.done()) return [["Review", "b", "primary"], ["Share", "l"], ["New", "n"], ["Board", "Escape"], info];
    return [["Undo", "u"], [this.hidden ? "Show board" : "Hide board", "h"], ["New", "n"], info];
  }

  keys() {
    if (this.review) return super.keys();
    if (this.done()) {
      return [
        ["b", "step through the engine's play", "b"],
        ["l", "share the result", "l"],
        ["n", "new game", "n"],
        ["esc", "show/hide summary", "Escape"],
        ["u", "take back the last card", "u"],
      ];
    }
    return [
      ["0-9", "card turned over", null],
      ["u", "take back a card", "u"],
      ["h", this.hidden ? "show the engine's board" : "hide the engine's board", "h"],
      ["tab", "view", "Tab"],
      ["c", "centre", "c"],
      ["n", "new game", "n"],
      ["?", "help", "?"],
    ];
  }

  drawPopup() {
    const show = this.done() && !this.hideSummary && !this.review;
    this.s.popup.hidden = !show;
    if (show) this.s.popup.innerHTML = this.summary();
  }

  summary() {
    const g = this.game;
    let out = `<div class="box-title">Game over</div>`;
    out += `<div>The engine scored ${ui.bold(g.score(), theme.ACCENT)}</div><div class="gap"></div>`;
    out += ui.levelLines(g.levels()) + `<div class="gap"></div>`;
    out += `<div>${this.verdict()}</div>`;
    const best = this.analysis?.best;
    if (best) {
      const searching = this.analysis.stagesDone < this.analysis.stages;
      out += `<div>Best play found for these cards: ${ui.bold(best.score, theme.HINT)}${searching ? ui.muted("  (still searching…)") : ""}</div>`;
    } else out += `<div>${ui.muted("Searching for the best play of these cards…")}</div>`;
    out += `<div class="gap"></div><div class="actions">${[
      ["b", "step through the engine's play"],
      ["l", "share the result"],
      ["n", "new game"],
      ["esc", "view board", "Escape"],
    ]
      .map(([k, w, send]) => `<button class="key" data-send="${send ?? k}"><b>${k}</b> ${w}</button>`)
      .join("")}</div>`;
    return out;
  }
}
