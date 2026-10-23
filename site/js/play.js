// The game screen, after the terminal game (rust/cli app.rs and ui.rs): move
// the tile in hand with the keys or the mouse, ask the engine for the best
// move, and after the game see how the engine played the same tiles and the
// best play it can find for them, step by step. Also replays recorded plays.

import { Game, TopView, geometry, levels, sameMove, score, cells } from "./game.js";
import { BoardView, extent } from "./board.js";
import { Hint } from "./hint.js";
import { Background } from "./workers.js";
import { Coach, GameAnalysis, plural } from "./coach.js";
import * as theme from "./theme.js";
import * as ui from "./ui.js";

/// How many of the engine's top moves the player can browse.
export const SUGGESTIONS = 3;
/// Search stages for the best play of a finished deal, as (nesting level,
/// candidates per board level): quick passes first, then deeper ones.
const STAGES = [
  [1, 4],
  [2, 4],
  [2, 8],
  [3, 8],
];
const HINT_ROLLOUTS = 256;

const randomSeed = () => (Math.random() * 2 ** 32) >>> 0;

export class PlayApp {
  /// `opts`: `seed` for the deal, `replay` ({name, steps}) to show a
  /// recorded play instead, `onDeal(seed)` when a new deal starts,
  /// `onHelp()` for the help screen.
  constructor(section, engine, pool, opts = {}) {
    this.engine = engine;
    this.pool = pool;
    this.opts = opts;
    this.s = ui.stage(section);
    this.view = new BoardView(this.s.canvas);
    this.heightMap = false;
    this.background = new Background();
    this.wire();
    if (opts.replay) this.loadReplay(opts.replay);
    else this.newGame(opts.seed);
  }

  // --------------------------------------------------------------------
  // Game flow

  newGame(seed = randomSeed(), tiles = null) {
    this.stopWork();
    this.seed = seed;
    this.game = new Game(this.engine, tiles ?? this.engine.deal(seed));
    this.replayName = null;
    this.reset();
    this.opts.onDeal?.(seed);
    this.render();
  }

  reset() {
    this.hint = null;
    this.showHint = false;
    this.suggestion = 0;
    this.rival = null;
    this.analysis = null;
    this.review = null;
    this.hideSummary = false;
    this.message = null;
    this.confirm = null;
    this.view.fresh = true;
    this.stopCoach();
  }

  /// Drop the move-by-move analysis (the game changed).
  stopCoach() {
    this.gameAnalysis?.stop();
    this.gameAnalysis = null;
    this.coach = null;
    this.coachOn = false;
  }

  /// Re-draw at most once a frame (analysis results stream in fast).
  renderSoon() {
    if (this.pendingRender) return;
    this.pendingRender = requestAnimationFrame(() => {
      this.pendingRender = null;
      this.render();
    });
  }

  loadReplay({ name, steps }) {
    this.stopWork();
    const game = Game.fromPlay(this.engine, steps);
    if (!game) throw new Error(`${name} breaks the rules`);
    this.game = game;
    this.reset();
    this.replayName = name;
    this.showReview("yours", 1);
  }

  stopWork() {
    this.hint?.stop();
    this.background.stop();
    this.stopCoach();
  }

  say(text, tone = "info") {
    this.message = { text, tone };
    clearTimeout(this.messageTimer);
    this.messageTimer = setTimeout(() => {
      this.message = null;
      this.render();
    }, 4000);
  }

  /// The hint, if it is about the position on the board now.
  currentHint() {
    return this.hint && this.hint.placed === this.game.placed() ? this.hint : null;
  }

  ghostMove() {
    const p = this.game.preview();
    return p && !p.problem ? p.move : null;
  }

  askHint() {
    if (this.game.isOver()) return;
    const yours = this.ghostMove();
    const h = this.currentHint();
    if (h && sameMove(h.askedAt, yours) && (yours || !h.askedAt)) {
      this.showHint = true;
      return;
    }
    this.hint?.stop();
    this.hint = new Hint(this.pool, this.game, yours, HINT_ROLLOUTS, () => this.render());
    this.suggestion = 0;
    this.showHint = true;
  }

  /// Move the tile to suggestion `k` (0 = best), or with null to the next
  /// suggestion after the one shown.
  jumpToHint(k) {
    const ranking = this.currentHint()?.ranking() ?? [];
    if (!ranking.length) {
      this.say("Press h to ask for the best move first");
      return;
    }
    const shown = Math.min(ranking.length, SUGGESTIONS);
    if (k == null) {
      const onIt = this.showHint && sameMove(this.ghostMove(), ranking[Math.min(this.suggestion, shown - 1)].move);
      k = onIt ? (this.suggestion + 1) % shown : Math.min(this.suggestion, shown - 1);
    } else if (k >= shown) return;
    this.suggestion = k;
    this.game.jumpTo(ranking[k].move);
    this.showHint = true;
  }

  place() {
    if (this.coachOn && this.coach.practice) {
      this.coach.tryMove();
      return;
    }
    const r = this.game.place();
    if (r.problem) {
      this.say(r.problem, "bad");
      return;
    }
    const points = r.value * r.move.layer;
    this.say(r.move.layer === 0 ? `${r.value} placed on the ground` : `${r.value} on level ${r.move.layer}: +${points}`, points > 0 ? "good" : "info");
    this.showHint = false;
    this.afterPlace(r);
    if (this.game.isOver()) this.gameOver();
  }

  afterPlace() {}

  /// The engine plays the same tiles, then searches for their best play.
  async gameOver() {
    const tiles = this.game.tiles;
    if (!this.replayName) {
      // Grade every move, on the worker pool, while the engine plays below.
      this.gameAnalysis = new GameAnalysis(this.pool, this.engine, tiles, this.game.pieces, () => this.renderSoon());
    }
    this.rival = { score: null };
    this.analysis = { best: null, stagesDone: 0, stages: STAGES.length };
    try {
      const played = await this.background.run({ type: "play", tiles, seed: this.seed });
      this.rival = { score: played.score, moves: played.moves };
      this.render();
    } catch {
      return; // stopped (new game or undo)
    }
    await this.searchBestPlay();
  }

  /// Search for the best play of the finished deal, in stages, showing each
  /// improvement.
  async searchBestPlay() {
    this.analysis = { best: null, stagesDone: 0, stages: STAGES.length };
    try {
      await this.background.run({ type: "analyze", tiles: this.game.tiles, stages: STAGES }, (m) => {
        if (!this.analysis) return;
        const better = (b) => !this.analysis.best || b.score > this.analysis.best.score;
        if (m.type === "improved" && better(m)) this.analysis.best = { score: m.score, moves: m.moves };
        if (m.type === "stage") {
          this.analysis.stagesDone = m.stage;
          if (better(m.best)) this.analysis.best = m.best;
        }
        if (this.review?.play === "best") this.showReview("best", this.review.step);
        this.render();
      });
    } catch {
      // Stopped (new game or undo).
    }
  }

  undo() {
    if (this.game.undo()) {
      this.background.stop();
      this.stopCoach();
      this.rival = null;
      this.analysis = null;
      this.review = null;
      this.showHint = false;
      this.say("Took the last tile back");
    } else this.say("Nothing to undo");
  }

  // --------------------------------------------------------------------
  // Reviewing a finished game

  movesOf(play) {
    if (play === "yours") return this.game.pieces;
    const best = this.analysis?.best;
    return best ? best.moves.map((m, k) => ({ value: this.game.tiles[k], ...m })) : null;
  }

  showReview(play, step) {
    const all = this.movesOf(play);
    if (!all) {
      this.say("Still looking for the best play…");
      return;
    }
    const n = this.game.tiles.length;
    step = Math.max(1, Math.min(n, step));
    const pieces = all.slice(0, step);
    // Frame both plays the same way.
    let box;
    for (const p of ["yours", "best"]) {
      const m = this.movesOf(p);
      if (m) box = extent(m, box);
    }
    this.review = { play, step, pieces, top: new TopView(pieces), extent: box };
    this.view.centre(box);
  }

  onReviewKey(key) {
    const { play, step } = this.review;
    const other = play === "yours" ? "best" : "yours";
    switch (key) {
      case "ArrowLeft":
      case "a":
        this.showReview(play, step - 1);
        break;
      case "ArrowRight":
      case "d":
        this.showReview(play, step + 1);
        break;
      case "Home":
        this.showReview(play, 1);
        break;
      case "End":
        this.showReview(play, this.game.tiles.length);
        break;
      case "Tab":
      case "b":
        if (this.movesOf("best")) this.showReview(other, step);
        else this.heightMap = !this.heightMap;
        break;
      case "v":
        this.heightMap = !this.heightMap;
        break;
      case "Escape":
        this.review = null;
        this.hideSummary = false;
        break;
      case "?":
        this.opts.onHelp?.();
        break;
      default:
        return false;
    }
    return true;
  }

  // --------------------------------------------------------------------
  // Input

  /// Handle a key (a KeyboardEvent or a click on the key bar); returns
  /// whether it was used.
  onKey(e) {
    const pending = this.confirm;
    this.confirm = null;
    const key = e.key;
    let used = true;
    if (key === "info") this.infoOpen = !this.infoOpen;
    else if (this.coachOn) used = this.coach.onKey(e);
    else if (this.review) used = this.onReviewKey(key);
    else if (this.game.isOver()) used = this.onOverKey(key);
    else used = this.onPlayKey(e, pending);
    if (used) this.render();
    return used;
  }

  /// Go through the game move by move with the engine's verdicts.
  openCoach() {
    if (!this.gameAnalysis) return;
    this.coach ??= new Coach(this);
    this.coachOn = true;
    this.review = null;
    this.coach.open();
  }

  onOverKey(key) {
    switch (key) {
      case "a":
        this.openCoach();
        break;
      case "b":
        this.showReview(this.movesOf("best") ? "best" : "yours", this.game.tiles.length);
        break;
      case "n":
        this.newGame();
        break;
      case "r":
        this.newGame(this.seed, this.game.tiles);
        break;
      case "u":
      case "Backspace":
        this.undo();
        break;
      case "Escape":
      case " ":
        this.hideSummary = !this.hideSummary;
        break;
      case "?":
        this.opts.onHelp?.();
        break;
      default:
        return false;
    }
    return true;
  }

  onPlayKey(e, pending) {
    const g = this.game;
    const step = e.shiftKey ? 4 : 1;
    if (/^(Arrow|[wasdWASD]$)/.test(e.key)) this.mouseAim = false;
    switch (e.key) {
      case "ArrowLeft":
      case "a":
        g.shift(0, -step);
        break;
      case "ArrowRight":
      case "d":
        g.shift(0, step);
        break;
      case "ArrowUp":
      case "w":
        g.shift(-step, 0);
        break;
      case "ArrowDown":
      case "s":
        g.shift(step, 0);
        break;
      case "A":
        g.shift(0, -4);
        break;
      case "D":
        g.shift(0, 4);
        break;
      case "W":
        g.shift(-4, 0);
        break;
      case "S":
        g.shift(4, 0);
        break;
      case " ":
      case "r":
      case "x":
        g.rotate(1);
        break;
      case "R":
      case "z":
        if (e.ctrlKey || e.metaKey) this.undo();
        else g.rotate(-1);
        break;
      case "Enter":
        this.place();
        break;
      case "u":
      case "Backspace":
        this.undo();
        break;
      case "h":
        this.askHint();
        break;
      case "g":
        this.jumpToHint(null);
        break;
      case "1":
      case "2":
      case "3":
        this.jumpToHint(Number(e.key) - 1);
        break;
      case "Tab":
      case "v":
        this.heightMap = !this.heightMap;
        break;
      case "?":
        this.opts.onHelp?.();
        break;
      case "c":
        this.recentre();
        break;
      case "Escape":
        this.showHint = false;
        break;
      case "n":
        if (g.placed() <= 1 || pending === "n") this.newGame();
        else {
          this.confirm = "n";
          this.say("Press again to start a new game");
        }
        break;
      default:
        return false;
    }
    return true;
  }

  wire() {
    const c = this.s.canvas;
    const aim = (e) => {
      if (!this.game || this.review || this.game.current() == null) return null;
      const cell = this.view.cellAt(e.clientX, e.clientY);
      if (!cell) return null;
      const v = this.game.current();
      const s = geometry.shapes[v][this.game.rot];
      const h = 1 + Math.max(...s.map((x) => x[0]));
      const w = 1 + Math.max(...s.map((x) => x[1]));
      // Keep the whole tile in view: the mouse never scrolls the board.
      const vp = this.view.vp;
      const i = Math.max(vp.top, Math.min(cell[0] - Math.floor(h / 2), vp.top + vp.rows - h));
      const j = Math.max(vp.left, Math.min(cell[1] - Math.floor(w / 2), vp.left + vp.cols - w));
      return [cell, i, j];
    };
    // The mouse aims the tile and clicks to place it; dragging pans the board.
    let press = null;
    c.addEventListener("pointerdown", (e) => {
      if (e.pointerType !== "mouse" || e.button !== 0) return;
      c.setPointerCapture(e.pointerId);
      press = { x: e.clientX, y: e.clientY, top: this.view.vp.top, left: this.view.vp.left, panned: false };
    });
    c.addEventListener("pointermove", (e) => {
      if (e.pointerType !== "mouse") return;
      if (press) {
        const dx = e.clientX - press.x;
        const dy = e.clientY - press.y;
        if (!press.panned && Math.hypot(dx, dy) < 5) return;
        press.panned = true;
        c.style.cursor = "grabbing";
        const s = this.view.cell;
        this.view.panTo(press.top - Math.round(dy / s), press.left - Math.round(dx / s));
        this.mouseAim = true;
        this.render();
        return;
      }
      const a = aim(e);
      if (!a) return;
      this.mouseAim = true;
      const before = `${this.game.cursor.i},${this.game.cursor.j}`;
      this.game.moveTo(a[1], a[2]);
      if (before !== `${this.game.cursor.i},${this.game.cursor.j}`) this.render();
    });
    c.addEventListener("pointerup", (e) => {
      if (e.pointerType !== "mouse" || e.button !== 0) return;
      const panned = press?.panned;
      press = null;
      c.style.cursor = "";
      if (panned || !aim(e)) return;
      this.place();
      this.render();
    });
    this.touch(c, aim);
    c.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      if (!this.review && this.game?.current() != null) {
        this.game.rotate(1);
        this.render();
      }
    });
    c.addEventListener(
      "wheel",
      (e) => {
        if (this.review || this.game?.current() == null) return;
        e.preventDefault();
        this.game.rotate(e.deltaY > 0 ? 1 : -1);
        this.render();
      },
      { passive: false },
    );
    this.s.zoom.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-zoom]");
      if (b) this.view.zoom(Number(b.dataset.zoom));
      if (e.target.closest("button[data-centre]")) this.recentre();
      if (e.target.closest("button[data-map]")) this.heightMap = !this.heightMap;
      if (e.target.closest("button[data-unspot]")) this.pinnedLevel = null;
      this.render();
    });
    // Hovering over (or tapping) a level in the Levels box shows only the
    // tiles on that level; clicking keeps it that way.
    for (const el of [this.s.panel, this.s.popup]) {
      el.addEventListener("mouseover", (e) => {
        const l = e.target.closest("[data-level]")?.dataset.level;
        const level = l == null ? null : Number(l);
        if (level === this.hoverLevel) return;
        this.hoverLevel = level;
        this.drawBoard();
      });
      el.addEventListener("mouseleave", () => {
        this.hoverLevel = null;
        this.drawBoard();
      });
      el.addEventListener("click", (e) => {
        const l = e.target.closest("[data-level]")?.dataset.level;
        if (l == null) return;
        this.pinnedLevel = this.pinnedLevel === Number(l) ? null : Number(l);
        this.infoOpen = false; // on phones, show the board
        this.render();
      });
    }
    this.s.panel.addEventListener("click", (e) => {
      const send = e.target.closest("[data-send]");
      const play = e.target.closest("[data-play]");
      const step = e.target.closest("[data-step]");
      if (step && this.coachOn) {
        this.coach.go(Number(step.dataset.step));
        this.render();
      } else if (send) this.onKey({ key: send.dataset.send, preventDefault() {} });
      else if (play && this.review && this.movesOf(play.dataset.play)) {
        this.showReview(play.dataset.play, this.review.step);
        this.render();
      }
    });
    this.s.popup.addEventListener("click", (e) => {
      const b = e.target.closest("[data-send]");
      if (b) this.onKey({ key: b.dataset.send, preventDefault() {} });
    });
    new ResizeObserver(() => this.render()).observe(this.s.canvas.parentElement);
  }

  /// Touch: tap to move the tile there, tap the tile to place it; drag to
  /// slide it (like a trackpad, so the finger never hides it); pinch to zoom.
  touch(c, aim) {
    const fingers = new Map();
    let drag = null;
    let pinch = null;
    const spread = () => {
      const [a, b] = [...fingers.values()];
      return Math.hypot(a.x - b.x, a.y - b.y);
    };
    c.addEventListener("pointerdown", (e) => {
      if (e.pointerType === "mouse") return;
      c.setPointerCapture(e.pointerId);
      fingers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (fingers.size === 2) {
        pinch = { spread: spread(), cell: this.view.cell };
        drag = null;
      } else if (fingers.size === 1) {
        drag = { x: e.clientX, y: e.clientY, cursor: { ...this.game?.cursor }, moved: false };
      }
    });
    c.addEventListener("pointermove", (e) => {
      if (!fingers.has(e.pointerId)) return;
      fingers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pinch && fingers.size === 2) {
        this.view.setCell(pinch.cell * (spread() / pinch.spread));
        this.render();
        return;
      }
      if (!drag) return;
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      if (this.review) {
        // Swipe left or right to step through the play.
        if (Math.abs(dx) > 36 && Math.abs(dx) > Math.abs(dy)) {
          this.showReview(this.review.play, this.review.step + (dx < 0 ? 1 : -1));
          drag.x = e.clientX;
          drag.moved = true;
          this.render();
        }
        return;
      }
      if (this.game?.current() == null) return;
      if (!drag.moved && Math.hypot(dx, dy) < 8) return;
      drag.moved = true;
      const s = this.view.cell;
      this.game.moveTo(drag.cursor.i + Math.round(dy / s), drag.cursor.j + Math.round(dx / s));
      this.render();
    });
    const lift = (e) => {
      if (!fingers.delete(e.pointerId)) return;
      if (pinch) {
        if (fingers.size === 0) pinch = null;
        return;
      }
      const tap = drag && !drag.moved && e.type === "pointerup";
      drag = null;
      const a = tap && aim(e);
      if (!a) return;
      const p = this.game.preview();
      const onTile = cells(p.value, p.move).some(([i, j]) => i === a[0][0] && j === a[0][1]);
      if (onTile) this.place();
      else this.game.moveTo(a[1], a[2]);
      this.render();
    };
    c.addEventListener("pointerup", lift);
    c.addEventListener("pointercancel", lift);
  }

  /// What the played game is called in reviews.
  yoursLabel() {
    return this.replayName ? "Play" : "You";
  }

  yoursTitle() {
    return this.replayName ?? "Your game";
  }

  /// Bring the stack (and the tile in hand) back into the middle.
  recentre() {
    if (this.review) this.view.centre(this.review.extent);
    else this.view.fresh = true;
    this.mouseAim = false;
  }

  // --------------------------------------------------------------------
  // Drawing

  /// The level whose tiles stand out (the others dimmed), if any.
  spotlight() {
    return this.hoverLevel ?? this.pinnedLevel ?? null;
  }

  /// The button that turns the spotlight off.
  drawUnspot() {
    const b = this.s.zoom.querySelector("[data-unspot]");
    b.hidden = this.pinnedLevel == null;
    b.textContent = `✕ level ${this.pinnedLevel}`;
  }

  render() {
    if (!this.game) return;
    this.view.resize();
    this.drawBoard();
    this.drawUnspot();
    const body = this.coachOn
      ? this.coach.practice
        ? this.coach.practicePanel()
        : this.coach.panel()
      : this.review
        ? this.reviewPanel()
        : this.panel();
    this.s.panel.innerHTML = `<button class="sheet-close" data-send="info">Close ✕</button>` + body;
    this.s.panel.classList.toggle("open", !!this.infoOpen);
    this.s.status.innerHTML = this.statusHtml();
    ui.touchBar(this.s.touchbar, this.touchButtons(), (e) => this.onKey(e));
    this.drawKeys();
    this.drawPopup();
  }

  /// The facts for the status line (shown instead of the panel on phones).
  statusFacts() {
    const g = this.game;
    const n = g.tiles.length;
    const turn = Math.min(n, g.placed() + (g.isOver() ? 0 : 1));
    return `Score ${ui.bold(g.score())}  ${ui.muted(`tile ${turn}/${n}`)}`;
  }

  statusHtml() {
    if (this.coachOn) return this.coach.statusHtml();
    const g = this.game;
    if (this.review) {
      const r = this.review;
      const v = g.tiles[r.step - 1];
      const m = r.pieces[r.step - 1];
      const which = r.play === "best" ? "Best" : this.yoursLabel();
      const pts = v * m.layer;
      return (
        `<div>${ui.muted(`Tile ${r.step}/${g.tiles.length}:`)} ${ui.bold(`a ${v}`, theme.valueCss(v))} ${ui.esc(`level ${m.layer}${pts ? `  +${pts}` : ""}`)}</div>` +
        `<div>${ui.bold(which, theme.ACCENT)} ${ui.muted("score so far")} ${ui.bold(score(r.pieces))}</div>`
      );
    }
    const facts = this.statusFacts();
    let line1 = "";
    const v = g.current();
    if (v != null) {
      const p = g.preview();
      const where = p.problem
        ? ui.span(`✗ ${p.problem}`, `color:${theme.BAD}`)
        : ui.span(`✓ level ${p.move.layer}${p.value * p.move.layer ? `  +${p.value * p.move.layer}` : ""}`, `color:${theme.GOOD}`);
      line1 = `<span class="in-hand">${ui.tileShape(v, g.rot, "mini")}${ui.bold(v, theme.valueCss(v))}</span> ${where}`;
    }
    let line2;
    const h = this.showHint ? this.currentHint() : null;
    const pick = h?.ranking()[this.suggestion];
    if (pick) {
      const digits = h.digits ?? 1;
      const exp = Number.isNaN(pick.expected) ? "" : ` ${h.exact ? "=" : "≈"} ${pick.expected.toFixed(digits)}`;
      line2 = ui.span(`▸ ${this.suggestion + 1}. ${describe(h.value, pick.move).trim()}${exp}`, `color:${theme.HINT};font-weight:700`);
      const yours = h.yoursRated();
      if (yours && sameMove(this.ghostMove(), h.askedAt) && yours.rank > 1) {
        line2 += ui.muted(`  yours −${(h.ranking()[0].expected - yours.expected).toFixed(digits)}`);
      }
      if (!h.finished()) line2 += ui.muted(`  thinking… ${Math.round(h.progress() * 100)}%`);
    } else if (g.isOver()) {
      line2 = this.overLine();
    } else if (this.message) {
      const color = { info: "inherit", good: theme.GOOD, bad: theme.BAD }[this.message.tone];
      line2 = ui.span(this.message.text, `color:${color}`);
    } else {
      line2 = ui.muted("Tap: aim · drag: slide · tap tile: place");
    }
    return (line1 ? `<div>${line1}</div>` : "") + `<div>${facts}</div><div>${line2}</div>`;
  }

  /// After the game: how it compares.
  overLine() {
    if (this.replayName) return ui.muted("Swipe or ◀ ▶ to step through it");
    const ai = this.rival?.score;
    const best = this.analysis?.best?.score;
    return `${ui.muted("engine")} ${ui.bold(ai ?? "…")}  ${ui.muted("best found")} ${ui.bold(best ?? "…", theme.HINT)}`;
  }

  /// Buttons for touch screens.
  touchButtons() {
    if (this.coachOn) return this.coach.touchButtons();
    const info = ["Info", "info"];
    if (this.review) {
      const other = this.movesOf("best") ? ["Switch", "Tab"] : ["Map", "Tab"];
      return [["◀", "ArrowLeft"], ["▶", "ArrowRight", "primary"], ["⏮", "Home"], ["⏭", "End"], other, ["Back", "Escape"], info];
    }
    if (this.game.isOver()) {
      if (this.replayName) return [["Step through", "b", "primary"], info];
      return [["Analysis", "a", "primary"], ["Best", "b"], ["New", "n"], ["Again", "r"], ["Board", "Escape"], info];
    }
    const hint = this.showHint && this.currentHint() ? ["Next", "g"] : ["Hint", "h"];
    return [["↺", "R"], ["↻", " "], hint, ["Place", "Enter", "primary"], ["Undo", "u"], ["New", "n"], info];
  }

  drawBoard() {
    if (this.coachOn && !this.coach.practice) return this.coach.drawBoard();
    const g = this.game;
    const n = g.tiles.length;
    if (this.review) {
      const r = this.review;
      const which = r.play === "best" ? "Best play found" : this.yoursTitle();
      this.s.title.textContent = `${which} · tile ${r.step} of ${n}${this.heightMap ? " · height map" : ""}`;
      this.view.draw(r.pieces, r.top, { heightMap: this.heightMap, spotlight: this.spotlight() });
      return;
    }
    this.s.title.textContent = this.heightMap ? "Board · height map" : "Board";
    const v = g.current();
    // Keys move the view with the tile; with the mouse it stays put.
    if (v != null) this.view.follow(g.cursor, v, g.rot, g.pieces, !this.mouseAim);
    else if (this.view.fresh) this.view.centre(extent(g.pieces));
    const hint = this.showHint ? this.currentHint() : null;
    const pick = hint?.ranking()[this.suggestion];
    this.view.draw(g.pieces, g.top, {
      heightMap: this.heightMap,
      spotlight: this.spotlight(),
      ghost: g.preview(),
      hint: pick ? { value: hint.value, move: pick.move } : null,
    });
  }

  messageLine() {
    if (!this.message) return "";
    const color = { info: "inherit", good: theme.GOOD, bad: theme.BAD }[this.message.tone];
    return `<div class="message" style="color:${color}">${ui.esc(this.message.text)}</div>`;
  }

  panel() {
    const g = this.game;
    const n = g.tiles.length;
    const turn = Math.min(n, g.placed() + (g.isOver() ? 0 : 1));
    const header =
      `<div>${ui.bold("NMBR 9", theme.ACCENT)}   ${ui.muted(`tile ${turn} of ${n}`)}</div>` +
      `<div>Score ${ui.bold(g.score())}   ${ui.muted(`deal #${this.seed}`)}</div>`;
    let hand;
    const v = g.current();
    if (v != null) {
      hand =
        `<div class="hand">${ui.tileShape(v, g.rot)}<div><div>${ui.bold(v, theme.valueCss(v))}</div>` +
        `<div>${ui.muted(`↻ ${g.rot * 90}°`)}</div></div></div>${placementLine(g.preview())}`;
    } else hand = ui.muted("All tiles placed");
    return (
      ui.box("Game", header) +
      ui.box("Tile in hand", hand) +
      ui.box("Still to come", ui.bagLines(g.stillToCome())) +
      ui.box("Levels", ui.levelLines(g.levels()) || "&nbsp;", "grow") +
      ui.box("Best move", this.hintBody()) +
      this.messageLine()
    );
  }

  hintBody() {
    const h = this.showHint ? this.currentHint() : null;
    if (!h) {
      if (this.game.isOver()) return ui.muted("Game over");
      return `${ui.bold("h", theme.ACCENT)} ${ui.muted("ask the engine")}`;
    }
    let out;
    const digits = h.digits ?? 1;
    if (h.forced()) out = `<div>${ui.muted("Only one way to place it")}</div>`;
    else if (h.finished()) out = `<div>${ui.muted(h.label ?? "Expected final score")}</div>`;
    else out = gauge(h.progress(), `thinking… ${Math.round(h.progress() * 100)}%`);
    const ranking = h.ranking();
    ranking.slice(0, SUGGESTIONS).forEach((r, k) => {
      const shown = k === this.suggestion;
      const style = shown ? `color:${theme.HINT};font-weight:700` : "";
      const exp = Number.isNaN(r.expected) ? "" : `${h.exact ? "=" : "≈"} ${r.expected.toFixed(digits).padStart(5)}`;
      out += `<div class="row" data-send="${k + 1}">${ui.span(`${shown ? "▸ " : "  "}${k + 1}. ${describe(h.value, r.move)}${exp}`, style)}</div>`;
    });
    const ghost = this.ghostMove();
    const yours = h.yoursRated();
    if (yours && sameMove(ghost, h.askedAt)) {
      const gap = ranking[0].expected - yours.expected;
      const verdict =
        yours.rank === 1 ? ui.bold(" best!", theme.GOOD) : ui.span(` −${gap.toFixed(digits)}`, `color:${gap < 1 ? theme.GOOD : theme.BAD}`);
      const exp = `${h.exact ? "=" : "≈"} ${yours.expected.toFixed(digits).padStart(5)}`;
      out += `<div class="gap"></div><div>${ui.esc(`  You ${`#${yours.rank}`.padEnd(4)}${describe(h.value, yours.move)}${exp}`)}${verdict}</div>`;
    } else if (ghost && !ranking.slice(0, SUGGESTIONS).some((r) => sameMove(r.move, ghost))) {
      out += `<div class="gap"></div><div>  ${ui.bold("h", theme.ACCENT)} ${ui.muted("rate the spot you're on")}</div>`;
    }
    out += `<div class="gap"></div><div>  ${ui.bold("g", theme.ACCENT)} ${ui.muted("next suggestion")}  ${ui.bold("1-3", theme.ACCENT)} ${ui.muted("pick")}</div>`;
    return out;
  }

  reviewPanel() {
    const g = this.game;
    const r = this.review;
    const n = g.tiles.length;
    const k = r.step - 1;
    const value = g.tiles[k];
    const yours = this.movesOf("yours");
    const best = this.movesOf("best");
    const after = (moves) => score(moves.slice(0, r.step));
    const describeAt = (m) => (m ? (m.layer && value ? `level ${m.layer}  +${value * m.layer}` : `level ${m.layer}`) : "…");
    const row = (label, play, m, total) => {
      const on = r.play === play;
      const style = on ? `font-weight:700;color:${theme.ACCENT}` : "";
      return `<div class="row" data-play="${play}">${ui.span(`${on ? "▸" : " "} ${label.padEnd(7)}${describeAt(m).padEnd(13)}`, style)}${ui.muted(total == null ? "" : `score ${total}`)}</div>`;
    };
    let out = `<div>${ui.muted(`Tile ${r.step} of ${n}: `)}${ui.bold(`a ${value}`, theme.valueCss(value))}</div><div class="gap"></div>`;
    out += row(this.yoursLabel(), "yours", yours[k], after(yours));
    if (this.analysis) out += row("Best", "best", best?.[k], best ? after(best) : null);
    out += `<div class="gap"></div>`;
    const finalLine = (label, s) => `<div>  ${ui.esc(label.padEnd(12))}${ui.bold(s ?? "…")}</div>`;
    if (this.replayName) out += finalLine("Final score", g.score());
    else {
      out += `<div>${ui.muted("Final scores")}</div>` + finalLine(this.yoursLabel(), g.score());
      if (this.analysis) out += finalLine("Best found", this.analysis.best?.score);
      if (this.rival) out += finalLine("Engine", this.rival.score);
    }
    out += `<div class="gap"></div>${ui.levelLines(levels(r.pieces))}`;
    if (this.analysis && this.analysis.stagesDone < this.analysis.stages) {
      out += `<div class="gap"></div><div>${ui.muted("Still searching for a better play…")}</div>`;
    }
    return ui.box("Review", out, "grow") + this.messageLine();
  }

  keys() {
    if (this.coachOn) return this.coach.keys();
    const g = this.game;
    if (this.review && !this.analysis) {
      return [
        ["←→", "step", "ArrowRight"],
        ["home/end", "first/last", "Home"],
        ["tab", "tiles / height map", "Tab"],
        ["esc", "summary", "Escape"],
      ];
    }
    if (this.review) {
      return [
        ["←→", "step", "ArrowRight"],
        ["home/end", "first/last", "Home"],
        ["tab", "your game / best play", "Tab"],
        ["v", "view", "v"],
        ["esc", "back", "Escape"],
      ];
    }
    if (g.isOver() && this.replayName) {
      return [
        ["b", "step through it", "b"],
        ["esc", "show/hide summary", "Escape"],
      ];
    }
    if (g.isOver()) {
      return [
        ["a", "game analysis", "a"],
        ["b", "review best play", "b"],
        ["n", "new game", "n"],
        ["r", "replay tiles", "r"],
        ["esc", "show/hide summary", "Escape"],
        ["u", "undo", "u"],
      ];
    }
    return [
      ["←↑↓→", "move", null],
      ["space", "rotate", " "],
      ["enter", "place", "Enter"],
      ["h", "hint", "h"],
      ["g", "next hint", "g"],
      ["u", "undo", "u"],
      ["tab", "view", "Tab"],
      ["?", "help", "?"],
      ["n", "new game", "n"],
    ];
  }

  drawKeys() {
    const keys = this.keys();
    const sig = JSON.stringify(keys);
    if (sig !== this.keySig) {
      this.keySig = sig;
      ui.keyBar(this.s.keys, keys, (e) => this.onKey(e));
    }
  }

  drawPopup() {
    const g = this.game;
    const show = g.isOver() && !this.hideSummary && !this.review && !this.coachOn;
    this.s.popup.hidden = !show;
    if (show) this.s.popup.innerHTML = this.summary();
  }

  summary() {
    const g = this.game;
    const s = g.score();
    let out = `<div class="box-title">${this.replayName ? "Replay" : "Game over"}</div>`;
    out += `<div>Final score  ${ui.bold(s, theme.ACCENT)}</div><div class="gap"></div>`;
    out += ui.levelLines(g.levels()) + `<div class="gap"></div>`;
    if (this.replayName) {
      out += `<div>${ui.muted(`Replay of ${this.replayName}`)}</div><div class="gap"></div>`;
      out += actions([["b", "step through it"]]);
      return out;
    }
    const ai = this.rival?.score;
    if (ai == null) out += `<div>${ui.muted("The engine is playing these tiles…")}</div>`;
    else {
      const [verdict, color] =
        s > ai ? [`you won by ${s - ai}!`, theme.GOOD] : s === ai ? ["a tie!", theme.ACCENT] : [`it won by ${ai - s}`, theme.BAD];
      out += `<div>The engine scored ${ai} on these tiles: ${ui.bold(verdict, color)}</div>`;
    }
    const best = this.analysis?.best;
    if (best) {
      const searching = this.analysis.stagesDone < this.analysis.stages;
      out += `<div>Best play found on these tiles: ${ui.bold(best.score, theme.HINT)}${searching ? ui.muted("  (still searching…)") : ""}</div>`;
    } else out += `<div>${ui.muted("Searching for the best play of these tiles…")}</div>`;
    const a = this.gameAnalysis;
    if (a) {
      out += `<div class="gap"></div>`;
      if (a.finished()) {
        const sum = a.summary();
        const worst = ["blunder", "mistake", "inaccuracy"]
          .filter((id) => sum.counts[id])
          .map((id) => plural(sum.counts[id], id));
        out += `<div>Your accuracy: ${ui.bold(`${sum.accuracy.toFixed(0)}%`, theme.ACCENT)}  ${ui.muted(worst.length ? worst.join(", ") : "no mistakes!")}</div>`;
      } else out += `<div>${ui.muted(`Analysing your moves… ${Math.round(a.progress() * 100)}%`)}</div>`;
    }
    out += `<div class="gap"></div>`;
    out += actions([
      ...(a ? [["a", "game analysis"]] : []),
      ["b", "review the best play"],
      ["n", "new game"],
      ["r", "replay these tiles"],
      ["esc", "view board", "Escape"],
    ]);
    return out;
  }
}

function actions(list) {
  return `<div class="actions">${list
    .map(([k, what, send]) => `<button class="key" data-send="${ui.esc(send ?? k)}"><b>${ui.esc(k)}</b> ${ui.esc(what)}</button>`)
    .join("")}</div>`;
}

function gauge(fraction, label) {
  return `<div class="gauge"><div style="width:${(fraction * 100).toFixed(1)}%"></div><span>${ui.esc(label)}</span></div>`;
}

/// "level 2  +14" for a move of `value`.
export function describe(value, m) {
  const points = value * m.layer;
  return `level ${m.layer}  ${(points > 0 ? `+${points}` : "").padEnd(3)} `;
}

export function placementLine(p) {
  if (!p) return "";
  if (!p.problem) {
    const points = p.value * p.move.layer;
    return `<div>${ui.span("✓ ", `color:${theme.GOOD}`)}Level ${p.move.layer}${points > 0 ? ui.bold(`  +${points} pts`, theme.GOOD) : ""}</div>`;
  }
  return `<div>${ui.span(`✗ ${p.problem}`, `color:${theme.BAD}`)}</div>`;
}
