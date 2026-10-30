// Coaching after a game, in the spirit of a chess engine's game review.
//
// Every decision the player made is checked again: at the position before
// it, the engine plays the rest of the game out from each candidate move,
// the player's own included (the same Monte Carlo rollouts as the hints),
// and the player's move is judged by the expected final points it gave away
// compared with the engine's own pick. That loss plays the part of a chess
// engine's centipawn loss: it grades each move (engine's pick, excellent, good,
// inaccuracy, mistake, blunder), gives the game an accuracy, draws an
// evaluation graph, and points at the moments that mattered. Each moment
// gets a reason in words and can be replayed to look for a better move.
// The engine is a strong heuristic, not perfect play, so the words used are
// "the engine's pick", never "best".

import { Game, TopView, cells, levels, samePlacement } from "./game.js";
import { extent } from "./board.js";
import { Hint } from "./hint.js";
import * as theme from "./theme.js";
import * as ui from "./ui.js";

const CHUNK = 8;
const ROLLOUTS = 256;
const PER_LAYER = 6;

/// Grades by expected points lost, as chess grades by centipawns.
export const GRADES = [
  { id: "best", name: "Engine's pick", mark: "!", max: 0.05, color: "rgb(120,220,120)" },
  { id: "excellent", name: "Excellent", mark: "", max: 0.75, color: "rgb(150,210,120)" },
  { id: "good", name: "Good", mark: "", max: 2, color: "rgb(170,190,140)" },
  { id: "inaccuracy", name: "Inaccuracy", mark: "?!", max: 4, color: "rgb(240,200,80)" },
  { id: "mistake", name: "Mistake", mark: "?", max: 8, color: "rgb(250,150,70)" },
  { id: "blunder", name: "Blunder", mark: "??", max: Infinity, color: "rgb(255,95,95)" },
];
const FORCED = { id: "forced", name: "Only move", mark: "", max: 0, color: "rgb(140,145,160)" };
const grade = (loss) => GRADES.find((g) => loss <= g.max);

/// How good a move was, from 100 (the engine's pick) down: an exponential
/// in the points lost, like chess accuracy scores.
const moveAccuracy = (loss) => 100 * Math.exp(-loss / 6);

const PHASES = [
  ["Opening", 1, 7],
  ["Middle game", 7, 14],
  ["Endgame", 14, 20],
];

/// Sides of the tile at `move` that touch another tile on its level.
function edgesTouching(pieces, value, move) {
  const occupied = new Set();
  for (const p of pieces) if (p.layer === move.layer) for (const [i, j] of cells(p.value, p)) occupied.add(`${i},${j}`);
  let n = 0;
  for (const [i, j] of cells(value, move)) {
    for (const [a, b] of [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]) if (occupied.has(`${a},${b}`)) n++;
  }
  return n;
}

/// The engine's verdict on every move of a finished game, filling in as the
/// rollouts come back from the worker pool.
export class GameAnalysis {
  constructor(pool, engine, tiles, pieces, onUpdate) {
    this.tiles = tiles;
    this.pieces = pieces.map((p) => ({ ...p }));
    this.positions = [];
    const jobs = [];
    // Move k places tile k on the first k tiles (the first goes in the middle).
    for (let k = 1; k < pieces.length; k++) {
      const before = this.pieces.slice(0, k);
      const value = tiles[k];
      engine.load(before);
      const candidates = engine.hintCandidates(value, PER_LAYER);
      const yours = this.pieces[k];
      let mine = candidates.findIndex((m) => samePlacement(value, m, yours));
      if (mine < 0) {
        candidates.push({ layer: yours.layer, i: yours.i, j: yours.j, rot: yours.rot });
        mine = candidates.length - 1;
      }
      const pos = { k, value, candidates, mine, totals: candidates.map(() => 0), done: 0, target: ROLLOUTS };
      if (candidates.length <= 1) pos.done = pos.target;
      else {
        for (let c = 0; c < ROLLOUTS / CHUNK; c++) {
          jobs.push({ type: "rollouts", pieces: before, value, seed: c, n: CHUNK, candidates, k });
        }
      }
      this.positions.push(pos);
    }
    engine.load(this.pieces);
    this.cancel = pool.submit(jobs, (reply, job) => {
      if (reply.type !== "done") return;
      const pos = this.positions[job.k - 1];
      reply.totals.forEach((t, c) => (pos.totals[c] += t));
      pos.done += CHUNK;
      this.cached = null;
      onUpdate();
    });
  }

  stop() {
    this.cancel?.();
  }

  progress() {
    const done = this.positions.reduce((s, p) => s + p.done, 0);
    return done / (this.positions.length * ROLLOUTS);
  }

  finished() {
    return this.positions.every((p) => p.done >= p.target);
  }

  /// The verdict on move k (1 = the second tile), once its rollouts are in:
  /// {k, value, yours, best, eYours, eBest, loss, rank, grade, ...}.
  move(k) {
    const pos = this.positions[k - 1];
    if (!pos || pos.done < pos.target) return null;
    const yours = this.pieces[k];
    const before = this.pieces.slice(0, k);
    const remaining = this.tiles.slice(k + 1);
    if (pos.candidates.length <= 1) {
      return { k, value: pos.value, yours, best: yours, eYours: null, eBest: null, loss: 0, rank: 1, grade: FORCED, before, remaining };
    }
    const e = pos.totals.map((t) => t / pos.done);
    const bestIdx = e.indexOf(Math.max(...e));
    const loss = Math.max(0, e[bestIdx] - e[pos.mine]);
    const rank = 1 + e.filter((x) => x > e[pos.mine] + 1e-9).length;
    return {
      k,
      value: pos.value,
      yours,
      best: pos.candidates[bestIdx],
      eYours: e[pos.mine],
      eBest: e[bestIdx],
      loss,
      rank,
      grade: rank === 1 ? GRADES[0] : grade(Math.max(loss, 0.051)),
      before,
      remaining,
      edgesYours: edgesTouching(before, pos.value, yours),
      edgesBest: edgesTouching(before, pos.value, pos.candidates[bestIdx]),
    };
  }

  moves() {
    return this.positions.map((p) => this.move(p.k)).filter(Boolean);
  }

  /// The whole game: accuracy, grades, phases, lessons and luck.
  summary() {
    if (this.cached) return this.cached;
    const moves = this.moves();
    const judged = moves.filter((m) => m.grade !== FORCED);
    const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
    const accuracy = mean(judged.map((m) => moveAccuracy(m.loss)));
    const counts = Object.fromEntries([...GRADES, FORCED].map((g) => [g.id, 0]));
    for (const m of moves) counts[m.grade.id]++;
    const phases = PHASES.map(([name, a, b]) => {
      const ms = judged.filter((m) => m.k >= a && m.k < b);
      return { name, accuracy: mean(ms.map((m) => moveAccuracy(m.loss))), lost: ms.reduce((s, m) => s + m.loss, 0) };
    });
    const lost = judged.reduce((s, m) => s + m.loss, 0);
    this.cached = { moves, accuracy, counts, phases, lost, lessons: lessons(judged) };
    return this.cached;
  }
}

/// What kind of slip a move was, by how it differs from the engine's pick.
export function slip(m) {
  if (m.loss < 0.75) return null;
  if (m.best.layer > m.yours.layer) return "climb";
  if (m.yours.layer > m.best.layer) return m.value <= 3 ? "lowHigh" : "early";
  if (m.edgesBest > m.edgesYours) return "loose";
  return "shape";
}

const LESSONS = {
  climb: {
    title: "Missed chances to go up",
    advice: "When a tile can sit a level higher, take the points, above all with 6 to 9.",
  },
  lowHigh: {
    title: "Low numbers placed high",
    advice: "Keep 0 to 3 as foundations: space up high is worth more under the big numbers still to come.",
  },
  early: {
    title: "Climbing too early",
    advice: "Going up now can spend support the bigger tiles will want later. Build a wide base first.",
  },
  loose: {
    title: "Loose placements",
    advice: "Tiles that touch more edges make a compact base, with more places for later tiles to go up.",
  },
  shape: {
    title: "Spots with less room to grow",
    advice: "On the same level, prefer the spot that leaves shapes the tiles still to come can sit on.",
  },
};

function lessons(judged) {
  const by = {};
  for (const m of judged) {
    const t = slip(m);
    if (!t) continue;
    by[t] ??= { id: t, ...LESSONS[t], moves: 0, lost: 0 };
    by[t].moves++;
    by[t].lost += m.loss;
  }
  return Object.values(by).sort((a, b) => b.lost - a.lost);
}

const lvl = (m, v) => `level ${m.layer}${v * m.layer ? ` (+${v * m.layer})` : ""}`;

/// Why the engine prefers its move, in words.
export function explain(m) {
  const v = m.value;
  if (m.grade === FORCED) return `There was only one way to place this ${v}.`;
  if (m.rank === 1) return "The engine's top choice too.";
  if (m.loss < 0.75) return `About as good as the engine's choice, ${lvl(m.best, v)}: the difference is within its margin of error.`;
  const big = m.remaining.filter((t) => t >= 6).sort((a, b) => b - a);
  const toCome = big.length ? ` Still to come: ${big.join(", ")}.` : "";
  switch (slip(m)) {
    case "climb":
      return (
        `This ${v} could have gone up to level ${m.best.layer} for +${v * m.best.layer} straight away; you put it on ${lvl(m.yours, v)}.` +
        (v >= 6 ? " High numbers earn the most up high: when a 9 can climb, it usually should." : "")
      );
    case "lowHigh":
      return `A ${v} on level ${m.yours.layer} takes space that a bigger number could use later. The engine kept it on ${lvl(m.best, v)}.${toCome}`;
    case "early":
      return `Going up to level ${m.yours.layer} here cost more than its +${v * m.yours.layer}: the engine kept the ${v} on ${lvl(m.best, v)}, saving the support below for later tiles.${toCome}`;
    case "loose":
      return `Same level, but the engine's spot fits tighter (touching ${m.edgesBest} edges to your ${m.edgesYours}), keeping the stack compact so later tiles have more places to go up.`;
    default:
      return `Same level, but the engine's spot leaves more room to build on for the ${m.remaining.length} tiles still to come.${toCome}`;
  }
}

// ----------------------------------------------------------------------
// The coaching screen, run inside the game screen (a PlayApp).

export class Coach {
  constructor(app) {
    this.app = app;
    this.step = 1;
    this.show = "both"; // which moves are drawn: both, yours, best
    this.practice = null;
    this.retries = {};
  }

  get analysis() {
    return this.app.gameAnalysis;
  }

  /// Open on the biggest mistake (or the first move).
  open() {
    const moves = this.analysis.moves();
    const worst = moves.reduce((w, m) => (!w || m.loss > w.loss ? m : w), null);
    this.step = worst && worst.loss >= 0.75 ? worst.k : 1;
    this.frame();
  }

  /// Show the whole game if it fits, otherwise the move being looked at.
  frame() {
    const view = this.app.view;
    const all = extent(this.app.game.pieces);
    const fits = all[1] - all[0] <= view.vp.rows && all[3] - all[2] <= view.vp.cols;
    if (fits) {
      view.centre(all);
      return;
    }
    const m = this.current();
    const k = this.step;
    const shown = this.app.game.pieces.slice(Math.max(0, k - 3), k + 1);
    if (m?.best) shown.push({ value: m.value, ...m.best });
    view.centre(extent(shown));
  }

  current() {
    return this.analysis.move(this.step);
  }

  go(k) {
    const n = this.analysis.positions.length;
    this.step = Math.max(1, Math.min(n, k));
    this.app.message = null;
    this.frame();
  }

  /// The next (or previous) move that lost at least an inaccuracy's worth.
  jump(dir) {
    const moves = this.analysis.moves().filter((m) => m.loss >= GRADES[2].max);
    const next = dir > 0 ? moves.find((m) => m.k > this.step) : moves.reverse().find((m) => m.k < this.step);
    if (next) this.go(next.k);
    else this.app.say(dir > 0 ? "No more mistakes after this one" : "No mistakes before this one");
  }

  onKey(e) {
    if (this.practice) return this.onPracticeKey(e);
    switch (e.key) {
      case "ArrowLeft":
      case "a":
        this.go(this.step - 1);
        break;
      case "ArrowRight":
      case "d":
        this.go(this.step + 1);
        break;
      case "Home":
        this.go(1);
        break;
      case "End":
        this.go(Infinity);
        break;
      case "m":
      case "ArrowDown":
        this.jump(1);
        break;
      case "M":
      case "ArrowUp":
        this.jump(-1);
        break;
      case "t":
      case "Tab":
        this.show = { both: "yours", yours: "best", best: "both" }[this.show];
        break;
      case "r":
        this.startPractice();
        break;
      case "v":
        this.app.heightMap = !this.app.heightMap;
        break;
      case "c":
        this.frame();
        break;
      case "?":
        this.app.opts.onHelp?.();
        break;
      case "Escape":
      case "q":
        this.app.coachOn = false;
        this.app.hideSummary = false;
        break;
      default:
        return false;
    }
    return true;
  }

  // --- Retrying a move ------------------------------------------------

  /// Set up the position before this move and let the player try again.
  startPractice() {
    const m = this.current();
    if (!m || m.grade === FORCED) {
      this.app.say("Nothing to retry here");
      return;
    }
    const app = this.app;
    const g = new Game(app.engine, app.game.tiles);
    g.pieces = m.before.map((p) => ({ ...p }));
    g.refresh();
    g.jumpTo(m.yours);
    this.practice = { k: m.k, game: g, saved: app.game, checking: null };
    app.game = g;
    app.view.fresh = true;
    app.say(`Find a better place for this ${m.value}`);
  }

  endPractice() {
    if (!this.practice) return;
    this.practice.checking?.stop();
    this.app.game = this.practice.saved;
    this.app.game.refresh();
    this.practice = null;
    this.frame();
  }

  onPracticeKey(e) {
    if (e.key === "Escape" || e.key === "q") {
      this.endPractice();
      return true;
    }
    if (e.key === "h") {
      this.app.say("No hints here: find it yourself!");
      return true;
    }
    if (e.key === "Enter") {
      this.tryMove();
      return true;
    }
    if (["u", "Backspace", "n", "g", "1", "2", "3"].includes(e.key)) return true;
    return this.app.onPlayKey(e, null);
  }

  /// Judge the move the player just tried, with the same rollouts as the
  /// analysis (so the numbers compare).
  tryMove() {
    const pr = this.practice;
    if (!pr || pr.checking) return;
    const p = pr.game.preview();
    if (p.problem) {
      this.app.say(p.problem, "bad");
      return;
    }
    const m = this.analysis.move(pr.k);
    const pos = this.analysis.positions[pr.k - 1];
    const known = pos.candidates.findIndex((c) => samePlacement(m.value, c, p.move));
    const finish = (expected) => {
      const loss = Math.max(0, m.eBest - expected);
      this.retries[pr.k] = { move: p.move, expected, loss, grade: loss < 0.05 ? GRADES[0] : grade(loss) };
      this.endPractice();
      const r = this.retries[pr.k];
      const better = m.loss - loss;
      const verdict =
        better > 0.05 ? `${better.toFixed(1)} points better than in your game` : better < -0.05 ? "worse than in your game" : "no better than in your game";
      this.app.say(`${r.grade.name}: −${loss.toFixed(1)}, ${verdict}`, better > 0.05 ? "good" : "bad");
      this.app.render();
    };
    if (known >= 0) {
      finish(pos.totals[known] / pos.done);
      return;
    }
    pr.checking = new Hint(this.app.pool, pr.game, p.move, ROLLOUTS, () => {
      if (pr.checking?.finished()) finish(pr.checking.yoursRated().expected);
      else this.app.render();
    });
    this.app.say("Checking your move…");
  }

  // --- Drawing ----------------------------------------------------------

  drawBoard() {
    const app = this.app;
    const m = this.current();
    const n = this.analysis.positions.length + 1;
    app.s.title.textContent = `Game analysis · tile ${this.step + 1} of ${n}${app.heightMap ? " · height map" : ""}`;
    const pieces = app.game.pieces.slice(0, this.step);
    const top = new TopView(pieces);
    const yours = app.game.pieces[this.step];
    const retry = this.retries[this.step];
    const opts = { heightMap: app.heightMap, spotlight: app.spotlight(), last: pieces.length - 1 };
    if (this.show !== "best") opts.ghost = { value: yours.value, move: retry?.move ?? yours, problem: null };
    if (m && this.show !== "yours" && m.best && !samePlacement(m.value, m.best, yours)) opts.hint = { value: m.value, move: m.best };
    app.view.draw(pieces, top, opts);
  }

  verdictLine(m) {
    if (!m) return ui.muted("Analysing this move…");
    const g = m.grade;
    const badge = ui.span(`${g.mark ? `${g.mark} ` : ""}${g.name}`, `color:${g.color};font-weight:700`);
    if (g === FORCED) return badge;
    return `${badge}  ${m.loss >= 0.05 ? ui.span(`−${m.loss.toFixed(1)}`, `color:${g.color}`) : ""}`;
  }

  panel() {
    const a = this.analysis;
    const sum = a.summary();
    const m = this.current();
    const n = a.positions.length + 1;
    let out = "";

    // The report card.
    let card;
    if (!a.finished()) card = scores(this.app) + `<div>${ui.muted("Analysing your moves…")}</div>` + gauge(a.progress());
    else {
      const acc = sum.accuracy;
      card = scores(this.app) + `<div>Accuracy ${ui.bold(`${acc.toFixed(0)}%`, accColor(acc))}   ${ui.muted("points given away")} ${ui.bold(sum.lost.toFixed(1))}</div>`;
      card += `<div>${ui.muted("(estimated final points lost, move by move, next to the engine's picks)")}</div>`;
      card += `<div class="grades">${GRADES.slice(3)
        .reverse()
        .map((g) => ui.span(plural(sum.counts[g.id], g.name.toLowerCase()), `color:${sum.counts[g.id] ? g.color : theme.MUTED}`))
        .join(ui.muted(" · "))}</div>`;
      card += `<div>${sum.phases.map((p) => `${ui.muted(p.name)} ${p.accuracy == null ? "–" : ui.bold(`${p.accuracy.toFixed(0)}%`, accColor(p.accuracy))}`).join("  ")}</div>`;
    }
    out += ui.box("Game analysis", `<div class="wrap">${card}</div>`);
    out += ui.box("Estimated final score", graph(a, this.step));

    // This move.
    let move = `<div>${ui.muted(`Tile ${this.step + 1} of ${n}:`)} ${ui.bold(`a ${a.tiles[this.step]}`, theme.valueCss(a.tiles[this.step]))}</div><div>${this.verdictLine(m)}</div>`;
    if (m && m.grade !== FORCED) {
      const row = (label, mv, e, color) =>
        `<div>${ui.span(label.padEnd(7), `color:${color};font-weight:700`)}${ui.esc(lvl(mv, m.value).padEnd(16))}${ui.muted(`≈ ${e.toFixed(1)}`)}</div>`;
      move += row("You", m.yours, m.eYours, "#fff") + row("Engine", m.best, m.eBest, theme.HINT);
      const r = this.retries[this.step];
      if (r) move += row("Retry", r.move, r.expected, r.grade.color);
    }
    if (m) move += `<div class="gap"></div><div class="wrap">${ui.esc(explain(m))}</div>`;
    out += ui.box("This move", move);

    // Lessons.
    if (a.finished() && sum.lessons.length) {
      const ls = sum.lessons
        .slice(0, 3)
        .map((l) => `<div class="wrap"><b>${ui.esc(l.title)}</b> ${ui.muted(`(${l.moves} move${l.moves === 1 ? "" : "s"}, −${l.lost.toFixed(1)})`)}<br>${ui.esc(l.advice)}</div>`)
        .join(`<div class="gap"></div>`);
      out += ui.box("Lessons from this game", ls);
    }

    // Every move, as a chess move list.
    out += ui.box(
      "Moves",
      `<div class="movelist">${a.positions
        .map((p) => {
          const mv = a.move(p.k);
          const g = mv?.grade;
          const on = p.k === this.step ? " on" : "";
          return `<button class="mv${on}" data-step="${p.k}"><small>${p.k + 1}</small> <b style="color:${theme.valueCss(a.tiles[p.k])}">${a.tiles[p.k]}</b>${g ? ui.span(g.mark || "·", `color:${g.color}`) : ""}</button>`;
        })
        .join("")}</div>`,
    );
    return out + this.app.messageLine();
  }

  practicePanel() {
    const m = this.analysis.move(this.practice.k);
    const g = this.practice.game;
    const v = g.current();
    const hand =
      `<div class="hand">${ui.tileShape(v, g.rot)}<div><div>${ui.bold(v, theme.valueCss(v))}</div>` +
      `<div>${ui.muted(`↻ ${g.rot * 90}°`)}</div></div></div>`;
    const preview = g.preview();
    const where = preview.problem
      ? ui.span(`✗ ${preview.problem}`, `color:${theme.BAD}`)
      : ui.span(`✓ level ${preview.move.layer}`, `color:${theme.GOOD}`);
    const intro = `In your game this ${m.value} lost ${m.loss.toFixed(1)} points (${m.grade.name.toLowerCase()}). Find a better place, then press enter (or tap the tile).`;
    return (
      ui.box("Retry this move", `<div class="wrap">${ui.esc(intro)}</div>`) +
      ui.box("Tile in hand", hand + `<div>${where}</div>`) +
      ui.box("Levels", ui.levelLines(levels(g.pieces)) || "&nbsp;", "grow") +
      this.app.messageLine()
    );
  }

  statusHtml() {
    if (this.practice) {
      const p = this.practice.game.preview();
      const where = p.problem ? ui.span(`✗ ${p.problem}`, `color:${theme.BAD}`) : ui.span(`✓ level ${p.move.layer}`, `color:${theme.GOOD}`);
      const msg = this.app.message ? ui.esc(this.app.message.text) : ui.muted("Find a better place, then tap the tile");
      return `<div>${ui.bold("Retry", theme.ACCENT)} ${where}</div><div>${msg}</div>`;
    }
    const a = this.analysis;
    const m = this.current();
    const line1 = `${ui.muted(`Tile ${this.step + 1}:`)} ${ui.bold(`a ${a.tiles[this.step]}`, theme.valueCss(a.tiles[this.step]))}  ${this.verdictLine(m)}`;
    let line2 = "";
    if (m && m.grade !== FORCED) line2 = `${ui.muted("you")} ${ui.esc(lvl(m.yours, m.value))}  ${ui.span(`engine ${lvl(m.best, m.value)}`, `color:${theme.HINT}`)}`;
    const sum = a.summary();
    const line3 = a.finished()
      ? `${ui.muted("accuracy")} ${ui.bold(`${sum.accuracy.toFixed(0)}%`, accColor(sum.accuracy))}  ${ui.muted("given away")} ${ui.bold(sum.lost.toFixed(1))}`
      : ui.muted(`Analysing… ${Math.round(a.progress() * 100)}%`);
    return `<div>${line1}</div><div>${line2 || "&nbsp;"}</div><div>${line3}</div>`;
  }

  touchButtons() {
    if (this.practice) return [["↺", "R"], ["↻", " "], ["Try it", "Enter", "primary"], ["Cancel", "Escape"]];
    const show = { both: "Both", yours: "Yours", best: "Engine" }[this.show];
    return [["◀", "ArrowLeft"], ["▶", "ArrowRight", "primary"], ["Next ??", "m"], ["Retry", "r"], [show, "t"], ["Back", "Escape"], ["Info", "info"]];
  }

  keys() {
    if (this.practice) {
      return [
        ["←↑↓→", "move", null],
        ["space", "rotate", " "],
        ["enter", "try it", "Enter"],
        ["esc", "cancel", "Escape"],
      ];
    }
    return [
      ["←→", "move by move", "ArrowRight"],
      ["↓↑ / m", "next / previous mistake", "m"],
      ["t", "show yours / the engine's / both", "t"],
      ["r", "retry this move", "r"],
      ["v", "height map", "v"],
      ["esc", "back", "Escape"],
    ];
  }
}

/// The game's final scores: yours, and the engine's on the same tiles in the
/// same order (not perfect play).
function scores(app) {
  const col = (label, s, color) => `${ui.muted(label)} ${ui.bold(s ?? "…", color)}`;
  return (
    `<div>${[
      col("You", app.game.score(), theme.ACCENT),
      ...(app.rival ? [col("Engine", app.rival.score, theme.HINT)] : []),
    ].join("   ")}</div>` +
    `<div>${ui.muted("Engine: the computer playing these tiles in the same order, knowing no more than you did. It's a strong heuristic, not perfect play.")}</div>` +
    `<div class="gap"></div>`
  );
}

/// "2 mistakes", "1 inaccuracy", "3 inaccuracies".
export function plural(n, word) {
  if (n === 1) return `1 ${word}`;
  return `${n} ${word.endsWith("y") ? `${word.slice(0, -1)}ies` : `${word}s`}`;
}

function accColor(a) {
  return a >= 90 ? theme.GOOD : a >= 75 ? "rgb(240,200,80)" : theme.BAD;
}

function gauge(fraction) {
  return `<div class="gauge"><div style="width:${(fraction * 100).toFixed(1)}%"></div><span>${Math.round(fraction * 100)}%</span></div>`;
}

/// The evaluation graph: after each move, the expected final score of the
/// engine's pick (cyan) and of yours (white), with the gap between them
/// shaded and each move's dot coloured by its grade. Click to jump.
function graph(a, step) {
  const moves = a.positions.map((p) => a.move(p.k));
  const vals = moves.flatMap((m) => (m?.eBest != null ? [m.eBest, m.eYours] : []));
  if (vals.length < 2) return ui.muted("Analysing…");
  const W = 300;
  const H = 96;
  const pad = 6;
  const lo = Math.floor(Math.min(...vals) / 10) * 10;
  const hi = Math.ceil(Math.max(...vals) / 10) * 10 || lo + 10;
  const n = a.positions.length;
  const X = (k) => pad + ((k - 1) / Math.max(1, n - 1)) * (W - 2 * pad);
  const Y = (v) => H - pad - ((v - lo) / Math.max(1, hi - lo)) * (H - 2 * pad);
  const pts = moves.filter((m) => m?.eBest != null);
  const line = (key) => pts.map((m, i) => `${i ? "L" : "M"}${X(m.k).toFixed(1)},${Y(m[key]).toFixed(1)}`).join("");
  const area =
    pts.map((m, i) => `${i ? "L" : "M"}${X(m.k).toFixed(1)},${Y(m.eBest).toFixed(1)}`).join("") +
    pts
      .slice()
      .reverse()
      .map((m) => `L${X(m.k).toFixed(1)},${Y(m.eYours).toFixed(1)}`)
      .join("") +
    "Z";
  let svg = `<svg class="graph" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  for (const v of [lo, (lo + hi) / 2, hi]) {
    svg += `<line x1="0" x2="${W}" y1="${Y(v)}" y2="${Y(v)}" stroke="rgb(50,54,66)" stroke-width="1"/>`;
    svg += `<text x="${W - 2}" y="${Y(v) - 2}" text-anchor="end">${v.toFixed(0)}</text>`;
  }
  svg += `<line x1="${X(step)}" x2="${X(step)}" y1="0" y2="${H}" stroke="rgb(255,200,80)" stroke-width="1.5" stroke-dasharray="3 3"/>`;
  svg += `<path d="${area}" fill="rgba(255,95,95,0.28)"/>`;
  svg += `<path d="${line("eBest")}" fill="none" stroke="${theme.HINT}" stroke-width="1.6"/>`;
  svg += `<path d="${line("eYours")}" fill="none" stroke="#fff" stroke-width="1.6"/>`;
  for (const m of pts) {
    svg += `<circle data-step="${m.k}" cx="${X(m.k)}" cy="${Y(m.eYours)}" r="${m.k === step ? 4.5 : 3}" fill="${m.grade.color}"/>`;
  }
  for (let k = 1; k <= n; k++) {
    svg += `<rect data-step="${k}" x="${X(k) - (W / n) / 2}" y="0" width="${W / n}" height="${H}" fill="transparent"/>`;
  }
  return `${svg}</svg><div class="legend">${ui.span("— engine's pick", `color:${theme.HINT}`)}  ${ui.span("— yours", "color:#fff")}  ${ui.muted("click a move")}</div>`;
}
