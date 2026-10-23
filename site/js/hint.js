// The best move, from the engine's Monte Carlo rollouts (as in the terminal
// game's hint.rs): each candidate move is played out many times with the
// remaining tiles shuffled, and ranked by average final score. The work goes
// out in chunks of rollouts to the worker pool, so results stream in.

import { samePlacement, sameMove } from "./game.js";

const CHUNK = 8;
/// Candidate moves kept per level (more than the engine plays with, so
/// there are several options to compare).
const PER_LAYER = 6;

export class Hint {
  /// Search for the best place for the current tile of `game`, also rating
  /// the player's move `yours` if given; `onUpdate` hears each chunk.
  constructor(pool, game, yours, rollouts, onUpdate) {
    this.placed = game.placed();
    this.value = game.current();
    this.askedAt = yours;
    const picks = game.engine.hintCandidates(this.value, PER_LAYER);
    this.yours = null;
    if (yours) {
      const k = picks.findIndex((m) => samePlacement(this.value, m, yours));
      if (k >= 0) this.yours = k;
      else if (game.legalList.some((m) => sameMove(m, yours))) {
        picks.push(yours);
        this.yours = picks.length - 1;
      }
    }
    this.candidates = picks;
    this.totals = picks.map(() => 0);
    this.done = 0;
    const chunks = Math.max(1, Math.ceil(rollouts / CHUNK));
    this.target = chunks * CHUNK;
    if (picks.length <= 1) {
      // Nothing to compare: no need to simulate.
      this.done = this.target;
      this.forcedOnly = true;
      return;
    }
    const jobs = Array.from({ length: chunks }, (_, c) => ({
      type: "rollouts",
      pieces: game.pieces,
      value: this.value,
      seed: c,
      n: CHUNK,
      candidates: picks,
    }));
    this.cancel = pool.submit(jobs, (reply) => {
      if (reply.type !== "done") return;
      reply.totals.forEach((t, k) => (this.totals[k] += t));
      this.done += CHUNK;
      onUpdate();
    });
  }

  stop() {
    this.cancel?.();
  }

  finished() {
    return this.done >= this.target;
  }

  progress() {
    return Math.min(1, this.done / this.target);
  }

  /// Whether there is only one way to place the tile.
  forced() {
    return this.candidates.length <= 1;
  }

  rated(k) {
    const expected = this.forcedOnly ? NaN : this.totals[k] / Math.max(1, this.done);
    return { move: this.candidates[k], expected };
  }

  /// Candidates, best first (empty until results arrive).
  ranking() {
    if (this.done === 0) return [];
    if (this.forcedOnly) return this.candidates.map((_, k) => this.rated(k));
    return this.candidates
      .map((_, k) => k)
      .sort((a, b) => this.totals[b] - this.totals[a] || a - b)
      .map((k) => this.rated(k));
  }

  /// The player's move with its rank (1 = best), once there are results.
  yoursRated() {
    const k = this.yours;
    if (k == null || this.done === 0 || this.forcedOnly) return null;
    const rank = 1 + this.totals.filter((t) => t > this.totals[k]).length;
    return { rank, ...this.rated(k) };
  }
}
