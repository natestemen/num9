// A web worker with its own copy of the engine, for work that would freeze
// the page: hint rollouts, the engine playing a deal, the search for the
// best play. Each request carries an id that its
// replies repeat.

import { Engine } from "./engine.js";

let current = null;
const ready = Engine.load((score, moves) => {
  postMessage({ id: current, type: "improved", score, moves });
});

const handlers = {
  rollouts(e, m) {
    e.load(m.pieces);
    return { totals: e.rollouts(m.value, m.seed, m.n, m.candidates) };
  },
  play(e, m) {
    return e.play(m.tiles, m.seed);
  },
  analyze(e, m) {
    // Stages of increasing depth, each reported as it ends.
    let best = null;
    for (const [k, [level, perLayer]] of m.stages.entries()) {
      const sol = e.analyze(m.tiles, level, perLayer, k);
      if (!best || sol.score > best.score) best = sol;
      postMessage({ id: m.id, type: "stage", stage: k + 1, best });
    }
    return best;
  },
};

onmessage = async ({ data: m }) => {
  const engine = await ready;
  current = m.id;
  try {
    postMessage({ id: m.id, type: "done", ...handlers[m.type](engine, m) });
  } catch (err) {
    postMessage({ id: m.id, type: "error", message: String(err) });
  }
};
