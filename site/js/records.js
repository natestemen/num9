// Recorded plays with the tiles in any order: the 236 record and every
// distinct play above 200 that the search campaign found, replayed on the
// game screen.

import { parsePlay } from "./game.js";
import { PlayApp } from "./play.js";
import * as ui from "./ui.js";

export class RecordsApp {
  constructor(section, engine, pool, data, opts = {}) {
    this.data = data;
    this.opts = opts;
    section.insertAdjacentHTML(
      "beforeend",
      `<div class="record-list box"><div class="box-title">Plays above 200 (${data.plays.length})</div><div class="box-body"><div class="list"></div></div></div>`,
    );
    const list = section.querySelector(".list");
    list.innerHTML = data.plays
      .map((p, k) => `<button data-k="${k}"><b>${p.score}</b> <span>level ${p.top}</span> <small>${ui.esc(p.name)}</small></button>`)
      .join("");
    list.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-k]");
      if (b) this.select(Number(b.dataset.k));
    });
    this.list = list;
    this.app = new PlayApp(section, engine, pool, { ...opts, replay: this.replayOf(0) });
    this.select(Math.max(0, data.plays.findIndex((p) => p.name === opts.name)));
  }

  replayOf(k) {
    const p = this.data.plays[k];
    return { name: p.name, steps: parsePlay(p.moves) };
  }

  select(k) {
    this.app.loadReplay(this.replayOf(k));
    for (const b of this.list.querySelectorAll("button")) b.classList.toggle("on", Number(b.dataset.k) === k);
    this.app.render();
    this.opts.onSelect?.(this.data.plays[k].name);
  }

  onKey(e) {
    return this.app.onKey(e);
  }

  render() {
    this.app.render();
  }
}
