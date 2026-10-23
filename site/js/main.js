// The page: loads the engine, then shows one screen at a time (chosen by the
// address's #fragment) and sends it the keyboard.

import { Engine } from "./engine.js";
import { geometry } from "./game.js";
import { Pool } from "./workers.js";
import { PlayApp } from "./play.js";
import { PuzzleApp, DECKS } from "./puzzles.js";
import { RecordsApp } from "./records.js";

const screens = ["play", "puzzles", "records", "about"];
const apps = {};
let active = null;
let engine;
let pool;
let records;

const help = document.getElementById("help");
const openHelp = () => (help.hidden = false);
help.addEventListener("click", () => (help.hidden = true));

function route() {
  const [name, arg] = location.hash.replace(/^#/, "").split("/");
  return { name: screens.includes(name) ? name : "play", arg };
}

async function create(name, arg) {
  const section = document.getElementById(name);
  if (name === "play") {
    const seed = /^\d+$/.test(arg ?? "") ? Number(arg) >>> 0 : undefined;
    return new PlayApp(section, engine, pool, {
      seed,
      onHelp: openHelp,
      onDeal: (s) => history.replaceState(null, "", `#play/${s}`),
    });
  }
  if (name === "puzzles") {
    const picker = section.querySelector(".decks");
    picker.innerHTML = DECKS.map((d, k) => `<button data-k="${k}">${d.name}</button>`).join("");
    const mark = (deck) => {
      for (const b of picker.querySelectorAll("button")) b.classList.toggle("on", DECKS[b.dataset.k] === deck);
    };
    const app = new PuzzleApp(section, engine, pool, { onHelp: openHelp, onDeck: mark });
    picker.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-k]");
      if (b) app.chooseDeck(DECKS[b.dataset.k]);
    });
    mark(DECKS[0]);
    return app;
  }
  if (name === "records") {
    records ??= await (await fetch("data/records.json")).json();
    return new RecordsApp(section, engine, pool, records, {
      name: arg && decodeURIComponent(arg),
      onHelp: openHelp,
      onSelect: (name) => history.replaceState(null, "", `#records/${encodeURIComponent(name)}`),
    });
  }
  return null;
}

async function show() {
  const { name, arg } = route();
  for (const s of screens) document.getElementById(s).hidden = s !== name;
  for (const a of document.querySelectorAll("nav a")) a.classList.toggle("on", a.getAttribute("href") === `#${name}`);
  if (!(name in apps)) apps[name] = await create(name, arg);
  active = apps[name];
  active?.render();
}

document.addEventListener("keydown", (e) => {
  if (e.altKey || (e.metaKey && e.key !== "z") || (e.ctrlKey && e.key !== "z")) return;
  if (!help.hidden) {
    help.hidden = true;
    e.preventDefault();
    return;
  }
  if (active?.onKey(e)) e.preventDefault();
});

async function start() {
  const status = document.getElementById("loading");
  try {
    engine = await Engine.load();
  } catch (err) {
    status.textContent = `Couldn't load the engine (${err}). This page needs a browser with WebAssembly.`;
    return;
  }
  geometry.shapes = engine.shapes();
  Object.assign(geometry, engine.size());
  pool = Pool.default();
  status.remove();
  window.addEventListener("hashchange", show);
  await show();
}

start();
