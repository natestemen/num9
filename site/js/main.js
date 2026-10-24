// The page: loads the engine, then shows one screen at a time (chosen by the
// address's #fragment) and sends it the keyboard.

import { Engine } from "./engine.js";
import { geometry } from "./game.js";
import { Pool } from "./workers.js";
import { PlayApp } from "./play.js";
import { ChallengeApp, loadBests } from "./challenges.js";
import { SandboxApp } from "./sandbox.js";
import { RecordsApp } from "./records.js";
import { CompanionApp } from "./companion.js";

const screens = ["play", "companion", "challenges", "records", "about"];
const apps = {};
// The screens, for poking at from the browser console.
window.num9 = { apps };
let active = null;
let engine;
let pool;
let records;

const help = document.getElementById("help");
const openHelp = () => (help.hidden = false);
help.addEventListener("click", () => (help.hidden = true));

function route() {
  const [name, ...rest] = location.hash.replace(/^#/, "").split("/");
  return { name: screens.includes(name) ? name : "play", arg: rest.join("/") || undefined };
}

async function create(name, arg) {
  const section = document.getElementById(name);
  if (name === "play") {
    const app = new PlayApp(section, engine, pool, {
      onHelp: openHelp,
      onDeal: (hash) => history.replaceState(null, "", `#${hash}`),
    });
    app.route(arg);
    return app;
  }
  if (name === "companion") {
    return new CompanionApp(section, engine, pool, { onHelp: openHelp });
  }
  if (name === "challenges") {
    const { challenges } = await (await fetch("data/challenges.json")).json();
    const picker = section.querySelector(".decks");
    // Each challenge: its name and target, and a tick once you've reached it.
    const mark = (deck) => {
      history.replaceState(null, "", `#challenges/${deck.id}`);
      const bests = loadBests();
      picker.innerHTML = challenges
        .map((d, k) => {
          const best = bests[d.id];
          const done = best != null && best >= d.target ? " ✓" : "";
          return `<button data-k="${k}" class="${d === deck ? "on" : ""}">${d.name} · ${d.target}${done}</button>`;
        })
        .join("");
    };
    // With a mouse and room, every tile starts on the board to be arranged
    // freely; on phones, tiles are picked from a tray and placed in turn.
    const free = matchMedia("(pointer: fine) and (min-width: 861px)").matches;
    const App = free ? SandboxApp : ChallengeApp;
    const app = new App(section, engine, pool, { onHelp: openHelp, onDeck: mark, challenges });
    picker.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-k]");
      if (b) app.choose(challenges[b.dataset.k]);
    });
    app.route(arg);
    mark(app.deck);
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
  else apps[name]?.route?.(arg); // a link followed while the screen is open
  active = apps[name];
  active?.render();
}

document.addEventListener("keydown", (e) => {
  if (e.target.closest?.("input, textarea, select")) return;
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
