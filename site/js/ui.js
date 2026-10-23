// Layout shared by every screen, after the terminal game's: the board on the
// left in a bordered box, a side panel of boxes, and a bar of keys below
// (which can also be clicked or tapped).

import * as theme from "./theme.js";
import { geometry, size } from "./game.js";

export const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export const span = (text, style = "") => `<span style="${style}">${esc(text)}</span>`;
export const muted = (text) => span(text, `color:${theme.MUTED}`);
export const bold = (text, color = "") => span(text, `font-weight:700;${color ? `color:${color}` : ""}`);

export function box(title, body, cls = "") {
  return `<div class="box ${cls}"><div class="box-title">${esc(title)}</div><div class="box-body">${body}</div></div>`;
}

/// Build the board, panel and key bar inside `section`.
export function stage(section) {
  section.insertAdjacentHTML(
    "beforeend",
    `<div class="stage">
      <div class="box board-box">
        <div class="box-title board-title"></div>
        <div class="canvas-wrap"><canvas></canvas></div>
        <div class="zoom"><button data-centre title="Re-centre on the stack (c)">⌖</button><button data-zoom="-4" title="Zoom out">−</button><button data-zoom="4" title="Zoom in">+</button></div>
        <div class="popup" hidden></div>
      </div>
      <div class="status"></div>
      <aside class="panel"></aside>
    </div>
    <div class="touchbar"></div>
    <div class="keys"></div>`,
  );
  const q = (s) => section.querySelector(s);
  return {
    title: q(".board-title"),
    canvas: q("canvas"),
    popup: q(".popup"),
    panel: q(".panel"),
    keys: q(".keys"),
    zoom: q(".zoom"),
    status: q(".status"),
    touchbar: q(".touchbar"),
  };
}

/// Big buttons for touch screens: [label, key to send, class]; clicking one
/// sends its key to `onKey`.
export function touchBar(el, buttons, onKey) {
  const sig = JSON.stringify(buttons);
  if (el.dataset.sig === sig) return;
  el.dataset.sig = sig;
  el.innerHTML = buttons
    .map(([label, send, cls = ""]) => `<button class="${cls}" data-send="${esc(send)}">${esc(label)}</button>`)
    .join("");
  el.onclick = (e) => {
    const b = e.target.closest("button[data-send]");
    if (b) onKey({ key: b.dataset.send, shiftKey: false, ctrlKey: false, metaKey: false, preventDefault() {} });
  };
}

/// The key bar: [[key label, what it does, key to send]]; clicking a chip
/// sends that key to `onKey`.
export function keyBar(el, keys, onKey) {
  el.innerHTML = keys
    .map(([k, what, send]) => `<button class="key" data-send="${esc(send ?? k)}"><b>${esc(k)}</b> ${esc(what)}</button>`)
    .join("");
  el.onclick = (e) => {
    const b = e.target.closest("button[data-send]");
    if (b) onKey({ key: b.dataset.send, shiftKey: false, ctrlKey: false, metaKey: false, preventDefault() {} });
  };
}

/// A tile drawn as a little grid of blocks.
export function tileShape(value, rot, cls = "") {
  const cs = geometry.shapes[value][rot];
  const { h, w } = size(value, rot);
  const filled = new Set(cs.map(([r, c]) => `${r},${c}`));
  let html = `<div class="shape ${cls}" style="grid-template-columns:repeat(${w},1fr)">`;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      html += filled.has(`${r},${c}`) ? `<i style="background:${theme.valueCss(value)}"></i>` : "<i></i>";
    }
  }
  return html + "</div>";
}

/// "Level 2  12 pts" lines, top level first.
export function levelLines(levels) {
  return levels
    .slice()
    .reverse()
    .map(
      ([l, n, pts]) =>
        `<div>${bold(`Level ${l}`, l === 0 ? theme.MUTED : theme.levelCss(l))} ${esc(String(n).padStart(2))} tile${n === 1 ? " " : "s"}  ${muted(`${String(pts).padStart(3)} pts`)}</div>`,
    )
    .join("");
}

/// The tiles still to come, as dots.
export function bagLines(counts, copies = 2) {
  const line = (from, to) =>
    Array.from({ length: to - from }, (_, k) => from + k)
      .map((v) => {
        const n = counts[v];
        const color = n === 0 ? theme.MUTED : theme.valueCss(v);
        const dots = Array.from({ length: copies }, (_, k) => (k < n ? "●" : "○")).join("");
        return span(`${v}${dots}`, `color:${color};font-weight:700`);
      })
      .join("  ");
  return `<div>${line(0, 5)}</div><div>${line(5, 10)}</div>`;
}
