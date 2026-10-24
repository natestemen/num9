// Sharing: links to a deal or a challenge, and a short message with your
// score (one square per tile, coloured by the level it ended on) to paste
// to a friend. Phones get the system's share sheet; elsewhere it is copied.

/// A link to this site at `#hash`.
export function link(hash) {
  return `${location.origin}${location.pathname}#${hash}`;
}

const SQUARES = ["⬛", "🟦", "🟩", "🟨", "🟧", "🟥", "🟪", "🟫", "⬜"];

/// One square per tile, in the order placed: black on the ground, then
/// blue, green, yellow, orange, red, purple... going up.
export function squares(pieces) {
  return pieces.map((p) => SQUARES[Math.min(p.layer, SQUARES.length - 1)]).join("");
}

/// The deal as it goes in a link: twenty digits, the tiles in order.
export function dealCode(tiles) {
  return tiles.join("");
}

/// The tiles of a deal code, if it is two of each value 0-9.
export function parseDeal(code) {
  if (!/^\d{20}$/.test(code ?? "")) return null;
  const tiles = [...code].map(Number);
  const counts = Array(10).fill(0);
  for (const v of tiles) counts[v]++;
  return counts.every((c) => c === 2) ? tiles : null;
}

/// Share `text` and `url`: the share sheet on touch screens, otherwise the
/// clipboard. Resolves to "shared", "copied", "cancelled" or "failed".
export async function share(text, url) {
  const full = `${text}\n${url}`;
  if (navigator.share && matchMedia("(pointer: coarse)").matches) {
    try {
      await navigator.share({ text, url });
      return "shared";
    } catch (e) {
      if (e?.name === "AbortError") return "cancelled";
    }
  }
  try {
    await navigator.clipboard.writeText(full);
    return "copied";
  } catch {
    // No clipboard access (an old browser, or not allowed): the old way.
    const area = document.createElement("textarea");
    area.value = full;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok ? "copied" : "failed";
  }
}

/// What to tell the player after `share`.
export function said(outcome, what) {
  return {
    shared: [`Shared ${what}`, "good"],
    copied: [`Copied ${what}: paste it anywhere`, "good"],
    cancelled: ["Not shared", "info"],
    failed: ["Couldn't copy: your browser blocked it", "bad"],
  }[outcome];
}
