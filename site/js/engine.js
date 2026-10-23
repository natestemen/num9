// The num9 engine (Rust, compiled to WebAssembly; see rust/web). Each
// instance holds one board. Moves are {layer, i, j, rot}.

const WASM = new URL("../num9.wasm", import.meta.url);

export function decodeMoves(bytes) {
  const out = [];
  for (let k = 0; k + 3 < bytes.length; k += 4) {
    out.push({ layer: bytes[k], i: bytes[k + 1], j: bytes[k + 2], rot: bytes[k + 3] });
  }
  return out;
}

export function encodeMoves(moves) {
  return Uint8Array.from(moves.flatMap((m) => [m.layer, m.i, m.j, m.rot]));
}

/// Pieces ({value, move}) as the five-bytes-each format `board_load` reads.
export function encodePieces(pieces) {
  return Uint8Array.from(pieces.flatMap((p) => [p.value, p.layer, p.i, p.j, p.rot]));
}

export class Engine {
  constructor(exports) {
    this.x = exports;
  }

  /// Load the module. `onImproved(score, moves)` hears from `analyze`.
  static async load(onImproved = () => {}) {
    let engine;
    const imports = {
      env: {
        improved: (score, ptr, len) => onImproved(score, decodeMoves(engine.bytes(ptr, len))),
      },
    };
    let module;
    try {
      module = await WebAssembly.instantiateStreaming(fetch(WASM), imports);
    } catch {
      // Servers that send the wrong content type.
      const buf = await (await fetch(WASM)).arrayBuffer();
      module = await WebAssembly.instantiate(buf, imports);
    }
    engine = new Engine(module.instance.exports);
    return engine;
  }

  bytes(ptr, len) {
    return new Uint8Array(this.x.memory.buffer, ptr, len).slice();
  }

  write(bytes) {
    const ptr = this.x.io_input(bytes.length);
    new Uint8Array(this.x.memory.buffer, ptr, bytes.length).set(bytes);
  }

  output() {
    return this.bytes(this.x.io_output(), this.x.io_output_len());
  }

  /// shapes[value][rot] = [[row, col], ...]
  shapes() {
    this.x.shapes();
    const b = this.output();
    const out = [];
    let k = 0;
    for (let v = 0; v < 10; v++) {
      out.push([]);
      for (let rot = 0; rot < 4; rot++) {
        const n = b[k++];
        const cells = [];
        for (let c = 0; c < n; c++, k += 2) cells.push([b[k], b[k + 1]]);
        out[v].push(cells);
      }
    }
    return out;
  }

  size() {
    const s = this.x.board_size();
    return { height: Math.floor(s / 1000), width: s % 1000 };
  }

  deal(seed) {
    this.x.deal(seed >>> 0);
    return Array.from(this.output());
  }

  centerMove() {
    this.x.center_move();
    return decodeMoves(this.output())[0];
  }

  /// Set the board to `pieces`; false if a move was illegal.
  load(pieces) {
    this.write(encodePieces(pieces));
    return this.x.board_load() === 1;
  }

  validMoves(value) {
    this.x.valid_moves(value);
    return decodeMoves(this.output());
  }

  hintCandidates(value, perLayer) {
    this.x.hint_candidates(value, perLayer);
    return decodeMoves(this.output());
  }

  rollouts(value, seed, n, candidates) {
    this.write(encodeMoves(candidates));
    this.x.rollouts(value, seed >>> 0, n);
    return Array.from(new Int32Array(this.output().buffer));
  }

  play(tiles, seed) {
    this.write(Uint8Array.from(tiles));
    const score = this.x.engine_play(seed >>> 0);
    return { score, moves: decodeMoves(this.output()) };
  }

  analyze(tiles, level, perLayer, seed) {
    this.write(Uint8Array.from(tiles));
    const score = this.x.analyze(level, perLayer, seed >>> 0);
    return { score, moves: decodeMoves(this.output()) };
  }

  exactInit(deck, table) {
    const bytes = new Uint8Array(10 + table.length);
    bytes.set(deck);
    bytes.set(table, 10);
    this.write(bytes);
    this.x.exact_init();
  }

  exactValue() {
    return this.x.exact_value();
  }

  /// [{move, expected}] for each distinct placement of `value`.
  exactPlacements(value) {
    const n = this.x.exact_placements(value);
    const b = this.output();
    const view = new DataView(b.buffer);
    const out = [];
    for (let k = 0; k < n; k++) {
      const [move] = decodeMoves(b.subarray(12 * k, 12 * k + 4));
      out.push({ move, expected: view.getFloat64(12 * k + 4, true) });
    }
    return out;
  }
}
