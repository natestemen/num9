//! The num9 engine for the website, as a WebAssembly module with a plain C
//! interface (no bindings generator): numbers go in and out as arguments,
//! anything longer through two byte buffers. JavaScript writes inputs into
//! the buffer `io_input(len)` returns, and reads outputs from `io_output()`
//! (`io_output_len()` bytes).
//!
//! Each instance (the page, and every web worker) holds one board. Moves are
//! passed as four bytes: level, row, column, rotation.
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p num9-web
//! ```

use std::cell::RefCell;
use std::sync::atomic::AtomicBool;

use num9_core::exact::{self, Memo};
use num9_core::game::play_tiles;
use num9_core::piece::shape;
use num9_core::rollout::{candidate_moves, rollout_totals, MonteCarlo};
use num9_core::solver::Nested;
use num9_core::{Board, Move, Rng, Strategy};

thread_local! {
    static BOARD: RefCell<Board> = RefCell::new(Board::new());
    static INPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// The exact solver for a puzzle: its deck and what it has solved.
    static EXACT: RefCell<Option<([u8; 10], Memo)>> = const { RefCell::new(None) };
}

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
extern "C" {
    /// Called by `analyze` with every better play found (moves as bytes).
    fn improved(score: i32, moves: *const u8, len: usize);
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn improved(_score: i32, _moves: *const u8, _len: usize) {}

fn input() -> Vec<u8> {
    INPUT.with(|b| b.borrow().clone())
}

fn output(bytes: Vec<u8>) -> usize {
    let n = bytes.len();
    OUTPUT.with(|b| *b.borrow_mut() = bytes);
    n
}

fn encode(moves: &[Move]) -> Vec<u8> {
    moves
        .iter()
        .flat_map(|m| [m.layer as u8, m.i as u8, m.j as u8, m.rot as u8])
        .collect()
}

fn decode(bytes: &[u8]) -> Vec<Move> {
    bytes
        .chunks_exact(4)
        .map(|b| Move::new(b[0] as usize, b[1] as usize, b[2] as usize, b[3] as usize))
        .collect()
}

/// Room for `len` bytes of input; returns where to write them.
#[no_mangle]
pub extern "C" fn io_input(len: usize) -> *mut u8 {
    INPUT.with(|b| {
        let mut b = b.borrow_mut();
        b.resize(len, 0);
        b.as_mut_ptr()
    })
}

#[no_mangle]
pub extern "C" fn io_output() -> *const u8 {
    OUTPUT.with(|b| b.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn io_output_len() -> usize {
    OUTPUT.with(|b| b.borrow().len())
}

/// Every tile's cells: for each value and rotation, the cell count and then
/// (row, column) pairs.
#[no_mangle]
pub extern "C" fn shapes() -> usize {
    let mut out = Vec::new();
    for v in 0..10u8 {
        for rot in 0..4 {
            let cells: Vec<(usize, usize)> = shape(v, rot).cells().collect();
            out.push(cells.len() as u8);
            out.extend(cells.iter().flat_map(|&(r, c)| [r as u8, c as u8]));
        }
    }
    output(out)
}

/// The board's size: rows × 1000 + columns.
#[no_mangle]
pub extern "C" fn board_size() -> u32 {
    (num9_core::HEIGHT * 1000 + num9_core::WIDTH) as u32
}

/// Where the first tile goes (as a move in the output).
#[no_mangle]
pub extern "C" fn center_move() -> usize {
    output(encode(&[Board::center_move()]))
}

#[no_mangle]
pub extern "C" fn board_reset() {
    BOARD.with(|b| *b.borrow_mut() = Board::new());
}

fn place(board: &mut Board, value: u8, mv: Move) -> bool {
    let legal = if board.pieces().is_empty() {
        mv == Board::center_move()
    } else {
        board.is_valid_move(value, mv)
    };
    legal && board.place(value, mv).is_ok()
}

/// Place a tile if the move is legal; returns 1 if it was.
#[no_mangle]
pub extern "C" fn board_place(value: u8, layer: u8, i: u8, j: u8, rot: u8) -> i32 {
    let mv = Move::new(layer as usize, i as usize, j as usize, rot as usize);
    BOARD.with(|b| place(&mut b.borrow_mut(), value, mv)) as i32
}

/// Set the board to the tiles in the input (five bytes each: value, then
/// the move); returns 1 if every move was legal.
#[no_mangle]
pub extern "C" fn board_load() -> i32 {
    let bytes = input();
    BOARD.with(|b| {
        let mut b = b.borrow_mut();
        *b = Board::new();
        bytes
            .chunks_exact(5)
            .all(|t| place(&mut b, t[0], decode(&t[1..])[0])) as i32
    })
}

#[no_mangle]
pub extern "C" fn board_undo() {
    BOARD.with(|b| {
        b.borrow_mut().undo();
    });
}

#[no_mangle]
pub extern "C" fn board_score() -> i32 {
    BOARD.with(|b| b.borrow().score() as i32)
}

/// Every legal move of `value`; returns how many.
#[no_mangle]
pub extern "C" fn valid_moves(value: u8) -> usize {
    let moves = BOARD.with(|b| b.borrow().find_valid_moves(value));
    output(encode(&moves)) / 4
}

/// The moves worth simulating for a hint (see `candidate_moves`).
#[no_mangle]
pub extern "C" fn hint_candidates(value: u8, per_layer: usize) -> usize {
    let picks = BOARD.with(|b| {
        let b = b.borrow();
        let moves = b.find_valid_moves(value);
        candidate_moves(&b, value, &moves, per_layer)
            .into_iter()
            .map(|k| moves[k])
            .collect::<Vec<_>>()
    });
    output(encode(&picks)) / 4
}

/// Total final score over `rollouts` simulated games for each candidate
/// move in the input, as little-endian i32s (see `rollout_totals`).
#[no_mangle]
pub extern "C" fn rollouts(value: u8, seed: u32, rollouts: usize) -> usize {
    let candidates = decode(&input());
    let mc = MonteCarlo {
        rollouts,
        seed: seed as u64,
        ..MonteCarlo::default()
    };
    let picks: Vec<usize> = (0..candidates.len()).collect();
    let totals = BOARD.with(|b| rollout_totals(&b.borrow(), value, &candidates, &picks, &mc));
    output(
        totals
            .iter()
            .flat_map(|&t| (t as i32).to_le_bytes())
            .collect(),
    )
}

/// The engine (Monte Carlo) playing the tiles in the input, in order;
/// returns its score, with its moves in the output.
#[no_mangle]
pub extern "C" fn engine_play(seed: u32) -> i32 {
    let tiles = input();
    let ai = Strategy::MonteCarlo(MonteCarlo::default());
    let board = play_tiles(&ai, &tiles, &mut Rng::new(seed as u64));
    let moves: Vec<Move> = board.pieces().iter().map(|p| p.mv).collect();
    output(encode(&moves));
    board.score() as i32
}

/// Search for the best play of the tiles in the input (nested Monte Carlo
/// search), reporting each improvement through `improved`; returns the best
/// score, with its moves in the output.
#[no_mangle]
pub extern "C" fn analyze(level: usize, per_layer: usize, seed: u32) -> i32 {
    let tiles = input();
    let nested = Nested {
        level,
        per_layer,
        threads: 1,
        ..Nested::default()
    };
    let mut best = i64::MIN;
    let sol = nested.solve(
        &tiles,
        seed as u64,
        &AtomicBool::new(false),
        |_, partial| {
            if partial.score > best {
                best = partial.score;
                let bytes = encode(&partial.moves);
                unsafe { improved(partial.score as i32, bytes.as_ptr(), bytes.len()) };
            }
        },
    );
    output(encode(&sol.moves));
    sol.score as i32
}

/// Start a puzzle: the input is the deck (ten counts) followed by positions
/// solved in advance (see `Memo::export`), possibly none.
#[no_mangle]
pub extern "C" fn exact_init() {
    let bytes = input();
    let mut deck = [0u8; 10];
    deck.copy_from_slice(&bytes[..10]);
    let memo = Memo::new();
    memo.import(&bytes[10..]);
    EXACT.with(|e| *e.borrow_mut() = Some((deck, memo)));
}

/// The deck minus what is on the board (and minus `drawn`, if below 10).
fn left_after(deck: &[u8; 10], board: &Board, drawn: u8) -> [u8; 10] {
    let mut left = *deck;
    for p in board.pieces() {
        left[p.value as usize] -= 1;
    }
    if drawn < 10 {
        left[drawn as usize] -= 1;
    }
    left
}

/// The best expected final score from the board as it is, before the next
/// tile is drawn.
#[no_mangle]
pub extern "C" fn exact_value() -> f64 {
    EXACT.with(|e| {
        let e = e.borrow();
        let (deck, memo) = e.as_ref().expect("exact_init first");
        BOARD.with(|b| {
            let mut b = b.borrow_mut();
            let mut left = left_after(deck, &b, 10);
            exact::solve(memo, &mut b, &mut left).expected
        })
    })
}

/// Each distinct placement of `value` (just drawn) and the best expected
/// final score after it: twelve bytes each, the move then an f64.
#[no_mangle]
pub extern "C" fn exact_placements(value: u8) -> usize {
    let rated = EXACT.with(|e| {
        let e = e.borrow();
        let (deck, memo) = e.as_ref().expect("exact_init first");
        BOARD.with(|b| {
            let mut b = b.borrow_mut();
            let mut left = left_after(deck, &b, value);
            exact::placements(memo, &mut b, value, &mut left)
        })
    });
    let mut out = Vec::with_capacity(rated.len() * 12);
    for (m, node) in &rated {
        out.extend(encode(&[*m]));
        out.extend(node.expected.to_le_bytes());
    }
    output(out) / 12
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out() -> Vec<u8> {
        OUTPUT.with(|b| b.borrow().clone())
    }

    fn write(bytes: &[u8]) {
        let p = io_input(bytes.len());
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()) };
    }

    #[test]
    fn plays_a_game_through_the_interface() {
        board_reset();
        center_move();
        let c = out();
        assert_eq!(board_place(5, c[0], c[1], c[2], c[3]), 1);
        assert_eq!(board_place(5, c[0], c[1], c[2], c[3]), 0, "occupied");
        let n = valid_moves(7);
        let moves = out();
        assert_eq!(moves.len(), n * 4);
        let m = &moves[..4];
        assert_eq!(board_place(7, m[0], m[1], m[2], m[3]), 1);
        // Reload the same position from bytes.
        let mut pieces = vec![5];
        pieces.extend(&c);
        pieces.push(7);
        pieces.extend(m);
        write(&pieces);
        assert_eq!(board_load(), 1);
        assert_eq!(BOARD.with(|b| b.borrow().pieces().len()), 2);
        // Hint candidates and their rollouts.
        let k = hint_candidates(3, 4);
        write(&out());
        let totals = rollouts(3, 0, 4);
        assert_eq!(totals, k * 4);
        board_undo();
        assert_eq!(BOARD.with(|b| b.borrow().pieces().len()), 1);
    }

    #[test]
    fn engine_and_analysis_play_legal_games() {
        let tiles = num9_core::game::shuffled_tiles(&mut Rng::new(1));
        write(&tiles);
        let score = engine_play(1);
        let moves = decode(&out());
        assert_eq!(num9_core::solver::verify(&tiles, &moves), Ok(score as i64));
        write(&tiles);
        let score = analyze(1, 2, 0);
        let moves = decode(&out());
        assert_eq!(num9_core::solver::verify(&tiles, &moves), Ok(score as i64));
    }

    #[test]
    fn exact_values_match_the_solver() {
        let deck = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0];
        write(&deck);
        exact_init();
        board_reset();
        assert!((exact_value() - 7.0 / 3.0).abs() < 1e-9);
        center_move();
        let c = out();
        board_place(2, c[0], c[1], c[2], c[3]);
        // Before a draw, a position is worth the average over the tiles
        // that could come of their best placement.
        let before = exact_value();
        let mut sum = 0.0;
        for v in [0, 1, 3] {
            let n = exact_placements(v);
            let rated = out();
            assert_eq!(rated.len(), n * 12);
            sum += rated
                .chunks_exact(12)
                .map(|r| f64::from_le_bytes(r[4..].try_into().unwrap()))
                .fold(f64::MIN, f64::max);
        }
        assert!(
            (before - sum / 3.0).abs() < 1e-9,
            "{before} vs {}",
            sum / 3.0
        );
    }
}
