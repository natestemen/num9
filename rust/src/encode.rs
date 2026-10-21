//! Boards as stacks of bit planes, for machine learning.
//!
//! The game doesn't care where on the table the tiles are, so a board is
//! encoded as a `CROP` × `CROP` window around the pieces. Each level gets two
//! planes: which cells are occupied, and which occupied cells border a
//! *different* piece on the same level (where a tile above could rest on two
//! pieces).

use crate::board::{Board, Move, DEPTH, HEIGHT, WIDTH};
use crate::piece::shape;

/// Side of the encoded window, in cells.
pub const CROP: usize = 32;
/// Planes per board: (occupied, piece boundary) for each level.
pub const PLANES: usize = 2 * DEPTH;
/// Bytes per encoded board.
pub const BOARD_BYTES: usize = PLANES * CROP * CROP;

/// Top-left corner of the window, centred on the ground level's pieces.
pub fn crop_origin(board: &Board) -> (usize, usize) {
    let (ci, cj) = match board.layer_bbox(0) {
        Some((a, b, c, d)) => ((a + b).div_ceil(2), (c + d).div_ceil(2)),
        None => (HEIGHT / 2, WIDTH / 2),
    };
    (
        ci.saturating_sub(CROP / 2).min(HEIGHT - CROP),
        cj.saturating_sub(CROP / 2).min(WIDTH - CROP),
    )
}

/// Append `board`'s planes (`BOARD_BYTES` bytes of 0/1, plane-major) for
/// the window at `origin`. Cells outside the window are dropped.
pub fn encode(board: &Board, origin: (usize, usize), out: &mut Vec<u8>) {
    // Piece number (1-based) per cell of the window, per level.
    let mut ids = [[0u8; CROP * CROP]; DEPTH];
    for (k, p) in board.pieces().iter().enumerate() {
        for (r, c) in shape(p.value, p.mv.rot).cells() {
            let (i, j) = (p.mv.i + r, p.mv.j + c);
            if let (Some(a), Some(b)) = (i.checked_sub(origin.0), j.checked_sub(origin.1)) {
                if a < CROP && b < CROP {
                    ids[p.mv.layer][a * CROP + b] = k as u8 + 1;
                }
            }
        }
    }
    let start = out.len();
    out.resize(start + BOARD_BYTES, 0);
    let planes = &mut out[start..];
    for (l, grid) in ids.iter().enumerate() {
        let (occupied, boundary) = planes[2 * l * CROP * CROP..].split_at_mut(CROP * CROP);
        for a in 0..CROP {
            for b in 0..CROP {
                let id = grid[a * CROP + b];
                if id == 0 {
                    continue;
                }
                occupied[a * CROP + b] = 1;
                let differs = |x: usize, y: usize| {
                    let other = grid[x * CROP + y];
                    other != 0 && other != id
                };
                let edge = (a > 0 && differs(a - 1, b))
                    || (a + 1 < CROP && differs(a + 1, b))
                    || (b > 0 && differs(a, b - 1))
                    || (b + 1 < CROP && differs(a, b + 1));
                boundary[a * CROP + b] = edge as u8;
            }
        }
    }
}

/// The distinct placements of `value` (pieces 0 and 8 look the same under
/// rotations r and r + 2; only the first is kept).
pub fn distinct_moves(board: &Board, value: u8) -> Vec<Move> {
    let half_turn = shape(value, 0).rows == shape(value, 2).rows;
    let mut moves = board.find_valid_moves(value);
    if half_turn {
        moves.retain(|m| m.rot < 2);
    }
    moves
}

/// Encode the board after each of `moves` (all in the current board's
/// window, so they line up), appending to `out`.
pub fn encode_afterstates(board: &Board, value: u8, moves: &[Move], out: &mut Vec<u8>) {
    let origin = crop_origin(board);
    let mut b = board.clone();
    out.reserve(moves.len() * BOARD_BYTES);
    for &mv in moves {
        b.place(value, mv).expect("legal move");
        encode(&b, origin, out);
        b.undo();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::shuffled_tiles;
    use crate::rng::Rng;
    use crate::strategies::Strategy;

    fn sample_board() -> Board {
        let mut rng = Rng::new(8);
        let mut b = Board::new();
        for v in shuffled_tiles(&mut rng).into_iter().take(12) {
            let mv = Strategy::EdgesThenUp.choose(&b, v, &mut rng);
            b.place(v, mv).unwrap();
        }
        b
    }

    #[test]
    fn occupancy_matches_the_board() {
        let b = sample_board();
        let origin = crop_origin(&b);
        let mut out = Vec::new();
        encode(&b, origin, &mut out);
        assert_eq!(out.len(), BOARD_BYTES);
        for l in 0..DEPTH {
            let plane = &out[2 * l * CROP * CROP..(2 * l + 1) * CROP * CROP];
            let ones = plane.iter().filter(|&&x| x == 1).count();
            assert_eq!(ones as u32, b.layer_fill(l), "level {l} fits in the window");
            for a in 0..CROP {
                for c in 0..CROP {
                    let occupied = b.is_occupied(l, origin.0 + a, origin.1 + c);
                    assert_eq!(plane[a * CROP + c] == 1, occupied);
                }
            }
        }
    }

    #[test]
    fn boundaries_mark_cells_next_to_other_pieces() {
        let mut b = Board::new();
        b.place(9, Move::new(0, 80, 60, 0)).unwrap();
        b.place(9, Move::new(0, 80, 63, 0)).unwrap();
        let origin = crop_origin(&b);
        let mut out = Vec::new();
        encode(&b, origin, &mut out);
        let boundary = &out[CROP * CROP..2 * CROP * CROP];
        let marked: Vec<(usize, usize)> = (0..CROP * CROP)
            .filter(|&k| boundary[k] == 1)
            .map(|k| (origin.0 + k / CROP, origin.1 + k % CROP))
            .collect();
        // The two 9s meet along columns 62|63 in rows 80 and 81.
        assert_eq!(marked, vec![(80, 62), (80, 63), (81, 62), (81, 63)]);
    }

    #[test]
    fn afterstates_follow_the_moves_and_leave_the_board_alone() {
        let b = sample_board();
        let moves = distinct_moves(&b, 7);
        let mut out = Vec::new();
        encode_afterstates(&b, 7, &moves, &mut out);
        assert_eq!(out.len(), moves.len() * BOARD_BYTES);
        let origin = crop_origin(&b);
        for (k, &mv) in moves.iter().enumerate().step_by(7) {
            let mut after = b.clone();
            after.place(7, mv).unwrap();
            let mut one = Vec::new();
            encode(&after, origin, &mut one);
            assert_eq!(&out[k * BOARD_BYTES..(k + 1) * BOARD_BYTES], &one[..]);
        }
        assert_eq!(distinct_moves(&b, 8).len() * 2, b.find_valid_moves(8).len());
    }
}
