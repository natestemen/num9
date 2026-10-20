//! Board state and move generation, mirroring `num9/board.py`.
//!
//! Each layer is stored as one `u128` bitmask per row (bit `j` = column `j`),
//! so overlap, adjacency and support checks are a handful of word operations.

use std::fmt;

use crate::piece::{shape, Shape};

pub const DEPTH: usize = 6;
pub const HEIGHT: usize = 160;
pub const WIDTH: usize = 120;
const COL_MASK: u128 = (1u128 << WIDTH) - 1;

/// A placement: piece's top-left corner at (`i`, `j`) on `layer`, rotated `rot`
/// clockwise quarter turns. Same order as the Python `(layer, i, j, rot)` tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Move {
    pub layer: usize,
    pub i: usize,
    pub j: usize,
    pub rot: usize,
}

impl Move {
    pub fn new(layer: usize, i: usize, j: usize, rot: usize) -> Move {
        Move { layer, i, j, rot }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placed {
    pub value: u8,
    pub mv: Move,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaceError {
    OutOfBounds,
    Overlap,
    BadValue,
}

impl fmt::Display for PlaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlaceError::OutOfBounds => write!(f, "placement extends outside the board"),
            PlaceError::Overlap => write!(f, "placement overlaps an existing piece"),
            PlaceError::BadValue => write!(f, "piece value must be in 0..=9"),
        }
    }
}

impl std::error::Error for PlaceError {}

/// Inclusive bounding box of the occupied cells of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BBox {
    min_i: usize,
    max_i: usize,
    min_j: usize,
    max_j: usize,
}

impl BBox {
    fn union(self, other: BBox) -> BBox {
        BBox {
            min_i: self.min_i.min(other.min_i),
            max_i: self.max_i.max(other.max_i),
            min_j: self.min_j.min(other.min_j),
            max_j: self.max_j.max(other.max_j),
        }
    }
}

#[derive(Clone)]
pub struct Board {
    occ: [[u128; HEIGHT]; DEPTH],
    bbox: [Option<BBox>; DEPTH],
    pieces: Vec<Placed>,
    score: i64,
}

impl Default for Board {
    fn default() -> Self {
        Board::new()
    }
}

impl Board {
    pub fn new() -> Board {
        Board {
            occ: [[0; HEIGHT]; DEPTH],
            bbox: [None; DEPTH],
            pieces: Vec::with_capacity(20),
            score: 0,
        }
    }

    /// Where the first piece goes (`Board.center_coords` plus rotation 0).
    pub fn center_move() -> Move {
        Move::new(0, HEIGHT / 2, WIDTH / 2, 0)
    }

    pub fn pieces(&self) -> &[Placed] {
        &self.pieces
    }

    pub fn score(&self) -> i64 {
        self.score
    }

    pub fn layer_is_empty(&self, layer: usize) -> bool {
        self.bbox[layer].is_none()
    }

    /// Number of occupied cells on a layer.
    pub fn layer_fill(&self, layer: usize) -> u32 {
        match self.bbox[layer] {
            None => 0,
            Some(b) => self.occ[layer][b.min_i..=b.max_i]
                .iter()
                .map(|r| r.count_ones())
                .sum(),
        }
    }

    /// Inclusive `(min_i, max_i, min_j, max_j)` of a layer's occupied cells.
    pub fn layer_bbox(&self, layer: usize) -> Option<(usize, usize, usize, usize)> {
        self.bbox[layer].map(|b| (b.min_i, b.max_i, b.min_j, b.max_j))
    }

    pub fn is_occupied(&self, layer: usize, i: usize, j: usize) -> bool {
        (self.occ[layer][i] >> j) & 1 == 1
    }

    #[inline]
    fn row_mask(s: &Shape, r: usize, j: usize) -> u128 {
        (s.rows[r] as u128) << j
    }

    /// Ring row `rr` positioned on the board (columns `j - 1 ..`), clipped to the board.
    #[inline]
    fn ring_mask(s: &Shape, rr: usize, j: usize) -> u128 {
        (((s.ring[rr] as u128) << j) >> 1) & COL_MASK
    }

    fn in_bounds(s: &Shape, mv: Move) -> bool {
        mv.layer < DEPTH && mv.i + s.h <= HEIGHT && mv.j + s.w <= WIDTH
    }

    fn overlaps(&self, s: &Shape, layer: usize, i: usize, j: usize) -> bool {
        let rows = &self.occ[layer];
        (0..s.h).any(|r| rows[i + r] & Self::row_mask(s, r, j) != 0)
    }

    /// Ring rows that fall on the board, as (board row, ring row index).
    fn ring_rows(s: &Shape, i: usize) -> impl Iterator<Item = (usize, usize)> {
        (0..s.h + 2).filter_map(move |rr| {
            let row = (i + rr).checked_sub(1)?;
            (row < HEIGHT).then_some((row, rr))
        })
    }

    fn touches(&self, s: &Shape, layer: usize, i: usize, j: usize) -> bool {
        let rows = &self.occ[layer];
        Self::ring_rows(s, i).any(|(row, rr)| rows[row] & Self::ring_mask(s, rr, j) != 0)
    }

    /// `Board.num_edges_touching`: occupied cells adjacent to the piece on its layer.
    pub fn num_edges_touching(&self, value: u8, mv: Move) -> u32 {
        let s = shape(value, mv.rot);
        let rows = &self.occ[mv.layer];
        Self::ring_rows(s, mv.i)
            .map(|(row, rr)| (rows[row] & Self::ring_mask(s, rr, mv.j)).count_ones())
            .sum()
    }

    /// Number of piece tiles that sit on occupied cells of the layer below.
    pub fn support_overlap(&self, value: u8, mv: Move) -> u32 {
        if mv.layer == 0 {
            return 0;
        }
        let s = shape(value, mv.rot);
        let below = &self.occ[mv.layer - 1];
        (0..s.h)
            .map(|r| (below[mv.i + r] & Self::row_mask(s, r, mv.j)).count_ones())
            .sum()
    }

    fn fully_supported(&self, s: &Shape, layer: usize, i: usize, j: usize) -> bool {
        let below = &self.occ[layer - 1];
        (0..s.h).all(|r| Self::row_mask(s, r, j) & !below[i + r] == 0)
    }

    /// Whether the piece rests on at least two distinct pieces of the layer below.
    fn spans_two_pieces(&self, s: &Shape, layer: usize, i: usize, j: usize) -> bool {
        let mut count = 0;
        for p in self.pieces.iter().filter(|p| p.mv.layer == layer - 1) {
            let ps = shape(p.value, p.mv.rot);
            let lo = i.max(p.mv.i);
            let hi = (i + s.h).min(p.mv.i + ps.h);
            let touches = (lo..hi).any(|row| {
                Self::row_mask(s, row - i, j) & Self::row_mask(ps, row - p.mv.i, p.mv.j) != 0
            });
            if touches {
                count += 1;
                if count >= 2 {
                    return true;
                }
            }
        }
        false
    }

    /// The legality test of `find_valid_moves`: `validate_touching` and
    /// `validate_supported` from `num9/board.py`.
    fn is_legal(&self, s: &Shape, layer: usize, i: usize, j: usize) -> bool {
        if !self.layer_is_empty(layer)
            && (self.overlaps(s, layer, i, j) || !self.touches(s, layer, i, j))
        {
            return false;
        }
        if layer > 0
            && !(self.fully_supported(s, layer, i, j) && self.spans_two_pieces(s, layer, i, j))
        {
            return false;
        }
        true
    }

    /// Whether `mv` is one of the moves `find_valid_moves(value)` returns.
    pub fn is_valid_move(&self, value: u8, mv: Move) -> bool {
        if value > 9 || mv.rot > 3 {
            return false;
        }
        if self.pieces.is_empty() {
            return mv == Self::center_move();
        }
        let s = shape(value, mv.rot);
        if !Self::in_bounds(s, mv) {
            return false;
        }
        if mv.layer > 0 && self.layer_is_empty(mv.layer - 1) {
            return false;
        }
        self.is_legal(s, mv.layer, mv.i, mv.j)
    }

    /// All legal placements of piece `value`, in the same order as
    /// `Board.find_valid_moves`: by layer, then rotation, then row, then column.
    ///
    /// Like the Python version this lists symmetric rotations separately, so
    /// e.g. pieces 0 and 8 yield each placement twice (rotations r and r + 2).
    pub fn find_valid_moves(&self, value: u8) -> Vec<Move> {
        let mut out = Vec::new();
        self.for_each_valid_move(value, |mv| out.push(mv));
        out
    }

    /// Calls `f` on every legal placement, in `find_valid_moves` order.
    pub fn for_each_valid_move(&self, value: u8, mut f: impl FnMut(Move)) {
        if self.pieces.is_empty() {
            f(Self::center_move());
            return;
        }
        for layer in 0..DEPTH {
            if layer > 0 && self.layer_is_empty(layer - 1) {
                break;
            }
            // Search around this layer's pieces, or the layer below's if it is empty.
            let bb = self.bbox[layer]
                .or(self.bbox[layer.saturating_sub(1)])
                .unwrap();
            for rot in 0..4 {
                let s = shape(value, rot);
                // Python searches `range(min - size, max + size)`; positions outside
                // the board are clipped (Python's behaviour there is undefined).
                let i_lo = bb.min_i.saturating_sub(s.h);
                let i_hi = (bb.max_i + s.h).min(HEIGHT - s.h + 1);
                let j_lo = bb.min_j.saturating_sub(s.w);
                let j_hi = (bb.max_j + s.w).min(WIDTH - s.w + 1);
                for i in i_lo..i_hi {
                    for j in j_lo..j_hi {
                        if self.is_legal(s, layer, i, j) {
                            f(Move::new(layer, i, j, rot));
                        }
                    }
                }
            }
        }
    }

    /// Put a piece on the board. Like `Board.place`, this only refuses
    /// overlapping (or off-board) placements; use `is_valid_move` for the
    /// full rules.
    pub fn place(&mut self, value: u8, mv: Move) -> Result<(), PlaceError> {
        if value > 9 {
            return Err(PlaceError::BadValue);
        }
        let s = shape(value, mv.rot);
        if !Self::in_bounds(s, mv) {
            return Err(PlaceError::OutOfBounds);
        }
        if self.overlaps(s, mv.layer, mv.i, mv.j) {
            return Err(PlaceError::Overlap);
        }
        let mut min_j = usize::MAX;
        let mut max_j = 0;
        for r in 0..s.h {
            self.occ[mv.layer][mv.i + r] |= Self::row_mask(s, r, mv.j);
            if s.rows[r] != 0 {
                min_j = min_j.min(mv.j + s.rows[r].trailing_zeros() as usize);
                max_j = max_j.max(mv.j + 7 - s.rows[r].leading_zeros() as usize);
            }
        }
        let piece_bb = BBox {
            min_i: mv.i,
            max_i: mv.i + s.h - 1,
            min_j,
            max_j,
        };
        self.bbox[mv.layer] = Some(match self.bbox[mv.layer] {
            Some(b) => b.union(piece_bb),
            None => piece_bb,
        });
        self.pieces.push(Placed { value, mv });
        self.score += value as i64 * mv.layer as i64;
        Ok(())
    }

    /// The board as `Board.board` would hold it: `values[layer][i][j]`, flattened,
    /// with each cell holding its piece's value (10 for piece 0) or 0 if empty.
    pub fn values(&self) -> Vec<i32> {
        let mut out = vec![0i32; DEPTH * HEIGHT * WIDTH];
        for p in &self.pieces {
            let s = shape(p.value, p.mv.rot);
            let v = if p.value == 0 { 10 } else { p.value as i32 };
            for (r, c) in s.cells() {
                out[(p.mv.layer * HEIGHT + p.mv.i + r) * WIDTH + p.mv.j + c] = v;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_move_is_centre() {
        let b = Board::new();
        assert_eq!(b.find_valid_moves(7), vec![Move::new(0, 80, 60, 0)]);
    }

    #[test]
    fn second_piece_must_touch_first() {
        let mut b = Board::new();
        b.place(9, Board::center_move()).unwrap();
        let moves = b.find_valid_moves(5);
        assert!(!moves.is_empty());
        for mv in &moves {
            assert_eq!(mv.layer, 0, "no layer 1 with a single piece below");
            assert!(b.num_edges_touching(5, *mv) > 0);
        }
    }

    #[test]
    fn overlap_is_rejected() {
        let mut b = Board::new();
        b.place(9, Board::center_move()).unwrap();
        assert_eq!(b.place(1, Board::center_move()), Err(PlaceError::Overlap));
        assert_eq!(b.pieces().len(), 1);
    }

    #[test]
    fn upper_layer_needs_two_pieces_below() {
        let mut b = Board::new();
        // Two 9s side by side (each 4x3 with a 2x2 lower-left block).
        b.place(9, Move::new(0, 80, 60, 0)).unwrap();
        b.place(9, Move::new(0, 80, 63, 0)).unwrap();
        let moves = b.find_valid_moves(1);
        let up: Vec<_> = moves.iter().filter(|m| m.layer == 1).collect();
        assert!(!up.is_empty());
        for mv in up {
            // Must straddle the seam between columns 62 and 63.
            let s = shape(1, mv.rot);
            assert!(mv.j <= 62 && mv.j + s.w > 63);
        }
    }

    #[test]
    fn score_counts_layers() {
        let mut b = Board::new();
        b.place(9, Move::new(0, 80, 60, 0)).unwrap();
        b.place(9, Move::new(0, 80, 63, 0)).unwrap();
        let mv = *b.find_valid_moves(1).iter().find(|m| m.layer == 1).unwrap();
        b.place(1, mv).unwrap();
        assert_eq!(b.score(), 1);
    }
}
