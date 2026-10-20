//! Board state and move generation, ported from the Python `num9/board.py`.
//!
//! Each layer is stored as one `u128` bitmask per row (bit `j` = column `j`),
//! so overlap, adjacency and support checks are a handful of word operations.

use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not, Shl, Shr};

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
    /// Occupied cells per layer.
    fill: [u32; DEPTH],
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
            fill: [0; DEPTH],
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
        self.fill[layer]
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
    /// `validate_supported` from the Python `num9/board.py`.
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
    ///
    /// Works a row at a time: for a given top row `i`, each check below yields
    /// a bitmask over all columns `j` at once (bit `j` = the piece placed at
    /// column `j`), built by shifting the board rows under each tile. This is
    /// equivalent to running `is_legal` at every position, which
    /// `is_valid_move` still does and the tests compare against.
    pub fn for_each_valid_move(&self, value: u8, mut f: impl FnMut(Move)) {
        if self.pieces.is_empty() {
            f(Self::center_move());
            return;
        }
        let mut single = Vec::new();
        let mut cur64 = [0u64; HEIGHT];
        let mut adj64 = [0u64; HEIGHT];
        let mut below64 = [0u64; HEIGHT];
        for layer in 0..DEPTH {
            if layer > 0 && self.layer_is_empty(layer - 1) {
                break;
            }
            // Search around this layer's pieces, or the layer below's if it is empty.
            let bb = self.bbox[layer]
                .or(self.bbox[layer.saturating_sub(1)])
                .unwrap();
            // Pieces are at most 4 cells across, so the scan reads columns
            // `bb.min_j - 5 ..= bb.max_j + 7` and rows `bb.min_i - 5 .. bb.max_i + 9`.
            // When those columns fit in 64 bits (always, in practice), scan a
            // 64-bit copy of them: u64 shifts are much cheaper than u128 ones.
            let base = bb.min_j.saturating_sub(5);
            let rows = bb.min_i.saturating_sub(5)..(bb.max_i + 9).min(HEIGHT);
            let occ = &self.occ[layer];
            if bb.max_j + 8 <= base + 64 {
                for r in rows {
                    cur64[r] = (occ[r] >> base) as u64;
                    adj64[r] = (Self::adjacent(occ, r) >> base) as u64;
                    if layer > 0 {
                        below64[r] = (self.occ[layer - 1][r] >> base) as u64;
                    }
                }
                let maps = (&cur64, &adj64, &below64);
                self.scan_layer(value, layer, bb, base, maps, &mut single, &mut f);
            } else {
                let mut adj = [0u128; HEIGHT];
                for r in rows {
                    adj[r] = Self::adjacent(occ, r);
                }
                let maps = (occ, &adj, &self.occ[layer.saturating_sub(1)]);
                self.scan_layer(value, layer, bb, 0, maps, &mut single, &mut f);
            }
        }
    }

    /// Row `r` of the cells orthogonally adjacent to an occupied cell of `occ`.
    #[inline]
    fn adjacent(occ: &[u128; HEIGHT], r: usize) -> u128 {
        let up = if r > 0 { occ[r - 1] } else { 0 };
        let down = if r + 1 < HEIGHT { occ[r + 1] } else { 0 };
        (up | down | (occ[r] << 1) | (occ[r] >> 1)) & COL_MASK
    }

    /// Emit the legal moves on one layer. `maps` holds this layer's occupied
    /// cells, the cells adjacent to them, and the layer below's occupied cells,
    /// all shifted right by `base` columns.
    #[allow(clippy::too_many_arguments)]
    fn scan_layer<W: Word>(
        &self,
        value: u8,
        layer: usize,
        bb: BBox,
        base: usize,
        maps: (&[W; HEIGHT], &[W; HEIGHT], &[W; HEIGHT]),
        single: &mut Vec<(usize, u128)>,
        f: &mut impl FnMut(Move),
    ) {
        let (cur, adj, below) = maps;
        let cur_empty = self.layer_is_empty(layer);
        for rot in 0..4 {
            let s = shape(value, rot);
            // Python searches `range(min - size, max + size)`; positions outside
            // the board are clipped (Python's behaviour there is undefined).
            let mut i_lo = bb.min_i.saturating_sub(s.h);
            let mut i_hi = (bb.max_i + s.h).min(HEIGHT - s.h + 1);
            let j_lo = bb.min_j.saturating_sub(s.w);
            let j_hi = (bb.max_j + s.w).min(WIDTH - s.w + 1);
            if j_lo >= j_hi {
                continue;
            }
            let window = W::from_u128((((1u128 << j_hi) - 1) & !((1u128 << j_lo) - 1)) >> base);
            if layer > 0 {
                // A fully supported piece lies within the layer below's rows.
                let below_bb = self.bbox[layer - 1].unwrap();
                i_lo = i_lo.max(below_bb.min_i);
                i_hi = i_hi.min((below_bb.max_i + 2).saturating_sub(s.h));
                self.single_piece_positions(s, layer - 1, single);
            }
            for i in i_lo..i_hi {
                let mut ok = window;
                if layer > 0 {
                    ok &= all_tiles(below, s, i);
                    for &(_, m) in single.iter().filter(|(si, _)| *si == i) {
                        ok &= !W::from_u128(m >> base);
                    }
                }
                if ok != W::ZERO && !cur_empty {
                    // With no tile on an occupied cell, the piece touches one
                    // exactly when some tile is adjacent to an occupied cell.
                    let (overlap, touch) = any_tile2(cur, adj, s, i);
                    ok &= !overlap & touch;
                }
                while ok != W::ZERO {
                    f(Move::new(
                        layer,
                        i,
                        ok.trailing_zeros() as usize + base,
                        rot,
                    ));
                    ok = ok.clear_lowest();
                }
            }
        }
    }

    /// Positions (as `(row, column mask)`) where `s` would rest entirely on a
    /// single piece of `layer`, i.e. fail the "two pieces below" rule.
    fn single_piece_positions(&self, s: &Shape, layer: usize, out: &mut Vec<(usize, u128)>) {
        out.clear();
        for p in self.pieces.iter().filter(|p| p.mv.layer == layer) {
            let ps = shape(p.value, p.mv.rot);
            if ps.h < s.h || ps.tiles < s.tiles {
                continue;
            }
            for i in p.mv.i..=p.mv.i + ps.h - s.h {
                let mut m = !0u128;
                for r in 0..s.h {
                    let x = Self::row_mask(ps, i + r - p.mv.i, p.mv.j);
                    let mut bits = s.rows[r];
                    while bits != 0 {
                        m &= x >> bits.trailing_zeros();
                        bits &= bits - 1;
                    }
                }
                if m != 0 {
                    out.push((i, m));
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
        self.fill[mv.layer] += s.tiles;
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

/// A row of cells as a bitset (bit `j` = column `j`, relative to some base).
trait Word:
    Copy
    + PartialEq
    + BitAnd<Output = Self>
    + BitAndAssign
    + BitOr<Output = Self>
    + BitOrAssign
    + Not<Output = Self>
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
{
    const ZERO: Self;
    const ONES: Self;
    /// Keep the low bits that fit (truncating).
    fn from_u128(x: u128) -> Self;
    fn trailing_zeros(self) -> u32;
    fn clear_lowest(self) -> Self;
}

macro_rules! impl_word {
    ($t:ty) => {
        impl Word for $t {
            const ZERO: Self = 0;
            const ONES: Self = !0;
            #[inline]
            fn from_u128(x: u128) -> Self {
                x as $t
            }
            #[inline]
            fn trailing_zeros(self) -> u32 {
                <$t>::trailing_zeros(self)
            }
            #[inline]
            fn clear_lowest(self) -> Self {
                self & self.wrapping_sub(1)
            }
        }
    };
}
impl_word!(u64);
impl_word!(u128);

/// Bit `j`: every tile of `s` at (`i`, `j`) lies on a set cell of `rows`.
#[inline]
fn all_tiles<W: Word>(rows: &[W; HEIGHT], s: &Shape, i: usize) -> W {
    let mut m = W::ONES;
    for r in 0..s.h {
        let x = rows[i + r];
        let mut bits = s.rows[r];
        while bits != 0 {
            m &= x >> bits.trailing_zeros();
            bits &= bits - 1;
        }
    }
    m
}

/// Bit `j` of each: some tile of `s` at (`i`, `j`) lies on a set cell of `a`
/// (resp. `b`).
#[inline]
fn any_tile2<W: Word>(a: &[W; HEIGHT], b: &[W; HEIGHT], s: &Shape, i: usize) -> (W, W) {
    let (mut ma, mut mb) = (W::ZERO, W::ZERO);
    for r in 0..s.h {
        let (x, y) = (a[i + r], b[i + r]);
        let mut bits = s.rows[r];
        while bits != 0 {
            let c = bits.trailing_zeros();
            ma |= x >> c;
            mb |= y >> c;
            bits &= bits - 1;
        }
    }
    (ma, mb)
}

/// Emoji rendering, as the Python `Board.__str__` printed it: every non-empty layer,
/// cropped to the occupied area plus a one-cell border.
impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(bb) = self.bbox.iter().flatten().copied().reduce(BBox::union) else {
            return Ok(());
        };
        let values = self.values();
        let top = (0..DEPTH).rev().find(|&l| !self.layer_is_empty(l)).unwrap();
        let width = bb.max_j - bb.min_j + 3;
        let mut lines = Vec::new();
        for layer in 0..=top {
            lines.push(format!(
                "{:-^w$}",
                format!(" LAYER {layer} "),
                w = 2 * width
            ));
            for i in bb.min_i - 1..=bb.max_i + 1 {
                let row: String = (bb.min_j - 1..=bb.max_j + 1)
                    .map(|j| match values[(layer * HEIGHT + i) * WIDTH + j] {
                        0 => "⬜️",
                        10 => "⬛️",
                        1 => "🟫",
                        2 => "🟧",
                        3 => "🟨",
                        4 => "🟩",
                        5 => "🔵",
                        6 => "🟦",
                        7 => "🟪",
                        8 => "💓",
                        _ => "🟥",
                    })
                    .collect();
                lines.push(row);
            }
        }
        write!(f, "{}", lines.join("\n"))
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

    /// Every legal move by checking each position on its own (`is_valid_move`),
    /// in `find_valid_moves` order.
    fn brute_force_moves(b: &Board, value: u8) -> Vec<Move> {
        if b.pieces().is_empty() {
            return b.find_valid_moves(value);
        }
        let mut out = Vec::new();
        for layer in 0..DEPTH {
            for rot in 0..4 {
                for i in 0..HEIGHT {
                    for j in 0..WIDTH {
                        let mv = Move::new(layer, i, j, rot);
                        if b.is_valid_move(value, mv) {
                            out.push(mv);
                        }
                    }
                }
            }
        }
        out
    }

    /// Wide boards (beyond the 64-column fast path) and ordinary ones: the
    /// row-at-a-time generator must agree with the per-position rule check.
    #[test]
    fn move_generation_matches_per_position_check() {
        use crate::rng::Rng;
        let mut rng = Rng::new(9);
        let mut wide_positions = 0;
        for game in 0..12 {
            let mut b = Board::new();
            for k in 0..20 {
                let value = rng.below(10) as u8;
                if let Some((_, _, lo, hi)) = b.layer_bbox(0) {
                    // Wider than 52 columns, the scan no longer fits in a u64.
                    wide_positions += usize::from(hi - lo + 1 > 52);
                }
                let moves = b.find_valid_moves(value);
                assert_eq!(moves, brute_force_moves(&b, value), "game {game} ply {k}");
                // Stretch the first 14 pieces sideways in half the games, then build up.
                let mv = if game % 2 == 0 && k < 14 {
                    let side = if k % 2 == 0 {
                        moves.iter().min_by_key(|m| m.j)
                    } else {
                        moves.iter().max_by_key(|m| m.j)
                    };
                    *side.unwrap()
                } else {
                    let top = moves.iter().map(|m| m.layer).max().unwrap();
                    let up: Vec<_> = moves.iter().filter(|m| m.layer == top).collect();
                    **rng.choose(&up)
                };
                b.place(value, mv).unwrap();
            }
        }
        assert!(
            wide_positions >= 10,
            "only {wide_positions} positions used the u128 path"
        );
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
