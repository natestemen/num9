//! Stranded support: cells of a level that no tile still to come could ever
//! rest on.
//!
//! A tile above the ground must lie entirely on the level below and on at
//! least two of its tiles, so the exposed cells of a level are the only
//! foundation for the next. Placing a tile on level L covers some of level
//! L-1's exposed cells, but it can also cut others off: leave pockets too
//! small or the wrong shape for any tile still to come, or sitting on a
//! single tile. Those cells can never be built on again. [`stranded`] counts
//! how many a move would cut off, so a strategy can avoid wasting support.
//!
//! "Could rest on" ignores the rule that a tile must touch another on its
//! level: later tiles could provide that contact.

use crate::board::{Board, Move};
use crate::piece::shape;

/// Tiles still to be drawn after `value` is placed: which values remain.
pub fn values_to_come(board: &Board, value: u8) -> [bool; 10] {
    let mut counts = [2i8; 10];
    for p in board.pieces() {
        counts[p.value as usize] -= 1;
    }
    counts[value as usize] -= 1;
    counts.map(|c| c > 0)
}

/// The level below `layer` and what's already on `layer`, over the level
/// below's bounding box: which tile of the level below (1-based, 0 for none)
/// is under each cell, and whether `layer` covers it.
#[derive(Clone)]
pub struct Support {
    top: usize,
    left: usize,
    rows: usize,
    cols: usize,
    owner: Vec<u8>,
    covered: Vec<bool>,
}

impl Support {
    /// `None` on the ground (nothing below it) or with nothing below yet.
    pub fn new(board: &Board, layer: usize) -> Option<Support> {
        if layer == 0 {
            return None;
        }
        let (a, b, c, d) = board.layer_bbox(layer - 1)?;
        let (rows, cols) = (b - a + 1, d - c + 1);
        let mut s = Support {
            top: a,
            left: c,
            rows,
            cols,
            owner: vec![0; rows * cols],
            covered: vec![false; rows * cols],
        };
        for (k, p) in board.pieces().iter().enumerate() {
            if p.mv.layer + 1 == layer {
                for (r, col) in shape(p.value, p.mv.rot).cells() {
                    let at = s.at(p.mv.i + r, p.mv.j + col).expect("inside the bbox");
                    s.owner[at] = (k % 255 + 1) as u8;
                }
            } else if p.mv.layer == layer {
                s.cover(p.value, p.mv);
            }
        }
        Some(s)
    }

    fn at(&self, i: usize, j: usize) -> Option<usize> {
        let (r, c) = (i.checked_sub(self.top)?, j.checked_sub(self.left)?);
        (r < self.rows && c < self.cols).then_some(r * self.cols + c)
    }

    /// Mark the cells a tile at `mv` would cover.
    pub fn cover(&mut self, value: u8, mv: Move) {
        for (r, c) in shape(value, mv.rot).cells() {
            if let Some(at) = self.at(mv.i + r, mv.j + c) {
                self.covered[at] = true;
            }
        }
    }

    /// Which exposed cells some tile of the `values` could still rest on.
    pub fn usable(&self, values: &[bool; 10]) -> Vec<bool> {
        let mut out = vec![false; self.rows * self.cols];
        for v in (0..10u8).filter(|&v| values[v as usize]) {
            let half_turn = shape(v, 0).rows == shape(v, 2).rows;
            for rot in 0..if half_turn { 2 } else { 4 } {
                let s = shape(v, rot);
                if s.h > self.rows || s.w > self.cols {
                    continue;
                }
                let cells: Vec<usize> = s.cells().map(|(r, c)| r * self.cols + c).collect();
                for i in 0..=self.rows - s.h {
                    for j in 0..=self.cols - s.w {
                        let base = i * self.cols + j;
                        let mut first = 0u8;
                        let mut two = false;
                        let fits = cells.iter().all(|&k| {
                            let o = self.owner[base + k];
                            if o == 0 || self.covered[base + k] {
                                return false;
                            }
                            if first == 0 {
                                first = o;
                            } else if o != first {
                                two = true;
                            }
                            true
                        });
                        if fits && two {
                            for &k in &cells {
                                out[base + k] = true;
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

impl Support {
    /// Cells usable `before` (see [`Support::usable`]) that a tile at `mv`
    /// leaves uncovered and makes unusable.
    pub fn stranded_by(&self, before: &[bool], value: u8, mv: Move, values: &[bool; 10]) -> usize {
        let mut s = self.clone();
        s.cover(value, mv);
        let after = s.usable(values);
        (0..before.len())
            .filter(|&k| before[k] && !after[k] && !s.covered[k])
            .count()
    }
}

/// Cells of the level below `mv` that a tile of the `values` could rest on
/// now, that `mv` leaves uncovered, and that none could rest on after it.
pub fn stranded(board: &Board, value: u8, mv: Move, values: &[bool; 10]) -> usize {
    let Some(s) = Support::new(board, mv.layer) else {
        return 0;
    };
    s.stranded_by(&s.usable(values), value, mv, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [bool; 10] = [true; 10];

    /// Two 0s side by side on the ground (each a 3x4 ring).
    fn two_zeros() -> Board {
        let mut b = Board::new();
        b.place(0, Board::center_move()).unwrap();
        let c = Board::center_move();
        let next = b
            .find_valid_moves(0)
            .into_iter()
            .find(|m| m.rot == 0 && m.i == c.i && m.j == c.j + 3)
            .expect("a 0 right next to the first");
        b.place(0, next).unwrap();
        b
    }

    #[test]
    fn a_single_tile_supports_nothing() {
        let mut b = Board::new();
        b.place(9, Board::center_move()).unwrap();
        let s = Support::new(&b, 1).unwrap();
        assert!(s.usable(&ALL).iter().all(|&u| !u));
        assert!(Support::new(&b, 0).is_none());
    }

    #[test]
    fn tiles_can_rest_across_two_tiles() {
        let b = two_zeros();
        let s = Support::new(&b, 1).unwrap();
        let usable = s.usable(&ALL).iter().filter(|&&u| u).count();
        assert!(usable > 0);
        // Only the shapes to come count: with no tiles left nothing is usable.
        assert_eq!(s.usable(&[false; 10]).iter().filter(|&&u| u).count(), 0);
        assert!(usable <= s.owner.iter().filter(|&&o| o > 0).count());
    }

    #[test]
    fn counts_the_cells_a_move_cuts_off() {
        let b = two_zeros();
        let values = ALL;
        for m in b.find_valid_moves(1).into_iter().filter(|m| m.layer == 1) {
            let n = stranded(&b, 1, m, &values);
            // Never more than what was usable to begin with.
            let s = Support::new(&b, 1).unwrap();
            assert!(
                n <= s.usable(&values).iter().filter(|&&u| u).count(),
                "{m:?}"
            );
        }
        // Covering the middle of the two rings leaves both outer columns
        // standing alone on one ring each: unusable.
        let worst = b
            .find_valid_moves(1)
            .into_iter()
            .filter(|m| m.layer == 1)
            .map(|m| stranded(&b, 1, m, &values))
            .max()
            .unwrap();
        assert!(worst > 0);
    }
}
