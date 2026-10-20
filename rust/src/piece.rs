//! Piece shapes, mirroring `num9/piece.py`.

use std::sync::OnceLock;

/// Piece layouts at rotation 0, copied from `num9/piece.py` (1 = tile).
const BASE: [&[&[u8]]; 10] = [
    &[&[1, 1, 1], &[1, 0, 1], &[1, 0, 1], &[1, 1, 1]],
    &[&[1, 1], &[0, 1], &[0, 1], &[0, 1]],
    &[&[0, 1, 1], &[0, 1, 1], &[1, 1, 0], &[1, 1, 1]],
    &[&[1, 1, 1], &[0, 0, 1], &[0, 1, 1], &[1, 1, 1]],
    &[&[0, 1, 1], &[0, 1, 0], &[1, 1, 1], &[0, 1, 1]],
    &[&[1, 1, 1], &[1, 1, 1], &[0, 0, 1], &[1, 1, 1]],
    &[&[1, 1, 0], &[1, 0, 0], &[1, 1, 1], &[1, 1, 1]],
    &[&[1, 1, 1], &[0, 1, 0], &[1, 1, 0], &[1, 0, 0]],
    &[&[0, 1, 1], &[0, 1, 1], &[1, 1, 0], &[1, 1, 0]],
    &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 0], &[1, 1, 0]],
];

/// One rotation of one piece, stored as per-row bitmasks.
#[derive(Clone, Debug)]
pub struct Shape {
    pub h: usize,
    pub w: usize,
    /// `rows[r]` bit `c` set <=> tile at (r, c). Rows `>= h` are zero.
    pub rows: [u8; 4],
    /// Cells orthogonally adjacent to the piece but not part of it (including
    /// enclosed holes), matching `Piece.surrounding_indices`. `ring[rr]` bit `c`
    /// refers to the cell at offset (rr - 1, c - 1) from the piece's top-left.
    pub ring: [u8; 6],
    pub tiles: u32,
}

impl Shape {
    fn from_grid(grid: &[Vec<u8>]) -> Shape {
        let h = grid.len();
        let w = grid[0].len();
        let mut rows = [0u8; 4];
        for (r, row) in grid.iter().enumerate() {
            for (c, &cell) in row.iter().enumerate() {
                if cell != 0 {
                    rows[r] |= 1 << c;
                }
            }
        }
        // Dilate the piece by one cell (in a frame padded by 1) and remove the piece.
        let mut padded = [0u8; 6];
        for r in 0..h {
            padded[r + 1] = rows[r] << 1;
        }
        let mut ring = [0u8; 6];
        for rr in 0..h + 2 {
            let mut d = padded[rr] | (padded[rr] << 1) | (padded[rr] >> 1);
            if rr > 0 {
                d |= padded[rr - 1];
            }
            if rr + 1 < 6 {
                d |= padded[rr + 1];
            }
            ring[rr] = d & !padded[rr];
        }
        let tiles = rows.iter().map(|r| r.count_ones()).sum();
        Shape {
            h,
            w,
            rows,
            ring,
            tiles,
        }
    }

    /// The shape as a 0/1 grid.
    pub fn grid(&self) -> Vec<Vec<u8>> {
        (0..self.h)
            .map(|r| (0..self.w).map(|c| (self.rows[r] >> c) & 1).collect())
            .collect()
    }

    /// Tile coordinates in row-major order.
    pub fn cells(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.h).flat_map(move |r| {
            (0..self.w).filter_map(move |c| ((self.rows[r] >> c) & 1 == 1).then_some((r, c)))
        })
    }
}

/// Rotate 90 degrees clockwise, as `zip(*array[::-1])` does in `Piece.array`.
fn rotate(grid: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let h = grid.len();
    let w = grid[0].len();
    (0..w)
        .map(|i| (0..h).map(|j| grid[h - 1 - j][i]).collect())
        .collect()
}

static SHAPES: OnceLock<Vec<[Shape; 4]>> = OnceLock::new();

fn build_shapes() -> Vec<[Shape; 4]> {
    BASE.iter()
        .map(|base| {
            let mut grid: Vec<Vec<u8>> = base.iter().map(|r| r.to_vec()).collect();
            let mut out = Vec::with_capacity(4);
            for _ in 0..4 {
                out.push(Shape::from_grid(&grid));
                grid = rotate(&grid);
            }
            out.try_into().unwrap()
        })
        .collect()
}

/// Shape of piece `value` (0..=9) at `rotation` (0..=3, clockwise quarter turns).
pub fn shape(value: u8, rotation: usize) -> &'static Shape {
    &SHAPES.get_or_init(build_shapes)[value as usize][rotation % 4]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_counts() {
        let expected = [10, 5, 9, 9, 8, 10, 9, 7, 8, 10];
        for v in 0..10u8 {
            for r in 0..4 {
                assert_eq!(shape(v, r).tiles, expected[v as usize]);
            }
        }
    }

    #[test]
    fn four_rotations_is_identity() {
        for v in 0..10u8 {
            let g = shape(v, 0).grid();
            let mut r = g.clone();
            for _ in 0..4 {
                r = rotate(&r);
            }
            assert_eq!(g, r);
        }
    }

    #[test]
    fn ring_of_zero_includes_hole() {
        let s = shape(0, 0);
        // The hole cells (1,1) and (2,1) sit at ring offset (+1, +1).
        assert_eq!((s.ring[2] >> 2) & 1, 1);
        assert_eq!((s.ring[3] >> 2) & 1, 1);
        // Corners diagonal to the piece are not neighbours.
        assert_eq!(s.ring[0] & 1, 0);
    }
}
