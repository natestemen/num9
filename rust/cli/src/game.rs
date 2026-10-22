//! Game state for interactive play: the deal, the board, and the "ghost" tile
//! the player is moving around before placing it.
//!
//! The player positions a tile in 2D; like a physical tile it comes to rest
//! on top of whatever is underneath. That is always the only level where it
//! could legally go (every cell of a level is also occupied on the level
//! below), so the level never needs choosing.

use num9_core::board::{Board, Move, DEPTH, HEIGHT, WIDTH};
use num9_core::piece::shape;

/// Number of tiles in a game: two of each value 0-9.
pub const TILES: usize = 20;

/// What is visible in a cell when looking down on the stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Top {
    pub value: u8,
    pub layer: usize,
    /// Index of the piece in placement order.
    pub piece: usize,
}

/// The board seen from above: the topmost piece in each cell.
#[derive(Clone)]
pub struct TopView {
    cells: Vec<Option<Top>>,
}

impl TopView {
    pub fn new(board: &Board) -> TopView {
        let mut cells = vec![None; HEIGHT * WIDTH];
        for (piece, p) in board.pieces().iter().enumerate() {
            for (r, c) in shape(p.value, p.mv.rot).cells() {
                let cell = &mut cells[(p.mv.i + r) * WIDTH + p.mv.j + c];
                if cell.is_none_or(|t: Top| t.layer < p.mv.layer) {
                    *cell = Some(Top {
                        value: p.value,
                        layer: p.mv.layer,
                        piece,
                    });
                }
            }
        }
        TopView { cells }
    }

    pub fn get(&self, i: usize, j: usize) -> Option<Top> {
        self.cells[i * WIDTH + j]
    }

    /// Number of tiles stacked in a cell.
    pub fn height(&self, i: usize, j: usize) -> usize {
        self.get(i, j).map_or(0, |t| t.layer + 1)
    }
}

/// Why a tile cannot go where the ghost is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Problem {
    /// Some of the tile would sit over a lower spot than the rest.
    Overhang,
    /// The tile would rest on a single tile; it needs at least two.
    OnePieceBelow,
    /// Tiles on a level must touch another tile on that level.
    NotTouching { layer: usize },
    /// The stack is already as tall as the board allows.
    TooHigh,
}

impl Problem {
    pub fn describe(self) -> String {
        match self {
            Problem::Overhang => "Would hang over a gap".into(),
            Problem::OnePieceBelow => "Must rest on at least two tiles".into(),
            Problem::NotTouching { layer } => format!("Must touch a tile on level {layer}"),
            Problem::TooHigh => "The stack can't go any higher".into(),
        }
    }
}

/// Where the ghost tile would land and whether it may.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Preview {
    pub value: u8,
    pub mv: Move,
    pub legal: Result<(), Problem>,
}

impl Preview {
    pub fn points(&self) -> i64 {
        self.value as i64 * self.mv.layer as i64
    }
}

pub struct Game {
    /// The deal, in drawing order.
    pub tiles: Vec<u8>,
    pub board: Board,
    pub top: TopView,
    /// Top-left corner of the ghost tile.
    pub cursor: (usize, usize),
    /// Rotation of the ghost tile (clockwise quarter turns).
    pub rot: usize,
}

impl Game {
    /// Start a game with `tiles` (two of each value, in drawing order). The
    /// first tile is placed straight away: on an empty board every spot is
    /// equivalent, so the engine always starts in the middle.
    pub fn new(tiles: Vec<u8>) -> Game {
        let mut board = Board::new();
        let first = Board::center_move();
        board.place(tiles[0], first).expect("empty board");
        let top = TopView::new(&board);
        // Start the next tile just to the right of the first.
        let cursor = (first.i, first.j + shape(tiles[0], 0).w);
        let mut game = Game {
            tiles,
            board,
            top,
            cursor,
            rot: 0,
        };
        game.snap_to_legal();
        game
    }

    /// Move the ghost to the legal spot nearest where it is now, keeping its
    /// rotation if possible, so each tile starts somewhere it could go.
    pub fn snap_to_legal(&mut self) {
        let Some(value) = self.current() else { return };
        let (ci, cj) = (self.cursor.0 as isize, self.cursor.1 as isize);
        let nearest = self
            .board
            .find_valid_moves(value)
            .into_iter()
            .min_by_key(|m| {
                let d = (m.i as isize - ci).abs() + (m.j as isize - cj).abs();
                (m.rot != self.rot, d, std::cmp::Reverse(m.layer))
            });
        if let Some(mv) = nearest {
            self.jump_to(mv);
        }
    }

    /// Tiles placed so far.
    pub fn placed(&self) -> usize {
        self.board.pieces().len()
    }

    pub fn is_over(&self) -> bool {
        self.placed() == self.tiles.len()
    }

    /// The tile to place now.
    pub fn current(&self) -> Option<u8> {
        self.tiles.get(self.placed()).copied()
    }

    /// How many of each value are still to come after the current tile.
    pub fn still_to_come(&self) -> [u8; 10] {
        let mut counts = [0u8; 10];
        for &v in self.tiles.iter().skip(self.placed() + 1) {
            counts[v as usize] += 1;
        }
        counts
    }

    fn ghost_size(&self) -> (usize, usize) {
        let s = shape(self.current().unwrap_or(0), self.rot);
        (s.h, s.w)
    }

    /// Where the current tile would land at the cursor, and whether it may.
    pub fn preview(&self) -> Option<Preview> {
        let value = self.current()?;
        Some(self.preview_at(value, self.cursor.0, self.cursor.1, self.rot))
    }

    pub fn preview_at(&self, value: u8, i: usize, j: usize, rot: usize) -> Preview {
        let s = shape(value, rot);
        let cells: Vec<(usize, usize)> = s.cells().map(|(r, c)| (i + r, j + c)).collect();
        let layer = cells
            .iter()
            .map(|&(a, b)| self.top.height(a, b))
            .max()
            .unwrap();
        let mv = Move::new(layer.min(DEPTH - 1), i, j, rot);
        let legal = if layer >= DEPTH {
            Err(Problem::TooHigh)
        } else if self.board.is_valid_move(value, mv) {
            Ok(())
        } else {
            Err(self.problem(&cells, layer))
        };
        Preview { value, mv, legal }
    }

    /// Explain why a footprint at `layer` breaks the rules.
    fn problem(&self, cells: &[(usize, usize)], layer: usize) -> Problem {
        if cells.iter().any(|&(a, b)| self.top.height(a, b) != layer) {
            return Problem::Overhang;
        }
        if layer > 0 {
            let first = self.top.get(cells[0].0, cells[0].1).map(|t| t.piece);
            if cells
                .iter()
                .all(|&(a, b)| self.top.get(a, b).map(|t| t.piece) == first)
            {
                return Problem::OnePieceBelow;
            }
        }
        Problem::NotTouching { layer }
    }

    fn clamp_cursor(&mut self) {
        let (h, w) = self.ghost_size();
        self.cursor.0 = self.cursor.0.min(HEIGHT - h);
        self.cursor.1 = self.cursor.1.min(WIDTH - w);
    }

    /// Move the ghost tile, staying on the board.
    pub fn shift(&mut self, di: isize, dj: isize) {
        let (h, w) = self.ghost_size();
        self.cursor.0 = self.cursor.0.saturating_add_signed(di).min(HEIGHT - h);
        self.cursor.1 = self.cursor.1.saturating_add_signed(dj).min(WIDTH - w);
    }

    /// Put the ghost tile's top-left corner at (`i`, `j`), staying on the board.
    pub fn move_to(&mut self, i: usize, j: usize) {
        self.cursor = (i, j);
        self.clamp_cursor();
    }

    /// Rotate the ghost tile by `quarter_turns` clockwise, about its centre.
    pub fn rotate(&mut self, quarter_turns: isize) {
        let (h0, w0) = self.ghost_size();
        self.rot = (self.rot as isize + quarter_turns).rem_euclid(4) as usize;
        let (h1, w1) = self.ghost_size();
        let di = (h0 as isize - h1 as isize) / 2;
        let dj = (w0 as isize - w1 as isize) / 2;
        self.cursor.0 = self.cursor.0.saturating_add_signed(di);
        self.cursor.1 = self.cursor.1.saturating_add_signed(dj);
        self.clamp_cursor();
    }

    /// Show `mv` with the ghost tile.
    pub fn jump_to(&mut self, mv: Move) {
        self.rot = mv.rot;
        self.cursor = (mv.i, mv.j);
        self.clamp_cursor();
    }

    /// Place the current tile where the ghost is.
    pub fn place(&mut self) -> Result<Move, Problem> {
        let preview = self.preview().ok_or(Problem::TooHigh)?;
        preview.legal?;
        self.board
            .place(preview.value, preview.mv)
            .expect("legal move");
        self.top = TopView::new(&self.board);
        self.snap_to_legal();
        Ok(preview.mv)
    }

    /// Take back the last tile (the automatic first tile stays). Returns
    /// whether anything was undone; the ghost goes back to where it was.
    pub fn undo(&mut self) -> bool {
        if self.placed() <= 1 {
            return false;
        }
        let p = self.board.undo().expect("pieces to undo");
        self.top = TopView::new(&self.board);
        self.jump_to(p.mv);
        true
    }

    /// Per level: (level, tiles on it, points scored there), for levels in use.
    pub fn levels(&self) -> Vec<(usize, usize, i64)> {
        let mut out: Vec<(usize, usize, i64)> = (0..DEPTH).map(|l| (l, 0, 0)).collect();
        for p in self.board.pieces() {
            out[p.mv.layer].1 += 1;
            out[p.mv.layer].2 += p.value as i64 * p.mv.layer as i64;
        }
        out.retain(|&(_, n, _)| n > 0);
        out
    }

    /// Cell where a piece's label is drawn (see [`label_cell`]).
    pub fn label_cell(&self, piece: usize) -> (usize, usize) {
        label_cell(&self.board, piece)
    }
}

/// Cell where a piece's label is drawn: its first tile in reading order.
pub fn label_cell(board: &Board, piece: usize) -> (usize, usize) {
    let p = board.pieces()[piece];
    let (r, c) = shape(p.value, p.mv.rot).cells().next().unwrap();
    (p.mv.i + r, p.mv.j + c)
}

/// The board after playing the first `steps` of `moves` for `tiles`.
pub fn replay(tiles: &[u8], moves: &[Move], steps: usize) -> Board {
    let mut board = Board::new();
    for (&v, &mv) in tiles.iter().zip(moves).take(steps) {
        board.place(v, mv).expect("a recorded legal play");
    }
    board
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_tiles() -> Vec<u8> {
        vec![9, 9, 5, 3, 0, 1, 2, 4, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    }

    #[test]
    fn each_tile_starts_in_a_legal_spot() {
        let mut g = Game::new(standard_tiles());
        while !g.is_over() {
            assert_eq!(g.preview().unwrap().legal, Ok(()));
            g.place().unwrap();
        }
    }

    #[test]
    fn first_tile_is_placed_automatically() {
        let g = Game::new(standard_tiles());
        assert_eq!(g.placed(), 1);
        assert_eq!(g.current(), Some(9));
        assert_eq!(g.still_to_come()[9], 0);
        assert_eq!(g.still_to_come()[5], 2);
    }

    #[test]
    fn preview_agrees_with_the_engine_everywhere() {
        let mut g = Game::new(standard_tiles());
        // Build a small structure with the engine's own moves.
        for _ in 0..6 {
            let v = g.current().unwrap();
            let mv = *g
                .board
                .find_valid_moves(v)
                .iter()
                .max_by_key(|m| (m.layer, m.j))
                .unwrap();
            g.jump_to(mv);
            assert_eq!(g.place(), Ok(mv));
        }
        let v = g.current().unwrap();
        let legal = g.board.find_valid_moves(v);
        for rot in 0..4 {
            for i in 70..95 {
                for j in 50..80 {
                    let p = g.preview_at(v, i, j, rot);
                    // Legal exactly when the engine lists the move at the resting level.
                    assert_eq!(p.legal.is_ok(), legal.contains(&p.mv), "{p:?}");
                }
            }
        }
        // Every legal move is reachable by positioning the ghost.
        for mv in legal {
            assert_eq!(g.preview_at(v, mv.i, mv.j, mv.rot).mv, mv);
        }
    }

    #[test]
    fn problems_are_explained() {
        let mut g = Game::new(standard_tiles());
        // Far away from the first tile: must touch it.
        g.move_to(20, 20);
        assert_eq!(
            g.preview().unwrap().legal,
            Err(Problem::NotTouching { layer: 0 })
        );
        // Directly on top of the single tile: one piece below.
        g.move_to(80, 60);
        assert_eq!(g.preview().unwrap().legal, Err(Problem::OnePieceBelow));
        // Half on, half off: overhang.
        g.move_to(80, 62);
        assert_eq!(g.preview().unwrap().legal, Err(Problem::Overhang));
        assert!(g.place().is_err());
        assert_eq!(g.placed(), 1);
    }

    #[test]
    fn rotation_keeps_the_tile_roughly_in_place_and_on_the_board() {
        let mut g = Game::new(standard_tiles());
        g.move_to(50, 50);
        for _ in 0..4 {
            g.rotate(1);
        }
        assert_eq!((g.cursor, g.rot), ((50, 50), 0));
        g.move_to(HEIGHT, WIDTH);
        g.rotate(1);
        let (h, w) = g.ghost_size();
        assert!(g.cursor.0 + h <= HEIGHT && g.cursor.1 + w <= WIDTH);
    }

    #[test]
    fn undo_returns_the_tile_and_the_ghost() {
        let mut g = Game::new(standard_tiles());
        let mv = g.board.find_valid_moves(9)[0];
        g.jump_to(mv);
        g.place().unwrap();
        g.move_to(10, 10);
        assert!(g.undo());
        assert_eq!(
            (g.placed(), g.current(), g.cursor, g.rot),
            (1, Some(9), (mv.i, mv.j), mv.rot)
        );
        assert!(!g.undo(), "the first tile stays");
    }

    #[test]
    fn levels_and_labels() {
        let mut g = Game::new(standard_tiles());
        while !g.is_over() {
            let v = g.current().unwrap();
            let mv = *g
                .board
                .find_valid_moves(v)
                .iter()
                .max_by_key(|m| m.layer)
                .unwrap();
            g.jump_to(mv);
            g.place().unwrap();
        }
        let total: i64 = g.levels().iter().map(|l| l.2).sum();
        assert_eq!(total, g.board.score());
        assert_eq!(g.levels().iter().map(|l| l.1).sum::<usize>(), TILES);
        for k in 0..TILES {
            let (i, j) = g.label_cell(k);
            assert!(g.board.is_occupied(g.board.pieces()[k].mv.layer, i, j));
        }
    }
}
