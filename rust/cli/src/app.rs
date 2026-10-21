//! The interactive session: turns key presses and mouse events into game
//! actions, and keeps track of hints, messages and overlays.

use std::sync::mpsc::{channel, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use num9_core::game::{play_tiles, shuffled_tiles};
use num9_core::rollout::MonteCarlo;
use num9_core::{Rng, Strategy};
use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::Rect;

use crate::game::Game;
use crate::hint::Hint;

/// How many of the engine's top moves the player can browse.
pub const SUGGESTIONS: usize = 3;

/// How the board is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    /// Tiles coloured by value, brighter the higher they sit.
    Tiles,
    /// A height map: colour by level.
    Height,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    Info,
    Good,
    Bad,
}

pub struct Message {
    pub text: String,
    pub tone: Tone,
    until: Instant,
}

/// The engine playing the same deal, for comparison at the end.
pub struct Rival {
    rx: Receiver<i64>,
    pub score: Option<i64>,
}

/// Which part of the board is on screen (in board cells).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Viewport {
    pub top: usize,
    pub left: usize,
    pub rows: usize,
    pub cols: usize,
}

pub struct App {
    pub game: Game,
    pub seed: u64,
    pub hint_rollouts: usize,
    pub view: View,
    pub help: bool,
    /// Whether the game-over summary is hidden (to look at the board).
    pub hide_summary: bool,
    pub hint: Option<Hint>,
    /// Whether the hint's best move is drawn on the board.
    pub show_hint: bool,
    /// Which of the top suggestions is shown on the board (0 = best).
    pub suggestion: usize,
    pub rival: Option<Rival>,
    pub message: Option<Message>,
    /// A destructive key (quit / new game) waiting for confirmation.
    pub confirm: Option<KeyCode>,
    pub quit: bool,
    pub viewport: Option<Viewport>,
    /// Where the board was drawn last frame (for the mouse).
    pub board_area: Rect,
}

impl App {
    pub fn new(seed: u64, tiles: Option<Vec<u8>>, hint_rollouts: usize) -> App {
        let tiles = tiles.unwrap_or_else(|| shuffled_tiles(&mut Rng::new(seed)));
        App {
            game: Game::new(tiles),
            seed,
            hint_rollouts,
            view: View::Tiles,
            help: false,
            hide_summary: false,
            hint: None,
            show_hint: false,
            suggestion: 0,
            rival: None,
            message: None,
            confirm: None,
            quit: false,
            viewport: None,
            board_area: Rect::default(),
        }
    }

    fn say(&mut self, text: impl Into<String>, tone: Tone) {
        self.message = Some(Message {
            text: text.into(),
            tone,
            until: Instant::now() + Duration::from_secs(4),
        });
    }

    /// Collect background results and expire old messages. Returns whether
    /// anything changed on screen.
    pub fn tick(&mut self) -> bool {
        let mut changed = false;
        if let Some(h) = &mut self.hint {
            changed |= h.poll();
        }
        if let Some(r) = &mut self.rival {
            if let Ok(score) = r.rx.try_recv() {
                r.score = Some(score);
                changed = true;
            }
        }
        if self
            .message
            .as_ref()
            .is_some_and(|m| Instant::now() >= m.until)
        {
            self.message = None;
            changed = true;
        }
        changed
    }

    /// Whether background work is running (the UI should refresh often).
    pub fn busy(&self) -> bool {
        self.hint.as_ref().is_some_and(|h| !h.finished())
            || self.rival.as_ref().is_some_and(|r| r.score.is_none())
    }

    /// The hint, if it is about the position on the board now.
    pub fn current_hint(&self) -> Option<&Hint> {
        self.hint
            .as_ref()
            .filter(|h| h.placed == self.game.placed())
    }

    /// The player's current move, if the ghost is on a legal spot.
    fn ghost_move(&self) -> Option<num9_core::Move> {
        self.game
            .preview()
            .filter(|p| p.legal.is_ok())
            .map(|p| p.mv)
    }

    fn ask_hint(&mut self) {
        if self.game.is_over() {
            return;
        }
        let yours = self.ghost_move();
        // Same position and same spot: show the hint we have.
        if self.current_hint().is_some_and(|h| h.asked_at == yours) {
            self.show_hint = true;
            return;
        }
        let value = self.game.current().unwrap();
        self.hint = Some(Hint::start(
            &self.game.board,
            value,
            yours,
            self.hint_rollouts,
        ));
        self.suggestion = 0;
        self.show_hint = true;
    }

    /// Move the ghost to suggestion `k` (0 = best), or with `None` to the
    /// next suggestion after the one shown (the best first).
    fn jump_to_hint(&mut self, k: Option<usize>) {
        let ranking = self.current_hint().map(|h| h.ranking()).unwrap_or_default();
        if ranking.is_empty() {
            self.say("Press h to ask for the best move first", Tone::Info);
            return;
        }
        let shown = ranking.len().min(SUGGESTIONS);
        let k = match k {
            Some(k) if k < shown => k,
            Some(_) => return,
            None if self.show_hint
                && self.ghost_move() == Some(ranking[self.suggestion.min(shown - 1)].mv) =>
            {
                (self.suggestion + 1) % shown
            }
            None => self.suggestion.min(shown - 1),
        };
        self.suggestion = k;
        self.game.jump_to(ranking[k].mv);
        self.show_hint = true;
    }

    fn place(&mut self) {
        match self.game.place() {
            Ok(mv) => {
                let value = self.game.board.pieces().last().unwrap().value;
                let points = value as i64 * mv.layer as i64;
                let text = match mv.layer {
                    0 => format!("{value} placed on the ground"),
                    l => format!("{value} on level {l}: +{points}"),
                };
                self.say(text, if points > 0 { Tone::Good } else { Tone::Info });
                self.show_hint = false;
                if self.game.is_over() {
                    self.start_rival();
                }
            }
            Err(problem) => self.say(problem.describe(), Tone::Bad),
        }
    }

    fn start_rival(&mut self) {
        let (tx, rx) = channel();
        let tiles = self.game.tiles.clone();
        let seed = self.seed;
        thread::spawn(move || {
            let ai = Strategy::MonteCarlo(MonteCarlo::default());
            let board = play_tiles(&ai, &tiles, &mut Rng::new(seed));
            let _ = tx.send(board.score());
        });
        self.rival = Some(Rival { rx, score: None });
    }

    fn undo(&mut self) {
        if self.game.undo() {
            self.rival = None;
            self.show_hint = false;
            self.say("Took the last tile back", Tone::Info);
        } else {
            self.say("Nothing to undo", Tone::Info);
        }
    }

    /// Start over with `tiles` (a fresh deal if `None`).
    fn restart(&mut self, tiles: Option<Vec<u8>>) {
        if tiles.is_none() {
            self.seed = self.seed.wrapping_add(1);
        }
        *self = App::new(self.seed, tiles, self.hint_rollouts);
    }

    /// Run `key`'s action now if it is harmless; otherwise ask to press it
    /// again first (when a game is in progress).
    fn confirmed(&mut self, key: KeyCode, what: &str) -> bool {
        if self.game.is_over() || self.game.placed() <= 1 || self.confirm == Some(key) {
            return true;
        }
        self.confirm = Some(key);
        let name = match key {
            KeyCode::Char(c) => c.to_string(),
            _ => "it".into(),
        };
        self.say(format!("Press {name} again to {what}"), Tone::Info);
        false
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        let pending = self.confirm.take();
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let step = if shift { 4 } else { 1 };

        if self.help {
            self.help = false;
            if matches!(
                key.code,
                KeyCode::Char('?') | KeyCode::Esc | KeyCode::Char('q')
            ) {
                return;
            }
        }
        if self.game.is_over() {
            match key.code {
                KeyCode::Char('n') => self.restart(None),
                KeyCode::Char('r') => self.restart(Some(self.game.tiles.clone())),
                KeyCode::Char('u') | KeyCode::Backspace => self.undo(),
                KeyCode::Char('q') => self.quit = true,
                KeyCode::Esc | KeyCode::Char(' ') => self.hide_summary = !self.hide_summary,
                KeyCode::Char('c') if ctrl => self.quit = true,
                KeyCode::Char('?') => self.help = true,
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Char('c') if ctrl => self.quit = true,
            KeyCode::Char('z') if ctrl => self.undo(),
            KeyCode::Left | KeyCode::Char('a') => self.game.shift(0, -step),
            KeyCode::Right | KeyCode::Char('d') => self.game.shift(0, step),
            KeyCode::Up | KeyCode::Char('w') => self.game.shift(-step, 0),
            KeyCode::Down | KeyCode::Char('s') => self.game.shift(step, 0),
            KeyCode::Char('A') => self.game.shift(0, -4),
            KeyCode::Char('D') => self.game.shift(0, 4),
            KeyCode::Char('W') => self.game.shift(-4, 0),
            KeyCode::Char('S') => self.game.shift(4, 0),
            KeyCode::Char(' ') | KeyCode::Char('r') | KeyCode::Char('x') => self.game.rotate(1),
            KeyCode::Char('R') | KeyCode::Char('z') => self.game.rotate(-1),
            KeyCode::Enter => self.place(),
            KeyCode::Char('u') | KeyCode::Backspace => self.undo(),
            KeyCode::Char('h') => self.ask_hint(),
            KeyCode::Char('g') => self.jump_to_hint(None),
            KeyCode::Char(c @ '1'..='9') => self.jump_to_hint(Some(c as usize - '1' as usize)),
            KeyCode::Tab | KeyCode::Char('v') => {
                self.view = match self.view {
                    View::Tiles => View::Height,
                    View::Height => View::Tiles,
                };
            }
            KeyCode::Char('?') => self.help = true,
            KeyCode::Esc => self.show_hint = false,
            KeyCode::Char(c @ ('q' | 'n')) => {
                self.confirm = pending;
                let what = if c == 'q' { "quit" } else { "start a new game" };
                if self.confirmed(key.code, what) {
                    self.confirm = None;
                    if c == 'q' {
                        self.quit = true;
                    } else {
                        self.restart(None);
                    }
                }
            }
            _ => {}
        }
    }

    /// Board cell under a screen position, if it is on the board.
    fn cell_at(&self, column: u16, row: u16) -> Option<(usize, usize)> {
        let a = self.board_area;
        let v = self.viewport?;
        if column < a.x || row < a.y || column >= a.x + a.width || row >= a.y + a.height {
            return None;
        }
        let r = (row - a.y) as usize;
        let c = (column - a.x) as usize / 2;
        (r < v.rows && c < v.cols).then_some((v.top + r, v.left + c))
    }

    pub fn on_mouse(&mut self, mouse: MouseEvent) {
        if self.help || self.game.is_over() {
            return;
        }
        match mouse.kind {
            MouseEventKind::Moved | MouseEventKind::Drag(_) | MouseEventKind::Down(_) => {
                if let Some((i, j)) = self.cell_at(mouse.column, mouse.row) {
                    // Centre the tile on the pointer.
                    let s = num9_core::piece::shape(self.game.current().unwrap(), self.game.rot);
                    self.game
                        .move_to(i.saturating_sub(s.h / 2), j.saturating_sub(s.w / 2));
                    if mouse.kind == MouseEventKind::Down(MouseButton::Left) {
                        self.place();
                    } else if mouse.kind == MouseEventKind::Down(MouseButton::Right) {
                        self.game.rotate(1);
                    }
                }
            }
            MouseEventKind::ScrollDown => self.game.rotate(1),
            MouseEventKind::ScrollUp => self.game.rotate(-1),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::KeyEventKind;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new_with_kind(code, KeyModifiers::NONE, KeyEventKind::Press)
    }

    fn app() -> App {
        App::new(7, None, 16)
    }

    fn wait_for_hint(app: &mut App) {
        let start = Instant::now();
        while !app.current_hint().unwrap().finished() {
            app.tick();
            assert!(start.elapsed() < Duration::from_secs(60));
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn arrows_move_and_space_rotates() {
        let mut a = app();
        let (i, j) = a.game.cursor;
        a.on_key(key(KeyCode::Left));
        a.on_key(key(KeyCode::Down));
        assert_eq!(a.game.cursor, (i + 1, j - 1));
        a.on_key(key(KeyCode::Char(' ')));
        assert_eq!(a.game.rot, 1);
        a.on_key(key(KeyCode::Char('R')));
        assert_eq!(a.game.rot, 0);
    }

    #[test]
    fn illegal_placement_explains_itself() {
        let mut a = app();
        a.game.move_to(10, 10);
        a.on_key(key(KeyCode::Enter));
        assert_eq!(a.game.placed(), 1);
        let m = a.message.as_ref().unwrap();
        assert_eq!(
            (m.text.as_str(), m.tone),
            ("Must touch a tile on level 0", Tone::Bad)
        );
    }

    #[test]
    fn hint_then_jump_then_place_plays_the_best_move() {
        let mut a = app();
        a.on_key(key(KeyCode::Char('h')));
        wait_for_hint(&mut a);
        let best = a.current_hint().unwrap().ranking()[0];
        a.on_key(key(KeyCode::Char('g')));
        assert_eq!(a.game.preview().unwrap().mv, best.mv);
        a.on_key(key(KeyCode::Enter));
        assert_eq!(a.game.board.pieces().last().unwrap().mv, best.mv);
        assert!(
            a.current_hint().is_none(),
            "hint is for the previous position"
        );
    }

    #[test]
    fn g_cycles_through_suggestions_and_moving_reasks() {
        let mut a = app();
        for _ in 0..4 {
            let mv = a.game.board.find_valid_moves(a.game.current().unwrap())[0];
            a.game.jump_to(mv);
            a.on_key(key(KeyCode::Enter));
        }
        a.on_key(key(KeyCode::Char('h')));
        wait_for_hint(&mut a);
        let ranking = a.current_hint().unwrap().ranking();
        let mut seen = Vec::new();
        for _ in 0..4 {
            a.on_key(key(KeyCode::Char('g')));
            seen.push(a.game.preview().unwrap().mv);
        }
        let top: Vec<_> = ranking.iter().take(SUGGESTIONS).map(|r| r.mv).collect();
        assert_eq!(&seen[..3], &top[..3]);
        assert_eq!(seen[3], top[0], "wraps around");
        a.on_key(key(KeyCode::Char('2')));
        assert_eq!((a.suggestion, a.game.preview().unwrap().mv), (1, top[1]));
        // Pressing h on the same spot reuses the hint; elsewhere it re-rates.
        let first = a.current_hint().unwrap() as *const Hint;
        a.on_key(key(KeyCode::Char('h')));
        assert_eq!(a.current_hint().unwrap() as *const Hint, first);
        a.game.snap_to_legal();
        a.on_key(key(KeyCode::Char('g')));
        a.on_key(key(KeyCode::Right));
        a.game.snap_to_legal();
        if a.ghost_move() != Some(top[0]) {
            a.on_key(key(KeyCode::Char('h')));
            assert_eq!(a.current_hint().unwrap().asked_at, a.ghost_move());
        }
    }

    #[test]
    fn quitting_mid_game_needs_confirmation() {
        let mut a = app();
        let mv = a.game.board.find_valid_moves(a.game.current().unwrap())[0];
        a.game.jump_to(mv);
        a.on_key(key(KeyCode::Enter));
        a.on_key(key(KeyCode::Char('q')));
        assert!(!a.quit);
        a.on_key(key(KeyCode::Left));
        a.on_key(key(KeyCode::Char('q')));
        assert!(!a.quit, "an intervening key cancels the confirmation");
        a.on_key(key(KeyCode::Char('q')));
        assert!(a.quit);
    }

    #[test]
    fn finishing_a_game_starts_the_rival_and_r_replays_the_deal() {
        let mut a = app();
        while !a.game.is_over() {
            let v = a.game.current().unwrap();
            let mv = a.game.board.find_valid_moves(v)[0];
            a.game.jump_to(mv);
            a.on_key(key(KeyCode::Enter));
        }
        assert!(a.rival.is_some());
        let tiles = a.game.tiles.clone();
        a.on_key(key(KeyCode::Char('r')));
        assert_eq!((a.game.placed(), &a.game.tiles), (1, &tiles));
        assert!(a.rival.is_none());
    }
}
