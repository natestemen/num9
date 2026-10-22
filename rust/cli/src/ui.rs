//! Drawing the game: the board on the left, a side panel with the tile in
//! hand, what's still to come, the score and hints, and a key bar below.

use num9_core::board::{Move, HEIGHT, WIDTH};
use num9_core::piece::shape;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Gauge, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Play, Review, Tone, View, Viewport, SUGGESTIONS};
use crate::game::{label_cell, Preview, Problem, TILES};
use crate::theme;

const PANEL_WIDTH: u16 = 36;
const MIN_WIDTH: u16 = PANEL_WIDTH + 24;
const MIN_HEIGHT: u16 = 24;

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        let text = format!(
            "num9 needs a terminal of at least {MIN_WIDTH}×{MIN_HEIGHT}\n(this one is {}×{})",
            area.width, area.height
        );
        frame.render_widget(Paragraph::new(text).centered(), area);
        return;
    }
    let [main, keys] = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(area);
    let [board, panel] =
        Layout::horizontal([Constraint::Min(1), Constraint::Length(PANEL_WIDTH)]).areas(main);
    draw_board(frame, app, board);
    match &app.review {
        Some(r) => draw_review_panel(frame, app, r, panel),
        None => draw_panel(frame, app, panel),
    }
    draw_keys(frame, app, keys);
    if app.game.is_over() && !app.hide_summary && app.review.is_none() {
        draw_game_over(frame, app, board);
    }
    if app.help {
        draw_help(frame, area);
    }
}

fn block(title: &str) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::MUTED))
        .title(Span::styled(format!(" {title} "), Style::new().bold()))
}

// ----------------------------------------------------------------------------
// Board

/// Keep the ghost tile on screen, centring on the pieces the first time.
fn follow(app: &mut App, rows: usize, cols: usize) -> Viewport {
    let (ci, cj) = app.game.cursor;
    let s = shape(app.game.current().unwrap_or(0), app.game.rot);
    let mut v = app
        .viewport
        .filter(|v| v.rows == rows && v.cols == cols)
        .unwrap_or_else(|| {
            // Centre on everything placed plus the ghost.
            let (mut lo_i, mut hi_i, mut lo_j, mut hi_j) = (ci, ci + s.h, cj, cj + s.w);
            if let Some((a, b, c, d)) = app.game.board.layer_bbox(0) {
                (lo_i, hi_i, lo_j, hi_j) =
                    (lo_i.min(a), hi_i.max(b + 1), lo_j.min(c), hi_j.max(d + 1));
            }
            Viewport {
                top: ((lo_i + hi_i) / 2).saturating_sub(rows / 2),
                left: ((lo_j + hi_j) / 2).saturating_sub(cols / 2),
                rows,
                cols,
            }
        });
    let margin_i = 2.min(rows.saturating_sub(s.h) / 2);
    let margin_j = 3.min(cols.saturating_sub(s.w) / 2);
    if ci < v.top + margin_i {
        v.top = ci.saturating_sub(margin_i);
    }
    if ci + s.h + margin_i > v.top + rows {
        v.top = (ci + s.h + margin_i).saturating_sub(rows);
    }
    if cj < v.left + margin_j {
        v.left = cj.saturating_sub(margin_j);
    }
    if cj + s.w + margin_j > v.left + cols {
        v.left = (cj + s.w + margin_j).saturating_sub(cols);
    }
    v.top = v.top.min(HEIGHT.saturating_sub(rows));
    v.left = v.left.min(WIDTH.saturating_sub(cols));
    app.viewport = Some(v);
    v
}

/// Footprint cells of `mv` for `value`.
fn footprint(value: u8, mv: Move) -> impl Iterator<Item = (usize, usize)> {
    shape(value, mv.rot)
        .cells()
        .map(move |(r, c)| (mv.i + r, mv.j + c))
}

/// Centre the view on a reviewed game (both plays, so switching is steady).
fn frame_review(r: &Review, rows: usize, cols: usize) -> Viewport {
    let (a, b, c, d) = r.extent;
    Viewport {
        top: ((a + b) / 2)
            .saturating_sub(rows / 2)
            .min(HEIGHT.saturating_sub(rows)),
        left: ((c + d) / 2)
            .saturating_sub(cols / 2)
            .min(WIDTH.saturating_sub(cols)),
        rows,
        cols,
    }
}

fn draw_board(frame: &mut Frame, app: &mut App, area: Rect) {
    let title = match (&app.review, app.view) {
        (Some(r), _) => match r.play {
            Play::Yours => format!("Your game · tile {} of {TILES}", r.step),
            Play::Best => format!("Best play found · tile {} of {TILES}", r.step),
        },
        (None, View::Tiles) => "Board".to_string(),
        (None, View::Height) => "Board · height map".to_string(),
    };
    let outer = block(&title);
    let inner = outer.inner(area);
    frame.render_widget(outer, area);
    app.board_area = inner;
    let rows = inner.height as usize;
    let cols = inner.width as usize / 2;
    let v = match &app.review {
        Some(r) => frame_review(r, rows, cols),
        None => follow(app, rows, cols),
    };
    let game = &app.game;
    let (board, top) = match &app.review {
        Some(r) => (&r.board, &r.top),
        None => (&game.board, &game.top),
    };
    let buf = frame.buffer_mut();
    let last = board.pieces().len().checked_sub(1);

    let put = |buf: &mut Buffer, i: usize, j: usize, text: [char; 2], style: Style| {
        if i < v.top || j < v.left || i >= v.top + rows || j >= v.left + cols {
            return;
        }
        let x = inner.x + 2 * (j - v.left) as u16;
        let y = inner.y + (i - v.top) as u16;
        for (k, ch) in text.into_iter().enumerate() {
            if let Some(cell) = buf.cell_mut((x + k as u16, y)) {
                cell.set_char(ch).set_style(style);
            }
        }
    };

    // Tiles, as seen from above.
    for i in v.top..(v.top + rows).min(HEIGHT) {
        for j in v.left..(v.left + cols).min(WIDTH) {
            match top.get(i, j) {
                None => put(buf, i, j, ['·', ' '], Style::new().fg(theme::EMPTY_DOT)),
                Some(t) => {
                    let bg = match app.view {
                        View::Tiles => theme::tile(t.value, t.layer),
                        View::Height => theme::level(t.layer),
                    };
                    let mut style = Style::new().bg(bg).fg(theme::text_on(bg));
                    let text = if label_cell(board, t.piece) == (i, j) {
                        if Some(t.piece) == last {
                            style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                        }
                        [char::from(b'0' + t.value), theme::superscript(t.layer)]
                    } else {
                        [' ', ' ']
                    };
                    put(buf, i, j, text, style);
                }
            }
        }
    }

    if app.review.is_some() {
        return;
    }

    // The engine's suggestion being browsed, if asked for.
    if app.show_hint {
        let hint = app.current_hint();
        let pick = hint.and_then(|h| h.ranking().get(app.suggestion).copied());
        if let (Some(h), Some(pick)) = (hint, pick) {
            for (i, j) in footprint(h.value, pick.mv) {
                let bg = game.top.get(i, j).map(|t| theme::tile(t.value, t.layer));
                let mut style = Style::new().fg(theme::HINT).add_modifier(Modifier::BOLD);
                if let Some(bg) = bg {
                    style = style.bg(bg);
                }
                put(buf, i, j, ['▒', '▒'], style);
            }
        }
    }

    // The tile in hand.
    if let Some(p) = game.preview() {
        let cells: Vec<_> = footprint(p.value, p.mv).collect();
        for (k, &(i, j)) in cells.iter().enumerate() {
            let (style, text) = match p.legal {
                Ok(()) => {
                    let bg = theme::value(p.value);
                    let style = Style::new().bg(bg).fg(theme::text_on(bg));
                    if k == 0 {
                        let label = [char::from(b'0' + p.value), theme::superscript(p.mv.layer)];
                        (style.add_modifier(Modifier::BOLD), label)
                    } else {
                        (style, ['░', '░'])
                    }
                }
                Err(_) => {
                    let style = Style::new().bg(theme::ILLEGAL_BG).fg(theme::ILLEGAL_FG);
                    (style, if k == 0 { ['✗', ' '] } else { ['░', '░'] })
                }
            };
            put(buf, i, j, text, style);
        }
    }
}

// ----------------------------------------------------------------------------
// Side panel

fn draw_panel(frame: &mut Frame, app: &App, area: Rect) {
    let game = &app.game;
    let hint_height = if app.current_hint().is_some() && app.show_hint {
        10
    } else {
        4
    };
    let [header, tile, bag, levels, hint, message] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Length(9),
        Constraint::Length(5),
        Constraint::Min(3),
        Constraint::Length(hint_height),
        Constraint::Length(3),
    ])
    .areas(area);

    // Score and progress.
    let turn = (game.placed() + usize::from(!game.is_over())).min(TILES);
    let header_lines = vec![
        Line::from(vec![
            Span::styled("NMBR 9", Style::new().bold().fg(theme::ACCENT)),
            Span::styled(
                format!("   tile {turn} of {}", game.tiles.len()),
                Style::new().fg(theme::MUTED),
            ),
        ]),
        Line::from(vec![
            Span::raw("Score "),
            Span::styled(game.board.score().to_string(), Style::new().bold()),
        ]),
    ];
    frame.render_widget(Paragraph::new(header_lines).block(block("Game")), header);

    // The tile in hand and where it would go.
    let preview = game.preview();
    let mut lines: Vec<Line> = Vec::new();
    if let Some(value) = game.current() {
        let s = shape(value, game.rot);
        let bg = theme::value(value);
        for r in 0..4 {
            let mut spans = vec![Span::raw("  ")];
            for c in 0..4 {
                let filled = r < s.h && c < s.w && (s.rows[r] >> c) & 1 == 1;
                spans.push(if filled {
                    Span::styled("  ", Style::new().bg(bg))
                } else {
                    Span::raw("  ")
                });
            }
            if r == 1 {
                spans.push(Span::styled(
                    format!("   {value}"),
                    Style::new().bold().fg(bg),
                ));
            }
            if r == 2 {
                spans.push(Span::styled(
                    format!("   ↻ {}°", game.rot * 90),
                    Style::new().fg(theme::MUTED),
                ));
            }
            lines.push(Line::from(spans));
        }
        lines.push(Line::raw(""));
        lines.push(placement_line(preview));
    } else {
        lines.push(Line::styled(
            "All tiles placed",
            Style::new().fg(theme::MUTED),
        ));
    }
    frame.render_widget(Paragraph::new(lines).block(block("Tile in hand")), tile);

    // What's still in the bag.
    let counts = game.still_to_come();
    let bag_line = |values: std::ops::Range<u8>| {
        let mut spans = Vec::new();
        for v in values {
            let n = counts[v as usize];
            let fg = if n == 0 {
                theme::MUTED
            } else {
                theme::value(v)
            };
            spans.push(Span::styled(format!("{v}"), Style::new().fg(fg).bold()));
            let dots: String = (0..2).map(|k| if k < n { '●' } else { '○' }).collect();
            spans.push(Span::styled(format!("{dots}  "), Style::new().fg(fg)));
        }
        Line::from(spans)
    };
    frame.render_widget(
        Paragraph::new(vec![bag_line(0..5), bag_line(5..10)]).block(block("Still to come")),
        bag,
    );

    // Points per level.
    let mut level_lines: Vec<Line> = game
        .levels()
        .into_iter()
        .map(|(l, n, pts)| {
            Line::from(vec![
                Span::styled(
                    format!("Level {l} "),
                    Style::new().fg(theme::level(l)).bold(),
                ),
                Span::raw(format!("{n:>2} tile{}  ", if n == 1 { " " } else { "s" })),
                Span::styled(format!("{pts:>3} pts"), Style::new().fg(theme::MUTED)),
            ])
        })
        .collect();
    if level_lines.is_empty() {
        level_lines.push(Line::raw(""));
    }
    frame.render_widget(Paragraph::new(level_lines).block(block("Levels")), levels);

    draw_hint(frame, app, hint);

    // Latest message.
    if let Some(m) = &app.message {
        let fg = match m.tone {
            Tone::Info => Color::Reset,
            Tone::Good => theme::GOOD,
            Tone::Bad => theme::BAD,
        };
        frame.render_widget(
            Paragraph::new(Line::styled(m.text.clone(), Style::new().fg(fg)))
                .wrap(Wrap { trim: true }),
            message.inner(ratatui::layout::Margin::new(1, 0)),
        );
    }
}

fn placement_line(preview: Option<Preview>) -> Line<'static> {
    match preview {
        Some(
            p @ Preview {
                legal: Ok(()), mv, ..
            },
        ) => {
            let points = p.points();
            let mut spans = vec![
                Span::styled("✓ ", Style::new().fg(theme::GOOD)),
                Span::raw(format!("Level {}", mv.layer)),
            ];
            if points > 0 {
                spans.push(Span::styled(
                    format!("  +{points} pts"),
                    Style::new().fg(theme::GOOD).bold(),
                ));
            }
            Line::from(spans)
        }
        Some(Preview {
            legal: Err(problem),
            ..
        }) => Line::from(vec![
            Span::styled("✗ ", Style::new().fg(theme::BAD)),
            Span::styled(Problem::describe(problem), Style::new().fg(theme::BAD)),
        ]),
        None => Line::raw(""),
    }
}

/// "level 2  +14" for a move of `value`.
fn describe_move(value: u8, m: Move) -> String {
    let points = value as usize * m.layer;
    let points = if points > 0 {
        format!("+{points}")
    } else {
        String::new()
    };
    format!("level {}  {points:<3}", m.layer)
}

fn draw_hint(frame: &mut Frame, app: &App, area: Rect) {
    let outer = block("Best move");
    let inner = outer.inner(area);
    frame.render_widget(outer, area);
    let Some(h) = app.current_hint().filter(|_| app.show_hint) else {
        let line = if app.game.is_over() {
            Line::styled("Game over", Style::new().fg(theme::MUTED))
        } else {
            Line::from(vec![
                Span::styled("h", Style::new().fg(theme::ACCENT).bold()),
                Span::styled(" ask the engine", Style::new().fg(theme::MUTED)),
            ])
        };
        frame.render_widget(Paragraph::new(line), inner);
        return;
    };
    let [gauge, list] = Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(inner);
    if h.finished() {
        let text = if h.forced() {
            "Only one way to place it"
        } else {
            "Expected final score"
        };
        frame.render_widget(Line::styled(text, Style::new().fg(theme::MUTED)), gauge);
    } else {
        frame.render_widget(
            Gauge::default()
                .gauge_style(Style::new().fg(theme::HINT).bg(Color::Rgb(40, 44, 52)))
                .ratio(h.progress().min(1.0))
                .label(format!("thinking… {:.0}%", h.progress() * 100.0)),
            gauge,
        );
    }
    let ranking = h.ranking();
    let mut lines = Vec::new();
    for (k, r) in ranking.iter().take(SUGGESTIONS).enumerate() {
        let shown = k == app.suggestion;
        let style = if shown {
            Style::new().fg(theme::HINT).bold()
        } else {
            Style::new()
        };
        lines.push(Line::from(vec![
            Span::styled(if shown { "▸ " } else { "  " }, style),
            Span::styled(
                format!("{}. {}", k + 1, describe_move(h.value, r.mv)),
                style,
            ),
            Span::styled(format!("≈ {:>5.1}", r.expected), style),
        ]));
    }
    let ghost = app.game.preview().filter(|p| p.legal.is_ok()).map(|p| p.mv);
    match h.yours() {
        Some((rank, r)) if ghost == h.asked_at => {
            let gap = ranking.first().map_or(0.0, |b| b.expected - r.expected);
            let verdict = if rank == 1 {
                Span::styled(" best!", Style::new().fg(theme::GOOD).bold())
            } else {
                let fg = if gap < 1.0 { theme::GOOD } else { theme::BAD };
                Span::styled(format!(" −{gap:.1}"), Style::new().fg(fg))
            };
            lines.push(Line::raw(""));
            lines.push(Line::from(vec![
                Span::raw(format!("  You #{rank:<2}{}", describe_move(h.value, r.mv))),
                Span::raw(format!("≈ {:>5.1}", r.expected)),
                verdict,
            ]));
        }
        _ if ghost.is_some_and(|g| !ranking.iter().take(SUGGESTIONS).any(|r| r.mv == g)) => {
            lines.push(Line::raw(""));
            lines.push(Line::from(vec![
                Span::styled("  h", Style::new().fg(theme::ACCENT).bold()),
                Span::styled(" rate the spot you're on", Style::new().fg(theme::MUTED)),
            ]));
        }
        _ => {}
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(vec![
        Span::styled("  g", Style::new().fg(theme::ACCENT).bold()),
        Span::styled(" next suggestion  ", Style::new().fg(theme::MUTED)),
        Span::styled("1-3", Style::new().fg(theme::ACCENT).bold()),
        Span::styled(" pick", Style::new().fg(theme::MUTED)),
    ]));
    frame.render_widget(Paragraph::new(lines), list);
}

// ----------------------------------------------------------------------------
// Key bar and overlays

/// The two plays side by side at the step being reviewed.
fn draw_review_panel(frame: &mut Frame, app: &App, r: &Review, area: Rect) {
    let tiles = &app.game.tiles;
    let yours: Vec<Move> = app.game.board.pieces().iter().map(|p| p.mv).collect();
    let best = app.analysis.as_ref().and_then(|a| a.best.as_ref());
    let k = r.step - 1;
    let value = tiles[k];
    let score_after = |moves: &[Move]| -> i64 {
        tiles
            .iter()
            .zip(moves)
            .take(r.step)
            .map(|(&v, m)| v as i64 * m.layer as i64)
            .sum()
    };
    let describe = |m: Move| {
        let pts = value as usize * m.layer;
        if pts > 0 {
            format!("level {}  +{pts}", m.layer)
        } else {
            format!("level {}", m.layer)
        }
    };
    let row = |label: &str, play: Play, mv: Option<Move>, total: Option<i64>| {
        let style = if r.play == play {
            Style::new().bold().fg(theme::ACCENT)
        } else {
            Style::new()
        };
        Line::from(vec![
            Span::styled(
                format!("{} {label:<5}", if r.play == play { "▸" } else { " " }),
                style,
            ),
            Span::styled(format!("{:<13}", mv.map_or("…".into(), describe)), style),
            Span::styled(
                total.map_or(String::new(), |t| format!("score {t}")),
                Style::new().fg(theme::MUTED),
            ),
        ])
    };
    let mut lines = vec![
        Line::from(vec![
            Span::styled(
                format!("Tile {} of {TILES}: ", r.step),
                Style::new().fg(theme::MUTED),
            ),
            Span::styled(
                format!("a {value}"),
                Style::new().bold().fg(theme::value(value)),
            ),
        ]),
        Line::raw(""),
        row(
            "You",
            Play::Yours,
            Some(yours[k]),
            Some(score_after(&yours)),
        ),
        row(
            "Best",
            Play::Best,
            best.map(|b| b.moves[k]),
            best.map(|b| score_after(&b.moves)),
        ),
        Line::raw(""),
    ];
    let final_line = |label: &str, score: Option<i64>| {
        Line::from(vec![
            Span::raw(format!("  {label:<11}")),
            Span::styled(
                score.map_or("…".into(), |s| s.to_string()),
                Style::new().bold(),
            ),
        ])
    };
    lines.push(Line::styled("Final scores", Style::new().fg(theme::MUTED)));
    lines.push(final_line("You", Some(app.game.board.score())));
    lines.push(final_line("Best found", best.map(|b| b.score)));
    lines.push(final_line(
        "Engine",
        app.rival.as_ref().and_then(|r| r.score),
    ));
    if app.analysis.as_ref().is_some_and(|a| !a.finished()) {
        lines.push(Line::raw(""));
        lines.push(Line::styled(
            "Still searching for a better play…",
            Style::new().fg(theme::MUTED),
        ));
    }
    frame.render_widget(Paragraph::new(lines).block(block("Review")), area);
}

fn draw_keys(frame: &mut Frame, app: &App, area: Rect) {
    let keys: &[(&str, &str)] = if app.review.is_some() {
        &[
            ("←→", "step"),
            ("home/end", "first/last"),
            ("tab", "your game / best play"),
            ("esc", "back"),
        ]
    } else if app.game.is_over() {
        &[
            ("b", "review best play"),
            ("n", "new game"),
            ("r", "replay tiles"),
            ("esc", "show/hide summary"),
            ("u", "undo"),
            ("q", "quit"),
        ]
    } else {
        &[
            ("←↑↓→", "move"),
            ("space", "rotate"),
            ("enter", "place"),
            ("h", "hint"),
            ("g", "next hint"),
            ("u", "undo"),
            ("tab", "view"),
            ("?", "help"),
            ("q", "quit"),
        ]
    };
    let mut spans = Vec::new();
    for (key, what) in keys {
        spans.push(Span::styled(
            format!(" {key} "),
            Style::new().bold().fg(Color::Black).bg(theme::MUTED),
        ));
        spans.push(Span::styled(
            format!(" {what}  "),
            Style::new().fg(theme::MUTED),
        ));
    }
    frame.render_widget(Line::from(spans), area);
}

/// A centred rectangle of at most `width` × `height` inside `area`.
fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let w = width.min(area.width);
    let h = height.min(area.height);
    Rect::new(
        area.x + (area.width - w) / 2,
        area.y + (area.height - h) / 2,
        w,
        h,
    )
}

fn draw_game_over(frame: &mut Frame, app: &App, area: Rect) {
    let game = &app.game;
    let score = game.board.score();
    let mut lines = vec![
        Line::from(vec![
            Span::raw("Final score  "),
            Span::styled(score.to_string(), Style::new().bold().fg(theme::ACCENT)),
        ]),
        Line::raw(""),
    ];
    for (l, n, pts) in game.levels() {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  Level {l} "),
                Style::new().fg(theme::level(l)).bold(),
            ),
            Span::raw(format!(
                "{n:>2} tile{} {pts:>4} pts",
                if n == 1 { " " } else { "s" }
            )),
        ]));
    }
    lines.push(Line::raw(""));
    let rival = match app.rival.as_ref().and_then(|r| r.score) {
        Some(ai) => {
            let (verdict, color) = match score.cmp(&ai) {
                std::cmp::Ordering::Greater => (format!("you won by {}!", score - ai), theme::GOOD),
                std::cmp::Ordering::Equal => ("a tie!".to_string(), theme::ACCENT),
                std::cmp::Ordering::Less => (format!("it won by {}", ai - score), theme::BAD),
            };
            Line::from(vec![
                Span::raw(format!("The engine scored {ai} on these tiles: ")),
                Span::styled(verdict, Style::new().fg(color).bold()),
            ])
        }
        None => Line::styled(
            "The engine is playing these tiles…",
            Style::new().fg(theme::MUTED),
        ),
    };
    lines.push(rival);
    let analysis = app.analysis.as_ref();
    let best = match analysis.and_then(|a| a.best.as_ref()) {
        Some(b) => {
            let searching = analysis.is_some_and(|a| !a.finished());
            Line::from(vec![
                Span::raw("Best play found on these tiles: "),
                Span::styled(b.score.to_string(), Style::new().bold().fg(theme::HINT)),
                Span::styled(
                    if searching {
                        "  (still searching…)"
                    } else {
                        ""
                    },
                    Style::new().fg(theme::MUTED),
                ),
            ])
        }
        None => Line::styled(
            "Searching for the best play of these tiles…",
            Style::new().fg(theme::MUTED),
        ),
    };
    lines.push(best);
    lines.push(Line::raw(""));
    lines.push(Line::styled(
        "b review the best play · n new game · r replay these tiles",
        Style::new().fg(theme::MUTED),
    ));
    lines.push(Line::styled(
        "esc view board · q quit",
        Style::new().fg(theme::MUTED),
    ));
    let popup = centered(area, 60, lines.len() as u16 + 2);
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines)
            .block(block("Game over"))
            .wrap(Wrap { trim: false }),
        popup,
    );
}

fn draw_help(frame: &mut Frame, area: Rect) {
    let key = |k: &str| Span::styled(format!("{k:<14}"), Style::new().fg(theme::ACCENT).bold());
    let heading = |t: &str| Line::styled(t.to_string(), Style::new().bold());
    let lines = vec![
        heading("How to play"),
        Line::raw("Place the 20 tiles one at a time, in the order they're drawn."),
        Line::raw("Each tile scores its value × its level (the ground is level 0)."),
        Line::raw("A tile must touch another tile on its own level. To go up a"),
        Line::raw("level it must sit entirely on tiles below, on at least two."),
        Line::raw("Tiles land on whatever is underneath, so just slide them around."),
        Line::raw(""),
        heading("Keys"),
        Line::from(vec![
            key("←↑↓→ / wasd"),
            Span::raw("move (with shift: 4 cells)"),
        ]),
        Line::from(vec![
            key("space / r"),
            Span::raw("rotate clockwise (R: anticlockwise)"),
        ]),
        Line::from(vec![key("enter"), Span::raw("place the tile")]),
        Line::from(vec![key("u"), Span::raw("undo")]),
        Line::from(vec![
            key("h"),
            Span::raw("ask the engine for the best move"),
        ]),
        Line::from(vec![
            key("g / 1-3"),
            Span::raw("move the tile to the engine's picks"),
        ]),
        Line::from(vec![
            key("tab"),
            Span::raw("switch between tiles and height map"),
        ]),
        Line::from(vec![
            key("mouse"),
            Span::raw("hover to aim, click to place, scroll to rotate"),
        ]),
        Line::from(vec![
            key("b"),
            Span::raw("after a game: step through the best play found"),
        ]),
        Line::from(vec![key("n / q"), Span::raw("new game / quit")]),
        Line::raw(""),
        heading("Reading the board"),
        Line::raw("Tiles show their value and level (7² = a 7 on level 2) and get"),
        Line::raw("brighter the higher they sit. The last tile placed is underlined."),
        Line::raw(""),
        Line::styled("Press any key to close", Style::new().fg(theme::MUTED)),
    ];
    let popup = centered(area, 70, lines.len() as u16 + 2);
    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(lines).block(block("Help")), popup);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn screen(app: &mut App, w: u16, h: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        terminal.draw(|f| draw(f, app)).unwrap();
        let buf = terminal.backend().buffer().clone();
        (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn renders_the_main_screen() {
        let mut app = App::new(3, None, 16);
        let s = screen(&mut app, 100, 32);
        for text in [
            "Board",
            "NMBR 9",
            "tile 2 of 20",
            "Tile in hand",
            "Still to come",
            "Best move",
            "place",
        ] {
            assert!(s.contains(text), "missing {text:?} in\n{s}");
        }
        // The first tile's label: its value on level 0.
        let first = app.game.tiles[0];
        assert!(s.contains(&format!("{first}⁰")), "{s}");
    }

    #[test]
    fn explains_an_illegal_spot_and_shows_help() {
        let mut app = App::new(3, None, 16);
        app.game.move_to(10, 10);
        let s = screen(&mut app, 100, 32);
        assert!(s.contains("Must touch a tile on level 0"), "{s}");
        app.help = true;
        let s = screen(&mut app, 100, 32);
        assert!(s.contains("How to play"), "{s}");
    }

    #[test]
    fn renders_the_summary_and_the_review() {
        let mut app = App::new(3, None, 16);
        app.analysis_stages = &[(1, 2)];
        while !app.game.is_over() {
            let v = app.game.current().unwrap();
            let mv = app.game.board.find_valid_moves(v)[0];
            app.game.jump_to(mv);
            app.game.place().unwrap();
        }
        app.analysis = Some(crate::analysis::Analysis::start(
            app.game.tiles.clone(),
            &[(1, 2)],
        ));
        let start = std::time::Instant::now();
        while !app.analysis.as_ref().unwrap().finished() {
            app.tick();
            assert!(start.elapsed().as_secs() < 60);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        app.tick();
        let s = screen(&mut app, 110, 34);
        assert!(s.contains("Best play found on these tiles"), "{s}");
        app.on_key(ratatui::crossterm::event::KeyEvent::from(
            ratatui::crossterm::event::KeyCode::Char('b'),
        ));
        let s = screen(&mut app, 110, 34);
        for text in [
            "Best play found · tile 20 of 20",
            "Review",
            "Final scores",
            "Best found",
            "your game / best play",
        ] {
            assert!(s.contains(text), "missing {text:?} in\n{s}");
        }
    }

    #[test]
    fn small_terminals_get_a_message() {
        let mut app = App::new(3, None, 16);
        let s = screen(&mut app, 40, 10);
        assert!(s.contains("needs a terminal"), "{s}");
    }
}
