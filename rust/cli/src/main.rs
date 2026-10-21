//! `num9`: play NMBR 9 in the terminal, with the engine on hand for hints.

use std::io;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::Parser;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind,
};
use ratatui::crossterm::execute;

use num9_cli::app::App;
use num9_cli::ui;

/// Play NMBR 9 in your terminal.
///
/// Place twenty tiles, one at a time, scoring each tile's value times the
/// level it sits on. Press h at any point to ask the engine for the best move.
#[derive(Parser)]
#[command(name = "num9", version)]
struct Args {
    /// Seed for the order the tiles are drawn in (random if omitted).
    #[arg(long)]
    seed: Option<u64>,

    /// Draw the tiles in this order instead: two of each value 0-9,
    /// comma separated, e.g. 8,3,7,9,...
    #[arg(long, value_delimiter = ',', value_parser = clap::value_parser!(u8).range(0..10))]
    tiles: Option<Vec<u8>>,

    /// Simulated continuations per candidate move when asking for a hint.
    #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(u64).range(8..))]
    hint_rollouts: u64,

    /// Leave the mouse alone (lets the terminal select text).
    #[arg(long)]
    no_mouse: bool,
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    if let Some(tiles) = &args.tiles {
        let mut sorted = tiles.clone();
        sorted.sort_unstable();
        let standard: Vec<u8> = (0..10).flat_map(|v| [v, v]).collect();
        if sorted != standard {
            eprintln!("error: --tiles must list two of each value 0-9 (20 tiles)");
            std::process::exit(2);
        }
    }
    let seed = args.seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    });
    let mut app = App::new(seed, args.tiles, args.hint_rollouts as usize);

    let mouse = !args.no_mouse;
    if mouse {
        // Release the mouse even if we panic (ratatui restores the rest).
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = execute!(io::stdout(), DisableMouseCapture);
            hook(info);
        }));
    }
    let mut terminal = ratatui::init();
    if mouse {
        execute!(io::stdout(), EnableMouseCapture)?;
    }
    let result = run(&mut terminal, &mut app);
    if mouse {
        execute!(io::stdout(), DisableMouseCapture)?;
    }
    ratatui::restore();
    result
}

fn run(terminal: &mut ratatui::DefaultTerminal, app: &mut App) -> io::Result<()> {
    let mut dirty = true;
    while !app.quit {
        dirty |= app.tick();
        if dirty {
            terminal.draw(|frame| ui::draw(frame, app))?;
            dirty = false;
        }
        let timeout = if app.busy() {
            Duration::from_millis(40)
        } else {
            Duration::from_millis(500)
        };
        if event::poll(timeout)? {
            match event::read()? {
                Event::Key(key) if key.kind != KeyEventKind::Release => app.on_key(key),
                Event::Mouse(mouse) => app.on_mouse(mouse),
                Event::Resize(..) => {}
                _ => continue,
            }
            dirty = true;
        }
    }
    Ok(())
}
