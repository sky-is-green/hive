//! A small ratatui dashboard: Status, Models, Stacks, Runs.
//!
//! The Rust TUI is intentionally a reader: it uses the same records, library
//! rows and stack summaries as the CLI, so the two surfaces cannot disagree.
//! `q` quits, `Tab`/arrows switch tabs and move, `r` refreshes.

use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use hive_core::hardware;
use hive_core::json;
use hive_core::paths::Paths;
use hive_models::Library;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row as TableRow, Table, TableState, Tabs, Wrap};
use ratatui::{Frame, Terminal};
use serde_json::Value;

const TABS: [&str; 4] = ["Status", "Models", "Stacks", "Runs"];
const HEADER: Color = Color::Rgb(255, 221, 0);
const AMBER: Color = Color::Rgb(255, 183, 3);
const GRAY: Color = Color::Rgb(128, 128, 128);

/// Whether a bare `hive` should take over the terminal.
pub fn available() -> bool {
    if std::env::var_os("HIVE_NO_TUI").is_some() {
        return false;
    }
    if std::env::var("TERM")
        .map(|term| term == "dumb")
        .unwrap_or(false)
    {
        return false;
    }
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

pub fn run(paths: &Paths) -> Result<(), String> {
    if !available() {
        return Err(
            "an interactive terminal is required (use the plain commands in a pipe, \
             or HIVE_NO_TUI=1)"
                .into(),
        );
    }
    let mut app = App::new(paths);
    app.refresh(true);
    enable_raw_mode().map_err(|error| error.to_string())?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).map_err(|error| error.to_string())?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(|error| error.to_string())?;
    terminal.clear().map_err(|error| error.to_string())?;
    let result = app.event_loop(&mut terminal);
    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();
    result
}

struct RowData {
    cells: Vec<String>,
    detail: String,
    selectable: bool,
}

struct App<'a> {
    paths: &'a Paths,
    tab: usize,
    selected: usize,
    rows: Vec<RowData>,
    resources: String,
    message: String,
    last_refresh: Instant,
}

impl<'a> App<'a> {
    fn new(paths: &'a Paths) -> Self {
        Self {
            paths,
            tab: 0,
            selected: 0,
            rows: Vec::new(),
            resources: String::new(),
            message: String::new(),
            last_refresh: Instant::now(),
        }
    }

    fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> Result<(), String> {
        loop {
            terminal
                .draw(|frame| self.draw(frame))
                .map_err(|error| error.to_string())?;
            if event::poll(Duration::from_millis(500)).map_err(|error| error.to_string())? {
                match event::read().map_err(|error| error.to_string())? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        if !self.handle(key.code) {
                            return Ok(());
                        }
                    }
                    Event::Resize(_, _) => {}
                    _ => {}
                }
            }
            if self.last_refresh.elapsed() >= Duration::from_secs(2) {
                self.refresh(false);
            }
        }
    }

    fn handle(&mut self, code: KeyCode) -> bool {
        let previous_tab = self.tab;
        match code {
            KeyCode::Char('q') | KeyCode::Esc => return false,
            KeyCode::Tab | KeyCode::Right => self.tab = (self.tab + 1) % TABS.len(),
            KeyCode::BackTab | KeyCode::Left => self.tab = (self.tab + TABS.len() - 1) % TABS.len(),
            KeyCode::Char('1') => self.tab = 0,
            KeyCode::Char('2') => self.tab = 1,
            KeyCode::Char('3') => self.tab = 2,
            KeyCode::Char('4') => self.tab = 3,
            KeyCode::Down | KeyCode::Char('j') => self.move_selection(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_selection(-1),
            KeyCode::Char('r') => self.refresh(false),
            _ => {}
        }
        if self.tab != previous_tab {
            self.refresh(true);
        } else if !matches!(
            code,
            KeyCode::Down | KeyCode::Up | KeyCode::Char('j') | KeyCode::Char('k')
        ) {
            self.refresh(false);
        }
        true
    }

    fn move_selection(&mut self, delta: isize) {
        if self.rows.is_empty() {
            return;
        }
        let mut index = self.selected as isize;
        loop {
            index += delta;
            if index < 0 || index as usize >= self.rows.len() {
                return;
            }
            if self.rows[index as usize].selectable {
                self.selected = index as usize;
                return;
            }
        }
    }

    fn refresh(&mut self, reset_selection: bool) {
        let previous = self
            .rows
            .get(self.selected)
            .map(|row| row.cells.first().cloned().unwrap_or_default());
        self.last_refresh = Instant::now();
        self.resources = resources();
        self.message.clear();
        self.rows = match self.tab {
            0 => self.status_rows(),
            1 => self.model_rows(),
            2 => self.stack_rows(),
            _ => self.run_rows(),
        };
        if self.rows.is_empty() {
            self.message = match self.tab {
                0 => "No serves are running.  Start one: hive serve <stack> --detach".into(),
                1 => "No GGUF models yet.  Download one with: hive models download <repo>".into(),
                2 => format!("No saved stacks in {}", self.paths.stacks_dir.display()),
                _ => format!(
                    "No cascade runs in {}",
                    self.paths.repo.join("experiments/cascade/runs").display()
                ),
            };
        }
        self.selected = if reset_selection {
            self.rows.iter().position(|row| row.selectable).unwrap_or(0)
        } else {
            previous
                .and_then(|name| {
                    self.rows
                        .iter()
                        .position(|row| row.selectable && row.cells.first() == Some(&name))
                })
                .or_else(|| self.rows.iter().position(|row| row.selectable))
                .unwrap_or(0)
        };
    }

    fn status_rows(&self) -> Vec<RowData> {
        let records = hive_state::list_records(&self.paths.serve_dir);
        let mut rows = Vec::new();
        for record in records {
            let live = hive_state::pid_matches(&record);
            let started = record
                .get("started_at")
                .and_then(Value::as_f64)
                .unwrap_or_else(hive_state::now);
            let uptime = hive_state::format_duration((hive_state::now() - started).max(0.0));
            let name = record
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_owned();
            let pid = record.get("pid").and_then(Value::as_i64).unwrap_or(0);
            let mode = record
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or("foreground")
                .to_owned();
            let state = if live { "running" } else { "stale" };
            rows.push(RowData {
                cells: vec![name.clone(), pid.to_string(), mode, uptime, state.into()],
                detail: json::to_string_pretty(&Value::Object(record.clone())),
                selectable: true,
            });
            if let Some(tiers) = record.get("tiers").and_then(Value::as_array) {
                let host = record
                    .get("host")
                    .and_then(Value::as_str)
                    .unwrap_or("127.0.0.1");
                for tier in tiers {
                    let role = tier.get("role").and_then(Value::as_str).unwrap_or("?");
                    let endpoint = tier_endpoint(tier, host);
                    let ctx = tier.get("ctx").and_then(Value::as_i64);
                    let backend = tier.get("backend").and_then(Value::as_str).unwrap_or("");
                    rows.push(RowData {
                        cells: vec![
                            format!("  {role}"),
                            endpoint.clone(),
                            ctx.map(|ctx| format!("ctx {ctx}")).unwrap_or_default(),
                            backend.to_owned(),
                            String::new(),
                        ],
                        detail: format!("{endpoint}\n\n{}", json::to_string_pretty(tier)),
                        selectable: false,
                    });
                }
            }
        }
        rows
    }

    fn model_rows(&self) -> Vec<RowData> {
        Library::new(&self.paths.models_dir)
            .rows()
            .into_iter()
            .map(|row| RowData {
                cells: vec![
                    row.get("name").and_then(Value::as_str).unwrap_or("").into(),
                    format!(
                        "{:.2} GiB",
                        row.get("size_gb").and_then(Value::as_f64).unwrap_or(0.0)
                    ),
                    row.get("quantization")
                        .and_then(Value::as_str)
                        .unwrap_or("-")
                        .into(),
                    row.get("context_length")
                        .map(|ctx| ctx.to_string())
                        .unwrap_or_else(|| "-".into()),
                ],
                detail: json::to_string_pretty(&row),
                selectable: true,
            })
            .collect()
    }

    fn stack_rows(&self) -> Vec<RowData> {
        hive_core::stacks::list_summaries(&self.paths.stacks_dir)
            .into_iter()
            .map(|row| RowData {
                cells: vec![row.name.clone(), row.tiers.to_string(), row.roles.join(",")],
                detail: json::to_string_pretty(&row.to_json()),
                selectable: true,
            })
            .collect()
    }

    fn run_rows(&self) -> Vec<RowData> {
        let directory = self.paths.repo.join("experiments/cascade/runs");
        let Ok(entries) = std::fs::read_dir(&directory) else {
            return Vec::new();
        };
        let mut runs: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.join("report.json").is_file())
            .collect();
        runs.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
        runs.into_iter()
            .map(|path| {
                let report = path.join("report.json");
                let summary = std::fs::read_to_string(&report)
                    .ok()
                    .and_then(|text| serde_json::from_str::<Value>(&text).ok())
                    .and_then(|data| data.get("summary").cloned())
                    .unwrap_or(Value::Object(Default::default()));
                let tasks = summary
                    .get("tasks")
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".into());
                let cascade = percent(summary.get("cascade_accuracy"));
                let escalation = percent(summary.get("escalation_rate"));
                RowData {
                    cells: vec![
                        path.file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                        tasks,
                        cascade,
                        escalation,
                    ],
                    detail: format!(
                        "{}\n\n{}",
                        report.display(),
                        json::to_string_pretty(&summary)
                    ),
                    selectable: true,
                }
            })
            .collect()
    }

    fn draw(&self, frame: &mut Frame) {
        let area = frame.area();
        if area.width < 50 || area.height < 12 {
            frame.render_widget(
                Paragraph::new("terminal too small (need at least 50x12)")
                    .style(Style::default().fg(Color::White)),
                area,
            );
            return;
        }
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Length(2),
                Constraint::Min(6),
                Constraint::Length(2),
            ])
            .split(area);

        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    format!("  hive {}", hive_core::version()),
                    Style::default().fg(AMBER).add_modifier(Modifier::BOLD),
                ),
                Span::raw("   "),
                Span::styled(self.resources.clone(), Style::default().fg(GRAY)),
            ])),
            chunks[0],
        );
        let titles: Vec<Line> = TABS
            .iter()
            .enumerate()
            .map(|(index, name)| Line::from(format!(" {} {} ", index + 1, name)))
            .collect();
        frame.render_widget(
            Tabs::new(titles)
                .select(self.tab)
                .highlight_style(
                    Style::default()
                        .fg(Color::Black)
                        .bg(HEADER)
                        .add_modifier(Modifier::BOLD),
                )
                .style(Style::default().fg(GRAY)),
            chunks[1],
        );

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(chunks[2]);
        self.draw_table(frame, body[0]);
        self.draw_detail(frame, body[1]);

        let footer = if let Some(row) = self.rows.get(self.selected) {
            let detail_hint = if row.selectable {
                "enter/space details"
            } else {
                ""
            };
            format!(" q quit · Tab switch · ↑↓ move · r refresh · {detail_hint} ")
        } else {
            " q quit · Tab switch · r refresh ".to_owned()
        };
        frame.render_widget(
            Paragraph::new(self.message.clone())
                .style(Style::default().fg(GRAY))
                .wrap(Wrap { trim: true }),
            chunks[3],
        );
        let hint_area = Rect {
            y: chunks[3].y + 1,
            height: 1,
            ..chunks[3]
        };
        frame.render_widget(
            Paragraph::new(footer).style(Style::default().fg(Color::Black).bg(HEADER)),
            hint_area,
        );
    }

    fn draw_table(&self, frame: &mut Frame, area: Rect) {
        let headers: &[&str] = match self.tab {
            0 => &["NAME", "PID", "MODE", "UPTIME", "STATE"],
            1 => &["NAME", "SIZE", "QUANT", "CTX"],
            2 => &["NAME", "TIERS", "ROLES"],
            _ => &["RUN", "TASKS", "CASCADE", "ESCALATE"],
        };
        let widths: &[Constraint] = match self.tab {
            0 => &[
                Constraint::Length(24),
                Constraint::Length(8),
                Constraint::Length(12),
                Constraint::Length(10),
                Constraint::Length(8),
            ],
            1 => &[
                Constraint::Length(34),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(8),
            ],
            2 => &[
                Constraint::Length(28),
                Constraint::Length(7),
                Constraint::Min(20),
            ],
            _ => &[
                Constraint::Length(40),
                Constraint::Length(7),
                Constraint::Length(10),
                Constraint::Length(10),
            ],
        };
        let rows: Vec<TableRow> = self
            .rows
            .iter()
            .map(|row| {
                let cells: Vec<Span> = row
                    .cells
                    .iter()
                    .map(|cell| Span::raw(cell.clone()))
                    .collect();
                TableRow::new(cells)
            })
            .collect();
        let table = Table::new(rows, widths)
            .header(
                TableRow::new(
                    headers
                        .iter()
                        .map(|header| Span::styled(*header, Style::default().fg(AMBER)))
                        .collect::<Vec<_>>(),
                )
                .style(Style::default().add_modifier(Modifier::BOLD)),
            )
            .block(Block::default().borders(Borders::NONE))
            .row_highlight_style(Style::default().bg(Color::DarkGray).fg(Color::White));
        let mut state = TableState::default().with_selected(Some(self.selected));
        frame.render_stateful_widget(table, area, &mut state);
    }

    fn draw_detail(&self, frame: &mut Frame, area: Rect) {
        let title = self
            .rows
            .get(self.selected)
            .map(|_| " details ")
            .unwrap_or(" details ");
        let text = self
            .rows
            .get(self.selected)
            .map(|row| row.detail.as_str())
            .unwrap_or("Nothing selected.");
        frame.render_widget(
            Paragraph::new(text)
                .block(Block::default().borders(Borders::ALL).title(title))
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: false }),
            area,
        );
    }
}

fn tier_endpoint(tier: &Value, host: &str) -> String {
    if let Some(base) = tier.get("base_url").and_then(Value::as_str)
        && !base.is_empty()
    {
        return base.to_owned();
    }
    match tier.get("port").and_then(Value::as_i64) {
        Some(port) if port > 0 => format!("http://{host}:{port}/v1"),
        _ => String::new(),
    }
}

fn percent(value: Option<&Value>) -> String {
    match value.and_then(Value::as_f64) {
        Some(value) => format!("{:.0}%", value * 100.0),
        None => "-".into(),
    }
}

fn resources() -> String {
    let mut parts = Vec::new();
    if let Some((available, total)) = hardware::memory_gib() {
        parts.push(format!("RAM {available:.1}/{total:.1} GiB free"));
    }
    if let Ok(devices) = hardware::amd_devices() {
        let free: f64 = devices.iter().filter_map(|device| device.free_gb).sum();
        let total: f64 = devices.iter().filter_map(|device| device.memory_gb).sum();
        if total > 0.0 {
            parts.push(format!("VRAM {free:.1}/{total:.1} GiB free"));
        }
        let gtt_used: f64 = devices.iter().filter_map(|device| device.gtt_used_gb).sum();
        let gtt_total: f64 = devices
            .iter()
            .filter_map(|device| device.gtt_total_gb)
            .sum();
        if gtt_total > 0.0 {
            parts.push(format!("GTT {gtt_used:.1}/{gtt_total:.1} GiB"));
        }
    }
    parts.join(" · ")
}

/// Exposed for a future `hive ui` hand-off.
pub fn repo_runs_dir(repo: &Path) -> PathBuf {
    repo.join("experiments/cascade/runs")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn percentages_match_python_percent_format() {
        assert_eq!(percent(Some(&json!(0.75))), "75%");
        assert_eq!(percent(Some(&json!(0.0))), "0%");
        assert_eq!(percent(None), "-");
    }

    #[test]
    fn tier_endpoints_prefer_a_declared_base_url() {
        assert_eq!(
            tier_endpoint(&json!({"base_url": "http://x/v1", "port": 9}), "host"),
            "http://x/v1"
        );
        assert_eq!(
            tier_endpoint(&json!({"port": 1234}), "127.0.0.1"),
            "http://127.0.0.1:1234/v1"
        );
    }
}
