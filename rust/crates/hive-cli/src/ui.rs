//! `hive ui`: switch between the terminal dashboard and the Rust daemon.
//!
//! The web surface is the Rust daemon serving the existing `harness/webui`
//! shell, with the Python sidecar behind it. `status` reports both surfaces
//! without starting anything; `web` runs the daemon in the foreground or
//! detaches it with a pidfile and a log.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use hive_core::http;
use hive_core::paths::Paths;
use hive_core::python;
use serde_json::{Map, Value};

type Cmd = Result<u8, String>;

pub struct Options {
    pub mode: String,
    pub host: String,
    pub port: u16,
    pub detach: bool,
    pub no_open: bool,
    pub mock: bool,
    pub state_dir: Option<PathBuf>,
    pub as_json: bool,
}

pub fn run(paths: &Paths, options: Options) -> Cmd {
    match options.mode.as_str() {
        "terminal" => match hive_tui::run(paths) {
            Ok(()) => Ok(0),
            Err(error) => {
                eprintln!("hive ui terminal: {error}");
                Ok(1)
            }
        },
        "web" => web(paths, &options),
        _ => status(&options),
    }
}

fn status(options: &Options) -> Cmd {
    let api_url = format!("http://{}:{}", options.host, options.port);
    // The Rust daemon serves the shell on the API port; Python used a second
    // console port. Report the real surface, not the historical default.
    let web_url = api_url.clone();
    let api_alive = alive(&api_url, "/health");
    let web_alive = alive(&web_url, "/");
    let mut payload = Map::new();
    payload.insert(
        "web".into(),
        serde_json::json!({"running": web_alive, "url": web_url}),
    );
    payload.insert(
        "api".into(),
        serde_json::json!({"running": api_alive, "url": api_url}),
    );
    payload.insert(
        "terminal".into(),
        serde_json::json!({"command": "hive ui terminal"}),
    );
    payload.insert(
        "switch".into(),
        serde_json::json!({"web": "hive ui web", "terminal": "hive ui terminal"}),
    );
    let payload = Value::Object(payload);
    if options.as_json {
        print!("{}", hive_core::json::pretty(&payload));
        return Ok(0);
    }
    println!("interfaces");
    println!(
        "  web    {} · {}",
        web_url,
        if web_alive { "running" } else { "not running" }
    );
    println!(
        "  api    {} · {}",
        api_url,
        if api_alive { "running" } else { "not running" }
    );
    println!("  terminal  hive ui terminal");
    println!("  start web hive ui web");
    Ok(0)
}

fn web(paths: &Paths, options: &Options) -> Cmd {
    let daemon = daemon_binary()?;
    let log_dir = &paths.log_dir;
    std::fs::create_dir_all(log_dir)
        .map_err(|error| format!("cannot create {}: {error}", log_dir.display()))?;
    let log_path = log_dir.join("daemon.log");
    let pid_path = log_dir.join("daemon.pid");
    let url = format!("http://{}:{}/", options.host, options.port);
    let mut command = Command::new(&daemon);
    command
        .arg("--host")
        .arg(&options.host)
        .arg("--port")
        .arg(options.port.to_string());
    if options.mock {
        command.arg("--mock");
    }
    if let Some(state_dir) = &options.state_dir {
        command.env("HIVE_STATE_DIR", state_dir);
    }
    command.env(
        "HIVE_TOOL_URL",
        format!("http://{}:{}", options.host, options.port),
    );
    command.env("HIVE_CONSOLE_URL", &url);

    if !options.detach {
        let mut child = command
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("cannot start {}: {error}", daemon.display()))?;
        if !options.no_open {
            open_when_ready(url.clone());
        }
        let status = child
            .wait()
            .map_err(|error| format!("cannot wait for {}: {error}", daemon.display()))?;
        return Ok(if status.success() { 0 } else { 1 });
    }

    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| format!("cannot open {}: {error}", log_path.display()))?;
    let stderr = log
        .try_clone()
        .map_err(|error| format!("cannot clone the log handle: {error}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|error| format!("cannot start {}: {error}", daemon.display()))?;
    std::fs::write(&pid_path, format!("{}\n", child.id()))
        .map_err(|error| format!("cannot write {}: {error}", pid_path.display()))?;
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            let tail = std::fs::read_to_string(&log_path)
                .unwrap_or_default()
                .lines()
                .rev()
                .take(12)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n");
            return Err(format!(
                "daemon exited during startup ({status}); see {}:\n{tail}",
                log_path.display()
            ));
        }
        if alive(
            &format!("http://{}:{}", options.host, options.port),
            "/health",
        ) {
            if !options.no_open {
                open_browser(&url);
            }
            println!(
                "Hive daemon starting at {url} (pid {}; log {})",
                child.id(),
                log_path.display()
            );
            return Ok(0);
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let _ = child.kill();
    Err(format!(
        "daemon did not answer on {url}; see {}",
        log_path.display()
    ))
}

fn daemon_binary() -> Result<PathBuf, String> {
    if let Some(raw) = std::env::var_os("HIVE_DAEMON_BIN") {
        let path = PathBuf::from(raw);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "HIVE_DAEMON_BIN points at {}, which is not a file",
            path.display()
        ));
    }
    if let Ok(executable) = std::env::current_exe() {
        let sibling = executable.with_file_name("hive-daemon");
        if sibling.is_file() {
            return Ok(sibling);
        }
    }
    python::which("hive-daemon")
        .ok_or_else(|| "hive-daemon is not installed next to hive or on PATH".to_owned())
}

fn alive(base: &str, path: &str) -> bool {
    http::get(
        &format!("{}{}", base.trim_end_matches('/'), path),
        Duration::from_millis(800),
    )
    .map(|response| response.status == 200)
    .unwrap_or(false)
}

fn open_when_ready(url: String) {
    std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if alive(url.trim_end_matches('/'), "/health") {
                open_browser(&url);
                return;
            }
            std::thread::sleep(Duration::from_millis(300));
        }
    });
}

fn open_browser(url: &str) {
    for command in ["xdg-open", "open"] {
        if python::which(command).is_some()
            && Command::new(command)
                .arg(url)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .is_ok()
        {
            return;
        }
    }
}

#[allow(dead_code)]
fn _assert_writable(mut handle: impl Write) {
    let _ = handle.flush();
}
