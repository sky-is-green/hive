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
        _ => status(paths, &options),
    }
}

fn read_descriptors(state_dir: &std::path::Path) -> Vec<(PathBuf, Value)> {
    let mut rows = Vec::new();
    if let Ok(entries) = std::fs::read_dir(state_dir.join("daemons")) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "json") {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&path)
                && let Ok(value) = serde_json::from_str::<Value>(&text)
            {
                rows.push((path, value));
            }
        }
    }
    let single = state_dir.join("daemon.json");
    if let Ok(text) = std::fs::read_to_string(&single)
        && let Ok(value) = serde_json::from_str::<Value>(&text)
        && !rows.iter().any(|(_, row)| {
            row.get("pid") == value.get("pid") && row.get("api_url") == value.get("api_url")
        })
    {
        rows.push((single, value));
    }
    rows.sort_by_key(|(_, value)| value.get("port").and_then(Value::as_u64).unwrap_or(0));
    rows
}

fn status(paths: &Paths, options: &Options) -> Cmd {
    let fallback_api = format!("http://{}:{}", options.host, options.port);
    let state_dir = options
        .state_dir
        .clone()
        .unwrap_or_else(|| paths.state_dir.clone());
    let descriptor_path = state_dir.join("daemon.json");
    let descriptors = read_descriptors(&state_dir);

    let mut daemons = Vec::new();
    let mut primary: Option<(String, String, Option<u64>, bool, bool)> = None;
    for (path, value) in &descriptors {
        let api_url = value
            .get("api_url")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| fallback_api.clone());
        let web_url = value
            .get("web_url")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| api_url.clone());
        let pid = value.get("pid").and_then(Value::as_u64);
        let process_alive = pid
            .map(|pid| hive_state::pid_alive(pid as i64))
            .unwrap_or(true);
        let api_alive = process_alive && alive(&api_url, "/health");
        let web_alive = process_alive && alive(&web_url, "/");
        let running = api_alive || web_alive;
        let entry = serde_json::json!({
            "pid": pid,
            "url": web_url,
            "web_url": web_url,
            "api_url": api_url,
            "running": running,
            "descriptor": path.to_string_lossy(),
        });
        daemons.push(entry);
        if primary.is_none() || (running && !primary.as_ref().map(|row| row.3).unwrap_or(false)) {
            primary = Some((web_url, api_url, pid, running, web_alive));
        }
    }

    let (web_url, api_url, pid, api_alive, web_alive) =
        if let Some((web_url, api_url, pid, _running, web_alive)) = primary {
            let api_alive = pid
                .map(|pid| hive_state::pid_alive(pid as i64))
                .unwrap_or(true)
                && alive(&api_url, "/health");
            (web_url, api_url, pid, api_alive, web_alive)
        } else {
            let api_alive = alive(&fallback_api, "/health");
            let web_alive = alive(&fallback_api, "/");
            (
                fallback_api.clone(),
                fallback_api,
                None,
                api_alive,
                web_alive,
            )
        };

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
    payload.insert("pid".into(), pid.map(Value::from).unwrap_or(Value::Null));
    payload.insert(
        "descriptor".into(),
        Value::String(descriptor_path.to_string_lossy().into_owned()),
    );
    payload.insert("daemons".into(), Value::Array(daemons));
    let payload = Value::Object(payload);
    if options.as_json {
        print!("{}", hive_core::json::pretty(&payload));
        return Ok(0);
    }
    let suffix = |running: bool, pid: Option<u64>| match (running, pid) {
        (true, Some(pid)) => format!("running · pid {pid}"),
        (true, None) => "running".to_owned(),
        _ => "not running".to_owned(),
    };
    if let Some(entries) = payload.get("daemons").and_then(Value::as_array)
        && entries.len() > 1
    {
        println!("daemons");
        for entry in entries {
            let url = entry.get("url").and_then(Value::as_str).unwrap_or("?");
            let running = entry
                .get("running")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let pid = entry.get("pid").and_then(Value::as_u64);
            println!("  {url} · {}", suffix(running, pid));
        }
        println!("  terminal  hive ui terminal");
        return Ok(0);
    }
    println!("interfaces");
    println!("  web    {} · {}", web_url, suffix(web_alive, pid));
    println!("  api    {} · {}", api_url, suffix(api_alive, pid));
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
    let deadline = Instant::now() + Duration::from_secs(60);
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

fn health_ready(base: &str) -> bool {
    http::get(
        &format!("{}/health", base.trim_end_matches('/')),
        Duration::from_millis(800),
    )
    .ok()
    .and_then(|response| response.json().ok())
    .and_then(|payload| payload.get("ok").and_then(serde_json::Value::as_bool))
    .unwrap_or(false)
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
            if health_ready(&url) {
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
