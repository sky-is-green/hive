//! `hive serve`: apply a stack, keep it up, unload it on exit.

use std::fs::OpenOptions;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use chrono::Local;
use hive_core::backends::Manifest;
use hive_core::paths::Paths;
use hive_core::stacks;
use hive_core::{json, py_repr};
use hive_state::Record;
use hive_supervisor::Applied;
use serde_json::{Map, Value};

pub struct Options {
    pub stack: Option<String>,
    pub list: bool,
    pub backends: bool,
    pub model: Option<String>,
    pub file: Option<String>,
    #[allow(dead_code)]
    pub cards: Option<u32>,
    #[allow(dead_code)]
    pub devices: Option<String>,
    #[allow(dead_code)]
    pub ctx: u32,
    pub backend: Option<String>,
    pub fetch: bool,
    #[allow(dead_code)]
    pub role: String,
    pub save: Option<String>,
    pub name: Option<String>,
    pub detach: bool,
    pub replace: bool,
    pub llama_server: Option<PathBuf>,
    pub log_dir: Option<PathBuf>,
    pub host: String,
    pub timeout: f64,
    pub once: bool,
    pub as_json: bool,
    pub child: bool,
}

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_stop(_signal: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

pub fn run(paths: &Paths, options: Options) -> Result<u8, String> {
    if options.list {
        return list_stacks(paths, options.as_json);
    }
    if options.backends {
        return list_backends(paths, options.as_json);
    }
    if options.detach && options.once {
        eprintln!("hive serve: --detach and --once are mutually exclusive");
        return Ok(2);
    }
    if options.model.is_some() || options.file.is_some() || options.save.is_some() {
        eprintln!(
            "hive serve: generated stacks (--model/--file/--save) are not ported \
             yet; write a stack file and pass its name"
        );
        return Ok(2);
    }
    if options.fetch {
        eprintln!("hive serve: --fetch is not ported yet");
        return Ok(2);
    }
    let Some(target) = options.stack.as_deref() else {
        eprintln!("hive serve: a stack name, a path, or --model is required (or --list)");
        return Ok(2);
    };

    let (stack, stack_file) = match hive_supervisor::load_stack(target, &paths.stacks_dir) {
        Ok(pair) => pair,
        Err(error) => {
            eprintln!("hive serve: {error}");
            return Ok(2);
        }
    };
    let stack_name = stack
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("stack")
        .to_owned();
    let name = options.name.clone().unwrap_or_else(|| stack_name.clone());
    if let Err(error) = hive_state::validate_name(&name) {
        eprintln!("hive serve: {error}");
        return Ok(2);
    }
    if let Err(error) = check_existing(paths, &name, options.replace) {
        eprintln!("hive serve: {error}");
        return Ok(1);
    }

    if options.detach && !options.child {
        return detach(paths, &options, target, &name, &stack);
    }

    let run_options = hive_supervisor::Options {
        host: options.host.clone(),
        base_port: 1234,
        llama_server: options.llama_server.clone(),
        repo: paths.repo.clone(),
        models_dir: paths.models_dir.clone(),
        state_dir: paths.state_dir.clone(),
        log_dir: options
            .log_dir
            .clone()
            .unwrap_or_else(|| paths.log_dir.clone()),
        startup_timeout: options.timeout,
    };
    let applied = match hive_supervisor::apply(&stack, &run_options) {
        Ok(applied) => applied,
        Err(error) => {
            eprintln!("hive serve: {error}");
            return Ok(1);
        }
    };

    if options.once {
        print_status(
            &stack_name,
            &applied.tiers,
            &applied.warnings,
            options.as_json,
            &options.host,
        );
        applied.unload();
        if !options.as_json {
            println!("  unloaded");
        }
        return Ok(0);
    }

    let record = build_record(paths, &options, &name, &stack_name, stack_file, &applied);
    let written = match hive_state::write_record(&paths.serve_dir, &record) {
        Ok(_) => true,
        Err(error) => {
            eprintln!("hive serve: cannot write the serve record: {error}");
            false
        }
    };

    if options.as_json {
        print!("{}", json::pretty(&Value::Object(record.clone())));
    } else {
        println!("stack {} applied", py_repr(&stack_name));
        print_tiers(&applied.tiers, &options.host);
        println!(
            "  name {}  pid {}  {}",
            record.get("name").and_then(Value::as_str).unwrap_or("?"),
            record.get("pid").and_then(Value::as_i64).unwrap_or(0),
            record
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or("foreground")
        );
        if let Some(log_path) = record.get("log_path").and_then(Value::as_str) {
            println!("  log  {log_path}");
        }
        println!("  serving  Ctrl-C to stop");
    }

    install_signals();
    while !STOP.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(200));
    }
    applied.unload();
    if written {
        hive_state::remove_record(&paths.serve_dir, &name);
    }
    println!("stopped");
    Ok(0)
}

fn list_stacks(paths: &Paths, as_json: bool) -> Result<u8, String> {
    let rows = stacks::list_summaries(&paths.stacks_dir);
    if as_json {
        print!(
            "{}",
            json::pretty(&Value::Array(
                rows.iter().map(stacks::StackSummary::to_json).collect()
            ))
        );
        return Ok(0);
    }
    println!("stacks  {}", paths.stacks_dir.display());
    if rows.is_empty() {
        println!("  no saved stacks");
        return Ok(0);
    }
    for row in rows {
        println!("  {:<24} {}", row.name, row.roles.join(","));
    }
    Ok(0)
}

fn list_backends(paths: &Paths, as_json: bool) -> Result<u8, String> {
    let manifest = Manifest::load(paths.manifest.as_deref(), &paths.home)
        .map_err(|error| format!("backend manifest: {error}"))?;
    let rows = manifest.rows();
    if as_json {
        print!("{}", json::pretty(&Value::Array(rows)));
        return Ok(0);
    }
    println!("backends");
    for row in rows {
        println!(
            "  {:<10} {:<10} {:<24} {}",
            row.get("name").and_then(Value::as_str).unwrap_or(""),
            if row
                .get("installed")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                "installed"
            } else {
                "-"
            },
            row.get("accelerators")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default(),
            row.get("label").and_then(Value::as_str).unwrap_or("")
        );
    }
    Ok(0)
}

fn check_existing(paths: &Paths, name: &str, replace: bool) -> Result<(), String> {
    let record = match hive_state::read_record(&paths.serve_dir, name) {
        Ok(record) => record,
        Err(_) => {
            hive_state::remove_record(&paths.serve_dir, name);
            return Ok(());
        }
    };
    let Some(record) = record else {
        return Ok(());
    };
    if hive_state::pid_matches(&record) {
        if !replace {
            return Err(format!(
                "serve {} is already running (pid {}); use --replace or --name",
                py_repr(name),
                record.get("pid").and_then(Value::as_i64).unwrap_or(0)
            ));
        }
        let pid = record.get("pid").and_then(Value::as_i64).unwrap_or(0);
        let (ok, message) = hive_state::terminate(pid, false, 15.0);
        if !ok {
            return Err(format!("cannot replace serve {}: {message}", py_repr(name)));
        }
    }
    hive_state::remove_record(&paths.serve_dir, name);
    Ok(())
}

fn build_record(
    paths: &Paths,
    options: &Options,
    name: &str,
    stack_name: &str,
    stack_file: Option<PathBuf>,
    applied: &Applied,
) -> Record {
    let pid = std::process::id() as i64;
    let mut record = Map::new();
    record.insert("version".into(), Value::from(1));
    record.insert("name".into(), Value::String(name.to_owned()));
    record.insert("pid".into(), Value::from(pid));
    record.insert(
        "mode".into(),
        Value::String(
            if options.child {
                "detached"
            } else {
                "foreground"
            }
            .into(),
        ),
    );
    record.insert("stack".into(), Value::String(stack_name.to_owned()));
    record.insert(
        "stack_file".into(),
        stack_file
            .map(|path| Value::String(path.to_string_lossy().into_owned()))
            .unwrap_or(Value::Null),
    );
    record.insert("host".into(), Value::String(options.host.clone()));
    record.insert("started_at".into(), Value::from(hive_state::now()));
    record.insert(
        "started_iso".into(),
        Value::String(Local::now().format("%Y-%m-%dT%H:%M:%S%z").to_string()),
    );
    record.insert(
        "cmdline".into(),
        Value::String(hive_state::pid_cmdline(pid)),
    );
    record.insert(
        "log_path".into(),
        if options.child {
            hive_state::log_path(&paths.serve_dir, name)
                .map(|path| Value::String(path.to_string_lossy().into_owned()))
                .unwrap_or(Value::Null)
        } else {
            Value::Null
        },
    );
    record.insert(
        "log_dir".into(),
        Value::String(
            options
                .log_dir
                .clone()
                .unwrap_or_else(|| paths.log_dir.clone())
                .to_string_lossy()
                .into_owned(),
        ),
    );
    record.insert("tiers".into(), Value::Array(applied.tiers.clone()));
    record.insert(
        "warnings".into(),
        Value::Array(
            applied
                .warnings
                .iter()
                .cloned()
                .map(Value::String)
                .collect(),
        ),
    );
    record
}

fn detach(
    paths: &Paths,
    options: &Options,
    target: &str,
    name: &str,
    stack: &Value,
) -> Result<u8, String> {
    if let Err(error) = paths.ensure_serve_dir() {
        eprintln!(
            "hive serve: cannot create {}: {error}",
            paths.serve_dir.display()
        );
        return Ok(1);
    }
    let log = hive_state::log_path(&paths.serve_dir, name)
        .map_err(|error| format!("hive serve: {error}"))?;
    let handle = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log)
        .map_err(|error| format!("hive serve: cannot open {}: {error}", log.display()))?;
    let stderr = handle
        .try_clone()
        .map_err(|error| format!("hive serve: cannot clone the log handle: {error}"))?;

    let exe = std::env::current_exe()
        .map_err(|error| format!("hive serve: cannot find the executable: {error}"))?;
    let mut command = Command::new(exe);
    command
        .arg("serve")
        .arg(target)
        .arg("--name")
        .arg(name)
        .arg("--host")
        .arg(&options.host)
        .arg("--timeout")
        .arg(options.timeout.to_string())
        .arg("--stacks-dir")
        .arg(&paths.stacks_dir)
        .arg("--models-dir")
        .arg(&paths.models_dir)
        .arg("--_child");
    if let Some(log_dir) = &options.log_dir {
        command.arg("--log-dir").arg(log_dir);
    }
    if let Some(binary) = &options.llama_server {
        command.arg("--llama-server").arg(binary);
    }
    if let Some(backend) = &options.backend {
        command.arg("--backend").arg(backend);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(handle))
        .stderr(Stdio::from(stderr))
        .env("HIVE_SERVE_CHILD", "1");
    #[cfg(target_os = "linux")]
    unsafe {
        use std::os::unix::process::CommandExt;
        command.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("hive serve: cannot start the background serve: {error}"))?;
    let child_pid = child.id() as i64;
    let deadline =
        Instant::now() + Duration::from_secs_f64((options.timeout.max(10.0) + 15.0).max(10.0));
    let mut record = None;
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            let tail = hive_state::tail_lines(&log, 20).unwrap_or_default();
            eprintln!("hive serve: {} exited with code {status}", py_repr(name));
            eprintln!("  log  {}", log.display());
            for line in tail {
                eprintln!("  {line}");
            }
            return Ok(1);
        }
        if let Ok(Some(candidate)) = hive_state::read_record(&paths.serve_dir, name)
            && candidate.get("pid").and_then(Value::as_i64) == Some(child_pid)
        {
            record = Some(candidate);
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let Some(record) = record else {
        let _ = hive_state::terminate(child_pid, true, 5.0);
        let _ = child.wait();
        eprintln!(
            "hive serve: {} did not become ready within {:.0}s",
            py_repr(name),
            options.timeout
        );
        eprintln!("  log  {}", log.display());
        for line in hive_state::tail_lines(&log, 20).unwrap_or_default() {
            eprintln!("  {line}");
        }
        return Ok(1);
    };

    if options.as_json {
        print!("{}", json::pretty(&Value::Object(record)));
    } else {
        println!(
            "serving {} · pid {} · detached",
            py_repr(name),
            record.get("pid").and_then(Value::as_i64).unwrap_or(0)
        );
        let tiers = record
            .get("tiers")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        print_tiers(&tiers, &options.host);
        println!("  log  {}", log.display());
        for warning in stack
            .get("warnings")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            if let Some(warning) = warning.as_str() {
                println!("  ! {warning}");
            }
        }
    }
    Ok(0)
}

fn print_status(stack: &str, tiers: &[Value], warnings: &[String], as_json: bool, host: &str) {
    if as_json {
        let mut payload = Map::new();
        payload.insert("ok".into(), Value::Bool(true));
        payload.insert("stack".into(), Value::String(stack.into()));
        payload.insert("tiers".into(), Value::Array(tiers.to_vec()));
        payload.insert(
            "warnings".into(),
            Value::Array(warnings.iter().cloned().map(Value::String).collect()),
        );
        print!("{}", json::pretty(&Value::Object(payload)));
    } else {
        println!("stack {} applied", py_repr(stack));
        print_tiers(tiers, host);
        for warning in warnings {
            println!("  ! {warning}");
        }
    }
}

fn print_tiers(tiers: &[Value], host: &str) {
    let width = tiers
        .iter()
        .filter_map(|tier| tier.get("role").and_then(Value::as_str))
        .map(str::len)
        .max()
        .unwrap_or(4);
    for tier in tiers {
        let role = tier.get("role").and_then(Value::as_str).unwrap_or("?");
        let endpoint = tier
            .get("base_url")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| {
                format!(
                    "http://{host}:{}/v1",
                    tier.get("port").and_then(Value::as_i64).unwrap_or(0)
                )
            });
        println!("  {role:<width$}  {endpoint}");
        let ctx = tier.get("ctx").and_then(Value::as_i64);
        let backend = tier.get("backend").and_then(Value::as_str).unwrap_or("");
        let extra = match ctx {
            Some(ctx) if !backend.is_empty() => format!("{backend} ctx {ctx}"),
            Some(ctx) => format!("ctx {ctx}"),
            None => backend.to_owned(),
        };
        if !extra.is_empty() {
            println!("  {:width$}  {extra}", "");
        }
    }
}

fn install_signals() {
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGINT, on_stop as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_stop as *const () as libc::sighandler_t);
    }
}
