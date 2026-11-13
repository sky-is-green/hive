//! `hive serve`: apply a stack, keep it up, unload it on exit.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
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
    if options.model.is_some() && options.stack.is_some() {
        eprintln!("hive serve: give either a stack name or --model, not both");
        return Ok(2);
    }
    if options.model.is_none() && (options.file.is_some() || options.save.is_some()) {
        eprintln!("hive serve: --file/--save require --model");
        return Ok(2);
    }

    let (stack, stack_file, target, generated_warnings) =
        if let Some(model) = options.model.as_deref() {
            match generate_stack(paths, &options, model) {
                Ok((stack, saved, warnings)) => {
                    let target = saved
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned());
                    (stack, None, target, warnings)
                }
                Err(error) => {
                    eprintln!("hive serve: {error}");
                    return Ok(2);
                }
            }
        } else {
            let Some(target) = options.stack.as_deref() else {
                eprintln!("hive serve: a stack name, a path, or --model is required (or --list)");
                return Ok(2);
            };
            match hive_supervisor::load_stack(target, &paths.stacks_dir) {
                Ok((stack, stack_file)) => (stack, stack_file, Some(target.to_owned()), Vec::new()),
                Err(error) => {
                    eprintln!("hive serve: {error}");
                    return Ok(2);
                }
            }
        };
    let stack_name = stack
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("stack")
        .to_owned();
    if options.fetch
        && let Err(error) = ensure_fetch(paths, &options, &stack)
    {
        eprintln!("hive serve: {error}");
        return Ok(1);
    }
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
        let Some(target) = target.as_deref() else {
            eprintln!(
                "hive serve: --model --detach needs --save so the background \
                 process can reload the generated stack"
            );
            return Ok(2);
        };
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
    let mut applied = match hive_supervisor::apply(&stack, &run_options) {
        Ok(applied) => applied,
        Err(error) => {
            eprintln!("hive serve: {error}");
            return Ok(1);
        }
    };
    applied.warnings.extend(generated_warnings);

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

fn ensure_fetch(paths: &Paths, options: &Options, stack: &Value) -> Result<(), String> {
    let manifest = Manifest::load(paths.manifest.as_deref(), &paths.home)
        .map_err(|error| format!("backend manifest: {error}"))?;
    let tiers = stack
        .get("tiers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut missing = Vec::new();
    for tier in tiers {
        if tier
            .get("endpoint")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .is_some()
        {
            continue;
        }
        let name = tier
            .get("backend")
            .and_then(Value::as_str)
            .unwrap_or("vulkan")
            .to_owned();
        if missing.contains(&name) {
            continue;
        }
        let backend = manifest.resolve(&name).map_err(|error| error.to_string())?;
        if !backend.installed {
            missing.push(name);
        }
    }
    for name in missing {
        if !options.as_json {
            eprintln!("fetching {name} ...");
        }
        let mut payload = None;
        match crate::backends_cmd::fetch_capture(
            paths,
            Some(&name),
            None,
            false,
            options.as_json,
            false,
            &mut payload,
        ) {
            Ok(0) => {
                if !options.as_json {
                    eprintln!("installed {name}");
                }
            }
            Err(error) => {
                let detail = error
                    .strip_prefix("hive backends fetch: ")
                    .unwrap_or(&error);
                return Err(format!("cannot fetch backend {name:?}: {detail}"));
            }
            Ok(code) => return Err(format!("cannot fetch backend {name:?} (exit {code})")),
        }
    }
    Ok(())
}

fn generate_stack(
    paths: &Paths,
    options: &Options,
    model: &str,
) -> Result<(Value, Option<PathBuf>, Vec<String>), String> {
    let candidate = hive_core::paths::expand(Path::new(model));
    let remote_ref = !candidate
        .extension()
        .map(|suffix| suffix.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
        || model
            .split_once(':')
            .map(|(repo, file)| !file.is_empty() && repo.contains('/'))
            .unwrap_or(false);
    let model_path = if candidate.is_file() {
        if options.file.is_some() {
            return Err("--file is only for a HuggingFace repo, not a local path".into());
        }
        candidate
            .canonicalize()
            .map_err(|error| format!("cannot resolve {}: {error}", candidate.display()))?
    } else if !remote_ref {
        return Err(format!("no such GGUF file: {}", candidate.display()));
    } else {
        let (repo, file) = match options.file.as_deref() {
            Some(file) => (model.to_owned(), file.to_owned()),
            None => match model.split_once(':') {
                Some((repo, file)) if !file.is_empty() && repo.contains('/') => {
                    (repo.to_owned(), file.to_owned())
                }
                _ => {
                    let rows = crate::models_cmd::hub_files(model)
                        .map_err(|error| format!("cannot list {model}: {error}"))?;
                    if rows.is_empty() {
                        return Err(format!("{model} has no .gguf files"));
                    }
                    if rows.len() > 1 {
                        let names = rows
                            .iter()
                            .filter_map(|row| row.get("file").and_then(Value::as_str))
                            .collect::<Vec<_>>()
                            .join(", ");
                        return Err(format!(
                            "{model} has several GGUFs; choose one with --file: {names}"
                        ));
                    }
                    let file = rows[0]
                        .get("file")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    (model.to_owned(), file)
                }
            },
        };
        let mut payload = None;
        match crate::models_cmd::download_capture(
            paths,
            &repo,
            Some(&file),
            "main",
            options.as_json,
            false,
            &mut payload,
        ) {
            Ok(0) => {}
            Ok(_) => return Err(format!("cannot download {repo}:{file}")),
            Err(error) => return Err(error),
        }
        let Some(path) = payload
            .as_ref()
            .and_then(|payload| payload.get("path"))
            .and_then(Value::as_str)
        else {
            return Err(format!(
                "cannot resolve the downloaded path for {repo}:{file}"
            ));
        };
        let path = PathBuf::from(path);
        path.canonicalize()
            .map_err(|error| format!("cannot resolve {}: {error}", path.display()))?
    };
    if options.ctx == 0 {
        return Err("--ctx must be positive".into());
    }

    let all_devices = hive_supervisor::residency::visible_devices();
    let devices = select_devices(&all_devices, options.cards, options.devices.as_deref())?;
    let manifest = Manifest::load(paths.manifest.as_deref(), &paths.home).ok();
    let backend = if let Some(name) = &options.backend {
        if let Some(manifest) = &manifest {
            let resolved = manifest.resolve(name).map_err(|error| error.to_string())?;
            if !resolved.installed {
                return Err(format!(
                    "backend {name:?} is not installed ({}); run `hive backends \
                     fetch {name}` or add --fetch",
                    resolved.binary.display()
                ));
            }
        }
        Some(name.clone())
    } else {
        manifest.as_ref().and_then(|manifest| {
            manifest
                .for_accelerator(if devices.is_empty() { "cpu" } else { "amd" })
                .map(str::to_owned)
        })
    };

    let mut tier = Map::new();
    tier.insert("role".into(), Value::String(options.role.clone()));
    tier.insert("repo".into(), Value::String("local".into()));
    tier.insert(
        "file".into(),
        Value::String(model_path.to_string_lossy().into_owned()),
    );
    tier.insert("ctx".into(), Value::from(options.ctx));
    tier.insert("ngl".into(), Value::from(99));
    if let Some(backend) = backend {
        tier.insert("backend".into(), Value::String(backend));
    }
    tier.insert("cache_k".into(), Value::String("q8_0".into()));
    tier.insert("cache_v".into(), Value::String("q8_0".into()));

    let mut warnings = Vec::new();
    if !devices.is_empty() {
        let pin = devices
            .iter()
            .map(|device| device.index.to_string())
            .collect::<Vec<_>>()
            .join(",");
        tier.insert("pin".into(), Value::String(pin));
        for device in &devices {
            if device.display {
                warnings.push(format!(
                    "card {} drives a display; expect desktop stutter",
                    device.index
                ));
            }
        }
        if devices.len() > 1 {
            let memory: Vec<f64> = devices
                .iter()
                .map(|device| device.memory_gb.unwrap_or(1.0))
                .collect();
            let min = memory.iter().copied().fold(f64::INFINITY, f64::min);
            let max = memory.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let ts = if max - min < 0.5 {
                vec!["1"; memory.len()].join(",")
            } else {
                memory
                    .iter()
                    .map(|value| format!("{value:.0}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            tier.insert("ts".into(), Value::String(ts));
        }
    }

    let stem = model_path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let cards = if devices.is_empty() {
        "cpu".to_owned()
    } else {
        format!("{}card", devices.len())
    };
    let name = options
        .save
        .clone()
        .or_else(|| options.name.clone())
        .unwrap_or_else(|| slug(&format!("{stem}-{cards}")));
    let mut data = Map::new();
    data.insert("name".into(), Value::String(name.clone()));
    data.insert("version".into(), Value::from(1));
    data.insert("tiers".into(), Value::Array(vec![Value::Object(tier)]));
    let stack = hive_core::stacks::parse_stack(&Value::Object(data.clone()))
        .map_err(|error| error.to_string())?;

    let mut saved = None;
    if let Some(target) = options.save.as_deref() {
        let name =
            hive_core::stacks::check_stack_name(if target.is_empty() { &name } else { target })?;
        let path = paths.stacks_dir.join(format!("{name}.json"));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        std::fs::write(&path, json::pretty(&Value::Object(data)))
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        if !options.as_json {
            println!("wrote {}", path.display());
        }
        saved = Some(path);
    }
    Ok((stack, saved, warnings))
}

fn select_devices(
    devices: &[hive_core::hardware::Device],
    count: Option<u32>,
    requested: Option<&str>,
) -> Result<Vec<hive_core::hardware::Device>, String> {
    if let Some(requested) = requested {
        let wanted: Vec<usize> = requested
            .replace(' ', "")
            .split(',')
            .filter(|part| !part.is_empty())
            .map(|part| {
                part.parse::<usize>()
                    .map_err(|_| "--devices must be comma-separated HIP indices, e.g. 0,1")
            })
            .collect::<Result<_, _>>()?;
        let by_index: std::collections::BTreeMap<usize, &hive_core::hardware::Device> = devices
            .iter()
            .map(|device| (device.index, device))
            .collect();
        let missing: Vec<usize> = wanted
            .iter()
            .copied()
            .filter(|index| !by_index.contains_key(index))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "--devices {missing:?} not among the detected cards {:?}",
                by_index.keys().collect::<Vec<_>>()
            ));
        }
        return Ok(wanted
            .into_iter()
            .filter_map(|index| by_index.get(&index).map(|device| (**device).clone()))
            .collect());
    }
    if let Some(count) = count
        && count as usize > devices.len()
    {
        return Err(format!(
            "--cards {count} but only {} GPU(s) detected",
            devices.len()
        ));
    }
    let take = count.map(|count| count as usize).unwrap_or(devices.len());
    Ok(devices.iter().take(take).cloned().collect())
}

fn slug(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut dash = false;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            out.push(ch);
            dash = false;
        } else if !dash {
            out.push('-');
            dash = true;
        }
    }
    let trimmed = out.trim_matches(|ch| ch == '-' || ch == '.').to_owned();
    if trimmed.is_empty() {
        "stack".into()
    } else {
        trimmed.chars().take(64).collect()
    }
}

fn install_signals() {
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGINT, on_stop as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_stop as *const () as libc::sighandler_t);
    }
}
