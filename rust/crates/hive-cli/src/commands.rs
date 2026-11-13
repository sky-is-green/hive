use std::collections::HashSet;
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use hive_core::backends::Manifest;
use hive_core::paths::Paths;
use hive_core::{hardware, json, platform_display, py_repr, python, stacks, version};
use hive_models::Library;
use hive_state::Record;
use serde_json::{Map, Number, Value};

use crate::probe::probe;
use crate::{BackendsCommand, Cli, Command, ModelsCommand, StacksCommand};

type Cmd = Result<u8, String>;

pub fn run(cli: Cli) -> ExitCode {
    let (models_dir, stacks_dir) = overrides(&cli.command);
    let paths = Paths::discover(models_dir.as_deref(), stacks_dir.as_deref());
    let command = cli.command;
    let result = match command {
        None => {
            if hive_tui::available() {
                run_tui(&paths)
            } else {
                home(&paths)
            }
        }
        Some(Command::Tui) => run_tui(&paths),
        Some(Command::Ui {
            mode,
            host,
            port,
            console_port: _,
            detach,
            no_open,
            mock,
            state_dir,
            json: as_json,
        }) => crate::ui::run(
            &paths,
            crate::ui::Options {
                mode,
                host,
                port,
                detach,
                no_open,
                mock,
                state_dir,
                as_json,
            },
        ),
        Some(Command::Serve {
            stack,
            list,
            backends,
            model,
            file,
            cards,
            devices,
            ctx,
            backend,
            fetch,
            role,
            save,
            name,
            detach,
            replace,
            stacks_dir: _,
            models_dir: _,
            llama_server,
            log_dir,
            host,
            timeout,
            once,
            json: as_json,
            child,
        }) => crate::serve::run(
            &paths,
            crate::serve::Options {
                stack,
                list,
                backends,
                model,
                file,
                cards,
                devices,
                ctx,
                backend,
                fetch,
                role,
                save,
                name,
                detach,
                replace,
                llama_server,
                log_dir,
                host,
                timeout,
                once,
                as_json,
                child,
            },
        ),
        Some(Command::Version { json: as_json }) => cmd_version(&paths, as_json),
        Some(Command::Config { json: as_json, .. }) => cmd_config(&paths, as_json),
        Some(Command::Setup {
            no_fetch,
            json: as_json,
            ..
        }) => crate::backends_cmd::setup(&paths, no_fetch, as_json),
        Some(Command::Doctor {
            strict,
            json: as_json,
            ..
        }) => cmd_doctor(&paths, strict, as_json),
        Some(Command::Models { command }) => match command {
            ModelsCommand::List { json: as_json, .. } => cmd_models_list(&paths, as_json),
            ModelsCommand::Search {
                query,
                limit,
                json: as_json,
                ..
            } => crate::models_cmd::search(&query, limit, as_json),
            ModelsCommand::Files {
                repo,
                json: as_json,
                ..
            } => crate::models_cmd::files(&repo, as_json),
            ModelsCommand::Dir {
                path,
                set,
                json: as_json,
                ..
            } => crate::models_cmd::dir(&paths, path.as_deref(), set.as_deref(), as_json),
            ModelsCommand::Download {
                repo,
                file,
                revision,
                json: as_json,
                ..
            } => crate::models_cmd::download(&paths, &repo, file.as_deref(), &revision, as_json),
            ModelsCommand::Import {
                path,
                copy,
                json: as_json,
                ..
            } => crate::models_cmd::import(&paths, &path, copy, as_json),
            ModelsCommand::Remove {
                file,
                yes,
                json: as_json,
                ..
            } => crate::models_cmd::remove(&paths, &file, yes, as_json),
        },
        Some(Command::Stacks { command }) => match command {
            StacksCommand::List { json: as_json, .. } => cmd_stacks_list(&paths, as_json),
            StacksCommand::Show { stack, .. } => crate::stacks_cmd::show(&paths, &stack),
            StacksCommand::Validate {
                stack,
                strict,
                json: as_json,
                ..
            } => crate::stacks_cmd::validate(&paths, &stack, strict, as_json),
        },
        Some(Command::Backends {
            json: as_json,
            command,
        }) => match command {
            None => crate::backends_cmd::list(&paths, as_json),
            Some(BackendsCommand::List { json: sub_json }) => {
                crate::backends_cmd::list(&paths, sub_json || as_json)
            }
            Some(BackendsCommand::Fetch {
                name,
                tag,
                force,
                json: sub_json,
            }) => crate::backends_cmd::fetch(
                &paths,
                name.as_deref(),
                tag.as_deref(),
                force,
                sub_json || as_json,
            ),
        },
        Some(Command::Status {
            json: as_json,
            no_probe,
            prune,
        }) => cmd_status(&paths, as_json, no_probe, prune),
        Some(Command::Stop {
            name,
            all,
            force,
            timeout,
            json: as_json,
        }) => cmd_stop(&paths, name.as_deref(), all, force, timeout, as_json),
        Some(Command::Logs {
            name,
            tier,
            lines,
            follow,
            json: as_json,
        }) => cmd_logs(
            &paths,
            name.as_deref(),
            tier.as_deref(),
            lines,
            follow,
            as_json,
        ),
        Some(Command::Help {
            topic,
            json: as_json,
        }) => crate::help::run(topic, as_json),
    };
    match result {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("hive: {error}");
            ExitCode::FAILURE
        }
    }
}

/// The per-command `--models-dir`/`--stacks-dir` overrides.
fn overrides(command: &Option<Command>) -> (Option<PathBuf>, Option<PathBuf>) {
    match command {
        Some(Command::Serve {
            models_dir,
            stacks_dir,
            ..
        }) => (models_dir.clone(), stacks_dir.clone()),
        Some(Command::Models {
            command:
                ModelsCommand::List { models_dir, .. }
                | ModelsCommand::Search { models_dir, .. }
                | ModelsCommand::Files { models_dir, .. }
                | ModelsCommand::Dir { models_dir, .. }
                | ModelsCommand::Download { models_dir, .. }
                | ModelsCommand::Import { models_dir, .. }
                | ModelsCommand::Remove { models_dir, .. },
        }) => (models_dir.clone(), None),
        Some(Command::Stacks {
            command: StacksCommand::List { stacks_dir, .. } | StacksCommand::Show { stacks_dir, .. },
        }) => (None, stacks_dir.clone()),
        Some(Command::Stacks {
            command:
                StacksCommand::Validate {
                    models_dir,
                    stacks_dir,
                    ..
                },
        }) => (models_dir.clone(), stacks_dir.clone()),
        Some(Command::Doctor {
            models_dir,
            stacks_dir,
            ..
        })
        | Some(Command::Config {
            models_dir,
            stacks_dir,
            ..
        })
        | Some(Command::Setup {
            models_dir,
            stacks_dir,
            ..
        }) => (models_dir.clone(), stacks_dir.clone()),
        _ => (None, None),
    }
}

fn print_json(value: &Value) {
    print!("{}", json::pretty(value));
}

fn home(paths: &Paths) -> Cmd {
    let records = hive_state::list_records(&paths.serve_dir);
    let running = records
        .iter()
        .filter(|record| hive_state::pid_matches(record))
        .count();
    let stale = records.len().saturating_sub(running);
    let models = Library::new(&paths.models_dir);
    let model_count = models.count();
    let stack_count = stacks::list_summaries(&paths.stacks_dir).len();
    let installed = Manifest::load(paths.manifest.as_deref(), &paths.home)
        .map(|manifest| manifest.installed().len())
        .unwrap_or(0);
    let mut run_text = if running == 0 {
        "none".to_owned()
    } else {
        running.to_string()
    };
    if stale > 0 {
        run_text.push_str(&format!(" · {stale} stale"));
    }
    println!("hive {}", version());
    println!("  running  {run_text}");
    println!("  models   {model_count}");
    println!("  stacks   {stack_count}");
    println!("  backends {installed}");
    println!();
    println!("  hive doctor");
    println!("  hive serve peer-2tier --detach");
    println!("  hive status");
    println!("  hive models list");
    Ok(0)
}

fn run_tui(paths: &Paths) -> Cmd {
    match hive_tui::run(paths) {
        Ok(()) => Ok(0),
        Err(error) => {
            eprintln!("hive tui: {error}");
            Ok(1)
        }
    }
}

fn cmd_version(paths: &Paths, as_json: bool) -> Cmd {
    if as_json {
        let python_path = python::find(paths);
        let mut payload = Map::new();
        payload.insert("hive".into(), Value::String(version().to_owned()));
        payload.insert(
            "python".into(),
            Value::String(python::version(&python_path)),
        );
        payload.insert("platform".into(), Value::String(platform_display()));
        print_json(&Value::Object(payload));
    } else {
        println!("hive {}", version());
    }
    Ok(0)
}

fn cmd_config(paths: &Paths, as_json: bool) -> Cmd {
    paths
        .ensure_models_dir()
        .map_err(|error| format!("cannot use the models directory: {error}"))?;
    let manifest = Manifest::load(paths.manifest.as_deref(), &paths.home).ok();
    let installed = manifest
        .as_ref()
        .map(|manifest| manifest.installed())
        .unwrap_or_default();
    let default_backend = manifest.as_ref().and_then(|manifest| {
        manifest
            .for_accelerator("amd")
            .or_else(|| manifest.for_accelerator("cpu"))
            .map(str::to_owned)
    });
    let python_path = python::find(paths);

    let mut payload = Map::new();
    payload.insert("hive".into(), Value::String(version().to_owned()));
    payload.insert(
        "python".into(),
        Value::String(python_path.to_string_lossy().into_owned()),
    );
    payload.insert(
        "repo".into(),
        Value::String(paths.repo.to_string_lossy().into_owned()),
    );
    payload.insert(
        "models_dir".into(),
        Value::String(paths.models_dir.to_string_lossy().into_owned()),
    );
    payload.insert(
        "stacks_dir".into(),
        Value::String(paths.stacks_dir.to_string_lossy().into_owned()),
    );
    payload.insert(
        "state_dir".into(),
        Value::String(paths.state_dir.to_string_lossy().into_owned()),
    );
    payload.insert(
        "serve_dir".into(),
        Value::String(paths.serve_dir.to_string_lossy().into_owned()),
    );
    payload.insert(
        "log_dir".into(),
        Value::String(paths.log_dir.to_string_lossy().into_owned()),
    );
    payload.insert(
        "default_backend".into(),
        default_backend.map(Value::String).unwrap_or(Value::Null),
    );
    payload.insert(
        "installed_backends".into(),
        Value::Array(
            installed
                .into_iter()
                .map(|name| Value::String(name.to_owned()))
                .collect(),
        ),
    );
    if as_json {
        print_json(&Value::Object(payload));
    } else {
        println!("config");
        for (key, value) in payload {
            let text = match value {
                Value::String(text) => text,
                Value::Null => "-".to_owned(),
                Value::Array(items) => items
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", "),
                other => other.to_string(),
            };
            println!("  {key:<20} {text}");
        }
    }
    Ok(0)
}

fn cmd_doctor(paths: &Paths, strict: bool, as_json: bool) -> Cmd {
    let mut checks: Vec<Value> = Vec::new();
    let mut add = |status: &str, name: &str, detail: String| {
        let mut check = Map::new();
        check.insert("status".into(), Value::String(status.into()));
        check.insert("name".into(), Value::String(name.into()));
        check.insert("detail".into(), Value::String(detail));
        checks.push(Value::Object(check));
    };

    let python_path = python::find(paths);
    let python_version = python::version(&python_path);
    add(
        if python::meets_minimum(&python_version, (3, 10)) {
            "ok"
        } else {
            "fail"
        },
        "python",
        format!("{python_version} ({})", python_path.display()),
    );
    add("ok", "hive", version().to_owned());
    add("ok", "platform", platform_display());

    let models_dir = &paths.models_dir;
    let _ = paths.ensure_models_dir();
    if writable(models_dir) {
        let count = Library::new(models_dir).count();
        let free = hardware::disk_free_gib(models_dir).unwrap_or(0.0);
        add(
            "ok",
            "models",
            format!(
                "{} · {count} GGUF · {free:.1} GiB free",
                models_dir.display()
            ),
        );
    } else {
        add(
            "fail",
            "models",
            format!("{} · not writable", models_dir.display()),
        );
    }

    let stack_count = std::fs::read_dir(&paths.stacks_dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
                .count()
        })
        .unwrap_or(0);
    add(
        if paths.stacks_dir.is_dir() {
            "ok"
        } else {
            "warn"
        },
        "stacks",
        format!("{} · {stack_count} stack(s)", paths.stacks_dir.display()),
    );

    let _ = paths.ensure_serve_dir();
    add(
        if writable(&paths.serve_dir) {
            "ok"
        } else {
            "fail"
        },
        "state",
        paths.serve_dir.display().to_string(),
    );

    match Manifest::load(paths.manifest.as_deref(), &paths.home) {
        Ok(manifest) => {
            let installed = manifest.installed();
            if installed.is_empty() {
                add(
                    "fail",
                    "backends",
                    "none installed; run `hive backends fetch vulkan` \
                     (or see `hive backends`)"
                        .into(),
                );
            } else {
                add("ok", "backends", installed.join(", "));
            }
        }
        Err(error) => add("fail", "backends", format!("backend manifest: {error}")),
    }

    let devices = match hardware::amd_devices() {
        Ok(devices) => {
            if devices.is_empty() {
                add("warn", "cards", "no amdgpu cards found via sysfs".into());
            } else {
                let detail = devices
                    .iter()
                    .map(|device| device.detail())
                    .collect::<Vec<_>>()
                    .join("; ");
                add("ok", "cards", detail);
            }
            devices
        }
        Err(error) => {
            add(
                "warn",
                "cards",
                format!("cannot read amdgpu sysfs: {error}"),
            );
            Vec::new()
        }
    };

    if let Some((available, total)) = hardware::memory_gib() {
        let mut detail = format!("{available:.1}/{total:.1} GiB free");
        let gtt_used: Vec<f64> = devices.iter().filter_map(|d| d.gtt_used_gb).collect();
        let gtt_total: Vec<f64> = devices.iter().filter_map(|d| d.gtt_total_gb).collect();
        if !gtt_used.is_empty() && !gtt_total.is_empty() {
            detail.push_str(&format!(
                " · GTT {:.1}/{:.1} GiB",
                gtt_used.iter().sum::<f64>(),
                gtt_total.iter().sum::<f64>()
            ));
        }
        add("ok", "memory", detail);
    } else {
        add("warn", "memory", "cannot read host memory".into());
    }

    match hardware::rocm_tool() {
        Some(tool) => add("ok", "rocm", tool.display().to_string()),
        None if !devices.is_empty() => add("warn", "rocm", "rocminfo/rocm-smi not in PATH".into()),
        None => add(
            "warn",
            "rocm",
            "not found (only needed for a ROCm backend)".into(),
        ),
    }

    if hardware::has_hf_token() {
        add("ok", "hf-token", "set".into());
    } else {
        add(
            "warn",
            "hf-token",
            "not set (only needed for gated/private repos)".into(),
        );
    }

    for port in [1234u16, 8150u16] {
        if hardware::port_free(port) {
            add("ok", &format!("port {port}"), "free".into());
        } else {
            add(
                "warn",
                &format!("port {port}"),
                "in use (a serve may be running)".into(),
            );
        }
    }

    let failures = checks
        .iter()
        .filter(|check| check.get("status").and_then(Value::as_str) == Some("fail"))
        .count();
    let warnings = checks
        .iter()
        .filter(|check| check.get("status").and_then(Value::as_str) == Some("warn"))
        .count();
    let ok = failures == 0 && !(strict && warnings > 0);
    if as_json {
        let mut summary = Map::new();
        summary.insert("ok".into(), Value::from(checks.len() - failures - warnings));
        summary.insert("warn".into(), Value::from(warnings));
        summary.insert("fail".into(), Value::from(failures));
        let mut payload = Map::new();
        payload.insert("ok".into(), Value::Bool(ok));
        payload.insert("checks".into(), Value::Array(checks));
        payload.insert("summary".into(), Value::Object(summary));
        print_json(&Value::Object(payload));
    } else {
        println!("doctor  {}", platform_display());
        let width = checks
            .iter()
            .filter_map(|check| check.get("name").and_then(Value::as_str))
            .map(str::len)
            .max()
            .unwrap_or(0);
        for check in &checks {
            let status = check.get("status").and_then(Value::as_str).unwrap_or("");
            let name = check.get("name").and_then(Value::as_str).unwrap_or("");
            let detail = check.get("detail").and_then(Value::as_str).unwrap_or("");
            let mark = match status {
                "ok" => "✔",
                "warn" => "!",
                _ => "✘",
            };
            println!("  {mark} {name:<width$}  {detail}");
        }
        println!();
        if failures > 0 {
            println!("  {failures} failure(s) · fix them before serving");
        } else if strict && warnings > 0 {
            println!("  {warnings} warning(s) · --strict");
        } else if warnings > 0 {
            println!("  ready · {warnings} warning(s)");
        } else {
            println!("  ready");
        }
    }
    Ok(if ok { 0 } else { 1 })
}

fn writable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
            return false;
        };
        unsafe { libc::access(c_path.as_ptr(), libc::W_OK) == 0 }
    }
    #[cfg(not(unix))]
    {
        path.is_dir()
    }
}

fn cmd_models_list(paths: &Paths, as_json: bool) -> Cmd {
    paths
        .ensure_models_dir()
        .map_err(|error| format!("cannot use the models directory: {error}"))?;
    let rows = Library::new(&paths.models_dir).rows();
    if as_json {
        print_json(&Value::Array(rows));
        return Ok(0);
    }
    println!("models  {}", paths.models_dir.display());
    if rows.is_empty() {
        println!("  no GGUF models here yet");
        println!("  hive models download <repo>[:<file>]");
        return Ok(0);
    }
    for row in &rows {
        println!(
            "  {:<28} {:>8}  {:<10} {:<8} {}",
            row.get("name").and_then(Value::as_str).unwrap_or(""),
            format!(
                "{:.2} GiB",
                row.get("size_gb").and_then(Value::as_f64).unwrap_or(0.0)
            ),
            row.get("quantization")
                .and_then(Value::as_str)
                .unwrap_or("-"),
            row.get("context_length")
                .map(|ctx| ctx.to_string())
                .unwrap_or_else(|| "-".into()),
            row.get("file").and_then(Value::as_str).unwrap_or(""),
        );
    }
    println!("\n  {} model(s)", rows.len());
    Ok(0)
}

fn cmd_stacks_list(paths: &Paths, as_json: bool) -> Cmd {
    let rows = stacks::list_summaries(&paths.stacks_dir);
    if as_json {
        print_json(&Value::Array(
            rows.iter().map(stacks::StackSummary::to_json).collect(),
        ));
        return Ok(0);
    }
    println!("stacks  {}", paths.stacks_dir.display());
    if rows.is_empty() {
        println!("  no saved stacks here yet");
        return Ok(0);
    }
    for row in rows {
        println!(
            "  {:<24} {:>2}  {}",
            row.name,
            row.tiers,
            row.roles.join(",")
        );
    }
    Ok(0)
}

fn cmd_status(paths: &Paths, as_json: bool, no_probe: bool, prune: bool) -> Cmd {
    let records = hive_state::list_records(&paths.serve_dir);
    let mut pruned = Vec::new();
    let mut views: Vec<Record> = Vec::new();
    for mut record in records {
        let pid = record.get("pid").and_then(Value::as_i64).unwrap_or(0);
        let alive = hive_state::pid_alive(pid);
        let matches = if alive {
            hive_state::pid_matches(&record)
        } else {
            false
        };
        if prune && !matches {
            if let Some(name) = record.get("name").and_then(Value::as_str) {
                hive_state::remove_record(&paths.serve_dir, name);
                pruned.push(Value::String(name.to_owned()));
            }
            continue;
        }
        let started = record
            .get("started_at")
            .and_then(Value::as_f64)
            .unwrap_or_else(hive_state::now);
        let uptime = (hive_state::now() - started).max(0.0);
        record.insert("alive".into(), Value::Bool(alive));
        record.insert("matches".into(), Value::Bool(matches));
        record.insert(
            "state".into(),
            Value::String(if matches { "running" } else { "stale" }.into()),
        );
        record.insert("uptime_s".into(), number(round1(uptime)));
        record.insert(
            "tiers".into(),
            match record.get("tiers") {
                Some(Value::Array(tiers)) => Value::Array(tiers.clone()),
                _ => Value::Array(Vec::new()),
            },
        );
        views.push(record);
    }

    if !no_probe {
        let mut endpoints: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for view in &views {
            if view.get("state").and_then(Value::as_str) != Some("running") {
                continue;
            }
            if let Some(tiers) = view.get("tiers").and_then(Value::as_array) {
                for tier in tiers {
                    let endpoint = tier_endpoint(view, tier);
                    if !endpoint.is_empty() && seen.insert(endpoint.clone()) {
                        endpoints.push(endpoint);
                    }
                }
            }
        }
        if !endpoints.is_empty() {
            let timeout = Duration::from_millis(800);
            let mut results: Vec<(String, Value)> = Vec::new();
            std::thread::scope(|scope| {
                let handles: Vec<_> = endpoints
                    .iter()
                    .map(|endpoint| {
                        let endpoint = endpoint.clone();
                        scope.spawn(move || {
                            let answer = probe(&endpoint, timeout);
                            (endpoint, answer)
                        })
                    })
                    .collect();
                for handle in handles {
                    if let Ok(result) = handle.join() {
                        results.push(result);
                    }
                }
            });
            let probes: std::collections::BTreeMap<String, Value> = results.into_iter().collect();
            for view in &mut views {
                if view.get("state").and_then(Value::as_str) != Some("running") {
                    continue;
                }
                let host = view
                    .get("host")
                    .and_then(Value::as_str)
                    .unwrap_or("127.0.0.1")
                    .to_owned();
                let Some(Value::Array(tiers)) = view.get_mut("tiers") else {
                    continue;
                };
                for tier in tiers {
                    let endpoint = tier_endpoint_with_host(tier, &host);
                    if let Some(answer) = probes.get(&endpoint)
                        && let Some(tier) = tier.as_object_mut()
                    {
                        tier.insert("probe".into(), answer.clone());
                    }
                }
            }
        }
    }

    if as_json {
        let mut payload = Map::new();
        payload.insert("ok".into(), Value::Bool(true));
        payload.insert(
            "serves".into(),
            Value::Array(views.into_iter().map(Value::Object).collect()),
        );
        payload.insert("pruned".into(), Value::Array(pruned));
        print_json(&Value::Object(payload));
        return Ok(0);
    }

    if views.is_empty() {
        println!("  nothing running · start one with: hive serve <stack>");
        if !pruned.is_empty() {
            println!("  pruned {}", join_strings(&pruned));
        }
        return Ok(0);
    }
    println!("serves");
    for view in &views {
        let name = view.get("name").and_then(Value::as_str).unwrap_or("?");
        let pid = view.get("pid").and_then(Value::as_i64).unwrap_or(0);
        let mode = view
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("foreground");
        let uptime = view.get("uptime_s").and_then(Value::as_f64).unwrap_or(0.0);
        let state = if view.get("state").and_then(Value::as_str) == Some("running") {
            "running"
        } else {
            "stale"
        };
        println!(
            "  {name}  pid {pid}  {mode}  up {}  {state}",
            hive_state::format_duration(uptime)
        );
        if let Some(tiers) = view.get("tiers").and_then(Value::as_array) {
            for tier in tiers {
                let role = tier.get("role").and_then(Value::as_str).unwrap_or("?");
                let endpoint = tier_endpoint(view, tier);
                let ctx = tier.get("ctx").map(|c| c.to_string()).unwrap_or_default();
                let backend = tier.get("backend").and_then(Value::as_str).unwrap_or("");
                let status = match tier
                    .get("probe")
                    .and_then(|probe| probe.get("up"))
                    .and_then(Value::as_bool)
                {
                    Some(true) => "up",
                    Some(false) => "down",
                    None => "-",
                };
                println!(
                    "  {:<8} {:<42} ctx {:<6} {:<8} {status}",
                    role, endpoint, ctx, backend
                );
            }
        }
        if let Some(log_path) = view.get("log_path").and_then(Value::as_str) {
            println!("  log  {log_path}");
        } else if view.get("state").and_then(Value::as_str) == Some("running") {
            println!("  log  (foreground; logs are on its terminal)");
        }
        if view.get("state").and_then(Value::as_str) != Some("running") {
            println!("  prune with: hive status --prune");
        }
    }
    if !pruned.is_empty() {
        println!("  pruned {}", join_strings(&pruned));
    }
    Ok(0)
}

fn cmd_stop(
    paths: &Paths,
    name: Option<&str>,
    all: bool,
    force: bool,
    timeout: f64,
    as_json: bool,
) -> Cmd {
    if all && name.is_some() {
        eprintln!("hive stop: give a name or --all, not both");
        return Ok(2);
    }
    let records = hive_state::list_records(&paths.serve_dir);
    let targets: Vec<Record> = if all {
        records
    } else if let Some(name) = name {
        match hive_state::read_record(&paths.serve_dir, name) {
            Ok(Some(record)) => vec![record],
            Ok(None) => match resolve_one(&records, Some(name)) {
                Ok(record) => vec![record],
                Err(error) => {
                    eprintln!("hive stop: {error}");
                    return Ok(1);
                }
            },
            Err(error) => {
                eprintln!("hive stop: {error}");
                return Ok(1);
            }
        }
    } else if records.len() == 1 {
        records
    } else if records.is_empty() {
        println!("  nothing running");
        return Ok(0);
    } else {
        let names = records
            .iter()
            .filter_map(|record| record.get("name").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("hive stop: several serves are running ({names}); name one or use --all");
        return Ok(2);
    };

    if targets.is_empty() {
        println!("  nothing running");
        return Ok(0);
    }

    let mut results = Vec::new();
    let mut failed = false;
    for record in targets {
        let name = record
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_owned();
        let mut result = Map::new();
        result.insert("name".into(), Value::String(name.clone()));
        if !hive_state::pid_matches(&record) {
            hive_state::remove_record(&paths.serve_dir, &name);
            result.insert("stopped".into(), Value::Bool(true));
            result.insert("state".into(), Value::String("stale".into()));
            result.insert(
                "message".into(),
                Value::String("removed stale record (process was gone)".into()),
            );
        } else {
            let pid = record.get("pid").and_then(Value::as_i64).unwrap_or(0);
            let (stopped, message) = hive_state::terminate(pid, force, timeout);
            if stopped {
                hive_state::remove_record(&paths.serve_dir, &name);
            } else {
                failed = true;
            }
            result.insert("stopped".into(), Value::Bool(stopped));
            result.insert("state".into(), Value::String("running".into()));
            result.insert("message".into(), Value::String(message));
        }
        results.push(Value::Object(result));
    }

    if as_json {
        let mut payload = Map::new();
        payload.insert("ok".into(), Value::Bool(!failed));
        payload.insert("serves".into(), Value::Array(results));
        print_json(&Value::Object(payload));
    } else {
        for result in &results {
            let name = result.get("name").and_then(Value::as_str).unwrap_or("?");
            let stopped = result
                .get("stopped")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let message = result.get("message").and_then(Value::as_str).unwrap_or("");
            let detail = if message == "stopped" || message == "killed" {
                String::new()
            } else {
                format!(" · {message}")
            };
            if stopped {
                println!("  stopped {name}{detail}");
            } else {
                println!("  could not stop {name}{detail}");
            }
        }
    }
    Ok(if failed { 1 } else { 0 })
}

fn cmd_logs(
    paths: &Paths,
    name: Option<&str>,
    tier_role: Option<&str>,
    lines: i64,
    follow: bool,
    as_json: bool,
) -> Cmd {
    let records = hive_state::list_records(&paths.serve_dir);
    let record = match resolve_one(&records, name) {
        Ok(record) => record,
        Err(error) => {
            eprintln!("hive logs: {error}");
            return Ok(1);
        }
    };
    let name = record
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_owned();

    let (path, source) = if let Some(role) = tier_role {
        let tiers = record
            .get("tiers")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let Some(tier) = tiers
            .iter()
            .find(|tier| tier.get("role").and_then(Value::as_str) == Some(role))
        else {
            let roles = tiers
                .iter()
                .filter_map(|tier| tier.get("role").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "hive logs: tier {} is not in this serve (roles: {})",
                py_repr(role),
                if roles.is_empty() { "none" } else { &roles }
            );
            return Ok(2);
        };
        if tier.get("base_url").and_then(Value::as_str).is_some()
            || tier.get("port").and_then(Value::as_i64).is_none()
        {
            eprintln!(
                "hive logs: tier {} is an endpoint; its log belongs to that endpoint",
                py_repr(role)
            );
            return Ok(1);
        }
        let port = tier.get("port").and_then(Value::as_i64).unwrap_or(0);
        let dir = record
            .get("log_dir")
            .and_then(Value::as_str)
            .unwrap_or("logs");
        (
            PathBuf::from(dir).join(format!("llama_server_{port}.log")),
            format!("tier {role}"),
        )
    } else {
        match record.get("log_path").and_then(Value::as_str) {
            Some(raw) => (PathBuf::from(raw), "supervisor".to_owned()),
            None => {
                eprintln!(
                    "hive logs: this serve is in the foreground; its output is on \
                     that terminal (use --tier for the engine log)"
                );
                return Ok(1);
            }
        }
    };

    let lines = match hive_state::tail_lines(&path, lines) {
        Ok(lines) => lines,
        Err(error) => {
            eprintln!("hive logs: cannot read {}: {error}", path.display());
            return Ok(1);
        }
    };
    if as_json {
        let mut payload = Map::new();
        payload.insert("name".into(), Value::String(name));
        payload.insert("source".into(), Value::String(source));
        payload.insert(
            "path".into(),
            Value::String(path.to_string_lossy().into_owned()),
        );
        payload.insert(
            "lines".into(),
            Value::Array(lines.into_iter().map(Value::String).collect()),
        );
        print_json(&Value::Object(payload));
        return Ok(0);
    }
    for line in &lines {
        println!("{line}");
    }
    if !follow {
        return Ok(0);
    }

    eprintln!("-- following {} (Ctrl-C to stop) --", path.display());
    let mut handle = match std::fs::File::open(&path) {
        Ok(handle) => handle,
        Err(error) => {
            eprintln!("hive logs: cannot follow {}: {error}", path.display());
            return Ok(1);
        }
    };
    let _ = handle.seek(SeekFrom::End(0));
    let pid = record.get("pid").and_then(Value::as_i64).unwrap_or(0);
    let mut reader = BufReader::new(handle);
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                if !hive_state::pid_alive(pid) {
                    eprintln!("-- the serve has stopped --");
                    break;
                }
                std::thread::sleep(Duration::from_millis(300));
            }
            Ok(_) => {
                print!("{line}");
                let _ = std::io::stdout().flush();
            }
            Err(error) => {
                eprintln!("hive logs: cannot follow {}: {error}", path.display());
                return Ok(1);
            }
        }
    }
    Ok(0)
}

fn resolve_one(records: &[Record], name: Option<&str>) -> Result<Record, String> {
    if let Some(name) = name {
        for record in records {
            if record.get("name").and_then(Value::as_str) == Some(name) {
                return Ok(record.clone());
            }
        }
        return Err(format!("no such serve {}", py_repr(name)));
    }
    if records.is_empty() {
        return Err("nothing is running".into());
    }
    if records.len() > 1 {
        let names = records
            .iter()
            .filter_map(|record| record.get("name").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!("several serves are running ({names}); name one"));
    }
    Ok(records[0].clone())
}

fn tier_endpoint(record: &Record, tier: &Value) -> String {
    let host = record
        .get("host")
        .and_then(Value::as_str)
        .unwrap_or("127.0.0.1");
    tier_endpoint_with_host(tier, host)
}

fn tier_endpoint_with_host(tier: &Value, host: &str) -> String {
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

fn number(value: f64) -> Value {
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn join_strings(values: &[Value]) -> String {
    values
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
        .join(", ")
}
