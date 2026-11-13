//! Managed endpoint tiers: generate the engine config, start its server, and
//! wait for the OpenAI surface.
//!
//! This is the Rust port of `StrataLauncher` / `DeepInferLauncher` resolution
//! and config generation. The engine itself stays its own process; the
//! supervisor owns that process and stops it on unload.

use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use hive_core::http;
use hive_core::paths::{expand, expand_vars};
use serde_json::{Map, Value};

use crate::Options;

const STRATA_DEFAULT_TIMEOUT: f64 = 600.0;
const DEEPINFER_DEFAULT_TIMEOUT: f64 = 900.0;

pub(crate) struct Launched {
    pub pid: u32,
    pub port: u16,
    pub log: PathBuf,
    pub child: Child,
}

struct Resolved {
    kind: String,
    repo: PathBuf,
    config: PathBuf,
    python: PathBuf,
    port: u16,
    server: Option<String>,
}

pub(crate) fn start(
    launch: &Map<String, Value>,
    tier: &Value,
    stack_name: &str,
    role: &str,
    endpoint: &str,
    options: &Options,
) -> Result<Launched, String> {
    let resolved = resolve(launch, endpoint, options)?;
    let base = read_config(&resolved.config)?;
    let config = build_config(&base, launch, tier, endpoint, &resolved, options)?;

    let state_dir = options.state_dir.join(&resolved.kind);
    fs::create_dir_all(&state_dir)
        .map_err(|error| format!("cannot create {}: {error}", state_dir.display()))?;
    let slug = slug(&format!("{}-{stack_name}-{role}", resolved.kind));
    let generated = state_dir.join(format!("{slug}.json"));
    let payload = serde_json::to_string_pretty(&config)
        .map_err(|error| format!("cannot encode the generated config: {error}"))?;
    let temp = generated.with_extension("json.tmp");
    fs::write(&temp, format!("{payload}\n"))
        .map_err(|error| format!("cannot write {}: {error}", temp.display()))?;
    fs::rename(&temp, &generated)
        .map_err(|error| format!("cannot install {}: {error}", generated.display()))?;

    if let Some(parent) = options.log_dir.as_path().parent() {
        let _ = parent;
    }
    fs::create_dir_all(&options.log_dir)
        .map_err(|error| format!("cannot create {}: {error}", options.log_dir.display()))?;
    let log_path = options
        .log_dir
        .join(format!("{}_{}.log", resolved.kind, resolved.port));
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| format!("cannot open {}: {error}", log_path.display()))?;
    let stderr = log
        .try_clone()
        .map_err(|error| format!("cannot clone the log handle: {error}"))?;

    let mut command = Command::new(&resolved.python);
    if resolved.kind == "deepinfer" {
        let facade = options.repo.join("tools/deepinfer/serve.py");
        if !facade.is_file() {
            return Err(format!(
                "the DeepInfer facade is missing: {}",
                facade.display()
            ));
        }
        command
            .arg(facade)
            .arg("--config")
            .arg(&generated)
            .arg("--port")
            .arg(resolved.port.to_string());
        command.env("PYTHONPATH", deepinfer_pythonpath(&resolved.repo, options));
    } else {
        let server = resolved
            .server
            .clone()
            .unwrap_or_else(|| "serve/server.py".into());
        command
            .arg(server)
            .arg("--engine")
            .arg(&resolved.kind)
            .arg("--config")
            .arg(&generated)
            .arg("--port")
            .arg(resolved.port.to_string());
    }
    command
        .current_dir(&resolved.repo)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr));
    #[cfg(target_os = "linux")]
    unsafe {
        use std::os::unix::process::CommandExt;
        command.pre_exec(|| {
            libc::setsid();
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            if libc::getppid() == 1 {
                libc::_exit(1);
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("cannot start {} for tier {role}: {error}", resolved.kind))?;
    let timeout = launch.get("timeout_s").and_then(Value::as_f64).unwrap_or(
        if resolved.kind == "deepinfer" {
            DEEPINFER_DEFAULT_TIMEOUT
        } else {
            STRATA_DEFAULT_TIMEOUT
        },
    );
    if let Err(error) = wait_healthy(&mut child, endpoint, timeout, &log_path) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }
    Ok(Launched {
        pid: child.id(),
        port: resolved.port,
        log: log_path,
        child,
    })
}

fn resolve(
    launch: &Map<String, Value>,
    endpoint: &str,
    options: &Options,
) -> Result<Resolved, String> {
    let kind = launch
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("strata")
        .trim()
        .to_lowercase();
    let raw_repo = launch
        .get("repo")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let repo = expand(Path::new(&expand_vars(raw_repo)));
    if !repo.is_dir() {
        return Err(format!("{kind} repo not found: {}", repo.display()));
    }
    let repo = repo
        .canonicalize()
        .map_err(|error| format!("cannot resolve {}: {error}", repo.display()))?;

    let raw_config = launch
        .get("config")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let raw_config = expand(Path::new(&expand_vars(raw_config)));
    let external = launch
        .get("config_external")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let config = if raw_config.is_absolute() {
        raw_config
    } else if kind == "strata" && external {
        options.repo.join(raw_config)
    } else {
        repo.join(raw_config)
    };
    let config = config
        .canonicalize()
        .map_err(|_| format!("{kind} config not found: {}", config.display()))?;
    if kind == "strata" && !external && config != repo && !config.starts_with(&repo) {
        return Err(format!(
            "strata config {} must stay inside the repo {} (set config_external: true for a Hive-owned config)",
            config.display(),
            repo.display()
        ));
    }
    if kind == "deepinfer" && config != repo && !config.starts_with(&repo) {
        return Err(format!(
            "deepinfer config {} must stay inside the repo {}",
            config.display(),
            repo.display()
        ));
    }
    if !config.is_file() {
        return Err(format!("{kind} config not found: {}", config.display()));
    }

    let raw_python = launch
        .get("python")
        .and_then(Value::as_str)
        .unwrap_or(".venv/bin/python");
    let raw_python = expand(Path::new(&expand_vars(raw_python)));
    let python = if raw_python.is_absolute() {
        raw_python
    } else {
        repo.join(raw_python)
    };
    let python = std::path::absolute(&python)
        .map_err(|error| format!("cannot resolve {}: {error}", python.display()))?;
    if !python.is_file() {
        return Err(format!(
            "{kind} python not found: {} (set launch.python)",
            python.display()
        ));
    }
    let port = endpoint_port(endpoint);
    if port == 0 {
        return Err(format!("endpoint {endpoint:?} carries no usable port"));
    }
    let server = launch
        .get("server")
        .and_then(Value::as_str)
        .map(str::to_owned);
    Ok(Resolved {
        kind,
        repo,
        config,
        python,
        port,
        server,
    })
}

fn read_config(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read config {}: {error}", path.display()))?;
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(&bytes);
    let data: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("config {} is not valid JSON: {error}", path.display()))?;
    if !data.is_object() {
        return Err(format!("config {} must be a JSON object", path.display()));
    }
    Ok(data)
}

fn build_config(
    base: &Value,
    launch: &Map<String, Value>,
    tier: &Value,
    endpoint: &str,
    resolved: &Resolved,
    options: &Options,
) -> Result<Value, String> {
    let mut config = base.clone();
    expand_tree(&mut config);
    let object = config
        .as_object_mut()
        .ok_or_else(|| "generated config must be an object".to_owned())?;
    object.insert("port".into(), Value::from(resolved.port));
    if resolved.kind == "deepinfer" {
        object.insert(
            "ctx".into(),
            Value::from(tier.get("ctx").and_then(Value::as_i64).unwrap_or(4096)),
        );
        let engine_args: Vec<Value> = launch
            .get("engine_args")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !engine_args.is_empty() {
            object.insert("engine_args".into(), Value::Array(engine_args));
        }
    } else {
        let ctx = tier.get("ctx").and_then(Value::as_i64).unwrap_or(8192);
        let ctx_flag = launch
            .get("ctx_flag")
            .and_then(Value::as_str)
            .unwrap_or("--max-context");
        let mut overrides = vec![ctx_flag.to_owned(), ctx.to_string()];
        overrides.extend(
            launch
                .get("engine_args")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned),
        );
        let base_args: Vec<String> = object
            .get("args")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        let merged = merge_engine_args(&base_args, &overrides);
        object.insert(
            "args".into(),
            Value::Array(merged.into_iter().map(Value::String).collect()),
        );
        let engine_log = options
            .log_dir
            .join(format!("strata_{}.engine.log", resolved.port));
        object.insert(
            "log".into(),
            Value::String(engine_log.to_string_lossy().into_owned()),
        );
    }
    if let Some(extra) = launch.get("options").and_then(Value::as_object) {
        for (key, value) in extra {
            object.insert(key.clone(), value.clone());
        }
    }
    let _ = endpoint;
    Ok(config)
}

fn expand_tree(value: &mut Value) {
    match value {
        Value::String(text) => *text = expand_vars(text),
        Value::Array(items) => {
            for item in items {
                expand_tree(item);
            }
        }
        Value::Object(object) => {
            for item in object.values_mut() {
                expand_tree(item);
            }
        }
        _ => {}
    }
}

/// Merge `--flag value...` spans into a base argv, replacing an existing span.
fn merge_engine_args(base: &[String], overrides: &[String]) -> Vec<String> {
    let mut args = base.to_vec();
    let mut pending = overrides.to_vec();
    while !pending.is_empty() {
        let token = pending.remove(0);
        if !is_flag(&token) {
            args.push(token);
            continue;
        }
        let mut values = Vec::new();
        while !pending.is_empty() && !is_flag(&pending[0]) {
            values.push(pending.remove(0));
        }
        let mut replaced = false;
        let mut index = 0;
        while index < args.len() {
            if args[index] == token {
                let mut end = index + 1;
                while end < args.len() && !is_flag(&args[end]) {
                    end += 1;
                }
                args.splice(
                    index..end,
                    std::iter::once(token.clone()).chain(values.iter().cloned()),
                );
                replaced = true;
                break;
            }
            index += 1;
        }
        if !replaced {
            args.push(token);
            args.extend(values);
        }
    }
    args
}

fn is_flag(token: &str) -> bool {
    let stripped = token.strip_prefix("--").or_else(|| token.strip_prefix('-'));
    stripped
        .and_then(|rest| rest.chars().next())
        .map(|ch| ch.is_ascii_alphabetic())
        .unwrap_or(false)
}

fn endpoint_port(endpoint: &str) -> u16 {
    let rest = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"));
    let Some(rest) = rest else {
        return 0;
    };
    let authority = rest.split('/').next().unwrap_or("");
    authority
        .rsplit_once(':')
        .and_then(|(_, port)| port.parse().ok())
        .unwrap_or(if endpoint.starts_with("https://") {
            443
        } else {
            80
        })
}

fn wait_healthy(child: &mut Child, endpoint: &str, timeout: f64, log: &Path) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs_f64(timeout.max(1.0));
    let root = endpoint
        .trim_end_matches('/')
        .strip_suffix("/v1")
        .unwrap_or_else(|| endpoint.trim_end_matches('/'));
    let health_url = format!("{root}/health");
    let models_url = format!("{root}/v1/models");
    let mut last_error = String::new();
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!(
                "server for this tier exited during startup (code {status}); see {}: {}",
                log.display(),
                crate::log_tail(log, 20)
            ));
        }
        for (index, url) in [health_url.as_str(), models_url.as_str()]
            .iter()
            .enumerate()
        {
            if let Ok(response) = http::get(url, Duration::from_secs(5))
                && response.status == 200
            {
                if index == 0
                    && let Ok(data) = response.json()
                {
                    if data.get("loaded").and_then(Value::as_bool) == Some(false) {
                        last_error = "endpoint reports loaded=false".into();
                        continue;
                    }
                    if let Some(status) = data.get("status").and_then(Value::as_str)
                        && !status.trim().eq_ignore_ascii_case("ok")
                    {
                        last_error = format!("status={status:?}");
                        continue;
                    }
                }
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    Err(format!(
        "endpoint {endpoint} did not become healthy within {timeout:.0}s ({}); see {}",
        if last_error.is_empty() {
            "no health response"
        } else {
            &last_error
        },
        log.display()
    ))
}

fn deepinfer_pythonpath(repo: &Path, options: &Options) -> String {
    let mut paths = vec![
        repo.to_path_buf(),
        options.repo.join("vendor"),
        options.repo.join("vendor/deepinfer"),
        options.repo.clone(),
    ];
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(paths)
        .map(|paths| paths.to_string_lossy().into_owned())
        .unwrap_or_default()
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
    out.trim_matches('-').to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn merges_flag_spans_without_duplicating() {
        let base = vec![
            "--max-context".into(),
            "8192".into(),
            "--prefill".into(),
            "512".into(),
        ];
        let merged = merge_engine_args(
            &base,
            &[
                "--max-context".into(),
                "2048".into(),
                "--vram-reserve-mib".into(),
                "3072".into(),
            ],
        );
        assert_eq!(
            merged,
            vec![
                "--max-context",
                "2048",
                "--prefill",
                "512",
                "--vram-reserve-mib",
                "3072"
            ]
        );
    }

    #[test]
    fn negative_numbers_are_values_not_flags() {
        assert!(!is_flag("-1"));
        assert!(is_flag("-t"));
        assert!(is_flag("--ctx"));
    }

    #[test]
    fn expands_environment_variables_recursively() {
        unsafe { std::env::set_var("HIVE_LAUNCH_TEST", "/tmp/launch") };
        let mut value = json!({"args": ["--path", "$HIVE_LAUNCH_TEST/x"]});
        expand_tree(&mut value);
        assert_eq!(value["args"][1], "/tmp/launch/x");
        unsafe { std::env::remove_var("HIVE_LAUNCH_TEST") };
    }

    #[test]
    fn slugs_keep_safe_characters() {
        assert_eq!(slug("strata/my stack:face"), "strata-my-stack-face");
    }
}
