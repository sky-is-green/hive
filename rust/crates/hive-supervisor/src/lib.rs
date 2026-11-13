//! Stack application and engine supervision.
//!
//! One `apply` call validates a stack, spawns one `llama-server` per local
//! tier, adopts endpoint-backed tiers after a health check, and returns the
//! rows written to the serve record. `Applied::unload` stops exactly the
//! children this process started.
//!
//! Engine children request `PR_SET_PDEATHSIG(SIGTERM)`, so `kill -9` of the
//! supervisor does not leave a model process behind.

pub mod residency;

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use hive_core::http;
use serde_json::{Map, Value};

pub struct Options {
    pub host: String,
    pub base_port: u16,
    pub llama_server: Option<PathBuf>,
    pub repo: PathBuf,
    pub models_dir: PathBuf,
    pub log_dir: PathBuf,
    pub startup_timeout: f64,
}

/// A successfully applied stack.
pub struct Applied {
    pub tiers: Vec<Value>,
    pub warnings: Vec<String>,
    pub per_card: Vec<Value>,
    children: Vec<Engine>,
}

struct Engine {
    role: String,
    pid: u32,
    port: u16,
    log: PathBuf,
    child: Child,
}

impl Applied {
    pub fn pids(&self) -> Vec<i32> {
        self.children
            .iter()
            .map(|engine| engine.pid as i32)
            .collect()
    }

    pub fn tier_ports(&self) -> Vec<(String, u16, PathBuf)> {
        self.children
            .iter()
            .map(|engine| (engine.role.clone(), engine.port, engine.log.clone()))
            .collect()
    }

    /// Stop every engine this process started: SIGTERM, then SIGKILL.
    pub fn unload(mut self) {
        terminate_all(&mut self.children);
    }
}

impl Drop for Applied {
    fn drop(&mut self) {
        terminate_all(&mut self.children);
    }
}

/// Load a stack by name under `stacks_dir` or by explicit path.
pub fn load_stack(target: &str, stacks_dir: &Path) -> Result<(Value, Option<PathBuf>), String> {
    let candidate = PathBuf::from(target);
    let looks_like_path = target.ends_with(".json") || target.contains('/') || candidate.is_file();
    let path = if looks_like_path {
        let path = candidate;
        if !path.is_file() {
            return Err(format!("no stack file at {}", path.display()));
        }
        path
    } else {
        let path = stacks_dir.join(format!(
            "{}.json",
            hive_core::stacks::check_stack_name(target)?
        ));
        if !path.is_file() {
            return Err(format!("no saved stack at {}", path.display()));
        }
        path
    };
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read stack at {}: {error}", path.display()))?;
    let data: Value = serde_json::from_str(&text)
        .map_err(|error| format!("stack file {} is not valid JSON: {error}", path.display()))?;
    let stack = hive_core::stacks::parse_stack(&data).map_err(|error| error.to_string())?;
    validate(&stack)?;
    Ok((
        stack,
        looks_like_path.then(|| path.canonicalize().unwrap_or(path)),
    ))
}

/// Structural checks that need tiers to be usable for a serve.
pub fn validate(stack: &Value) -> Result<(), String> {
    let tiers = stack
        .get("tiers")
        .and_then(Value::as_array)
        .ok_or_else(|| "stack has no tiers".to_owned())?;
    if tiers.is_empty() {
        return Err("stack has no tiers (every stack needs a face tier)".into());
    }
    let mut seen = HashSet::new();
    for (position, tier) in tiers.iter().enumerate() {
        let role = tier
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("tier {}: missing tier role", position + 1))?;
        if !seen.insert(role.to_owned()) {
            return Err(format!(
                "tier {}: duplicate tier role {role:?}",
                position + 1
            ));
        }
        for key in ["repo", "file"] {
            if tier
                .get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .is_none()
            {
                return Err(format!("tier {}: missing {key}", position + 1));
            }
        }
    }
    Ok(())
}

/// Apply every tier, face first. On failure the already-spawned engines are
/// stopped before the error is returned.
pub fn apply(stack: &Value, options: &Options) -> Result<Applied, String> {
    let tiers = stack
        .get("tiers")
        .and_then(Value::as_array)
        .ok_or_else(|| "stack has no tiers".to_owned())?;
    let mut ordered: Vec<&Value> = Vec::with_capacity(tiers.len());
    ordered.extend(
        tiers
            .iter()
            .filter(|tier| tier.get("role").and_then(Value::as_str) == Some("face")),
    );
    ordered.extend(
        tiers
            .iter()
            .filter(|tier| tier.get("role").and_then(Value::as_str) != Some("face")),
    );

    let stack_name = stack
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("stack")
        .to_owned();
    let preflight = residency::preflight(stack, options)?;
    let per_card: Vec<Value> = preflight
        .per_card
        .iter()
        .map(residency::CardPlan::to_json)
        .collect();
    let mut applied = Applied {
        tiers: Vec::new(),
        warnings: preflight.warnings.clone(),
        per_card: per_card.clone(),
        children: Vec::new(),
    };
    let mut busy: HashSet<u16> = HashSet::new();
    let mut offset = 0u16;

    for tier in ordered {
        let role = tier
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_owned();
        let key = format!("{stack_name}-{role}");
        if let Some(endpoint) = endpoint(tier) {
            if let Some(port) = endpoint_port(&endpoint) {
                busy.insert(port);
            }
            match probe_endpoint(&endpoint) {
                Ok(health) => {
                    if let Some(expected) = tier.get("expect_model").and_then(Value::as_str)
                        && !expected.is_empty()
                        && let Some(reported) = health.model.as_deref()
                        && !reported.is_empty()
                        && reported != expected
                    {
                        return Err(format!(
                            "endpoint {endpoint} reports model {reported:?}, \
                                         expected {expected:?} (stale route or wrong server)"
                        ));
                    }
                    applied
                        .tiers
                        .push(endpoint_tier_row(tier, &key, &endpoint, &health, &per_card));
                }
                Err(error) => {
                    if tier.get("launch").is_some() {
                        applied.unload();
                        return Err(format!(
                            "stack '{stack_name}' failed on tier '{role}': {error} \
                             (launch-backed endpoints are not ported to the Rust \
                             supervisor yet)"
                        ));
                    }
                    applied.unload();
                    return Err(format!(
                        "stack '{stack_name}' failed on tier '{role}': {error}"
                    ));
                }
            }
            continue;
        }

        let result = spawn_local(tier, options, &busy, &mut offset);
        let engine = match result {
            Ok(engine) => engine,
            Err(error) => {
                applied.unload();
                return Err(format!(
                    "stack '{stack_name}' failed on tier '{role}': {error}"
                ));
            }
        };
        busy.insert(engine.port);
        applied
            .tiers
            .push(local_tier_row(tier, &key, &engine, &per_card));
        applied.children.push(engine);
    }
    Ok(applied)
}

struct EngineHealth {
    model: Option<String>,
    max_context: Option<i64>,
    loaded: bool,
}

fn endpoint(tier: &Value) -> Option<String> {
    tier.get("endpoint")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

fn endpoint_port(endpoint: &str) -> Option<u16> {
    let rest = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))?;
    let authority = rest.split('/').next()?;
    let port = authority.rsplit_once(':').map(|(_, port)| port)?;
    port.parse().ok()
}

fn probe_endpoint(endpoint: &str) -> Result<EngineHealth, String> {
    let root = endpoint
        .trim_end_matches('/')
        .strip_suffix("/v1")
        .unwrap_or_else(|| endpoint.trim_end_matches('/'));
    let mut errors = Vec::new();
    for (url, health) in [
        (format!("{root}/health"), true),
        (format!("{root}/v1/models"), false),
    ] {
        let response = match http::get(&url, Duration::from_secs(5)) {
            Ok(response) => response,
            Err(error) => {
                errors.push(format!("{url}: {error}"));
                continue;
            }
        };
        if response.status != 200 {
            errors.push(format!("{url}: HTTP {}", response.status));
            continue;
        }
        let data = response.json().unwrap_or(Value::Object(Map::new()));
        if health {
            if let Some(status) = data.get("status")
                && let Some(status) = status.as_str()
                && !status.trim().eq_ignore_ascii_case("ok")
            {
                errors.push(format!("{url}: status={status:?}"));
                continue;
            }
            if data.get("loaded").and_then(Value::as_bool) == Some(false) {
                return Err(format!(
                    "endpoint {endpoint} is up but its model is not loaded \
                     (/health says loaded=false); load it and apply again"
                ));
            }
        }
        let model = data
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                data.get("data")
                    .and_then(Value::as_array)
                    .and_then(|items| {
                        items
                            .iter()
                            .find_map(|item| item.get("id").and_then(Value::as_str))
                    })
                    .map(str::to_owned)
            });
        return Ok(EngineHealth {
            model,
            max_context: data.get("max_context").and_then(Value::as_i64),
            loaded: data.get("loaded").and_then(Value::as_bool).unwrap_or(true),
        });
    }
    Err(format!(
        "endpoint {endpoint} is not healthy: {}",
        if errors.is_empty() {
            "no response".to_owned()
        } else {
            errors.join("; ")
        }
    ))
}

fn endpoint_tier_row(
    tier: &Value,
    key: &str,
    endpoint: &str,
    health: &EngineHealth,
    per_card: &[Value],
) -> Value {
    let role = tier.get("role").and_then(Value::as_str).unwrap_or("?");
    let ctx = health
        .max_context
        .or_else(|| tier.get("ctx").and_then(Value::as_i64))
        .unwrap_or(8192);
    let model = health
        .model
        .clone()
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| {
            Path::new(tier.get("file").and_then(Value::as_str).unwrap_or(""))
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default()
        });
    let mut row = Map::new();
    row.insert("role".into(), Value::String(role.into()));
    row.insert("key".into(), Value::String(key.into()));
    row.insert(
        "port".into(),
        Value::from(endpoint_port(endpoint).unwrap_or(0)),
    );
    row.insert("ctx".into(), Value::from(ctx));
    row.insert("model".into(), Value::String(model));
    row.insert(
        "backend".into(),
        Value::String(
            tier.get("backend")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .unwrap_or("endpoint")
                .into(),
        ),
    );
    row.insert("resident".into(), Value::Bool(health.loaded));
    row.insert("vram_gb".into(), Value::Null);
    row.insert("tok_s".into(), Value::Null);
    row.insert(
        "base_url".into(),
        Value::String(endpoint.trim_end_matches('/').into()),
    );
    row.insert(
        "caps".into(),
        tier.get("caps").cloned().unwrap_or(Value::Null),
    );
    row.insert("managed".into(), Value::Bool(false));
    row.insert("pid".into(), Value::Null);
    row.insert("per_card".into(), Value::Array(per_card.to_vec()));
    Value::Object(row)
}

fn local_tier_row(tier: &Value, key: &str, engine: &Engine, per_card: &[Value]) -> Value {
    let role = tier.get("role").and_then(Value::as_str).unwrap_or("?");
    let ctx = tier.get("ctx").and_then(Value::as_i64).unwrap_or(8192);
    let model = health_model(engine).unwrap_or_else(|| {
        Path::new(tier.get("file").and_then(Value::as_str).unwrap_or(""))
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    let mut row = Map::new();
    row.insert("role".into(), Value::String(role.into()));
    row.insert("key".into(), Value::String(key.into()));
    row.insert("port".into(), Value::from(engine.port));
    row.insert("ctx".into(), Value::from(ctx));
    row.insert("model".into(), Value::String(model));
    row.insert(
        "backend".into(),
        Value::String(
            tier.get("backend")
                .and_then(Value::as_str)
                .unwrap_or("vulkan")
                .into(),
        ),
    );
    row.insert("resident".into(), Value::Bool(true));
    row.insert("vram_gb".into(), Value::Null);
    row.insert("tok_s".into(), Value::Null);
    row.insert("base_url".into(), Value::Null);
    row.insert(
        "caps".into(),
        tier.get("caps").cloned().unwrap_or(Value::Null),
    );
    row.insert("managed".into(), Value::Bool(false));
    row.insert("pid".into(), Value::from(engine.pid));
    row.insert("per_card".into(), Value::Array(per_card.to_vec()));
    Value::Object(row)
}

fn health_model(engine: &Engine) -> Option<String> {
    // The health probe already ran in spawn_local; keep this cheap by probing
    // once more only when the caller needs the advertised id.
    let response = http::get(
        &format!("http://127.0.0.1:{}/v1/models", engine.port),
        Duration::from_millis(500),
    )
    .ok()?;
    if response.status != 200 {
        return None;
    }
    let data = response.json().ok()?;
    data.get("data")
        .and_then(Value::as_array)
        .and_then(|items| {
            items
                .iter()
                .find_map(|item| item.get("id").and_then(Value::as_str))
        })
        .map(str::to_owned)
}

fn spawn_local(
    tier: &Value,
    options: &Options,
    busy: &HashSet<u16>,
    offset: &mut u16,
) -> Result<Engine, String> {
    let role = tier.get("role").and_then(Value::as_str).unwrap_or("?");
    let file = tier.get("file").and_then(Value::as_str).unwrap_or("");
    let repo = tier.get("repo").and_then(Value::as_str).unwrap_or("");
    let model = hive_models::resolve_model(file, &options.models_dir).ok_or_else(|| {
        if !repo.is_empty() && repo != "local" {
            format!(
                "model {file:?} is not local; the Rust supervisor does not \
                 download Hugging Face repos yet (fetch it with `hive models \
                 download`)"
            )
        } else {
            format!(
                "model {file:?} is neither a local file nor in {}",
                options.models_dir.display()
            )
        }
    })?;
    let backend = tier
        .get("backend")
        .and_then(Value::as_str)
        .unwrap_or("vulkan")
        .to_lowercase();
    let binary = binary_for(&backend, options)?;
    let port = next_port(options.base_port, busy, offset)?;
    let ctx = tier.get("ctx").and_then(Value::as_i64).unwrap_or(8192);
    let ngl = tier.get("ngl").and_then(Value::as_i64).unwrap_or(99);
    let log_path = options.log_dir.join(format!("llama_server_{port}.log"));
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }

    let mut command = Command::new(&binary);
    command
        .arg("-m")
        .arg(&model)
        .arg("--host")
        .arg(&options.host)
        .arg("--port")
        .arg(port.to_string())
        .arg("-ngl")
        .arg(ngl.to_string())
        .arg("-c")
        .arg(ctx.to_string());
    if let Some(cache_k) = tier.get("cache_k").and_then(Value::as_str)
        && !cache_k.is_empty()
    {
        command.arg("--cache-type-k").arg(cache_k);
    }
    if let Some(cache_v) = tier.get("cache_v").and_then(Value::as_str)
        && !cache_v.is_empty()
    {
        command.arg("--cache-type-v").arg(cache_v);
    }
    if let Some(mmproj) = tier.get("mmproj").and_then(Value::as_str)
        && !mmproj.is_empty()
    {
        command.arg("--mmproj").arg(mmproj);
    }
    if backend == "scion" {
        command.arg("--skip-chat-parsing");
    }
    command.arg("--jinja");
    if let Some(split) = tier.get("ts").and_then(Value::as_str)
        && !split.is_empty()
    {
        command
            .arg("--split-mode")
            .arg("layer")
            .arg("--tensor-split")
            .arg(split);
    }
    if let Some(spec) = tier.get("spec") {
        if let Some(spec) = spec.as_object() {
            if let Some(kind) = spec.get("type").and_then(Value::as_str) {
                command.arg("--spec-type").arg(kind);
            }
            if let Some(max) = spec.get("n_max").and_then(Value::as_i64) {
                command.arg("--draft-max").arg(max.to_string());
            }
        } else if let Some(kind) = spec.as_str() {
            command.arg("--spec-type").arg(kind);
        }
    }
    if let Some(engine_args) = tier.get("engine_args").and_then(Value::as_array) {
        for arg in engine_args.iter().filter_map(Value::as_str) {
            command.arg(arg);
        }
    }
    if let Some(pin) = tier.get("pin").and_then(Value::as_str) {
        let pin = pin.trim();
        if !pin.is_empty() {
            if let Some((name, value)) = pin.split_once('=') {
                command.env(name.trim(), value.trim());
            } else {
                command.env("HIP_VISIBLE_DEVICES", pin);
            }
        }
    }

    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| format!("cannot open {}: {error}", log_path.display()))?;
    let stderr = log
        .try_clone()
        .map_err(|error| format!("cannot clone the log handle: {error}"))?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr));
    #[cfg(target_os = "linux")]
    unsafe {
        use std::os::unix::process::CommandExt;
        command.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            if libc::getppid() == 1 {
                libc::_exit(1);
            }
            Ok(())
        });
    }
    let child = command
        .spawn()
        .map_err(|error| format!("cannot start {}: {error}", binary.display()))?;
    let pid = child.id();
    let mut engine = Engine {
        role: role.to_owned(),
        pid,
        port,
        log: log_path.clone(),
        child,
    };
    wait_healthy(&mut engine, options.startup_timeout)?;
    Ok(engine)
}

fn wait_healthy(engine: &mut Engine, timeout: f64) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs_f64(timeout.max(1.0));
    let url = format!("http://127.0.0.1:{}/v1/models", engine.port);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = engine.child.try_wait() {
            let tail = log_tail(&engine.log, 20);
            return Err(format!(
                "llama-server exited during startup (code {status}); see {}\n{tail}",
                engine.log.display()
            ));
        }
        if let Ok(response) = http::get(&url, Duration::from_secs(2))
            && response.status == 200
        {
            std::thread::sleep(Duration::from_millis(500));
            if let Ok(Some(status)) = engine.child.try_wait() {
                let tail = log_tail(&engine.log, 20);
                return Err(format!(
                    "llama-server exited during startup (code {status}); see {}\n{tail}",
                    engine.log.display()
                ));
            }
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    let _ = engine.child.kill();
    Err(format!(
        "llama-server did not become healthy within {timeout:.0}s; see {}",
        engine.log.display()
    ))
}

fn binary_for(backend: &str, options: &Options) -> Result<PathBuf, String> {
    if let Some(raw) = std::env::var_os("HARNESS_LLAMA_SERVER") {
        let path = PathBuf::from(raw);
        if path.is_file() {
            return Ok(path);
        }
    }
    if let Some(explicit) = &options.llama_server
        && explicit.is_file()
        && (backend == "vulkan" || backend.is_empty())
    {
        return Ok(explicit.clone());
    }
    for base in [
        options.repo.join("tools/backends"),
        options.repo.join("app/tools/backends"),
    ] {
        let candidate = base.join(backend).join("llama-server");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    let default = options
        .llama_server
        .clone()
        .or_else(|| Some(options.repo.join("tools/llama.cpp/llama-server")));
    if let Some(default) = default {
        if default.is_file() {
            return Ok(default);
        }
        return Err(format!(
            "llama-server not found at {}. Set HARNESS_LLAMA_SERVER or extract \
             a llama.cpp Vulkan release into tools/llama.cpp/.",
            default.display()
        ));
    }
    Err(format!(
        "no llama-server binary for backend {backend:?}; install one with \
         `hive backends fetch {backend}`"
    ))
}

fn next_port(base: u16, busy: &HashSet<u16>, offset: &mut u16) -> Result<u16, String> {
    for _ in 0..64 {
        let port = base.saturating_add(*offset);
        *offset = offset.saturating_add(1);
        if busy.contains(&port) {
            continue;
        }
        if TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)).is_ok() {
            return Ok(port);
        }
    }
    Err("no free port found in the llama-server range".into())
}

fn terminate_all(children: &mut Vec<Engine>) {
    for engine in children.iter() {
        unsafe {
            libc::kill(engine.pid as i32, libc::SIGTERM);
        }
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        let mut all_done = true;
        for engine in children.iter_mut() {
            match engine.child.try_wait() {
                Ok(Some(_)) => {}
                Ok(None) => all_done = false,
                Err(_) => {}
            }
        }
        if all_done {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    for engine in children.iter_mut() {
        if matches!(engine.child.try_wait(), Ok(None)) {
            let _ = engine.child.kill();
            let _ = engine.child.wait();
        }
    }
    children.clear();
}

fn log_tail(path: &Path, count: usize) -> String {
    let Ok(text) = std::fs::read_to_string(path) else {
        return String::new();
    };
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(count)..].join("\n")
}
