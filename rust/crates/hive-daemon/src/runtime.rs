//! Daemon-owned stack and model-server lifecycle.
//!
//! The Rust CLI already applies stacks through `hive-supervisor`; this module
//! gives the daemon the same engine. It holds at most one applied stack (the
//! same discipline as Python's `StackManager`) and turns supervisor rows into
//! the `/v1/stacks` and `/v1/server` payloads the console expects.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use hive_core::paths::{Paths, expand};
use hive_core::{hardware, http, stacks};
use hive_supervisor::{Applied, EngineInfo, Options};
use serde_json::{Map, Value};

#[derive(Debug)]
pub struct ApiError {
    pub status: u16,
    pub detail: String,
}

impl ApiError {
    fn new(status: u16, detail: impl Into<String>) -> Self {
        Self {
            status,
            detail: detail.into(),
        }
    }
}

struct AppliedStack {
    name: String,
    tiers: Vec<Value>,
    warnings: Vec<String>,
    engines: Vec<EngineInfo>,
    started_at: String,
    applied: Applied,
}

pub struct Runtime {
    paths: Paths,
    options: Options,
    apply_lock: tokio::sync::Mutex<()>,
    state: Mutex<Option<AppliedStack>>,
}

impl Runtime {
    pub fn new(paths: &Paths) -> Self {
        let llama_server = std::env::var_os("HIVE_LLAMA_SERVER")
            .map(PathBuf::from)
            .map(|path| expand(&path))
            .filter(|path| path.is_file());
        let startup_timeout = std::env::var("HIVE_STARTUP_TIMEOUT")
            .or_else(|_| std::env::var("HARNESS_STARTUP_TIMEOUT"))
            .ok()
            .and_then(|text| text.parse::<f64>().ok())
            .unwrap_or(300.0);
        let options = Options {
            host: "127.0.0.1".into(),
            base_port: 1234,
            llama_server,
            repo: paths.repo.clone(),
            models_dir: paths.models_dir.clone(),
            state_dir: paths.state_dir.clone(),
            log_dir: paths.log_dir.clone(),
            startup_timeout,
        };
        Self {
            paths: paths.clone(),
            options,
            apply_lock: tokio::sync::Mutex::new(()),
            state: Mutex::new(None),
        }
    }

    // ------------------------------------------------------------------
    // Saved stacks
    // ------------------------------------------------------------------
    pub fn list_stacks(&self) -> Value {
        Value::Array(
            stacks::list_summaries(&self.paths.stacks_dir)
                .iter()
                .map(|summary| summary.to_json())
                .collect(),
        )
    }

    pub fn load_stack(&self, name: &str) -> Result<Value, ApiError> {
        read_stack(&self.paths.stacks_dir, name)
    }

    pub fn save_stack(&self, name: &str, body: &Value) -> Result<Value, ApiError> {
        let name = stack_name(name)?;
        let mut payload = body
            .as_object()
            .cloned()
            .ok_or_else(|| ApiError::new(400, "stack body must be an object"))?;
        match payload.get("name").and_then(Value::as_str) {
            Some(given) if given != name => {
                return Err(ApiError::new(
                    400,
                    format!("body stack name {given:?} does not match path {name:?}"),
                ));
            }
            Some(_) => {}
            None => {
                payload.insert("name".into(), Value::String(name.clone()));
            }
        }
        let stack = stacks::parse_stack(&Value::Object(payload))
            .map_err(|error| ApiError::new(400, format!("invalid stack {name:?}: {error}")))?;
        let errors = stacks::validate_shape(&stack);
        if !errors.is_empty() {
            return Err(ApiError::new(
                400,
                format!("invalid stack: {}", errors.join("; ")),
            ));
        }
        write_stack(&self.paths.stacks_dir, &name, &stack)?;
        Ok(stack)
    }

    pub fn delete_stack(&self, name: &str) -> Result<Value, ApiError> {
        let name = stack_name(name)?;
        let path = self.paths.stacks_dir.join(format!("{name}.json"));
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(serde_json::json!({"ok": true, "deleted": name})),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(ApiError::new(404, format!("no such stack: {name}")))
            }
            Err(error) => Err(ApiError::new(500, format!("cannot delete {name}: {error}"))),
        }
    }

    pub fn validate_stack(&self, name: &str) -> Result<Value, ApiError> {
        let stack = self.load_stack(name)?;
        let devices = hive_supervisor::residency::visible_devices();
        let plan = hive_supervisor::residency::plan(&stack, &self.paths.models_dir, &devices);
        Ok(serde_json::json!({
            "ok": plan.ok,
            "per_card": plan.per_card.iter().map(|card| card.to_json()).collect::<Vec<_>>(),
            "warnings": plan.warnings,
        }))
    }

    pub fn status(&self) -> Value {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        match state.as_ref() {
            Some(applied) => serde_json::json!({
                "ok": true,
                "stack": applied.name,
                "tiers": applied.tiers.clone(),
                "warnings": applied.warnings.clone(),
            }),
            None => serde_json::json!({
                "ok": true,
                "stack": null,
                "tiers": [],
                "warnings": [],
            }),
        }
    }

    pub async fn apply_stack(
        self: &Arc<Self>,
        name: &str,
        base_port: Option<u16>,
    ) -> Result<Value, ApiError> {
        let stack = self.load_stack(name)?;
        let name = stack
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(name)
            .to_owned();
        self.apply_value(stack, name, base_port).await
    }

    pub async fn unload_stack(&self, name: &str) -> Result<Value, ApiError> {
        let _guard = self.apply_lock.lock().await;
        let current = self.take_applied();
        let Some(current) = current else {
            return Ok(serde_json::json!({"ok": true, "unloaded": []}));
        };
        if current.name != name {
            self.put_applied(current);
            return Ok(serde_json::json!({"ok": true, "unloaded": []}));
        }
        let roles: Vec<Value> = current
            .tiers
            .iter()
            .filter_map(|tier| tier.get("role").and_then(Value::as_str))
            .map(|role| Value::String(role.to_owned()))
            .collect();
        unload(current).await;
        Ok(serde_json::json!({"ok": true, "unloaded": roles}))
    }

    // ------------------------------------------------------------------
    // Model server
    // ------------------------------------------------------------------
    pub async fn start_server(self: &Arc<Self>, body: &Value) -> Result<Value, ApiError> {
        if flag(body, "embedding") {
            return Err(ApiError::new(
                422,
                "embedding launches are not native yet; use the Python sidecar",
            ));
        }
        if body.get("pooling").and_then(Value::as_str).is_some() {
            return Err(ApiError::new(422, "pooling requires embedding=True"));
        }
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty());
        let Some(model) = model else {
            return Err(ApiError::new(
                422,
                "server start needs a local model name or path; download a GGUF first",
            ));
        };
        let key = body
            .get("key")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| {
                Path::new(model)
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .filter(|stem| !stem.is_empty())
                    .unwrap_or_else(|| "model".to_owned())
            });
        let stack_name = stack_name(&key).unwrap_or_else(|_| "studio".to_owned());
        let ctx = body.get("ctx_size").and_then(Value::as_i64).unwrap_or(8192);
        let ngl = body.get("ngl").and_then(Value::as_i64).unwrap_or(999);
        let backend = body
            .get("backend")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .unwrap_or("vulkan")
            .to_owned();
        let mut tier = Map::new();
        tier.insert("role".into(), Value::String("face".into()));
        tier.insert(
            "repo".into(),
            Value::String(
                body.get("hf_repo")
                    .and_then(Value::as_str)
                    .unwrap_or("local")
                    .into(),
            ),
        );
        tier.insert("file".into(), Value::String(model.to_owned()));
        tier.insert("ctx".into(), Value::from(ctx));
        tier.insert("ngl".into(), Value::from(ngl));
        tier.insert("backend".into(), Value::String(backend.clone()));
        for source in ["cache_type_k", "cache_type_v", "mmproj"] {
            if let Some(value) = body.get(source).and_then(Value::as_str)
                && !value.trim().is_empty()
            {
                let target = match source {
                    "cache_type_k" => "cache_k",
                    "cache_type_v" => "cache_v",
                    _ => "mmproj",
                };
                tier.insert(target.into(), Value::String(value.trim().into()));
            }
        }
        if let Some(visible) = body.get("visible_devices").and_then(Value::as_str)
            && !visible.trim().is_empty()
        {
            let variable = if backend.eq_ignore_ascii_case("cuda") {
                "CUDA_VISIBLE_DEVICES"
            } else {
                "HIP_VISIBLE_DEVICES"
            };
            tier.insert(
                "pin".into(),
                Value::String(format!("{variable}={}", visible.trim())),
            );
        }
        let mut extra = Vec::new();
        if let Some(threads) = body.get("threads").and_then(Value::as_i64) {
            extra.push("-t".to_owned());
            extra.push(threads.to_string());
        }
        if flag(body, "flash_attn") {
            extra.push("-fa".into());
            extra.push("on".into());
        }
        if let Some(slots) = body.get("parallel_slots").and_then(Value::as_i64) {
            extra.push("-np".into());
            extra.push(slots.to_string());
        }
        for (field, flag_name) in [
            ("batch_size", "-b"),
            ("ubatch_size", "-ub"),
            ("alias", "--alias"),
            ("reasoning", "--reasoning"),
            ("api_key", "--api-key"),
        ] {
            if let Some(value) = body.get(field).and_then(Value::as_str)
                && !value.trim().is_empty()
            {
                extra.push(flag_name.into());
                extra.push(value.trim().into());
            }
        }
        if flag(body, "mlock") {
            extra.push("--mlock".into());
        }
        if flag(body, "no_mmap") {
            extra.push("--no-mmap".into());
        }
        if !extra.is_empty() {
            tier.insert(
                "engine_args".into(),
                Value::Array(extra.into_iter().map(Value::String).collect()),
            );
        }
        let stack = serde_json::json!({
            "name": stack_name,
            "version": 1,
            "tiers": [Value::Object(tier)],
            "routing": {},
        });
        let stack = stacks::parse_stack(&stack)
            .map_err(|error| ApiError::new(422, format!("invalid server launch: {error}")))?;
        let base_port = body
            .get("port")
            .and_then(Value::as_u64)
            .filter(|port| *port > 0 && *port <= u16::MAX as u64)
            .map(|port| port as u16);
        self.apply_value(stack, key.clone(), base_port).await?;
        self.rename_first_tier(&key);
        let tier = self.first_tier().unwrap_or(Value::Null);
        let engines = self.engines_snapshot();
        let instance = self.instance_for_tier(&tier, &engines, &timestamp());
        let mut out = instance;
        if let Some(object) = out.as_object_mut() {
            object.insert("ngl".into(), Value::from(ngl));
        }
        Ok(out)
    }

    pub async fn stop_server(&self, key: Option<&str>) -> Result<Value, ApiError> {
        let Some(key) = key else {
            let _guard = self.apply_lock.lock().await;
            let current = self.take_applied();
            let Some(current) = current else {
                return Ok(serde_json::json!({"ok": true, "stopped": 0}));
            };
            let stopped = current.engines.len();
            unload(current).await;
            return Ok(serde_json::json!({"ok": true, "stopped": stopped}));
        };
        let _guard = self.apply_lock.lock().await;
        let current = self.take_applied();
        let Some(current) = current else {
            return Err(ApiError::new(
                404,
                format!("no such loaded instance: {key}"),
            ));
        };
        let matches = current.tiers.iter().any(|tier| {
            tier.get("key").and_then(Value::as_str) == Some(key)
                || tier.get("role").and_then(Value::as_str) == Some(key)
        });
        if !matches {
            self.put_applied(current);
            return Err(ApiError::new(
                404,
                format!("no such loaded instance: {key}"),
            ));
        }
        unload(current).await;
        Ok(serde_json::json!({
            "ok": true,
            "unloaded": key,
            "drained": false,
            "port_free": true,
        }))
    }

    pub async fn unload_server(&self, key: &str) -> Result<Value, ApiError> {
        self.stop_server(Some(key)).await
    }

    pub async fn unload_loaded(&self) -> Value {
        let _guard = self.apply_lock.lock().await;
        let current = self.take_applied();
        let keys: Vec<Value> = current
            .as_ref()
            .map(|applied| {
                applied
                    .tiers
                    .iter()
                    .filter_map(|tier| tier.get("key").and_then(Value::as_str))
                    .map(|key| Value::String(key.to_owned()))
                    .collect()
            })
            .unwrap_or_default();
        if let Some(current) = current {
            unload(current).await;
        }
        serde_json::json!({
            "instances": keys,
            "retired": [],
            "strata": null,
            "drained": false,
        })
    }

    pub fn server_status(&self) -> Value {
        let (tiers, engines, started_at) = {
            let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            match state.as_ref() {
                Some(applied) => (
                    applied.tiers.clone(),
                    applied.engines.clone(),
                    applied.started_at.clone(),
                ),
                None => (Vec::new(), Vec::new(), String::new()),
            }
        };
        let instances: Vec<Value> = tiers
            .iter()
            .map(|tier| self.instance_for_tier(tier, &engines, &started_at))
            .collect();

        let primary = instances.first();
        let loaded: Vec<Value> = instances
            .iter()
            .map(|instance| {
                serde_json::json!({
                    "backend": instance.get("backend").cloned().unwrap_or(Value::Null),
                    "key": instance.get("key").cloned().unwrap_or(Value::Null),
                    "model": instance.get("model").cloned().unwrap_or(Value::Null),
                    "port": instance.get("port").cloned().unwrap_or(Value::Null),
                    "base_url": instance.get("base_url").cloned().unwrap_or(Value::Null),
                    "healthy": instance.get("healthy").cloned().unwrap_or(Value::Bool(false)),
                })
            })
            .collect();
        let port = primary
            .and_then(|instance| instance.get("port").cloned())
            .unwrap_or_else(|| Value::from(self.options.base_port));
        let base_url = primary
            .and_then(|instance| instance.get("base_url").cloned())
            .unwrap_or_else(|| {
                Value::String(format!("http://127.0.0.1:{}", self.options.base_port))
            });
        let binary = self.options.llama_server.clone();
        let local_models = hive_models::Library::new(&self.paths.models_dir).rows();
        let mut hardware = hardware_summary(&self.paths);
        hardware["source"] = Value::String("rust-daemon".into());
        serde_json::json!({
            "running": !instances.is_empty(),
            "healthy": !instances.is_empty(),
            "binary": binary.as_ref().map(|path| path.to_string_lossy().into_owned()).unwrap_or_default(),
            "binary_found": binary.as_ref().is_some_and(|path| path.is_file()),
            "port": port,
            "base_url": base_url,
            "model": primary.and_then(|instance| instance.get("model").cloned()).unwrap_or(Value::Null),
            "pid": primary.and_then(|instance| instance.get("pid").cloned()).unwrap_or(Value::Null),
            "started_at": if started_at.is_empty() { Value::Null } else { Value::String(started_at) },
            "instances": instances,
            "local_models": local_models,
            "downloads": [],
            "hardware": hardware,
            "strata": {"alive": false},
            "loaded": loaded,
            "loaded_model": loaded.first().and_then(|row| row.get("model").cloned()).unwrap_or(Value::Null),
        })
    }

    pub fn server_log(&self, tail: usize) -> Value {
        let mut lines: Vec<String> = Vec::new();
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&self.paths.log_dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("llama_server") && name.ends_with(".log"))
            })
            .collect();
        paths.sort();
        for path in paths {
            if let Ok(text) = std::fs::read_to_string(&path) {
                lines.extend(text.lines().map(str::to_owned));
            }
        }
        let tail = tail.max(1);
        let start = lines.len().saturating_sub(tail);
        serde_json::json!({"lines": &lines[start..]})
    }

    pub fn server_memory(&self, conversations: u64) -> Value {
        let (rss_mb, threads) = process_memory().unwrap_or((0.0, 0));
        serde_json::json!({
            "rss_mb": rss_mb,
            "conversations_in_memory": conversations,
            "max_conversations": Value::Null,
            "loggers": 0,
            "downloads": 0,
            "threads": threads,
        })
    }

    pub fn server_metrics(&self) -> Value {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let Some(applied) = state.as_ref() else {
            return serde_json::json!({"available": false});
        };
        let port = applied.tiers.iter().find_map(|tier| {
            tier.get("port")
                .and_then(Value::as_u64)
                .filter(|port| *port > 0)
        });
        drop(state);
        let Some(port) = port else {
            return serde_json::json!({"available": false});
        };
        let response = match http::get(
            &format!("http://127.0.0.1:{port}/metrics"),
            Duration::from_secs(3),
        ) {
            Ok(response) if response.status == 200 => response,
            _ => return serde_json::json!({"available": false}),
        };
        let body = String::from_utf8_lossy(&response.body);
        let mut gauges = Map::new();
        for line in body.lines() {
            if line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once(' ') else {
                continue;
            };
            if let Ok(value) = value.trim().parse::<f64>() {
                gauges.insert(
                    key.to_owned(),
                    serde_json::Number::from_f64(value)
                        .map(Value::Number)
                        .unwrap_or(Value::Null),
                );
            }
        }
        serde_json::json!({"available": true, "gauges": gauges})
    }

    pub fn server_fit(&self, model_gb: f64, ctx: i64, kv: &str) -> Value {
        let hardware = hardware_summary(&self.paths);
        let vram = hardware
            .get("combined_vram_gb")
            .and_then(Value::as_f64)
            .or_else(|| hardware.get("vram_gb").and_then(Value::as_f64))
            .unwrap_or(0.0);
        let ram = hardware
            .get("available_ram_gb")
            .and_then(Value::as_f64)
            .or_else(|| hardware.get("total_ram_gb").and_then(Value::as_f64))
            .unwrap_or(0.0);
        let disk = hardware
            .get("disk")
            .and_then(|disk| disk.get("repo"))
            .and_then(|repo| repo.get("free_gb"))
            .and_then(Value::as_f64);
        fit_plan(model_gb, ctx, vram, ram, kv, disk)
    }

    async fn apply_value(
        self: &Arc<Self>,
        stack: Value,
        name: String,
        base_port: Option<u16>,
    ) -> Result<Value, ApiError> {
        let _guard = self.apply_lock.lock().await;
        let previous = self.take_applied();
        if let Some(previous) = previous {
            unload(previous).await;
        }
        let mut options = self.options.clone();
        if let Some(port) = base_port {
            options.base_port = port;
        }
        let stack_for_apply = stack.clone();
        let applied =
            tokio::task::spawn_blocking(move || hive_supervisor::apply(&stack_for_apply, &options))
                .await
                .map_err(|error| ApiError::new(500, format!("stack worker failed: {error}")))?
                .map_err(|error| ApiError::new(409, format!("cannot apply {name:?}: {error}")))?;
        let tiers = applied.tiers.clone();
        let warnings = applied.warnings.clone();
        let engines = applied.engines();
        let rows = apply_rows(&tiers);
        let started_at = timestamp();
        *self.state.lock().unwrap_or_else(|error| error.into_inner()) = Some(AppliedStack {
            name,
            tiers,
            warnings,
            engines,
            started_at,
            applied,
        });
        Ok(serde_json::json!({"ok": true, "tiers": rows}))
    }

    fn engines_snapshot(&self) -> Vec<EngineInfo> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .map(|applied| applied.engines.clone())
            .unwrap_or_default()
    }

    fn instance_for_tier(&self, tier: &Value, engines: &[EngineInfo], started_at: &str) -> Value {
        let key = tier.get("key").and_then(Value::as_str).unwrap_or("");
        let port = tier.get("port").and_then(Value::as_u64).unwrap_or(0);
        let role = tier.get("role").and_then(Value::as_str).unwrap_or("");
        let engine = engines.iter().find(|engine| engine.role == role);
        serde_json::json!({
            "key": key,
            "running": true,
            "healthy": true,
            "port": port,
            "base_url": format!("http://127.0.0.1:{port}"),
            "model": tier.get("model").and_then(Value::as_str).unwrap_or(""),
            "pid": engine.map(|engine| engine.pid).or_else(|| tier.get("pid").and_then(Value::as_u64).map(|pid| pid as u32)),
            "adopted": tier.get("managed").and_then(Value::as_bool) == Some(false) && tier.get("base_url").and_then(Value::as_str).is_some(),
            "started_at": started_at,
            "backend": tier.get("backend").and_then(Value::as_str).unwrap_or(""),
            "binary": self.options.llama_server.as_ref().map(|path| path.to_string_lossy().into_owned()).unwrap_or_default(),
            "embedding": false,
        })
    }

    fn first_tier(&self) -> Option<Value> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .and_then(|applied| applied.tiers.first().cloned())
    }

    fn rename_first_tier(&self, key: &str) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(applied) = state.as_mut()
            && let Some(tier) = applied.tiers.first_mut()
            && let Some(object) = tier.as_object_mut()
        {
            object.insert("key".into(), Value::String(key.to_owned()));
        }
    }

    fn take_applied(&self) -> Option<AppliedStack> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
    }

    fn put_applied(&self, applied: AppliedStack) {
        *self.state.lock().unwrap_or_else(|error| error.into_inner()) = Some(applied);
    }
}

async fn unload(applied: AppliedStack) {
    let _ = tokio::task::spawn_blocking(move || applied.applied.unload()).await;
}

fn apply_rows(tiers: &[Value]) -> Vec<Value> {
    tiers
        .iter()
        .map(|tier| {
            serde_json::json!({
                "role": tier.get("role").cloned().unwrap_or(Value::Null),
                "port": tier.get("port").cloned().unwrap_or(Value::Null),
                "key": tier.get("key").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

fn stack_name(name: &str) -> Result<String, ApiError> {
    stacks::check_stack_name(name).map_err(|error| ApiError::new(400, error))
}

fn read_stack(directory: &Path, name: &str) -> Result<Value, ApiError> {
    let name = stack_name(name)?;
    let path = directory.join(format!("{name}.json"));
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(ApiError::new(404, format!("no such stack: {name}")));
        }
        Err(error) => {
            return Err(ApiError::new(
                500,
                format!("cannot read stack {name}: {error}"),
            ));
        }
    };
    let data: Value = serde_json::from_str(&text)
        .map_err(|error| ApiError::new(400, format!("invalid stack {name:?}: {error}")))?;
    stacks::parse_stack(&data)
        .map_err(|error| ApiError::new(400, format!("invalid stack {name:?}: {error}")))
}

fn write_stack(directory: &Path, name: &str, stack: &Value) -> Result<(), ApiError> {
    std::fs::create_dir_all(directory).map_err(|error| {
        ApiError::new(
            500,
            format!("cannot create {}: {error}", directory.display()),
        )
    })?;
    let path = directory.join(format!("{name}.json"));
    let temporary = directory.join(format!(".{name}.json.{}", std::process::id()));
    let text = serde_json::to_string_pretty(stack)
        .map_err(|error| ApiError::new(500, format!("cannot encode stack: {error}")))?;
    std::fs::write(&temporary, text + "\n").map_err(|error| {
        ApiError::new(
            500,
            format!("cannot write {}: {error}", temporary.display()),
        )
    })?;
    std::fs::rename(&temporary, &path)
        .map_err(|error| ApiError::new(500, format!("cannot save {}: {error}", path.display())))
}

fn flag(body: &Value, key: &str) -> bool {
    body.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn timestamp() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn process_memory() -> Option<(f64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let mut rss_mb = 0.0;
    for line in status.lines() {
        if let Some(value) = line.strip_prefix("VmRSS:") {
            let kib = value.split_whitespace().next()?.parse::<f64>().ok()?;
            rss_mb = (kib / 1024.0 * 10.0).round() / 10.0;
        }
    }
    let threads = std::fs::read_dir("/proc/self/task")
        .map(|entries| entries.count() as u64)
        .unwrap_or(0);
    Some((rss_mb, threads))
}

fn hardware_summary(paths: &Paths) -> Value {
    let devices = hardware::amd_devices().unwrap_or_default();
    let (available_ram_gb, total_ram_gb) = hardware::memory_gib().unwrap_or((8.0, 8.0));
    let (vram_total, vram_free) = devices.iter().fold((0.0, 0.0), |(total, free), device| {
        (
            total + device.memory_gb.unwrap_or(0.0),
            free + device.free_gb.unwrap_or(0.0),
        )
    });
    let rows: Vec<Value> = devices
        .iter()
        .map(|device| {
            serde_json::json!({
                "backend": "rocm",
                "name": device.card,
                "index": device.index,
                "memory_gb": device.memory_gb,
                "used_gb": device.used_gb,
                "free_gb": device.free_gb,
                "visible": true,
            })
        })
        .collect();
    let disk_free = hardware::disk_free_gib(&paths.repo).ok();
    let vram_gb = (!devices.is_empty()).then_some(round2(vram_total));
    serde_json::json!({
        "cpu_count": std::thread::available_parallelism().map(|count| count.get()).unwrap_or(0),
        "total_ram_gb": round2(total_ram_gb),
        "available_ram_gb": round2(available_ram_gb),
        "vram_gb": vram_gb,
        "vram_free_gb": (!devices.is_empty()).then_some(round2(vram_free)),
        "combined_vram_gb": vram_gb,
        "devices": rows,
        "vram_source": if devices.is_empty() { "ram" } else { "sysfs" },
        "available_gb": if devices.is_empty() { round2(available_ram_gb) } else { round2(vram_free) },
        "disk": {"repo": {"free_gb": disk_free}},
    })
}

fn fit_plan(
    model_gb: f64,
    ctx: i64,
    vram_gb: f64,
    ram_gb: f64,
    kv: &str,
    disk_free_gb: Option<f64>,
) -> Value {
    let model_gb = model_gb.max(0.0);
    let ctx = ctx.max(0) as f64;
    let per_1k = match kv.to_ascii_lowercase().as_str() {
        "f16" => 0.14,
        "q4_0" => 0.035,
        _ => 0.07,
    };
    let kv_gb = (ctx / 1024.0) * per_1k;
    let tier1 = model_gb.min(vram_gb.max(0.0));
    let remaining = model_gb - tier1;
    let ram = ram_gb.max(0.0);
    let ram_weights = remaining.min(ram);
    let weights_disk = remaining - ram_weights;
    let ram_left = ram - ram_weights;
    let ram_kv = kv_gb.min(ram_left.max(0.0));
    let kv_disk = kv_gb - ram_kv;
    let tier2 = ram_weights + ram_kv;
    let tier3 = weights_disk + kv_disk;
    let disk_full = disk_free_gb.is_some_and(|free| tier3 > 0.0 && free - tier3 < free * 0.2);
    let io_warning = kv_disk > 0.0 || ctx > 131072.0;
    let cap = if per_1k > 0.0 {
        (ram_left.max(0.0) * 1024.0 / per_1k) as i64
    } else {
        0
    };
    serde_json::json!({
        "metrics": {
            "total_gb": round2(model_gb + kv_gb),
            "model_gb": round2(model_gb),
            "kv_gb": round2(kv_gb),
            "tier1_vram_gb": round2(tier1),
            "tier2_ram_gb": round2(tier2),
            "tier3_disk_gb": round2(tier3),
        },
        "flags": {
            "io_latency_warning": io_warning,
            "disk_full": disk_full,
            "recommend_cap": cap,
        },
    })
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_plan_puts_the_overflow_in_the_next_tier() {
        let plan = fit_plan(10.0, 8192, 4.0, 4.0, "q8_0", Some(100.0));
        assert_eq!(plan["metrics"]["tier1_vram_gb"], 4.0);
        assert_eq!(plan["metrics"]["tier2_ram_gb"], 4.0);
        assert_eq!(plan["metrics"]["tier3_disk_gb"], 2.56);
        assert_eq!(plan["flags"]["io_latency_warning"], true);
    }

    #[test]
    fn stack_name_rejects_paths() {
        assert!(stack_name("../etc/passwd").is_err());
        assert_eq!(stack_name("fast").unwrap(), "fast");
    }
}
