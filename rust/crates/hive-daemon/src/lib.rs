//! The Phase 2 daemon: native health/config/version routes, with a transparent
//! proxy fallback to the Python sidecar.
//!
//! This is deliberately a small first slice. It proves the ownership split:
//! Rust owns the public port and the routes in `GROUP_ONE`; every other path is
//! streamed to the sidecar unchanged. Route groups move into Rust one at a
//! time.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path as UrlPath, Query, Request, State};
use axum::http::header::{
    CACHE_CONTROL, CONNECTION, CONTENT_TYPE, HOST, HeaderMap, HeaderName, HeaderValue,
    TRANSFER_ENCODING, UPGRADE,
};
use axum::http::{Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use hive_core::backends::Manifest;
use hive_core::paths::Paths;
use hive_core::{platform_display, python, version};
use percent_encoding::percent_decode_str;
use serde_json::{Map, Value};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::trace::TraceLayer;

pub mod runtime;
pub mod sidecar;
use runtime::{ApiError, Runtime};
use sidecar::Sidecar;

pub const OWNER_HEADER: &str = "x-hive-owner";
pub const REQUEST_ID_HEADER: &str = "x-hive-request-id";
static REQUEST_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub struct DaemonConfig {
    pub host: String,
    pub port: u16,
    /// An already-running sidecar. When set, the daemon never spawns one.
    pub sidecar_url: Option<String>,
    /// Spawn `python -m harness` on `sidecar_port` when no URL is configured.
    pub spawn_sidecar: bool,
    pub sidecar_port: u16,
    /// `harness/webui` root; `None` disables static console serving.
    pub webui: Option<PathBuf>,
    pub token: Option<String>,
    pub paths: Paths,
}

impl DaemonConfig {
    pub fn new(host: impl Into<String>, port: u16, sidecar_url: Option<String>) -> Self {
        let token = std::env::var("HARNESS_TOKEN")
            .ok()
            .map(|token| token.trim().to_owned())
            .filter(|token| !token.is_empty());
        let paths = Paths::discover(None, None);
        let webui = default_webui(&paths);
        Self {
            host: host.into(),
            port,
            sidecar_url: sidecar_url.filter(|url| !url.trim().is_empty()),
            spawn_sidecar: false,
            sidecar_port: 8766,
            webui,
            token,
            paths,
        }
    }
}

fn default_webui(paths: &Paths) -> Option<PathBuf> {
    let candidate = paths.repo.join("harness/webui");
    candidate.is_dir().then_some(candidate)
}

pub struct AppState {
    pub config: DaemonConfig,
    client: reqwest::Client,
    managed: Option<Arc<Sidecar>>,
    runtime: Arc<Runtime>,
}

impl AppState {
    pub fn new(config: DaemonConfig) -> Self {
        Self::with_sidecar(config, None)
    }

    fn with_sidecar(config: DaemonConfig, managed: Option<Arc<Sidecar>>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .unwrap_or_default();
        let runtime = Arc::new(Runtime::new(&config.paths));
        Self {
            config,
            client,
            managed,
            runtime,
        }
    }
}

pub fn router(config: DaemonConfig) -> Router {
    router_with_sidecar(config, None)
}

fn router_with_sidecar(config: DaemonConfig, managed: Option<Arc<Sidecar>>) -> Router {
    let state = Arc::new(AppState::with_sidecar(config, managed));
    Router::new()
        .route("/health", get(health))
        .route("/v1/setup/status", get(setup_status))
        .route("/v1/version", get(version_route))
        .route("/v1/config", get(config_route))
        .route("/v1/stacks", get(list_stacks))
        .route("/v1/stacks/status", get(stack_status))
        .route(
            "/v1/stacks/{name}",
            get(get_stack).put(put_stack).delete(delete_stack),
        )
        .route("/v1/stacks/{name}/validate", post(validate_stack))
        .route("/v1/stacks/{name}/apply", post(apply_stack))
        .route("/v1/stacks/{name}/unload", post(unload_stack))
        .route("/v1/server/status", get(server_status))
        .route("/v1/server/log", get(server_log))
        .route("/v1/server/memory", get(server_memory))
        .route("/v1/server/metrics", get(server_metrics))
        .route("/v1/server/fit", get(server_fit))
        .route("/v1/server/start", post(server_start))
        .route("/v1/server/stop", post(server_stop))
        .route("/v1/server/unload", post(server_unload))
        .route("/v1/server/unload-loaded", post(server_unload_loaded))
        .route(
            "/v1/models/local",
            get(local_models).delete(delete_local_model),
        )
        .route("/v1/models/hub", get(hub_search))
        .route("/v1/models/hub/files/{*repo}", get(hub_files))
        .route("/v1/models/hub/download", post(hub_download))
        .route("/v1/models/hub/downloads", get(hub_downloads))
        .fallback(dispatch)
        .layer(middleware::from_fn_with_state(state.clone(), token_guard))
        .layer(middleware::from_fn(request_id))
        .layer(cors_layer())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run(mut config: DaemonConfig) -> Result<(), String> {
    let bind = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .map_err(|error| format!("cannot bind {bind}: {error}"))?;

    let managed = if config.spawn_sidecar && config.sidecar_url.is_none() {
        let python = python::resolve(&config.paths)?;
        let spec = sidecar::SidecarSpec {
            python,
            repo: config.paths.repo.clone(),
            state_dir: config.paths.state_dir.clone(),
            models_dir: config.paths.models_dir.clone(),
            log_dir: config.paths.log_dir.clone(),
            port: config.sidecar_port,
        };
        let sidecar = Arc::new(Sidecar::start(spec)?);
        config.sidecar_url = Some(sidecar.url().to_owned());
        Some(sidecar)
    } else {
        None
    };

    tracing::info!(
        %bind,
        sidecar = ?config.sidecar_url,
        managed = managed.is_some(),
        "hive daemon listening"
    );
    let result = axum::serve(listener, router_with_sidecar(config, managed.clone()))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|error| error.to_string());
    // Always take the sidecar down, including after a serve error.
    if let Some(sidecar) = managed {
        sidecar.shutdown().await;
    }
    result
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

fn cors_layer() -> CorsLayer {
    let raw = std::env::var("HARNESS_CORS_ORIGINS").unwrap_or_default();
    if raw.trim() == "*" {
        return CorsLayer::permissive();
    }
    let values: Vec<HeaderValue> = if raw.trim().is_empty() {
        [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8765",
            "http://127.0.0.1:8765",
        ]
        .into_iter()
        .filter_map(|origin| HeaderValue::from_str(origin).ok())
        .collect()
    } else {
        raw.split(',')
            .map(str::trim)
            .filter(|origin| !origin.is_empty())
            .filter_map(|origin| HeaderValue::from_str(origin).ok())
            .collect()
    };
    CorsLayer::new()
        .allow_origin(AllowOrigin::list(values))
        .allow_methods(Any)
        .allow_headers(Any)
}

async fn request_id(request: Request, next: Next) -> Response {
    let id = format!(
        "{}-{}",
        std::process::id(),
        REQUEST_SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let Ok(value) = HeaderValue::from_str(&id) else {
        return next.run(request).await;
    };
    let mut request = request;
    request
        .headers_mut()
        .insert(HeaderName::from_static(REQUEST_ID_HEADER), value.clone());
    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert(HeaderName::from_static(REQUEST_ID_HEADER), value);
    response
}

async fn token_guard(State(state): State<Arc<AppState>>, request: Request, next: Next) -> Response {
    let required = state.config.token.as_deref();
    if let Some(required) = required
        && request.uri().path().starts_with("/v1/")
    {
        let supplied = request
            .headers()
            .get("x-splinter-token")
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        if !token_matches(supplied, required) {
            return owned(
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"detail": "invalid or missing token"})).into_response(),
            );
        }
    }
    next.run(request).await
}

fn token_matches(supplied: &str, required: &str) -> bool {
    let supplied = supplied.as_bytes();
    let required = required.as_bytes();
    if supplied.len() != required.len() {
        return false;
    }
    let mut difference = 0u8;
    for (left, right) in supplied.iter().zip(required) {
        difference |= left ^ right;
    }
    difference == 0
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    let (ok, conversations, error) = match sidecar_health(&state).await {
        Ok(conversations) => (true, conversations, None),
        Err(error) => (false, 0, Some(error)),
    };
    let mut payload = Map::new();
    payload.insert("ok".into(), Value::Bool(ok));
    payload.insert("conversations".into(), Value::from(conversations));
    let mut sidecar = Map::new();
    sidecar.insert("managed".into(), Value::Bool(state.managed.is_some()));
    if let Some(url) = &state.config.sidecar_url {
        sidecar.insert("url".into(), Value::String(url.clone()));
    }
    if let Some(managed) = &state.managed
        && let Some(pid) = managed.pid().await
    {
        sidecar.insert("pid".into(), Value::from(pid));
    }
    payload.insert("sidecar".into(), Value::Object(sidecar));
    if let Some(error) = error {
        payload.insert("error".into(), Value::String(error));
    }
    owned(StatusCode::OK, Json(Value::Object(payload)).into_response())
}

async fn sidecar_health(state: &AppState) -> Result<u64, String> {
    let Some(base) = &state.config.sidecar_url else {
        return Ok(0);
    };
    let url = format!("{}/health", base.trim_end_matches('/'));
    let response = state
        .client
        .get(url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(format!("sidecar health HTTP {}", response.status()));
    }
    let data: Value = response.json().await.map_err(|error| error.to_string())?;
    Ok(data
        .get("conversations")
        .and_then(Value::as_u64)
        .unwrap_or(0))
}

async fn setup_status(State(state): State<Arc<AppState>>, request: Request) -> Response {
    if cfg!(windows) {
        return proxy(State(state), request).await;
    }
    owned(
        StatusCode::GONE,
        Json(serde_json::json!({
            "detail": "the VHDX/WSL2 setup path is Windows-only and quarantined on Linux"
        }))
        .into_response(),
    )
}

async fn version_route(State(state): State<Arc<AppState>>) -> Response {
    let python_path = python::find(&state.config.paths);
    let payload = serde_json::json!({
        "hive": version(),
        "python": python::version(&python_path),
        "platform": platform_display(),
    });
    owned(StatusCode::OK, Json(payload).into_response())
}

async fn config_route(State(state): State<Arc<AppState>>) -> Response {
    let paths = &state.config.paths;
    let manifest = Manifest::load(paths.manifest.as_deref(), &paths.home).ok();
    let installed: Vec<Value> = manifest
        .as_ref()
        .map(|manifest| {
            manifest
                .installed()
                .into_iter()
                .map(|name| Value::String(name.to_owned()))
                .collect()
        })
        .unwrap_or_default();
    let default_backend = manifest
        .as_ref()
        .and_then(|manifest| {
            manifest
                .for_accelerator("amd")
                .or_else(|| manifest.for_accelerator("cpu"))
        })
        .map(str::to_owned)
        .map(Value::String)
        .unwrap_or(Value::Null);
    let python_path = python::find(paths);
    let payload = serde_json::json!({
        "hive": version(),
        "python": python_path.to_string_lossy().into_owned(),
        "repo": paths.repo.to_string_lossy().into_owned(),
        "models_dir": paths.models_dir.to_string_lossy().into_owned(),
        "stacks_dir": paths.stacks_dir.to_string_lossy().into_owned(),
        "state_dir": paths.state_dir.to_string_lossy().into_owned(),
        "serve_dir": paths.serve_dir.to_string_lossy().into_owned(),
        "log_dir": paths.log_dir.to_string_lossy().into_owned(),
        "default_backend": default_backend,
        "installed_backends": installed,
    });
    owned(StatusCode::OK, Json(payload).into_response())
}

fn runtime_json(result: Result<Value, ApiError>) -> Response {
    match result {
        Ok(value) => owned(StatusCode::OK, Json(value).into_response()),
        Err(error) => runtime_error(error),
    }
}

fn runtime_error(error: ApiError) -> Response {
    let status = StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    owned(
        status,
        Json(serde_json::json!({"detail": error.detail})).into_response(),
    )
}

async fn list_stacks(State(state): State<Arc<AppState>>) -> Response {
    runtime_json(Ok(state.runtime.list_stacks()))
}

async fn stack_status(State(state): State<Arc<AppState>>) -> Response {
    runtime_json(Ok(state.runtime.status()))
}

async fn get_stack(State(state): State<Arc<AppState>>, UrlPath(name): UrlPath<String>) -> Response {
    runtime_json(state.runtime.load_stack(&name))
}

async fn put_stack(
    State(state): State<Arc<AppState>>,
    UrlPath(name): UrlPath<String>,
    Json(body): Json<Value>,
) -> Response {
    runtime_json(state.runtime.save_stack(&name, &body))
}

async fn delete_stack(
    State(state): State<Arc<AppState>>,
    UrlPath(name): UrlPath<String>,
) -> Response {
    runtime_json(state.runtime.delete_stack(&name))
}

async fn validate_stack(
    State(state): State<Arc<AppState>>,
    UrlPath(name): UrlPath<String>,
) -> Response {
    let runtime = Arc::clone(&state.runtime);
    match tokio::task::spawn_blocking(move || runtime.validate_stack(&name)).await {
        Ok(result) => runtime_json(result),
        Err(error) => runtime_error(ApiError {
            status: 500,
            detail: format!("validate worker failed: {error}"),
        }),
    }
}

async fn apply_stack(
    State(state): State<Arc<AppState>>,
    UrlPath(name): UrlPath<String>,
) -> Response {
    runtime_json(state.runtime.apply_stack(&name, None).await)
}

async fn unload_stack(
    State(state): State<Arc<AppState>>,
    UrlPath(name): UrlPath<String>,
) -> Response {
    runtime_json(state.runtime.unload_stack(&name).await)
}

async fn server_status(State(state): State<Arc<AppState>>) -> Response {
    let runtime = Arc::clone(&state.runtime);
    match tokio::task::spawn_blocking(move || runtime.server_status()).await {
        Ok(value) => runtime_json(Ok(value)),
        Err(error) => runtime_error(ApiError {
            status: 500,
            detail: format!("status worker failed: {error}"),
        }),
    }
}

async fn server_log(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let tail = query
        .get("tail")
        .and_then(|text| text.parse::<usize>().ok())
        .unwrap_or(120);
    let runtime = Arc::clone(&state.runtime);
    match tokio::task::spawn_blocking(move || runtime.server_log(tail)).await {
        Ok(value) => runtime_json(Ok(value)),
        Err(error) => runtime_error(ApiError {
            status: 500,
            detail: format!("log worker failed: {error}"),
        }),
    }
}

async fn server_memory(State(state): State<Arc<AppState>>) -> Response {
    let conversations = sidecar_health(&state).await.unwrap_or(0);
    let runtime = Arc::clone(&state.runtime);
    match tokio::task::spawn_blocking(move || runtime.server_memory(conversations)).await {
        Ok(value) => runtime_json(Ok(value)),
        Err(error) => runtime_error(ApiError {
            status: 500,
            detail: format!("memory worker failed: {error}"),
        }),
    }
}

async fn server_metrics(State(state): State<Arc<AppState>>) -> Response {
    let runtime = Arc::clone(&state.runtime);
    match tokio::task::spawn_blocking(move || runtime.server_metrics()).await {
        Ok(value) => runtime_json(Ok(value)),
        Err(error) => runtime_error(ApiError {
            status: 500,
            detail: format!("metrics worker failed: {error}"),
        }),
    }
}

async fn server_fit(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let model_gb = query
        .get("model_gb")
        .and_then(|text| text.parse::<f64>().ok())
        .unwrap_or(0.0);
    let ctx = query
        .get("ctx")
        .and_then(|text| text.parse::<i64>().ok())
        .unwrap_or(8192);
    let kv = query.get("kv").cloned().unwrap_or_else(|| "q8_0".into());
    let runtime = Arc::clone(&state.runtime);
    match tokio::task::spawn_blocking(move || runtime.server_fit(model_gb, ctx, &kv)).await {
        Ok(value) => runtime_json(Ok(value)),
        Err(error) => runtime_error(ApiError {
            status: 500,
            detail: format!("fit worker failed: {error}"),
        }),
    }
}

async fn server_start(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    runtime_json(state.runtime.start_server(&body).await)
}

async fn server_stop(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    runtime_json(
        state
            .runtime
            .stop_server(query.get("key").map(String::as_str))
            .await,
    )
}

async fn server_unload(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let Some(key) = body
        .get("key")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|key| !key.is_empty())
    else {
        return runtime_error(ApiError {
            status: 422,
            detail: "key is required".into(),
        });
    };
    runtime_json(state.runtime.unload_server(key).await)
}

async fn server_unload_loaded(State(state): State<Arc<AppState>>) -> Response {
    runtime_json(Ok(state.runtime.unload_loaded().await))
}

async fn local_models(State(state): State<Arc<AppState>>) -> Response {
    let runtime = Arc::clone(&state.runtime);
    match tokio::task::spawn_blocking(move || runtime.local_models()).await {
        Ok(value) => runtime_json(Ok(value)),
        Err(error) => runtime_error(ApiError {
            status: 500,
            detail: format!("library worker failed: {error}"),
        }),
    }
}

async fn delete_local_model(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let Some(file) = query.get("file") else {
        return runtime_error(ApiError {
            status: 422,
            detail: "file is required".into(),
        });
    };
    runtime_json(state.runtime.delete_local(file))
}

async fn hub_search(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let q = query.get("q").cloned().unwrap_or_default();
    let limit = query
        .get("limit")
        .and_then(|text| text.parse::<i64>().ok())
        .unwrap_or(12);
    runtime_json(state.runtime.hub_search(&q, limit).await)
}

async fn hub_files(State(state): State<Arc<AppState>>, UrlPath(repo): UrlPath<String>) -> Response {
    runtime_json(state.runtime.hub_files(repo.trim_start_matches('/')).await)
}

async fn hub_download(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let repo = body.get("repo").and_then(Value::as_str).unwrap_or("");
    let file = body.get("file").and_then(Value::as_str).unwrap_or("");
    runtime_json(state.runtime.hub_download(repo, file))
}

async fn hub_downloads(State(state): State<Arc<AppState>>) -> Response {
    runtime_json(Ok(state.runtime.hub_downloads()))
}

/// The console shell, its `/api` prefix, and the Python-only API all arrive
/// here. Static assets win; everything else falls through to the sidecar.
async fn dispatch(State(state): State<Arc<AppState>>, request: Request) -> Response {
    let path = request.uri().path().to_owned();
    if path == "/" || path == "/index.html" {
        return static_response(&state, Path::new("index.html"), request.method(), true).await;
    }
    if path == "/api" || path.starts_with("/api/") {
        let raw = request
            .uri()
            .path_and_query()
            .map(|value| value.as_str())
            .unwrap_or("/api")
            .to_owned();
        let target = api_target(&raw);
        return proxy_to(state, request, target).await;
    }
    if is_sidecar_passthrough(&path) {
        return proxy(State(state), request).await;
    }
    if matches!(*request.method(), Method::GET | Method::HEAD)
        && let Some(relative) = safe_relative(&path)
        && let Some(root) = &state.config.webui
    {
        let candidate = root.join(&relative);
        if tokio::fs::metadata(&candidate)
            .await
            .map(|meta| meta.is_file())
            .unwrap_or(false)
        {
            let no_store = relative
                .extension()
                .and_then(|ext| ext.to_str())
                .is_none_or(|ext| ext.eq_ignore_ascii_case("html"));
            return static_response(&state, &relative, request.method(), no_store).await;
        }
    }
    proxy(State(state), request).await
}

fn is_sidecar_passthrough(path: &str) -> bool {
    path.starts_with("/v1/")
        || path.starts_with("/view/")
        || path == "/cascade"
        || path.starts_with("/cascade/")
        || path == "/docs"
        || path == "/openapi.json"
        || path == "/redoc"
}

/// `/api/v1/models` → `/v1/models`; the console's local role editor keeps its
/// own name and maps onto the sidecar's `/v1/roles`.
fn api_target(raw: &str) -> String {
    let (path, query) = match raw.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (raw, None),
    };
    let mapped = if path == "/api/roles" {
        "/v1/roles".to_owned()
    } else if let Some(rest) = path.strip_prefix("/api") {
        if rest.is_empty() {
            "/".to_owned()
        } else {
            rest.to_owned()
        }
    } else {
        path.to_owned()
    };
    match query {
        Some(query) => format!("{mapped}?{query}"),
        None => mapped,
    }
}

fn safe_relative(path: &str) -> Option<PathBuf> {
    let decoded = percent_decode_str(path).decode_utf8().ok()?;
    let mut relative = PathBuf::new();
    for component in Path::new(decoded.trim_start_matches('/')).components() {
        match component {
            Component::Normal(part) => relative.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    (!relative.as_os_str().is_empty()).then_some(relative)
}

async fn static_response(
    state: &AppState,
    relative: &Path,
    method: &Method,
    no_store: bool,
) -> Response {
    let Some(root) = &state.config.webui else {
        return not_found("console assets are disabled");
    };
    let path = root.join(relative);
    let bytes = match tokio::fs::read(&path).await {
        Ok(bytes) => bytes,
        Err(error) => {
            return owned(
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "detail": format!("no console asset at /{}: {error}", relative.display())
                }))
                .into_response(),
            );
        }
    };
    let content_type = mime_guess::from_path(relative)
        .first_or_octet_stream()
        .to_string();
    let cache = if no_store {
        "no-store"
    } else {
        "public, max-age=300"
    };
    let builder = Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, content_type)
        .header(CACHE_CONTROL, cache);
    let response = if *method == Method::HEAD {
        builder
            .header(axum::http::header::CONTENT_LENGTH, bytes.len())
            .body(Body::empty())
    } else {
        builder.body(Body::from(bytes))
    };
    match response {
        Ok(response) => owned(StatusCode::OK, response),
        Err(_) => not_found("cannot read console asset"),
    }
}

fn not_found(detail: &str) -> Response {
    owned(
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "detail": detail })).into_response(),
    )
}

async fn proxy(State(state): State<Arc<AppState>>, request: Request) -> Response {
    let path = request
        .uri()
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or("/")
        .to_owned();
    proxy_to(state, request, path).await
}

async fn proxy_to(state: Arc<AppState>, request: Request, path: String) -> Response {
    let Some(base) = &state.config.sidecar_url else {
        return owned(
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "no Python sidecar configured"})).into_response(),
        );
    };
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    let method = match Method::from_bytes(request.method().as_str().as_bytes()) {
        Ok(method) => method,
        Err(_) => {
            return owned(
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"detail": "invalid HTTP method"})).into_response(),
            );
        }
    };
    let headers = forward_headers(request.headers());
    let body = reqwest::Body::wrap_stream(request.into_body().into_data_stream());
    let upstream = match state
        .client
        .request(method, &url)
        .headers(headers)
        .body(body)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            return owned(
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "detail": format!("sidecar unavailable: {error}")
                }))
                .into_response(),
            );
        }
    };
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut headers = HeaderMap::new();
    for (name, value) in upstream.headers() {
        if is_hop_by_hop(name) || name == axum::http::header::CONTENT_LENGTH {
            continue;
        }
        headers.insert(name.clone(), value.clone());
    }
    headers.insert(
        HeaderName::from_static(OWNER_HEADER),
        HeaderValue::from_static("python"),
    );
    let stream = upstream.bytes_stream();
    Response::builder()
        .status(status)
        .body(Body::from_stream(stream))
        .map(|mut response| {
            *response.headers_mut() = headers;
            response
        })
        .unwrap_or_else(|_| {
            owned(
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"detail": "cannot build proxy response"})).into_response(),
            )
        })
}

fn forward_headers(headers: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, value) in headers {
        if is_hop_by_hop(name) || name == HOST {
            continue;
        }
        out.insert(name.clone(), value.clone());
    }
    out
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    name == CONNECTION || name == TRANSFER_ENCODING || name == UPGRADE
}

fn owned(status: StatusCode, response: Response) -> Response {
    let mut response = response;
    response.headers_mut().insert(
        HeaderName::from_static(OWNER_HEADER),
        HeaderValue::from_static("rust"),
    );
    *response.status_mut() = status;
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn app() -> Router {
        let mut config = DaemonConfig::new("127.0.0.1", 0, None);
        config.paths = Paths::discover(None, None);
        router(config)
    }

    #[tokio::test]
    async fn setup_status_is_410_on_linux() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/v1/setup/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::GONE);
    }

    #[tokio::test]
    async fn version_and_config_are_owned_by_rust() {
        for path in ["/v1/version", "/v1/config"] {
            let response = app()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            assert_eq!(
                response.headers().get(OWNER_HEADER).unwrap(),
                "rust",
                "{path}"
            );
        }
    }

    #[tokio::test]
    async fn a_missing_sidecar_is_a_clear_503() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
