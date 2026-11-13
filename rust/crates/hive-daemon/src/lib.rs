//! The Phase 2 daemon: native health/config/version routes, with a transparent
//! proxy fallback to the Python sidecar.
//!
//! This is deliberately a small first slice. It proves the ownership split:
//! Rust owns the public port and the routes in `GROUP_ONE`; every other path is
//! streamed to the sidecar unchanged. Route groups move into Rust one at a
//! time.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::header::{
    CONNECTION, HOST, HeaderMap, HeaderName, HeaderValue, TRANSFER_ENCODING, UPGRADE,
};
use axum::http::{Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use hive_core::backends::Manifest;
use hive_core::paths::Paths;
use hive_core::{platform_display, python, version};
use serde_json::{Map, Value};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub mod sidecar;
use sidecar::Sidecar;

pub const OWNER_HEADER: &str = "x-hive-owner";

#[derive(Clone)]
pub struct DaemonConfig {
    pub host: String,
    pub port: u16,
    /// An already-running sidecar. When set, the daemon never spawns one.
    pub sidecar_url: Option<String>,
    /// Spawn `python -m harness` on `sidecar_port` when no URL is configured.
    pub spawn_sidecar: bool,
    pub sidecar_port: u16,
    pub token: Option<String>,
    pub paths: Paths,
}

impl DaemonConfig {
    pub fn new(host: impl Into<String>, port: u16, sidecar_url: Option<String>) -> Self {
        let token = std::env::var("HARNESS_TOKEN")
            .ok()
            .map(|token| token.trim().to_owned())
            .filter(|token| !token.is_empty());
        Self {
            host: host.into(),
            port,
            sidecar_url: sidecar_url.filter(|url| !url.trim().is_empty()),
            spawn_sidecar: false,
            sidecar_port: 8766,
            token,
            paths: Paths::discover(None, None),
        }
    }
}

pub struct AppState {
    pub config: DaemonConfig,
    client: reqwest::Client,
    managed: Option<Arc<Sidecar>>,
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
        Self {
            config,
            client,
            managed,
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
        .fallback(proxy)
        .layer(middleware::from_fn_with_state(state.clone(), token_guard))
        .layer(CorsLayer::permissive())
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
        if supplied != required {
            return owned(
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"detail": "invalid or missing token"})).into_response(),
            );
        }
    }
    next.run(request).await
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

async fn proxy(State(state): State<Arc<AppState>>, request: Request) -> Response {
    let Some(base) = &state.config.sidecar_url else {
        return owned(
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"detail": "no Python sidecar configured"})).into_response(),
        );
    };
    let path = request
        .uri()
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or("/");
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
