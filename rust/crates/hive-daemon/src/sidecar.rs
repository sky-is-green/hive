//! Lifecycle for the Python Studio sidecar.
//!
//! The daemon owns the public port, but the Studio's routes still live in
//! Python. This module starts `python -m harness` on a private loopback port,
//! waits for `/health`, restarts it with capped exponential backoff, and takes
//! its process group down when the daemon stops.
//!
//! Interpreter selection is strict here: a set-but-unusable `HIVE_PYTHON` is a
//! startup error, not a reason to fall back to another Python.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, watch};
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep, timeout};

use hive_core::version;

const STARTUP_DEADLINE: Duration = Duration::from_secs(60);
const HEALTH_POLL: Duration = Duration::from_millis(250);
const MIN_BACKOFF: Duration = Duration::from_secs(1);
const MAX_BACKOFF: Duration = Duration::from_secs(30);
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

/// Everything needed to launch the sidecar. `port` 0 asks the OS for a free
/// port; the chosen port is reported by [`Sidecar::url`].
#[derive(Debug, Clone)]
pub struct SidecarSpec {
    pub python: PathBuf,
    pub repo: PathBuf,
    pub state_dir: PathBuf,
    pub models_dir: PathBuf,
    pub log_dir: PathBuf,
    pub port: u16,
}

/// A running (or restarting) sidecar supervisor.
pub struct Sidecar {
    base_url: String,
    pid: Arc<Mutex<Option<u32>>>,
    shutdown: watch::Sender<bool>,
    task: Mutex<Option<JoinHandle<()>>>,
}

impl Sidecar {
    /// Reserve a port, then start supervising the sidecar in the background.
    pub fn start(spec: SidecarSpec) -> Result<Self, String> {
        let port = pick_port(spec.port)?;
        let base_url = format!("http://127.0.0.1:{port}");
        let pid = Arc::new(Mutex::new(None));
        let (shutdown, receiver) = watch::channel(false);
        let supervisor = supervise(
            Arc::new(SidecarSpec { port, ..spec }),
            base_url.clone(),
            Arc::clone(&pid),
            receiver,
        );
        let task = tokio::spawn(supervisor);
        Ok(Self {
            base_url,
            pid,
            shutdown,
            task: Mutex::new(Some(task)),
        })
    }

    pub fn url(&self) -> &str {
        &self.base_url
    }

    /// The current sidecar PID, absent while it is between restarts.
    pub async fn pid(&self) -> Option<u32> {
        *self.pid.lock().await
    }

    /// Request a clean stop and wait for the supervisor. If the child ignores
    /// the grace period, the process group is killed.
    pub async fn shutdown(&self) {
        let _ = self.shutdown.send(true);
        let task = self.task.lock().await.take();
        if let Some(task) = task
            && timeout(SHUTDOWN_GRACE, task).await.is_err()
        {
            #[cfg(unix)]
            if let Some(pid) = *self.pid.lock().await {
                tracing::warn!(pid, "sidecar did not stop in time; killing process group");
                kill_group(pid, libc::SIGKILL);
            }
        }
    }
}

async fn supervise(
    spec: Arc<SidecarSpec>,
    base_url: String,
    pid: Arc<Mutex<Option<u32>>>,
    mut shutdown: watch::Receiver<bool>,
) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            tracing::error!("cannot build the sidecar health client: {error}");
            return;
        }
    };
    let mut backoff = MIN_BACKOFF;
    while !*shutdown.borrow() {
        let mut child = match spawn_child(&spec) {
            Ok(child) => child,
            Err(error) => {
                tracing::error!("cannot start the Python sidecar: {error}");
                if sleep_or_shutdown(backoff, &mut shutdown).await {
                    break;
                }
                backoff = next_backoff(backoff);
                continue;
            }
        };
        let child_pid = child.id();
        *pid.lock().await = child_pid;
        if let Some(child_pid) = child_pid {
            tracing::info!(pid = child_pid, url = %base_url, "starting Python sidecar");
        }

        let ready = handshake(&client, &base_url, &mut child, &mut shutdown).await;
        let healthy = match ready {
            Ok(sidecar_version) => {
                backoff = MIN_BACKOFF;
                if let Some(found) = sidecar_version.as_deref()
                    && found != version()
                {
                    tracing::warn!(
                        sidecar = found,
                        daemon = version(),
                        "sidecar version differs from the daemon"
                    );
                }
                tracing::info!(
                    pid = child_pid.unwrap_or_default(),
                    url = %base_url,
                    version = sidecar_version.as_deref().unwrap_or("unknown"),
                    "Python sidecar ready"
                );
                true
            }
            Err(error) => {
                tracing::warn!(
                    pid = child_pid.unwrap_or_default(),
                    "Python sidecar did not become ready: {error}"
                );
                false
            }
        };

        if healthy {
            tokio::select! {
                status = child.wait() => {
                    tracing::warn!(?status, "Python sidecar exited");
                }
                _ = wait_shutdown(&mut shutdown) => {
                    tracing::info!("stopping Python sidecar");
                    terminate(&mut child).await;
                    *pid.lock().await = None;
                    break;
                }
            }
        } else {
            terminate(&mut child).await;
        }
        *pid.lock().await = None;

        if sleep_or_shutdown(backoff, &mut shutdown).await {
            break;
        }
        backoff = next_backoff(backoff);
    }
    tracing::debug!("sidecar supervisor stopped");
}

fn spawn_child(spec: &SidecarSpec) -> Result<Child, String> {
    let port = spec.port.to_string();
    let mut command = Command::new(&spec.python);
    command
        .arg("-m")
        .arg("harness")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(&port)
        .arg("--no-console")
        .arg("--no-open")
        .arg("--log-dir")
        .arg(&spec.log_dir)
        .arg("--state-dir")
        .arg(&spec.state_dir)
        .arg("--models-dir")
        .arg(&spec.models_dir)
        .current_dir(&spec.repo)
        .stdin(Stdio::null())
        .env("PYTHONUNBUFFERED", "1")
        .kill_on_drop(false);
    #[cfg(unix)]
    {
        // One process group so descendants are stopped together.
        command.process_group(0);
    }
    let (stdout, stderr) = sidecar_log(&spec.log_dir);
    match (stdout, stderr) {
        (Some(stdout), Some(stderr)) => {
            command.stdout(stdout).stderr(stderr);
        }
        _ => {
            command.stdout(Stdio::null()).stderr(Stdio::inherit());
        }
    }
    command
        .spawn()
        .map_err(|error| format!("{}: {error}", spec.python.display()))
}

fn sidecar_log(log_dir: &Path) -> (Option<File>, Option<File>) {
    if std::fs::create_dir_all(log_dir).is_err() {
        return (None, None);
    }
    let path = log_dir.join("daemon-sidecar.log");
    let open = || {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok()
    };
    match (open(), open()) {
        (Some(stdout), Some(stderr)) => (Some(stdout), Some(stderr)),
        _ => (None, None),
    }
}

async fn handshake(
    client: &reqwest::Client,
    base_url: &str,
    child: &mut Child,
    shutdown: &mut watch::Receiver<bool>,
) -> Result<Option<String>, String> {
    let deadline = Instant::now() + STARTUP_DEADLINE;
    let health = format!("{base_url}/health");
    loop {
        if *shutdown.borrow() {
            return Err("daemon is shutting down".to_owned());
        }
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Err(format!("sidecar exited during startup ({status})"));
        }
        match client.get(&health).send().await {
            Ok(response) if response.status().is_success() => {
                // The body is JSON today; treating a malformed body as "not
                // ready yet" rides out the brief window before routes mount.
                if let Ok(data) = response.json::<Value>().await
                    && data.get("ok").and_then(Value::as_bool) != Some(false)
                {
                    if let Some(service) = data.get("service").and_then(Value::as_str)
                        && service != "hive-sidecar"
                    {
                        return Err(format!("unexpected sidecar service {service:?}"));
                    }
                    let version = data
                        .get("version")
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                    return Ok(version);
                }
            }
            Ok(response) => {
                tracing::debug!(status = %response.status(), "sidecar health not ready");
            }
            Err(_) => {}
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "timed out after {}s waiting for {health}",
                STARTUP_DEADLINE.as_secs()
            ));
        }
        if sleep_or_shutdown(HEALTH_POLL, shutdown).await {
            return Err("daemon is shutting down".to_owned());
        }
    }
}

async fn terminate(child: &mut Child) {
    if let Some(pid) = child.id() {
        #[cfg(unix)]
        kill_group(pid, libc::SIGTERM);
        #[cfg(not(unix))]
        {
            let _ = child.start_kill();
        }
    }
    if timeout(Duration::from_secs(5), child.wait()).await.is_err() {
        if let Some(pid) = child.id() {
            kill_group(pid, libc::SIGKILL);
        }
        let _ = child.kill().await;
        let _ = child.wait().await;
    }
}

#[cfg(unix)]
fn kill_group(pid: u32, signal: i32) {
    // The child leads its own process group (process_group(0)), so the group
    // id equals its pid. A negative pid targets every member.
    unsafe {
        libc::kill(-(pid as i32), signal);
    }
}

#[cfg(not(unix))]
fn kill_group(_pid: u32, _signal: i32) {}

async fn wait_shutdown(shutdown: &mut watch::Receiver<bool>) {
    loop {
        if *shutdown.borrow() {
            return;
        }
        if shutdown.changed().await.is_err() {
            return;
        }
    }
}

async fn sleep_or_shutdown(duration: Duration, shutdown: &mut watch::Receiver<bool>) -> bool {
    tokio::select! {
        _ = sleep(duration) => false,
        _ = wait_shutdown(shutdown) => true,
    }
}

fn next_backoff(current: Duration) -> Duration {
    (current * 2).min(MAX_BACKOFF)
}

fn pick_port(preferred: u16) -> Result<u16, String> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", preferred)).map_err(|error| {
        if preferred == 0 {
            format!("cannot reserve a sidecar port: {error}")
        } else {
            format!("sidecar port {preferred} is unavailable: {error}")
        }
    })?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_is_capped() {
        assert_eq!(next_backoff(MIN_BACKOFF), Duration::from_secs(2));
        assert_eq!(next_backoff(Duration::from_secs(20)), MAX_BACKOFF);
        assert_eq!(next_backoff(MAX_BACKOFF), MAX_BACKOFF);
    }

    #[test]
    fn an_ephemeral_port_is_reported() {
        let port = pick_port(0).unwrap();
        assert_ne!(port, 0);
    }
}
