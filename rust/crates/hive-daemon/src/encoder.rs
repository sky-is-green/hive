//! Managed native encoder service.
//!
//! When `HARNESS_EMBEDDING_BACKEND=rust`, the daemon starts `hive-encoder
//! serve` on a private loopback port and points the Python sidecar at it. The
//! sidecar then uses `sieve.rust_encoder.RustEncoderDrone` instead of importing
//! torch/sentence-transformers.

use std::fs::OpenOptions;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use hive_core::python;
use tokio::process::{Child, Command};

pub struct EncoderService {
    child: Child,
    pub port: u16,
}

impl EncoderService {
    pub async fn start(log_dir: &std::path::Path, port: u16) -> Result<Self, String> {
        let binary = binary()?;
        if let Some(parent) = log_path(log_dir).parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path(log_dir))
            .map_err(|error| format!("cannot open the encoder log: {error}"))?;
        let stderr = log.try_clone().map_err(|error| error.to_string())?;
        let mut command = Command::new(&binary);
        command
            .arg("serve")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(stderr));
        #[cfg(unix)]
        command.process_group(0);
        let mut child = command
            .spawn()
            .map_err(|error| format!("cannot start {}: {error}", binary.display()))?;
        let deadline = Instant::now() + Duration::from_secs(30);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|error| error.to_string())?;
        loop {
            if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
                return Err(format!(
                    "hive-encoder exited during startup ({status}); see {}",
                    log_path(log_dir).display()
                ));
            }
            let healthy = client
                .get(format!("http://127.0.0.1:{port}/health"))
                .send()
                .await
                .map(|response| response.status().is_success())
                .unwrap_or(false);
            if healthy {
                tracing::info!(port, binary = %binary.display(), "native encoder ready");
                return Ok(Self { child, port });
            }
            if Instant::now() >= deadline {
                let _ = child.kill().await;
                return Err(format!(
                    "hive-encoder did not become ready on :{port}; see {}",
                    log_path(log_dir).display()
                ));
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    pub async fn stop(&mut self) {
        #[cfg(unix)]
        if let Some(pid) = self.child.id() {
            unsafe {
                libc::kill(-(pid as i32), libc::SIGTERM);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = self.child.start_kill();
        }
        if tokio::time::timeout(Duration::from_secs(5), self.child.wait())
            .await
            .is_err()
        {
            #[cfg(unix)]
            if let Some(pid) = self.child.id() {
                unsafe {
                    libc::kill(-(pid as i32), libc::SIGKILL);
                }
            }
            let _ = self.child.kill().await;
            let _ = self.child.wait().await;
        }
    }
}

fn log_path(log_dir: &std::path::Path) -> PathBuf {
    log_dir.join("encoder.log")
}

fn binary() -> Result<PathBuf, String> {
    if let Some(raw) = std::env::var_os("HIVE_ENCODER_BIN") {
        let path = PathBuf::from(raw);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "HIVE_ENCODER_BIN points at {}, which is not a file",
            path.display()
        ));
    }
    if let Ok(executable) = std::env::current_exe() {
        let sibling = executable.with_file_name("hive-encoder");
        if sibling.is_file() {
            return Ok(sibling);
        }
    }
    python::which("hive-encoder")
        .ok_or_else(|| "hive-encoder is not installed next to hive-daemon or on PATH".to_owned())
}
