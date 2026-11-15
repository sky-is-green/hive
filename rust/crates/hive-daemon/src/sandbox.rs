//! Native Linux sandbox API.
//!
//! The daemon supervises `hive-sandbox`, a single-threaded helper that applies
//! Landlock filesystem rules and a seccomp deny-list before exec. Keeping the
//! isolation in a separate process avoids unsafe post-fork work in the
//! multithreaded daemon.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use hive_core::paths::Paths;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::runtime::ApiError;

const OUTPUT_CAP: usize = 10_000;

#[derive(Debug, Clone)]
struct Policy {
    mode: String,
    memory_mb: u64,
    pids: u64,
    nofile: u64,
    timeout_s: u64,
    allow_network: bool,
    raw: Value,
}

impl Policy {
    fn load(state_dir: &Path) -> Self {
        let path = state_dir.join("settings/sandbox.json");
        let raw = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .unwrap_or_else(|| json!({}));
        let mode = raw
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("auto")
            .to_owned();
        let memory_mb = raw
            .get("memory_mb")
            .and_then(Value::as_u64)
            .or_else(|| {
                raw.get("memory")
                    .and_then(Value::as_str)
                    .and_then(|memory| memory.trim_end_matches('m').parse::<u64>().ok())
            })
            .unwrap_or(512)
            .clamp(16, 8192);
        let timeout_s = raw
            .get("timeout_s")
            .and_then(Value::as_u64)
            .unwrap_or(60)
            .clamp(1, 120);
        let pids = raw
            .get("pids_limit")
            .and_then(Value::as_u64)
            .unwrap_or(256)
            .clamp(1, 4096);
        let nofile = raw
            .get("nofile")
            .and_then(Value::as_u64)
            .unwrap_or(256)
            .clamp(16, 4096);
        let allow_network = raw
            .get("allow_network")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Self {
            mode,
            memory_mb,
            pids,
            nofile,
            timeout_s,
            allow_network,
            raw,
        }
    }
}

pub struct SandboxConfig {
    binary: Option<PathBuf>,
    workspace_root: PathBuf,
    state_dir: PathBuf,
}

impl SandboxConfig {
    pub fn discover(paths: &Paths) -> Self {
        let binary = std::env::var_os("HIVE_SANDBOX_BIN")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .or_else(|| {
                std::env::current_exe()
                    .ok()
                    .and_then(|path| path.parent().map(|parent| parent.join("hive-sandbox")))
                    .filter(|path| path.is_file())
            })
            .or_else(|| {
                [
                    paths.repo.join("rust/target/release/hive-sandbox"),
                    paths.repo.join("rust/target/debug/hive-sandbox"),
                ]
                .into_iter()
                .find(|path| path.is_file())
            });
        let workspace_root = std::env::var_os("HIVE_SANDBOX_WORKSPACE_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/srv/hive-ai/workspaces"));
        Self {
            binary,
            workspace_root,
            state_dir: paths.state_dir.clone(),
        }
    }

    pub fn status(&self) -> Value {
        let policy = self.policy();
        let lsm = std::fs::read_to_string("/sys/kernel/security/lsm").unwrap_or_default();
        let landlock = lsm.split(',').any(|name| name.trim() == "landlock");
        let seccomp = Path::new("/proc/sys/kernel/seccomp/actions_avail").is_file();
        let available = self.binary.is_some() && landlock && seccomp;
        json!({
            "installed": available,
            "daemon": available,
            "binary": self.binary.as_ref().map(|path| path.to_string_lossy().into_owned()),
            "detail": if available { "native Landlock/seccomp sandbox" } else { "native sandbox unavailable" },
            "host": "",
            "rootless": false,
            "backend": "landlock",
            "available": available,
            "landlock": landlock,
            "seccomp": seccomp,
            "workspace_root": self.workspace_root.to_string_lossy(),
            "workspace_root_exists": self.workspace_root.is_dir(),
            "policy": policy.raw,
            "mode": policy.mode,
            "pinned": true,
            "pin_error": "",
            "fallback_requires_approval": false,
            "fallback": if available { "native sandbox" } else { "native sandbox unavailable" },
            "docker_installed": false,
            "docker_daemon": false,
        })
    }

    pub fn native_selected(&self) -> bool {
        !self.policy().mode.eq_ignore_ascii_case("docker")
    }

    pub fn available(&self) -> bool {
        let lsm = std::fs::read_to_string("/sys/kernel/security/lsm").unwrap_or_default();
        let landlock = lsm.split(',').any(|name| name.trim() == "landlock");
        let seccomp = Path::new("/proc/sys/kernel/seccomp/actions_avail").is_file();
        self.binary.is_some() && landlock && seccomp
    }

    pub fn strict_native(&self) -> bool {
        self.policy().mode.eq_ignore_ascii_case("landlock")
    }

    fn policy(&self) -> Policy {
        Policy::load(&self.state_dir)
    }

    pub async fn exec(&self, body: &Value) -> Result<Value, ApiError> {
        let Some(binary) = &self.binary else {
            return Err(ApiError {
                status: 503,
                detail: "hive-sandbox is not built or installed".into(),
            });
        };
        let command = body
            .get("command")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|command| !command.is_empty())
            .ok_or_else(|| ApiError {
                status: 422,
                detail: "command is required".into(),
            })?;
        if command.chars().count() > 16_000 {
            return Err(ApiError {
                status: 422,
                detail: "command is too long".into(),
            });
        }
        let policy = self.policy();
        let timeout = body
            .get("timeout")
            .and_then(Value::as_u64)
            .unwrap_or(policy.timeout_s)
            .clamp(1, 120);
        let memory_mb = body
            .get("memory_mb")
            .and_then(Value::as_u64)
            .unwrap_or(policy.memory_mb)
            .clamp(16, 8192);
        let cpu_seconds = body
            .get("cpu_seconds")
            .and_then(Value::as_u64)
            .unwrap_or(timeout)
            .clamp(1, 3600);
        let pids = body
            .get("pids")
            .and_then(Value::as_u64)
            .unwrap_or(policy.pids)
            .clamp(1, 4096);
        let nofile = body
            .get("nofile")
            .and_then(Value::as_u64)
            .unwrap_or(policy.nofile)
            .clamp(16, 4096);
        let allow_network = body
            .get("allow_network")
            .and_then(Value::as_bool)
            .unwrap_or(policy.allow_network);

        let workspace = match body.get("workspace").and_then(Value::as_str) {
            Some(raw) if !raw.trim().is_empty() => {
                Some(self.workspace(raw.trim()).map_err(|detail| ApiError {
                    status: 422,
                    detail,
                })?)
            }
            _ => None,
        };

        let scratch = self.state_dir.join("sandbox").join(format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|value| value.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&scratch).map_err(|error| ApiError {
            status: 500,
            detail: format!("cannot create sandbox scratch directory: {error}"),
        })?;

        let mut cmd = Command::new(binary);
        cmd.arg("--temp").arg(&scratch);
        if let Some(workspace) = &workspace {
            cmd.arg("--workspace").arg(workspace);
        }
        cmd.arg("--memory-mb")
            .arg(memory_mb.to_string())
            .arg("--cpu-seconds")
            .arg(cpu_seconds.to_string())
            .arg("--pids")
            .arg(pids.to_string())
            .arg("--nofile")
            .arg(nofile.to_string());
        if allow_network {
            cmd.arg("--allow-network");
        }
        cmd.arg("--")
            .arg("/bin/sh")
            .arg("-c")
            .arg(command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .kill_on_drop(true);
        let started = Instant::now();
        let child = cmd.spawn().map_err(|error| ApiError {
            status: 503,
            detail: format!("cannot start hive-sandbox: {error}"),
        })?;
        let pid = child.id();
        let output =
            tokio::time::timeout(Duration::from_secs(timeout), child.wait_with_output()).await;
        let _ = std::fs::remove_dir_all(&scratch);

        let (timed_out, exit_code, stdout, stderr) = match output {
            Ok(Ok(output)) => (
                false,
                output.status.code().unwrap_or(1),
                String::from_utf8_lossy(&output.stdout)
                    .chars()
                    .take(OUTPUT_CAP)
                    .collect::<String>(),
                String::from_utf8_lossy(&output.stderr)
                    .chars()
                    .take(OUTPUT_CAP)
                    .collect::<String>(),
            ),
            Ok(Err(error)) => {
                return Err(ApiError {
                    status: 500,
                    detail: format!("sandbox process failed: {error}"),
                });
            }
            Err(_) => {
                if let Some(pid) = pid {
                    // SAFETY: a negative pid addresses the sandbox's process group.
                    unsafe {
                        libc::kill(-(pid as i32), libc::SIGKILL);
                    }
                }
                (true, 124, String::new(), String::new())
            }
        };
        if exit_code == 78 {
            return Err(ApiError {
                status: 503,
                detail: format!("native sandbox is unavailable: {}", stderr.trim()),
            });
        }
        Ok(json!({
            "backend": "landlock",
            "docker_available": false,
            "image": "",
            "sandbox_policy": {
                "mode": "landlock",
                "memory_mb": memory_mb,
                "cpu_seconds": cpu_seconds,
                "pids": pids,
                "nofile": nofile,
                "allow_network": allow_network,
            },
            "workspace": workspace.map(|path| path.to_string_lossy().into_owned()).unwrap_or_default(),
            "container": "",
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "duration_ms": started.elapsed().as_millis() as u64,
        }))
    }

    fn workspace(&self, raw: &str) -> Result<PathBuf, String> {
        let root = std::fs::canonicalize(&self.workspace_root)
            .map_err(|error| format!("sandbox workspace root is unavailable: {error}"))?;
        let path = std::fs::canonicalize(raw)
            .map_err(|error| format!("workspace is not usable: {error}"))?;
        if !path.is_dir() {
            return Err("workspace is not a directory".into());
        }
        if path == root || !path.starts_with(&root) {
            return Err(format!(
                "workspace must be a subdirectory of {}",
                root.display()
            ));
        }
        Ok(path)
    }
}
