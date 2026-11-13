//! Python discovery. The Rust CLI no longer needs Python to run, but the
//! Studio and experiments still do, so `version`, `config` and `doctor`
//! report the interpreter they will use.

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::paths::{Paths, expand};

/// `HIVE_PYTHON`, then the active virtualenv, then the checkout venv, then PATH.
///
/// Reporting commands use this forgiving form. Processes that cannot run
/// without Python must use [`resolve`]: a set-but-unusable `HIVE_PYTHON` is an
/// operator error, not an invitation to silently pick a different interpreter.
pub fn find(paths: &Paths) -> PathBuf {
    resolve(paths).unwrap_or_else(|_| PathBuf::from("python3"))
}

/// Resolve an interpreter or explain why the configured one cannot be used.
///
/// Unlike [`find`], a present `HIVE_PYTHON` is authoritative: if it is missing,
/// not a file, or not executable, that is an error. The same applies to an
/// explicitly set, unusable `VIRTUAL_ENV`.
pub fn resolve(paths: &Paths) -> Result<PathBuf, String> {
    if let Some(raw) = env::var_os("HIVE_PYTHON") {
        let text = raw.to_string_lossy();
        if text.trim().is_empty() {
            return Err("HIVE_PYTHON is set but empty".to_owned());
        }
        let candidate = expand(Path::new(&raw));
        if usable(&candidate) {
            return Ok(candidate);
        }
        return Err(format!(
            "HIVE_PYTHON points at {}, which is not an executable file",
            candidate.display()
        ));
    }
    if let Some(raw) = env::var_os("VIRTUAL_ENV") {
        let candidate = PathBuf::from(raw).join("bin/python");
        if usable(&candidate) {
            return Ok(candidate);
        }
        return Err(format!(
            "VIRTUAL_ENV is set but {} is not an executable file; set HIVE_PYTHON or unset VIRTUAL_ENV",
            candidate.display()
        ));
    }
    for root in [Some(paths.repo.as_path()), Some(paths.home.as_path())]
        .into_iter()
        .flatten()
    {
        let candidate = root.join(".venv/bin/python");
        if usable(&candidate) {
            return Ok(candidate);
        }
    }
    for name in ["python3", "python"] {
        if let Some(found) = which(name) {
            return Ok(found);
        }
    }
    Err("no Python interpreter found; set HIVE_PYTHON".to_owned())
}

fn usable(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .map(|meta| meta.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// `platform.python_version()`, or a best-effort fallback.
pub fn version(executable: &Path) -> String {
    if let Ok(raw) = env::var("HIVE_PYTHON_VERSION") {
        let text = raw.trim();
        if !text.is_empty() {
            return text.to_owned();
        }
    }
    if let Some(found) = run_version(executable) {
        return found;
    }
    // A venv's pyvenv.cfg carries at least the major.minor pair.
    if let Some(parent) = executable.parent().and_then(Path::parent)
        && let Ok(text) = std::fs::read_to_string(parent.join("pyvenv.cfg"))
    {
        for line in text.lines() {
            if let Some(value) = line.strip_prefix("version_info") {
                let value = value.trim_start_matches([' ', '=']).trim();
                if !value.is_empty() {
                    return value.to_owned();
                }
            }
        }
    }
    "unknown".to_owned()
}

fn run_version(executable: &Path) -> Option<String> {
    let mut child = Command::new(executable)
        .arg("-c")
        .arg("import platform; print(platform.python_version())")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break,
            Ok(Some(_)) | Err(_) => return None,
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    let output = child.wait_with_output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!text.is_empty()).then_some(text)
}

/// Whether the interpreter is new enough for the Studio.
pub fn meets_minimum(version: &str, minimum: (u32, u32)) -> bool {
    let mut parts = version.split('.');
    let major = parts
        .next()
        .and_then(|p| p.parse::<u32>().ok())
        .unwrap_or(0);
    let minor = parts
        .next()
        .and_then(|p| p.parse::<u32>().ok())
        .unwrap_or(0);
    (major, minor) >= minimum
}

pub fn which(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for directory in env::split_paths(&path) {
        let candidate = directory.join(name);
        if is_executable(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(path) {
        Ok(meta) => meta.is_file() && meta.permissions().mode() & 0o111 != 0,
        Err(_) => false,
    }
}
