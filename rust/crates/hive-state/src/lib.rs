//! Serve records and process control.
//!
//! A serve is one supervisor process plus the engine children it started. The
//! record is advisory: `status` re-checks the PID and command line, and a
//! stale record is pruned rather than trusted. There is no daemon and no
//! database, just one JSON file per serve under `<state>/serve/`.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

pub type Record = Map<String, Value>;

pub fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// 1-64 letters, digits, `.`, `_` or `-`, starting with a letter or digit.
pub fn validate_name(name: &str) -> Result<String, String> {
    let text = name.trim();
    let mut chars = text.chars();
    let valid_start = chars
        .next()
        .map(|c| c.is_ascii_alphanumeric())
        .unwrap_or(false);
    let valid_rest = chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if text.len() > 64 || !valid_start || !valid_rest {
        return Err(format!(
            "invalid serve name {name:?}: use 1-64 letters, digits, '.', '_' or '-'"
        ));
    }
    Ok(text.to_owned())
}

pub fn record_path(serve_dir: &Path, name: &str) -> Result<PathBuf, String> {
    Ok(serve_dir.join(format!("{}.json", validate_name(name)?)))
}

pub fn log_path(serve_dir: &Path, name: &str) -> Result<PathBuf, String> {
    Ok(serve_dir.join(format!("{}.log", validate_name(name)?)))
}

fn coerce(data: Value, path: &Path) -> Result<Record, String> {
    let mut object = match data {
        Value::Object(object) => object,
        _ => {
            return Err(format!(
                "serve record {} is not a JSON object",
                path.display()
            ));
        }
    };
    let fallback = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or(&fallback);
    let name = validate_name(name)?;
    let pid = object
        .get("pid")
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("serve record {} has no numeric pid", path.display()))?;
    if pid <= 0 {
        return Err(format!("serve record {} has no valid pid", path.display()));
    }
    object.insert("name".into(), Value::String(name));
    object.insert("pid".into(), Value::from(pid));
    object.entry("version").or_insert(Value::from(1));
    object.entry("started_at").or_insert(Value::from(0.0));
    object
        .entry("tiers")
        .or_insert_with(|| Value::Array(Vec::new()));
    Ok(object)
}

pub fn read_record(serve_dir: &Path, name: &str) -> Result<Option<Record>, String> {
    let path = record_path(serve_dir, name)?;
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "cannot read serve record {}: {error}",
                path.display()
            ));
        }
    };
    let data: Value = serde_json::from_str(&text)
        .map_err(|error| format!("serve record {} is not valid JSON: {error}", path.display()))?;
    coerce(data, &path).map(Some)
}

/// Every valid record, oldest first. Corrupt files are skipped.
pub fn list_records(serve_dir: &Path) -> Vec<Record> {
    if fs::create_dir_all(serve_dir).is_err() {
        return Vec::new();
    }
    let mut files: Vec<PathBuf> = fs::read_dir(serve_dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
                .collect()
        })
        .unwrap_or_default();
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    let mut records = Vec::new();
    for path in files {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(data) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let Ok(mut record) = coerce(data, &path) else {
            continue;
        };
        record.insert(
            "record_path".into(),
            Value::String(path.to_string_lossy().into_owned()),
        );
        records.push(record);
    }
    records.sort_by(|a, b| {
        let left = a.get("started_at").and_then(Value::as_f64).unwrap_or(0.0);
        let right = b.get("started_at").and_then(Value::as_f64).unwrap_or(0.0);
        left.total_cmp(&right)
    });
    records
}

/// Write one record atomically. `updated_at` is refreshed.
pub fn write_record(serve_dir: &Path, record: &Record) -> Result<PathBuf, String> {
    let name = record
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "serve record needs a name".to_owned())?;
    let name = validate_name(name)?;
    let pid = record
        .get("pid")
        .and_then(Value::as_i64)
        .ok_or_else(|| "serve record needs a positive pid".to_owned())?;
    if pid <= 0 {
        return Err("serve record needs a positive pid".to_owned());
    }
    let mut output = record.clone();
    output.insert("updated_at".into(), Value::from(now()));
    let path = record_path(serve_dir, &name)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let text = hive_core::json::pretty(&Value::Object(output));
    let mut temp = tempfile::NamedTempFile::new_in(path.parent().unwrap_or(Path::new(".")))
        .map_err(|error| error.to_string())?;
    use std::io::Write;
    temp.write_all(text.as_bytes())
        .map_err(|error| error.to_string())?;
    temp.as_file()
        .sync_all()
        .map_err(|error| error.to_string())?;
    temp.persist(&path)
        .map_err(|error| error.error.to_string())?;
    Ok(path)
}

pub fn remove_record(serve_dir: &Path, name: &str) -> bool {
    match record_path(serve_dir, name) {
        Ok(path) => fs::remove_file(path).is_ok(),
        Err(_) => false,
    }
}

/// True when `pid` names a live, non-zombie process.
pub fn pid_alive(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if rc != 0 {
        let error = std::io::Error::last_os_error();
        return error.raw_os_error() == Some(libc::EPERM);
    }
    pid_state(pid) != "Z"
}

/// The Linux process state letter, or `""`.
fn pid_state(pid: i64) -> String {
    let Ok(raw) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return String::new();
    };
    let Some(end) = raw.rfind(')') else {
        return String::new();
    };
    raw[end + 2..]
        .chars()
        .next()
        .map(String::from)
        .unwrap_or_default()
}

pub fn pid_cmdline(pid: i64) -> String {
    let Ok(raw) = fs::read(format!("/proc/{pid}/cmdline")) else {
        return String::new();
    };
    String::from_utf8_lossy(
        &raw.iter()
            .map(|byte| if *byte == 0 { b' ' } else { *byte })
            .collect::<Vec<_>>(),
    )
    .trim()
    .to_owned()
}

/// Whether the live PID still looks like the serve that wrote the record.
pub fn pid_matches(record: &Record) -> bool {
    let pid = record.get("pid").and_then(Value::as_i64).unwrap_or(0);
    if !pid_alive(pid) {
        return false;
    }
    let stored = record
        .get("cmdline")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if stored.is_empty() {
        return true;
    }
    let current = pid_cmdline(pid);
    current.is_empty() || current == stored
}

/// Ask one serve supervisor to stop; force only after the timeout.
pub fn terminate(pid: i64, force: bool, timeout: f64) -> (bool, String) {
    if pid <= 0 || !pid_alive(pid) {
        return (true, "already stopped".into());
    }
    let rc = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    if rc != 0 {
        let error = std::io::Error::last_os_error();
        return match error.raw_os_error() {
            Some(libc::ESRCH) => (true, "already stopped".into()),
            Some(libc::EPERM) => (false, format!("permission denied for pid {pid}")),
            _ => (false, error.to_string()),
        };
    }
    let deadline = Instant::now() + Duration::from_secs_f64(timeout.max(0.5));
    while Instant::now() < deadline {
        if !pid_alive(pid) {
            return (true, "stopped".into());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    if !force {
        return (
            false,
            format!(
                "pid {pid} still running after {}s (use --force)",
                format_g(timeout)
            ),
        );
    }
    let pgid = unsafe { libc::getpgid(pid as libc::pid_t) };
    if pgid == pid as libc::pid_t {
        // a detached serve is its own session; take the engine children too
        unsafe { libc::killpg(pgid, libc::SIGKILL) };
    } else {
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if !pid_alive(pid) {
            return (true, "killed".into());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    (false, format!("pid {pid} survived SIGKILL"))
}

/// Python's `%g` for the timeout message: 15.0 -> "15", 0.5 -> "0.5".
pub fn format_g(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        format!("{value}")
    }
}

pub fn format_duration(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    let days = total / 86_400;
    let hours = (total % 86_400) / 3_600;
    let minutes = (total % 3_600) / 60;
    let secs = total % 60;
    if days > 0 {
        format!("{days}d {hours:02}:{minutes:02}:{secs:02}")
    } else {
        format!("{hours:02}:{minutes:02}:{secs:02}")
    }
}

/// The last `count` lines (all of them when `count <= 0`).
pub fn tail_lines(path: &Path, count: i64) -> std::io::Result<Vec<String>> {
    let text = fs::read_to_string(path)?;
    let text = text.replace("\r\n", "\n");
    let mut lines: Vec<String> = text.split('\n').map(str::to_owned).collect();
    if lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    if count <= 0 || lines.len() <= count as usize {
        return Ok(lines);
    }
    Ok(lines.split_off(lines.len() - count as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_reject_traversal() {
        assert!(validate_name("../x").is_err());
        assert!(validate_name("a/b").is_err());
        assert!(validate_name("-x").is_err());
        assert_eq!(validate_name("peer-2tier").unwrap(), "peer-2tier");
    }

    #[test]
    fn durations_match_the_python_format() {
        assert_eq!(format_duration(5.9), "00:00:05");
        assert_eq!(format_duration(86_400.0 + 3_661.0), "1d 01:01:01");
    }

    #[test]
    fn tail_keeps_the_last_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");
        fs::write(&path, "a\nb\nc\n").unwrap();
        assert_eq!(tail_lines(&path, 2).unwrap(), vec!["b", "c"]);
        assert_eq!(tail_lines(&path, 0).unwrap(), vec!["a", "b", "c"]);
    }
}
