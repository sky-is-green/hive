//! `hive models list`: one row per `.gguf`, newest file first.

use std::path::PathBuf;
use std::time::SystemTime;

use chrono::{DateTime, Local};
use serde_json::{Map, Number, Value};
use walkdir::WalkDir;

use crate::gguf::read_metadata;

pub struct Library {
    pub dir: PathBuf,
}

impl Library {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The rows the Python CLI prints for `models list --json`.
    pub fn rows(&self) -> Vec<Value> {
        let mut paths: Vec<PathBuf> = WalkDir::new(&self.dir)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .map(|entry| entry.into_path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "gguf"))
            .collect();
        paths.sort();

        let mut entries = Vec::new();
        for path in paths {
            let Ok(stat) = std::fs::metadata(&path) else {
                continue;
            };
            let relative = path
                .strip_prefix(&self.dir)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| {
                    path.file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_default()
                });
            let metadata = read_metadata(&path);
            let size_gb = round2(stat.len() as f64 / (1024.0 * 1024.0 * 1024.0));
            let modified = format_modified(stat.modified().ok());
            let mut row = Map::new();
            row.insert(
                "name".into(),
                Value::String(
                    path.file_stem()
                        .map(|stem| stem.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                ),
            );
            row.insert("file".into(), Value::String(relative));
            row.insert("size_gb".into(), number(size_gb));
            row.insert("sizeGb".into(), number(size_gb));
            row.insert("modified".into(), Value::String(modified.clone()));
            row.insert("lastModified".into(), Value::String(modified));
            if let Some(Value::String(arch)) = metadata
                .get("architecture")
                .or_else(|| metadata.get("general.architecture"))
            {
                row.insert("architecture".into(), Value::String(arch.clone()));
            }
            if let Some(Value::String(quant)) = metadata.get("quantization") {
                row.insert("quantization".into(), Value::String(quant.clone()));
            }
            if let Some(value) = metadata.get("context_length")
                && let Some(ctx) = value.as_i64()
            {
                row.insert("context_length".into(), Value::from(ctx));
                row.insert("contextLength".into(), Value::from(ctx));
            }
            if !metadata.is_empty() {
                row.insert("gguf_metadata".into(), Value::Object(metadata.clone()));
                row.insert("ggufMetadata".into(), Value::Object(metadata));
            }
            entries.push(Value::Object(row));
        }
        // Python sorts by the formatted `modified` string, newest first, stable.
        entries.sort_by(|a, b| {
            let left = a.get("modified").and_then(Value::as_str).unwrap_or("");
            let right = b.get("modified").and_then(Value::as_str).unwrap_or("");
            right.cmp(left)
        });
        entries
    }

    pub fn count(&self) -> usize {
        WalkDir::new(&self.dir)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "gguf"))
            .count()
    }
}

fn format_modified(modified: Option<SystemTime>) -> String {
    let Some(modified) = modified else {
        return String::new();
    };
    let datetime: DateTime<Local> = modified.into();
    datetime.format("%Y-%m-%d %H:%M").to_string()
}

fn number(value: f64) -> Value {
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_library_has_no_rows() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Library::new(dir.path()).rows().is_empty());
    }

    #[test]
    fn a_non_gguf_file_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("readme.txt"), b"x").unwrap();
        assert!(Library::new(dir.path()).rows().is_empty());
    }
}
