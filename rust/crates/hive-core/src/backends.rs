//! The backend manifest: which `llama-server` builds are declared, where they
//! live, and whether each one is installed. Mirrors `harness/backends.py`.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::error::{HiveError, Result};

/// Best-first accelerator preference, matching `_PREFERENCE`.
const PREFERENCE: &[&str] = &["rocm", "cuda", "sycl", "vulkan", "scion", "cpu"];

#[derive(Debug, Deserialize)]
struct ManifestDoc {
    backends: IndexMap<String, Map<String, Value>>,
}

#[derive(Debug, Clone)]
pub struct Backend {
    pub name: String,
    pub raw: Map<String, Value>,
    pub binary: PathBuf,
    pub installed: bool,
}

impl Backend {
    pub fn label(&self) -> String {
        self.raw
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    }

    pub fn accelerators(&self) -> Vec<String> {
        self.raw
            .get("accelerators")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// One `resolve()` row, including the derived `binary` and `installed`.
    pub fn to_json(&self) -> Value {
        let mut object = self.raw.clone();
        object.insert("name".into(), Value::String(self.name.clone()));
        object.insert(
            "binary".into(),
            Value::String(self.binary.to_string_lossy().into_owned()),
        );
        object.insert("installed".into(), Value::Bool(self.installed));
        Value::Object(object)
    }
}

#[derive(Debug, Clone)]
pub struct Manifest {
    pub entries: Vec<Backend>,
}

impl Manifest {
    /// Read the checkout manifest, or the packaged copy for a wheel.
    pub fn load(path: Option<&Path>, home: &Path) -> Result<Self> {
        let path = path.ok_or_else(|| {
            HiveError::not_found("backend manifest not found (tools/backends/manifest.json)")
        })?;
        let text = std::fs::read_to_string(path)?;
        let doc: ManifestDoc = serde_json::from_str(&text)?;
        if doc.backends.is_empty() {
            return Err(HiveError::invalid(format!(
                "backend manifest {} has no 'backends' object",
                path.display()
            )));
        }
        let entries = doc
            .backends
            .into_iter()
            .map(|(name, raw)| {
                let binary_dir = raw
                    .get("binary_dir")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("tools/backends/{name}"));
                let binary = home.join(binary_dir).join("llama-server");
                let installed = binary.is_file();
                Backend {
                    name,
                    raw,
                    binary,
                    installed,
                }
            })
            .collect();
        Ok(Self { entries })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|entry| entry.name.as_str())
    }

    pub fn resolve(&self, name: &str) -> Result<&Backend> {
        let key = name.trim().to_lowercase();
        self.entries
            .iter()
            .find(|entry| entry.name == key)
            .ok_or_else(|| {
                HiveError::invalid(format!(
                    "unknown backend {name:?}; known: {}",
                    self.names().collect::<Vec<_>>().join(", ")
                ))
            })
    }

    pub fn installed(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|entry| entry.installed)
            .map(|entry| entry.name.as_str())
            .collect()
    }

    /// The installed backend for an accelerator, best first. `None` means the
    /// caller falls back to the default `llama-server`.
    pub fn for_accelerator(&self, kind: &str) -> Option<&str> {
        let kind = kind.trim().to_lowercase();
        let mut candidates: Vec<&Backend> = self
            .entries
            .iter()
            .filter(|entry| entry.installed && entry.accelerators().iter().any(|a| a == &kind))
            .collect();
        candidates.sort_by_key(|entry| {
            PREFERENCE
                .iter()
                .position(|name| *name == entry.name)
                .unwrap_or(PREFERENCE.len())
        });
        candidates.first().map(|entry| entry.name.as_str())
    }

    pub fn rows(&self) -> Vec<Value> {
        self.entries.iter().map(Backend::to_json).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        let raw: Map<String, Value> = serde_json::from_str(
            r#"{"binary_dir":"tools/backends/vulkan","accelerators":["amd"],"label":"Vulkan"}"#,
        )
        .unwrap();
        Manifest {
            entries: vec![Backend {
                name: "vulkan".into(),
                raw,
                binary: PathBuf::from("/nonexistent/llama-server"),
                installed: false,
            }],
        }
    }

    #[test]
    fn resolves_case_insensitively_and_reports_unknown_names() {
        let manifest = manifest();
        assert_eq!(manifest.resolve("VULKAN").unwrap().label(), "Vulkan");
        assert!(
            manifest
                .resolve("nope")
                .unwrap_err()
                .to_string()
                .contains("known")
        );
    }

    #[test]
    fn an_uninstalled_backend_is_not_selected() {
        assert_eq!(manifest().for_accelerator("amd"), None);
    }
}
