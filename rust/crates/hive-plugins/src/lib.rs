//! WASM verifier/tool plugins.
//!
//! A plugin is a directory with `plugin.json` and a `.wasm` module. The module
//! has no imports; it exports `memory`, `alloc(size) -> ptr`, and
//! `run(ptr, len) -> out_ptr`, where `out_ptr` points to a little-endian u32
//! length followed by the JSON output. Execution is bounded by deterministic
//! fuel and a linear-memory cap, so a bad verifier cannot hang or exhaust the
//! daemon.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use wasmtime::{
    Config, Engine, Instance, Memory, Module, Store, StoreLimits, StoreLimitsBuilder, TypedFunc,
};

const DEFAULT_FUEL: u64 = 10_000_000;
const DEFAULT_MEMORY_MB: usize = 64;
const DEFAULT_MAX_OUTPUT: usize = 1_048_576;
const MAX_INPUT: usize = 1_048_576;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PluginManifest {
    pub name: String,
    #[serde(default)]
    pub version: String,
    pub module: String,
    #[serde(default)]
    pub sha256: String,
    #[serde(default = "default_fuel")]
    pub fuel: u64,
    #[serde(default = "default_memory_mb")]
    pub memory_mb: usize,
    #[serde(default = "default_max_output")]
    pub max_output_bytes: usize,
}

fn default_fuel() -> u64 {
    DEFAULT_FUEL
}

fn default_memory_mb() -> usize {
    DEFAULT_MEMORY_MB
}

fn default_max_output() -> usize {
    DEFAULT_MAX_OUTPUT
}

pub struct PluginHost {
    directory: PathBuf,
    engine: Engine,
}

pub struct Plugin {
    pub manifest: PluginManifest,
    module: Module,
    engine: Engine,
}

#[derive(Debug)]
pub struct PluginRun {
    pub output: Vec<u8>,
    pub fuel_used: u64,
    pub elapsed_ms: u64,
}

impl PluginHost {
    pub fn new(directory: impl Into<PathBuf>) -> Result<Self, String> {
        let mut config = Config::new();
        config.consume_fuel(true);
        let engine = Engine::new(&config)
            .map_err(|error| format!("cannot create wasmtime engine: {error}"))?;
        Ok(Self {
            directory: directory.into(),
            engine,
        })
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn list(&self) -> Vec<serde_json::Value> {
        let Ok(entries) = std::fs::read_dir(&self.directory) else {
            return Vec::new();
        };
        let mut rows: Vec<serde_json::Value> = entries
            .flatten()
            .filter(|entry| entry.path().is_dir())
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                let manifest_path = entry.path().join("plugin.json");
                let text = std::fs::read_to_string(&manifest_path).ok()?;
                let manifest: PluginManifest = serde_json::from_str(&text).ok()?;
                let module = entry.path().join(&manifest.module);
                let valid = module.is_file() && self.hash_matches(&manifest, &module);
                Some(serde_json::json!({
                    "name": manifest.name,
                    "version": manifest.version,
                    "directory": name,
                    "module": manifest.module,
                    "fuel": manifest.fuel,
                    "memory_mb": manifest.memory_mb,
                    "valid": valid,
                }))
            })
            .collect();
        rows.sort_by(|left, right| {
            left.get("name")
                .and_then(|value| value.as_str())
                .cmp(&right.get("name").and_then(|value| value.as_str()))
        });
        rows
    }

    pub fn load(&self, name: &str) -> Result<Plugin, String> {
        validate_name(name)?;
        let directory = self.directory.join(name);
        let manifest_path = directory.join("plugin.json");
        let text = std::fs::read_to_string(&manifest_path)
            .map_err(|error| format!("cannot read {}: {error}", manifest_path.display()))?;
        let manifest: PluginManifest = serde_json::from_str(&text)
            .map_err(|error| format!("invalid plugin manifest: {error}"))?;
        if manifest.name != name {
            return Err(format!(
                "plugin manifest name {:?} does not match directory {name:?}",
                manifest.name
            ));
        }
        let module_path = directory.join(&manifest.module);
        let canonical_dir = directory
            .canonicalize()
            .map_err(|error| format!("cannot resolve plugin directory: {error}"))?;
        let canonical_module = module_path
            .canonicalize()
            .map_err(|error| format!("cannot resolve plugin module: {error}"))?;
        if !canonical_module.starts_with(&canonical_dir) {
            return Err("plugin module must stay inside its directory".into());
        }
        if !self.hash_matches(&manifest, &canonical_module) {
            return Err("plugin module sha256 does not match the manifest".into());
        }
        let module = Module::from_file(&self.engine, &canonical_module)
            .map_err(|error| format!("cannot compile plugin module: {error}"))?;
        Ok(Plugin {
            manifest,
            module,
            engine: self.engine.clone(),
        })
    }

    fn hash_matches(&self, manifest: &PluginManifest, module: &Path) -> bool {
        if manifest.sha256.is_empty() {
            return true;
        }
        let Ok(bytes) = std::fs::read(module) else {
            return false;
        };
        let digest = format!("{:x}", Sha256::digest(&bytes));
        digest.eq_ignore_ascii_case(&manifest.sha256)
    }
}

impl Plugin {
    pub fn run(&self, input: &[u8]) -> Result<PluginRun, String> {
        if input.len() > MAX_INPUT {
            return Err(format!("plugin input exceeds {MAX_INPUT} bytes"));
        }
        let started = Instant::now();
        let limits = StoreLimitsBuilder::new()
            .memory_size(self.manifest.memory_mb.saturating_mul(1024 * 1024))
            .instances(1)
            .tables(1)
            .memories(1)
            .trap_on_grow_failure(true)
            .build();
        let mut store = Store::new(&self.engine, limits);
        store.limiter(|limits: &mut StoreLimits| limits);
        store
            .set_fuel(self.manifest.fuel)
            .map_err(|error| format!("cannot set plugin fuel: {error}"))?;
        let instance = Instance::new(&mut store, &self.module, &[])
            .map_err(|error| format!("cannot instantiate plugin: {error}"))?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| "plugin does not export memory".to_owned())?;
        let alloc: TypedFunc<i32, i32> = instance
            .get_typed_func(&mut store, "alloc")
            .map_err(|error| format!("plugin does not export alloc: {error}"))?;
        let run: TypedFunc<(i32, i32), i32> = instance
            .get_typed_func(&mut store, "run")
            .map_err(|error| format!("plugin does not export run: {error}"))?;

        let input_len = i32::try_from(input.len()).map_err(|_| "input is too large")?;
        let input_ptr = alloc
            .call(&mut store, input_len)
            .map_err(|error| format!("plugin alloc failed: {error}"))?;
        write_memory(&memory, &mut store, input_ptr, input)?;
        let output_ptr = run
            .call(&mut store, (input_ptr, input_len))
            .map_err(|error| format!("plugin run failed: {error}"))?;
        let mut length = [0u8; 4];
        memory
            .read(&store, output_ptr as usize, &mut length)
            .map_err(|error| format!("cannot read plugin output length: {error}"))?;
        let output_len = u32::from_le_bytes(length) as usize;
        if output_len > self.manifest.max_output_bytes {
            return Err(format!(
                "plugin output exceeds {} bytes",
                self.manifest.max_output_bytes
            ));
        }
        let mut output = vec![0u8; output_len];
        memory
            .read(&store, output_ptr as usize + 4, &mut output)
            .map_err(|error| format!("cannot read plugin output: {error}"))?;
        let remaining = store.get_fuel().unwrap_or(0);
        Ok(PluginRun {
            output,
            fuel_used: self.manifest.fuel.saturating_sub(remaining),
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}

fn write_memory(
    memory: &Memory,
    store: &mut Store<StoreLimits>,
    ptr: i32,
    bytes: &[u8],
) -> Result<(), String> {
    if ptr < 0 {
        return Err("plugin alloc returned a negative pointer".into());
    }
    let size = memory.data_size(&*store);
    let end = (ptr as usize)
        .checked_add(bytes.len())
        .ok_or_else(|| "plugin input pointer overflow".to_owned())?;
    if end > size {
        return Err("plugin input does not fit in memory".into());
    }
    memory
        .write(store, ptr as usize, bytes)
        .map_err(|error| format!("cannot write plugin input: {error}"))
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name.len() > 128
        || !name.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
    {
        return Err("invalid plugin name".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_names_are_bounded() {
        assert!(validate_name("verifier-1").is_ok());
        assert!(validate_name("../escape").is_err());
        assert!(validate_name("a/b").is_err());
    }
}
