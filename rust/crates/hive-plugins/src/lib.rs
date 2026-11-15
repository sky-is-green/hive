//! WASM verifier/tool plugins.
//!
//! A plugin is a directory with `plugin.json` and a `.wasm` module. The module
//! has no imports; it exports `memory`, `alloc(size) -> ptr`, and
//! `run(ptr, len) -> out_ptr`, where `out_ptr` points to a little-endian u32
//! length followed by the JSON output. Execution is bounded by deterministic
//! fuel and a linear-memory cap, so a bad verifier cannot hang or exhaust the
//! daemon.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use base64::Engine as _;
use ed25519_dalek::{Signature, VerifyingKey};
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

#[derive(Serialize)]
struct SigningPayload<'a> {
    name: &'a str,
    version: &'a str,
    module: &'a str,
    sha256: &'a str,
    fuel: u64,
    memory_mb: usize,
    max_output_bytes: usize,
    key_id: &'a str,
}

impl PluginManifest {
    pub fn signing_payload(&self, key_id: &str) -> Result<Vec<u8>, String> {
        serde_json::to_vec(&SigningPayload {
            name: &self.name,
            version: &self.version,
            module: &self.module,
            sha256: &self.sha256,
            fuel: self.fuel,
            memory_mb: self.memory_mb,
            max_output_bytes: self.max_output_bytes,
            key_id,
        })
        .map_err(|error| format!("cannot encode plugin signing payload: {error}"))
    }
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
    trusted_keys: BTreeMap<String, VerifyingKey>,
    require_signature: bool,
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
    pub fn new(directory: impl Into<PathBuf>, require_signature: bool) -> Result<Self, String> {
        let directory = directory.into();
        let trusted_keys = load_trusted_keys(&directory)?;
        Self::with_keys(directory, trusted_keys, require_signature)
    }

    pub fn new_unsigned(directory: impl Into<PathBuf>) -> Result<Self, String> {
        Self::with_keys(directory.into(), BTreeMap::new(), false)
    }

    pub fn with_keys(
        directory: impl Into<PathBuf>,
        trusted_keys: BTreeMap<String, VerifyingKey>,
        require_signature: bool,
    ) -> Result<Self, String> {
        let mut config = Config::new();
        config.consume_fuel(true);
        let engine = Engine::new(&config)
            .map_err(|error| format!("cannot create wasmtime engine: {error}"))?;
        Ok(Self {
            directory: directory.into(),
            engine,
            trusted_keys,
            require_signature,
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
                    "key_id": manifest.key_id,
                    "signed": manifest.signature.is_some(),
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
        self.verify_manifest(&manifest)?;
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

    fn verify_manifest(&self, manifest: &PluginManifest) -> Result<(), String> {
        if !self.require_signature && manifest.signature.is_none() {
            return Ok(());
        }
        let key_id = manifest
            .key_id
            .as_deref()
            .ok_or_else(|| "plugin manifest has no key_id".to_owned())?;
        let Some(key) = self.trusted_keys.get(key_id) else {
            return Err(format!("plugin signing key {key_id:?} is not trusted"));
        };
        let signature = manifest
            .signature
            .as_deref()
            .ok_or_else(|| "plugin manifest has no signature".to_owned())?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(signature)
            .map_err(|error| format!("plugin signature is not valid base64: {error}"))?;
        let signature = Signature::from_slice(&bytes)
            .map_err(|error| format!("plugin signature is malformed: {error}"))?;
        key.verify_strict(&manifest.signing_payload(key_id)?, &signature)
            .map_err(|_| "plugin signature does not verify".to_owned())
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

fn load_trusted_keys(directory: &Path) -> Result<BTreeMap<String, VerifyingKey>, String> {
    let path = std::env::var_os("HIVE_PLUGIN_TRUSTED_KEYS")
        .map(PathBuf::from)
        .unwrap_or_else(|| directory.join("trusted-keys.json"));
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(BTreeMap::new());
    };
    let data: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("invalid trusted plugin keys: {error}"))?;
    let mut keys = BTreeMap::new();
    let Some(entries) = data.get("keys").and_then(serde_json::Value::as_object) else {
        return Ok(keys);
    };
    for (id, value) in entries {
        let encoded = value
            .as_str()
            .ok_or_else(|| format!("trusted plugin key {id:?} is not a string"))?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|error| format!("trusted plugin key {id:?} is not base64: {error}"))?;
        let bytes: [u8; 32] = bytes
            .try_into()
            .map_err(|_| format!("trusted plugin key {id:?} is not 32 bytes"))?;
        let key = VerifyingKey::from_bytes(&bytes)
            .map_err(|error| format!("trusted plugin key {id:?} is invalid: {error}"))?;
        keys.insert(id.clone(), key);
    }
    Ok(keys)
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
    fn signing_payload_matches_the_documented_contract() {
        let manifest = PluginManifest {
            name: "demo".into(),
            version: "0.1.0".into(),
            module: "module.wasm".into(),
            sha256: "abc".into(),
            fuel: 1000,
            memory_mb: 1,
            max_output_bytes: 65536,
            key_id: None,
            signature: None,
        };
        let payload = manifest.signing_payload("release-test").unwrap();
        assert_eq!(
            String::from_utf8(payload).unwrap(),
            r#"{"name":"demo","version":"0.1.0","module":"module.wasm","sha256":"abc","fuel":1000,"memory_mb":1,"max_output_bytes":65536,"key_id":"release-test"}"#
        );
    }

    #[test]
    fn plugin_names_are_bounded() {
        assert!(validate_name("verifier-1").is_ok());
        assert!(validate_name("../escape").is_err());
        assert!(validate_name("a/b").is_err());
    }
}
