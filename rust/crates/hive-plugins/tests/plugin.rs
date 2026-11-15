use std::path::Path;

use std::collections::BTreeMap;

use base64::Engine as _;
use ed25519_dalek::Signer;
use hive_plugins::{PluginHost, PluginManifest};
use rand_core::OsRng;
use sha2::{Digest, Sha256};

fn install(root: &Path, name: &str, wat: &str, mutate: impl FnOnce(&mut serde_json::Value)) {
    let directory = root.join(name);
    std::fs::create_dir_all(&directory).unwrap();
    let wasm = wat::parse_str(wat).unwrap();
    std::fs::write(directory.join("module.wasm"), &wasm).unwrap();
    let digest = format!("{:x}", Sha256::digest(&wasm));
    let mut manifest = serde_json::json!({
        "name": name,
        "version": "0.1.0",
        "module": "module.wasm",
        "sha256": digest,
        "fuel": 1_000_000,
        "memory_mb": 2,
        "max_output_bytes": 65536,
    });
    mutate(&mut manifest);
    std::fs::write(
        directory.join("plugin.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

const ECHO: &str = r#"
(module
  (memory (export "memory") 2)
  (global $heap (mut i32) (i32.const 1024))
  (func (export "alloc") (param $size i32) (result i32)
    (local $ptr i32)
    (local.set $ptr (global.get $heap))
    (global.set $heap (i32.add (global.get $heap) (local.get $size)))
    (local.get $ptr))
  (func (export "run") (param $ptr i32) (param $len i32) (result i32)
    (local $out i32)
    (local.set $out (i32.const 8192))
    (i32.store (local.get $out) (local.get $len))
    (memory.copy (i32.add (local.get $out) (i32.const 4)) (local.get $ptr) (local.get $len))
    (local.get $out)))
"#;

const SPIN: &str = r#"
(module
  (memory (export "memory") 1)
  (func (export "alloc") (param i32) (result i32) (i32.const 1024))
  (func (export "run") (param i32 i32) (result i32)
    (loop $forever (br $forever))
    (i32.const 0)))
"#;

const GROW: &str = r#"
(module
  (memory (export "memory") 1)
  (func (export "alloc") (param i32) (result i32) (i32.const 1024))
  (func (export "run") (param i32 i32) (result i32)
    (drop (memory.grow (i32.const 100)))
    (i32.store (i32.const 4096) (i32.const 0))
    (i32.const 4096)))
"#;

#[test]
fn a_plugin_round_trips_json_under_limits() {
    let root = tempfile::tempdir().unwrap();
    install(root.path(), "echo", ECHO, |_| {});
    let host = PluginHost::new_unsigned(root.path()).unwrap();
    let plugin = host.load("echo").unwrap();
    let input = br#"{"verdict":"pass"}"#;
    let result = plugin.run(input).unwrap();
    assert_eq!(result.output, input);
    assert!(result.fuel_used > 0);
}

#[test]
fn an_infinite_loop_runs_out_of_fuel() {
    let root = tempfile::tempdir().unwrap();
    install(root.path(), "spin", SPIN, |manifest| {
        manifest["fuel"] = serde_json::json!(1000);
    });
    let host = PluginHost::new_unsigned(root.path()).unwrap();
    let plugin = host.load("spin").unwrap();
    let error = match plugin.run(b"{}") {
        Ok(_) => panic!("expected plugin error"),
        Err(error) => error,
    };
    assert!(
        error.contains("fuel") || error.contains("run failed"),
        "{error}"
    );
}

#[test]
fn memory_growth_is_bounded() {
    let root = tempfile::tempdir().unwrap();
    install(root.path(), "grow", GROW, |manifest| {
        manifest["memory_mb"] = serde_json::json!(1);
    });
    let host = PluginHost::new_unsigned(root.path()).unwrap();
    let plugin = host.load("grow").unwrap();
    let error = match plugin.run(b"{}") {
        Ok(_) => panic!("expected plugin error"),
        Err(error) => error,
    };
    assert!(
        error.contains("memory") || error.contains("run failed"),
        "{error}"
    );
}

#[test]
fn a_hash_mismatch_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    install(root.path(), "echo", ECHO, |manifest| {
        manifest["sha256"] = serde_json::json!("00");
    });
    let host = PluginHost::new_unsigned(root.path()).unwrap();
    let error = match host.load("echo") {
        Ok(_) => panic!("expected load error"),
        Err(error) => error,
    };
    assert!(error.contains("sha256"), "{error}");
}

#[test]
fn listing_reports_manifest_validity() {
    let root = tempfile::tempdir().unwrap();
    install(root.path(), "echo", ECHO, |_| {});
    let host = PluginHost::new_unsigned(root.path()).unwrap();
    let rows = host.list();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["name"], "echo");
    assert_eq!(rows[0]["valid"], true);
}

fn install_signed(root: &Path, name: &str, wat: &str, key_id: &str) -> ed25519_dalek::SigningKey {
    let directory = root.join(name);
    std::fs::create_dir_all(&directory).unwrap();
    let wasm = wat::parse_str(wat).unwrap();
    std::fs::write(directory.join("module.wasm"), &wasm).unwrap();
    let digest = format!("{:x}", Sha256::digest(&wasm));
    let mut manifest = serde_json::json!({
        "name": name,
        "version": "0.1.0",
        "module": "module.wasm",
        "sha256": digest,
        "fuel": 1_000_000,
        "memory_mb": 2,
        "max_output_bytes": 65536,
        "key_id": key_id,
    });
    let parsed: PluginManifest = serde_json::from_value(manifest.clone()).unwrap();
    let payload = parsed.signing_payload(key_id).unwrap();
    let key = ed25519_dalek::SigningKey::generate(&mut OsRng);
    let signature = key.sign(&payload);
    manifest["signature"] =
        serde_json::json!(base64::engine::general_purpose::STANDARD.encode(signature.to_bytes()));
    std::fs::write(
        directory.join("plugin.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    key
}

#[test]
fn a_signed_plugin_verifies_and_a_tampered_signature_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let key = install_signed(root.path(), "signed", ECHO, "release-test");
    let trusted = BTreeMap::from([("release-test".to_owned(), key.verifying_key())]);
    let host = PluginHost::with_keys(root.path(), trusted, true).unwrap();
    let plugin = host.load("signed").unwrap();
    let result = plugin.run(br#"{"verdict":"pass"}"#).unwrap();
    assert_eq!(result.output, br#"{"verdict":"pass"}"#);

    let manifest_path = root.path().join("signed/plugin.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["signature"] = serde_json::json!("AAAA");
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let error = match host.load("signed") {
        Ok(_) => panic!("expected signature failure"),
        Err(error) => error,
    };
    assert!(error.contains("signature"), "{error}");
}

#[test]
fn a_signed_plugin_requires_a_trusted_key() {
    let root = tempfile::tempdir().unwrap();
    install_signed(root.path(), "signed", ECHO, "release-test");
    let host = PluginHost::with_keys(root.path(), BTreeMap::new(), true).unwrap();
    let error = match host.load("signed") {
        Ok(_) => panic!("expected trust failure"),
        Err(error) => error,
    };
    assert!(error.contains("not trusted"), "{error}");
}
