//! Per-card residency preflight for `hive serve`.
//!
//! Ports the conservative part of `harness/stack/residency.py`: GGUF weights
//! plus architecture-aware KV, split by `pin`/`ts`, against live VRAM after the
//! desktop's share and a per-tier compute buffer. It refuses an over-budget
//! stack before any engine is spawned.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use hive_core::hardware::{self, Device};
use serde_json::{Map, Value};

use crate::Options;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const DESKTOP_FALLBACK_GIB: f64 = 1.6;
const BIG_TIER_GIB: f64 = 10.0;
const COMPUTE_BUFFER_BIG_GIB: f64 = 1.5;
const COMPUTE_BUFFER_SMALL_GIB: f64 = 0.5;
const DRAFT_HEAD_GIB: f64 = 1.2;
const DEFAULT_CTX: i64 = 8192;

#[derive(Debug, Clone, Copy)]
pub struct CardPlan {
    pub card: usize,
    pub weights: f64,
    pub kv: f64,
    pub total: f64,
    pub budget: f64,
}

impl CardPlan {
    pub fn to_json(&self) -> Value {
        let mut object = Map::new();
        object.insert("card".into(), Value::from(self.card as u64));
        object.insert("weights".into(), number(self.weights, 3));
        object.insert("kv".into(), number(self.kv, 3));
        object.insert("total".into(), number(self.total, 3));
        object.insert("budget".into(), number(self.budget, 3));
        Value::Object(object)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub ok: bool,
    pub per_card: Vec<CardPlan>,
    pub warnings: Vec<String>,
}

/// Cards the plan may place on, honouring `HIP_VISIBLE_DEVICES` indices.
pub fn visible_devices() -> Vec<Device> {
    let mut devices = hardware::amd_devices().unwrap_or_default();
    if let Some(raw) = std::env::var_os("HIP_VISIBLE_DEVICES") {
        let text = raw.to_string_lossy();
        if !text.trim().is_empty() {
            let indices: Option<Vec<i64>> = text
                .split(',')
                .map(str::trim)
                .filter(|token| !token.is_empty())
                .map(|token| token.parse::<i64>().ok())
                .collect();
            if let Some(indices) = indices {
                devices.retain(|device| indices.contains(&(device.index as i64)));
            }
        }
    }
    devices
}

/// Build the residency plan. No devices means the check is skipped with a
/// warning: a CPU-only box has no VRAM to budget.
pub fn plan(stack: &Value, models_dir: &Path, devices: &[Device]) -> Plan {
    let tiers = stack
        .get("tiers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut warnings = Vec::new();
    let local: Vec<&Value> = tiers
        .iter()
        .filter(|tier| endpoint(tier).is_none())
        .collect();
    if tiers.is_empty() {
        return Plan {
            ok: false,
            per_card: Vec::new(),
            warnings: vec!["stack has no tiers".into()],
        };
    }
    if local.is_empty() {
        for tier in &tiers {
            let role = role(tier);
            let endpoint = endpoint(tier).unwrap_or_default();
            warnings.push(format!(
                "tier {role}: endpoint-backed ({endpoint}); residency is external, not planned"
            ));
        }
        return Plan {
            ok: true,
            per_card: Vec::new(),
            warnings,
        };
    }
    if devices.is_empty() {
        warnings.push("no visible GPU devices: residency is not planned".into());
        return Plan {
            ok: true,
            per_card: Vec::new(),
            warnings,
        };
    }

    let by_index: BTreeMap<usize, &Device> = devices
        .iter()
        .map(|device| (device.index, device))
        .collect();
    let mut weights: BTreeMap<usize, f64> =
        devices.iter().map(|device| (device.index, 0.0)).collect();
    let mut kv: BTreeMap<usize, f64> = devices.iter().map(|device| (device.index, 0.0)).collect();
    let mut resident: BTreeMap<usize, Vec<f64>> = devices
        .iter()
        .map(|device| (device.index, Vec::new()))
        .collect();
    let mut resolved = true;

    for tier in &tiers {
        if endpoint(tier).is_some() {
            warnings.push(format!(
                "tier {}: endpoint-backed ({}); residency is external, not planned",
                role(tier),
                endpoint(tier).unwrap_or_default()
            ));
            continue;
        }
        let (charged, tier_ok) = charge_tier(
            tier,
            models_dir,
            &by_index,
            &mut weights,
            &mut kv,
            &mut resident,
            &mut warnings,
        );
        let _ = charged;
        resolved = resolved && tier_ok;
    }

    let mut per_card = Vec::new();
    for device in devices {
        let index = device.index;
        let budget = budget(device, &resident[&index], &mut warnings);
        per_card.push(CardPlan {
            card: index,
            weights: weights[&index],
            kv: kv[&index],
            total: weights[&index] + kv[&index],
            budget,
        });
    }
    per_card.sort_by_key(|plan| plan.card);
    let mut over = false;
    for card in &per_card {
        if card.total > card.budget {
            over = true;
            warnings.push(format!(
                "card {} is over budget: {:.1} GiB needed, {:.1} GiB usable",
                card.card, card.total, card.budget
            ));
        }
    }
    Plan {
        ok: resolved && !over,
        per_card,
        warnings,
    }
}

fn charge_tier(
    tier: &Value,
    models_dir: &Path,
    by_index: &BTreeMap<usize, &Device>,
    weights: &mut BTreeMap<usize, f64>,
    kv: &mut BTreeMap<usize, f64>,
    resident: &mut BTreeMap<usize, Vec<f64>>,
    warnings: &mut Vec<String>,
) -> (bool, bool) {
    let role = role(tier);
    let file = tier
        .get("file")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if file.is_empty() {
        warnings.push(format!("tier {role}: no GGUF file named"));
        return (false, false);
    }
    let Some(path) = hive_models::resolve_model(file, models_dir).or_else(|| {
        let probe = PathBuf::from(file);
        probe.is_file().then_some(probe)
    }) else {
        warnings.push(format!(
            "tier {role}: {file} is not in the local model library; download it before applying the stack"
        ));
        return (false, false);
    };
    let Some(size_gib) = file_size_gib(&path) else {
        warnings.push(format!("tier {role}: cannot size {}", path.display()));
        return (false, false);
    };

    let estimate = hive_models::kv::estimate_path(&path);
    let ctx = tier.get("ctx").and_then(Value::as_i64).unwrap_or(0);
    let ctx = if ctx <= 0 {
        warnings.push(format!(
            "tier {role}: ctx is {ctx}; pricing KV at the llama.cpp default of {DEFAULT_CTX}"
        ));
        DEFAULT_CTX
    } else {
        ctx
    };
    let mut kv_gib = 0.0;
    match estimate {
        Some(estimate) => {
            let cache_k = text(tier, "cache_k").unwrap_or_else(|| "f16".into());
            let cache_v = text(tier, "cache_v").unwrap_or_else(|| cache_k.clone());
            let bytes_k = estimate.bytes_at(ctx, &cache_k);
            kv_gib = if cache_v.eq_ignore_ascii_case(&cache_k) {
                bytes_k
            } else {
                bytes_k.max(estimate.bytes_at(ctx, &cache_v))
            } / GIB;
        }
        None => warnings.push(format!(
            "tier {role}: {} has no readable KV metadata (no block_count / kv head count); KV is charged as 0 and only the weights are budgeted",
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        )),
    }

    let mut tier_weights = size_gib;
    if let Some(mmproj) = text(tier, "mmproj") {
        match hive_models::resolve_model(&mmproj, models_dir).or_else(|| {
            let probe = PathBuf::from(&mmproj);
            probe.is_file().then_some(probe)
        }) {
            Some(path) => match file_size_gib(&path) {
                Some(size) => tier_weights += size,
                None => warnings.push(format!("tier {role}: cannot size {}", path.display())),
            },
            None => warnings.push(format!(
                "tier {role}: vision projector {mmproj} is not in the local model library; its weights are not budgeted"
            )),
        }
    }
    if uses_draft_head(tier) {
        tier_weights += DRAFT_HEAD_GIB;
        warnings.push(format!(
            "tier {role}: speculative decoding reserves {DRAFT_HEAD_GIB} GiB of draft head (LOCAL-STACKS.md §6)"
        ));
    }

    let cards = split(tier, by_index, warnings);
    for (card, share) in cards {
        *weights.entry(card).or_insert(0.0) += tier_weights * share;
        *kv.entry(card).or_insert(0.0) += kv_gib * share;
        resident.entry(card).or_default().push(tier_weights);
    }
    (true, true)
}

fn split(
    tier: &Value,
    by_index: &BTreeMap<usize, &Device>,
    warnings: &mut Vec<String>,
) -> Vec<(usize, f64)> {
    let role = role(tier);
    let mut cards = pinned_cards(tier, by_index, warnings);
    if cards.is_empty() {
        cards = by_index.keys().copied().collect();
    }
    if cards.len() <= 1 {
        return cards.into_iter().map(|card| (card, 1.0)).collect();
    }
    let raw_ts = tier.get("ts").and_then(Value::as_str).unwrap_or("").trim();
    if let Some(shares) = tensor_split(raw_ts, cards.len()) {
        return cards.into_iter().zip(shares).collect();
    }
    warnings.push(format!(
        "tier {role}: spans cards {cards:?} without a -ts that names each of them ({raw_ts:?}); charging the whole tier to card {}",
        cards[0]
    ));
    vec![(cards[0], 1.0)]
}

fn pinned_cards(
    tier: &Value,
    by_index: &BTreeMap<usize, &Device>,
    warnings: &mut Vec<String>,
) -> Vec<usize> {
    let role = role(tier);
    let raw = tier.get("pin").and_then(Value::as_str).unwrap_or("").trim();
    if raw.is_empty() {
        return Vec::new();
    }
    let value = raw.split_once('=').map(|(_, value)| value).unwrap_or(raw);
    let mut cards = Vec::new();
    for part in value.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let Ok(card) = part.parse::<i64>() else {
            warnings.push(format!(
                "tier {role}: pin {raw:?} is not a card list; planning on every visible card"
            ));
            return Vec::new();
        };
        let card = card as usize;
        if !by_index.contains_key(&card) {
            warnings.push(format!(
                "tier {role}: pin {raw:?} names card {card}, which is not a visible device; planning on the visible cards"
            ));
            return Vec::new();
        }
        if !cards.contains(&card) {
            cards.push(card);
        }
    }
    cards
}

fn tensor_split(raw: &str, count: usize) -> Option<Vec<f64>> {
    if raw.is_empty() {
        return None;
    }
    let parts: Vec<&str> = raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if parts.len() != count {
        return None;
    }
    let values: Vec<f64> = parts
        .iter()
        .map(|part| part.parse::<f64>().ok())
        .collect::<Option<_>>()?;
    if values.iter().any(|value| *value < 0.0) || values.iter().sum::<f64>() <= 0.0 {
        return None;
    }
    let total: f64 = values.iter().sum();
    Some(values.into_iter().map(|value| value / total).collect())
}

fn budget(device: &Device, tier_weights: &[f64], warnings: &mut Vec<String>) -> f64 {
    let total = device
        .memory_gb
        .or_else(|| match (device.free_gb, device.used_gb) {
            (free, used) if free.is_some() || used.is_some() => {
                Some(free.unwrap_or(0.0) + used.unwrap_or(0.0))
            }
            _ => None,
        })
        .unwrap_or(0.0);
    let mut desktop = 0.0;
    if device.display {
        desktop = match device.used_gb {
            Some(used) if used > 0.0 => used,
            _ => {
                warnings.push(format!(
                    "card {} drives a display but reports no used_gb; charging the measured {DESKTOP_FALLBACK_GIB} GiB desktop cost",
                    device.index
                ));
                DESKTOP_FALLBACK_GIB
            }
        };
        warnings.push(format!(
            "card {} drives a display: {desktop:.1} GiB desktop cost excluded from its budget",
            device.index
        ));
    }
    let buffers: f64 = tier_weights
        .iter()
        .map(|weight| {
            if *weight >= BIG_TIER_GIB {
                COMPUTE_BUFFER_BIG_GIB
            } else {
                COMPUTE_BUFFER_SMALL_GIB
            }
        })
        .sum();
    (total - desktop - buffers).max(0.0)
}

fn role(tier: &Value) -> String {
    tier.get("role")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_owned()
}

fn text(tier: &Value, key: &str) -> Option<String> {
    tier.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

fn endpoint(tier: &Value) -> Option<String> {
    text(tier, "endpoint")
}

fn file_size_gib(path: &Path) -> Option<f64> {
    std::fs::metadata(path)
        .ok()
        .map(|metadata| metadata.len() as f64 / GIB)
}

fn uses_draft_head(tier: &Value) -> bool {
    let Some(spec) = tier.get("spec").and_then(Value::as_object) else {
        return false;
    };
    let kind = spec
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_lowercase();
    kind.contains("draft") || kind.contains("mtp")
}

fn number(value: f64, decimals: u32) -> Value {
    let factor = 10f64.powi(decimals as i32);
    let rounded = (value * factor).round() / factor;
    serde_json::Number::from_f64(rounded)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// Apply the preflight to a stack before the supervisor spawns anything.
pub fn preflight(stack: &Value, options: &Options) -> Result<Plan, String> {
    let devices = visible_devices();
    let plan = plan(stack, &options.models_dir, &devices);
    if plan.ok {
        Ok(plan)
    } else {
        let name = stack.get("name").and_then(Value::as_str).unwrap_or("stack");
        let detail = if plan.warnings.is_empty() {
            "over budget".to_owned()
        } else {
            plan.warnings.join("; ")
        };
        Err(format!("stack '{name}' refused: {detail}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Options;
    use serde_json::json;
    use std::path::PathBuf;

    fn device(index: usize, display: bool, total: f64, used: f64) -> Device {
        Device {
            index,
            card: format!("card{index}"),
            display,
            memory_gb: Some(total),
            used_gb: Some(used),
            free_gb: Some(total - used),
            ..Device::default()
        }
    }

    fn options(models_dir: &Path) -> Options {
        Options {
            host: "127.0.0.1".into(),
            base_port: 1234,
            llama_server: None,
            repo: PathBuf::from("/nonexistent"),
            models_dir: models_dir.to_path_buf(),
            log_dir: PathBuf::from("/tmp"),
            startup_timeout: 1.0,
        }
    }

    #[test]
    fn an_over_budget_tier_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("big.gguf");
        let file = std::fs::File::create(&model).unwrap();
        file.set_len(25 * 1024 * 1024 * 1024).unwrap();
        // A sparse 25 GiB file keeps the test cheap: only the size matters.
        let stack = json!({
            "name": "big",
            "tiers": [{"role": "face", "repo": "local", "file": "big.gguf", "ctx": 2048}]
        });
        let plan = plan(&stack, dir.path(), &[device(0, false, 20.0, 0.0)]);
        assert!(!plan.ok);
        assert!(
            plan.warnings
                .iter()
                .any(|warning| warning.contains("over budget"))
        );
    }

    #[test]
    fn a_small_tier_fits_and_charges_the_desktop() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("small.gguf"), vec![0u8; 1024 * 1024]).unwrap();
        let stack = json!({
            "name": "small",
            "tiers": [{"role": "face", "repo": "local", "file": "small.gguf", "ctx": 2048}]
        });
        let plan = plan(&stack, dir.path(), &[device(0, true, 20.0, 2.0)]);
        assert!(plan.ok);
        assert!(
            plan.warnings
                .iter()
                .any(|warning| warning.contains("desktop cost"))
        );
    }

    #[test]
    fn no_devices_skips_the_check_with_a_warning() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("small.gguf"), b"x").unwrap();
        let stack = json!({
            "name": "small",
            "tiers": [{"role": "face", "repo": "local", "file": "small.gguf"}]
        });
        let plan = plan(&stack, dir.path(), &[]);
        assert!(plan.ok);
        assert!(
            plan.warnings
                .iter()
                .any(|warning| warning.contains("not planned"))
        );
        let _ = options(dir.path());
    }
}
