//! Architecture-aware KV cache sizing for one GGUF.
//!
//! Mirrors `harness.models.attention_kv_estimate` / `KVEstimate`: full-attention
//! layers grow with the context, sliding layers saturate at the window, and
//! linear/recurrent layers carry no growing cache at all.

use serde_json::{Map, Value};

const LINEAR_LABELS: &[&str] = &[
    "linear_attention",
    "linear",
    "mamba",
    "recurrent",
    "recurrent_attention",
    "ssm",
    "gated_delta_net",
    "gated_delta",
    "rwkv",
    "meson",
];
const SLIDING_LABELS: &[&str] = &["sliding_attention", "sliding_window", "sliding", "swa"];

const KV_TYPE_BYTES: &[(&str, f64)] = &[
    ("f32", 4.0),
    ("f16", 2.0),
    ("bf16", 2.0),
    ("q8_0", 34.0 / 32.0),
    ("q5_1", 24.0 / 32.0),
    ("q5_0", 22.0 / 32.0),
    ("q4_1", 20.0 / 32.0),
    ("q4_0", 18.0 / 32.0),
    ("q6_k", 210.0 / 256.0),
    ("q5_k", 176.0 / 256.0),
    ("q4_k", 144.0 / 256.0),
    ("q3_k", 110.0 / 256.0),
    ("q2_k", 84.0 / 256.0),
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvEstimate {
    pub n_layers: i64,
    pub full_layers: i64,
    pub sliding_layers: i64,
    pub linear_layers: i64,
    pub sliding_window: i64,
    pub kv_heads: i64,
    pub key_len: i64,
    pub val_len: i64,
}

impl KvEstimate {
    pub fn elements_per_layer(&self) -> i64 {
        self.kv_heads * (self.key_len + self.val_len)
    }

    /// Total KV bytes at a fully filled `ctx` window.
    pub fn bytes_at(&self, ctx: i64, kv_type: &str) -> f64 {
        let ctx = ctx.max(0);
        let per_layer = self.elements_per_layer() as f64 * bytes_per_element(kv_type);
        let growing = self.full_layers as f64 * ctx as f64;
        let sliding = self.sliding_layers as f64 * ctx.min(self.sliding_window.max(0)) as f64;
        (growing + sliding) * per_layer
    }
}

pub fn estimate(meta: &Map<String, Value>) -> Option<KvEstimate> {
    if meta.is_empty() {
        return None;
    }
    let n_layers = meta_first_int(meta, ".block_count")?;
    let head = meta_first_int(meta, ".attention.head_count");
    let emb = meta_first_int(meta, ".embedding_length");
    let kv_heads = meta_first_int(meta, ".attention.head_count_kv")?;
    let mut key_len = meta_first_int(meta, ".attention.key_length");
    let mut val_len = meta_first_int(meta, ".attention.value_length");
    if key_len.is_none()
        && let (Some(emb), Some(head)) = (emb, head)
        && head != 0
    {
        key_len = Some(emb / head);
    }
    if val_len.is_none() {
        val_len = key_len;
    }
    let (Some(key_len), Some(val_len)) = (key_len, val_len) else {
        return None;
    };
    if n_layers <= 0 || kv_heads <= 0 || key_len <= 0 || val_len <= 0 {
        return None;
    }

    let window = meta_first_int(meta, ".attention.sliding_window").unwrap_or(0);
    let pattern = meta_first_int(meta, ".attention.sliding_window_pattern").unwrap_or(0);
    let labels = layer_types(meta);
    let arch = meta
        .get("architecture")
        .or_else(|| meta.get("general.architecture"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_lowercase();

    let (mut full, mut sliding, mut linear) = (0i64, 0i64, 0i64);
    if !labels.is_empty() {
        for label in &labels {
            if LINEAR_LABELS.contains(&label.as_str()) {
                linear += 1;
            } else if SLIDING_LABELS.contains(&label.as_str()) {
                sliding += 1;
            } else {
                full += 1;
            }
        }
        full += (n_layers - labels.len() as i64).max(0);
    } else if window > 0 {
        if pattern > 0 {
            sliding = (0..n_layers).step_by(pattern.max(1) as usize).count() as i64;
        } else {
            sliding = n_layers;
        }
        full = n_layers - sliding;
    } else if let Some(period) = arch_period(&arch) {
        let main_layers = if n_layers % period == 1 {
            n_layers - 1
        } else {
            n_layers
        };
        full = (main_layers / period).max(1);
        linear = n_layers - full;
    } else {
        full = n_layers;
    }

    Some(KvEstimate {
        n_layers,
        full_layers: full,
        sliding_layers: sliding,
        linear_layers: linear,
        sliding_window: window,
        kv_heads,
        key_len,
        val_len,
    })
}

/// Read a GGUF header and estimate its KV cache.
pub fn estimate_path(path: &std::path::Path) -> Option<KvEstimate> {
    let meta = crate::read_metadata(path);
    estimate(&meta)
}

fn arch_period(arch: &str) -> Option<i64> {
    match arch {
        "qwen35" | "qwen3_5_moe" => Some(4),
        _ => None,
    }
}

fn bytes_per_element(kind: &str) -> f64 {
    let key = kind.trim().to_lowercase();
    KV_TYPE_BYTES
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(_, bytes)| *bytes)
        .unwrap_or(2.0)
}

fn meta_first_int(meta: &Map<String, Value>, suffix: &str) -> Option<i64> {
    for (key, value) in meta {
        if !key.ends_with(suffix) {
            continue;
        }
        if let Some(number) = value.as_i64() {
            return Some(number);
        }
        if let Some(text) = value.as_str()
            && let Ok(number) = text.trim().parse::<i64>()
        {
            return Some(number);
        }
    }
    None
}

fn layer_types(meta: &Map<String, Value>) -> Vec<String> {
    for (key, value) in meta {
        if !key.ends_with(".attention.layer_types") {
            continue;
        }
        if let Some(items) = value.as_array() {
            return items
                .iter()
                .filter_map(Value::as_str)
                .map(|label| label.trim().to_lowercase())
                .filter(|label| !label.is_empty())
                .take(4096)
                .collect();
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn dense() -> Map<String, Value> {
        json!({
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.attention.key_length": 128,
            "llama.attention.value_length": 128
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[test]
    fn a_dense_model_prices_every_layer() {
        let estimate = estimate(&dense()).unwrap();
        assert_eq!(estimate.full_layers, 32);
        assert_eq!(estimate.sliding_layers, 0);
        // 32 * 8 * (128+128) * 8192 * 1.0625 bytes ~= 0.53 GiB
        let gib = estimate.bytes_at(8192, "q8_0") / (1024.0 * 1024.0 * 1024.0);
        assert!((gib - 0.531).abs() < 0.01, "{gib}");
    }

    #[test]
    fn the_reference_hybrid_is_34_kib_per_token_at_q8() {
        let meta = json!({
            "general.architecture": "qwen3_5",
            "qwen3_5.block_count": 64,
            "qwen3_5.attention.head_count_kv": 8,
            "qwen3_5.attention.key_length": 128,
            "qwen3_5.attention.value_length": 128,
            "qwen3_5.attention.layer_types": (0..64)
                .map(|index| if index % 4 == 0 { "full_attention" } else { "linear_attention" })
                .collect::<Vec<_>>()
        })
        .as_object()
        .unwrap()
        .clone();
        let estimate = estimate(&meta).unwrap();
        assert_eq!(estimate.full_layers, 16);
        assert_eq!(estimate.linear_layers, 48);
        let per_token = estimate.bytes_at(65_536, "q8_0") / 65_536.0;
        assert!(
            (per_token / 1024.0 - 34.0).abs() < 0.5,
            "{} KiB",
            per_token / 1024.0
        );
    }

    #[test]
    fn a_thin_header_has_no_estimate() {
        assert!(estimate(&Map::new()).is_none());
        assert!(estimate(json!({"llama.block_count": 32}).as_object().unwrap()).is_none());
    }
}
