//! Port of the retention store and decay core (Phase 3).
//!
//! This is the first native memory slice: chunk storage, fingerprint dedup,
//! eviction, turn indexing, decay, and cosine retrieval. Hygiene/codec
//! normalization, comb, drift, dedup and full assembly remain in the Python
//! reference until their parity tests land.

pub mod assemble;
pub mod budget;
pub mod codec;
pub mod comb;
pub mod dedup;
pub mod drift;
pub mod hygiene;
pub mod remembrance;
pub mod vocab;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub content: String,
    pub turn: u32,
    #[serde(default)]
    pub fingerprint: String,
    #[serde(default)]
    pub timestamp: String,
    #[serde(default = "one")]
    pub decay_multiplier: f64,
    #[serde(default)]
    pub times_saved: u32,
    #[serde(default)]
    pub last_referenced_turn: Option<u32>,
    #[serde(default)]
    pub relevance_history: Vec<(u32, f64)>,
}

fn one() -> f64 {
    1.0
}

impl Chunk {
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "content": self.content,
            "turn": self.turn,
            "fingerprint": self.fingerprint,
            "timestamp": self.timestamp,
            "decay_multiplier": self.decay_multiplier,
            "times_saved": self.times_saved,
            "last_referenced_turn": self.last_referenced_turn,
            "relevance_history": self.relevance_history,
        })
    }
}

pub struct ContextStore {
    pub chunks: Vec<Chunk>,
    pub turn_index: BTreeMap<u32, Vec<String>>,
    pub max_chunks: Option<usize>,
    pub sanitize: bool,
    pub max_chunk_chars: usize,
    pub ingest_block_prefixes: Vec<String>,
}

impl Default for ContextStore {
    fn default() -> Self {
        Self {
            chunks: Vec::new(),
            turn_index: BTreeMap::new(),
            max_chunks: None,
            sanitize: true,
            max_chunk_chars: hygiene::DEFAULT_MAX_CHUNK_CHARS,
            ingest_block_prefixes: hygiene::default_ingest_block_prefixes()
                .iter()
                .map(|prefix| (*prefix).to_owned())
                .collect(),
        }
    }
}

impl ContextStore {
    pub fn new(max_chunks: Option<usize>) -> Self {
        Self {
            max_chunks,
            ..Self::default()
        }
    }

    pub fn add_chunk(
        &mut self,
        turn: u32,
        content: &str,
        chunk_id: Option<&str>,
    ) -> Option<String> {
        let content = self.prepare(content)?;
        let content_fingerprint = fingerprint(&content);
        if let Some(existing) = self
            .chunks
            .iter_mut()
            .find(|chunk| chunk.fingerprint == content_fingerprint)
        {
            existing.last_referenced_turn = Some(turn);
            self.turn_index
                .entry(turn)
                .or_default()
                .push(existing.id.clone());
            return Some(existing.id.clone());
        }
        let id = chunk_id
            .map(str::to_owned)
            .unwrap_or_else(|| fingerprint(&format!("{turn}:{content}")));
        self.chunks.push(Chunk {
            id: id.clone(),
            content,
            turn,
            fingerprint: content_fingerprint,
            timestamp: chrono::Utc::now().to_rfc3339(),
            decay_multiplier: 1.0,
            times_saved: 0,
            last_referenced_turn: Some(turn),
            relevance_history: Vec::new(),
        });
        self.turn_index.entry(turn).or_default().push(id.clone());
        self.evict_overflow();
        Some(id)
    }

    fn prepare(&self, text: &str) -> Option<String> {
        hygiene::prepare_for_storage(
            text,
            self.max_chunk_chars,
            &self.ingest_block_prefixes,
            self.sanitize,
        )
    }

    fn evict_overflow(&mut self) {
        let Some(max) = self.max_chunks else {
            return;
        };
        while self.chunks.len() > max {
            let Some((index, _)) = self
                .chunks
                .iter()
                .enumerate()
                .min_by_key(|(_, chunk)| chunk.last_referenced_turn.unwrap_or(0))
            else {
                break;
            };
            let removed = self.chunks.remove(index);
            self.remove_from_turns(&removed.id);
        }
    }

    fn remove_from_turns(&mut self, id: &str) {
        self.turn_index.retain(|_, ids| {
            ids.retain(|candidate| candidate != id);
            !ids.is_empty()
        });
    }

    pub fn get_recent_chunks(&self, n: usize) -> Vec<&Chunk> {
        let turns: Vec<u32> = self.turn_index.keys().rev().take(n).copied().collect();
        let mut out = Vec::new();
        for turn in turns.into_iter().rev() {
            if let Some(ids) = self.turn_index.get(&turn) {
                for id in ids {
                    if let Some(chunk) = self.chunks.iter().find(|chunk| &chunk.id == id) {
                        out.push(chunk);
                    }
                }
            }
        }
        out
    }

    pub fn get_deletion_candidates(&self) -> Vec<&Chunk> {
        let Some(max) = self.max_chunks else {
            return Vec::new();
        };
        if self.chunks.len() <= max {
            return Vec::new();
        }
        let mut ordered: Vec<&Chunk> = self.chunks.iter().collect();
        ordered.sort_by_key(|chunk| chunk.last_referenced_turn.unwrap_or(0));
        ordered.into_iter().take(self.chunks.len() - max).collect()
    }

    pub fn from_chunks(chunks: Vec<Chunk>) -> Self {
        let mut turn_index: BTreeMap<u32, Vec<String>> = BTreeMap::new();
        for chunk in &chunks {
            turn_index
                .entry(chunk.turn)
                .or_default()
                .push(chunk.id.clone());
        }
        Self {
            chunks,
            turn_index,
            ..Self::default()
        }
    }

    pub fn count(&self) -> usize {
        self.chunks.len()
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "max_chunks": self.max_chunks,
            "ingest_block_prefixes": self.ingest_block_prefixes,
            "chunks": self.chunks.iter().map(Chunk::to_json).collect::<Vec<_>>(),
            "turn_index": self.turn_index.iter().map(|(turn, ids)| {
                (turn.to_string(), Value::Array(ids.iter().cloned().map(Value::String).collect()))
            }).collect::<Map<String, Value>>(),
        })
    }
}

pub fn fingerprint(text: &str) -> String {
    format!("{:x}", md5::compute(text.as_bytes()))[..12].to_owned()
}

/// `effective_score = raw / (decay_multiplier ** age_factor)`, with the stale
/// wall and optional drift penalties from the Python `DecayMatrix`.
pub fn apply_decay(
    chunks: &[&Chunk],
    current_turn: u32,
    raw_scores: &BTreeMap<String, f64>,
    drift_penalties: &BTreeMap<String, f64>,
    exempt_ids: &[String],
    stale_threshold: Option<u32>,
    stale_factor: Option<f64>,
) -> BTreeMap<String, f64> {
    let threshold = stale_threshold.unwrap_or(20);
    let factor = stale_factor.unwrap_or(0.5);
    let mut effective = BTreeMap::new();
    for chunk in chunks {
        let raw = raw_scores.get(&chunk.id).copied().unwrap_or(0.0);
        if exempt_ids.contains(&chunk.id) {
            effective.insert(chunk.id.clone(), raw.max(0.0));
            continue;
        }
        let age = current_turn.saturating_sub(chunk.last_referenced_turn.unwrap_or(chunk.turn));
        let age_factor = (age as f64 / 10.0).min(3.0);
        let mut decayed = raw / chunk.decay_multiplier.powf(age_factor);
        decayed *= drift_penalties.get(&chunk.id).copied().unwrap_or(1.0);
        if age > threshold {
            decayed *= factor;
        }
        effective.insert(chunk.id.clone(), decayed.max(0.0));
    }
    effective
}

/// Cosine scores for one query against every chunk, in chunk order.
pub fn cosine_scores(query: &[f32], vectors: &[Vec<f32>]) -> Vec<f64> {
    let query_norm = norm(query);
    vectors
        .iter()
        .map(|vector| {
            let dot: f64 = query
                .iter()
                .zip(vector)
                .map(|(left, right)| *left as f64 * *right as f64)
                .sum();
            let denominator = query_norm * norm(vector);
            if denominator == 0.0 {
                0.0
            } else {
                dot / denominator
            }
        })
        .collect()
}

pub fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let denom = norm(left) * norm(right);
    if denom == 0.0 {
        return 0.0;
    }
    left.iter()
        .zip(right)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum::<f64>()
        / denom
}

pub(crate) fn norm(vector: &[f32]) -> f64 {
    vector
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        .sqrt()
}
