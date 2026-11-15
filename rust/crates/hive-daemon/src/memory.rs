//! Daemon-hosted native memory API.
//!
//! Per-conversation stores live here; the Python sidecar can load its persisted
//! chunks, add new ones, and ask for a native assembly. Embeddings come from
//! the managed `hive-encoder` service over loopback, so the daemon does not link
//! ONNX Runtime itself.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use hive_memory::assemble::{AssembleOptions, CombCandidate, Embedder, assemble};
use hive_memory::vocab::Vocabulary;
use hive_memory::{Chunk, ContextStore};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::runtime::ApiError;

struct Conversation {
    store: ContextStore,
}

#[derive(Deserialize)]
struct CombInput {
    #[serde(flatten)]
    chunk: Chunk,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

pub struct MemoryManager {
    conversations: Mutex<HashMap<String, Conversation>>,
    vocab: Arc<Vocabulary>,
    encoder_url: Option<String>,
    agent: ureq::Agent,
    embeddings: Arc<Mutex<HashMap<String, Vec<f32>>>>,
}

impl MemoryManager {
    pub fn new(encoder_url: Option<String>) -> Self {
        let paths = hive_core::paths::Paths::discover(None, None);
        let vocab_dir = paths.repo.join("splinter/splinter/vocab");
        let vocab = Vocabulary::load(&vocab_dir, &["code", "general"])
            .unwrap_or_else(|_| Vocabulary::new(Vec::<String>::new()));
        Self {
            conversations: Mutex::new(HashMap::new()),
            vocab: Arc::new(vocab),
            encoder_url,
            agent: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(120))
                .build(),
            embeddings: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn load(&self, conversation: &str, body: &Value) -> Result<Value, ApiError> {
        let chunks: Vec<Chunk> = body
            .get("chunks")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| ApiError {
                status: 400,
                detail: format!("invalid chunks: {error}"),
            })?
            .unwrap_or_default();
        let max_chunks = body
            .get("max_chunks")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        let mut store = ContextStore::from_chunks(chunks);
        store.max_chunks = max_chunks;
        if let Some(sanitize) = body.get("sanitize").and_then(Value::as_bool) {
            store.sanitize = sanitize;
        }
        if let Some(prefixes) = body.get("ingest_block_prefixes").and_then(Value::as_array) {
            store.ingest_block_prefixes = prefixes
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
        }
        let count = store.count();
        self.conversations
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(conversation.to_owned(), Conversation { store });
        Ok(json!({"ok": true, "conversation": conversation, "chunks": count}))
    }

    pub fn add(&self, conversation: &str, body: &Value) -> Result<Value, ApiError> {
        let turn = body.get("turn").and_then(Value::as_u64).unwrap_or(1) as u32;
        let content = body
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError {
                status: 422,
                detail: "content is required".into(),
            })?;
        let chunk_id = body.get("chunk_id").and_then(Value::as_str);
        let mut conversations = self
            .conversations
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let conversation = conversations
            .entry(conversation.to_owned())
            .or_insert_with(|| Conversation {
                store: ContextStore::default(),
            });
        let id = conversation.store.add_chunk(turn, content, chunk_id);
        Ok(json!({"id": id, "count": conversation.store.count()}))
    }

    pub fn reset(&self, conversation: &str) -> Value {
        self.conversations
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(conversation);
        json!({"ok": true, "conversation": conversation})
    }

    pub fn status(&self, conversation: &str) -> Value {
        let conversations = self
            .conversations
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match conversations.get(conversation) {
            Some(entry) => json!({
                "conversation": conversation,
                "loaded": true,
                "chunks": entry.store.count(),
            }),
            None => json!({"conversation": conversation, "loaded": false, "chunks": 0}),
        }
    }

    pub fn assemble(&self, conversation: &str, body: &Value) -> Result<Value, ApiError> {
        let Some(encoder_url) = &self.encoder_url else {
            return Err(ApiError {
                status: 503,
                detail: "native memory needs the managed encoder (HIVE_ENCODER_PORT)".into(),
            });
        };
        let query = body
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError {
                status: 422,
                detail: "query is required".into(),
            })?;
        let current_turn = body
            .get("current_turn")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let route = body
            .get("route")
            .and_then(Value::as_str)
            .unwrap_or("ultra_small");
        let max_context = body
            .get("max_context")
            .and_then(Value::as_u64)
            .unwrap_or(8192) as u32;
        let max_chunk_share = body
            .get("max_chunk_share")
            .and_then(Value::as_f64)
            .unwrap_or(0.5);
        let relevance_floor = body
            .get("relevance_floor")
            .and_then(Value::as_f64)
            .unwrap_or(0.25);
        let drift_threshold = body
            .get("drift_threshold")
            .and_then(Value::as_f64)
            .unwrap_or(0.6);
        let dedup_threshold = body
            .get("dedup_threshold")
            .and_then(Value::as_f64)
            .unwrap_or(0.92);
        let stale_threshold = body
            .get("stale_threshold")
            .and_then(Value::as_u64)
            .map(|value| value as u32);
        let stale_factor = body.get("stale_factor").and_then(Value::as_f64);
        let payload_fingerprints = body
            .get("payload_fingerprints")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let score_override: Option<BTreeMap<String, f64>> = body
            .get("score_override")
            .filter(|value| !value.is_null())
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| ApiError {
                status: 400,
                detail: format!("invalid score_override: {error}"),
            })?;
        let comb: Vec<CombInput> = body
            .get("comb_candidates")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| ApiError {
                status: 400,
                detail: format!("invalid comb_candidates: {error}"),
            })?
            .unwrap_or_default();
        let options = AssembleOptions {
            route: route.to_owned(),
            current_turn,
            max_context,
            relevance_floor,
            max_chunk_share,
            drift_threshold,
            dedup_threshold,
            stale_threshold,
            stale_factor,
            skip_remembrance: body
                .get("skip_remembrance")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            skip_dedup: body
                .get("skip_dedup")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            dedup_against_payload: body
                .get("dedup_against_payload")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            payload_fingerprints,
            drift_prior: body
                .get("drift_prior")
                .and_then(Value::as_str)
                .unwrap_or("off")
                .to_owned(),
            drift_prior_factor: body
                .get("drift_prior_factor")
                .and_then(Value::as_f64)
                .unwrap_or(0.5),
            vocab_boost: body
                .get("vocab_boost")
                .and_then(Value::as_f64)
                .unwrap_or(0.15),
            budget_override: body
                .get("budget")
                .and_then(Value::as_u64)
                .map(|value| value as u32),
            score_override,
        };

        let mut store = {
            let conversations = self
                .conversations
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let entry = conversations.get(conversation).ok_or_else(|| ApiError {
                status: 404,
                detail: format!("no native memory loaded for {conversation}"),
            })?;
            entry.store.clone()
        };
        let mut embedder = HttpEmbedder {
            agent: self.agent.clone(),
            url: encoder_url.clone(),
            cache: Arc::clone(&self.embeddings),
        };
        let comb_candidates: Vec<CombCandidate> = comb
            .into_iter()
            .map(|input| CombCandidate {
                chunk: input.chunk,
                embedding: input.embedding,
            })
            .collect();
        let assemble_started = std::time::Instant::now();
        let assembled = assemble(
            &mut store,
            query,
            &mut embedder,
            &self.vocab,
            &options,
            &comb_candidates,
        )
        .map_err(|error| ApiError {
            status: 502,
            detail: format!("native assembly failed: {error}"),
        })?;
        if std::env::var_os("HIVE_MEMORY_TRACE").is_some() {
            eprintln!(
                "[memory] chunks={} assemble={:?}",
                store.count(),
                assemble_started.elapsed()
            );
        }

        // Assembly mutates decay state, membrane refresh, and relevance
        // history, so the updated store replaces the conversation entry.
        if let Some(entry) = self
            .conversations
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get_mut(conversation)
        {
            entry.store = store;
        }

        Ok(json!({
            "content": assembled.content,
            "token_count": assembled.token_count,
            "budget": assembled.budget,
            "selected_ids": assembled.selected_ids,
            "drift_detected": assembled.drift_detected,
            "drift_score": assembled.drift_score,
            "top_raw_score": assembled.top_raw_score,
            "top_chunk_id": assembled.top_chunk_id,
            "raw_scores": assembled.raw_scores,
            "effective_scores": assembled.effective_scores,
            "payload_dedup_skipped": assembled.payload_dedup_skipped,
            "refresh": assembled.refresh,
            "relevance_updates": assembled.relevance_updates,
            "remembrance": assembled.remembrance.iter().map(|result| json!({
                "chunk_id": result.chunk_id,
                "saved": result.saved,
                "relevance_score": result.relevance_score,
                "compressed_content": result.compressed_content,
                "new_decay": result.new_decay,
                "times_saved": result.times_saved,
            })).collect::<Vec<_>>(),
        }))
    }
}

struct HttpEmbedder {
    agent: ureq::Agent,
    url: String,
    cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
}

impl Embedder for HttpEmbedder {
    fn encode(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let started = std::time::Instant::now();
        let missing: Vec<String> = {
            let cache = self.cache.lock().unwrap_or_else(|error| error.into_inner());
            texts
                .iter()
                .filter(|text| !cache.contains_key(text.as_str()))
                .cloned()
                .collect()
        };
        let missing_count = missing.len();
        if !missing.is_empty() {
            let payload = serde_json::to_string(&json!({"texts": missing}))
                .map_err(|error| error.to_string())?;
            let response = self
                .agent
                .post(&format!("{}/encode", self.url.trim_end_matches('/')))
                .set("content-type", "application/json")
                .send_string(&payload)
                .map_err(|error| format!("encoder request failed: {error}"))?;
            let text = response
                .into_string()
                .map_err(|error| format!("cannot read encoder response: {error}"))?;
            let payload: Value = serde_json::from_str(&text)
                .map_err(|error| format!("invalid encoder response: {error}"))?;
            let vectors = payload
                .get("vectors")
                .and_then(Value::as_array)
                .ok_or_else(|| "encoder response has no vectors".to_owned())?;
            let mut cache = self.cache.lock().unwrap_or_else(|error| error.into_inner());
            for (text, row) in missing.iter().zip(vectors) {
                let vector: Vec<f32> = row
                    .as_array()
                    .ok_or_else(|| "encoder vector is not an array".to_owned())?
                    .iter()
                    .filter_map(Value::as_f64)
                    .map(|value| value as f32)
                    .collect();
                cache.insert(text.clone(), vector);
            }
        }
        if std::env::var_os("HIVE_MEMORY_TRACE").is_some() {
            eprintln!(
                "[memory] encode texts={} missing={} elapsed={:?}",
                texts.len(),
                missing_count,
                started.elapsed()
            );
        }
        let cache = self.cache.lock().unwrap_or_else(|error| error.into_inner());
        texts
            .iter()
            .map(|text| {
                cache
                    .get(text.as_str())
                    .cloned()
                    .ok_or_else(|| format!("encoder returned no vector for {text:?}"))
            })
            .collect()
    }
}
