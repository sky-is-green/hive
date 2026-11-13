//! Daemon-hosted native memory API.
//!
//! Per-conversation stores live here; the Python sidecar can load its persisted
//! chunks, add new ones, and ask for a native assembly. Embeddings come from
//! the managed `hive-encoder` service over loopback, so the daemon does not link
//! ONNX Runtime itself.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use hive_memory::assemble::{AssembleOptions, Embedder, assemble};
use hive_memory::vocab::Vocabulary;
use hive_memory::{Chunk, ContextStore};
use serde_json::{Value, json};

use crate::runtime::ApiError;

struct Conversation {
    store: ContextStore,
}

pub struct MemoryManager {
    conversations: Mutex<HashMap<String, Conversation>>,
    vocab: Arc<Vocabulary>,
    encoder_url: Option<String>,
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
        let store = {
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
            url: encoder_url.clone(),
        };
        let options = AssembleOptions {
            route: route.to_owned(),
            current_turn,
            max_context,
            ..Default::default()
        };
        let assembled =
            assemble(&store, query, &mut embedder, &self.vocab, &options).map_err(|error| {
                ApiError {
                    status: 502,
                    detail: format!("native assembly failed: {error}"),
                }
            })?;
        Ok(json!({
            "content": assembled.content,
            "token_count": assembled.token_count,
            "budget": assembled.budget,
            "selected_ids": assembled.selected_ids,
            "drift_detected": assembled.drift_detected,
            "drift_score": assembled.drift_score,
            "raw_scores": assembled.raw_scores,
            "effective_scores": assembled.effective_scores,
        }))
    }
}

struct HttpEmbedder {
    url: String,
}

impl Embedder for HttpEmbedder {
    fn encode(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let payload =
            serde_json::to_string(&json!({"texts": texts})).map_err(|error| error.to_string())?;
        let response = ureq::post(&format!("{}/encode", self.url.trim_end_matches('/')))
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
        vectors
            .iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| "encoder vector is not an array".to_owned())
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_f64)
                            .map(|value| value as f32)
                            .collect()
                    })
            })
            .collect()
    }
}
