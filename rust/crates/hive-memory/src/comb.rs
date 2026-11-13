//! Comb (surplus) store: JSONL-backed frozen chunks with lexical recall.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use regex::Regex;

use crate::Chunk;

#[derive(Debug, Clone)]
pub struct CombRecord {
    pub id: String,
    pub content: String,
    pub turn: u32,
    pub fingerprint: String,
    pub timestamp: String,
    pub decay_multiplier: f64,
    pub times_saved: u32,
    pub last_referenced_turn: Option<u32>,
    pub relevance_history: Vec<(u32, f64)>,
    pub embedding: Option<Vec<f32>>,
}

impl CombRecord {
    pub fn from_chunk(chunk: &Chunk, embedding: Option<Vec<f32>>) -> Self {
        Self {
            id: chunk.id.clone(),
            content: chunk.content.clone(),
            turn: chunk.turn,
            fingerprint: chunk.fingerprint.clone(),
            timestamp: chunk.timestamp.clone(),
            decay_multiplier: chunk.decay_multiplier,
            times_saved: chunk.times_saved,
            last_referenced_turn: chunk.last_referenced_turn.or(Some(chunk.turn)),
            relevance_history: chunk.relevance_history.clone(),
            embedding,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut object = serde_json::json!({
            "id": self.id,
            "content": self.content,
            "turn": self.turn,
            "fingerprint": self.fingerprint,
            "timestamp": self.timestamp,
            "decay_multiplier": self.decay_multiplier,
            "times_saved": self.times_saved,
            "last_referenced_turn": self.last_referenced_turn,
            "relevance_history": self.relevance_history,
        });
        if let Some(embedding) = &self.embedding {
            object["embedding"] = serde_json::json!(embedding);
        }
        object
    }

    pub fn from_json(value: &serde_json::Value) -> Option<Self> {
        Some(Self {
            id: value.get("id")?.as_str()?.to_owned(),
            content: value.get("content")?.as_str()?.to_owned(),
            turn: value.get("turn")?.as_u64()? as u32,
            fingerprint: value
                .get("fingerprint")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned(),
            timestamp: value
                .get("timestamp")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned(),
            decay_multiplier: value
                .get("decay_multiplier")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0),
            times_saved: value
                .get("times_saved")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            last_referenced_turn: value
                .get("last_referenced_turn")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
            relevance_history: value
                .get("relevance_history")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| {
                            let pair = item.as_array()?;
                            Some((pair.first()?.as_u64()? as u32, pair.get(1)?.as_f64()?))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            embedding: value
                .get("embedding")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_f64().map(|v| v as f32))
                        .collect()
                }),
        })
    }
}

pub struct CombStore {
    pub path: PathBuf,
    pub max_records: usize,
    pub records: BTreeMap<String, CombRecord>,
    pub put_count: usize,
}

impl CombStore {
    pub fn new(path: &Path, max_records: usize) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let mut store = Self {
            path: path.to_path_buf(),
            max_records,
            records: BTreeMap::new(),
            put_count: 0,
        };
        store.load()?;
        Ok(store)
    }

    fn load(&mut self) -> Result<(), String> {
        let Ok(text) = std::fs::read_to_string(&self.path) else {
            return Ok(());
        };
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(line)
                && let Some(record) = CombRecord::from_json(&value)
            {
                self.records.insert(record.id.clone(), record);
            }
        }
        Ok(())
    }

    pub fn put(&mut self, chunk: &Chunk, embedding: Option<Vec<f32>>) -> Result<(), String> {
        let record = CombRecord::from_chunk(chunk, embedding);
        self.records.insert(record.id.clone(), record.clone());
        self.append(&record)?;
        self.put_count += 1;
        if self.records.len() > self.max_records {
            self.prune(None, None, None)?;
        }
        Ok(())
    }

    pub fn retrieve(&self, query: &str, k: usize) -> Vec<&CombRecord> {
        let word = Regex::new(r"[a-z0-9]{4,}").expect("word regex");
        let mut query_words: Vec<String> = word
            .find_iter(&query.to_lowercase())
            .map(|m| m.as_str().to_owned())
            .collect();
        if query_words.is_empty() {
            query_words.push(query.to_lowercase());
        }
        let mut ranked: Vec<&CombRecord> = self
            .records
            .values()
            .filter(|record| overlap(record, &query_words, &word) > 0)
            .collect();
        ranked.sort_by(|left, right| {
            overlap(right, &query_words, &word)
                .cmp(&overlap(left, &query_words, &word))
                .then_with(|| {
                    (right.last_referenced_turn.unwrap_or(0))
                        .cmp(&left.last_referenced_turn.unwrap_or(0))
                })
        });
        ranked.truncate(k);
        ranked
    }

    pub fn touch(&mut self, ids: &[String], turn: u32) -> Result<usize, String> {
        let mut changed = 0;
        for id in ids {
            if let Some(record) = self.records.get_mut(id) {
                record.last_referenced_turn =
                    Some(record.last_referenced_turn.unwrap_or(0).max(turn));
                changed += 1;
            }
        }
        if changed > 0 {
            self.flush()?;
        }
        Ok(changed)
    }

    pub fn prune(
        &mut self,
        max_records: Option<usize>,
        max_age_turns: Option<u32>,
        current_turn: Option<u32>,
    ) -> Result<usize, String> {
        let mut dropped = 0;
        if let (Some(max_age), Some(current)) = (max_age_turns, current_turn) {
            let cutoff = current.saturating_sub(max_age);
            let remove: Vec<String> = self
                .records
                .iter()
                .filter(|(_, record)| record.last_referenced_turn.unwrap_or(0) < cutoff)
                .map(|(id, _)| id.clone())
                .collect();
            for id in remove {
                self.records.remove(&id);
                dropped += 1;
            }
        }
        let cap = max_records.unwrap_or(self.max_records);
        while self.records.len() > cap {
            if let Some(id) = self
                .records
                .values()
                .min_by_key(|record| record.last_referenced_turn.unwrap_or(0))
                .map(|record| record.id.clone())
            {
                self.records.remove(&id);
                dropped += 1;
            }
        }
        if dropped > 0 {
            self.flush()?;
        }
        Ok(dropped)
    }

    pub fn flush(&self) -> Result<(), String> {
        use std::io::Write;
        let mut file = std::fs::File::create(&self.path).map_err(|error| error.to_string())?;
        for record in self.records.values() {
            writeln!(
                file,
                "{}",
                serde_json::to_string(&record.to_json()).unwrap_or_default()
            )
            .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn append(&self, record: &CombRecord) -> Result<(), String> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|error| error.to_string())?;
        writeln!(
            file,
            "{}",
            serde_json::to_string(&record.to_json()).unwrap_or_default()
        )
        .map_err(|error| error.to_string())?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

fn overlap(record: &CombRecord, query_words: &[String], word: &Regex) -> usize {
    let content_words: Vec<String> = word
        .find_iter(&record.content.to_lowercase())
        .map(|m| m.as_str().to_owned())
        .collect();
    query_words
        .iter()
        .filter(|query| content_words.contains(query))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, content: &str, turn: u32) -> Chunk {
        Chunk {
            id: id.into(),
            content: content.into(),
            turn,
            fingerprint: String::new(),
            timestamp: String::new(),
            decay_multiplier: 1.0,
            times_saved: 0,
            last_referenced_turn: Some(turn),
            relevance_history: Vec::new(),
        }
    }

    #[test]
    fn comb_round_trips_and_ranks_lexically() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("comb.jsonl");
        let mut store = CombStore::new(&path, 10).unwrap();
        store
            .put(
                &chunk("a", "the retention store keeps chunks", 1),
                Some(vec![1.0, 0.0]),
            )
            .unwrap();
        store
            .put(&chunk("b", "quantization trades precision", 2), None)
            .unwrap();
        assert_eq!(store.len(), 2);
        let hits = store.retrieve("retention chunks", 3);
        assert_eq!(hits[0].id, "a");
        assert_eq!(store.touch(&["a".to_owned()], 5).unwrap(), 1);
        store.prune(Some(1), None, None).unwrap();
        assert_eq!(store.len(), 1);

        let reloaded = CombStore::new(&path, 10).unwrap();
        assert_eq!(reloaded.len(), 1);
        assert_eq!(reloaded.records["a"].last_referenced_turn, Some(5));
    }
}
