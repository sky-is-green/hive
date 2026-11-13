//! Remembrance pass (retention layer).

use regex::Regex;

use crate::Chunk;
use crate::vocab::Vocabulary;

pub struct RemembranceResult {
    pub chunk_id: String,
    pub saved: bool,
    pub relevance_score: f64,
    pub compressed_content: Option<String>,
    pub new_decay: Option<f64>,
}

pub struct RemembrancePass<'a> {
    pub vocab: &'a Vocabulary,
    pub threshold: f64,
}

impl<'a> RemembrancePass<'a> {
    pub fn new(vocab: &'a Vocabulary) -> Self {
        Self {
            vocab,
            threshold: 0.65,
        }
    }

    pub fn process<F>(&self, chunks: &mut [Chunk], topic: &str, score: F) -> Vec<RemembranceResult>
    where
        F: Fn(&str) -> f64,
    {
        let mut results = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let relevance = score(&chunk.content);
            if relevance >= self.threshold {
                let compressed = self.compress(&chunk.content);
                let new_decay = chunk.decay_multiplier * decay_factor(chunk.times_saved);
                chunk.times_saved += 1;
                chunk.decay_multiplier = new_decay;
                results.push(RemembranceResult {
                    chunk_id: chunk.id.clone(),
                    saved: true,
                    relevance_score: relevance,
                    compressed_content: Some(compressed),
                    new_decay: Some(new_decay),
                });
            } else {
                results.push(RemembranceResult {
                    chunk_id: chunk.id.clone(),
                    saved: false,
                    relevance_score: relevance,
                    compressed_content: None,
                    new_decay: None,
                });
            }
        }
        let _ = topic;
        results
    }

    pub fn compress(&self, content: &str) -> String {
        let patterns = [
            r"^(I think|I believe|I feel|basically|essentially|actually),?\s*",
            r"(just|really|very|quite|rather)\s+",
            r"(kind of|sort of|a bit|a little)\s+",
            r"(as I mentioned|as discussed|per our conversation),?\s*",
            r"^(so|well|okay|right|sure),?\s*",
        ];
        let mut cleaned = content.to_owned();
        let mut changed = true;
        while changed {
            changed = false;
            for pattern in patterns {
                let Ok(regex) = Regex::new(&format!("(?i){pattern}")) else {
                    continue;
                };
                let next = regex.replace_all(&cleaned, "").into_owned();
                if next != cleaned {
                    cleaned = next;
                    changed = true;
                }
            }
        }
        let sentences: Vec<&str> = cleaned.trim().split_inclusive(['.', '!', '?']).collect();
        let important: Vec<&str> = sentences
            .iter()
            .copied()
            .filter(|sentence| self.vocab.matches(sentence))
            .collect();
        if important.is_empty() {
            cleaned
        } else {
            important.join(" ")
        }
    }
}

fn decay_factor(times_saved: u32) -> f64 {
    1.8 + (times_saved as f64 * 0.3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, content: &str) -> Chunk {
        Chunk {
            id: id.into(),
            content: content.into(),
            turn: 1,
            fingerprint: String::new(),
            timestamp: String::new(),
            decay_multiplier: 1.0,
            times_saved: 0,
            last_referenced_turn: Some(1),
            relevance_history: Vec::new(),
        }
    }

    #[test]
    fn a_relevant_chunk_is_saved_with_rising_decay() {
        let vocab = Vocabulary::new(["retention".to_owned()]);
        let pass = RemembrancePass::new(&vocab);
        let mut chunks = vec![chunk("a", "basically the retention store keeps chunks")];
        let results = pass.process(&mut chunks, "retention", |_| 0.9);
        assert!(results[0].saved);
        assert_eq!(results[0].new_decay, Some(1.8));
        assert_eq!(chunks[0].times_saved, 1);
        assert_eq!(chunks[0].decay_multiplier, 1.8);
    }
}
