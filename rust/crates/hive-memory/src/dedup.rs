//! Semantic duplicate detection (membrane layer).

use std::collections::{BTreeMap, BTreeSet};

use regex::Regex;

use crate::Chunk;
use crate::vocab::Vocabulary;

pub struct ContextDeduplicator<'a> {
    pub vocab: &'a Vocabulary,
    pub threshold: f64,
}

impl<'a> ContextDeduplicator<'a> {
    pub fn new(vocab: &'a Vocabulary, threshold: f64) -> Self {
        Self { vocab, threshold }
    }

    /// Returns surviving indices into `chunks` and the refresh map.
    pub fn deduplicate(
        &self,
        chunks: &[&Chunk],
        embeddings: &[Vec<f32>],
    ) -> (Vec<usize>, BTreeMap<String, u32>) {
        if chunks.len() <= 1 {
            return ((0..chunks.len()).collect(), BTreeMap::new());
        }
        let matrix = cosine_matrix(embeddings);
        let mut keep: BTreeSet<usize> = (0..chunks.len()).collect();
        let mut refresh = BTreeMap::new();
        for i in 0..chunks.len() {
            if !keep.contains(&i) {
                continue;
            }
            for j in (i + 1)..chunks.len() {
                if !keep.contains(&j) {
                    continue;
                }
                if matrix[i][j] > self.threshold {
                    let keep_index = if self.info_density(&chunks[i].content)
                        >= self.info_density(&chunks[j].content)
                    {
                        keep.remove(&j);
                        i
                    } else {
                        keep.remove(&i);
                        j
                    };
                    let freshest = chunks[i].turn.max(chunks[j].turn);
                    let entry = refresh.entry(chunks[keep_index].id.clone()).or_insert(0);
                    *entry = (*entry).max(freshest);
                }
            }
        }
        (keep.into_iter().collect(), refresh)
    }

    fn info_density(&self, content: &str) -> f64 {
        let word_re = Regex::new(r"\w+").expect("word regex");
        let words: Vec<&str> = word_re.find_iter(content).map(|m| m.as_str()).collect();
        let total_words = words.len().max(1) as f64;
        let term_count = self.vocab.matched_terms(content).len() as f64;
        let sentences: Vec<&str> = content.trim().split_inclusive(['.', '!', '?']).collect();
        let avg_sent_len = sentences
            .iter()
            .map(|sentence| word_re.find_iter(sentence).count() as f64)
            .sum::<f64>()
            / sentences.len().max(1) as f64;
        (term_count / total_words) * avg_sent_len
    }
}

fn cosine_matrix(rows: &[Vec<f32>]) -> Vec<Vec<f64>> {
    let count = rows.len();
    let mut out = vec![vec![0.0; count]; count];
    for i in 0..count {
        for j in 0..count {
            let dot: f64 = rows[i]
                .iter()
                .zip(&rows[j])
                .map(|(left, right)| *left as f64 * *right as f64)
                .sum();
            let denom = crate::norm(&rows[i]) * crate::norm(&rows[j]);
            out[i][j] = if denom == 0.0 { 0.0 } else { dot / denom };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocabulary {
        Vocabulary::new(["retention".to_owned(), "sandbox".to_owned()])
    }

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
    fn near_duplicates_keep_the_denser_chunk_and_refresh() {
        let vocab = vocab();
        let dedup = ContextDeduplicator::new(&vocab, 0.92);
        let first = chunk("a", "retention store keeps chunks", 2);
        let second = chunk("b", "retention store keeps chunks and sandbox", 5);
        let (survivors, refresh) = dedup.deduplicate(
            &[&first, &second],
            &[vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]],
        );
        assert_eq!(survivors, vec![1]);
        assert_eq!(refresh.get("b"), Some(&5));
    }
}
