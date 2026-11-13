//! Topic drift detection (membrane layer).

use crate::Chunk;

pub struct TopicDriftDetector {
    pub threshold: f64,
}

impl Default for TopicDriftDetector {
    fn default() -> Self {
        Self { threshold: 0.6 }
    }
}

impl TopicDriftDetector {
    pub fn check<F>(&self, recent: &[&Chunk], all: &[&Chunk], embed: F) -> (f64, bool)
    where
        F: Fn(&str) -> Vec<f32>,
    {
        if all.len() < 5 || recent.is_empty() {
            return (0.0, false);
        }
        let recent_text = recent
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let historical_text = all[..all.len() / 2]
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let recent_embedding = embed(&recent_text);
        let historical_embedding = embed(&historical_text);
        let similarity = crate::cosine(&recent_embedding, &historical_embedding);
        let score = 1.0 - similarity;
        (score, score > self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str) -> Chunk {
        Chunk {
            id: id.into(),
            content: format!("chunk {id}"),
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
    fn short_stores_never_drift() {
        let chunks: Vec<Chunk> = (0..4).map(|index| chunk(&index.to_string())).collect();
        let refs: Vec<&Chunk> = chunks.iter().collect();
        let (score, reset) = TopicDriftDetector::default().check(&refs, &refs, |_| vec![1.0, 0.0]);
        assert_eq!(score, 0.0);
        assert!(!reset);
    }
}
