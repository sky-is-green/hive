//! Native context assembly (focal layer), minus comb/remembrance for now.

use std::collections::BTreeMap;

use hive_encoder::Encoder;

use crate::budget::AdaptiveBudget;
use crate::dedup::ContextDeduplicator;
use crate::drift::TopicDriftDetector;
use crate::vocab::Vocabulary;
use crate::{ContextStore, apply_decay, cosine_scores};

pub struct AssembleOptions {
    pub route: String,
    pub max_context: u32,
    pub current_turn: u32,
    pub relevance_floor: f64,
    pub max_chunk_share: f64,
    pub drift_threshold: f64,
    pub dedup_threshold: f64,
    pub stale_threshold: Option<u32>,
    pub stale_factor: Option<f64>,
}

impl Default for AssembleOptions {
    fn default() -> Self {
        Self {
            route: "ultra_small".into(),
            max_context: 8192,
            current_turn: 0,
            relevance_floor: 0.25,
            max_chunk_share: 0.5,
            drift_threshold: 0.6,
            dedup_threshold: 0.92,
            stale_threshold: None,
            stale_factor: None,
        }
    }
}

pub struct Assembled {
    pub content: String,
    pub token_count: u32,
    pub budget: u32,
    pub selected_ids: Vec<String>,
    pub raw_scores: BTreeMap<String, f64>,
    pub effective_scores: BTreeMap<String, f64>,
    pub drift_detected: bool,
    pub drift_score: f64,
}

pub fn assemble(
    store: &ContextStore,
    query: &str,
    encoder: &mut Encoder,
    vocab: &Vocabulary,
    options: &AssembleOptions,
) -> Result<Assembled, String> {
    if store.chunks.is_empty() {
        return Ok(Assembled {
            content: String::new(),
            token_count: 0,
            budget: AdaptiveBudget::default().compute(&options.route, 0, options.max_context),
            selected_ids: Vec::new(),
            raw_scores: BTreeMap::new(),
            effective_scores: BTreeMap::new(),
            drift_detected: false,
            drift_score: 0.0,
        });
    }
    let mut texts = vec![query.to_owned()];
    texts.extend(store.chunks.iter().map(|chunk| chunk.content.clone()));
    let vectors = encoder.encode(&texts)?;
    let raw_values = cosine_scores(&vectors[0], &vectors[1..]);
    let raw_scores: BTreeMap<String, f64> = store
        .chunks
        .iter()
        .zip(raw_values)
        .map(|(chunk, score)| (chunk.id.clone(), score))
        .collect();

    let chunk_refs: Vec<&crate::Chunk> = store.chunks.iter().collect();
    let dedup = ContextDeduplicator::new(vocab, options.dedup_threshold);
    let (survivors, _refresh) = dedup.deduplicate(&chunk_refs, &vectors[1..]);
    let surviving: Vec<&crate::Chunk> = survivors.iter().map(|index| chunk_refs[*index]).collect();

    let recent = store.get_recent_chunks(3);
    let detector = TopicDriftDetector {
        threshold: options.drift_threshold,
    };
    let (drift_score, drift_detected) = detector.check(&recent, &chunk_refs, |text| {
        // The joined texts are not in the batch; this closure cannot borrow
        // the mutable encoder. Drift is computed separately below.
        let _ = text;
        vec![0.0]
    });
    let drift = if chunk_refs.len() >= 5 && !recent.is_empty() {
        let recent_text = recent
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let historical_text = chunk_refs[..chunk_refs.len() / 2]
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let drift_vectors = encoder.encode(&[recent_text, historical_text])?;
        let similarity = crate::cosine(&drift_vectors[0], &drift_vectors[1]);
        let score = 1.0 - similarity;
        (score, score > options.drift_threshold)
    } else {
        (0.0, false)
    };
    let _ = (drift_score, drift_detected);
    let (drift_score, drift_detected) = drift;

    let mut penalties = BTreeMap::new();
    if drift_detected {
        let recent_ids: Vec<&str> = recent.iter().map(|chunk| chunk.id.as_str()).collect();
        for chunk in &surviving {
            penalties.insert(
                chunk.id.clone(),
                if recent_ids.contains(&chunk.id.as_str()) {
                    1.0
                } else {
                    0.1
                },
            );
        }
    }

    let effective = apply_decay(
        &surviving,
        options.current_turn,
        &raw_scores,
        &penalties,
        &[],
        options.stale_threshold,
        options.stale_factor,
    );
    let high_relevance = effective.values().filter(|score| **score > 0.6).count();
    let budget =
        AdaptiveBudget::default().compute(&options.route, high_relevance, options.max_context);
    let mut scored: Vec<(&crate::Chunk, f64)> = surviving
        .iter()
        .map(|chunk| (*chunk, effective.get(&chunk.id).copied().unwrap_or(0.0)))
        .collect();
    scored.sort_by(|left, right| right.1.total_cmp(&left.1));

    let share_cap = (budget as f64 * options.max_chunk_share) as u32;
    let mut selected: Vec<crate::Chunk> = Vec::new();
    let mut used = 0u32;
    for (chunk, _) in scored {
        if raw_scores.get(&chunk.id).copied().unwrap_or(0.0) < options.relevance_floor {
            continue;
        }
        let mut content = chunk.content.clone();
        let mut cost = estimate_tokens(&content);
        if share_cap > 0 && cost > share_cap {
            content = truncate_to_token_cap(&content, share_cap);
            if content.chars().count() < 32 {
                continue;
            }
            cost = estimate_tokens(&content);
        }
        if used + cost > budget {
            continue;
        }
        let mut copy = chunk.clone();
        copy.content = content;
        selected.push(copy);
        used += cost;
    }
    let content = selected
        .iter()
        .map(|chunk| chunk.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let token_count = selected
        .iter()
        .map(|chunk| estimate_tokens(&chunk.content))
        .sum();
    Ok(Assembled {
        content,
        token_count,
        budget,
        selected_ids: selected.iter().map(|chunk| chunk.id.clone()).collect(),
        raw_scores,
        effective_scores: effective,
        drift_detected,
        drift_score,
    })
}

pub fn estimate_tokens(text: &str) -> u32 {
    ((text.chars().count() / 4) as u32).max(1)
}

fn truncate_to_token_cap(text: &str, token_cap: u32) -> String {
    if token_cap == 0 {
        return String::new();
    }
    if estimate_tokens(text) <= token_cap {
        return text.to_owned();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut lo = 0usize;
    let mut hi = chars.len();
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if estimate_tokens(&chars[..mid].iter().collect::<String>()) <= token_cap {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    chars[..lo].iter().collect()
}
