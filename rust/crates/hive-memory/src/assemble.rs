//! Native context assembly (focal layer), including comb and remembrance.

use std::collections::{BTreeMap, BTreeSet};

use regex::Regex;

use crate::budget::AdaptiveBudget;
use crate::codec::normalize_codec;
use crate::dedup::ContextDeduplicator;
use crate::remembrance::{RemembrancePass, RemembranceResult};
use crate::vocab::Vocabulary;
use crate::{Chunk, ContextStore, apply_decay, cosine_scores};

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
    pub skip_remembrance: bool,
    pub skip_dedup: bool,
    pub dedup_against_payload: bool,
    pub payload_fingerprints: Vec<String>,
    pub drift_prior: String,
    pub drift_prior_factor: f64,
    /// Added to the cosine of candidates that match the domain vocabulary,
    /// mirroring `UltraSmallDrone.vocab_boost`.
    pub vocab_boost: f64,
    /// A fixed token budget (used by the P4/P11 pressure replays). When absent
    /// the adaptive budget is computed from the effective scores.
    pub budget_override: Option<u32>,
    /// Route-specific relevance scores computed by the sidecar's drone fleet
    /// (medium/escalation tiers live in Python). Keyed by chunk id and already
    /// boosted; when present native assembly skips its own ultra-small pass.
    pub score_override: Option<BTreeMap<String, f64>>,
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
            skip_remembrance: false,
            skip_dedup: false,
            dedup_against_payload: true,
            payload_fingerprints: Vec::new(),
            drift_prior: "off".into(),
            drift_prior_factor: 0.5,
            vocab_boost: 0.15,
            budget_override: None,
            score_override: None,
        }
    }
}

#[derive(Clone)]
pub struct CombCandidate {
    pub chunk: Chunk,
    pub embedding: Option<Vec<f32>>,
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
    pub top_raw_score: f64,
    pub top_chunk_id: Option<String>,
    pub payload_dedup_skipped: usize,
    pub remembrance: Vec<RemembranceResult>,
    pub refresh: BTreeMap<String, u32>,
    pub relevance_updates: BTreeMap<String, Vec<(u32, f64)>>,
}

pub trait Embedder {
    fn encode(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
}

#[cfg(feature = "native-encoder")]
impl Embedder for hive_encoder::Encoder {
    fn encode(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        hive_encoder::Encoder::encode(self, texts)
    }
}

pub fn assemble(
    store: &mut ContextStore,
    query: &str,
    encoder: &mut dyn Embedder,
    vocab: &Vocabulary,
    options: &AssembleOptions,
    comb_candidates: &[CombCandidate],
) -> Result<Assembled, String> {
    let remembrance = if options.skip_remembrance {
        Vec::new()
    } else {
        run_remembrance(store, query, encoder, vocab, options.vocab_boost)?
    };

    // Comb records join the active store in one candidate pool; they compete
    // for the same token budget and never enlarge the context window.
    let store_len = store.chunks.len();
    let mut pool: Vec<Chunk> = store.chunks.clone();
    pool.extend(
        comb_candidates
            .iter()
            .map(|candidate| candidate.chunk.clone()),
    );
    let comb_ids: BTreeSet<String> = comb_candidates
        .iter()
        .map(|candidate| candidate.chunk.id.clone())
        .collect();

    let prefixes = store.ingest_block_prefixes.clone();
    let scored_pairs: Vec<(usize, String)> = pool
        .iter()
        .enumerate()
        .filter_map(|(index, chunk)| {
            let normalized = normalize_for_scoring(&chunk.content, &prefixes);
            (!normalized.is_empty()).then_some((index, normalized))
        })
        .collect();

    if scored_pairs.is_empty() {
        return Ok(Assembled {
            content: String::new(),
            token_count: 0,
            budget: budget_for(options, 0),
            selected_ids: Vec::new(),
            raw_scores: BTreeMap::new(),
            effective_scores: BTreeMap::new(),
            drift_detected: false,
            drift_score: 0.0,
            top_raw_score: 0.0,
            top_chunk_id: None,
            payload_dedup_skipped: 0,
            remembrance,
            refresh: BTreeMap::new(),
            relevance_updates: BTreeMap::new(),
        });
    }

    // Scoring pass: the exact normalized forms go through the same encoder, so
    // meta/tool control text cannot skew the query or win as a candidate.
    let mut raw_scores = BTreeMap::new();
    let mut raw_order: Vec<(String, f64)> = Vec::new();
    if let Some(overrides) = &options.score_override {
        for (index, _) in &scored_pairs {
            let chunk = &pool[*index];
            let score = overrides.get(&chunk.id).copied().unwrap_or(0.0);
            raw_scores.insert(chunk.id.clone(), score);
            raw_order.push((chunk.id.clone(), score));
        }
    } else {
        let scoring_query = normalize_for_scoring(query, &prefixes);
        let mut texts = Vec::with_capacity(scored_pairs.len() + 1);
        texts.push(scoring_query);
        texts.extend(scored_pairs.iter().map(|(_, text)| text.clone()));
        let vectors = encoder.encode(&texts)?;
        if vectors.len() != texts.len() {
            return Err(format!(
                "encoder returned {} vectors for {} texts",
                vectors.len(),
                texts.len()
            ));
        }
        let similarities = cosine_scores(&vectors[0], &vectors[1..]);
        for ((index, normalized), similarity) in scored_pairs.iter().zip(similarities) {
            let chunk = &pool[*index];
            let boost = if vocab.matches(normalized) {
                options.vocab_boost
            } else {
                0.0
            };
            let score = similarity + boost;
            raw_scores.insert(chunk.id.clone(), score);
            raw_order.push((chunk.id.clone(), score));
        }
    }
    let top_raw = raw_order
        .iter()
        .fold(None::<&(String, f64)>, |best, pair| match best {
            Some(current) if current.1 >= pair.1 => Some(current),
            _ => Some(pair),
        })
        .cloned();
    let (top_raw_score, top_chunk_id) = match top_raw {
        Some((id, score)) => (score, Some(id)),
        None => (0.0, None),
    };

    // Dedup embeddings use raw content (the reference store caches those).
    let mut embeddings = if options.skip_dedup {
        Vec::new()
    } else {
        let raw_texts: Vec<String> = scored_pairs
            .iter()
            .map(|(index, _)| pool[*index].content.clone())
            .collect();
        encoder.encode(&raw_texts)?
    };
    if !options.skip_dedup {
        for (slot, (index, _)) in scored_pairs.iter().enumerate() {
            if *index >= store_len
                && let Some(embedding) = &comb_candidates[*index - store_len].embedding
            {
                embeddings[slot] = embedding.clone();
            }
        }
    }
    let (survivors, refresh) = {
        let chunk_refs: Vec<&Chunk> = scored_pairs
            .iter()
            .map(|(index, _)| &pool[*index])
            .collect();
        if options.skip_dedup {
            ((0..chunk_refs.len()).collect(), BTreeMap::new())
        } else {
            let dedup = ContextDeduplicator::new(vocab, options.dedup_threshold);
            dedup.deduplicate(&chunk_refs, &embeddings)
        }
    };
    store.apply_refresh(&refresh);
    for (id, turn) in &refresh {
        if let Some(chunk) = pool.iter_mut().find(|chunk| &chunk.id == id) {
            chunk.last_referenced_turn = Some(chunk.last_referenced_turn.unwrap_or(0).max(*turn));
        }
    }
    let surviving: Vec<&Chunk> = survivors
        .iter()
        .map(|index| &pool[scored_pairs[*index].0])
        .collect();

    let recent = store.get_recent_chunks(3);
    let all_refs: Vec<&Chunk> = pool.iter().collect();
    let (drift_score, drift_detected) = if all_refs.len() >= 5 && !recent.is_empty() {
        let recent_text = recent
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let historical_text = all_refs[..all_refs.len() / 2]
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let drift_vectors = encoder.encode(&[recent_text, historical_text])?;
        let score = 1.0 - crate::cosine(&drift_vectors[0], &drift_vectors[1]);
        (score, score > options.drift_threshold)
    } else {
        (0.0, false)
    };

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
        if options.drift_prior.eq_ignore_ascii_case("downweight") && !query.is_empty() {
            let query_words = content_words(query);
            for chunk in &surviving {
                if !recent_ids.contains(&chunk.id.as_str()) || chunk.content.is_empty() {
                    continue;
                }
                let chunk_words = content_words(&chunk.content);
                if !chunk_words.is_empty()
                    && chunk_words.intersection(&query_words).count() as f64
                        / chunk_words.len() as f64
                        >= 0.8
                {
                    penalties.insert(chunk.id.clone(), options.drift_prior_factor);
                }
            }
        }
    }

    let exempt: Vec<String> = comb_ids.iter().cloned().collect();
    let mut effective = apply_decay(
        &surviving,
        options.current_turn,
        &raw_scores,
        &penalties,
        &exempt,
        options.stale_threshold,
        options.stale_factor,
    );

    let mut payload_dedup_skipped = 0;
    if options.dedup_against_payload && !options.payload_fingerprints.is_empty() {
        let fingerprints: BTreeSet<&str> = options
            .payload_fingerprints
            .iter()
            .map(String::as_str)
            .collect();
        let before = effective.len();
        effective.retain(|id, _| {
            pool.iter()
                .find(|chunk| &chunk.id == id)
                .map(|chunk| !fingerprints.contains(chunk.fingerprint.as_str()))
                .unwrap_or(true)
        });
        payload_dedup_skipped = before - effective.len();
    }

    let high_relevance = effective.values().filter(|score| **score > 0.6).count();
    let budget = options.budget_override.unwrap_or_else(|| {
        AdaptiveBudget::default().compute(&options.route, high_relevance, options.max_context)
    });

    let mut scored: Vec<(&Chunk, f64)> = surviving
        .iter()
        .filter(|chunk| effective.contains_key(&chunk.id))
        .map(|chunk| (*chunk, effective.get(&chunk.id).copied().unwrap_or(0.0)))
        .collect();
    scored.sort_by(|left, right| right.1.total_cmp(&left.1));

    let share_cap = (budget as f64 * options.max_chunk_share) as u32;
    let mut selected: Vec<Chunk> = Vec::new();
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

    // Selection is curation: record it on the store chunk so the surplus tier
    // (comb) can distinguish what the splinter once judged relevant.
    let mut relevance_updates = BTreeMap::new();
    for chunk in &selected {
        if comb_ids.contains(&chunk.id) {
            continue;
        }
        if let Some(target) = store.chunks.iter_mut().find(|item| item.id == chunk.id) {
            target.relevance_history.push((
                options.current_turn,
                (raw_scores.get(&chunk.id).copied().unwrap_or(0.0) * 1000.0).round() / 1000.0,
            ));
            if target.relevance_history.len() > 10 {
                let excess = target.relevance_history.len() - 10;
                target.relevance_history.drain(0..excess);
            }
            relevance_updates.insert(target.id.clone(), target.relevance_history.clone());
        }
    }

    let content = selected
        .iter()
        .map(|chunk| normalize_codec(&chunk.content))
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
        top_raw_score,
        top_chunk_id,
        payload_dedup_skipped,
        remembrance,
        refresh,
        relevance_updates,
    })
}

fn budget_for(options: &AssembleOptions, high_relevance: usize) -> u32 {
    options.budget_override.unwrap_or_else(|| {
        AdaptiveBudget::default().compute(&options.route, high_relevance, options.max_context)
    })
}

fn run_remembrance(
    store: &mut ContextStore,
    query: &str,
    encoder: &mut dyn Embedder,
    vocab: &Vocabulary,
    vocab_boost: f64,
) -> Result<Vec<RemembranceResult>, String> {
    let ids: Vec<String> = store
        .get_deletion_candidates()
        .iter()
        .map(|chunk| chunk.id.clone())
        .collect();
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut texts = Vec::with_capacity(ids.len() + 1);
    texts.push(query.to_owned());
    let mut indices = Vec::with_capacity(ids.len());
    for id in &ids {
        if let Some(index) = store.chunks.iter().position(|chunk| &chunk.id == id) {
            texts.push(store.chunks[index].content.clone());
            indices.push(index);
        }
    }
    let vectors = encoder.encode(&texts)?;
    if vectors.len() != texts.len() {
        return Err(format!(
            "encoder returned {} vectors for {} remembrance texts",
            vectors.len(),
            texts.len()
        ));
    }
    let similarities = cosine_scores(&vectors[0], &vectors[1..]);
    let pass = RemembrancePass::new(vocab);
    let mut results = Vec::with_capacity(indices.len());
    for (index, similarity) in indices.into_iter().zip(similarities) {
        let boost = if vocab.matches(&store.chunks[index].content) {
            vocab_boost
        } else {
            0.0
        };
        results.push(pass.process_one(&mut store.chunks[index], similarity + boost));
    }
    Ok(results)
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

/// The reference `focal.assembly.normalize_for_scoring`: strip harness control
/// lines and expand contractions so both sides of the similarity compare use
/// the same form.
pub fn normalize_for_scoring(text: &str, prefixes: &[String]) -> String {
    if text.trim().is_empty() {
        return String::new();
    }
    let stripped = crate::hygiene::strip_boilerplate(text, prefixes);
    if stripped.trim().is_empty() {
        return String::new();
    }
    let mut out = stripped;
    for (pattern, replacement) in contraction_patterns() {
        out = pattern.replace_all(&out, *replacement).into_owned();
    }
    out.trim().to_owned()
}

fn contraction_patterns() -> &'static [(Regex, &'static str)] {
    use std::sync::OnceLock;
    static PATTERNS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        [
            ("don't", "do not"),
            ("don’t", "do not"),
            ("can't", "cannot"),
            ("can’t", "cannot"),
            ("couldn't", "could not"),
            ("couldn’t", "could not"),
            ("wouldn't", "would not"),
            ("wouldn’t", "would not"),
            ("i'm", "i am"),
            ("i’m", "i am"),
            ("i've", "i have"),
            ("i’ve", "i have"),
            ("won't", "will not"),
            ("won’t", "will not"),
            ("it's", "it is"),
            ("it’s", "it is"),
        ]
        .into_iter()
        .filter_map(|(short, long)| {
            Regex::new(&format!("(?i){}", regex::escape(short)))
                .ok()
                .map(|pattern| (pattern, long))
        })
        .collect()
    })
}

fn content_words(text: &str) -> BTreeSet<String> {
    let word = Regex::new(r"[a-z0-9]{4,}").expect("word regex");
    word.find_iter(&text.to_lowercase())
        .map(|found| found.as_str().to_owned())
        .collect()
}
