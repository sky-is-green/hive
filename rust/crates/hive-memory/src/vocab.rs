//! Domain vocabulary subset used by dedup/remembrance scoring.

use std::path::Path;

use regex::Regex;

pub struct Vocabulary {
    terms: Vec<String>,
    pattern: Option<Regex>,
}

impl Vocabulary {
    pub fn new(terms: impl IntoIterator<Item = String>) -> Self {
        let mut terms: Vec<String> = terms
            .into_iter()
            .map(|term| term.trim().to_lowercase())
            .filter(|term| !term.is_empty())
            .collect();
        terms.sort_by_key(|term| std::cmp::Reverse(term.len()));
        terms.dedup();
        let pattern = if terms.is_empty() {
            None
        } else {
            let alternation = terms
                .iter()
                .map(|term| regex::escape(term))
                .collect::<Vec<_>>()
                .join("|");
            Regex::new(&format!("(?i){alternation}")).ok()
        };
        Self { terms, pattern }
    }

    pub fn load(dir: &Path, domains: &[&str]) -> Result<Self, String> {
        let mut terms = Vec::new();
        for domain in domains {
            let path = dir.join(format!("{domain}.json"));
            let text = std::fs::read_to_string(&path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            let data: serde_json::Value = serde_json::from_str(&text)
                .map_err(|error| format!("invalid {}: {error}", path.display()))?;
            collect_terms(&data, &mut terms);
        }
        Ok(Self::new(terms))
    }

    pub fn matches(&self, text: &str) -> bool {
        self.pattern
            .as_ref()
            .is_some_and(|pattern| pattern.is_match(text))
    }

    pub fn matched_terms(&self, text: &str) -> Vec<String> {
        let Some(pattern) = &self.pattern else {
            return Vec::new();
        };
        let mut out: Vec<String> = Vec::new();
        for found in pattern.find_iter(text) {
            let term = found.as_str().to_lowercase();
            if !out.contains(&term) {
                out.push(term);
            }
        }
        out
    }

    pub fn size(&self) -> usize {
        self.terms.len()
    }
}

fn collect_terms(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                if let Some(text) = item.as_str() {
                    out.push(text.to_owned());
                }
            }
        }
        serde_json::Value::Object(map) => {
            for (key, item) in map {
                if key == "domain" || key == "description" {
                    continue;
                }
                if let Some(text) = item.as_str() {
                    out.push(text.to_owned());
                } else if item.is_array() {
                    collect_terms(item, out);
                }
            }
        }
        _ => {}
    }
}
