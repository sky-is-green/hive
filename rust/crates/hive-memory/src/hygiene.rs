//! Write-side retention hygiene, ported from `retention/hygiene.py`.

use regex::Regex;

use crate::codec::normalize_codec;

pub const PREFIX_WINDOW: usize = 80;
pub const DEFAULT_MAX_CHUNK_CHARS: usize = 4000;
pub const TRUNCATION_MARK: &str = "\n…[truncated]";

pub fn default_ingest_block_prefixes() -> [&'static str; 2] {
    [
        "You have access to enabled tools. If a tool is needed to satisfy the user's request or complete the action you described, call ed...",
        "You have used all available tool calls. Based on everything you have found so far, provide your final answer now. Do not call any...",
    ]
}

pub fn is_boilerplate_line(line: &str, prefixes: &[String]) -> bool {
    let stripped = line.trim_start();
    prefixes
        .iter()
        .filter(|prefix| !prefix.is_empty())
        .any(|prefix| stripped.starts_with(&prefix[..prefix.len().min(PREFIX_WINDOW)]))
}

pub fn strip_boilerplate(text: &str, prefixes: &[String]) -> String {
    if prefixes.is_empty() || text.trim().is_empty() {
        return if text.trim().is_empty() {
            String::new()
        } else {
            text.to_owned()
        };
    }
    let lines: Vec<&str> = text.lines().collect();
    let kept: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|line| !is_boilerplate_line(line, prefixes))
        .collect();
    if kept.len() == lines.len() {
        return text.to_owned();
    }
    let remainder = kept.join("\n");
    remainder.trim().to_owned()
}

pub fn content_fingerprint(text: &str) -> String {
    format!("{:x}", md5::compute(text.as_bytes()))[..12].to_owned()
}

pub fn sanitize_for_storage(text: &str, max_chars: usize) -> String {
    let mut out = text.to_owned();
    for (pattern, replacement) in secret_rules() {
        out = pattern.replace_all(&out, replacement).into_owned();
    }
    let base64 = Regex::new(r"[A-Za-z0-9+/]{256,}={0,2}").expect("base64 regex");
    out = base64
        .replace_all(&out, "[base64 blob stripped]")
        .into_owned();
    let count = out.chars().count();
    if count > max_chars {
        out = out.chars().take(max_chars).collect::<String>() + TRUNCATION_MARK;
    }
    out
}

pub fn prepare_for_storage(
    text: &str,
    max_chars: usize,
    block_prefixes: &[String],
    sanitize: bool,
) -> Option<String> {
    let cleaned = normalize_codec(text);
    let cleaned = strip_boilerplate(&cleaned, block_prefixes);
    if cleaned.trim().is_empty() {
        return None;
    }
    Some(if sanitize {
        sanitize_for_storage(&cleaned, max_chars)
    } else {
        cleaned
    })
}

fn secret_rules() -> Vec<(Regex, &'static str)> {
    vec![
        (
            Regex::new(r"\bsk-[A-Za-z0-9_-]{16,}\b").expect("openai key"),
            "[redacted-secret]",
        ),
        (
            Regex::new(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b").expect("github token"),
            "[redacted-secret]",
        ),
        (
            Regex::new(r"\bAKIA[0-9A-Z]{16}\b").expect("aws key"),
            "[redacted-secret]",
        ),
        (
            Regex::new(r"(?i)\b(authorization\s*[:=]\s*)bearer\s+[A-Za-z0-9._~+/=-]+")
                .expect("bearer"),
            "$1[redacted]",
        ),
        (
            Regex::new(
                r#"(?i)\b(api[_-]?key|apikey|passwd|password|secret|token)(\s*[:=]\s*)("[^"]{4,}"|'[^']{4,}'|[^\s,;"']{4,})"#,
            )
            .expect("assignment secret"),
            "$1$2[redacted]",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boilerplate_is_dropped_and_secrets_are_redacted() {
        let prefixes: Vec<String> = default_ingest_block_prefixes()
            .iter()
            .map(|prefix| (*prefix).to_owned())
            .collect();
        let text = format!("{}\nReal question about the GPU.", prefixes[0]);
        assert_eq!(
            prepare_for_storage(&text, 4000, &prefixes, true).as_deref(),
            Some("Real question about the GPU.")
        );
        let secret = "key: sk-abcdef1234567890abcdef ok";
        let out = prepare_for_storage(secret, 4000, &prefixes, true).unwrap();
        assert!(!out.contains("sk-abcdef"));
        assert!(out.contains("[redacted-secret]"));
    }
}
