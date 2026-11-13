//! UTF-8 / legacy-codec repair, ported from `retention/codec.py`.

const MAX_PASSES: usize = 2;

pub fn normalize_codec(text: &str) -> String {
    if text.is_empty() || !text.chars().any(is_suspect) {
        return text.to_owned();
    }
    let mut current = text.to_owned();
    loop {
        let foreign = segmented_repair(&current, is_foreign);
        let all_high = segmented_repair(&current, |ch| ch as u32 > 0xFF);
        let best = if corruption_score(&foreign) <= corruption_score(&all_high) {
            foreign
        } else {
            all_high
        };
        if corruption_score(&best) >= corruption_score(&current) {
            break;
        }
        current = best;
    }
    current
}

fn is_suspect(ch: char) -> bool {
    matches!(ch as u32, 0x80..=0x9F) || matches!(ch, 'Â' | 'Ã' | 'â')
}

fn is_foreign(ch: char) -> bool {
    (ch as u32) > 0xFF && !cp1252_c2().contains(&ch)
}

fn corruption_score(text: &str) -> usize {
    text.chars().filter(|ch| is_suspect(*ch)).count() + text.matches('\u{FFFD}').count()
}

fn segmented_repair<F>(text: &str, separator: F) -> String
where
    F: Fn(char) -> bool,
{
    let mut out = String::with_capacity(text.len());
    let mut buffer = String::new();
    for ch in text.chars() {
        if separator(ch) {
            if !buffer.is_empty() {
                out.push_str(&repair_to_fixed_point(&buffer));
                buffer.clear();
            }
            out.push(ch);
        } else {
            buffer.push(ch);
        }
    }
    if !buffer.is_empty() {
        out.push_str(&repair_to_fixed_point(&buffer));
    }
    out
}

fn repair_to_fixed_point(text: &str) -> String {
    let mut current = text.to_owned();
    loop {
        let better = best_within_depth(&current);
        if better == current {
            break;
        }
        current = better;
    }
    current
}

fn best_within_depth(text: &str) -> String {
    let mut best = text.to_owned();
    let mut best_score = corruption_score(text);
    let mut frontier = vec![text.to_owned()];
    let mut seen = std::collections::HashSet::new();
    seen.insert(text.to_owned());
    for _ in 0..MAX_PASSES {
        let mut next = Vec::new();
        for node in &frontier {
            for candidate in [reverse_cp1252(node), reverse_latin1(node)]
                .into_iter()
                .flatten()
            {
                if candidate == *node || !seen.insert(candidate.clone()) {
                    continue;
                }
                let score = corruption_score(&candidate);
                next.push(candidate.clone());
                if score < best_score {
                    best = candidate;
                    best_score = score;
                }
            }
        }
        frontier = next;
    }
    best
}

fn reverse_cp1252(text: &str) -> Option<String> {
    let (bytes, _, errors) = encoding_rs::WINDOWS_1252.encode(text);
    if errors {
        return None;
    }
    String::from_utf8(bytes.into_owned()).ok()
}

fn reverse_latin1(text: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let value = ch as u32;
        if value > 0xFF {
            return None;
        }
        bytes.push(value as u8);
    }
    String::from_utf8(bytes).ok()
}

fn cp1252_c2() -> std::collections::HashSet<char> {
    (0x80u8..0xA0)
        .filter_map(|byte| {
            let bytes = [byte];
            let (decoded, _, errors) = encoding_rs::WINDOWS_1252.decode(&bytes);
            if errors { None } else { decoded.chars().next() }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn misdecode(text: &str, codec: &str) -> String {
        // Simulate the Python test helper for the two legacy encodings.
        match codec {
            "cp1252" => {
                let (decoded, _, _) = encoding_rs::WINDOWS_1252.decode(text.as_bytes());
                decoded.into_owned()
            }
            "latin-1" => text.as_bytes().iter().map(|byte| *byte as char).collect(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn repairs_emdash_cp1252_and_arrow_latin1() {
        for original in [
            "Fix applied — verified this round.",
            "ingest → store → re-inject",
        ] {
            for codec in ["cp1252", "latin-1"] {
                let mangled = misdecode(original, codec);
                assert_eq!(normalize_codec(&mangled), original, "{codec}: {mangled:?}");
            }
        }
    }

    #[test]
    fn clean_text_is_unchanged_and_idempotent() {
        for text in [
            "plain ascii",
            "café naïve château — done → next © “fine” • ok",
            "mâchoire rêve cœur",
        ] {
            assert_eq!(normalize_codec(text), text);
            assert_eq!(normalize_codec(&normalize_codec(text)), text);
        }
    }
}
