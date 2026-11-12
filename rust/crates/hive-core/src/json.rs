//! JSON output matching Python's `json.dumps(..., indent=2, sort_keys=True)`.
//!
//! `serde_json` sorts object keys by default (its map is a `BTreeMap`), so the
//! only difference left is non-ASCII escaping: Python keeps `ensure_ascii=True`
//! unless told otherwise, while serde writes UTF-8. Escaping the non-ASCII
//! characters after serialising is valid JSON and makes the bytes match.

use serde_json::Value;

/// Pretty JSON with a trailing newline, like `print(json.dumps(...))`.
pub fn pretty(value: &Value) -> String {
    let mut text = to_string_pretty(value);
    text.push('\n');
    text
}

/// Pretty JSON without a trailing newline, ASCII-only.
pub fn to_string_pretty(value: &Value) -> String {
    let raw = serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".into());
    escape_non_ascii(&raw)
}

/// Escape every non-ASCII scalar as `\uXXXX` (surrogate pairs above the BMP).
fn escape_non_ascii(input: &str) -> String {
    if input.is_ascii() {
        return input.to_owned();
    }
    let mut out = String::with_capacity(input.len() + 16);
    for ch in input.chars() {
        let code = ch as u32;
        if code < 0x80 {
            out.push(ch);
        } else if code <= 0xFFFF {
            out.push_str(&format!("\\u{code:04x}"));
        } else {
            let adjusted = code - 0x1_0000;
            let high = 0xD800 + (adjusted >> 10);
            let low = 0xDC00 + (adjusted & 0x3FF);
            out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sorts_keys_and_escapes_non_ascii() {
        let value = json!({"b": "·", "a": 1});
        assert_eq!(
            to_string_pretty(&value),
            "{\n  \"a\": 1,\n  \"b\": \"\\u00b7\"\n}"
        );
    }

    #[test]
    fn empty_containers_match_python() {
        assert_eq!(to_string_pretty(&json!([])), "[]");
        assert_eq!(to_string_pretty(&json!({})), "{}");
    }
}
