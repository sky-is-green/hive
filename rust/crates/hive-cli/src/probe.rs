//! One cheap liveness + model check against an OpenAI-compatible base.

use std::time::Duration;

use hive_core::http;
use serde_json::Value;

/// Return `{"up": bool, "model": ...}` or `{"up": false, "error": ...}`.
pub fn probe(base_url: &str, timeout: Duration) -> Value {
    if base_url.trim().is_empty() {
        return down("no endpoint");
    }
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let response = match http::get(&url, timeout) {
        Ok(response) => response,
        Err(error) => return down(&error),
    };
    if !(200..300).contains(&response.status) {
        return down(&format!(
            "HTTP Error {}: {}",
            response.status, response.reason
        ));
    }
    let data: Value = match response.json() {
        Ok(data) => data,
        Err(error) => return down(&error),
    };
    let model = data
        .get("data")
        .and_then(Value::as_array)
        .and_then(|items| {
            items
                .iter()
                .find_map(|item| item.get("id").and_then(Value::as_str))
        })
        .map(|id| Value::String(id.to_owned()))
        .unwrap_or(Value::Null);
    let mut out = serde_json::Map::new();
    out.insert("up".into(), Value::Bool(true));
    out.insert("model".into(), model);
    Value::Object(out)
}

fn down(error: &str) -> Value {
    let mut out = serde_json::Map::new();
    out.insert("up".into(), Value::Bool(false));
    out.insert(
        "error".into(),
        Value::String(error.chars().take(160).collect()),
    );
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_base_is_down_without_connecting() {
        assert_eq!(probe("", Duration::from_millis(50))["error"], "no endpoint");
    }
}
