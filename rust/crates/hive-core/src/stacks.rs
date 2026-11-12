//! Saved stack summaries and the shape checks the CLI can do without hardware.
//!
//! The full stack schema (tier launch blocks, endpoint URLs, engine args) lives
//! in `harness/stack/schema.py`. The Rust CLI needs the same on-disk contract;
//! this module implements the stack/tier object shape and the `stacks list`
//! summary. Anything it cannot parse is skipped from the list, exactly like
//! the Python route.

use std::fs;
use std::path::Path;

use serde_json::{Map, Value};

use crate::error::Result;

const STACK_FIELDS: &[&str] = &["name", "version", "tiers", "routing"];
const TIER_FIELDS: &[&str] = &[
    "role",
    "repo",
    "file",
    "ctx",
    "ngl",
    "backend",
    "cache_k",
    "cache_v",
    "spec",
    "mmproj",
    "pin",
    "ts",
    "engine_args",
    "endpoint",
    "caps",
    "launch",
    "expect_model",
];
const ROLES: &[&str] = &["face", "worker", "agency", "mechanics"];

/// One `{"name","tiers","roles"}` row from `stacks list`.
#[derive(Debug, Clone)]
pub struct StackSummary {
    pub name: String,
    pub tiers: usize,
    pub roles: Vec<String>,
}

impl StackSummary {
    pub fn to_json(&self) -> Value {
        let mut object = Map::new();
        object.insert("name".into(), Value::String(self.name.clone()));
        object.insert("tiers".into(), Value::from(self.tiers as u64));
        object.insert(
            "roles".into(),
            Value::Array(self.roles.iter().cloned().map(Value::String).collect()),
        );
        Value::Object(object)
    }
}

/// Read one stack document, applying the same object/field checks as Python's
/// `Stack.from_dict`. Shape checks that need no I/O are included.
pub fn parse_stack(data: &Value) -> Result<Value> {
    let object = data
        .as_object()
        .ok_or_else(|| crate::HiveError::invalid("stack must be an object"))?;
    reject_unknown(object, STACK_FIELDS, "stack")?;
    let name = text(object.get("name"), "stack name")?;
    let version = integer(
        object.get("version").unwrap_or(&Value::from(1)),
        "stack version",
        1,
    )?;
    if version != 1 {
        return Err(crate::HiveError::invalid(format!(
            "unsupported stack version {version}; this build reads version 1"
        )));
    }
    let raw_tiers = match object.get("tiers") {
        None | Some(Value::Null) => &[] as &[Value],
        Some(Value::Array(items)) => items.as_slice(),
        Some(_) => return Err(crate::HiveError::invalid("stack tiers must be a list")),
    };
    let mut tiers = Vec::with_capacity(raw_tiers.len());
    for tier in raw_tiers {
        tiers.push(parse_tier(tier)?);
    }
    if let Some(routing) = object.get("routing")
        && !routing.is_null()
        && !routing.is_object()
    {
        return Err(crate::HiveError::invalid("stack routing must be an object"));
    }
    let mut out = object.clone();
    out.insert("name".into(), Value::String(name));
    out.insert("version".into(), Value::from(version));
    out.insert("tiers".into(), Value::Array(tiers));
    if !out.contains_key("routing") {
        let mut routing = Map::new();
        routing.insert("workers_as".into(), Value::String("subagent".into()));
        out.insert("routing".into(), Value::Object(routing));
    }
    Ok(Value::Object(out))
}

fn parse_tier(data: &Value) -> Result<Value> {
    let object = data
        .as_object()
        .ok_or_else(|| crate::HiveError::invalid("tier must be an object"))?;
    reject_unknown(object, TIER_FIELDS, "tier")?;
    for key in ["role", "repo", "file"] {
        text(object.get(key), &format!("tier {key}"))?;
    }
    if let Some(ctx) = object.get("ctx") {
        integer(ctx, "tier ctx", 1)?;
    }
    if let Some(ngl) = object.get("ngl") {
        integer(ngl, "tier ngl", 0)?;
    }
    if let Some(endpoint) = object.get("endpoint") {
        if !endpoint.is_null() && endpoint.as_str().is_none() {
            return Err(crate::HiveError::invalid("tier endpoint must be a string"));
        }
        if let Some(text) = endpoint.as_str()
            && !text.is_empty()
            && !(text.starts_with("http://") || text.starts_with("https://"))
        {
            return Err(crate::HiveError::invalid(
                "tier endpoint must be an http(s) URL with a host",
            ));
        }
    }
    Ok(Value::Object(object.clone()))
}

fn reject_unknown(object: &Map<String, Value>, known: &[&str], where_: &str) -> Result<()> {
    let mut unknown: Vec<&str> = object
        .keys()
        .map(String::as_str)
        .filter(|key| !known.contains(key))
        .collect();
    if !unknown.is_empty() {
        unknown.sort_unstable();
        return Err(crate::HiveError::invalid(format!(
            "{where_} has unknown field(s): {}; known: {}",
            unknown.join(", "),
            known.join(", ")
        )));
    }
    Ok(())
}

fn text(value: Option<&Value>, field: &str) -> Result<String> {
    let Some(value) = value else {
        return Err(crate::HiveError::invalid(format!("{field} is missing")));
    };
    let text = value
        .as_str()
        .ok_or_else(|| crate::HiveError::invalid(format!("{field} must be a string")))?;
    let text = text.trim();
    if text.is_empty() {
        return Err(crate::HiveError::invalid(format!(
            "{field} must not be empty"
        )));
    }
    Ok(text.to_owned())
}

fn integer(value: &Value, field: &str, minimum: i64) -> Result<i64> {
    let parsed = if let Some(number) = value.as_i64() {
        number
    } else if let Some(text) = value.as_str() {
        text.trim()
            .parse::<i64>()
            .map_err(|_| crate::HiveError::invalid(format!("{field} must be an integer")))?
    } else {
        return Err(crate::HiveError::invalid(format!(
            "{field} must be an integer"
        )));
    };
    if parsed < minimum {
        return Err(crate::HiveError::invalid(format!(
            "{field} must be >= {minimum}, got {parsed}"
        )));
    }
    Ok(parsed)
}

/// A stack name is a bare filename stem (the `stacks/<name>.json` convention).
pub fn check_stack_name(name: &str) -> std::result::Result<String, String> {
    let mut stem = name.trim();
    if stem.is_empty() {
        return Err("stack name must not be empty".into());
    }
    if let Some(stripped) = stem.strip_suffix(".json") {
        stem = stripped.trim();
    }
    if stem.is_empty() || stem == "." || stem == ".." || stem.starts_with('.') {
        return Err(format!(
            "stack name must be a bare filename, got {}",
            crate::py_repr(name)
        ));
    }
    if stem.contains('/') || stem.contains('\\') || stem.contains('\0') {
        return Err(format!(
            "stack name must be a bare filename, got {}",
            crate::py_repr(name)
        ));
    }
    Ok(stem.to_owned())
}

/// Structural errors only: `[]` means the shape is valid.
pub fn validate_shape(stack: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let Some(object) = stack.as_object() else {
        return vec!["expected a stack object".into()];
    };
    if let Some(name) = object.get("name").and_then(Value::as_str) {
        if let Err(error) = check_stack_name(name) {
            errors.push(error);
        }
    } else {
        errors.push("stack name is missing".into());
    }
    if let Some(routing) = object.get("routing")
        && !routing.is_object()
    {
        errors.push(format!(
            "routing must be an object, got {}",
            json_type(routing)
        ));
    }
    let tiers = match object.get("tiers") {
        Some(Value::Array(tiers)) => tiers,
        None | Some(Value::Null) => {
            errors.push("stack has no tiers (every stack needs a face tier)".into());
            return errors;
        }
        Some(other) => {
            errors.push(format!(
                "stack tiers must be a list, got {}",
                json_type(other)
            ));
            return errors;
        }
    };
    if tiers.is_empty() {
        errors.push("stack has no tiers (every stack needs a face tier)".into());
        return errors;
    }
    let mut seen: Vec<(String, usize)> = Vec::new();
    for (index, tier) in tiers.iter().enumerate() {
        let position = index + 1;
        let label = format!("tier {position}");
        let role = tier.get("role").and_then(Value::as_str).unwrap_or("");
        if role.trim().is_empty() {
            errors.push(format!("{label}: missing tier role"));
        } else if !ROLES.contains(&role.trim()) {
            errors.push(format!(
                "{label}: role {} is not one of {}",
                crate::py_repr(role),
                ROLES.join(", ")
            ));
        } else if let Some((_, first)) = seen.iter().find(|(name, _)| name == role.trim()) {
            errors.push(format!(
                "{label}: duplicate tier role {} (already used by tier {first})",
                crate::py_repr(role)
            ));
        } else {
            seen.push((role.trim().to_owned(), position));
        }
        for key in ["repo", "file"] {
            if tier
                .get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .is_none()
            {
                errors.push(format!("{label}: missing {key}"));
            }
        }
        match tier.get("ctx") {
            None => {}
            Some(value) => match value.as_i64() {
                Some(ctx) if ctx > 0 => {}
                _ => errors.push(format!(
                    "{label}: ctx must be a positive integer, got {value}"
                )),
            },
        }
        if let Some(launch) = tier.get("launch")
            && !launch.is_null()
        {
            if !launch.is_object() {
                errors.push(format!(
                    "{label}: launch must be an object, got {}",
                    json_type(launch)
                ));
            } else if tier
                .get("endpoint")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .is_none()
            {
                errors.push(format!(
                    "{label}: launch requires an endpoint (the contract it brings up)"
                ));
            }
        }
    }
    errors
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// `stacks list` rows, sorted by name. Unreadable or malformed files are skipped.
pub fn list_summaries(directory: &Path) -> Vec<StackSummary> {
    if !directory.is_dir() {
        return Vec::new();
    }
    let mut files: Vec<_> = match fs::read_dir(directory) {
        Ok(entries) => entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
            .collect(),
        Err(_) => return Vec::new(),
    };
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    let mut rows = Vec::new();
    for path in files {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(data) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let Ok(stack) = parse_stack(&data) else {
            continue;
        };
        let roles = stack
            .get("tiers")
            .and_then(Value::as_array)
            .map(|tiers| {
                tiers
                    .iter()
                    .filter_map(|tier| tier.get("role").and_then(Value::as_str))
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        rows.push(StackSummary {
            name: stack
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            tiers: stack
                .get("tiers")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0),
            roles,
        });
    }
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_a_minimal_stack() {
        let data = json!({
            "name": "peer-2tier",
            "version": 1,
            "tiers": [
                {"role": "face", "repo": "org/repo", "file": "a.gguf"},
                {"role": "worker", "repo": "org/repo", "file": "b.gguf"}
            ]
        });
        let stack = parse_stack(&data).unwrap();
        assert_eq!(stack["tiers"].as_array().unwrap().len(), 2);
        assert_eq!(stack["routing"]["workers_as"], "subagent");
    }

    #[test]
    fn rejects_unknown_fields_and_duplicate_shapes() {
        let data = json!({"name": "x", "version": 1, "tiers": [], "typo": true});
        assert!(parse_stack(&data).is_err());
    }

    #[test]
    fn a_bad_role_is_refused_by_the_shape_check() {
        let data = json!({
            "name": "x", "version": 1,
            "tiers": [{"role": "boss", "repo": "r", "file": "f"}]
        });
        let stack = parse_stack(&data).unwrap();
        assert!(!validate_shape(&stack).is_empty());
    }
}
