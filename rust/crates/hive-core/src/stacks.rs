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

const LAUNCH_KINDS: &[&str] = &["strata", "deepinfer"];
const LAUNCH_FIELDS: &[&str] = &[
    "kind",
    "repo",
    "config",
    "config_external",
    "python",
    "server",
    "engine_args",
    "options",
    "timeout_s",
    "ctx_flag",
];
const LAUNCH_ARGS_DENIED: &[&str] = &[
    "--engine",
    "--config",
    "--host",
    "--port",
    "--api-key",
    "--open",
    "--script",
    "--mcp-config",
    "--lazy",
];
const LAUNCH_OPTIONS_DENIED: &[&str] = &["args", "exe", "cwd", "port", "log"];
const ENGINE_ARGS_ALLOWED: &[&str] = &[
    "--device",
    "-dev",
    "--n-cpu-moe",
    "-ncmoe",
    "--cpu-moe",
    "-cmoe",
    "--n-cpu-ffn",
    "-ncffn",
    "--moe-slot-cache",
    "-msc",
    "--lazy-mode",
    "-lzm",
    "--flash-attn",
    "-fa",
    "--threads",
    "-t",
    "--threads-batch",
    "-tb",
    "--parallel",
    "-np",
    "--load-mode",
    "-lm",
    "--batch-size",
    "-b",
    "--ubatch-size",
    "-ub",
    "--split-mode",
    "-sm",
    "--main-gpu",
    "-mg",
    "--n-gpu-layers",
    "-ngl",
    "--no-mmap",
    "--mlock",
    "--no-op-offload",
    "--op-offload",
    "--model-draft",
    "-md",
    "--spec-draft-n-max",
    "--draft-max",
];

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
    if let Some(spec) = object.get("spec")
        && !spec.is_null()
        && !spec.is_object()
    {
        return Err(crate::HiveError::invalid(format!(
            "tier spec must be an object, got {}",
            json_type(spec)
        )));
    }
    if let Some(caps) = object.get("caps")
        && !caps.is_null()
        && !caps.as_str().map(str::is_empty).unwrap_or(false)
        && !caps.is_object()
    {
        return Err(crate::HiveError::invalid(format!(
            "tier caps must be an object, got {}",
            json_type(caps)
        )));
    }
    if let Some(endpoint) = object.get("endpoint") {
        if !endpoint.is_null() && endpoint.as_str().is_none() {
            return Err(crate::HiveError::invalid("tier endpoint must be a string"));
        }
        if let Some(text) = endpoint.as_str() {
            let text = text.trim();
            if !text.is_empty() && !valid_endpoint(text) {
                return Err(crate::HiveError::invalid(format!(
                    "tier endpoint must be an http(s) URL with a host, got {}",
                    crate::py_repr(text)
                )));
            }
        }
    }
    let mut out = object.clone();
    if let Some(engine_args) = object.get("engine_args")
        && !engine_args.is_null()
    {
        out.insert(
            "engine_args".into(),
            Value::Array(
                validate_engine_args(engine_args, "tier engine_args")?
                    .into_iter()
                    .map(Value::String)
                    .collect(),
            ),
        );
    }
    if let Some(launch) = object.get("launch")
        && !launch.is_null()
    {
        out.insert("launch".into(), validate_launch(launch, "tier launch")?);
    }
    Ok(Value::Object(out))
}

fn optional_text(value: Option<&Value>, field: &str) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => {
            let text = text.trim();
            if text.is_empty() {
                Ok(None)
            } else {
                Ok(Some(text.to_owned()))
            }
        }
        Some(other) => Err(crate::HiveError::invalid(format!(
            "{field} must be a string, got {}",
            json_type(other)
        ))),
    }
}

fn validate_engine_args(value: &Value, where_: &str) -> Result<Vec<String>> {
    let items = value.as_array().ok_or_else(|| {
        crate::HiveError::invalid(format!(
            "{where_} must be a list of strings, got {}",
            json_type(value)
        ))
    })?;
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let position = index + 1;
        let Some(text) = item.as_str() else {
            return Err(crate::HiveError::invalid(format!(
                "{where_}[{position}] must be a non-blank string, got {}",
                display_value(item)
            )));
        };
        let token = text.trim();
        if token.is_empty() {
            return Err(crate::HiveError::invalid(format!(
                "{where_}[{position}] must be a non-blank string, got {}",
                crate::py_repr(text)
            )));
        }
        if token.starts_with('-') && !is_numeric(token) && !ENGINE_ARGS_ALLOWED.contains(&token) {
            return Err(crate::HiveError::invalid(format!(
                "{where_}[{position}]: {} is not an allowed engine flag; allowed: {}",
                crate::py_repr(token),
                ENGINE_ARGS_ALLOWED.join(", ")
            )));
        }
        out.push(token.to_owned());
    }
    Ok(out)
}

fn validate_launch_args(value: &Value) -> Result<Vec<String>> {
    let where_ = "tier launch.engine_args";
    let items = value.as_array().ok_or_else(|| {
        crate::HiveError::invalid(format!(
            "{where_} must be a list of strings, got {}",
            json_type(value)
        ))
    })?;
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let position = index + 1;
        let Some(text) = item.as_str() else {
            return Err(crate::HiveError::invalid(format!(
                "{where_}[{position}] must be a non-blank string, got {}",
                display_value(item)
            )));
        };
        let token = text.trim();
        if token.is_empty() {
            return Err(crate::HiveError::invalid(format!(
                "{where_}[{position}] must be a non-blank string, got {}",
                crate::py_repr(text)
            )));
        }
        let flag = if token.starts_with('-') {
            token.split_once('=').map(|(flag, _)| flag).unwrap_or(token)
        } else {
            token
        };
        if LAUNCH_ARGS_DENIED.contains(&flag) {
            return Err(crate::HiveError::invalid(format!(
                "{where_}[{position}]: {} is owned by the managed launch and cannot be set per tier",
                crate::py_repr(flag)
            )));
        }
        out.push(token.to_owned());
    }
    Ok(out)
}

fn validate_launch(value: &Value, where_: &str) -> Result<Value> {
    let object = value.as_object().ok_or_else(|| {
        crate::HiveError::invalid(format!(
            "{where_} must be an object, got {}",
            json_type(value)
        ))
    })?;
    reject_unknown(object, LAUNCH_FIELDS, where_)?;
    if object.get("kind").is_none() {
        return Err(crate::HiveError::invalid(format!(
            "{where_} is missing required field: kind"
        )));
    }
    let kind = text(object.get("kind"), &format!("{where_} kind"))?;
    if !LAUNCH_KINDS.contains(&kind.as_str()) {
        return Err(crate::HiveError::invalid(format!(
            "{where_} kind {} is not one of {}",
            crate::py_repr(&kind),
            LAUNCH_KINDS.join(", ")
        )));
    }
    if object.get("repo").is_none() {
        return Err(crate::HiveError::invalid(format!(
            "{where_} is missing required field: repo"
        )));
    }
    if object.get("config").is_none() {
        return Err(crate::HiveError::invalid(format!(
            "{where_} is missing required field: config"
        )));
    }
    let repo = text(object.get("repo"), &format!("{where_} repo"))?;
    let config = text(object.get("config"), &format!("{where_} config"))?;
    let mut out = Map::new();
    out.insert("kind".into(), Value::String(kind));
    out.insert("repo".into(), Value::String(repo));
    out.insert("config".into(), Value::String(config));
    if let Some(external) = object.get("config_external") {
        let Some(external) = external.as_bool() else {
            return Err(crate::HiveError::invalid(format!(
                "{where_} config_external must be true or false, got {}",
                json_type(external)
            )));
        };
        out.insert("config_external".into(), Value::Bool(external));
    }
    for key in ["python", "server"] {
        if let Some(value) = optional_text(object.get(key), &format!("{where_} {key}"))? {
            out.insert(key.into(), Value::String(value));
        }
    }
    if let Some(engine_args) = object.get("engine_args")
        && !engine_args.is_null()
    {
        out.insert(
            "engine_args".into(),
            Value::Array(
                validate_launch_args(engine_args)?
                    .into_iter()
                    .map(Value::String)
                    .collect(),
            ),
        );
    }
    if let Some(ctx_flag) = optional_text(object.get("ctx_flag"), &format!("{where_} ctx_flag"))? {
        if !ctx_flag.starts_with("--") {
            return Err(crate::HiveError::invalid(format!(
                "{where_} ctx_flag {} must be a flag (start with '--')",
                crate::py_repr(&ctx_flag)
            )));
        }
        out.insert("ctx_flag".into(), Value::String(ctx_flag));
    }
    if let Some(options) = object.get("options") {
        let empty = options.is_null() || options.as_str().map(str::is_empty).unwrap_or(false);
        if !empty && !options.is_object() {
            return Err(crate::HiveError::invalid(format!(
                "{where_} options must be an object, got {}",
                json_type(options)
            )));
        }
        if let Some(options) = options.as_object() {
            let mut owned: Vec<&str> = options
                .keys()
                .map(String::as_str)
                .filter(|key| LAUNCH_OPTIONS_DENIED.contains(key))
                .collect();
            if !owned.is_empty() {
                owned.sort_unstable();
                return Err(crate::HiveError::invalid(format!(
                    "{where_} options may not override launcher-owned key(s): {}",
                    owned.join(", ")
                )));
            }
            out.insert("options".into(), Value::Object(options.clone()));
        }
    }
    if let Some(timeout) = object.get("timeout_s") {
        let value = if let Some(number) = timeout.as_f64() {
            if number <= 0.0 {
                return Err(crate::HiveError::invalid(format!(
                    "{where_} timeout_s must be > 0, got {number}"
                )));
            }
            timeout.clone()
        } else {
            return Err(crate::HiveError::invalid(format!(
                "{where_} timeout_s must be a number, got {}",
                json_type(timeout)
            )));
        };
        out.insert("timeout_s".into(), value);
    }
    Ok(Value::Object(out))
}

fn valid_endpoint(text: &str) -> bool {
    let rest = text
        .strip_prefix("http://")
        .or_else(|| text.strip_prefix("https://"));
    let Some(rest) = rest else {
        return false;
    };
    let authority = rest.split('/').next().unwrap_or("");
    !authority.is_empty() && !authority.chars().any(char::is_whitespace)
}

fn is_numeric(token: &str) -> bool {
    let text = token.strip_prefix('-').unwrap_or(token);
    let mut parts = text.split('.');
    let Some(integer) = parts.next() else {
        return false;
    };
    if integer.is_empty() || !integer.chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }
    match parts.next() {
        Some(fraction) => {
            !fraction.is_empty()
                && fraction.chars().all(|ch| ch.is_ascii_digit())
                && parts.next().is_none()
        }
        None => true,
    }
}

fn display_value(value: &Value) -> String {
    match value {
        Value::String(text) => crate::py_repr(text),
        Value::Null => "None".into(),
        Value::Bool(value) => {
            if *value {
                "True".into()
            } else {
                "False".into()
            }
        }
        Value::Number(number) => number.to_string(),
        Value::Array(_) | Value::Object(_) => json_type(value).into(),
    }
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
    fn allowed_engine_args_are_trimmed_and_negative_values_pass() {
        let data = json!({
            "name": "x", "version": 1,
            "tiers": [{
                "role": "face", "repo": "local", "file": "f",
                "engine_args": [" -t ", "4", "--flash-attn", "on", "-ngl", "-1"]
            }]
        });
        let stack = parse_stack(&data).unwrap();
        assert_eq!(
            stack["tiers"][0]["engine_args"],
            json!(["-t", "4", "--flash-attn", "on", "-ngl", "-1"])
        );
    }

    #[test]
    fn a_disallowed_engine_flag_is_refused() {
        let data = json!({
            "name": "x", "version": 1,
            "tiers": [{
                "role": "face", "repo": "local", "file": "f",
                "engine_args": ["--host", "0.0.0.0"]
            }]
        });
        let error = parse_stack(&data).unwrap_err().to_string();
        assert!(
            error.contains("'--host' is not an allowed engine flag"),
            "{error}"
        );
    }

    #[test]
    fn a_valid_launch_block_round_trips_its_fields() {
        let data = json!({
            "name": "x", "version": 1,
            "tiers": [{
                "role": "face", "repo": "local", "file": "f",
                "endpoint": "http://127.0.0.1:8080/v1",
                "launch": {
                    "kind": "strata",
                    "repo": "~/strata",
                    "config": "cfg.json",
                    "config_external": true,
                    "python": ".venv/bin/python",
                    "engine_args": ["--prefill", "512"],
                    "options": {"foo": 1},
                    "timeout_s": 30
                }
            }]
        });
        let stack = parse_stack(&data).unwrap();
        let launch = &stack["tiers"][0]["launch"];
        assert_eq!(launch["kind"], "strata");
        assert_eq!(launch["config_external"], true);
        assert_eq!(launch["engine_args"], json!(["--prefill", "512"]));
        assert_eq!(launch["options"], json!({"foo": 1}));
    }

    #[test]
    fn launch_denied_args_unknown_fields_and_kinds_are_refused() {
        for launch in [
            json!({"kind": "strata", "repo": "r", "config": "c",
                   "engine_args": ["--config", "x"]}),
            json!({"kind": "strata", "repo": "r", "config": "c", "typo": true}),
            json!({"kind": "boss", "repo": "r", "config": "c"}),
            json!({"kind": "strata", "repo": "r", "config": "c",
                   "options": {"port": 9}}),
            json!({"kind": "strata", "repo": "r", "config": "c", "timeout_s": 0}),
        ] {
            let data = json!({
                "name": "x", "version": 1,
                "tiers": [{
                    "role": "face", "repo": "local", "file": "f",
                    "endpoint": "http://127.0.0.1:8080/v1",
                    "launch": launch
                }]
            });
            assert!(parse_stack(&data).is_err(), "{data}");
        }
    }

    #[test]
    fn an_endpoint_without_a_host_is_refused() {
        let data = json!({
            "name": "x", "version": 1,
            "tiers": [{
                "role": "face", "repo": "local", "file": "f",
                "endpoint": "http:///v1"
            }]
        });
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
